"""
Problem module - the main container for a column generation problem.

The Problem class ties together all components:
- Network (graph structure)
- Resources (constraints on paths)
- Cover constraints (what items must be covered)
- Objective (minimize/maximize what)
- Configuration (solver settings, tolerances)

This is the main user-facing class for defining a problem.

Design Notes:
------------
- Problem is a container, not a solver
- Solving is delegated to solver classes (MasterProblem, PricingProblem)
- Problem can be serialized/deserialized for reproducibility
- Metadata supports experimental tracking

Usage Patterns:
--------------
1. Direct construction: User builds Network, Resources, and Problem in code
2. Parser construction: Parser reads files and creates Problem
3. Generator construction: Synthetic generator creates Problem

Future Extensions:
-----------------
- Add support for side constraints (beyond set covering/partitioning)
- Add support for multiple objective functions
- Add support for warm starting with initial columns
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from opencg.core.network import Network
from opencg.core.resource import Resource
from opencg.core.column import Column, ColumnPool


class ObjectiveSense(Enum):
    """Direction of optimization."""
    MINIMIZE = auto()
    MAXIMIZE = auto()


class CoverType(Enum):
    """
    Type of covering constraints.

    SET_COVERING: Each item covered at least once (>= 1)
    SET_PARTITIONING: Each item covered exactly once (= 1)
    GENERALIZED: Each item covered exactly rhs[i] times (= rhs[i])
    """
    SET_COVERING = auto()     # >= 1
    SET_PARTITIONING = auto() # = 1
    GENERALIZED = auto()      # = rhs[i] (user-specified)


@dataclass
class CoverConstraint:
    """
    A single covering constraint.

    Represents a constraint that certain columns must "cover" this item.

    Attributes:
        item_id: Unique identifier for the item (e.g., flight index)
        name: Human-readable name (e.g., "FL123")
        rhs: Right-hand side (1 for partitioning, >= 1 for covering)
        is_equality: True for partitioning (=), False for covering (>=)
    """
    item_id: int
    name: str = ""
    rhs: float = 1.0
    is_equality: bool = True  # True for set partitioning

    def __hash__(self) -> int:
        return hash(self.item_id)


@dataclass
class Problem:
    """
    Container for a column generation problem.

    This is the main class users interact with to define a problem.
    It contains all the data needed to solve the problem:
    - Network structure (nodes, arcs)
    - Resources and their constraints
    - Items to cover (flights, tasks, etc.)
    - Objective function

    The Problem itself doesn't solve anything - it's passed to solver
    classes (MasterProblem, PricingProblem) for actual solving.

    Attributes:
        name: Problem name (for identification)
        network: The graph structure
        resources: List of resource constraints
        cover_constraints: Items that must be covered
        cover_type: Type of covering (partitioning, covering, generalized)
        objective_sense: Minimize or maximize
        initial_columns: Optional initial columns (warm start)
        metadata: Additional problem information

    Example:
        >>> from opencg import Problem, Network
        >>> from opencg.core import AccumulatingResource
        >>>
        >>> # Build network
        >>> network = Network()
        >>> source = network.add_source()
        >>> sink = network.add_sink()
        >>> # ... add more nodes and arcs ...
        >>>
        >>> # Define resources
        >>> duty_time = AccumulatingResource("duty_time", max_value=10.0)
        >>> flight_time = AccumulatingResource("flight_time", max_value=8.0)
        >>>
        >>> # Define problem
        >>> problem = Problem(
        ...     name="CrewPairing_Instance1",
        ...     network=network,
        ...     resources=[duty_time, flight_time],
        ...     cover_constraints=cover_constraints,  # List of CoverConstraint
        ...     cover_type=CoverType.SET_PARTITIONING,
        ... )
        >>>
        >>> # Now pass to solver
        >>> # solution = solver.solve(problem)
    """
    # Identification
    name: str = "unnamed_problem"

    # Core components
    network: Network = field(default_factory=Network)
    resources: List[Resource] = field(default_factory=list)

    # Cover constraints
    cover_constraints: List[CoverConstraint] = field(default_factory=list)
    cover_type: CoverType = CoverType.SET_PARTITIONING

    # Objective
    objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE

    # Initial solution (warm start)
    initial_columns: List[Column] = field(default_factory=list)

    # Metadata for tracking experiments
    metadata: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # Convenience Methods for Building
    # =========================================================================

    def add_resource(self, resource: Resource) -> None:
        """
        Add a resource constraint.

        Args:
            resource: Resource to add
        """
        self.resources.append(resource)

    def add_cover_constraint(
        self,
        item_id: int,
        name: str = "",
        rhs: float = 1.0,
        is_equality: bool = True
    ) -> CoverConstraint:
        """
        Add a covering constraint.

        Args:
            item_id: Unique identifier for the item
            name: Human-readable name
            rhs: Right-hand side value
            is_equality: True for = constraint, False for >= constraint

        Returns:
            The created CoverConstraint
        """
        constraint = CoverConstraint(
            item_id=item_id,
            name=name or f"item_{item_id}",
            rhs=rhs,
            is_equality=is_equality
        )
        self.cover_constraints.append(constraint)
        return constraint

    def add_initial_column(self, column: Column) -> None:
        """
        Add an initial column (for warm start).

        Args:
            column: Column to add
        """
        self.initial_columns.append(column)

    # =========================================================================
    # Resource Access
    # =========================================================================

    def get_resource(self, name: str) -> Optional[Resource]:
        """
        Get a resource by name.

        Args:
            name: Resource name

        Returns:
            Resource or None if not found
        """
        for resource in self.resources:
            if resource.name == name:
                return resource
        return None

    def get_resource_names(self) -> List[str]:
        """Get list of resource names."""
        return [r.name for r in self.resources]

    # =========================================================================
    # Cover Constraint Access
    # =========================================================================

    def get_cover_constraint(self, item_id: int) -> Optional[CoverConstraint]:
        """
        Get a cover constraint by item ID.

        Args:
            item_id: Item identifier

        Returns:
            CoverConstraint or None if not found
        """
        for constraint in self.cover_constraints:
            if constraint.item_id == item_id:
                return constraint
        return None

    @property
    def num_cover_constraints(self) -> int:
        """Number of covering constraints."""
        return len(self.cover_constraints)

    @property
    def cover_item_ids(self) -> Set[int]:
        """Set of all item IDs that must be covered."""
        return {c.item_id for c in self.cover_constraints}

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> List[str]:
        """
        Validate the problem definition.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate network
        network_errors = self.network.validate()
        errors.extend(network_errors)

        # Check resources
        if not self.resources:
            errors.append("No resources defined")

        # Check for duplicate resource names
        resource_names = [r.name for r in self.resources]
        if len(resource_names) != len(set(resource_names)):
            errors.append("Duplicate resource names")

        # Check cover constraints
        if not self.cover_constraints:
            errors.append("No cover constraints defined")

        # Check for duplicate item IDs
        item_ids = [c.item_id for c in self.cover_constraints]
        if len(item_ids) != len(set(item_ids)):
            errors.append("Duplicate item IDs in cover constraints")

        # Check initial columns
        for i, column in enumerate(self.initial_columns):
            # Check that covered items exist in constraints
            valid_items = self.cover_item_ids
            for item in column.covered_items:
                if item not in valid_items:
                    errors.append(
                        f"Initial column {i} covers invalid item {item}"
                    )

        return errors

    def is_valid(self) -> bool:
        """Check if problem is valid."""
        return len(self.validate()) == 0

    # =========================================================================
    # Summary and Display
    # =========================================================================

    def summary(self) -> str:
        """
        Return a human-readable summary.

        Returns:
            Summary string
        """
        lines = [
            f"Problem: {self.name}",
            f"  Objective: {self.objective_sense.name}",
            f"  Cover type: {self.cover_type.name}",
            "",
            "  Network:",
            f"    Nodes: {self.network.num_nodes}",
            f"    Arcs: {self.network.num_arcs}",
            "",
            "  Resources:",
        ]

        for resource in self.resources:
            lines.append(f"    - {resource}")

        lines.extend([
            "",
            f"  Cover constraints: {self.num_cover_constraints}",
            f"  Initial columns: {len(self.initial_columns)}",
        ])

        if self.metadata:
            lines.append("")
            lines.append("  Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Problem('{self.name}', "
            f"nodes={self.network.num_nodes}, "
            f"arcs={self.network.num_arcs}, "
            f"resources={len(self.resources)}, "
            f"constraints={self.num_cover_constraints})"
        )


# =============================================================================
# Problem Builder (Fluent Interface)
# =============================================================================


class ProblemBuilder:
    """
    Fluent builder for constructing Problem instances.

    Provides a chainable API for building problems step by step.

    Example:
        >>> problem = (
        ...     ProblemBuilder("MyProblem")
        ...     .with_network(network)
        ...     .add_resource(AccumulatingResource("duty_time", max_value=10.0))
        ...     .add_resource(AccumulatingResource("flight_time", max_value=8.0))
        ...     .with_cover_type(CoverType.SET_PARTITIONING)
        ...     .add_cover_constraints_from_arcs(flight_arcs)
        ...     .build()
        ... )
    """

    def __init__(self, name: str = "unnamed_problem"):
        """
        Start building a problem.

        Args:
            name: Problem name
        """
        self._problem = Problem(name=name)

    def with_network(self, network: Network) -> 'ProblemBuilder':
        """Set the network."""
        self._problem.network = network
        return self

    def add_resource(self, resource: Resource) -> 'ProblemBuilder':
        """Add a resource."""
        self._problem.add_resource(resource)
        return self

    def with_resources(self, resources: List[Resource]) -> 'ProblemBuilder':
        """Set all resources."""
        self._problem.resources = list(resources)
        return self

    def with_cover_type(self, cover_type: CoverType) -> 'ProblemBuilder':
        """Set the cover constraint type."""
        self._problem.cover_type = cover_type
        return self

    def add_cover_constraint(
        self,
        item_id: int,
        name: str = "",
        rhs: float = 1.0
    ) -> 'ProblemBuilder':
        """Add a cover constraint."""
        is_equality = self._problem.cover_type != CoverType.SET_COVERING
        self._problem.add_cover_constraint(item_id, name, rhs, is_equality)
        return self

    def with_cover_constraints(
        self,
        constraints: List[CoverConstraint]
    ) -> 'ProblemBuilder':
        """Set all cover constraints."""
        self._problem.cover_constraints = list(constraints)
        return self

    def add_cover_constraints_from_arcs(
        self,
        arcs: List,  # List of Arc
        name_attr: str = "flight_number"
    ) -> 'ProblemBuilder':
        """
        Create cover constraints from arcs (e.g., flights).

        Each arc becomes a cover constraint.

        Args:
            arcs: List of arcs to cover
            name_attr: Attribute to use for constraint name
        """
        is_equality = self._problem.cover_type != CoverType.SET_COVERING
        for arc in arcs:
            name = arc.get_attribute(name_attr, f"arc_{arc.index}")
            self._problem.add_cover_constraint(
                item_id=arc.index,
                name=name,
                is_equality=is_equality
            )
        return self

    def with_objective(self, sense: ObjectiveSense) -> 'ProblemBuilder':
        """Set objective sense."""
        self._problem.objective_sense = sense
        return self

    def minimize(self) -> 'ProblemBuilder':
        """Set objective to minimize."""
        return self.with_objective(ObjectiveSense.MINIMIZE)

    def maximize(self) -> 'ProblemBuilder':
        """Set objective to maximize."""
        return self.with_objective(ObjectiveSense.MAXIMIZE)

    def with_initial_columns(self, columns: List[Column]) -> 'ProblemBuilder':
        """Set initial columns."""
        self._problem.initial_columns = list(columns)
        return self

    def with_metadata(self, **kwargs) -> 'ProblemBuilder':
        """Add metadata."""
        self._problem.metadata.update(kwargs)
        return self

    def build(self) -> Problem:
        """
        Build and return the Problem.

        Returns:
            The constructed Problem

        Raises:
            ValueError: If problem is not valid
        """
        errors = self._problem.validate()
        if errors:
            raise ValueError(
                f"Invalid problem:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return self._problem

    def build_unchecked(self) -> Problem:
        """
        Build without validation.

        Use this if you want to build incrementally and validate later.

        Returns:
            The constructed Problem (may be invalid)
        """
        return self._problem
