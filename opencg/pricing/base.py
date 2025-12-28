"""
Pricing problem abstract base class.

This module defines the interface that all pricing problem solvers must implement.
Users can either:
1. Use the provided LabelingAlgorithm (default SPPRC implementation)
2. Implement their own by subclassing PricingProblem

The pricing problem finds paths (columns) with negative reduced cost.
The reduced cost of a column is: c_j - sum_i(pi_i * a_ij)
where:
- c_j is the column's cost
- pi_i is the dual value for constraint i
- a_ij = 1 if column j covers item i, 0 otherwise

Design Philosophy:
-----------------
- Required methods are abstract and must be implemented
- Optional methods have default implementations
- Hooks allow customization without full reimplementation
- The interface supports various SPPRC variants (elementary, ng-route, etc.)

Customization Guide:
-------------------
To create a custom pricing solver:

1. Subclass PricingProblem
2. Implement required methods: _solve_impl
3. Optionally override hooks for custom behavior
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from opencg.core.column import Column
from opencg.core.problem import Problem
from opencg.pricing.label import Label


class PricingStatus(Enum):
    """
    Status of the pricing problem solution.
    """
    OPTIMAL = auto()          # Found optimal (most negative RC) column
    COLUMNS_FOUND = auto()    # Found column(s) with negative RC
    NO_COLUMNS = auto()        # No columns with negative RC found
    TIME_LIMIT = auto()       # Time limit reached
    ITERATION_LIMIT = auto()  # Iteration/label limit reached
    INFEASIBLE = auto()       # No feasible path exists
    ERROR = auto()            # Error occurred


@dataclass
class PricingSolution:
    """
    Result of solving the pricing problem.

    Attributes:
        status: Solution status
        columns: List of columns with negative reduced cost
        best_reduced_cost: Most negative reduced cost found
        num_labels_created: Total labels created during search
        num_labels_dominated: Labels pruned by dominance
        solve_time: Time spent solving in seconds
        iterations: Number of iterations (node extensions)
    """
    status: PricingStatus = PricingStatus.NO_COLUMNS
    columns: list[Column] = field(default_factory=list)
    best_reduced_cost: Optional[float] = None
    num_labels_created: int = 0
    num_labels_dominated: int = 0
    solve_time: float = 0.0
    iterations: int = 0

    @property
    def has_negative_reduced_cost(self) -> bool:
        """Check if any column with negative reduced cost was found."""
        return (
            self.best_reduced_cost is not None and
            self.best_reduced_cost < -1e-6
        )

    @property
    def num_columns(self) -> int:
        """Number of columns found."""
        return len(self.columns)

    def get_best_column(self) -> Optional[Column]:
        """Get the column with most negative reduced cost."""
        if not self.columns:
            return None
        return min(self.columns, key=lambda c: c.reduced_cost or float('inf'))

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "PricingSolution:",
            f"  Status: {self.status.name}",
            f"  Columns found: {self.num_columns}",
        ]

        if self.best_reduced_cost is not None:
            lines.append(f"  Best reduced cost: {self.best_reduced_cost:.6f}")

        lines.extend([
            f"  Labels created: {self.num_labels_created}",
            f"  Labels dominated: {self.num_labels_dominated}",
            f"  Solve time: {self.solve_time:.3f}s",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        rc_str = f", rc={self.best_reduced_cost:.4f}" if self.best_reduced_cost else ""
        return f"PricingSolution({self.status.name}, cols={self.num_columns}{rc_str})"


@dataclass
class PricingConfig:
    """
    Configuration for pricing problem solving.

    Attributes:
        max_columns: Maximum columns to return (0 = all with negative RC)
        max_labels: Maximum labels to create (0 = unlimited)
        max_time: Maximum solve time in seconds (0 = unlimited)
        reduced_cost_threshold: Only return columns with RC below this
        check_elementarity: Whether to enforce elementary paths
        use_dominance: Whether to use dominance pruning
        bidirectional: Whether to use bidirectional search
    """
    max_columns: int = 0  # 0 = all columns with negative RC
    max_labels: int = 0   # 0 = unlimited
    max_time: float = 0.0  # 0 = unlimited
    reduced_cost_threshold: float = -1e-6  # Only columns with RC < this
    check_elementarity: bool = False  # True = elementary SPPRC
    use_dominance: bool = True  # Use dominance pruning
    bidirectional: bool = False  # Bidirectional search (advanced)


class PricingProblem(ABC):
    """
    Abstract base class for pricing problem solvers.

    The pricing problem in column generation finds paths with negative
    reduced cost. This is typically formulated as a Shortest Path Problem
    with Resource Constraints (SPPRC).

    The reduced cost of a path P is:
        RC(P) = cost(P) - sum_{i in items(P)} pi_i

    where pi_i is the dual value for covering item i.

    Lifecycle:
    ---------
    1. Create: pricing = LabelingAlgorithm(problem)
    2. Set duals: pricing.set_dual_values(duals)
    3. Solve: solution = pricing.solve()
    4. Get columns: columns = solution.columns
    5. Update duals and repeat

    Attributes:
        problem: The Problem instance
        config: Pricing configuration
    """

    def __init__(self, problem: Problem, config: Optional[PricingConfig] = None):
        """
        Initialize the pricing problem.

        Args:
            problem: The Problem instance
            config: Optional configuration (uses defaults if not provided)
        """
        self._problem = problem
        self._config = config or PricingConfig()

        # Dual values (item_id -> pi)
        self._dual_values: dict[int, float] = {}

        # Item coverage mapping (arc_index -> list of item_ids it covers)
        self._arc_to_items: dict[int, list[int]] = {}
        self._build_arc_item_mapping()

    def _build_arc_item_mapping(self) -> None:
        """Build mapping from arcs to the items they cover.

        This method builds a mapping from arc indices to the item IDs they cover.
        It supports two modes:
        1. VRP-style: Arcs have 'customer_id' attribute indicating covered item
        2. Crew pairing-style: Arc index equals item ID for flight arcs
        """
        network = self._problem.network
        item_ids = {c.item_id for c in self._problem.cover_constraints}

        # First, try to build mapping from arc attributes
        found_via_attribute = False
        for arc in network.arcs:
            # Check for customer_id attribute (VRP-style)
            customer_id = arc.get_attribute('customer_id', None)
            if customer_id is not None and customer_id in item_ids:
                if arc.index not in self._arc_to_items:
                    self._arc_to_items[arc.index] = []
                self._arc_to_items[arc.index].append(customer_id)
                found_via_attribute = True

        # If no arc attributes found, fall back to arc_index == item_id assumption
        if not found_via_attribute:
            for constraint in self._problem.cover_constraints:
                item_id = constraint.item_id
                if item_id not in self._arc_to_items:
                    self._arc_to_items[item_id] = []
                self._arc_to_items[item_id].append(item_id)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def problem(self) -> Problem:
        """The underlying Problem instance."""
        return self._problem

    @property
    def config(self) -> PricingConfig:
        """Pricing configuration."""
        return self._config

    @property
    def dual_values(self) -> dict[int, float]:
        """Current dual values."""
        return self._dual_values.copy()

    @property
    def network(self):
        """Shortcut to problem's network."""
        return self._problem.network

    @property
    def resources(self):
        """Shortcut to problem's resources."""
        return self._problem.resources

    # =========================================================================
    # Public API
    # =========================================================================

    def set_dual_values(self, dual_values: dict[int, float]) -> None:
        """
        Set dual values from the master problem.

        These are used to compute reduced costs.

        Args:
            dual_values: Mapping from item_id to dual value (pi)
        """
        self._dual_values = dual_values.copy()
        self._on_duals_updated()

    def solve(self) -> PricingSolution:
        """
        Solve the pricing problem.

        Finds paths with negative reduced cost using the configured algorithm.

        Returns:
            PricingSolution with found columns and statistics
        """
        # Pre-solve hook
        self._before_solve()

        # Solve
        solution = self._solve_impl()

        # Post-solve hook
        solution = self._after_solve(solution)

        return solution

    def compute_reduced_cost(self, arc_indices: list[int]) -> float:
        """
        Compute the reduced cost of a path.

        Args:
            arc_indices: List of arc indices forming the path

        Returns:
            Reduced cost of the path
        """
        network = self._problem.network
        total_cost = 0.0
        total_dual = 0.0
        covered_items = set()

        for arc_idx in arc_indices:
            arc = network.get_arc(arc_idx)
            if arc is None:
                continue

            total_cost += arc.cost

            # Get items covered by this arc
            if arc_idx in self._arc_to_items:
                for item_id in self._arc_to_items[arc_idx]:
                    if item_id not in covered_items:
                        covered_items.add(item_id)
                        total_dual += self._dual_values.get(item_id, 0.0)

        return total_cost - total_dual

    def get_arc_reduced_cost(self, arc_index: int) -> float:
        """
        Get the reduced cost contribution of an arc.

        This is: arc.cost - sum of duals for items covered by this arc.

        Args:
            arc_index: Arc index

        Returns:
            Reduced cost contribution
        """
        arc = self._problem.network.get_arc(arc_index)
        if arc is None:
            return 0.0

        cost = arc.cost

        # Subtract duals for covered items
        if arc_index in self._arc_to_items:
            for item_id in self._arc_to_items[arc_index]:
                cost -= self._dual_values.get(item_id, 0.0)

        return cost

    def get_items_covered_by_arc(self, arc_index: int) -> list[int]:
        """
        Get item IDs covered by an arc.

        Args:
            arc_index: Arc index

        Returns:
            List of item IDs
        """
        return self._arc_to_items.get(arc_index, [])

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def _solve_impl(self) -> PricingSolution:
        """
        Implementation of the pricing algorithm.

        Subclasses must implement this method to solve the SPPRC.

        Returns:
            PricingSolution with found columns
        """
        pass

    # =========================================================================
    # Hooks (override for custom behavior)
    # =========================================================================

    def _on_duals_updated(self) -> None:
        """
        Hook called when dual values are updated.

        Override to perform precomputation based on new duals.
        """
        pass

    def _before_solve(self) -> None:
        """
        Hook called before solving.

        Override to perform setup before solve.
        """
        pass

    def _after_solve(self, solution: PricingSolution) -> PricingSolution:
        """
        Hook called after solving.

        Override to modify or augment the solution.

        Args:
            solution: The raw solution

        Returns:
            Possibly modified solution
        """
        return solution

    def _create_column_from_label(self, label: Label) -> Column:
        """
        Create a Column from a label at the sink.

        Override to customize column creation.

        Args:
            label: Label at sink node

        Returns:
            Column object
        """
        arc_indices = label.get_arc_indices()

        return Column(
            arc_indices=arc_indices,
            cost=label.cost,
            resource_values=dict(label.resource_values),
            covered_items=label.covered_items,
            reduced_cost=label.reduced_cost,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"PricingProblem: {self._problem.name}",
            f"  Network: {self._problem.network.num_nodes} nodes, {self._problem.network.num_arcs} arcs",
            f"  Resources: {len(self._problem.resources)}",
            f"  Items to cover: {len(self._problem.cover_constraints)}",
            f"  Duals set: {len(self._dual_values) > 0}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(problem={self._problem.name!r})"
