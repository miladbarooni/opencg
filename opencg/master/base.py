"""
Master problem abstract base class.

This module defines the interface that all master problem solvers must implement.
Users can either:
1. Use the provided HiGHSMasterProblem (default implementation)
2. Implement their own by subclassing MasterProblem

Design Philosophy:
-----------------
- Required methods are abstract and must be implemented
- Optional methods have default implementations (no-op or NotImplementedError)
- Hooks allow customization without full reimplementation
- The interface supports both LP and IP solving
- Dual stabilization is built into the interface

Customization Guide:
-------------------
To create a custom master problem solver:

1. Subclass MasterProblem
2. Implement required methods: _build_model, _add_column_impl, _solve_lp_impl, etc.
3. Optionally override hooks for custom behavior

Example:
    >>> class MyGurobiMaster(MasterProblem):
    ...     def _build_model(self) -> None:
    ...         self._model = gurobipy.Model()
    ...         # ... build constraints ...
    ...
    ...     def _add_column_impl(self, column: Column) -> int:
    ...         # Add column to Gurobi model
    ...         ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from opencg.core.column import Column
from opencg.core.problem import Problem
from opencg.master.solution import MasterSolution


@dataclass
class StabilizationConfig:
    """
    Configuration for dual stabilization.

    Dual stabilization helps prevent oscillation of dual values during
    column generation, which can slow convergence.

    Methods supported:
    - Boxstep: Constrain duals to be within [pi - delta, pi + delta] of previous
    - Smoothing: Use weighted average of current and previous duals

    Attributes:
        enabled: Whether stabilization is active
        method: Stabilization method ('boxstep', 'smoothing', 'none')
        boxstep_delta: Initial box size for boxstep method
        boxstep_shrink: Factor to shrink box when no improving columns found
        smoothing_alpha: Weight for current duals in smoothing (1.0 = no smoothing)
        center: Current stabilization center (previous duals)
    """
    enabled: bool = False
    method: str = 'none'  # 'boxstep', 'smoothing', 'none'

    # Boxstep parameters
    boxstep_delta: float = 10.0
    boxstep_shrink: float = 0.5

    # Smoothing parameters
    smoothing_alpha: float = 0.5  # alpha * current + (1-alpha) * previous

    # Internal state
    center: dict[int, float] = field(default_factory=dict)


class MasterProblem(ABC):
    """
    Abstract base class for master problem solvers.

    The master problem in column generation is typically a set partitioning
    or set covering problem:

    Set Partitioning:
        min  sum_j c_j * lambda_j
        s.t. sum_j a_ij * lambda_j = 1   for all items i
             lambda_j >= 0

    Set Covering:
        min  sum_j c_j * lambda_j
        s.t. sum_j a_ij * lambda_j >= 1  for all items i
             lambda_j >= 0

    Where:
    - lambda_j is the decision variable for column j
    - c_j is the cost of column j
    - a_ij = 1 if column j covers item i, 0 otherwise

    The dual values (pi_i) from the LP relaxation are used in the pricing
    subproblem to find columns with negative reduced cost.

    Lifecycle:
    ---------
    1. Create: master = HiGHSMasterProblem(problem)
    2. Add initial columns: master.add_columns(initial_columns)
    3. Solve LP: solution = master.solve_lp()
    4. Get duals for pricing: duals = master.get_dual_values()
    5. Add new columns from pricing: master.add_column(new_col)
    6. Repeat 3-5 until no negative reduced cost columns
    7. Solve IP if needed: solution = master.solve_ip()

    Attributes:
        problem: The Problem instance defining the optimization problem
        stabilization: Configuration for dual stabilization
    """

    def __init__(self, problem: Problem):
        """
        Initialize the master problem.

        Args:
            problem: The Problem instance with network, resources, and constraints
        """
        self._problem = problem
        self._stabilization = StabilizationConfig()

        # Column tracking
        self._columns: list[Column] = []
        self._column_id_to_index: dict[int, int] = {}

        # Item ID to constraint index mapping
        self._item_to_constraint_idx: dict[int, int] = {}
        for idx, constraint in enumerate(problem.cover_constraints):
            self._item_to_constraint_idx[constraint.item_id] = idx

        # Branching state: column_id -> (lower_bound, upper_bound)
        self._column_bounds: dict[int, tuple] = {}

        # Build the model
        self._build_model()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def problem(self) -> Problem:
        """The underlying Problem instance."""
        return self._problem

    @property
    def num_columns(self) -> int:
        """Number of columns currently in the master problem."""
        return len(self._columns)

    @property
    def num_constraints(self) -> int:
        """Number of covering constraints."""
        return len(self._problem.cover_constraints)

    @property
    def columns(self) -> list[Column]:
        """List of columns in the master problem."""
        return self._columns.copy()

    @property
    def stabilization(self) -> StabilizationConfig:
        """Stabilization configuration."""
        return self._stabilization

    # =========================================================================
    # Abstract Methods (MUST be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _build_model(self) -> None:
        """
        Build the initial master problem model.

        This method is called once during initialization. It should:
        1. Create the solver model object
        2. Add covering constraints (one per item)
        3. Set objective sense (min/max)
        4. Configure solver options

        The constraints should be:
        - sum_j a_ij * lambda_j = 1 (set partitioning)
        - sum_j a_ij * lambda_j >= 1 (set covering)

        Initially there are no columns, so constraints have no variables yet.
        """
        pass

    @abstractmethod
    def _add_column_impl(self, column: Column) -> int:
        """
        Add a column to the solver model.

        This is the low-level implementation that adds the column to the
        actual solver (HiGHS, Gurobi, etc.).

        Args:
            column: The column to add

        Returns:
            The index of the column in the solver model
        """
        pass

    @abstractmethod
    def _solve_lp_impl(self) -> MasterSolution:
        """
        Solve the LP relaxation.

        This method should:
        1. Call the solver's LP solve method
        2. Extract solution status
        3. Extract primal values (column values)
        4. Extract dual values (for pricing)
        5. Return MasterSolution

        Returns:
            MasterSolution with status, objective, primals, and duals
        """
        pass

    @abstractmethod
    def _solve_ip_impl(self) -> MasterSolution:
        """
        Solve as integer program.

        This method should:
        1. Set column variables to binary/integer
        2. Call the solver's IP solve method
        3. Extract solution
        4. Return MasterSolution

        Returns:
            MasterSolution with status, objective, and primal values
        """
        pass

    @abstractmethod
    def _get_dual_values_impl(self) -> dict[int, float]:
        """
        Extract dual values from the solver.

        Returns:
            Mapping from item_id to dual value (pi)
        """
        pass

    # =========================================================================
    # Public API - Column Management
    # =========================================================================

    def add_column(self, column: Column) -> int:
        """
        Add a column to the master problem.

        Args:
            column: The column to add (must have column_id set)

        Returns:
            Index of the column in the master problem

        Raises:
            ValueError: If column has no column_id
        """
        if column.column_id is None:
            raise ValueError("Column must have column_id set before adding to master")

        # Track the column
        idx = len(self._columns)
        self._columns.append(column)
        self._column_id_to_index[column.column_id] = idx

        # Add to solver
        solver_idx = self._add_column_impl(column)

        # Hook for subclasses
        self._on_column_added(column, idx, solver_idx)

        return idx

    def add_columns(self, columns: list[Column]) -> list[int]:
        """
        Add multiple columns to the master problem.

        Args:
            columns: List of columns to add

        Returns:
            List of indices of added columns
        """
        return [self.add_column(col) for col in columns]

    def get_column(self, column_id: int) -> Optional[Column]:
        """
        Get a column by its ID.

        Args:
            column_id: The column's unique identifier

        Returns:
            The column, or None if not found
        """
        idx = self._column_id_to_index.get(column_id)
        if idx is None:
            return None
        return self._columns[idx]

    def remove_column(self, column_id: int) -> bool:
        """
        Remove a column from the master problem.

        Note: Not all solvers support efficient column removal. The default
        implementation raises NotImplementedError. Subclasses should override
        if they support this operation.

        Args:
            column_id: The column's unique identifier

        Returns:
            True if column was removed, False if not found
        """
        return self._remove_column_impl(column_id)

    # =========================================================================
    # Public API - Solving
    # =========================================================================

    def solve_lp(self) -> MasterSolution:
        """
        Solve the LP relaxation.

        This is the main method called during column generation.

        Returns:
            MasterSolution with status, objective, primals, and duals
        """
        # Pre-solve hook
        self._before_solve_lp()

        # Solve
        solution = self._solve_lp_impl()

        # Post-solve hook
        solution = self._after_solve_lp(solution)

        return solution

    def solve_ip(self) -> MasterSolution:
        """
        Solve as integer program.

        Call this after column generation converges to get integer solution.

        Returns:
            MasterSolution with status, objective, and primal values
        """
        # Pre-solve hook
        self._before_solve_ip()

        # Solve
        solution = self._solve_ip_impl()

        # Post-solve hook
        solution = self._after_solve_ip(solution)

        return solution

    def solve(self, as_ip: bool = False) -> MasterSolution:
        """
        Solve the master problem.

        Convenience method that calls solve_lp or solve_ip.

        Args:
            as_ip: If True, solve as IP; otherwise solve LP relaxation

        Returns:
            MasterSolution
        """
        return self.solve_ip() if as_ip else self.solve_lp()

    # =========================================================================
    # Public API - Dual Values
    # =========================================================================

    def get_dual_values(self) -> dict[int, float]:
        """
        Get dual values for pricing.

        If stabilization is enabled, returns stabilized duals.

        Returns:
            Mapping from item_id to dual value (pi)
        """
        raw_duals = self._get_dual_values_impl()

        if self._stabilization.enabled:
            return self._apply_stabilization(raw_duals)

        return raw_duals

    def get_raw_dual_values(self) -> dict[int, float]:
        """
        Get dual values without stabilization.

        Returns:
            Mapping from item_id to dual value (pi)
        """
        return self._get_dual_values_impl()

    # =========================================================================
    # Public API - Dual Stabilization
    # =========================================================================

    def enable_stabilization(
        self,
        method: str = 'boxstep',
        **kwargs
    ) -> None:
        """
        Enable dual stabilization.

        Args:
            method: Stabilization method ('boxstep' or 'smoothing')
            **kwargs: Method-specific parameters
        """
        self._stabilization.enabled = True
        self._stabilization.method = method

        if method == 'boxstep':
            self._stabilization.boxstep_delta = kwargs.get('delta', 10.0)
            self._stabilization.boxstep_shrink = kwargs.get('shrink', 0.5)
        elif method == 'smoothing':
            self._stabilization.smoothing_alpha = kwargs.get('alpha', 0.5)

    def disable_stabilization(self) -> None:
        """Disable dual stabilization."""
        self._stabilization.enabled = False
        self._stabilization.method = 'none'
        self._stabilization.center.clear()

    def update_stabilization_center(self, duals: dict[int, float]) -> None:
        """
        Update the stabilization center.

        Call this when a good set of duals is found (e.g., after finding
        improving columns).

        Args:
            duals: The new center (typically current duals)
        """
        self._stabilization.center = duals.copy()

    def shrink_stabilization(self) -> None:
        """
        Shrink the stabilization region.

        Call this when no improving columns are found, to allow duals
        to move further from center.
        """
        if self._stabilization.method == 'boxstep':
            self._stabilization.boxstep_delta *= self._stabilization.boxstep_shrink

    # =========================================================================
    # Public API - Branching Support
    # =========================================================================

    def fix_column(self, column_id: int, value: float) -> None:
        """
        Fix a column to a specific value (for branching).

        Args:
            column_id: The column to fix
            value: The value to fix it to (typically 0 or 1)
        """
        self._set_column_bounds(column_id, value, value)

    def set_column_bounds(
        self,
        column_id: int,
        lower: float = 0.0,
        upper: float = 1.0
    ) -> None:
        """
        Set bounds on a column variable.

        Args:
            column_id: The column to bound
            lower: Lower bound
            upper: Upper bound
        """
        self._set_column_bounds(column_id, lower, upper)

    def reset_column_bounds(self, column_id: int) -> None:
        """
        Reset column bounds to default [0, 1].

        Args:
            column_id: The column to reset
        """
        self._set_column_bounds(column_id, 0.0, 1.0)
        if column_id in self._column_bounds:
            del self._column_bounds[column_id]

    def get_fixed_columns(self) -> dict[int, float]:
        """
        Get columns that are fixed to a value.

        Returns:
            Mapping from column_id to fixed value
        """
        return {
            col_id: bounds[0]
            for col_id, bounds in self._column_bounds.items()
            if bounds[0] == bounds[1]
        }

    # =========================================================================
    # Public API - Warm Starting
    # =========================================================================

    def set_basis(self, basis: Any) -> None:
        """
        Set a warm start basis.

        Args:
            basis: Solver-specific basis representation
        """
        self._set_basis_impl(basis)

    def get_basis(self) -> Optional[Any]:
        """
        Get the current basis for warm starting.

        Returns:
            Solver-specific basis representation, or None
        """
        return self._get_basis_impl()

    # =========================================================================
    # Public API - Cutting Planes
    # =========================================================================

    def add_cut(
        self,
        coefficients: dict[int, float],
        sense: str,
        rhs: float
    ) -> int:
        """
        Add a cutting plane.

        Args:
            coefficients: Mapping from column_id to coefficient
            sense: Constraint sense ('<=', '>=', '=')
            rhs: Right-hand side value

        Returns:
            Index of the added cut
        """
        return self._add_cut_impl(coefficients, sense, rhs)

    # =========================================================================
    # Optional Implementation Methods (override in subclasses)
    # =========================================================================

    def _remove_column_impl(self, column_id: int) -> bool:
        """
        Remove a column from the solver.

        Default implementation raises NotImplementedError.
        Override in subclasses that support column removal.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support column removal"
        )

    def _set_column_bounds(
        self,
        column_id: int,
        lower: float,
        upper: float
    ) -> None:
        """
        Set bounds on a column variable in the solver.

        Default implementation raises NotImplementedError.
        Override in subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support column bounds"
        )

    def _set_basis_impl(self, basis: Any) -> None:
        """
        Set warm start basis in the solver.

        Default implementation does nothing.
        """
        pass

    def _get_basis_impl(self) -> Optional[Any]:
        """
        Get current basis from the solver.

        Default implementation returns None.
        """
        return None

    def _add_cut_impl(
        self,
        coefficients: dict[int, float],
        sense: str,
        rhs: float
    ) -> int:
        """
        Add a cutting plane to the solver.

        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support cutting planes"
        )

    # =========================================================================
    # Hooks (override for custom behavior)
    # =========================================================================

    def _on_column_added(
        self,
        column: Column,
        index: int,
        solver_index: int
    ) -> None:
        """
        Hook called after a column is added.

        Override this to perform custom actions when columns are added.

        Args:
            column: The added column
            index: Index in self._columns
            solver_index: Index in the solver model
        """
        pass

    def _before_solve_lp(self) -> None:
        """
        Hook called before solving LP.

        Override this to perform setup before LP solve.
        """
        pass

    def _after_solve_lp(self, solution: MasterSolution) -> MasterSolution:
        """
        Hook called after solving LP.

        Override this to modify or augment the solution.

        Args:
            solution: The raw solution from the solver

        Returns:
            Possibly modified solution
        """
        return solution

    def _before_solve_ip(self) -> None:
        """
        Hook called before solving IP.

        Override this to perform setup before IP solve.
        """
        pass

    def _after_solve_ip(self, solution: MasterSolution) -> MasterSolution:
        """
        Hook called after solving IP.

        Override this to modify or augment the solution.

        Args:
            solution: The raw solution from the solver

        Returns:
            Possibly modified solution
        """
        return solution

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _apply_stabilization(self, raw_duals: dict[int, float]) -> dict[int, float]:
        """
        Apply dual stabilization.

        Args:
            raw_duals: Raw dual values from solver

        Returns:
            Stabilized dual values
        """
        if not self._stabilization.center:
            # No center yet, use raw duals as center
            self._stabilization.center = raw_duals.copy()
            return raw_duals

        method = self._stabilization.method

        if method == 'boxstep':
            return self._apply_boxstep(raw_duals)
        elif method == 'smoothing':
            return self._apply_smoothing(raw_duals)
        else:
            return raw_duals

    def _apply_boxstep(self, raw_duals: dict[int, float]) -> dict[int, float]:
        """Apply boxstep stabilization."""
        delta = self._stabilization.boxstep_delta
        center = self._stabilization.center

        stabilized = {}
        for item_id, dual in raw_duals.items():
            center_val = center.get(item_id, 0.0)
            # Clamp to [center - delta, center + delta]
            stabilized[item_id] = max(
                center_val - delta,
                min(center_val + delta, dual)
            )

        return stabilized

    def _apply_smoothing(self, raw_duals: dict[int, float]) -> dict[int, float]:
        """Apply smoothing stabilization."""
        alpha = self._stabilization.smoothing_alpha
        center = self._stabilization.center

        stabilized = {}
        for item_id, dual in raw_duals.items():
            center_val = center.get(item_id, dual)
            # Weighted average
            stabilized[item_id] = alpha * dual + (1 - alpha) * center_val

        return stabilized

    def _get_coefficient_matrix_entry(
        self,
        column: Column,
        item_id: int
    ) -> float:
        """
        Get the coefficient a_ij for column j and item i.

        Default is 1.0 if column covers item, 0.0 otherwise.
        Override for generalized covering with non-unit coefficients.

        Args:
            column: The column
            item_id: The item identifier

        Returns:
            Coefficient value (typically 0 or 1)
        """
        return 1.0 if column.covers_item(item_id) else 0.0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compute_reduced_cost(
        self,
        column: Column,
        dual_values: dict[int, float]
    ) -> float:
        """
        Compute the reduced cost of a column.

        reduced_cost = c_j - sum_i (pi_i * a_ij)

        Args:
            column: The column
            dual_values: Dual values (item_id -> pi)

        Returns:
            Reduced cost
        """
        dual_sum = sum(
            dual_values.get(item_id, 0.0) * self._get_coefficient_matrix_entry(column, item_id)
            for item_id in column.covered_items
        )
        return column.cost - dual_sum

    def summary(self) -> str:
        """
        Return a human-readable summary.

        Returns:
            Summary string
        """
        lines = [
            f"MasterProblem: {self._problem.name}",
            f"  Columns: {self.num_columns}",
            f"  Constraints: {self.num_constraints}",
            f"  Cover type: {self._problem.cover_type.name}",
            f"  Objective: {self._problem.objective_sense.name}",
        ]

        if self._stabilization.enabled:
            lines.append(f"  Stabilization: {self._stabilization.method}")

        if self._column_bounds:
            fixed = len(self.get_fixed_columns())
            lines.append(f"  Fixed columns: {fixed}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"columns={self.num_columns}, "
            f"constraints={self.num_constraints})"
        )
