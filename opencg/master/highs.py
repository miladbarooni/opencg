"""
HiGHS implementation of the master problem.

This module provides a ready-to-use master problem solver using HiGHS,
a high-performance open-source LP/MIP solver.

HiGHS is the default solver for OpenCG because:
- Open source (MIT license)
- High performance (competitive with commercial solvers)
- Good Python bindings (highspy)
- Supports warm starting
- Active development

Usage:
    >>> from opencg.master import HiGHSMasterProblem
    >>> master = HiGHSMasterProblem(problem)
    >>> master.add_columns(initial_columns)
    >>> solution = master.solve_lp()
    >>> duals = master.get_dual_values()
"""

import time
from typing import Any, Dict, List, Optional

try:
    import highspy
    HIGHS_AVAILABLE = True
except ImportError:
    HIGHS_AVAILABLE = False

from opencg.core.column import Column
from opencg.core.problem import Problem, CoverType, ObjectiveSense
from opencg.master.base import MasterProblem
from opencg.master.solution import MasterSolution, SolutionStatus


# HiGHS status mapping
def _map_highs_status(status: int) -> SolutionStatus:
    """Map HiGHS model status to our SolutionStatus."""
    if not HIGHS_AVAILABLE:
        return SolutionStatus.ERROR

    status_map = {
        highspy.HighsModelStatus.kNotset: SolutionStatus.NOT_SOLVED,
        highspy.HighsModelStatus.kLoadError: SolutionStatus.ERROR,
        highspy.HighsModelStatus.kModelError: SolutionStatus.ERROR,
        highspy.HighsModelStatus.kPresolveError: SolutionStatus.ERROR,
        highspy.HighsModelStatus.kSolveError: SolutionStatus.ERROR,
        highspy.HighsModelStatus.kPostsolveError: SolutionStatus.ERROR,
        highspy.HighsModelStatus.kModelEmpty: SolutionStatus.ERROR,
        highspy.HighsModelStatus.kOptimal: SolutionStatus.OPTIMAL,
        highspy.HighsModelStatus.kInfeasible: SolutionStatus.INFEASIBLE,
        highspy.HighsModelStatus.kUnbounded: SolutionStatus.UNBOUNDED,
        highspy.HighsModelStatus.kUnboundedOrInfeasible: SolutionStatus.INF_OR_UNBOUNDED,
        highspy.HighsModelStatus.kTimeLimit: SolutionStatus.TIME_LIMIT,
        highspy.HighsModelStatus.kIterationLimit: SolutionStatus.ITERATION_LIMIT,
    }

    return status_map.get(status, SolutionStatus.ERROR)


class HiGHSMasterProblem(MasterProblem):
    """
    Master problem solver using HiGHS.

    This is the default implementation for solving the master problem
    in column generation. It uses HiGHS via the highspy Python bindings.

    Features:
    - LP and IP solving
    - Warm starting via basis
    - Column bounds for branching
    - Efficient incremental column addition

    Example:
        >>> from opencg.master import HiGHSMasterProblem
        >>> from opencg import Problem
        >>>
        >>> # Create master problem from a Problem instance
        >>> master = HiGHSMasterProblem(problem)
        >>>
        >>> # Add initial columns
        >>> for col in initial_columns:
        ...     master.add_column(col)
        >>>
        >>> # Solve LP relaxation
        >>> solution = master.solve_lp()
        >>> print(f"Objective: {solution.objective_value}")
        >>>
        >>> # Get duals for pricing
        >>> duals = master.get_dual_values()
        >>> for item_id, pi in duals.items():
        ...     print(f"  pi[{item_id}] = {pi}")

    Attributes:
        time_limit: Maximum solve time in seconds (None = no limit)
        verbosity: HiGHS output level (0 = silent, 1 = normal, 2 = verbose)
    """

    def __init__(
        self,
        problem: Problem,
        time_limit: Optional[float] = None,
        verbosity: int = 0
    ):
        """
        Initialize the HiGHS master problem.

        Args:
            problem: The Problem instance
            time_limit: Maximum solve time in seconds (None = no limit)
            verbosity: HiGHS output level (0 = silent)

        Raises:
            ImportError: If highspy is not installed
        """
        if not HIGHS_AVAILABLE:
            raise ImportError(
                "HiGHS is not available. Install it with: pip install highspy"
            )

        self._time_limit = time_limit
        self._verbosity = verbosity

        # HiGHS model (created in _build_model)
        self._highs: Optional[highspy.Highs] = None

        # Track solver column indices (different from our column indices)
        self._column_to_solver_idx: Dict[int, int] = {}
        # Reverse lookup: solver index -> column ID (for O(1) lookup)
        self._solver_idx_to_column_id: Dict[int, int] = {}

        # Track if we need to switch to IP mode
        self._is_ip_mode: bool = False

        # Call parent init (which calls _build_model)
        super().__init__(problem)

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _build_model(self) -> None:
        """Build the HiGHS model with covering constraints."""
        self._highs = highspy.Highs()

        # Set options
        self._highs.setOptionValue('output_flag', self._verbosity > 0)
        self._highs.setOptionValue('log_to_console', self._verbosity > 0)

        if self._time_limit is not None:
            self._highs.setOptionValue('time_limit', self._time_limit)

        # Set objective sense
        if self._problem.objective_sense == ObjectiveSense.MINIMIZE:
            self._highs.changeObjectiveSense(highspy.ObjSense.kMinimize)
        else:
            self._highs.changeObjectiveSense(highspy.ObjSense.kMaximize)

        # Add covering constraints (initially empty, columns added later)
        # For each item, we have: sum_j a_ij * lambda_j {>=, =} rhs_i
        for constraint in self._problem.cover_constraints:
            if self._problem.cover_type == CoverType.SET_COVERING:
                # >= 1
                lower = constraint.rhs
                upper = highspy.kHighsInf
            elif self._problem.cover_type == CoverType.SET_PARTITIONING:
                # = 1
                lower = constraint.rhs
                upper = constraint.rhs
            else:
                # Generalized: use constraint settings
                if constraint.is_equality:
                    lower = constraint.rhs
                    upper = constraint.rhs
                else:
                    lower = constraint.rhs
                    upper = highspy.kHighsInf

            # Add empty row (no columns yet)
            self._highs.addRow(lower, upper, 0, [], [])

    def _add_column_impl(self, column: Column) -> int:
        """Add a column to the HiGHS model."""
        # Get constraint indices and coefficients for this column
        indices = []
        values = []

        for item_id in column.covered_items:
            if item_id in self._item_to_constraint_idx:
                constraint_idx = self._item_to_constraint_idx[item_id]
                coeff = self._get_coefficient_matrix_entry(column, item_id)
                if coeff != 0.0:
                    indices.append(constraint_idx)
                    values.append(coeff)

        # Column bounds: 0 <= lambda <= 1 (or inf for LP)
        lower = 0.0
        upper = highspy.kHighsInf  # Will be set to 1 for IP

        # Add the column
        # addCol(cost, lower, upper, num_nz, indices, values)
        self._highs.addCol(
            column.cost,
            lower,
            upper,
            len(indices),
            indices,
            values
        )

        # Track solver index (both forward and reverse lookup)
        solver_idx = self._highs.getNumCol() - 1
        self._column_to_solver_idx[column.column_id] = solver_idx
        self._solver_idx_to_column_id[solver_idx] = column.column_id

        return solver_idx

    def _solve_lp_impl(self) -> MasterSolution:
        """Solve the LP relaxation."""
        start_time = time.time()

        # Ensure we're in LP mode (columns are continuous)
        if self._is_ip_mode:
            self._set_all_columns_continuous()
            self._is_ip_mode = False

        # Run the solver
        self._highs.run()

        solve_time = time.time() - start_time

        # Get status
        status = _map_highs_status(self._highs.getModelStatus())

        # Build solution
        solution = MasterSolution(
            status=status,
            solve_time=solve_time,
            iterations=self._highs.getInfo().simplex_iteration_count,
            num_columns=self.num_columns,
        )

        # Extract solution if available
        if solution.has_solution or status == SolutionStatus.OPTIMAL:
            info = self._highs.getInfo()
            solution.objective_value = info.objective_function_value

            # Get primal values
            sol = self._highs.getSolution()
            for col_id, solver_idx in self._column_to_solver_idx.items():
                value = sol.col_value[solver_idx]
                if abs(value) > 1e-10:  # Only store non-zero
                    solution.column_values[col_id] = value

            # Get dual values
            solution.dual_values = self._get_dual_values_impl()

            # Get reduced costs
            for col_id, solver_idx in self._column_to_solver_idx.items():
                rc = sol.col_dual[solver_idx]
                if abs(rc) > 1e-10:
                    solution.reduced_costs[col_id] = rc

            # Store basis
            solution.basis = self._get_basis_impl()

        return solution

    def _solve_ip_impl(self) -> MasterSolution:
        """Solve as integer program."""
        start_time = time.time()

        # Set columns to binary
        self._set_all_columns_binary()
        self._is_ip_mode = True

        # Run the solver
        self._highs.run()

        solve_time = time.time() - start_time

        # Get status
        status = _map_highs_status(self._highs.getModelStatus())

        # Build solution
        solution = MasterSolution(
            status=status,
            solve_time=solve_time,
            iterations=self._highs.getInfo().simplex_iteration_count,
            nodes=self._highs.getInfo().mip_node_count,
            num_columns=self.num_columns,
        )

        # Extract solution if available
        if solution.has_solution or status == SolutionStatus.OPTIMAL:
            info = self._highs.getInfo()
            solution.objective_value = info.objective_function_value

            # Get gap if available
            if hasattr(info, 'mip_gap'):
                solution.gap = info.mip_gap

            # Get primal values
            sol = self._highs.getSolution()
            for col_id, solver_idx in self._column_to_solver_idx.items():
                value = sol.col_value[solver_idx]
                if abs(value) > 1e-10:
                    solution.column_values[col_id] = value

        return solution

    def _get_dual_values_impl(self) -> Dict[int, float]:
        """Extract dual values from HiGHS."""
        duals = {}

        sol = self._highs.getSolution()
        if sol.row_dual is None:
            return duals

        for constraint in self._problem.cover_constraints:
            constraint_idx = self._item_to_constraint_idx[constraint.item_id]
            dual = sol.row_dual[constraint_idx]
            duals[constraint.item_id] = dual

        return duals

    # =========================================================================
    # Optional Method Implementations
    # =========================================================================

    def _set_column_bounds(
        self,
        column_id: int,
        lower: float,
        upper: float
    ) -> None:
        """Set bounds on a column variable."""
        if column_id not in self._column_to_solver_idx:
            raise ValueError(f"Column {column_id} not found in master")

        solver_idx = self._column_to_solver_idx[column_id]
        self._highs.changeColBounds(solver_idx, lower, upper)

        # Track for later reset
        self._column_bounds[column_id] = (lower, upper)

    def _remove_column_impl(self, column_id: int) -> bool:
        """Remove a column from HiGHS."""
        if column_id not in self._column_to_solver_idx:
            return False

        solver_idx = self._column_to_solver_idx[column_id]

        # HiGHS doesn't have direct column removal that's efficient
        # We can set the column cost to infinity and bounds to [0, 0]
        # This effectively removes it from the solution
        self._highs.changeColCost(solver_idx, 1e20)  # Very high cost
        self._highs.changeColBounds(solver_idx, 0.0, 0.0)

        # Note: This is a "soft" removal. The column still exists in the model.
        # For true removal, we'd need to rebuild the model.

        return True

    def _set_basis_impl(self, basis: Any) -> None:
        """Set warm start basis."""
        if basis is None:
            return

        col_status, row_status = basis
        highs_basis = highspy.HighsBasis()
        highs_basis.col_status = col_status
        highs_basis.row_status = row_status
        self._highs.setBasis(highs_basis)

    def _get_basis_impl(self) -> Optional[Any]:
        """Get current basis."""
        basis = self._highs.getBasis()
        if basis is None or not basis.valid:
            return None
        return (list(basis.col_status), list(basis.row_status))

    def _add_cut_impl(
        self,
        coefficients: Dict[int, float],
        sense: str,
        rhs: float
    ) -> int:
        """Add a cutting plane."""
        # Convert column IDs to solver indices
        indices = []
        values = []

        for col_id, coeff in coefficients.items():
            if col_id in self._column_to_solver_idx:
                indices.append(self._column_to_solver_idx[col_id])
                values.append(coeff)

        # Determine bounds based on sense
        if sense == '<=':
            lower = -highspy.kHighsInf
            upper = rhs
        elif sense == '>=':
            lower = rhs
            upper = highspy.kHighsInf
        else:  # '='
            lower = rhs
            upper = rhs

        # Add the row
        self._highs.addRow(lower, upper, len(indices), indices, values)

        return self._highs.getNumRow() - 1

    # =========================================================================
    # HiGHS-specific Methods
    # =========================================================================

    def _set_all_columns_continuous(self) -> None:
        """Set all columns to continuous (for LP)."""
        num_cols = self._highs.getNumCol()
        for i in range(num_cols):
            self._highs.changeColIntegrality(
                i, highspy.HighsVarType.kContinuous
            )
            # Reset upper bound to infinity for LP
            if i in self._column_to_solver_idx.values():
                # Check if this column has custom bounds
                col_id = self._get_column_id_from_solver_idx(i)
                if col_id is not None and col_id in self._column_bounds:
                    _, upper = self._column_bounds[col_id]
                    self._highs.changeColBounds(i, 0.0, upper)
                else:
                    self._highs.changeColBounds(i, 0.0, highspy.kHighsInf)

    def _set_all_columns_binary(self) -> None:
        """Set all columns to binary (for IP)."""
        num_cols = self._highs.getNumCol()
        for i in range(num_cols):
            self._highs.changeColIntegrality(
                i, highspy.HighsVarType.kInteger
            )
            # Set upper bound to 1 for binary
            col_id = self._get_column_id_from_solver_idx(i)
            if col_id is not None and col_id in self._column_bounds:
                lower, upper = self._column_bounds[col_id]
                self._highs.changeColBounds(i, lower, min(upper, 1.0))
            else:
                self._highs.changeColBounds(i, 0.0, 1.0)

    def _get_column_id_from_solver_idx(self, solver_idx: int) -> Optional[int]:
        """Get column ID from solver index (O(1) reverse lookup)."""
        return self._solver_idx_to_column_id.get(solver_idx)

    def set_time_limit(self, seconds: float) -> None:
        """
        Set the solver time limit.

        Args:
            seconds: Maximum solve time in seconds
        """
        self._time_limit = seconds
        self._highs.setOptionValue('time_limit', seconds)

    def set_verbosity(self, level: int) -> None:
        """
        Set the solver verbosity level.

        Args:
            level: 0 = silent, 1 = normal, 2 = verbose
        """
        self._verbosity = level
        self._highs.setOptionValue('output_flag', level > 0)
        self._highs.setOptionValue('log_to_console', level > 0)

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model.

        Returns:
            Dictionary with model statistics
        """
        return {
            'num_columns': self._highs.getNumCol(),
            'num_rows': self._highs.getNumRow(),
            'num_nonzeros': self._highs.getNumNz(),
        }
