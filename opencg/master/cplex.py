"""
CPLEX implementation of the master problem.

This module provides a master problem solver using IBM CPLEX,
a high-performance commercial LP/MIP solver.

CPLEX is a powerful commercial solver that offers:
- Excellent LP and MIP performance
- Advanced presolve and cutting planes
- Parallel solving capabilities
- Robust numerical handling

Usage:
    >>> from opencg.master import CPLEXMasterProblem
    >>> master = CPLEXMasterProblem(problem)
    >>> master.add_columns(initial_columns)
    >>> solution = master.solve_lp()
    >>> duals = master.get_dual_values()

Requirements:
    - IBM CPLEX must be installed
    - docplex package: pip install docplex
    - CPLEX Python API configured
"""

import time
from typing import Any, Optional

try:
    from docplex.mp.model import Model as CplexModel
    from docplex.mp.solution import SolveSolution
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False
    CplexModel = None

from opencg.core.column import Column
from opencg.core.problem import CoverType, ObjectiveSense, Problem
from opencg.master.base import MasterProblem
from opencg.master.solution import MasterSolution, SolutionStatus


def _map_cplex_status(solve_status: Any) -> SolutionStatus:
    """Map CPLEX solve status to our SolutionStatus."""
    if not CPLEX_AVAILABLE:
        return SolutionStatus.ERROR

    # Check solve status
    if solve_status is None:
        return SolutionStatus.NOT_SOLVED

    status_name = str(solve_status).lower()

    if 'optimal' in status_name:
        return SolutionStatus.OPTIMAL
    elif 'infeasible' in status_name:
        return SolutionStatus.INFEASIBLE
    elif 'unbounded' in status_name:
        return SolutionStatus.UNBOUNDED
    elif 'time' in status_name or 'limit' in status_name:
        return SolutionStatus.TIME_LIMIT
    elif 'feasible' in status_name:
        return SolutionStatus.FEASIBLE
    else:
        return SolutionStatus.ERROR


class CPLEXMasterProblem(MasterProblem):
    """
    Master problem solver using IBM CPLEX.

    This implementation uses the docplex Python API for CPLEX.
    It provides all the functionality needed for column generation:
    - LP and IP solving
    - Dual value extraction
    - Warm starting
    - Column bounds for branching

    Example:
        >>> from opencg.master import CPLEXMasterProblem
        >>> from opencg import Problem
        >>>
        >>> # Create master problem
        >>> master = CPLEXMasterProblem(problem, time_limit=300)
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

    Attributes:
        time_limit: Maximum solve time in seconds (None = no limit)
        verbosity: CPLEX output level (0 = silent, 1+ = verbose)
        threads: Number of threads to use (0 = automatic)
    """

    def __init__(
        self,
        problem: Problem,
        time_limit: Optional[float] = None,
        verbosity: int = 0,
        threads: int = 0,
        mip_gap: float = 1e-4,
    ):
        """
        Initialize the CPLEX master problem.

        Args:
            problem: The Problem instance
            time_limit: Maximum solve time in seconds (None = no limit)
            verbosity: CPLEX output level (0 = silent)
            threads: Number of threads (0 = automatic)
            mip_gap: MIP optimality gap tolerance

        Raises:
            ImportError: If docplex/CPLEX is not available
        """
        if not CPLEX_AVAILABLE:
            raise ImportError(
                "CPLEX is not available. Install it with: pip install docplex\n"
                "Also ensure IBM CPLEX is installed and configured."
            )

        self._time_limit = time_limit
        self._verbosity = verbosity
        self._threads = threads
        self._mip_gap = mip_gap

        # CPLEX model (created in _build_model)
        self._model: Optional[CplexModel] = None

        # Variables for each column: column_id -> variable
        self._column_vars: dict[int, Any] = {}

        # Constraints for each item: item_id -> constraint
        self._item_constraints: dict[int, Any] = {}

        # Track if we're in IP mode
        self._is_ip_mode: bool = False

        # Last LP solution for dual extraction
        self._last_lp_solution: Optional[SolveSolution] = None

        # Pending constraints (created lazily when first column is added)
        self._pending_constraints: dict[int, tuple] = {}

        # Call parent init (which calls _build_model)
        super().__init__(problem)

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _build_model(self) -> None:
        """Build the CPLEX model with covering constraints."""
        self._model = CplexModel(name=self._problem.name or "master_problem")

        # Set solver parameters
        if self._verbosity == 0:
            self._model.context.solver.log_output = False
        else:
            self._model.context.solver.log_output = True

        if self._time_limit is not None:
            self._model.set_time_limit(self._time_limit)

        if self._threads > 0:
            self._model.context.cplex_parameters.threads = self._threads

        self._model.parameters.mip.tolerances.mipgap = self._mip_gap

        # Docplex doesn't allow trivially infeasible constraints (like 0 >= 1)
        # We'll store constraint info and create constraints when first column is added
        self._pending_constraints = {}
        for constraint in self._problem.cover_constraints:
            item_id = constraint.item_id
            rhs = constraint.rhs

            if self._problem.cover_type == CoverType.SET_COVERING:
                self._pending_constraints[item_id] = ('>=', rhs)
            elif self._problem.cover_type == CoverType.SET_PARTITIONING:
                self._pending_constraints[item_id] = ('==', rhs)
            else:
                if constraint.is_equality:
                    self._pending_constraints[item_id] = ('==', rhs)
                else:
                    self._pending_constraints[item_id] = ('>=', rhs)

        # Set objective sense (with dummy expression, will be updated when columns are added)
        if self._problem.objective_sense == ObjectiveSense.MINIMIZE:
            self._model.minimize(0)
        else:
            self._model.maximize(0)

    def _add_column_impl(self, column: Column) -> int:
        """Add a column to the CPLEX model."""
        col_id = column.column_id

        # Create continuous variable for LP (will be changed to binary for IP)
        var = self._model.continuous_var(
            lb=0,
            name=f"lambda_{col_id}"
        )
        self._column_vars[col_id] = var

        # Add to objective
        obj = self._model.get_objective_expr()
        if obj is None:
            if self._problem.objective_sense == ObjectiveSense.MINIMIZE:
                self._model.minimize(column.cost * var)
            else:
                self._model.maximize(column.cost * var)
        else:
            # Check if objective is the constant 0 (initial state)
            try:
                is_zero = obj.is_constant() and obj.constant == 0
            except AttributeError:
                is_zero = False

            if is_zero:
                if self._problem.objective_sense == ObjectiveSense.MINIMIZE:
                    self._model.minimize(column.cost * var)
                else:
                    self._model.maximize(column.cost * var)
            else:
                self._model.set_objective(
                    self._model.objective_sense,
                    obj + column.cost * var
                )

        # Add to covering constraints (creating them lazily as needed)
        for item_id in column.covered_items:
            coeff = self._get_coefficient_matrix_entry(column, item_id)
            if coeff == 0.0:
                continue

            if item_id in self._item_constraints:
                # Constraint exists, add variable to it
                ct = self._item_constraints[item_id]
                ct.lhs += coeff * var
            elif item_id in self._pending_constraints:
                # Create the constraint now with this variable
                sense, rhs = self._pending_constraints[item_id]
                if sense == '>=':
                    ct = self._model.add_constraint(
                        coeff * var >= rhs,
                        ctname=f"cover_{item_id}"
                    )
                else:  # '=='
                    ct = self._model.add_constraint(
                        coeff * var == rhs,
                        ctname=f"cover_{item_id}"
                    )
                self._item_constraints[item_id] = ct
                del self._pending_constraints[item_id]

        return len(self._column_vars) - 1

    def _solve_lp_impl(self) -> MasterSolution:
        """Solve the LP relaxation."""
        start_time = time.time()

        # Ensure we're in LP mode (variables are continuous)
        if self._is_ip_mode:
            self._set_all_columns_continuous()
            self._is_ip_mode = False

        # Solve
        solution = self._model.solve()
        self._last_lp_solution = solution

        solve_time = time.time() - start_time

        # Get status
        if solution is None:
            status = SolutionStatus.INFEASIBLE
        else:
            status = _map_cplex_status(self._model.solve_status)

        # Build solution
        master_solution = MasterSolution(
            status=status,
            solve_time=solve_time,
            num_columns=self.num_columns,
        )

        # Extract solution if available
        if solution is not None:
            master_solution.objective_value = solution.objective_value

            # Get primal values
            for col_id, var in self._column_vars.items():
                value = solution.get_value(var)
                if value is not None and abs(value) > 1e-10:
                    master_solution.column_values[col_id] = value

            # Get dual values
            master_solution.dual_values = self._get_dual_values_impl()

            # Get reduced costs using model method
            try:
                col_ids = list(self._column_vars.keys())
                vars_list = list(self._column_vars.values())
                if vars_list:
                    rcs = self._model.reduced_costs(vars_list)
                    for col_id, rc in zip(col_ids, rcs):
                        if rc is not None and abs(rc) > 1e-10:
                            master_solution.reduced_costs[col_id] = rc
            except Exception:
                pass  # Reduced costs not critical

        return master_solution

    def _solve_ip_impl(self) -> MasterSolution:
        """Solve as integer program."""
        start_time = time.time()

        # Set variables to binary
        self._set_all_columns_binary()
        self._is_ip_mode = True

        # Solve
        solution = self._model.solve()

        solve_time = time.time() - start_time

        # Get status
        if solution is None:
            status = SolutionStatus.INFEASIBLE
        else:
            status = _map_cplex_status(self._model.solve_status)

        # Build solution
        master_solution = MasterSolution(
            status=status,
            solve_time=solve_time,
            num_columns=self.num_columns,
        )

        # Extract solution if available
        if solution is not None:
            master_solution.objective_value = solution.objective_value

            # Get MIP gap if available
            try:
                master_solution.gap = self._model.solve_details.mip_relative_gap
            except (AttributeError, RuntimeError):
                pass

            # Get primal values
            for col_id, var in self._column_vars.items():
                value = solution.get_value(var)
                if value is not None and abs(value) > 0.5:  # Binary threshold
                    master_solution.column_values[col_id] = round(value)

        return master_solution

    def _get_dual_values_impl(self) -> dict[int, float]:
        """Extract dual values from CPLEX."""
        duals = {}

        if self._last_lp_solution is None:
            return duals

        # Use model's dual_values method (more reliable than solution object)
        try:
            constraints = list(self._item_constraints.values())
            item_ids = list(self._item_constraints.keys())

            if constraints:
                dual_values = self._model.dual_values(constraints)
                for item_id, dual in zip(item_ids, dual_values):
                    if dual is not None:
                        duals[item_id] = dual
                    else:
                        duals[item_id] = 0.0
        except Exception:
            # Fallback: set all duals to 0
            for item_id in self._item_constraints:
                duals[item_id] = 0.0

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
        if column_id not in self._column_vars:
            raise ValueError(f"Column {column_id} not found in master")

        var = self._column_vars[column_id]
        var.lb = lower
        var.ub = upper

        # Track for later reset
        self._column_bounds[column_id] = (lower, upper)

    def _remove_column_impl(self, column_id: int) -> bool:
        """Remove a column from CPLEX."""
        if column_id not in self._column_vars:
            return False

        var = self._column_vars[column_id]

        # Set bounds to [0, 0] to effectively remove
        var.lb = 0.0
        var.ub = 0.0

        return True

    def _add_cut_impl(
        self,
        coefficients: dict[int, float],
        sense: str,
        rhs: float
    ) -> int:
        """Add a cutting plane."""
        # Build linear expression
        expr = self._model.linear_expr()
        for col_id, coeff in coefficients.items():
            if col_id in self._column_vars:
                expr += coeff * self._column_vars[col_id]

        # Add constraint based on sense
        if sense == '<=':
            self._model.add_constraint(expr <= rhs)
        elif sense == '>=':
            self._model.add_constraint(expr >= rhs)
        else:  # '='
            self._model.add_constraint(expr == rhs)

        return self._model.number_of_constraints - 1

    # =========================================================================
    # CPLEX-specific Methods
    # =========================================================================

    def _set_all_columns_continuous(self) -> None:
        """Set all column variables to continuous (for LP)."""
        for col_id, var in self._column_vars.items():
            # Change variable type to continuous
            var.set_vartype('C')

            # Reset bounds
            if col_id in self._column_bounds:
                lower, upper = self._column_bounds[col_id]
                var.lb = lower
                var.ub = upper
            else:
                var.lb = 0.0
                var.ub = float('inf')

    def _set_all_columns_binary(self) -> None:
        """Set all column variables to binary (for IP)."""
        for col_id, var in self._column_vars.items():
            # Change variable type to binary
            var.set_vartype('B')

            # Set bounds for binary
            if col_id in self._column_bounds:
                lower, upper = self._column_bounds[col_id]
                var.lb = max(0, lower)
                var.ub = min(1, upper)
            else:
                var.lb = 0.0
                var.ub = 1.0

    def set_time_limit(self, seconds: float) -> None:
        """
        Set the solver time limit.

        Args:
            seconds: Maximum solve time in seconds
        """
        self._time_limit = seconds
        self._model.set_time_limit(seconds)

    def set_verbosity(self, level: int) -> None:
        """
        Set the solver verbosity level.

        Args:
            level: 0 = silent, 1+ = verbose
        """
        self._verbosity = level
        self._model.context.solver.log_output = (level > 0)

    def set_threads(self, num_threads: int) -> None:
        """
        Set the number of threads to use.

        Args:
            num_threads: Number of threads (0 = automatic)
        """
        self._threads = num_threads
        if num_threads > 0:
            self._model.context.cplex_parameters.threads = num_threads

    def set_mip_gap(self, gap: float) -> None:
        """
        Set the MIP optimality gap tolerance.

        Args:
            gap: Relative MIP gap (e.g., 0.01 for 1%)
        """
        self._mip_gap = gap
        self._model.parameters.mip.tolerances.mipgap = gap

    def get_model_stats(self) -> dict[str, Any]:
        """
        Get statistics about the model.

        Returns:
            Dictionary with model statistics
        """
        return {
            'num_columns': len(self._column_vars),
            'num_constraints': self._model.number_of_constraints,
            'num_variables': self._model.number_of_variables,
        }

    def export_model(self, filename: str) -> None:
        """
        Export the model to a file.

        Supported formats: .lp, .mps, .sav

        Args:
            filename: Path to export file
        """
        self._model.export_as_lp(filename)

    def get_cplex_model(self) -> CplexModel:
        """
        Get the underlying CPLEX model for advanced operations.

        Returns:
            The docplex Model object
        """
        return self._model
