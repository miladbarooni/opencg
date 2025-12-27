"""
Master problem module - LP/MIP solvers for the master problem.

The master problem in column generation is typically:
- Set Partitioning: sum(a_ij * lambda_j) = 1 for each item i
- Set Covering: sum(a_ij * lambda_j) >= 1 for each item i

This module provides:
- MasterProblem: Abstract base class for custom implementations
- HiGHSMasterProblem: Default implementation using HiGHS solver
- MasterSolution: Solution data structure
- SolutionStatus: Enum for solution status

Usage:
------
Using the default HiGHS solver:

    >>> from opencg.master import HiGHSMasterProblem
    >>> master = HiGHSMasterProblem(problem)
    >>> master.add_columns(initial_columns)
    >>> solution = master.solve_lp()
    >>> duals = master.get_dual_values()

Creating a custom solver:

    >>> from opencg.master import MasterProblem, MasterSolution
    >>>
    >>> class MyGurobiMaster(MasterProblem):
    ...     def _build_model(self):
    ...         # Build Gurobi model
    ...         ...
    ...
    ...     def _add_column_impl(self, column):
    ...         # Add column to Gurobi
    ...         ...
    ...
    ...     def _solve_lp_impl(self):
    ...         # Solve with Gurobi
    ...         ...
    ...
    ...     def _solve_ip_impl(self):
    ...         # Solve IP with Gurobi
    ...         ...
    ...
    ...     def _get_dual_values_impl(self):
    ...         # Extract duals from Gurobi
    ...         ...

Customization Points:
--------------------
The MasterProblem ABC provides several hooks for customization:

1. Required methods (must implement):
   - _build_model(): Build the solver model
   - _add_column_impl(): Add a column to the solver
   - _solve_lp_impl(): Solve LP relaxation
   - _solve_ip_impl(): Solve as IP
   - _get_dual_values_impl(): Extract dual values

2. Optional methods (override for custom behavior):
   - _remove_column_impl(): Remove a column
   - _set_column_bounds(): Set variable bounds (for branching)
   - _set_basis_impl(): Set warm start basis
   - _get_basis_impl(): Get current basis
   - _add_cut_impl(): Add cutting planes

3. Hooks (override to customize behavior):
   - _on_column_added(): Called after adding a column
   - _before_solve_lp(): Called before LP solve
   - _after_solve_lp(): Called after LP solve
   - _before_solve_ip(): Called before IP solve
   - _after_solve_ip(): Called after IP solve
"""

from opencg.master.solution import MasterSolution, SolutionStatus
from opencg.master.base import MasterProblem, StabilizationConfig

# Try to import HiGHS implementation
try:
    from opencg.master.highs import HiGHSMasterProblem, HIGHS_AVAILABLE
except ImportError:
    HIGHS_AVAILABLE = False
    HiGHSMasterProblem = None  # type: ignore

# Try to import CPLEX implementation
try:
    from opencg.master.cplex import CPLEXMasterProblem, CPLEX_AVAILABLE
except ImportError:
    CPLEX_AVAILABLE = False
    CPLEXMasterProblem = None  # type: ignore


__all__ = [
    # Solution
    'MasterSolution',
    'SolutionStatus',

    # Base class
    'MasterProblem',
    'StabilizationConfig',

    # HiGHS implementation
    'HiGHSMasterProblem',
    'HIGHS_AVAILABLE',

    # CPLEX implementation
    'CPLEXMasterProblem',
    'CPLEX_AVAILABLE',
]
