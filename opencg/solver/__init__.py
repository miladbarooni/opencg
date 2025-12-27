"""
Solver module - Column Generation algorithm implementation.

This module provides the main column generation algorithm that coordinates
the master problem and pricing subproblem to solve large-scale optimization
problems.

This module provides:
- ColumnGeneration: Main algorithm controller
- CGConfig: Configuration options
- CGSolution: Solution data structure
- CGStatus: Solution status enum
- CGIteration: Per-iteration information

Usage:
------
Basic usage:

    >>> from opencg.solver import ColumnGeneration, CGConfig
    >>> config = CGConfig(max_iterations=100, verbose=True, solve_ip=True)
    >>> cg = ColumnGeneration(problem, config)
    >>> solution = cg.solve()
    >>> if solution.is_optimal:
    ...     print(f"Optimal value: {solution.objective_value}")
    ...     for col in solution.columns:
    ...         print(f"  Column {col.column_id}: value={col.value:.4f}")

With custom components:

    >>> from opencg.solver import ColumnGeneration
    >>> from opencg.master import HiGHSMasterProblem
    >>> from opencg.pricing import LabelingAlgorithm
    >>>
    >>> cg = ColumnGeneration(problem)
    >>> cg.set_master(HiGHSMasterProblem(problem, verbosity=1))
    >>> cg.set_pricing(LabelingAlgorithm(problem, PricingConfig(max_columns=10)))
    >>> solution = cg.solve()

With callbacks for monitoring:

    >>> def progress_callback(cg, iteration):
    ...     print(f"Iter {iteration.iteration}: obj={iteration.master_objective:.2f}")
    ...     return iteration.iteration < 50  # Stop after 50 iterations
    >>>
    >>> cg = ColumnGeneration(problem)
    >>> cg.add_callback(progress_callback)
    >>> solution = cg.solve()

Algorithm Overview:
------------------
1. Initialize master problem with initial/artificial columns
2. Solve master problem LP relaxation
3. Extract dual values (optionally with stabilization)
4. Solve pricing subproblem to find columns with negative reduced cost
5. If improving columns found, add to master and go to step 2
6. If no improving columns, LP is optimal
7. Optionally solve IP for integer solution

Configuration Options:
--------------------
- max_iterations: Maximum CG iterations
- max_time: Time limit in seconds
- max_columns: Maximum columns to generate
- optimality_tolerance: RC threshold for optimality
- solve_ip: Whether to solve IP after LP converges
- use_stabilization: Enable dual stabilization
- verbose: Print progress information

Customization:
-------------
The algorithm can be customized by:
1. Providing custom master/pricing implementations
2. Adding callbacks for monitoring/early stopping
3. Configuring pricing via PricingConfig
4. Using different stabilization methods
"""

from opencg.solver.solution import CGSolution, CGStatus, CGIteration
from opencg.solver.column_generation import ColumnGeneration, CGConfig, CGCallback

__all__ = [
    # Main class
    'ColumnGeneration',

    # Configuration
    'CGConfig',
    'CGCallback',

    # Solution
    'CGSolution',
    'CGStatus',
    'CGIteration',
]
