"""
Column Generation controller.

This module implements the column generation algorithm that coordinates
the master problem and pricing subproblem to solve large-scale optimization
problems.

Algorithm Overview:
------------------
1. Initialize with initial columns (or artificial columns if none)
2. Solve master problem LP relaxation
3. Extract dual values
4. Solve pricing problem to find columns with negative reduced cost
5. If negative RC columns found, add them to master and go to step 2
6. If no negative RC columns, LP is optimal
7. Optionally solve IP for integer solution

Key Features:
------------
- Configurable stopping criteria (iterations, time, gap)
- Callback hooks for monitoring and customization
- Support for warm starting
- Iteration history tracking
- Automatic initial column generation (artificial variables)

References:
----------
- Desaulniers, G., Desrosiers, J., & Solomon, M. M. (Eds.). (2006).
  Column generation. Springer Science & Business Media.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

from opencg.core.column import Column, ColumnPool
from opencg.core.problem import Problem
from opencg.master import (
    HIGHS_AVAILABLE,
    HiGHSMasterProblem,
    MasterProblem,
    MasterSolution,
)
from opencg.pricing import (
    LabelingAlgorithm,
    PricingConfig,
    PricingProblem,
)
from opencg.solver.solution import CGIteration, CGSolution, CGStatus


@dataclass
class CGConfig:
    """
    Configuration for the column generation algorithm.

    Attributes:
        max_iterations: Maximum number of CG iterations (0 = unlimited)
        max_time: Maximum total time in seconds (0 = unlimited)
        max_columns: Maximum columns to generate (0 = unlimited)
        optimality_tolerance: Stop if best RC > -tolerance
        improvement_tolerance: Stop if objective doesn't improve by this much
        max_no_improvement: Stop after this many iterations without improvement
        solve_ip: Whether to solve IP after LP converges
        verbose: Print progress information
        pricing_config: Configuration for pricing subproblem
        use_stabilization: Whether to use dual stabilization
        stabilization_method: Stabilization method ('boxstep', 'smoothing')
        require_integral_lp: If True, continue until LP has no artificial columns
        artificial_tolerance: Max total artificial usage before considering LP integral
    """
    max_iterations: int = 1000
    max_time: float = 3600.0  # 1 hour default
    max_columns: int = 0  # Unlimited
    optimality_tolerance: float = 1e-6
    improvement_tolerance: float = 1e-6
    max_no_improvement: int = 10
    solve_ip: bool = False
    verbose: bool = False
    pricing_config: Optional[PricingConfig] = None
    use_stabilization: bool = False
    stabilization_method: str = 'boxstep'
    require_integral_lp: bool = False
    artificial_tolerance: float = 0.01  # Allow up to 1% artificial usage


# Type alias for callback functions
CGCallback = Callable[['ColumnGeneration', CGIteration], bool]


class ColumnGeneration:
    """
    Column generation algorithm controller.

    This class coordinates the master problem and pricing subproblem to
    implement the column generation algorithm for solving set partitioning,
    set covering, and related problems.

    The algorithm:
    1. Solves the LP relaxation of the master problem
    2. Uses dual values to formulate and solve the pricing subproblem
    3. Adds columns with negative reduced cost to the master
    4. Repeats until no improving columns are found

    Example:
        >>> from opencg.solver import ColumnGeneration, CGConfig
        >>> config = CGConfig(max_iterations=100, verbose=True)
        >>> cg = ColumnGeneration(problem, config)
        >>> solution = cg.solve()
        >>> print(f"Optimal value: {solution.objective_value}")

    Customization:
        You can provide custom master and pricing implementations:

        >>> from opencg.solver import ColumnGeneration
        >>> cg = ColumnGeneration(problem)
        >>> cg.set_master(MyCustomMaster(problem))
        >>> cg.set_pricing(MyCustomPricing(problem))
        >>> solution = cg.solve()

    Callbacks:
        Register callbacks to monitor progress:

        >>> def my_callback(cg, iteration):
        ...     print(f"Iteration {iteration.iteration}: obj={iteration.master_objective}")
        ...     return True  # Continue solving
        >>> cg.add_callback(my_callback)
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[CGConfig] = None,
    ):
        """
        Initialize the column generation controller.

        Args:
            problem: The Problem instance to solve
            config: Configuration options (uses defaults if not provided)
        """
        self._problem = problem
        self._config = config or CGConfig()

        # Components (created lazily or can be set externally)
        self._master: Optional[MasterProblem] = None
        self._pricing: Optional[PricingProblem] = None

        # Column pool for tracking all generated columns
        self._column_pool = ColumnPool()

        # Callbacks
        self._callbacks: list[CGCallback] = []

        # State
        self._is_solved = False
        self._solution: Optional[CGSolution] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def problem(self) -> Problem:
        """The underlying Problem instance."""
        return self._problem

    @property
    def config(self) -> CGConfig:
        """Configuration options."""
        return self._config

    @property
    def master(self) -> Optional[MasterProblem]:
        """The master problem solver."""
        return self._master

    @property
    def pricing(self) -> Optional[PricingProblem]:
        """The pricing problem solver."""
        return self._pricing

    @property
    def column_pool(self) -> ColumnPool:
        """The column pool containing all generated columns."""
        return self._column_pool

    @property
    def is_solved(self) -> bool:
        """Whether solve() has been called."""
        return self._is_solved

    @property
    def solution(self) -> Optional[CGSolution]:
        """The solution (None if not yet solved)."""
        return self._solution

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_master(self, master: MasterProblem) -> None:
        """
        Set a custom master problem solver.

        Args:
            master: Custom MasterProblem implementation
        """
        self._master = master

    def set_pricing(self, pricing: PricingProblem) -> None:
        """
        Set a custom pricing problem solver.

        Args:
            pricing: Custom PricingProblem implementation
        """
        self._pricing = pricing

    def add_callback(self, callback: CGCallback) -> None:
        """
        Add a callback function.

        Callbacks are called after each iteration with the ColumnGeneration
        instance and iteration info. Return False to stop the algorithm.

        Args:
            callback: Function taking (ColumnGeneration, CGIteration) -> bool
        """
        self._callbacks.append(callback)

    def add_initial_columns(self, columns: list[Column]) -> None:
        """
        Add initial columns for warm starting.

        Args:
            columns: List of columns to add
        """
        for col in columns:
            col_with_id = self._column_pool.add(col)
            if self._master is not None:
                self._master.add_column(col_with_id)

    # =========================================================================
    # Main Algorithm
    # =========================================================================

    def solve(self) -> CGSolution:
        """
        Run the column generation algorithm.

        Returns:
            CGSolution with results and statistics
        """
        start_time = time.time()

        # Initialize components
        self._initialize()

        # Add initial columns
        self._add_initial_columns()

        # Main CG loop
        solution = self._run_column_generation(start_time)

        # Optionally solve IP
        if self._config.solve_ip and solution.is_feasible:
            solution = self._solve_ip(solution, start_time)

        # Finalize
        solution.total_time = time.time() - start_time
        self._solution = solution
        self._is_solved = True

        return solution

    def _initialize(self) -> None:
        """Initialize master and pricing if not already set."""
        # Create master if not provided
        if self._master is None:
            if not HIGHS_AVAILABLE:
                raise RuntimeError(
                    "HiGHS is not available. Install it with: pip install highspy\n"
                    "Or provide a custom MasterProblem implementation."
                )
            self._master = HiGHSMasterProblem(
                self._problem,
                verbosity=1 if self._config.verbose else 0,
            )

            # Enable stabilization if configured
            if self._config.use_stabilization:
                self._master.enable_stabilization(
                    self._config.stabilization_method
                )

        # Create pricing if not provided
        if self._pricing is None:
            pricing_config = self._config.pricing_config or PricingConfig()
            self._pricing = LabelingAlgorithm(self._problem, pricing_config)

    def _add_initial_columns(self) -> None:
        """Add initial columns from problem or create artificial columns."""
        # Add columns from problem definition
        if self._problem.initial_columns:
            for col in self._problem.initial_columns:
                col_with_id = self._column_pool.add(col)
                self._master.add_column(col_with_id)

        # Add columns already in pool (from add_initial_columns called before solve)
        for col in self._column_pool.all_columns():
            if col.column_id is not None:
                # Already has ID, add to master
                self._master.add_column(col)

        # If no columns, create artificial columns
        if self._master.num_columns == 0:
            self._create_artificial_columns()

    def _create_artificial_columns(self) -> None:
        """
        Create artificial columns to ensure initial feasibility.

        Each artificial column covers exactly one item with high cost.
        This guarantees the master problem is feasible initially.
        """
        if self._config.verbose:
            print("Creating artificial columns...")

        big_m = 1e6  # Large cost for artificial columns

        # Track artificial column IDs for later checking
        self._artificial_column_ids: set = set()

        for constraint in self._problem.cover_constraints:
            # Create an artificial column covering just this item
            col = Column(
                arc_indices=(),  # No real path
                cost=big_m,
                covered_items=frozenset({constraint.item_id}),
                attributes={'artificial': True},
            )
            col_with_id = self._column_pool.add(col)
            self._master.add_column(col_with_id)
            self._artificial_column_ids.add(col_with_id.column_id)

        if self._config.verbose:
            print(f"  Added {self._problem.num_cover_constraints} artificial columns")

    def _get_artificial_usage(self, master_solution: MasterSolution) -> tuple[float, set]:
        """
        Compute total artificial column usage in LP solution.

        Returns:
            (total_usage, set of item_ids still using artificial)
        """
        if not hasattr(self, '_artificial_column_ids'):
            return 0.0, set()

        total = 0.0
        items_on_artificial = set()

        for col_id, val in master_solution.column_values.items():
            if val > 1e-10 and col_id in self._artificial_column_ids:
                total += val
                # Get which item this artificial covers
                col = self._column_pool.get(col_id)
                if col:
                    items_on_artificial.update(col.covered_items)

        return total, items_on_artificial

    def _run_column_generation(self, start_time: float) -> CGSolution:
        """
        Run the main column generation loop.

        Args:
            start_time: Algorithm start time

        Returns:
            CGSolution with LP results
        """
        iteration_history: list[CGIteration] = []
        total_master_time = 0.0
        total_pricing_time = 0.0

        best_objective = float('inf')
        no_improvement_count = 0

        iteration = 0
        status = CGStatus.NOT_SOLVED
        last_master_solution: Optional[MasterSolution] = None

        while True:
            iteration += 1

            # Check stopping criteria
            if self._config.max_iterations > 0 and iteration > self._config.max_iterations:
                status = CGStatus.ITERATION_LIMIT
                break

            elapsed = time.time() - start_time
            if self._config.max_time > 0 and elapsed >= self._config.max_time:
                status = CGStatus.TIME_LIMIT
                break

            if self._config.max_columns > 0 and self._column_pool.size >= self._config.max_columns:
                status = CGStatus.COLUMN_LIMIT
                break

            # Solve master problem
            master_start = time.time()
            master_solution = self._master.solve_lp()
            master_time = time.time() - master_start
            total_master_time += master_time

            last_master_solution = master_solution

            if not master_solution.is_optimal:
                if master_solution.is_infeasible:
                    status = CGStatus.INFEASIBLE
                elif master_solution.is_unbounded:
                    status = CGStatus.UNBOUNDED
                else:
                    status = CGStatus.ERROR
                break

            master_obj = master_solution.objective_value

            # Get dual values
            if self._config.use_stabilization:
                duals = self._master.get_dual_values()
                # Update stabilization center if improving
                if master_obj < best_objective:
                    self._master.update_stabilization_center(
                        self._master.get_raw_dual_values()
                    )
            else:
                duals = self._master.get_dual_values()

            # Solve pricing problem
            pricing_start = time.time()
            self._pricing.set_dual_values(duals)
            pricing_solution = self._pricing.solve()
            pricing_time = time.time() - pricing_start
            total_pricing_time += pricing_time

            # Get best reduced cost
            best_rc = pricing_solution.best_reduced_cost

            # Record iteration
            iter_info = CGIteration(
                iteration=iteration,
                master_objective=master_obj,
                best_reduced_cost=best_rc,
                num_columns_added=0,  # Will update below
                master_time=master_time,
                pricing_time=pricing_time,
                total_columns=self._column_pool.size,
            )

            # Check if we found improving columns
            if not pricing_solution.has_negative_reduced_cost:
                # No improving columns from standard pricing

                # Check if we need to continue for integral LP
                if self._config.require_integral_lp:
                    art_usage, items_on_art = self._get_artificial_usage(master_solution)
                    total_items = len(self._problem.cover_constraints)

                    if art_usage > self._config.artificial_tolerance * total_items:
                        # Still using too many artificials - try priority pricing
                        if self._config.verbose:
                            print(f"Iteration {iteration}: {len(items_on_art)} items on artificial, "
                                  f"continuing with priority pricing...")

                        # Set priority items for pricing
                        if hasattr(self._pricing, 'set_priority_items'):
                            self._pricing.set_priority_items(items_on_art)

                        # Retry pricing with relaxed threshold
                        old_threshold = self._pricing._config.reduced_cost_threshold
                        self._pricing._config.reduced_cost_threshold = 1e-3  # Accept near-zero RC
                        pricing_solution = self._pricing.solve()
                        self._pricing._config.reduced_cost_threshold = old_threshold

                        if pricing_solution.columns:
                            # Found columns for priority items
                            num_added = 0
                            for col in pricing_solution.columns:
                                col_with_id = self._column_pool.add(col)
                                self._master.add_column(col_with_id)
                                num_added += 1

                            iter_info.num_columns_added = num_added

                            if self._config.verbose:
                                print(f"  Added {num_added} priority columns")

                            iteration_history.append(iter_info)
                            self._invoke_callbacks(iter_info)
                            continue  # Continue the loop

                # Truly optimal - no more columns possible
                status = CGStatus.OPTIMAL

                # Shrink stabilization if using it
                if self._config.use_stabilization:
                    self._master.shrink_stabilization()

                if self._config.verbose:
                    print(f"Iteration {iteration}: No improving columns found. LP optimal.")

                iteration_history.append(iter_info)
                self._invoke_callbacks(iter_info)
                break

            # Add columns with negative reduced cost
            num_added = 0
            for col in pricing_solution.columns:
                col_with_id = self._column_pool.add(col)
                self._master.add_column(col_with_id)
                num_added += 1

            iter_info.num_columns_added = num_added

            # Check for improvement
            if master_obj < best_objective - self._config.improvement_tolerance:
                best_objective = master_obj
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self._config.max_no_improvement:
                status = CGStatus.NO_IMPROVEMENT
                if self._config.verbose:
                    print(f"Stopping: no improvement in {no_improvement_count} iterations")
                iteration_history.append(iter_info)
                self._invoke_callbacks(iter_info)
                break

            # Print progress
            if self._config.verbose:
                print(
                    f"Iteration {iteration}: "
                    f"obj={master_obj:.4f}, "
                    f"best_rc={best_rc:.4f}, "
                    f"added={num_added}, "
                    f"total={self._column_pool.size}"
                )

            # Record and invoke callbacks
            iteration_history.append(iter_info)
            if not self._invoke_callbacks(iter_info):
                status = CGStatus.FEASIBLE
                break

        # Build solution
        solution = self._build_solution(
            status=status,
            master_solution=last_master_solution,
            iteration_history=iteration_history,
            total_master_time=total_master_time,
            total_pricing_time=total_pricing_time,
        )

        return solution

    def _solve_ip(self, lp_solution: CGSolution, start_time: float) -> CGSolution:
        """
        Solve the IP after LP converges.

        Args:
            lp_solution: Solution from LP phase
            start_time: Algorithm start time

        Returns:
            Updated solution with IP results
        """
        if self._config.verbose:
            print("\nSolving IP...")

        ip_start = time.time()
        ip_solution = self._master.solve_ip()
        ip_time = time.time() - ip_start

        if ip_solution.is_optimal:
            lp_solution.ip_objective = ip_solution.objective_value
            lp_solution.objective_value = ip_solution.objective_value
            lp_solution.status = CGStatus.INTEGER_OPTIMAL

            # Update columns with IP values
            ip_columns = []
            for col_id, value in ip_solution.column_values.items():
                if value > 1e-6:
                    col = self._column_pool.get(col_id)
                    if col is not None:
                        ip_columns.append(col.with_value(value))
            lp_solution.columns = ip_columns

            if self._config.verbose:
                print(f"IP optimal: {ip_solution.objective_value:.4f}")

            # Compute gap
            if lp_solution.lp_objective is not None:
                lp_solution.gap = (
                    (ip_solution.objective_value - lp_solution.lp_objective)
                    / max(abs(ip_solution.objective_value), 1e-6)
                )
        else:
            if self._config.verbose:
                print(f"IP solve status: {ip_solution.status.name}")

        lp_solution.master_time += ip_time

        return lp_solution

    def _build_solution(
        self,
        status: CGStatus,
        master_solution: Optional[MasterSolution],
        iteration_history: list[CGIteration],
        total_master_time: float,
        total_pricing_time: float,
    ) -> CGSolution:
        """Build the CGSolution from components."""
        # Get columns with positive value
        active_columns = []
        if master_solution is not None and master_solution.has_solution:
            for col_id, value in master_solution.column_values.items():
                if value > 1e-6:
                    col = self._column_pool.get(col_id)
                    if col is not None:
                        active_columns.append(col.with_value(value))

        # Compute objective
        obj_value = None
        if master_solution is not None and master_solution.objective_value is not None:
            obj_value = master_solution.objective_value

        solution = CGSolution(
            status=status,
            objective_value=obj_value,
            lp_objective=obj_value,
            columns=active_columns,
            total_columns=self._column_pool.size,
            iterations=len(iteration_history),
            master_time=total_master_time,
            pricing_time=total_pricing_time,
            iteration_history=iteration_history,
        )

        return solution

    def _invoke_callbacks(self, iteration: CGIteration) -> bool:
        """
        Invoke all callbacks.

        Args:
            iteration: Current iteration info

        Returns:
            True to continue, False to stop
        """
        for callback in self._callbacks:
            if not callback(self, iteration):
                return False
        return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_iteration_history(self) -> list[CGIteration]:
        """
        Get the history of all iterations.

        Returns:
            List of CGIteration objects
        """
        if self._solution is None:
            return []
        return self._solution.iteration_history

    def reset(self) -> None:
        """
        Reset the solver for a new solve.

        Clears all state but keeps configuration and components.
        """
        self._column_pool = ColumnPool()
        self._is_solved = False
        self._solution = None

        # Re-initialize master with fresh model
        self._master = None
        self._initialize()

    def summary(self) -> str:
        """
        Return a human-readable summary.

        Returns:
            Summary string
        """
        lines = [
            f"ColumnGeneration: {self._problem.name}",
            "  Config:",
            f"    Max iterations: {self._config.max_iterations}",
            f"    Max time: {self._config.max_time}s",
            f"    Solve IP: {self._config.solve_ip}",
            f"    Stabilization: {self._config.use_stabilization}",
        ]

        if self._is_solved and self._solution:
            lines.extend([
                "",
                self._solution.summary(),
            ])
        else:
            lines.append("\n  Status: Not yet solved")

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "solved" if self._is_solved else "not solved"
        return f"ColumnGeneration(problem={self._problem.name!r}, {status})"
