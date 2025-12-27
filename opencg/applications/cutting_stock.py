"""
Cutting Stock Problem (1D Bin Packing) via Column Generation.

The Cutting Stock Problem (CSP) asks: given a set of items with sizes and demands,
and rolls of fixed width, find the minimum number of rolls needed to cut all items.

This is a classic application of column generation where:
- Master problem: Select patterns (columns) to minimize roll usage
- Pricing problem: Knapsack to find patterns with negative reduced cost

Mathematical Formulation:
------------------------
Master Problem (Set Covering):
    min  sum_p x_p                    (minimize number of rolls)
    s.t. sum_p a_ip * x_p >= d_i      (meet demand for item i)
         x_p >= 0, integer

Where:
- x_p = number of times to use pattern p
- a_ip = number of copies of item i in pattern p
- d_i = demand for item i

Pricing Subproblem (Bounded Knapsack):
    max  sum_i pi_i * y_i             (maximize dual value)
    s.t. sum_i s_i * y_i <= W         (respect roll width)
         0 <= y_i <= d_i              (respect demand bounds)
         y_i integer

Where:
- pi_i = dual value for item i (from master)
- y_i = copies of item i in new pattern
- s_i = size of item i
- W = roll width (capacity)

A pattern has negative reduced cost if: 1 - sum_i pi_i * y_i < 0

Usage:
------
    from opencg.applications import CuttingStockInstance, solve_cutting_stock

    # Define instance
    instance = CuttingStockInstance(
        roll_width=100,
        item_sizes=[45, 36, 31, 14],
        item_demands=[97, 610, 395, 211],
    )

    # Solve
    solution = solve_cutting_stock(instance)
    print(f"Rolls needed: {solution.num_rolls}")
    for pattern, count in solution.patterns:
        print(f"  Use pattern {pattern} x {count}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from opencg.core.column import Column
from opencg.core.problem import Problem, CoverConstraint, CoverType, ObjectiveSense
from opencg.core.network import Network
from opencg.master.base import MasterProblem
from opencg.master.solution import MasterSolution, SolutionStatus
from opencg.pricing.base import PricingProblem, PricingConfig, PricingSolution, PricingStatus


@dataclass
class CuttingStockInstance:
    """
    A Cutting Stock Problem instance.

    Attributes:
        roll_width: Width of each roll (capacity)
        item_sizes: Size of each item type
        item_demands: Number of each item type needed
        item_names: Optional names for items
        name: Optional instance name
    """
    roll_width: float
    item_sizes: List[float]
    item_demands: List[int]
    item_names: Optional[List[str]] = None
    name: Optional[str] = None

    def __post_init__(self):
        if len(self.item_sizes) != len(self.item_demands):
            raise ValueError("item_sizes and item_demands must have same length")
        if self.item_names is None:
            self.item_names = [f"item_{i}" for i in range(len(self.item_sizes))]
        if len(self.item_names) != len(self.item_sizes):
            raise ValueError("item_names must have same length as item_sizes")

    @property
    def num_items(self) -> int:
        """Number of item types."""
        return len(self.item_sizes)

    def max_copies(self, item_idx: int) -> int:
        """Maximum copies of an item that fit in one roll."""
        return int(self.roll_width // self.item_sizes[item_idx])

    @property
    def total_demand(self) -> int:
        """Total number of items demanded."""
        return sum(self.item_demands)

    @classmethod
    def from_bpplib(cls, filepath: str) -> 'CuttingStockInstance':
        """
        Load a cutting stock instance from BPPLIB format.

        BPPLIB format (for CSP):
            Line 1: Number of item types
            Line 2: Roll/bin capacity
            Lines 3+: size<tab>demand for each item type

        Args:
            filepath: Path to the BPPLIB .txt file

        Returns:
            CuttingStockInstance

        Example:
            >>> instance = CuttingStockInstance.from_bpplib("data/bpplib/Scholl_1/N1C1W1_A.txt")
        """
        import os
        name = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        num_types = int(lines[0])
        capacity = int(lines[1])

        sizes = []
        demands = []
        for i in range(2, 2 + num_types):
            parts = lines[i].split()
            sizes.append(int(parts[0]))
            demands.append(int(parts[1]))

        return cls(
            roll_width=capacity,
            item_sizes=sizes,
            item_demands=demands,
            name=name,
        )


def create_cutting_stock_problem(instance: CuttingStockInstance) -> Problem:
    """
    Create a Problem instance for cutting stock.

    The Problem uses:
    - Empty network (no graph structure needed)
    - Cover constraints for each item with rhs = demand
    - Columns store patterns in 'pattern' attribute

    Args:
        instance: The cutting stock instance

    Returns:
        Problem configured for cutting stock
    """
    # Create minimal network (just source and sink, no arcs)
    network = Network()
    network.add_source()
    network.add_sink()

    # Create cover constraints for each item
    # Each item must be covered at least demand times
    cover_constraints = []
    for i in range(instance.num_items):
        cover_constraints.append(CoverConstraint(
            item_id=i,
            name=instance.item_names[i],
            rhs=instance.item_demands[i],
            is_equality=False,  # >= constraint (covering)
        ))

    # Create problem
    problem = Problem(
        name="CuttingStock",
        network=network,
        resources=[],  # No resources needed
        cover_constraints=cover_constraints,
        cover_type=CoverType.SET_COVERING,
        objective_sense=ObjectiveSense.MINIMIZE,
        metadata={
            'roll_width': instance.roll_width,
            'item_sizes': instance.item_sizes,
            'item_demands': instance.item_demands,
        }
    )

    return problem


class CuttingStockMaster:
    """
    Master problem for Cutting Stock using HiGHS directly.

    This bypasses the standard MasterProblem to properly handle
    non-unit coefficients in the constraint matrix.

    The constraint matrix entry a_ip = number of copies of item i in pattern p.
    """

    def __init__(self, instance: CuttingStockInstance, verbosity: int = 0):
        """
        Initialize cutting stock master.

        Args:
            instance: The cutting stock instance
            verbosity: HiGHS output level
        """
        try:
            import highspy
        except ImportError:
            raise ImportError("HiGHS is required. Install with: pip install highspy")

        self._instance = instance
        self._highs = highspy.Highs()
        self._highs.setOptionValue('output_flag', verbosity > 0)
        self._highs.setOptionValue('log_to_console', verbosity > 0)

        # Minimize objective
        self._highs.changeObjectiveSense(highspy.ObjSense.kMinimize)

        # Add demand constraints: sum_p a_ip * x_p >= d_i
        for i in range(instance.num_items):
            # >= demand constraint (initially empty)
            self._highs.addRow(
                float(instance.item_demands[i]),  # lower bound = demand
                highspy.kHighsInf,                # upper bound = infinity
                0, [], []                         # no columns yet
            )

        # Track columns
        self._columns: List[Column] = []
        self._column_to_solver_idx: Dict[int, int] = {}

    @property
    def num_columns(self) -> int:
        return len(self._columns)

    def add_column(self, column: Column) -> int:
        """Add a pattern column to the master."""
        import highspy

        pattern = column.attributes.get('pattern', {})

        # Build constraint coefficients
        indices = []
        values = []
        for item_id, count in pattern.items():
            if count > 0:
                indices.append(item_id)
                values.append(float(count))

        # Add column: cost=1, bounds=[0, inf]
        self._highs.addCol(
            1.0,                    # cost (one roll)
            0.0,                    # lower bound
            highspy.kHighsInf,      # upper bound
            len(indices),
            indices,
            values
        )

        solver_idx = self._highs.getNumCol() - 1
        self._columns.append(column)
        self._column_to_solver_idx[column.column_id] = solver_idx

        return solver_idx

    def solve_lp(self) -> MasterSolution:
        """Solve LP relaxation."""
        import highspy

        self._highs.run()
        status = self._highs.getModelStatus()

        if status == highspy.HighsModelStatus.kOptimal:
            sol_status = 'OPTIMAL'
        elif status == highspy.HighsModelStatus.kInfeasible:
            sol_status = 'INFEASIBLE'
        else:
            sol_status = 'ERROR'

        solution = MasterSolution(
            status=getattr(SolutionStatus, sol_status, SolutionStatus.ERROR),
            solve_time=0.0,
            num_columns=self.num_columns,
        )

        if sol_status == 'OPTIMAL':
            info = self._highs.getInfo()
            solution.objective_value = info.objective_function_value

            sol = self._highs.getSolution()
            for col in self._columns:
                solver_idx = self._column_to_solver_idx[col.column_id]
                value = sol.col_value[solver_idx]
                if abs(value) > 1e-10:
                    solution.column_values[col.column_id] = value

            # Get duals
            for i in range(self._instance.num_items):
                solution.dual_values[i] = sol.row_dual[i]

        return solution

    def solve_ip(self) -> MasterSolution:
        """Solve as integer program."""
        import highspy

        # Set all columns to integer
        num_cols = self._highs.getNumCol()
        for i in range(num_cols):
            self._highs.changeColIntegrality(i, highspy.HighsVarType.kInteger)

        self._highs.run()
        status = self._highs.getModelStatus()

        if status == highspy.HighsModelStatus.kOptimal:
            sol_status = 'OPTIMAL'
        else:
            sol_status = 'ERROR'

        solution = MasterSolution(
            status=getattr(SolutionStatus, sol_status, SolutionStatus.ERROR),
            solve_time=0.0,
            num_columns=self.num_columns,
        )

        if sol_status == 'OPTIMAL':
            info = self._highs.getInfo()
            solution.objective_value = info.objective_function_value

            sol = self._highs.getSolution()
            for col in self._columns:
                solver_idx = self._column_to_solver_idx[col.column_id]
                value = sol.col_value[solver_idx]
                if abs(value) > 0.5:
                    solution.column_values[col.column_id] = round(value)

        # Reset to continuous for future LP solves
        for i in range(num_cols):
            self._highs.changeColIntegrality(i, highspy.HighsVarType.kContinuous)

        return solution

    def get_dual_values(self) -> Dict[int, float]:
        """Get dual values from last LP solve."""
        sol = self._highs.getSolution()
        return {i: sol.row_dual[i] for i in range(self._instance.num_items)}

    def get_column(self, column_id: int) -> Optional[Column]:
        """Get a column by ID."""
        for col in self._columns:
            if col.column_id == column_id:
                return col
        return None


class CuttingStockPricing(PricingProblem):
    """
    Pricing problem for Cutting Stock (Bounded Knapsack).

    Finds patterns (columns) with negative reduced cost by solving:

        max  sum_i pi_i * y_i
        s.t. sum_i s_i * y_i <= W
             0 <= y_i <= max_copies_i
             y_i integer

    Uses dynamic programming for exact solution.
    """

    def __init__(
        self,
        instance: CuttingStockInstance,
        problem: Problem,
        config: Optional[PricingConfig] = None,
    ):
        """
        Initialize pricing for cutting stock.

        Args:
            instance: The cutting stock instance
            problem: The Problem instance
            config: Pricing configuration
        """
        super().__init__(problem, config)
        self._instance = instance
        self._roll_width = instance.roll_width
        self._item_sizes = instance.item_sizes
        self._item_demands = instance.item_demands
        self._num_items = instance.num_items

        # Discretize if needed (for DP)
        self._use_dp = all(
            isinstance(s, int) or s == int(s)
            for s in instance.item_sizes
        ) and isinstance(instance.roll_width, int) or instance.roll_width == int(instance.roll_width)

    def _solve_impl(self) -> PricingSolution:
        """Solve the knapsack pricing problem."""
        start_time = time.time()

        # Get dual values
        duals = [self._dual_values.get(i, 0.0) for i in range(self._num_items)]

        # Solve knapsack
        if self._use_dp:
            pattern, total_value = self._solve_knapsack_dp(duals)
        else:
            pattern, total_value = self._solve_knapsack_greedy(duals)

        # Reduced cost = 1 (pattern cost) - dual value
        reduced_cost = 1.0 - total_value

        solve_time = time.time() - start_time

        # Check if pattern has negative reduced cost
        threshold = self._config.reduced_cost_threshold or -1e-6
        if reduced_cost < threshold and any(v > 0 for v in pattern.values()):
            # Create column for this pattern
            column = self._create_column(pattern, reduced_cost)
            return PricingSolution(
                status=PricingStatus.COLUMNS_FOUND,
                columns=[column],
                best_reduced_cost=reduced_cost,
                solve_time=solve_time,
                iterations=1,
            )
        else:
            return PricingSolution(
                status=PricingStatus.NO_COLUMNS,
                columns=[],
                best_reduced_cost=reduced_cost,
                solve_time=solve_time,
                iterations=1,
            )

    def _solve_knapsack_dp(self, duals: List[float]) -> Tuple[Dict[int, int], float]:
        """
        Solve bounded knapsack using dynamic programming with backtracking.

        This optimized version only stores values during DP (not patterns),
        then reconstructs the pattern at the end via backtracking.
        This is much faster for large instances.

        Returns:
            (pattern, total_value) where pattern is {item_id: count}
        """
        W = int(self._roll_width)
        n = self._num_items

        # Precompute item info: (original_idx, size, value, max_copies)
        # We'll create "virtual items" using binary decomposition
        virtual_items = []  # (original_idx, size, value, copies_in_this_item)

        for i in range(n):
            size = int(self._item_sizes[i])
            value = duals[i]
            max_copies = min(
                self._instance.max_copies(i),
                self._item_demands[i]
            )

            if size <= 0 or value <= 0 or max_copies <= 0:
                continue

            # Binary decomposition: represent max_copies as sum of powers of 2
            # e.g., max_copies=13 -> items with copies 1, 2, 4, 6 (1+2+4+6=13)
            copies_left = max_copies
            k = 1
            while copies_left > 0:
                take = min(k, copies_left)
                virtual_items.append((i, size * take, value * take, take))
                copies_left -= take
                k *= 2

        if not virtual_items:
            return {}, 0.0

        num_virtual = len(virtual_items)

        # DP: dp[w] = max value achievable with capacity w
        dp = [0.0] * (W + 1)

        # choice[w] = index of last virtual item added to achieve dp[w], or -1
        choice = [-1] * (W + 1)

        # Process each virtual item (0-1 knapsack style)
        for vi, (orig_idx, item_size, item_value, item_copies) in enumerate(virtual_items):
            # Process in reverse to ensure each virtual item used at most once
            for w in range(W, item_size - 1, -1):
                new_value = dp[w - item_size] + item_value
                if new_value > dp[w]:
                    dp[w] = new_value
                    choice[w] = vi

        # Backtrack to reconstruct the pattern
        pattern = {}
        w = W
        while w > 0 and choice[w] >= 0:
            vi = choice[w]
            orig_idx, item_size, item_value, item_copies = virtual_items[vi]
            pattern[orig_idx] = pattern.get(orig_idx, 0) + item_copies
            w -= item_size

        return pattern, dp[W]

    def _solve_knapsack_greedy(self, duals: List[float]) -> Tuple[Dict[int, int], float]:
        """
        Solve knapsack using greedy heuristic (for non-integer sizes).

        Returns:
            (pattern, total_value) where pattern is {item_id: count}
        """
        W = self._roll_width
        pattern = {}
        remaining = W
        total_value = 0.0

        # Sort by value density (dual / size)
        items = []
        for i in range(self._num_items):
            if self._item_sizes[i] > 0 and duals[i] > 0:
                density = duals[i] / self._item_sizes[i]
                items.append((density, i))

        items.sort(reverse=True)

        for _, i in items:
            size = self._item_sizes[i]
            value = duals[i]
            max_copies = min(
                int(remaining // size),
                self._item_demands[i]
            )

            if max_copies > 0:
                pattern[i] = max_copies
                remaining -= max_copies * size
                total_value += max_copies * value

        return pattern, total_value

    def _create_column(self, pattern: Dict[int, int], reduced_cost: float) -> Column:
        """Create a Column from a pattern."""
        # covered_items = items that appear in pattern
        covered = frozenset(i for i, count in pattern.items() if count > 0)

        # Cost is 1 (one roll used)
        return Column(
            arc_indices=(),  # No arcs for cutting stock
            cost=1.0,
            covered_items=covered,
            reduced_cost=reduced_cost,
            attributes={'pattern': pattern.copy()},
        )


@dataclass
class CuttingStockSolution:
    """Solution to a cutting stock problem."""
    num_rolls: float  # LP relaxation may be fractional
    num_rolls_ip: Optional[int]  # Integer solution
    patterns: List[Tuple[Dict[int, int], float]]  # (pattern, count)
    lp_objective: float
    ip_objective: Optional[float]
    solve_time: float
    iterations: int
    num_columns: int
    lower_bound: Optional[float] = None  # L2 lower bound


def _generate_ffd_patterns(instance: CuttingStockInstance) -> List[Dict[int, int]]:
    """
    Generate patterns using First Fit Decreasing heuristic.

    FFD sorts items by size (decreasing) and greedily packs them.
    This provides good initial columns for column generation.

    Returns:
        List of patterns (each pattern is {item_id: count})
    """
    # Create list of (size, item_id, demand_remaining)
    items = []
    for i in range(instance.num_items):
        for _ in range(instance.item_demands[i]):
            items.append((instance.item_sizes[i], i))

    # Sort by size descending
    items.sort(reverse=True)

    patterns = []
    W = instance.roll_width

    for size, item_id in items:
        # Try to fit in existing pattern
        placed = False
        for pattern in patterns:
            used = sum(instance.item_sizes[i] * cnt for i, cnt in pattern.items())
            if used + size <= W:
                pattern[item_id] = pattern.get(item_id, 0) + 1
                placed = True
                break

        if not placed:
            # Create new pattern
            patterns.append({item_id: 1})

    return patterns


def _compute_l2_lower_bound(instance: CuttingStockInstance) -> float:
    """
    Compute L2 (continuous) lower bound for cutting stock.

    L2 = ceil(sum of all item areas / roll capacity)

    This is a simple but tight lower bound.
    """
    total_area = sum(
        instance.item_sizes[i] * instance.item_demands[i]
        for i in range(instance.num_items)
    )
    import math
    return math.ceil(total_area / instance.roll_width)


def solve_cutting_stock(
    instance: CuttingStockInstance,
    max_iterations: int = 100,
    verbose: bool = False,
    solve_ip: bool = True,
    use_ffd_init: bool = True,
) -> CuttingStockSolution:
    """
    Solve a cutting stock problem using column generation.

    Args:
        instance: The problem instance
        max_iterations: Maximum CG iterations
        verbose: Print progress
        solve_ip: Whether to solve IP after CG
        use_ffd_init: Use FFD heuristic for initial columns (recommended)

    Returns:
        CuttingStockSolution with results
    """
    start_time = time.time()

    # Compute lower bound
    lower_bound = _compute_l2_lower_bound(instance)
    if verbose:
        print(f"L2 lower bound: {lower_bound}")

    # Create problem (for pricing)
    problem = create_cutting_stock_problem(instance)

    # Create specialized master for cutting stock
    master = CuttingStockMaster(instance, verbosity=1 if verbose else 0)

    # Create pricing
    pricing = CuttingStockPricing(instance, problem)

    # Generate initial columns
    next_col_id = 0

    if use_ffd_init:
        # Use FFD patterns for better initial solution
        ffd_patterns = _generate_ffd_patterns(instance)
        if verbose:
            print(f"FFD heuristic: {len(ffd_patterns)} patterns (upper bound)")

        for pattern in ffd_patterns:
            covered = frozenset(i for i, count in pattern.items() if count > 0)
            col = Column(
                arc_indices=(),
                cost=1.0,
                covered_items=covered,
                column_id=next_col_id,
                attributes={'pattern': pattern.copy()},
            )
            master.add_column(col)
            next_col_id += 1

    # Also add trivial patterns (needed for feasibility)
    for i in range(instance.num_items):
        max_in_roll = instance.max_copies(i)
        if max_in_roll > 0:
            pattern = {i: max_in_roll}
            col = Column(
                arc_indices=(),
                cost=1.0,
                covered_items=frozenset([i]),
                column_id=next_col_id,
                attributes={'pattern': pattern},
            )
            master.add_column(col)
            next_col_id += 1

    # Column generation loop
    iterations = 0
    lp_sol = None

    for iteration in range(max_iterations):
        iterations = iteration + 1

        # Solve LP
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            if verbose:
                print(f"LP not optimal: {lp_sol.status}")
            break

        if verbose:
            print(f"Iter {iteration}: obj={lp_sol.objective_value:.4f}, cols={master.num_columns}")

        # Get duals and run pricing
        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        # Check for new columns
        if not pricing_sol.columns:
            if verbose:
                print("Converged - no columns with negative reduced cost")
            break

        # Add new columns
        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

    # Extract LP solution
    lp_objective = lp_sol.objective_value if lp_sol else 0
    patterns = []
    if lp_sol:
        for col_id, value in lp_sol.column_values.items():
            if value > 1e-6:
                col = master.get_column(col_id)
                if col:
                    pattern = col.attributes.get('pattern', {})
                    patterns.append((pattern, value))

    # Solve IP if requested
    ip_objective = None
    ip_rolls = None
    if solve_ip and lp_sol:
        ip_sol = master.solve_ip()
        if ip_sol.status.name == 'OPTIMAL':
            ip_objective = ip_sol.objective_value
            ip_rolls = int(round(ip_objective))

            # Update patterns with IP values
            patterns = []
            for col_id, value in ip_sol.column_values.items():
                if value > 0.5:  # Integer solution
                    col = master.get_column(col_id)
                    if col:
                        pattern = col.attributes.get('pattern', {})
                        patterns.append((pattern, int(round(value))))

    solve_time = time.time() - start_time

    return CuttingStockSolution(
        num_rolls=lp_objective,
        num_rolls_ip=ip_rolls,
        patterns=patterns,
        lp_objective=lp_objective,
        ip_objective=ip_objective,
        solve_time=solve_time,
        iterations=iterations,
        num_columns=master.num_columns,
        lower_bound=lower_bound,
    )
