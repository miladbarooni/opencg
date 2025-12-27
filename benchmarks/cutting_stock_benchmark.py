#!/usr/bin/env python3
"""
Cutting Stock Benchmark: OpenCG vs Direct Formulation vs Known Optima.

This script compares:
1. OpenCG column generation approach
2. Direct IP formulation (enumerate all patterns)
3. Known optimal solutions (from BPPLIB)

For cutting stock, the key metrics are:
- LP objective (lower bound quality)
- IP objective (solution quality)
- Time to LP optimality
- Time to IP optimality
- Number of columns generated vs total possible
"""

import time
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opencg.applications.cutting_stock import (
    CuttingStockInstance,
    solve_cutting_stock,
    CuttingStockSolution,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    instance_name: str
    num_items: int
    roll_width: int
    total_demand: int

    # OpenCG results
    opencg_lp: float
    opencg_ip: Optional[int]
    opencg_time: float
    opencg_iterations: int
    opencg_columns: int

    # Direct solve results (if available)
    direct_ip: Optional[int] = None
    direct_time: Optional[float] = None
    direct_num_patterns: Optional[int] = None

    # Known optimal (if available)
    known_optimal: Optional[int] = None

    # Bounds
    l2_lower_bound: int = 0

    @property
    def gap_to_optimal(self) -> Optional[float]:
        """Gap between our IP and known optimal."""
        if self.known_optimal and self.opencg_ip:
            return (self.opencg_ip - self.known_optimal) / self.known_optimal * 100
        return None

    @property
    def lp_gap(self) -> Optional[float]:
        """Integrality gap (LP vs IP)."""
        if self.opencg_ip and self.opencg_lp > 0:
            return (self.opencg_ip - self.opencg_lp) / self.opencg_lp * 100
        return None


def count_all_patterns(instance: CuttingStockInstance) -> int:
    """
    Count total number of feasible patterns (for reference).

    This shows how many columns would need to be enumerated
    for a direct formulation.
    """
    # Use recursive enumeration with memoization
    from functools import lru_cache

    W = int(instance.roll_width)
    sizes = [int(s) for s in instance.item_sizes]
    max_copies = [min(instance.max_copies(i), instance.item_demands[i])
                  for i in range(instance.num_items)]

    @lru_cache(maxsize=None)
    def count_patterns(remaining: int, item_idx: int) -> int:
        if item_idx >= instance.num_items:
            return 1  # Empty pattern counts

        count = 0
        for copies in range(max_copies[item_idx] + 1):
            used = copies * sizes[item_idx]
            if used <= remaining:
                count += count_patterns(remaining - used, item_idx + 1)
        return count

    # Subtract 1 for the empty pattern
    return count_patterns(W, 0) - 1


def solve_direct_ip(instance: CuttingStockInstance, time_limit: float = 60.0) -> Tuple[Optional[int], float, int]:
    """
    Solve cutting stock by enumerating ALL patterns and solving IP directly.

    This is the "naive" approach that commercial solvers would use
    without column generation. It's exponential in the number of items.

    Returns:
        (optimal_value, solve_time, num_patterns)
    """
    try:
        import highspy
    except ImportError:
        return None, 0.0, 0

    start_time = time.time()

    # Generate all feasible patterns
    patterns = []
    W = int(instance.roll_width)
    sizes = [int(s) for s in instance.item_sizes]
    n = instance.num_items
    max_copies = [min(instance.max_copies(i), instance.item_demands[i])
                  for i in range(n)]

    def generate_patterns(remaining: int, item_idx: int, current: List[int]):
        if item_idx >= n:
            if any(c > 0 for c in current):
                patterns.append(current.copy())
            return

        for copies in range(max_copies[item_idx] + 1):
            used = copies * sizes[item_idx]
            if used <= remaining:
                current.append(copies)
                generate_patterns(remaining - used, item_idx + 1, current)
                current.pop()

    generate_patterns(W, 0, [])
    num_patterns = len(patterns)

    if num_patterns > 100000:
        # Too many patterns, skip direct solve
        return None, time.time() - start_time, num_patterns

    # Build and solve IP
    highs = highspy.Highs()
    highs.setOptionValue('output_flag', False)
    highs.setOptionValue('time_limit', time_limit)
    highs.changeObjectiveSense(highspy.ObjSense.kMinimize)

    # Add demand constraints
    for i in range(n):
        highs.addRow(
            float(instance.item_demands[i]),
            highspy.kHighsInf,
            0, [], []
        )

    # Add pattern columns
    for p, pattern in enumerate(patterns):
        indices = [i for i in range(n) if pattern[i] > 0]
        values = [float(pattern[i]) for i in indices]
        highs.addCol(1.0, 0.0, highspy.kHighsInf, len(indices), indices, values)
        highs.changeColIntegrality(p, highspy.HighsVarType.kInteger)

    highs.run()
    solve_time = time.time() - start_time

    status = highs.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        info = highs.getInfo()
        return int(round(info.objective_function_value)), solve_time, num_patterns

    return None, solve_time, num_patterns


def run_benchmark(
    instance: CuttingStockInstance,
    known_optimal: Optional[int] = None,
    run_direct: bool = True,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run benchmark on a single instance."""

    # L2 lower bound
    total_area = sum(
        instance.item_sizes[i] * instance.item_demands[i]
        for i in range(instance.num_items)
    )
    l2_bound = math.ceil(total_area / instance.roll_width)

    # Solve with OpenCG
    if verbose:
        print(f"\n{'='*60}")
        print(f"Instance: {instance.name or 'unnamed'}")
        print(f"  Items: {instance.num_items}, Roll width: {instance.roll_width}")
        print(f"  L2 lower bound: {l2_bound}")
        print(f"{'='*60}")

    opencg_start = time.time()
    solution = solve_cutting_stock(
        instance,
        max_iterations=200,
        verbose=verbose,
        solve_ip=True,
    )
    opencg_time = time.time() - opencg_start

    # Direct solve (if requested and feasible)
    direct_ip = None
    direct_time = None
    direct_patterns = None

    if run_direct and instance.num_items <= 20:
        if verbose:
            print("\nRunning direct IP solve...")
        direct_ip, direct_time, direct_patterns = solve_direct_ip(instance)
        if verbose:
            if direct_ip:
                print(f"  Direct IP: {direct_ip} in {direct_time:.2f}s ({direct_patterns} patterns)")
            else:
                print(f"  Direct IP: failed/timeout ({direct_patterns} patterns)")
    elif run_direct:
        # Just count patterns
        direct_patterns = count_all_patterns(instance)
        if verbose:
            print(f"\nTotal feasible patterns: {direct_patterns} (too many for direct solve)")

    return BenchmarkResult(
        instance_name=instance.name or "unnamed",
        num_items=instance.num_items,
        roll_width=int(instance.roll_width),
        total_demand=instance.total_demand,
        opencg_lp=solution.lp_objective,
        opencg_ip=solution.num_rolls_ip,
        opencg_time=opencg_time,
        opencg_iterations=solution.iterations,
        opencg_columns=solution.num_columns,
        direct_ip=direct_ip,
        direct_time=direct_time,
        direct_num_patterns=direct_patterns,
        known_optimal=known_optimal,
        l2_lower_bound=l2_bound,
    )


def create_test_instances() -> List[Tuple[CuttingStockInstance, Optional[int]]]:
    """Create test instances with known optima."""
    instances = []

    # Classic Gilmore-Gomory example
    instances.append((
        CuttingStockInstance(
            roll_width=100,
            item_sizes=[45, 36, 31, 14],
            item_demands=[97, 610, 395, 211],
            name="gilmore_gomory_1"
        ),
        None  # Unknown optimal
    ))

    # Small instance with known optimal
    instances.append((
        CuttingStockInstance(
            roll_width=10,
            item_sizes=[6, 4, 3, 2],
            item_demands=[2, 3, 4, 5],
            name="small_1"
        ),
        5  # Known optimal
    ))

    # Medium instance
    instances.append((
        CuttingStockInstance(
            roll_width=100,
            item_sizes=[70, 50, 40, 30, 20, 10],
            item_demands=[10, 20, 30, 40, 50, 60],
            name="medium_1"
        ),
        None
    ))

    # Instance where LP = IP (tight)
    instances.append((
        CuttingStockInstance(
            roll_width=100,
            item_sizes=[50, 25, 20],
            item_demands=[100, 200, 250],
            name="tight_lp_1"
        ),
        None
    ))

    # Larger instance
    instances.append((
        CuttingStockInstance(
            roll_width=1000,
            item_sizes=[465, 333, 287, 241, 198, 167, 145, 122, 99, 81],
            item_demands=[15, 22, 31, 18, 25, 33, 41, 28, 35, 42],
            name="large_1"
        ),
        None
    ))

    return instances


def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Header
    print(f"{'Instance':<20} {'Items':>6} {'L2 LB':>6} {'LP':>8} {'IP':>6} "
          f"{'Opt':>6} {'Gap%':>6} {'Time':>8} {'Iters':>6} {'Cols':>8} {'Direct':>10}")
    print("-" * 100)

    for r in results:
        gap_str = f"{r.gap_to_optimal:.1f}" if r.gap_to_optimal is not None else "-"
        opt_str = str(r.known_optimal) if r.known_optimal else "-"
        direct_str = f"{r.direct_ip}({r.direct_time:.1f}s)" if r.direct_ip else "-"

        print(f"{r.instance_name:<20} {r.num_items:>6} {r.l2_lower_bound:>6} "
              f"{r.opencg_lp:>8.2f} {r.opencg_ip or '-':>6} {opt_str:>6} {gap_str:>6} "
              f"{r.opencg_time:>7.2f}s {r.opencg_iterations:>6} {r.opencg_columns:>8} {direct_str:>10}")

    print("-" * 100)

    # Summary stats
    total_time = sum(r.opencg_time for r in results)
    total_cols = sum(r.opencg_columns for r in results)

    print(f"\nSummary:")
    print(f"  Total instances: {len(results)}")
    print(f"  Total OpenCG time: {total_time:.2f}s")
    print(f"  Total columns generated: {total_cols}")

    # Check optimality
    optimal_count = sum(1 for r in results if r.known_optimal and r.opencg_ip == r.known_optimal)
    known_count = sum(1 for r in results if r.known_optimal)
    if known_count > 0:
        print(f"  Optimal solutions found: {optimal_count}/{known_count}")

    # LP = IP cases
    tight_count = sum(1 for r in results
                      if r.opencg_ip and abs(r.opencg_lp - r.opencg_ip) < 0.01)
    print(f"  Instances with LP = IP: {tight_count}/{len(results)}")


def benchmark_scaling(max_items: int = 100, step: int = 10):
    """
    Benchmark scaling: how does performance change with problem size?
    """
    print("\n" + "=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)
    print(f"{'Items':>8} {'Time(s)':>10} {'Iters':>8} {'Columns':>10} {'Patterns*':>12}")
    print("-" * 60)

    import random
    random.seed(42)

    for n_items in range(10, max_items + 1, step):
        # Generate random instance
        roll_width = 1000
        sizes = [random.randint(50, 400) for _ in range(n_items)]
        demands = [random.randint(10, 50) for _ in range(n_items)]

        instance = CuttingStockInstance(
            roll_width=roll_width,
            item_sizes=sizes,
            item_demands=demands,
            name=f"random_{n_items}"
        )

        # Solve with OpenCG
        start = time.time()
        solution = solve_cutting_stock(instance, max_iterations=200, verbose=False)
        elapsed = time.time() - start

        # Estimate total patterns (rough approximation)
        avg_fit = roll_width / (sum(sizes) / n_items)
        est_patterns = int(math.pow(avg_fit, n_items / 2))  # Very rough

        print(f"{n_items:>8} {elapsed:>10.2f} {solution.iterations:>8} "
              f"{solution.num_columns:>10} {est_patterns:>12,}")


def run_bpplib_benchmark(benchmark_set: str, max_instances: int = 20, verbose: bool = False):
    """
    Run benchmark on BPPLIB instances.

    Args:
        benchmark_set: One of 'scholl1', 'scholl2', 'scholl3', 'schwerin1', 'schwerin2', 'hard28', 'waescher'
        max_instances: Maximum number of instances to run
        verbose: Verbose output
    """
    base_path = Path(__file__).parent.parent / "data" / "bpplib" / "Instances" / "Benchmarks" / "extracted"

    # Map benchmark set names to paths
    path_map = {
        'scholl1': base_path / "Scholl_CSP" / "Scholl_1",
        'scholl2': base_path / "Scholl_CSP" / "Scholl_2",
        'scholl3': base_path / "Scholl_CSP" / "Scholl_3",
        'schwerin1': base_path / "Schwerin_CSP" / "Schwerin_1",
        'schwerin2': base_path / "Schwerin_CSP" / "Schwerin_2",
        'hard28': base_path / "Hard28_CSP",
        'waescher': base_path / "Waescher_CSP",
    }

    if benchmark_set.lower() not in path_map:
        print(f"Unknown benchmark set: {benchmark_set}")
        print(f"Available sets: {list(path_map.keys())}")
        return

    instance_path = path_map[benchmark_set.lower()]
    if not instance_path.exists():
        print(f"Benchmark path not found: {instance_path}")
        print("Please extract the RAR files first.")
        return

    instances = sorted(instance_path.glob("*.txt"))[:max_instances]

    if not instances:
        print(f"No instances found in {instance_path}")
        return

    print(f"\n{'='*90}")
    print(f"BPPLIB Benchmark: {benchmark_set.upper()}")
    print(f"Path: {instance_path}")
    print(f"Running {len(instances)} instances")
    print(f"{'='*90}")
    print(f"{'Instance':<25} {'Items':>6} {'Width':>8} {'L2':>6} {'LP':>10} {'IP':>6} "
          f"{'Gap%':>6} {'Time':>8} {'Iters':>6}")
    print("-" * 90)

    total_time = 0
    total_optimal = 0  # LP = IP
    total_instances = 0

    for inst_path in instances:
        instance = CuttingStockInstance.from_bpplib(str(inst_path))

        # Compute L2 bound
        total_area = sum(
            instance.item_sizes[i] * instance.item_demands[i]
            for i in range(instance.num_items)
        )
        l2_bound = math.ceil(total_area / instance.roll_width)

        start = time.time()
        solution = solve_cutting_stock(instance, max_iterations=200, verbose=verbose)
        elapsed = time.time() - start
        total_time += elapsed
        total_instances += 1

        # Calculate integrality gap
        gap = ""
        if solution.num_rolls_ip and solution.lp_objective > 0:
            gap_val = (solution.num_rolls_ip - solution.lp_objective) / solution.lp_objective * 100
            gap = f"{gap_val:.1f}"
            if gap_val < 0.01:
                total_optimal += 1

        print(f"{instance.name:<25} {instance.num_items:>6} {int(instance.roll_width):>8} "
              f"{l2_bound:>6} {solution.lp_objective:>10.2f} {solution.num_rolls_ip or '-':>6} "
              f"{gap:>6} {elapsed:>7.2f}s {solution.iterations:>6}")

    print("-" * 90)
    print(f"Summary: {total_instances} instances, {total_time:.2f}s total, "
          f"{total_optimal}/{total_instances} optimal (LP=IP)")


def main():
    """Run the benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Cutting Stock Benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--no-direct", action="store_true", help="Skip direct IP solve")
    parser.add_argument("--bpplib", type=str, help="Path to BPPLIB instance file")
    parser.add_argument("--benchmark", type=str,
                       help="BPPLIB benchmark set: scholl1, scholl2, scholl3, schwerin1, schwerin2, hard28, waescher")
    parser.add_argument("--max-instances", type=int, default=20,
                       help="Maximum instances to run for benchmark sets")
    args = parser.parse_args()

    if args.benchmark:
        # Run on BPPLIB benchmark set
        run_bpplib_benchmark(args.benchmark, args.max_instances, args.verbose)
    elif args.bpplib:
        # Run on single BPPLIB instance
        instance = CuttingStockInstance.from_bpplib(args.bpplib)
        result = run_benchmark(instance, run_direct=not args.no_direct, verbose=args.verbose)
        print_results_table([result])
    elif args.scaling:
        # Run scaling benchmark
        benchmark_scaling()
    else:
        # Run on test instances
        instances = create_test_instances()
        results = []

        for instance, known_opt in instances:
            result = run_benchmark(
                instance,
                known_optimal=known_opt,
                run_direct=not args.no_direct,
                verbose=args.verbose,
            )
            results.append(result)

        print_results_table(results)


if __name__ == "__main__":
    main()
