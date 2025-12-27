#!/usr/bin/env python3
"""
Benchmark script for Cutting Stock Problem on BPPLIB instances.

BPPLIB is the standard benchmark library for bin packing and cutting stock problems.
Reference: Delorme, Iori & Martello (2018) "BPPLIB: a library for bin packing and cutting stock problems"
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from opencg.applications.cutting_stock import CuttingStockInstance, solve_cutting_stock


def load_optimal_solutions():
    """Load known optimal solutions for benchmarks."""
    # From BPPLIB solutions
    # These are selected instances with known optimal values
    return {
        # Scholl_1 instances (n=50, capacity=100)
        'N1C1W1_A': 25, 'N1C1W1_B': 31, 'N1C1W1_C': 20,
        'N1C1W1_D': 28, 'N1C1W1_E': 26, 'N1C1W1_F': 27,
        # Wäscher instances (hard instances)
        'Waescher_TEST0005': 42, 'Waescher_TEST0014': 27,
        # Hard28 instances (very challenging)
        'Hard28_BPP13': 55, 'Hard28_BPP14': 45,
    }


def benchmark_instance(filepath: str, optimal: int = None, verbose: bool = False):
    """Benchmark a single instance."""
    instance = CuttingStockInstance.from_bpplib(filepath)

    start = time.time()
    solution = solve_cutting_stock(instance, max_iterations=100, verbose=verbose)
    elapsed = time.time() - start

    # Use L2 bound as lower bound, or optimal if known
    lb = solution.lower_bound or 0
    if optimal:
        lb = max(lb, optimal)

    gap = None
    if lb > 0 and solution.num_rolls_ip:
        gap = 100.0 * (solution.num_rolls_ip - lb) / lb

    return {
        'name': instance.name,
        'n_types': instance.num_items,
        'capacity': int(instance.roll_width),
        'total_demand': instance.total_demand,
        'l2_lb': solution.lower_bound,
        'lp_obj': solution.lp_objective,
        'ip_obj': solution.num_rolls_ip,
        'optimal': optimal,
        'gap': gap,
        'time': elapsed,
        'iterations': solution.iterations,
        'columns': solution.num_columns,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Cutting Stock on BPPLIB')
    parser.add_argument('--dataset', choices=['scholl1', 'waescher', 'hard28', 'all'],
                        default='scholl1', help='Which dataset to run')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of instances')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    base_path = Path(__file__).parent / 'data' / 'bpplib' / 'Instances' / 'Benchmarks' / 'extracted'

    datasets = {
        'scholl1': base_path / 'Scholl_CSP' / 'Scholl_1',
        'waescher': base_path / 'Waescher_CSP',
        'hard28': base_path / 'Hard28_CSP',
    }

    if args.dataset == 'all':
        paths = list(datasets.values())
    else:
        paths = [datasets[args.dataset]]

    optimal_solutions = load_optimal_solutions()

    results = []
    for dataset_path in paths:
        instances = sorted(dataset_path.glob('*.txt'))
        if args.limit:
            instances = instances[:args.limit]

        print(f"\n{'='*90}")
        print(f"Dataset: {dataset_path.name}")
        print(f"{'='*90}")
        print(f"{'Instance':<25} {'n':>5} {'C':>6} {'L2':>5} {'LP':>8} {'IP':>6} {'Gap%':>6} {'Time':>7} {'Cols':>6}")
        print('-' * 90)

        for instance_path in instances:
            name = instance_path.stem
            optimal = optimal_solutions.get(name)

            result = benchmark_instance(str(instance_path), optimal, args.verbose)
            results.append(result)

            gap_str = f"{result['gap']:.1f}" if result['gap'] is not None else '-'
            l2_str = str(result['l2_lb']) if result['l2_lb'] else '-'

            print(f"{result['name']:<25} {result['n_types']:>5} {result['capacity']:>6} "
                  f"{l2_str:>5} {result['lp_obj']:>8.2f} {result['ip_obj']:>6} {gap_str:>6} "
                  f"{result['time']:>6.2f}s {result['columns']:>6}")

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0

    # Count optimal matches
    optimal_matches = sum(1 for r in results if r['optimal'] and r['ip_obj'] == r['optimal'])
    instances_with_opt = sum(1 for r in results if r['optimal'])

    print(f"Total instances: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per instance: {avg_time:.3f}s")
    if instances_with_opt > 0:
        print(f"Optimal solutions found: {optimal_matches}/{instances_with_opt} ({100*optimal_matches/instances_with_opt:.1f}%)")


if __name__ == '__main__':
    main()
