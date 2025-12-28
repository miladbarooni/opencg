#!/usr/bin/env python3
"""
Benchmark script for all 7 Kasirzadeh instances.

Compares our results with published literature (Kasirzadeh et al. 2017).

Usage:
    python scripts/benchmark_all_instances.py
    python scripts/benchmark_all_instances.py --quick  # Faster, fewer iterations
    python scripts/benchmark_all_instances.py --output results.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.solver import ColumnGeneration, CGConfig
from opencg.pricing import PricingConfig

# Try C++ backend
try:
    from opencg.pricing import FastPerSourcePricing
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    from opencg.pricing import PerSourcePricing


# Literature results from Kasirzadeh et al. (2017)
# Table 4 and 5 from the paper
LITERATURE_RESULTS = {
    'instance1': {
        'name': 'I1-727',
        'flights': 727,
        'lp_ip_gap': 0.00,
        'bnp_nodes': 7,
        'cg_iterations': 239,
        'cpu_time_min': 0.16,
        'uncovered_pct': 0.00,
    },
    'instance2': {
        'name': 'I2-DC9',
        'flights': 709,
        'lp_ip_gap': 0.18,
        'bnp_nodes': 30,
        'cg_iterations': 1968,
        'cpu_time_min': 0.45,
        'uncovered_pct': 0.00,
    },
    'instance3': {
        'name': 'I3-D94',
        'flights': 856,
        'lp_ip_gap': 0.02,
        'bnp_nodes': 14,
        'cg_iterations': 466,
        'cpu_time_min': 1.81,
        'uncovered_pct': 0.00,
    },
    'instance4': {
        'name': 'I4-D95',
        'flights': 1799,
        'lp_ip_gap': 0.38,
        'bnp_nodes': 129,
        'cg_iterations': 2417,
        'cpu_time_min': 47.59,
        'uncovered_pct': 0.04,
    },
    'instance5': {
        'name': 'I5-757',
        'flights': 2111,
        'lp_ip_gap': 2.91,
        'bnp_nodes': 172,
        'cg_iterations': 4531,
        'cpu_time_min': 149.58,
        'uncovered_pct': 1.81,
    },
    'instance6': {
        'name': 'I6-319',
        'flights': 2098,
        'lp_ip_gap': 0.49,
        'bnp_nodes': 168,
        'cg_iterations': 2975,
        'cpu_time_min': 75.31,
        'uncovered_pct': 0.14,
    },
    'instance7': {
        'name': 'I7-320',
        'flights': 2520,
        'lp_ip_gap': 0.37,
        'bnp_nodes': 195,
        'cg_iterations': 4011,
        'cpu_time_min': 184.76,
        'uncovered_pct': 0.86,
    },
}


def get_parser_config():
    """Get fixed parser configuration (closes connection gap)."""
    return {
        'min_connection_time': 0.5,
        'max_connection_time': 4.0,
        'min_layover_time': 4.0,  # FIXED: Close the gap
        'max_layover_time': 24.0,
        'max_duty_time': 14.0,
        'max_flight_time': 8.0,
        'max_pairing_days': 5,
    }


def solve_instance(
    instance_path: Path,
    max_iterations: int = 50,
    use_cpp: bool = True,
    num_threads: int = 4,
    verbose: bool = False,
) -> Dict:
    """
    Solve a single instance and return results.

    Returns:
        Dict with solve statistics
    """
    instance_name = instance_path.name

    print(f"\n{'='*70}", flush=True)
    print(f"INSTANCE: {instance_name}", flush=True)
    print(f"{'='*70}", flush=True)

    # Parse instance
    parser_config = ParserConfig(
        verbose=False,
        validate=False,
        options=get_parser_config()
    )
    parser = KasirzadehParser(parser_config)

    parse_start = time.time()
    problem = parser.parse(instance_path)
    parse_time = time.time() - parse_start

    num_flights = len(problem.cover_constraints)
    num_arcs = problem.network.num_arcs

    print(f"Parsed: {num_flights} flights, {num_arcs} arcs ({parse_time:.2f}s)", flush=True)

    # Configure pricing
    pricing_config = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    # Configure column generation
    cg_config = CGConfig(
        max_iterations=max_iterations,
        max_time=600.0,  # 10 minutes max
        solve_ip=True,
        verbose=verbose,
        pricing_config=pricing_config,
    )

    cg = ColumnGeneration(problem, cg_config)

    # Set fast pricing if available
    if use_cpp and HAS_CPP:
        pricing = FastPerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
            num_threads=num_threads,
        )
        cg.set_pricing(pricing)
        print(f"Using FastPerSourcePricing (C++) with {num_threads} threads", flush=True)
    else:
        print("Using PerSourcePricing (Python)", flush=True)

    # Solve
    solve_start = time.time()
    solution = cg.solve()
    solve_time = time.time() - solve_start

    # Analyze solution
    covered = set()
    uncovered = set()

    for item_id in range(num_flights):
        item_covered = False
        for col_id, val in solution.ip_column_values.items():
            if val > 0.5:
                col = cg._column_pool.get(col_id)
                if col and item_id in col.covered_items:
                    item_covered = True
                    break

        if item_covered:
            covered.add(item_id)
        else:
            uncovered.add(item_id)

    coverage_pct = 100.0 * len(covered) / num_flights
    uncovered_pct = 100.0 - coverage_pct

    # Calculate LP-IP gap (if both available)
    lp_ip_gap = 0.0
    if solution.lp_objective and solution.ip_objective:
        if solution.lp_objective > 0:
            lp_ip_gap = 100.0 * (solution.ip_objective - solution.lp_objective) / solution.lp_objective

    # Print results
    print(f"\n{'Results':-^70}", flush=True)
    print(f"  Status:          {solution.status}", flush=True)
    print(f"  LP Objective:    {solution.lp_objective:.2f}", flush=True)
    print(f"  IP Objective:    {solution.ip_objective:.2f}", flush=True)
    print(f"  LP-IP Gap:       {lp_ip_gap:.2f}%", flush=True)
    print(f"  Iterations:      {solution.iterations}", flush=True)
    print(f"  Total Columns:   {solution.total_columns}", flush=True)
    print(f"  Coverage:        {coverage_pct:.2f}% ({len(covered)}/{num_flights})", flush=True)
    print(f"  Uncovered:       {len(uncovered)} flights ({uncovered_pct:.2f}%)", flush=True)
    print(f"  Parse Time:      {parse_time:.2f}s", flush=True)
    print(f"  Solve Time:      {solve_time:.2f}s", flush=True)
    print(f"  Total Time:      {parse_time + solve_time:.2f}s", flush=True)

    # Compare with literature if available
    if instance_name in LITERATURE_RESULTS:
        lit = LITERATURE_RESULTS[instance_name]
        print(f"\n{'Literature Comparison (Kasirzadeh 2017)':-^70}", flush=True)
        print(f"  Name:            {lit['name']}", flush=True)
        print(f"  LP-IP Gap:       {lit['lp_ip_gap']:.2f}% (ours: {lp_ip_gap:.2f}%)", flush=True)
        print(f"  CG Iterations:   {lit['cg_iterations']} (ours: {solution.iterations})", flush=True)
        print(f"  CPU Time:        {lit['cpu_time_min']:.2f}min (ours: {solve_time/60:.2f}min)", flush=True)
        print(f"  Uncovered:       {lit['uncovered_pct']:.2f}% (ours: {uncovered_pct:.2f}%)", flush=True)

        # Comparison verdict
        print(f"\n{'Assessment':-^70}", flush=True)
        if coverage_pct >= 99.5:
            print("  ✅ EXCELLENT: Coverage ≥ 99.5% (literature average: 99.5%)", flush=True)
        elif coverage_pct >= 99.0:
            print("  ✅ GOOD: Coverage ≥ 99.0%", flush=True)
        elif coverage_pct >= 95.0:
            print("  ⚠️  ACCEPTABLE: Coverage ≥ 95.0%", flush=True)
        else:
            print("  ❌ NEEDS IMPROVEMENT: Coverage < 95%", flush=True)

        if solve_time < lit['cpu_time_min'] * 60:
            speedup = (lit['cpu_time_min'] * 60) / solve_time
            print(f"  ✅ FASTER: {speedup:.1f}x faster than literature", flush=True)
        else:
            slowdown = solve_time / (lit['cpu_time_min'] * 60)
            print(f"  ⚠️  SLOWER: {slowdown:.1f}x slower than literature", flush=True)

    # Return results dictionary
    return {
        'instance': instance_name,
        'num_flights': num_flights,
        'num_arcs': num_arcs,
        'lp_objective': solution.lp_objective,
        'ip_objective': solution.ip_objective,
        'lp_ip_gap_pct': lp_ip_gap,
        'iterations': solution.iterations,
        'total_columns': solution.total_columns,
        'covered': len(covered),
        'uncovered': len(uncovered),
        'coverage_pct': coverage_pct,
        'uncovered_pct': uncovered_pct,
        'parse_time_s': parse_time,
        'solve_time_s': solve_time,
        'total_time_s': parse_time + solve_time,
        'status': str(solution.status),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark all Kasirzadeh instances")
    parser.add_argument('--quick', action='store_true',
                        help="Quick mode (fewer iterations)")
    parser.add_argument('--no-cpp', action='store_true',
                        help="Don't use C++ backend")
    parser.add_argument('--threads', type=int, default=4,
                        help="Number of threads for parallel pricing")
    parser.add_argument('--verbose', action='store_true',
                        help="Verbose output during solve")
    parser.add_argument('--output', type=str,
                        help="Output CSV file path")
    parser.add_argument('--instances', nargs='+',
                        help="Specific instances to run (e.g., instance1 instance2)")
    args = parser.parse_args()

    # Determine data directory
    data_dir = Path(__file__).parent.parent / "data" / "kasirzadeh"
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Get instance list
    if args.instances:
        instances = [data_dir / inst for inst in args.instances]
        for inst_path in instances:
            if not inst_path.exists():
                print(f"Error: Instance not found: {inst_path}")
                return 1
    else:
        instances = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('instance')])

    if not instances:
        print("Error: No instances found")
        return 1

    print("=" * 70, flush=True)
    print("OPENCG BENCHMARK SUITE", flush=True)
    print("=" * 70, flush=True)
    print(f"Instances to test: {len(instances)}", flush=True)
    print(f"C++ backend: {'available' if HAS_CPP else 'NOT available'}", flush=True)
    print(f"Threads: {args.threads}", flush=True)
    print(f"Max iterations: {20 if args.quick else 50}", flush=True)
    print(flush=True)

    # Run benchmarks
    all_results = []
    for instance_path in instances:
        try:
            result = solve_instance(
                instance_path,
                max_iterations=20 if args.quick else 50,
                use_cpp=not args.no_cpp,
                num_threads=args.threads,
                verbose=args.verbose,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR solving {instance_path.name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Table header
    print(f"{'Instance':<12} {'Flights':<8} {'Coverage':<10} {'Gap':<8} {'Iters':<7} {'Time':<10}", flush=True)
    print("-" * 70, flush=True)

    total_time = 0.0
    avg_coverage = 0.0

    for r in all_results:
        print(f"{r['instance']:<12} "
              f"{r['num_flights']:<8} "
              f"{r['coverage_pct']:<9.2f}% "
              f"{r['lp_ip_gap_pct']:<7.2f}% "
              f"{r['iterations']:<7} "
              f"{r['solve_time_s']:<9.2f}s", flush=True)

        total_time += r['total_time_s']
        avg_coverage += r['coverage_pct']

    if all_results:
        avg_coverage /= len(all_results)

        print(flush=True)
        print(f"Average coverage: {avg_coverage:.2f}%", flush=True)
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)", flush=True)

        # Overall assessment
        print(f"\n{'Overall Assessment':-^70}", flush=True)
        if avg_coverage >= 99.5:
            print("✅ EXCELLENT: Average coverage ≥ 99.5% (matches/exceeds literature)", flush=True)
        elif avg_coverage >= 99.0:
            print("✅ GOOD: Average coverage ≥ 99.0%", flush=True)
        else:
            print("⚠️  NEEDS WORK: Average coverage < 99.0%", flush=True)

    # Save to CSV if requested
    if args.output and all_results:
        output_path = Path(args.output)
        with open(output_path, 'w', newline='') as f:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n✅ Results saved to: {output_path}", flush=True)

        # Also save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Results saved to: {json_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
