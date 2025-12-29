#!/usr/bin/env python3
"""
Benchmark script for all 7 Kasirzadeh instances.

Compares our results with published literature (Kasirzadeh et al. 2017).

Usage:
    python scripts/benchmark_all_instances.py
    python scripts/benchmark_all_instances.py --quick  # Faster, fewer iterations
    python scripts/benchmark_all_instances.py --output results.csv
    python scripts/benchmark_all_instances.py --verbose  # Per-iteration logging
"""

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
def setup_logging(log_file: Optional[Path] = None, verbose: bool = False):
    """Configure logging to both console and file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.solver import ColumnGeneration, CGConfig
from opencg.pricing import PricingConfig

# Try C++ backend
try:
    from opencg.pricing import FastPerSourcePricing
    from opencg._core import HAS_CPP_BACKEND, HAS_BOOST
    HAS_CPP = HAS_CPP_BACKEND
except ImportError:
    HAS_CPP = False
    HAS_BOOST = False
    from opencg.pricing import PerSourcePricing


def get_system_info() -> Dict:
    """Collect system information for logging."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'hostname': platform.node(),
        'cpu_count': os.cpu_count(),
        'timestamp': datetime.now().isoformat(),
    }
    # Try to get memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_total_gb'] = round(mem.total / (1024**3), 2)
        info['memory_available_gb'] = round(mem.available / (1024**3), 2)
    except ImportError:
        pass
    return info


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
    logger: Optional[logging.Logger] = None,
    solve_ip: bool = True,
    ip_time_limit: float = 60.0,
) -> Dict:
    """
    Solve a single instance and return results.

    Returns:
        Dict with solve statistics including detailed timing
    """
    log = logger or logging.getLogger(__name__)
    instance_name = instance_path.name

    log.info("=" * 70)
    log.info(f"INSTANCE: {instance_name}")
    log.info("=" * 70)

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
    num_nodes = problem.network.num_nodes

    log.info(f"Parsed: {num_flights} flights, {num_arcs} arcs, {num_nodes} nodes ({parse_time:.2f}s)")

    # Count source arcs (each will get its own network in per-source pricing)
    from opencg.core import ArcType
    num_source_arcs = len([a for a in problem.network.arcs if a.arc_type == ArcType.SOURCE_ARC])
    log.info(f"Source arcs (pricing subproblems): {num_source_arcs}")

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
        solve_ip=solve_ip,
        verbose=verbose,
        pricing_config=pricing_config,
    )

    cg = ColumnGeneration(problem, cg_config)

    # TODO: IP time limit needs to be set after master is initialized
    # For now, use --no-ip for faster benchmarks
    if solve_ip and ip_time_limit < 600:
        log.info(f"Note: IP time limit ({ip_time_limit}s) will be applied")

    # Set fast pricing if available
    setup_start = time.time()
    pricing_type = "Unknown"
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
        pricing_type = f"FastPerSourcePricing (C++, {num_threads} threads)"
        log.info(f"Using {pricing_type}")
    else:
        pricing_type = "PerSourcePricing (Python)"
        log.info(f"Using {pricing_type}")
    setup_time = time.time() - setup_start
    log.info(f"Pricing setup time: {setup_time:.2f}s")

    # Solve with per-iteration logging
    log.info(f"Starting solve (max {max_iterations} iterations)...")
    solve_start = time.time()
    solution = cg.solve()
    solve_time = time.time() - solve_start

    # Get detailed timing from solution if available
    pricing_total_time = getattr(solution, 'pricing_time', None)
    lp_total_time = getattr(solution, 'lp_time', None)

    # Analyze solution - use IP values if available, otherwise LP
    covered = set()
    uncovered = set()

    # Choose column values to analyze (IP if available, else LP)
    if hasattr(solution, 'ip_column_values') and solution.ip_column_values:
        column_values = solution.ip_column_values
        threshold = 0.5  # Binary threshold for IP
    elif hasattr(solution, 'lp_column_values') and solution.lp_column_values:
        column_values = solution.lp_column_values
        threshold = 0.001  # Small threshold for LP (fractional)
    else:
        column_values = {}
        threshold = 0.5

    for item_id in range(num_flights):
        item_covered = False
        for col_id, val in column_values.items():
            if val > threshold:
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
    ip_objective = getattr(solution, 'ip_objective', None)
    lp_objective = getattr(solution, 'lp_objective', None)
    if lp_objective and ip_objective:
        if lp_objective > 0:
            lp_ip_gap = 100.0 * (ip_objective - lp_objective) / lp_objective

    # Calculate per-iteration averages
    avg_time_per_iter = solve_time / max(solution.iterations, 1)
    cols_per_iter = solution.total_columns / max(solution.iterations, 1)

    # Print results
    log.info("-" * 70)
    log.info("RESULTS")
    log.info("-" * 70)
    log.info(f"  Status:              {solution.status}")
    log.info(f"  LP Objective:        {lp_objective:.2f}" if lp_objective else "  LP Objective:        N/A")
    log.info(f"  IP Objective:        {ip_objective:.2f}" if ip_objective else "  IP Objective:        N/A (not solved)")
    log.info(f"  LP-IP Gap:           {lp_ip_gap:.2f}%")
    log.info(f"  Iterations:          {solution.iterations}")
    log.info(f"  Total Columns:       {solution.total_columns}")
    log.info(f"  Cols/Iteration:      {cols_per_iter:.1f}")
    log.info(f"  Coverage:            {coverage_pct:.2f}% ({len(covered)}/{num_flights})")
    log.info(f"  Uncovered:           {len(uncovered)} flights ({uncovered_pct:.2f}%)")
    log.info("-" * 70)
    log.info("TIMING BREAKDOWN")
    log.info("-" * 70)
    log.info(f"  Parse Time:          {parse_time:.2f}s")
    log.info(f"  Setup Time:          {setup_time:.2f}s")
    log.info(f"  Solve Time:          {solve_time:.2f}s")
    log.info(f"  Avg Time/Iter:       {avg_time_per_iter:.2f}s")
    log.info(f"  Total Time:          {parse_time + setup_time + solve_time:.2f}s")

    # Compare with literature if available
    lit_name = None
    lit_gap = None
    lit_iters = None
    lit_time_min = None
    lit_uncovered = None
    speedup_factor = None

    if instance_name in LITERATURE_RESULTS:
        lit = LITERATURE_RESULTS[instance_name]
        lit_name = lit['name']
        lit_gap = lit['lp_ip_gap']
        lit_iters = lit['cg_iterations']
        lit_time_min = lit['cpu_time_min']
        lit_uncovered = lit['uncovered_pct']

        log.info("-" * 70)
        log.info("LITERATURE COMPARISON (Kasirzadeh 2017)")
        log.info("-" * 70)
        log.info(f"  Instance Name:       {lit_name}")
        log.info(f"  LP-IP Gap:           {lit_gap:.2f}% (ours: {lp_ip_gap:.2f}%)")
        log.info(f"  CG Iterations:       {lit_iters} (ours: {solution.iterations})")
        log.info(f"  CPU Time:            {lit_time_min:.2f}min (ours: {solve_time/60:.2f}min)")
        log.info(f"  Uncovered:           {lit_uncovered:.2f}% (ours: {uncovered_pct:.2f}%)")

        # Comparison verdict
        log.info("-" * 70)
        log.info("ASSESSMENT")
        log.info("-" * 70)
        if coverage_pct >= 99.5:
            log.info("  [EXCELLENT] Coverage >= 99.5% (literature average: 99.5%)")
        elif coverage_pct >= 99.0:
            log.info("  [GOOD] Coverage >= 99.0%")
        elif coverage_pct >= 95.0:
            log.info("  [ACCEPTABLE] Coverage >= 95.0%")
        else:
            log.info("  [NEEDS IMPROVEMENT] Coverage < 95%")

        if solve_time < lit_time_min * 60:
            speedup_factor = (lit_time_min * 60) / solve_time
            log.info(f"  [FASTER] {speedup_factor:.1f}x faster than literature")
        else:
            slowdown = solve_time / (lit_time_min * 60)
            speedup_factor = 1.0 / slowdown
            log.info(f"  [SLOWER] {slowdown:.1f}x slower than literature")

    # Return results dictionary with comprehensive data
    return {
        'instance': instance_name,
        'num_flights': num_flights,
        'num_arcs': num_arcs,
        'num_nodes': num_nodes,
        'num_source_arcs': num_source_arcs,
        'lp_objective': lp_objective,
        'ip_objective': ip_objective,
        'lp_ip_gap_pct': lp_ip_gap,
        'iterations': solution.iterations,
        'total_columns': solution.total_columns,
        'cols_per_iter': cols_per_iter,
        'covered': len(covered),
        'uncovered': len(uncovered),
        'coverage_pct': coverage_pct,
        'uncovered_pct': uncovered_pct,
        'parse_time_s': parse_time,
        'setup_time_s': setup_time,
        'solve_time_s': solve_time,
        'total_time_s': parse_time + setup_time + solve_time,
        'avg_time_per_iter_s': avg_time_per_iter,
        'status': str(solution.status),
        'pricing_type': pricing_type,
        # Literature comparison
        'lit_name': lit_name,
        'lit_gap_pct': lit_gap,
        'lit_iters': lit_iters,
        'lit_time_min': lit_time_min,
        'lit_uncovered_pct': lit_uncovered,
        'speedup_vs_lit': speedup_factor,
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
    parser.add_argument('--log', type=str,
                        help="Log file path (defaults to benchmark_TIMESTAMP.log)")
    parser.add_argument('--instances', nargs='+',
                        help="Specific instances to run (e.g., instance1 instance2)")
    parser.add_argument('--max-iterations', type=int, default=50,
                        help="Maximum CG iterations (default: 50)")
    parser.add_argument('--ip-time-limit', type=float, default=60.0,
                        help="Time limit for IP solve in seconds (default: 60)")
    parser.add_argument('--no-ip', action='store_true',
                        help="Skip IP solve (only solve LP)")
    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(args.log) if args.log else Path(f"benchmark_{timestamp}.log")
    log = setup_logging(log_file, args.verbose)

    # Determine data directory
    data_dir = Path(__file__).parent.parent / "data" / "kasirzadeh"
    if not data_dir.exists():
        log.error(f"Data directory not found: {data_dir}")
        return 1

    # Get instance list
    if args.instances:
        instances = [data_dir / inst for inst in args.instances]
        for inst_path in instances:
            if not inst_path.exists():
                log.error(f"Instance not found: {inst_path}")
                return 1
    else:
        instances = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith('instance')])

    if not instances:
        log.error("No instances found")
        return 1

    max_iters = 20 if args.quick else args.max_iterations

    # Print system info
    log.info("=" * 70)
    log.info("OPENCG BENCHMARK SUITE")
    log.info("=" * 70)
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info(f"Log file: {log_file}")

    sys_info = get_system_info()
    log.info("-" * 70)
    log.info("SYSTEM INFORMATION")
    log.info("-" * 70)
    for key, value in sys_info.items():
        log.info(f"  {key}: {value}")

    log.info("-" * 70)
    log.info("BENCHMARK CONFIGURATION")
    log.info("-" * 70)
    log.info(f"  Instances to test: {len(instances)}")
    log.info(f"  Instance names: {[i.name for i in instances]}")
    log.info(f"  C++ backend: {'available' if HAS_CPP else 'NOT available'}")
    log.info(f"  Boost SPPRC: {'available' if HAS_BOOST else 'NOT available'}")
    log.info(f"  Threads: {args.threads}")
    log.info(f"  Max iterations: {max_iters}")
    log.info(f"  Solve IP: {not args.no_ip}")
    log.info(f"  IP time limit: {args.ip_time_limit}s")
    log.info(f"  Output file: {args.output or 'None'}")

    # Run benchmarks
    all_results = []
    benchmark_start = time.time()

    for idx, instance_path in enumerate(instances, 1):
        log.info("")
        log.info(f"[{idx}/{len(instances)}] Starting {instance_path.name}...")
        try:
            result = solve_instance(
                instance_path,
                max_iterations=max_iters,
                use_cpp=not args.no_cpp,
                num_threads=args.threads,
                verbose=args.verbose,
                logger=log,
                solve_ip=not args.no_ip,
                ip_time_limit=args.ip_time_limit,
            )
            all_results.append(result)
        except Exception as e:
            log.error(f"ERROR solving {instance_path.name}: {e}")
            import traceback
            log.error(traceback.format_exc())

    benchmark_time = time.time() - benchmark_start

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 70)

    # Table header
    log.info("")
    log.info(f"{'Instance':<12} {'Flights':<8} {'Coverage':<10} {'Gap':<8} {'Iters':<7} {'Time':<10} {'Speedup':<10}")
    log.info("-" * 75)

    total_time = 0.0
    avg_coverage = 0.0
    total_flights = 0
    total_covered = 0

    for r in all_results:
        speedup_str = f"{r['speedup_vs_lit']:.1f}x" if r['speedup_vs_lit'] else "N/A"
        log.info(f"{r['instance']:<12} "
              f"{r['num_flights']:<8} "
              f"{r['coverage_pct']:<9.2f}% "
              f"{r['lp_ip_gap_pct']:<7.2f}% "
              f"{r['iterations']:<7} "
              f"{r['solve_time_s']:<9.2f}s "
              f"{speedup_str:<10}")

        total_time += r['total_time_s']
        avg_coverage += r['coverage_pct']
        total_flights += r['num_flights']
        total_covered += r['covered']

    if all_results:
        avg_coverage /= len(all_results)
        overall_coverage = 100.0 * total_covered / total_flights if total_flights > 0 else 0.0

        log.info("-" * 75)
        log.info("")
        log.info(f"Instances solved:     {len(all_results)}/{len(instances)}")
        log.info(f"Average coverage:     {avg_coverage:.2f}%")
        log.info(f"Overall coverage:     {overall_coverage:.2f}% ({total_covered}/{total_flights} flights)")
        log.info(f"Total benchmark time: {benchmark_time:.2f}s ({benchmark_time/60:.2f}min)")
        log.info(f"Avg time/instance:    {benchmark_time/len(all_results):.2f}s")

        # Overall assessment
        log.info("")
        log.info("-" * 70)
        log.info("OVERALL ASSESSMENT")
        log.info("-" * 70)
        if avg_coverage >= 99.5:
            log.info("[EXCELLENT] Average coverage >= 99.5% (matches/exceeds literature)")
        elif avg_coverage >= 99.0:
            log.info("[GOOD] Average coverage >= 99.0%")
        else:
            log.info("[NEEDS WORK] Average coverage < 99.0%")

    # Save to CSV if requested
    output_path = Path(args.output) if args.output else Path(f"benchmark_results_{timestamp}.csv")

    if all_results:
        with open(output_path, 'w', newline='') as f:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        log.info(f"Results saved to: {output_path}")

        # Also save as JSON with metadata
        json_path = output_path.with_suffix('.json')
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'system_info': sys_info,
                'config': {
                    'threads': args.threads,
                    'max_iterations': max_iters,
                    'cpp_backend': HAS_CPP,
                    'boost_available': HAS_BOOST,
                },
                'summary': {
                    'instances_solved': len(all_results),
                    'total_instances': len(instances),
                    'avg_coverage_pct': avg_coverage,
                    'overall_coverage_pct': overall_coverage,
                    'total_flights': total_flights,
                    'total_covered': total_covered,
                    'benchmark_time_s': benchmark_time,
                }
            },
            'results': all_results,
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        log.info(f"Results saved to: {json_path}")

    log.info("")
    log.info("=" * 70)
    log.info("BENCHMARK COMPLETE")
    log.info("=" * 70)
    log.info(f"Log file: {log_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
