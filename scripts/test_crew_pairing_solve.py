#!/usr/bin/env python3
"""
Test script for solving crew pairing with different configurations.

This script tests the full solve pipeline with:
1. Default (buggy) configuration
2. Fixed configuration (closed connection gap)
3. FastPerSourcePricing with fixed config

Usage:
    python scripts/test_crew_pairing_solve.py [instance_path]
    python scripts/test_crew_pairing_solve.py --config fixed
    python scripts/test_crew_pairing_solve.py --config default
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.core.arc import ArcType
from opencg.core.column import Column
from opencg.master import HiGHSMasterProblem
from opencg.pricing import PricingConfig

# Try C++ backend
try:
    from opencg.pricing import FastPerSourcePricing
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    FastPerSourcePricing = None

# Fallback to Python pricing
from opencg.pricing import PerSourcePricing


def get_config_options(config_name: str) -> Dict:
    """Get configuration options by name."""
    configs = {
        'default': {
            # This has the connection gap bug!
            'min_connection_time': 0.5,
            'max_connection_time': 4.0,
            'min_layover_time': 10.0,  # GAP: 4-10h dead zone
            'max_layover_time': 24.0,
            'max_duty_time': 14.0,
            'max_flight_time': 8.0,
            'max_pairing_days': 5,
        },
        'fixed': {
            # Fixed: no connection gap
            'min_connection_time': 0.5,
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # FIXED: Close the gap
            'max_layover_time': 24.0,
            'max_duty_time': 14.0,
            'max_flight_time': 8.0,
            'max_pairing_days': 5,
        },
        'relaxed': {
            # Relaxed constraints for better coverage
            'min_connection_time': 0.5,
            'max_connection_time': 6.0,  # Allow longer connections
            'min_layover_time': 6.0,     # Close the gap
            'max_layover_time': 30.0,    # Allow longer layovers
            'max_duty_time': 16.0,       # More duty time
            'max_flight_time': 10.0,     # More flight time
            'max_pairing_days': 6,       # Allow 6-day pairings
        }
    }
    return configs.get(config_name, configs['fixed'])


def solve_crew_pairing(
    instance_path: Path,
    config_name: str = 'fixed',
    max_iterations: int = 20,
    cols_per_source: int = 5,
    use_cpp: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Solve crew pairing and return results.

    Returns:
        Dict with solve results
    """
    print(f"\nSolving with config='{config_name}'...")
    print("-" * 60)

    # Parse instance
    options = get_config_options(config_name)
    parser_config = ParserConfig(verbose=verbose, validate=False, options=options)
    parser = KasirzadehParser(parser_config)

    start = time.time()
    problem = parser.parse(instance_path)
    parse_time = time.time() - start

    print(f"Parsed: {len(problem.cover_constraints)} flights, "
          f"{problem.network.num_arcs} arcs ({parse_time:.2f}s)")

    # Check gap
    has_gap = parser._min_layover > parser._max_connection
    if has_gap:
        print(f"  WARNING: Connection gap {parser._max_connection}h - {parser._min_layover}h")

    # Create master problem
    master = HiGHSMasterProblem(problem, verbosity=0)

    # Add artificial columns
    big_m = 1e6
    next_col_id = 0
    for constraint in problem.cover_constraints:
        art_col = Column(
            arc_indices=(),
            cost=big_m,
            covered_items=frozenset([constraint.item_id]),
            column_id=next_col_id,
            attributes={'artificial': True},
        )
        master.add_column(art_col)
        next_col_id += 1

    # Create pricing
    pricing_config = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    if use_cpp and HAS_CPP:
        print("Using FastPerSourcePricing (C++)")
        pricing = FastPerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=cols_per_source,
            time_per_source=0.1,
            num_threads=4,
        )
    else:
        print("Using PerSourcePricing (Python)")
        pricing = PerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=cols_per_source,
            time_per_source=0.1,
        )

    # Generate initial columns
    print("Generating initial columns...")
    init_pricing_config = PricingConfig(
        max_columns=500,
        max_time=60.0,
        reduced_cost_threshold=1e10,  # Accept any
        check_elementarity=True,
        use_dominance=True,
    )
    pricing._config = init_pricing_config
    pricing.set_dual_values({})
    sol = pricing.solve()

    for col in sol.columns:
        col_with_id = col.with_id(next_col_id)
        next_col_id += 1
        master.add_column(col_with_id)

    print(f"  Generated {len(sol.columns)} initial columns")

    # Switch to CG config
    pricing._config = pricing_config

    # Column generation loop
    cg_start = time.time()
    best_coverage = 0.0
    lp_obj = None

    for iteration in range(max_iterations):
        # Solve LP
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            print(f"  LP not optimal: {lp_sol.status}")
            break

        lp_obj = lp_sol.objective_value

        # Compute coverage
        covered = set()
        for col_id, val in lp_sol.column_values.items():
            if val > 1e-6:
                col = master.get_column(col_id)
                if col and not col.attributes.get('artificial'):
                    covered.update(col.covered_items)

        coverage_pct = 100.0 * len(covered) / len(problem.cover_constraints)
        best_coverage = max(best_coverage, coverage_pct)

        # Get duals and run pricing
        duals = master.get_dual_values()

        # Set priority items (uncovered flights)
        all_items = set(range(len(problem.cover_constraints)))
        uncovered = all_items - covered
        if hasattr(pricing, 'set_priority_items'):
            pricing.set_priority_items(uncovered)

        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if verbose or iteration % 5 == 0:
            print(f"  Iter {iteration:2d}: LP={lp_obj:10.2f}, "
                  f"Coverage={coverage_pct:5.1f}%, "
                  f"Cols={len(pricing_sol.columns):3d}")

        if not pricing_sol.columns:
            print(f"  Converged at iteration {iteration}")
            break

        # Add columns
        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

    cg_time = time.time() - cg_start
    total_time = time.time() - start

    # Final coverage
    lp_sol = master.solve_lp()
    covered = set()
    for col_id, val in lp_sol.column_values.items():
        if val > 1e-6:
            col = master.get_column(col_id)
            if col and not col.attributes.get('artificial'):
                covered.update(col.covered_items)

    final_coverage = 100.0 * len(covered) / len(problem.cover_constraints)
    uncovered_count = len(problem.cover_constraints) - len(covered)

    results = {
        'config': config_name,
        'has_gap': has_gap,
        'num_flights': len(problem.cover_constraints),
        'num_arcs': problem.network.num_arcs,
        'lp_objective': lp_sol.objective_value,
        'coverage_pct': final_coverage,
        'uncovered': uncovered_count,
        'total_columns': master.num_columns,
        'parse_time': parse_time,
        'cg_time': cg_time,
        'total_time': total_time,
    }

    print()
    print(f"Results for '{config_name}':")
    print(f"  LP Objective:  {results['lp_objective']:.2f}")
    print(f"  Coverage:      {results['coverage_pct']:.1f}% "
          f"({len(covered)}/{results['num_flights']})")
    print(f"  Uncovered:     {results['uncovered']}")
    print(f"  Total Columns: {results['total_columns']}")
    print(f"  Time:          {results['total_time']:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test crew pairing solve")
    parser.add_argument("instance", nargs="?", default=None)
    parser.add_argument("--config", choices=['default', 'fixed', 'relaxed', 'all'],
                       default='all', help="Configuration to test")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--cols-per-source", type=int, default=5)
    parser.add_argument("--no-cpp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Determine instance path
    if args.instance:
        instance_path = Path(args.instance)
    else:
        instance_path = Path(__file__).parent.parent / "data" / "kasirzadeh" / "instance1"

    if not instance_path.exists():
        print(f"Error: Instance not found: {instance_path}")
        return 1

    print("=" * 70)
    print("CREW PAIRING SOLVE TEST")
    print("=" * 70)
    print(f"Instance: {instance_path}")
    print(f"C++ backend: {'available' if HAS_CPP else 'NOT available'}")
    print()

    all_results = []

    if args.config == 'all':
        configs = ['default', 'fixed']
    else:
        configs = [args.config]

    for config in configs:
        try:
            results = solve_crew_pairing(
                instance_path,
                config_name=config,
                max_iterations=args.max_iter,
                cols_per_source=args.cols_per_source,
                use_cpp=not args.no_cpp,
                verbose=args.verbose
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error with config '{config}': {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        print()
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Config':<10} {'Gap?':<6} {'Arcs':<8} {'LP Obj':<12} "
              f"{'Coverage':<10} {'Uncovered':<10} {'Time':<8}")
        print("-" * 70)

        for r in all_results:
            print(f"{r['config']:<10} "
                  f"{'YES' if r['has_gap'] else 'NO':<6} "
                  f"{r['num_arcs']:<8} "
                  f"{r['lp_objective']:<12.2f} "
                  f"{r['coverage_pct']:<9.1f}% "
                  f"{r['uncovered']:<10} "
                  f"{r['total_time']:<7.2f}s")

        print()

        # Analysis
        if len(all_results) >= 2:
            default_r = next((r for r in all_results if r['config'] == 'default'), None)
            fixed_r = next((r for r in all_results if r['config'] == 'fixed'), None)

            if default_r and fixed_r:
                improvement = fixed_r['coverage_pct'] - default_r['coverage_pct']
                arc_diff = fixed_r['num_arcs'] - default_r['num_arcs']

                print("ANALYSIS:")
                print(f"  Closing the gap added {arc_diff} arcs (overnight connections)")
                print(f"  Coverage improved by {improvement:.1f}%")

                if fixed_r['coverage_pct'] >= 99.5:
                    print(f"  [SUCCESS] Achieved near-complete coverage ({fixed_r['coverage_pct']:.1f}%)")
                elif fixed_r['coverage_pct'] >= 95:
                    print(f"  [GOOD] Achieved good coverage ({fixed_r['coverage_pct']:.1f}%)")
                else:
                    print(f"  [NEEDS WORK] Coverage still needs improvement")

    return 0


if __name__ == "__main__":
    sys.exit(main())
