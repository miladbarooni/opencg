#!/usr/bin/env python3
"""
Test script for the new require_integral_lp feature.

This script compares:
1. Standard CG (may leave some flights on artificials)
2. CG with require_integral_lp=True (pushes until LP is integral)

Usage:
    python scripts/test_integral_lp.py
"""

import sys
import time
from pathlib import Path

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


def test_integral_lp(instance_path: Path):
    """Test require_integral_lp feature."""

    print("=" * 70)
    print("TESTING require_integral_lp FEATURE")
    print("=" * 70)
    print(f"Instance: {instance_path}")
    print()

    # Parse instance (fixed config)
    parser_config = ParserConfig(
        verbose=False,
        validate=False,
        options={
            'min_connection_time': 0.5,
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # Fixed: no gap
            'max_layover_time': 24.0,
            'max_duty_time': 14.0,
            'max_flight_time': 8.0,
            'max_pairing_days': 5,
        }
    )

    print("Parsing instance...")
    parser = KasirzadehParser(parser_config)
    problem = parser.parse(instance_path)
    print(f"  {len(problem.cover_constraints)} flights, {problem.network.num_arcs} arcs")
    print()

    # Test 1: Standard CG
    print("-" * 70)
    print("TEST 1: Standard Column Generation (solve_ip=True)")
    print("-" * 70)

    pricing_config = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    cg_config = CGConfig(
        max_iterations=50,
        max_time=300.0,
        solve_ip=True,
        verbose=True,
        pricing_config=pricing_config,
        require_integral_lp=False,  # Standard CG
    )

    cg = ColumnGeneration(problem, cg_config)

    # Set custom pricing if using C++
    if HAS_CPP:
        pricing = FastPerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
            num_threads=4,
        )
        cg.set_pricing(pricing)

    start = time.time()
    solution = cg.solve()
    time1 = time.time() - start

    print()
    print(f"Results:")
    print(f"  Status: {solution.status}")
    print(f"  LP Objective: {solution.lp_objective:.2f}")
    print(f"  IP Objective: {solution.ip_objective:.2f}")
    print(f"  Iterations: {solution.iterations}")
    print(f"  Total Columns: {solution.total_columns}")
    print(f"  Time: {time1:.2f}s")

    # Check coverage
    covered = set()
    uncovered = set()
    for item_id in range(len(problem.cover_constraints)):
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

    coverage_pct = 100.0 * len(covered) / len(problem.cover_constraints)
    print(f"  Coverage: {coverage_pct:.1f}% ({len(covered)}/{len(problem.cover_constraints)})")
    print(f"  Uncovered flights: {len(uncovered)}")
    if uncovered:
        print(f"    Flight IDs: {sorted(list(uncovered)[:10])}" +
              (f" ... ({len(uncovered)} total)" if len(uncovered) > 10 else ""))

    # Test 2: require_integral_lp
    print()
    print("-" * 70)
    print("TEST 2: Column Generation with require_integral_lp=True")
    print("-" * 70)

    cg_config2 = CGConfig(
        max_iterations=100,  # More iterations allowed
        max_time=600.0,
        solve_ip=True,
        verbose=True,
        pricing_config=pricing_config,
        require_integral_lp=True,  # NEW FEATURE
        artificial_tolerance=0.001,  # Very tight (0.1%)
    )

    cg2 = ColumnGeneration(problem, cg_config2)

    # Set custom pricing if using C++
    if HAS_CPP:
        pricing2 = FastPerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
            num_threads=4,
        )
        cg2.set_pricing(pricing2)

    start = time.time()
    solution2 = cg2.solve()
    time2 = time.time() - start

    print()
    print(f"Results:")
    print(f"  Status: {solution2.status}")
    print(f"  LP Objective: {solution2.lp_objective:.2f}")
    print(f"  IP Objective: {solution2.ip_objective:.2f}")
    print(f"  Iterations: {solution2.iterations}")
    print(f"  Total Columns: {solution2.total_columns}")
    print(f"  Time: {time2:.2f}s")

    # Check coverage
    covered2 = set()
    uncovered2 = set()
    for item_id in range(len(problem.cover_constraints)):
        item_covered = False
        for col_id, val in solution2.ip_column_values.items():
            if val > 0.5:
                col = cg2._column_pool.get(col_id)
                if col and item_id in col.covered_items:
                    item_covered = True
                    break
        if item_covered:
            covered2.add(item_id)
        else:
            uncovered2.add(item_id)

    coverage_pct2 = 100.0 * len(covered2) / len(problem.cover_constraints)
    print(f"  Coverage: {coverage_pct2:.1f}% ({len(covered2)}/{len(problem.cover_constraints)})")
    print(f"  Uncovered flights: {len(uncovered2)}")
    if uncovered2:
        print(f"    Flight IDs: {sorted(list(uncovered2)[:10])}" +
              (f" ... ({len(uncovered2)} total)" if len(uncovered2) > 10 else ""))

    # Comparison
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Standard CG':<20} {'require_integral_lp':<20}")
    print("-" * 70)
    print(f"{'Coverage':<30} {coverage_pct:<19.1f}% {coverage_pct2:<19.1f}%")
    print(f"{'Uncovered Flights':<30} {len(uncovered):<20} {len(uncovered2):<20}")
    print(f"{'LP Objective':<30} {solution.lp_objective:<20.2f} {solution2.lp_objective:<20.2f}")
    print(f"{'IP Objective':<30} {solution.ip_objective:<20.2f} {solution2.ip_objective:<20.2f}")
    print(f"{'Total Columns':<30} {solution.total_columns:<20} {solution2.total_columns:<20}")
    print(f"{'Iterations':<30} {solution.iterations:<20} {solution2.iterations:<20}")
    print(f"{'Solve Time (s)':<30} {time1:<20.2f} {time2:<20.2f}")
    print()

    # Analysis
    improvement = len(covered2) - len(covered)

    if improvement > 0:
        print(f"✅ SUCCESS: require_integral_lp improved coverage by {improvement} flights!")
        print(f"   ({coverage_pct:.1f}% → {coverage_pct2:.1f}%)")
    elif improvement == 0:
        if len(uncovered) == 0:
            print(f"✅ EXCELLENT: Both achieve 100% coverage!")
        else:
            print(f"⚠️  SAME RESULT: Both methods cover {len(covered)} flights")
            print(f"   The uncovered flight(s) may be structurally impossible to cover")
    else:
        print(f"❌ UNEXPECTED: require_integral_lp decreased coverage by {-improvement} flights")

    print()

    # Time overhead
    overhead_pct = 100.0 * (time2 - time1) / time1 if time1 > 0 else 0
    print(f"Time overhead: +{time2 - time1:.2f}s ({overhead_pct:+.1f}%)")

    return {
        'standard': {
            'coverage_pct': coverage_pct,
            'uncovered': len(uncovered),
            'time': time1,
            'columns': solution.total_columns,
        },
        'integral_lp': {
            'coverage_pct': coverage_pct2,
            'uncovered': len(uncovered2),
            'time': time2,
            'columns': solution2.total_columns,
        }
    }


def main():
    # Default to instance 1
    instance_path = Path(__file__).parent.parent / "data" / "kasirzadeh" / "instance1"

    if not instance_path.exists():
        print(f"Error: Instance not found: {instance_path}")
        return 1

    print(f"C++ backend: {'available' if HAS_CPP else 'NOT available'}")
    if not HAS_CPP:
        print("WARNING: Using Python pricing (slower)")
    print()

    results = test_integral_lp(instance_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
