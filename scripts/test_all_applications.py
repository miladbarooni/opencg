#!/usr/bin/env python3
"""
Test script for all OpenCG applications.

Tests:
1. Cutting Stock Problem (CSP)
2. Vehicle Routing with Time Windows (VRPTW)
3. Crew Pairing (with and without connection gap fix)

Usage:
    python scripts/test_all_applications.py
    python scripts/test_all_applications.py --verbose
    python scripts/test_all_applications.py --app crew_pairing
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    """Result of a test."""
    app: str
    name: str
    passed: bool
    message: str
    time: float
    details: Optional[Dict] = None


def test_cutting_stock(verbose: bool = False) -> List[TestResult]:
    """Test Cutting Stock Problem."""
    from opencg.applications.cutting_stock import CuttingStockInstance, solve_cutting_stock

    results = []
    print("\n" + "=" * 60)
    print("CUTTING STOCK PROBLEM")
    print("=" * 60)

    # Test 1: Simple instance
    print("\nTest 1: Simple CSP instance")
    start = time.time()
    try:
        instance = CuttingStockInstance(
            roll_width=100,
            item_sizes=[45, 36, 31, 14],
            item_demands=[97, 610, 395, 211],
            name="simple_csp"
        )
        solution = solve_cutting_stock(instance, max_iterations=50, verbose=verbose)

        # L2 lower bound: ceiling of total volume / roll capacity
        total_volume = sum(s * d for s, d in zip(instance.item_sizes, instance.item_demands))
        l2 = int((total_volume + instance.roll_width - 1) // instance.roll_width)

        passed = solution.num_rolls_ip is not None and solution.num_rolls_ip >= l2
        results.append(TestResult(
            app="CSP",
            name="simple_csp",
            passed=passed,
            message=f"LP={solution.lp_objective:.2f}, IP={solution.num_rolls_ip}, L2={l2}",
            time=time.time() - start,
            details={'lp': solution.lp_objective, 'ip': solution.num_rolls_ip, 'l2': l2}
        ))
        print(f"  [{'PASS' if passed else 'FAIL'}] LP={solution.lp_objective:.2f}, IP={solution.num_rolls_ip}")
    except Exception as e:
        results.append(TestResult(
            app="CSP", name="simple_csp", passed=False,
            message=str(e), time=time.time() - start
        ))
        print(f"  [FAIL] Error: {e}")

    # Test 2: BPPLIB instance (if available)
    bpplib_path = Path("data/bpplib/Instances/Benchmarks/extracted/Scholl_CSP/Scholl_1")
    instances = list(bpplib_path.glob("*.txt"))[:1] if bpplib_path.exists() else []

    if instances:
        print("\nTest 2: BPPLIB Scholl instance")
        start = time.time()
        try:
            inst_path = instances[0]
            instance = CuttingStockInstance.from_bpplib(str(inst_path))
            solution = solve_cutting_stock(instance, max_iterations=50, verbose=verbose)

            passed = solution.num_rolls_ip is not None
            results.append(TestResult(
                app="CSP",
                name=f"bpplib_{inst_path.stem}",
                passed=passed,
                message=f"LP={solution.lp_objective:.2f}, IP={solution.num_rolls_ip}",
                time=time.time() - start
            ))
            print(f"  [{'PASS' if passed else 'FAIL'}] {inst_path.name}: LP={solution.lp_objective:.2f}, IP={solution.num_rolls_ip}")
        except Exception as e:
            results.append(TestResult(
                app="CSP", name="bpplib", passed=False,
                message=str(e), time=time.time() - start
            ))
            print(f"  [FAIL] Error: {e}")
    else:
        print("\nTest 2: BPPLIB instances not found (skipped)")

    return results


def test_vrptw(verbose: bool = False) -> List[TestResult]:
    """Test Vehicle Routing with Time Windows."""
    from opencg.applications.vrp import VRPTWInstance, solve_vrptw, VRPTWConfig

    results = []
    print("\n" + "=" * 60)
    print("VEHICLE ROUTING WITH TIME WINDOWS")
    print("=" * 60)

    # Test 1: Small instance
    print("\nTest 1: Small VRPTW instance (5 customers)")
    start = time.time()
    try:
        instance = VRPTWInstance(
            depot=(0, 0),
            customers=[
                (10, 0), (0, 10), (-10, 0), (0, -10), (5, 5)
            ],
            demands=[20, 15, 25, 10, 30],
            time_windows=[
                (0, 50), (20, 80), (40, 120), (0, 100), (10, 60)
            ],
            service_times=[10, 10, 10, 10, 10],
            vehicle_capacity=50,
            depot_time_window=(0, 200),
            name="small_vrptw"
        )

        config = VRPTWConfig(verbose=verbose, max_iterations=50)
        solution = solve_vrptw(instance, config)

        # Check all customers covered
        all_customers = set()
        for route in solution.routes:
            all_customers.update(route)

        coverage = len(all_customers) / instance.num_customers * 100
        passed = coverage == 100.0

        results.append(TestResult(
            app="VRPTW",
            name="small_vrptw",
            passed=passed,
            message=f"Dist={solution.total_distance_ip:.2f}, Vehicles={solution.num_vehicles}, Coverage={coverage:.0f}%",
            time=time.time() - start,
            details={'distance': solution.total_distance_ip, 'vehicles': solution.num_vehicles}
        ))
        print(f"  [{'PASS' if passed else 'FAIL'}] Distance={solution.total_distance_ip:.2f}, Vehicles={solution.num_vehicles}")
    except Exception as e:
        results.append(TestResult(
            app="VRPTW", name="small_vrptw", passed=False,
            message=str(e), time=time.time() - start
        ))
        print(f"  [FAIL] Error: {e}")

    # Test 2: Tight time windows
    print("\nTest 2: Tight time windows instance")
    start = time.time()
    try:
        instance = VRPTWInstance(
            depot=(0, 0),
            customers=[
                (10, 10), (-10, 10), (10, -10), (-10, -10)
            ],
            demands=[10, 10, 10, 10],
            time_windows=[
                (0, 25), (0, 25), (50, 75), (50, 75)
            ],
            service_times=[5, 5, 5, 5],
            vehicle_capacity=100,
            depot_time_window=(0, 200),
            name="tight_tw"
        )

        config = VRPTWConfig(verbose=verbose, max_iterations=50)
        solution = solve_vrptw(instance, config)

        all_customers = set()
        for route in solution.routes:
            all_customers.update(route)

        coverage = len(all_customers) / instance.num_customers * 100
        passed = coverage == 100.0

        results.append(TestResult(
            app="VRPTW",
            name="tight_tw",
            passed=passed,
            message=f"Dist={solution.total_distance_ip:.2f}, Vehicles={solution.num_vehicles}",
            time=time.time() - start
        ))
        print(f"  [{'PASS' if passed else 'FAIL'}] Distance={solution.total_distance_ip:.2f}, Vehicles={solution.num_vehicles}")
    except Exception as e:
        results.append(TestResult(
            app="VRPTW", name="tight_tw", passed=False,
            message=str(e), time=time.time() - start
        ))
        print(f"  [FAIL] Error: {e}")

    return results


def test_crew_pairing(verbose: bool = False) -> List[TestResult]:
    """Test Crew Pairing (Kasirzadeh instances)."""
    from opencg.parsers import KasirzadehParser
    from opencg.parsers.base import ParserConfig
    from opencg.core.column import Column
    from opencg.core.arc import ArcType
    from opencg.master import HiGHSMasterProblem
    from opencg.pricing import PricingConfig

    try:
        from opencg.pricing import FastPerSourcePricing
        HAS_CPP = True
    except ImportError:
        HAS_CPP = False
        from opencg.pricing import PerSourcePricing

    results = []
    print("\n" + "=" * 60)
    print("CREW PAIRING")
    print("=" * 60)

    instance_path = Path("data/kasirzadeh/instance1")
    if not instance_path.exists():
        print("\nKasirzadeh instance not found (skipped)")
        return results

    # Test 1: With connection gap (default buggy config)
    print("\nTest 1: Default config (with connection gap)")
    start = time.time()
    try:
        config = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 10.0,  # GAP!
            'max_duty_time': 14.0,
        })
        parser = KasirzadehParser(config)
        problem = parser.parse(instance_path)

        # Quick solve (limited iterations)
        coverage = _quick_cg_solve(problem, max_iter=10, use_cpp=HAS_CPP, verbose=verbose)

        passed = coverage > 90  # Even with gap, should get decent coverage
        results.append(TestResult(
            app="Crew",
            name="with_gap",
            passed=passed,
            message=f"Coverage={coverage:.1f}% (gap: 4-10h)",
            time=time.time() - start,
            details={'coverage': coverage, 'has_gap': True}
        ))
        print(f"  [{'PASS' if passed else 'FAIL'}] Coverage={coverage:.1f}%")
    except Exception as e:
        results.append(TestResult(
            app="Crew", name="with_gap", passed=False,
            message=str(e), time=time.time() - start
        ))
        print(f"  [FAIL] Error: {e}")

    # Test 2: Fixed config (no gap)
    print("\nTest 2: Fixed config (no connection gap)")
    start = time.time()
    try:
        config = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # FIXED!
            'max_duty_time': 14.0,
        })
        parser = KasirzadehParser(config)
        problem = parser.parse(instance_path)

        coverage = _quick_cg_solve(problem, max_iter=10, use_cpp=HAS_CPP, verbose=verbose)

        passed = coverage > 95  # Should be better with fix
        results.append(TestResult(
            app="Crew",
            name="fixed",
            passed=passed,
            message=f"Coverage={coverage:.1f}% (no gap)",
            time=time.time() - start,
            details={'coverage': coverage, 'has_gap': False}
        ))
        print(f"  [{'PASS' if passed else 'FAIL'}] Coverage={coverage:.1f}%")
    except Exception as e:
        results.append(TestResult(
            app="Crew", name="fixed", passed=False,
            message=str(e), time=time.time() - start
        ))
        print(f"  [FAIL] Error: {e}")

    return results


def _quick_cg_solve(problem, max_iter: int = 10, use_cpp: bool = True, verbose: bool = False) -> float:
    """Quick CG solve to test coverage."""
    from opencg.core.column import Column
    from opencg.master import HiGHSMasterProblem
    from opencg.pricing import PricingConfig

    if use_cpp:
        from opencg.pricing import FastPerSourcePricing as Pricing
    else:
        from opencg.pricing import PerSourcePricing as Pricing

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

    if use_cpp:
        pricing = Pricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
            num_threads=4,
        )
    else:
        pricing = Pricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
        )

    # Quick CG iterations
    for _ in range(max_iter):
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            break

        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if not pricing_sol.columns:
            break

        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

    # Compute coverage
    lp_sol = master.solve_lp()
    covered = set()
    for col_id, val in lp_sol.column_values.items():
        if val > 1e-6:
            col = master.get_column(col_id)
            if col and not col.attributes.get('artificial'):
                covered.update(col.covered_items)

    return 100.0 * len(covered) / len(problem.cover_constraints)


def main():
    parser = argparse.ArgumentParser(description="Test all OpenCG applications")
    parser.add_argument("--app", choices=['csp', 'vrptw', 'crew', 'all'],
                       default='all', help="Which application to test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenCG Application Tests")
    print("=" * 60)

    all_results = []

    if args.app in ['all', 'csp']:
        all_results.extend(test_cutting_stock(args.verbose))

    if args.app in ['all', 'vrptw']:
        all_results.extend(test_vrptw(args.verbose))

    if args.app in ['all', 'crew']:
        all_results.extend(test_crew_pairing(args.verbose))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total_time = sum(r.time for r in all_results)

    print(f"\nTests: {len(all_results)} total, {passed} passed, {failed} failed")
    print(f"Total time: {total_time:.2f}s")

    if failed > 0:
        print("\nFailed tests:")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.app}/{r.name}: {r.message}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
