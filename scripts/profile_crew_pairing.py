#!/usr/bin/env python3
"""
Profile crew pairing pricing to identify bottlenecks.

This script profiles the pricing phase of column generation to identify
performance bottlenecks. It uses cProfile for function-level profiling
and provides detailed timing breakdowns.

Usage:
    python scripts/profile_crew_pairing.py [instance_path]
    python scripts/profile_crew_pairing.py --cpp  # Use C++ backend
    python scripts/profile_crew_pairing.py --detailed  # Line-by-line profiling
"""

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
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


class PricingProfiler:
    """Profile pricing performance."""

    def __init__(self, instance_path: Path, use_cpp: bool = False):
        self.instance_path = instance_path
        self.use_cpp = use_cpp and HAS_CPP
        self.timings: Dict[str, List[float]] = {
            'parse': [],
            'setup': [],
            'pricing_solve': [],
            'add_columns': [],
        }

    def setup_problem(self) -> Tuple:
        """Parse instance and setup master/pricing."""
        print("Setting up problem...")

        # Fixed config (no connection gap)
        options = {
            'min_connection_time': 0.5,
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,
            'max_layover_time': 24.0,
            'max_duty_time': 14.0,
            'max_flight_time': 8.0,
            'max_pairing_days': 5,
        }

        # Parse
        start = time.time()
        parser_config = ParserConfig(verbose=False, validate=False, options=options)
        parser = KasirzadehParser(parser_config)
        problem = parser.parse(self.instance_path)
        self.timings['parse'].append(time.time() - start)

        print(f"  Flights: {len(problem.cover_constraints)}")
        print(f"  Arcs: {problem.network.num_arcs}")
        print(f"  Resources: {len(problem.resources)}")

        # Setup master
        start = time.time()
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

        if self.use_cpp:
            print("  Using FastPerSourcePricing (C++)")
            pricing = FastPerSourcePricing(
                problem,
                config=pricing_config,
                max_labels_per_node=30,
                cols_per_source=5,
                time_per_source=0.1,
                num_threads=4,
            )
        else:
            print("  Using PerSourcePricing (Python)")
            pricing = PerSourcePricing(
                problem,
                config=pricing_config,
                max_labels_per_node=30,
                cols_per_source=5,
                time_per_source=0.1,
            )

        self.timings['setup'].append(time.time() - start)

        return problem, master, pricing, next_col_id

    def generate_initial_columns(self, problem, pricing, master, next_col_id) -> int:
        """Generate initial columns without duals."""
        print("Generating initial columns...")

        init_config = PricingConfig(
            max_columns=500,
            max_time=60.0,
            reduced_cost_threshold=1e10,
            check_elementarity=True,
            use_dominance=True,
        )

        pricing._config = init_config
        pricing.set_dual_values({})

        start = time.time()
        sol = pricing.solve()
        elapsed = time.time() - start

        for col in sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

        print(f"  Generated {len(sol.columns)} columns in {elapsed:.2f}s")
        return next_col_id

    def run_cg_iterations(self, problem, master, pricing, next_col_id,
                          max_iterations: int = 10) -> Dict:
        """Run CG iterations and profile pricing."""
        print(f"\nRunning {max_iterations} CG iterations...")

        # Reset to CG config
        cg_config = PricingConfig(
            max_columns=200,
            max_time=30.0,
            reduced_cost_threshold=-1e-6,
            check_elementarity=True,
            use_dominance=True,
        )
        pricing._config = cg_config

        iteration_data = []

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Solve LP
            lp_start = time.time()
            lp_sol = master.solve_lp()
            lp_time = time.time() - lp_start

            if lp_sol.status.name != 'OPTIMAL':
                print(f"  Iter {iteration}: LP not optimal")
                break

            # Compute coverage
            covered = set()
            for col_id, val in lp_sol.column_values.items():
                if val > 1e-6:
                    col = master.get_column(col_id)
                    if col and not col.attributes.get('artificial'):
                        covered.update(col.covered_items)

            coverage_pct = 100.0 * len(covered) / len(problem.cover_constraints)

            # Get duals
            dual_start = time.time()
            duals = master.get_dual_values()
            dual_time = time.time() - dual_start

            # Set priority items
            all_items = set(range(len(problem.cover_constraints)))
            uncovered = all_items - covered
            if hasattr(pricing, 'set_priority_items'):
                pricing.set_priority_items(uncovered)

            # Run pricing (MAIN BOTTLENECK)
            pricing_start = time.time()
            pricing.set_dual_values(duals)
            pricing_sol = pricing.solve()
            pricing_time = time.time() - pricing_start
            self.timings['pricing_solve'].append(pricing_time)

            # Add columns
            add_start = time.time()
            for col in pricing_sol.columns:
                col_with_id = col.with_id(next_col_id)
                next_col_id += 1
                master.add_column(col_with_id)
            add_time = time.time() - add_start
            self.timings['add_columns'].append(add_time)

            iter_time = time.time() - iter_start

            iteration_data.append({
                'iteration': iteration,
                'lp_time': lp_time,
                'dual_time': dual_time,
                'pricing_time': pricing_time,
                'add_time': add_time,
                'total_time': iter_time,
                'lp_obj': lp_sol.objective_value,
                'coverage': coverage_pct,
                'num_cols': len(pricing_sol.columns),
            })

            print(f"  Iter {iteration:2d}: LP={lp_sol.objective_value:10.2f}, "
                  f"Coverage={coverage_pct:5.1f}%, "
                  f"Cols={len(pricing_sol.columns):3d}, "
                  f"Pricing={pricing_time:.3f}s")

            if not pricing_sol.columns:
                print(f"  Converged at iteration {iteration}")
                break

        return iteration_data

    def print_summary(self, iteration_data: List[Dict]):
        """Print profiling summary."""
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)

        # Overall timings
        print("\nPhase Timings:")
        print(f"  Parse:        {sum(self.timings['parse']):.3f}s")
        print(f"  Setup:        {sum(self.timings['setup']):.3f}s")
        print(f"  Pricing (total): {sum(self.timings['pricing_solve']):.3f}s "
              f"({len(self.timings['pricing_solve'])} calls)")
        print(f"  Add Columns:  {sum(self.timings['add_columns']):.3f}s")

        # Pricing breakdown
        if self.timings['pricing_solve']:
            avg_pricing = sum(self.timings['pricing_solve']) / len(self.timings['pricing_solve'])
            max_pricing = max(self.timings['pricing_solve'])
            min_pricing = min(self.timings['pricing_solve'])

            print(f"\nPricing Statistics:")
            print(f"  Average: {avg_pricing:.3f}s")
            print(f"  Min:     {min_pricing:.3f}s")
            print(f"  Max:     {max_pricing:.3f}s")

        # Iteration breakdown
        if iteration_data:
            print(f"\nPer-Iteration Breakdown:")
            print(f"  {'Iter':<6} {'LP':<8} {'Dual':<8} {'Pricing':<10} "
                  f"{'Add':<8} {'Total':<8} {'Cols':<6}")
            print("  " + "-" * 60)

            for data in iteration_data[:10]:  # First 10 iterations
                print(f"  {data['iteration']:<6} "
                      f"{data['lp_time']:<8.3f} "
                      f"{data['dual_time']:<8.3f} "
                      f"{data['pricing_time']:<10.3f} "
                      f"{data['add_time']:<8.3f} "
                      f"{data['total_time']:<8.3f} "
                      f"{data['num_cols']:<6}")

            # Time breakdown percentages
            total_iter_time = sum(d['total_time'] for d in iteration_data)
            total_pricing_time = sum(d['pricing_time'] for d in iteration_data)
            total_lp_time = sum(d['lp_time'] for d in iteration_data)

            pricing_pct = 100.0 * total_pricing_time / total_iter_time
            lp_pct = 100.0 * total_lp_time / total_iter_time

            print(f"\nTime Distribution:")
            print(f"  Pricing: {pricing_pct:.1f}% ({total_pricing_time:.2f}s)")
            print(f"  LP Solve: {lp_pct:.1f}% ({total_lp_time:.2f}s)")
            print(f"  Other: {100 - pricing_pct - lp_pct:.1f}%")

        # Recommendations
        print(f"\nBottleneck Analysis:")
        if iteration_data:
            pricing_pct = 100.0 * sum(d['pricing_time'] for d in iteration_data) / \
                         sum(d['total_time'] for d in iteration_data)

            if pricing_pct > 70:
                print(f"  [HIGH PRIORITY] Pricing is {pricing_pct:.0f}% of total time")
                print(f"    - Consider C++ backend (FastPerSourcePricing)")
                print(f"    - Increase parallel threads")
                print(f"    - Optimize resource extension in SPPRC")
            elif pricing_pct > 40:
                print(f"  [MEDIUM] Pricing is {pricing_pct:.0f}% of total time")
                print(f"    - Already using C++ backend" if self.use_cpp else
                      f"    - Try C++ backend for 3-5x speedup")
            else:
                print(f"  [LOW] Pricing is only {pricing_pct:.0f}% of total time")
                print(f"    - Focus on other optimizations")


def profile_with_cprofile(profiler: PricingProfiler, max_iterations: int = 10):
    """Run profiling with cProfile."""
    print("\n" + "=" * 70)
    print("FUNCTION-LEVEL PROFILING (cProfile)")
    print("=" * 70)

    pr = cProfile.Profile()

    # Setup (not profiled)
    problem, master, pricing, next_col_id = profiler.setup_problem()
    next_col_id = profiler.generate_initial_columns(problem, pricing, master, next_col_id)

    # Profile CG iterations
    pr.enable()
    iteration_data = profiler.run_cg_iterations(problem, master, pricing, next_col_id, max_iterations)
    pr.disable()

    # Print profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')

    print("\nTop 20 Functions by Cumulative Time:")
    ps.print_stats(20)
    print(s.getvalue())

    # Print custom summary
    profiler.print_summary(iteration_data)

    return iteration_data


def main():
    parser = argparse.ArgumentParser(description="Profile crew pairing pricing")
    parser.add_argument("instance", nargs="?", default=None)
    parser.add_argument("--cpp", action="store_true", help="Use C++ backend")
    parser.add_argument("--max-iter", type=int, default=10,
                       help="Number of CG iterations to profile")
    parser.add_argument("--detailed", action="store_true",
                       help="Detailed line-by-line profiling (slower)")
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
    print("CREW PAIRING PROFILING")
    print("=" * 70)
    print(f"Instance: {instance_path}")
    print(f"Backend: {'C++ (FastPerSourcePricing)' if args.cpp and HAS_CPP else 'Python (PerSourcePricing)'}")
    print(f"Max iterations: {args.max_iter}")
    print()

    if args.cpp and not HAS_CPP:
        print("WARNING: C++ backend not available, falling back to Python")
        print()

    # Create profiler
    profiler = PricingProfiler(instance_path, use_cpp=args.cpp)

    # Run profiling
    iteration_data = profile_with_cprofile(profiler, max_iterations=args.max_iter)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
