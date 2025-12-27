"""
Example: High-coverage crew pairing using Per-Source Pricing.

This example demonstrates how to achieve near-complete flight coverage
(99%+) on Kasirzadeh instances using the Per-Source Pricing algorithm.

The key insight is that standard beam-search pricing only explores paths
from a few "best" source arcs, missing many flights. Per-Source Pricing
builds isolated networks for each source arc, ensuring comprehensive coverage.

Usage:
    python examples/crew_pairing/solve_kasirzadeh_coverage.py [instance_path]
    python examples/crew_pairing/solve_kasirzadeh_coverage.py --solver cplex
    python examples/crew_pairing/solve_kasirzadeh_coverage.py --solver highs
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path (for running without installation)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.pricing import PerSourcePricing, PricingConfig
from opencg.master import HiGHSMasterProblem, HIGHS_AVAILABLE
from opencg.core.column import Column

# Try to import CPLEX
try:
    from opencg.master import CPLEXMasterProblem, CPLEX_AVAILABLE
except ImportError:
    CPLEX_AVAILABLE = False
    CPLEXMasterProblem = None


def get_data_path() -> Path:
    """Get the data directory path."""
    import os
    env_path = os.environ.get('OPENCG_DATA_PATH')
    if env_path:
        return Path(env_path)
    return Path(__file__).parent.parent.parent / "data"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve Kasirzadeh crew scheduling with high coverage"
    )
    parser.add_argument(
        "instance",
        nargs="?",
        default=None,
        help="Path to instance directory (default: data/kasirzadeh/instance1)"
    )
    parser.add_argument(
        "--solver",
        choices=["highs", "cplex"],
        default="highs",
        help="Master problem solver (default: highs)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum CG iterations (default: 50)"
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Maximum solve time in seconds (default: 300)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def compute_coverage(master, problem, lp_solution):
    """Compute flight coverage from LP solution."""
    covered = set()
    for col_id, val in lp_solution.column_values.items():
        if val > 1e-6:
            col = master.get_column(col_id)
            if col and not col.attributes.get('artificial'):
                covered.update(col.covered_items)
    return covered, 100.0 * len(covered) / len(problem.cover_constraints)


def main():
    """Main entry point."""
    args = parse_args()

    # Check solver availability
    if args.solver == "cplex":
        if not CPLEX_AVAILABLE:
            print("Error: CPLEX solver not available.")
            print("Install CPLEX and docplex: pip install docplex")
            return 1
        MasterClass = CPLEXMasterProblem
        solver_name = "CPLEX"
    else:
        if not HIGHS_AVAILABLE:
            print("Error: HiGHS solver not available.")
            print("Install with: pip install highspy")
            return 1
        MasterClass = HiGHSMasterProblem
        solver_name = "HiGHS"

    # Determine instance path
    if args.instance:
        instance_path = Path(args.instance)
    else:
        instance_path = get_data_path() / "kasirzadeh" / "instance1"

    if not instance_path.exists():
        print(f"Error: Instance not found: {instance_path}")
        return 1

    print("=" * 70)
    print("OpenCG - High-Coverage Crew Pairing Solver")
    print("=" * 70)
    print(f"Instance: {instance_path}")
    print(f"Solver:   {solver_name}")
    print()

    # Parse the instance
    print("Loading instance...")
    start_time = time.time()

    parser_config = ParserConfig(
        verbose=args.verbose,
        validate=True,
        options={
            'min_connection_time': 0.5,
            'max_connection_time': 4.0,
            'min_layover_time': 10.0,
            'max_layover_time': 24.0,
            'max_duty_time': 14.0,
            'max_pairing_days': 5,
        }
    )
    parser = KasirzadehParser(parser_config)

    if not parser.can_parse(instance_path):
        print(f"Error: Cannot parse {instance_path}")
        return 1

    problem = parser.parse(instance_path)
    load_time = time.time() - start_time

    print(f"  Loaded in {format_time(load_time)}")
    print(f"  Flights: {len(problem.cover_constraints)}")
    print(f"  Network: {problem.network.num_nodes} nodes, {problem.network.num_arcs} arcs")
    print()

    # Create master problem
    print(f"Creating {solver_name} master problem...")
    master = MasterClass(problem, verbosity=0)

    # Add artificial columns for feasibility
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

    # Create PerSourcePricing - high threshold for initial columns
    pricing_init = PricingConfig(
        max_columns=500,
        max_time=60.0,
        reduced_cost_threshold=1e10,  # Accept any column initially
        check_elementarity=True,
        use_dominance=True,
    )

    print("Creating PerSourcePricing algorithm...")
    pricing = PerSourcePricing(
        problem,
        config=pricing_init,
        max_labels_per_node=50,
        cols_per_source=5,
        time_per_source=0.1,
    )

    # Generate initial columns
    print("Generating initial columns...")
    solve_start = time.time()
    pricing.set_dual_values({})
    sol = pricing.solve()
    init_time = time.time() - solve_start

    print(f"  Found {len(sol.columns)} initial columns in {format_time(init_time)}")

    for col in sol.columns:
        col_with_id = col.with_id(next_col_id)
        next_col_id += 1
        master.add_column(col_with_id)

    # Switch to normal threshold for CG
    pricing_cg = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )
    pricing._config = pricing_cg

    # Run column generation
    print()
    print("Running Column Generation...")
    print("-" * 70)
    print(f"{'Iter':>5} {'Objective':>15} {'Columns':>10} {'New':>6} {'Coverage':>10} {'Time':>8}")
    print("-" * 70)

    cg_start = time.time()
    lp_sol = None
    final_coverage = 0.0

    for iteration in range(args.max_iter):
        # Check time limit
        elapsed = time.time() - cg_start
        if elapsed >= args.max_time:
            print(f"\nTime limit reached ({format_time(elapsed)})")
            break

        # Solve LP
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            print(f"\nLP not optimal: {lp_sol.status}")
            break

        # Get duals and run pricing
        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        # Compute coverage
        covered, pct = compute_coverage(master, problem, lp_sol)
        final_coverage = pct

        # Check convergence
        if not pricing_sol.columns:
            print(f"{iteration:>5} {lp_sol.objective_value:>15.2f} {master.num_columns:>10} "
                  f"{0:>6} {pct:>9.1f}% {elapsed:>7.1f}s")
            print()
            print("*** CONVERGED ***")
            break

        # Add new columns
        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

        # Progress output
        if iteration % 5 == 0 or len(pricing_sol.columns) < 10:
            print(f"{iteration:>5} {lp_sol.objective_value:>15.2f} {master.num_columns:>10} "
                  f"{len(pricing_sol.columns):>6} {pct:>9.1f}% {elapsed:>7.1f}s")

    solve_time = time.time() - solve_start

    # Final results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    if lp_sol:
        # Get final coverage
        covered, pct = compute_coverage(master, problem, lp_sol)
        uncovered = set(range(len(problem.cover_constraints))) - covered

        print(f"LP Objective:    {lp_sol.objective_value:.2f}")
        print(f"Total Columns:   {master.num_columns}")
        print(f"Coverage:        {pct:.1f}% ({len(covered)}/{len(problem.cover_constraints)} flights)")
        print()

        if uncovered and len(uncovered) <= 10:
            print(f"Uncovered flights: {sorted(uncovered)}")
        elif uncovered:
            print(f"Uncovered flights: {len(uncovered)} (first 10: {sorted(uncovered)[:10]})")
        print()

    print("Timing:")
    print(f"  Load time:    {format_time(load_time)}")
    print(f"  Solve time:   {format_time(solve_time)}")
    print(f"  Total time:   {format_time(load_time + solve_time)}")
    print()

    print("=" * 70)
    print("Done!")
    print("=" * 70)

    return 0 if final_coverage > 95.0 else 1


if __name__ == "__main__":
    sys.exit(main())
