"""
Example: Solving a Kasirzadeh crew scheduling instance using Column Generation.

This example demonstrates the complete workflow:
1. Load a Kasirzadeh benchmark instance
2. Set up the column generation algorithm
3. Solve the crew pairing problem
4. Analyze and report results

Usage:
    python examples/crew_pairing/solve_kasirzadeh.py [instance_path] [--max-iter N] [--verbose]

Prerequisites:
    - Unzip one of the Kasirzadeh instances (e.g., instance1.zip)
    - Install opencg with HiGHS: pip install -e ".[solvers]"
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path (for running without installation)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.solver import ColumnGeneration, CGConfig, CGStatus
from opencg.pricing import LabelingAlgorithm, PricingConfig
from opencg.master import HiGHSMasterProblem, HIGHS_AVAILABLE


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
        description="Solve Kasirzadeh crew scheduling instance using Column Generation"
    )
    parser.add_argument(
        "instance",
        nargs="?",
        default=None,
        help="Path to instance directory (default: data/kasirzadeh/instance1)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum CG iterations (default: 100)"
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Maximum solve time in seconds (default: 300)"
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=50,
        help="Maximum columns per pricing iteration (default: 50)"
    )
    parser.add_argument(
        "--solve-ip",
        action="store_true", 
        help="Solve integer program after LP converges"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-stabilization",
        action="store_true",
        help="Disable dual stabilization"
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


def print_progress(cg, iteration):
    """Callback to print progress during solving."""
    lp_obj = iteration.master_objective if iteration.master_objective is not None else 0.0
    rc = iteration.best_reduced_cost if iteration.best_reduced_cost is not None else 0.0
    cols = iteration.num_columns_added if iteration.num_columns_added is not None else 0
    print(f"  Iter {iteration.iteration:3d}: "
          f"LP={lp_obj:12.4f}  "
          f"RC={rc:10.4f}  "
          f"Cols={cols:3d}  "
          f"Total={cg.column_pool.size:5d}  "
          f"Time={iteration.master_time + iteration.pricing_time:.2f}s")
    return True  # Continue solving


def main():
    """Main entry point."""
    args = parse_args()

    # Check HiGHS availability
    if not HIGHS_AVAILABLE:
        print("Error: HiGHS solver not available.")
        print("Install with: pip install highspy")
        return 1

    # Determine instance path
    if args.instance:
        instance_path = Path(args.instance)
    else:
        instance_path = get_data_path() / "kasirzadeh" / "instance1"

    if not instance_path.exists():
        print(f"Error: Instance not found: {instance_path}")
        return 1

    print("=" * 70)
    print("OpenCG - Crew Pairing Solver")
    print("=" * 70)
    print(f"Instance: {instance_path}")
    print()

    # Parse the instance
    print("Loading instance...")
    start_time = time.time()

    # Configure parser with constraints from Kasirzadeh et al. paper
    parser_config = ParserConfig(
        verbose=args.verbose,
        validate=True,
        options={
            # Short connections (count as duty time)
            'min_connection_time': 0.5,  # Minimum 30 min connection
            'max_connection_time': 4.0,  # Max 4 hours for short connection

            # Overnight layovers (reset duty time) - per paper: >= 10 hours
            'min_layover_time': 10.0,   # Minimum 10 hours rest
            'max_layover_time': 24.0,   # Max 24 hours layover

            # Duty constraints
            'max_duty_time': 14.0,  # Maximum duty period in hours

            # Pairing constraints - per paper: 1-5 days
            'max_pairing_days': 5,  # Maximum 5 duty days per pairing
        }
    )
    parser = KasirzadehParser(parser_config)

    if not parser.can_parse(instance_path):
        print(f"Error: Cannot parse {instance_path}")
        print("Make sure the directory contains listOfBases.csv and day_*.csv files")
        return 1

    problem = parser.parse(instance_path)
    load_time = time.time() - start_time

    print(f"  Loaded in {format_time(load_time)}")
    print(f"  Flights: {problem.num_cover_constraints}")
    print(f"  Network: {problem.network.num_nodes} nodes, {problem.network.num_arcs} arcs")
    print()

    # Configure column generation
    cg_config = CGConfig(
        max_iterations=args.max_iter,
        max_time=args.max_time,
        solve_ip=args.solve_ip,
        verbose=args.verbose,
        use_stabilization=not args.no_stabilization,
    )

    # Configure pricing algorithm
    pricing_config = PricingConfig(
        max_columns=args.max_columns,
        reduced_cost_threshold=-1e-6,
        use_dominance=True,
        check_elementarity=True,  # Elementary paths for crew scheduling
    )

    print("Setting up Column Generation...")
    print(f"  Max iterations: {cg_config.max_iterations}")
    print(f"  Max time: {format_time(cg_config.max_time)}")
    print(f"  Solve IP: {cg_config.solve_ip}")
    print(f"  Stabilization: {cg_config.use_stabilization}")
    print(f"  Max columns/iter: {pricing_config.max_columns}")
    print()

    # Create column generation solver
    cg = ColumnGeneration(problem, cg_config)

    # Set custom pricing with our configuration
    pricing = LabelingAlgorithm(problem, pricing_config)
    cg.set_pricing(pricing)

    # Add progress callback
    if not args.verbose:
        print("Solving...")
        print("-" * 70)
        cg.add_callback(print_progress)

    # Solve
    solve_start = time.time()
    solution = cg.solve()
    solve_time = time.time() - solve_start

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Status: {solution.status.name}")
    print()

    if solution.is_feasible:
        print(f"LP Objective: {solution.lp_objective:.4f}")
        if solution.ip_objective is not None:
            print(f"IP Objective: {solution.ip_objective:.4f}")
            gap = abs(solution.ip_objective - solution.lp_objective) / max(1e-10, abs(solution.lp_objective)) * 100
            print(f"Gap: {gap:.2f}%")
        print()

    print(f"Iterations: {solution.iterations}")
    print(f"Total columns generated: {solution.total_columns}")
    print()

    print("Timing:")
    print(f"  Load time:    {format_time(load_time)}")
    print(f"  Solve time:   {format_time(solve_time)}")
    print(f"  Master time:  {format_time(solution.master_time)}")
    print(f"  Pricing time: {format_time(solution.pricing_time)}")
    print(f"  Total time:   {format_time(load_time + solve_time)}")
    print()

    # Print solution columns if IP was solved
    if solution.is_integer and solution.columns:
        print("Solution pairings:")
        print("-" * 70)
        for i, col in enumerate(solution.columns):
            if col.value and col.value > 0.5:  # Integer solution
                flights_covered = len(col.covered_items) if col.covered_items else 0
                print(f"  Pairing {i+1}: {flights_covered} flights, cost={col.cost:.2f}")
                if args.verbose and col.arc_indices:
                    print(f"    Arcs: {col.arc_indices[:10]}...")
        print()

    # Print convergence summary
    if solution.iteration_history:
        print("Convergence:")
        print("-" * 70)
        history = solution.get_convergence_history()
        if len(history) > 10:
            # Show first 5 and last 5
            for i, obj in enumerate(history[:5]):
                print(f"  Iter {i+1:3d}: {obj:.4f}")
            print("  ...")
            for i, obj in enumerate(history[-5:], len(history)-4):
                print(f"  Iter {i:3d}: {obj:.4f}")
        else:
            for i, obj in enumerate(history):
                print(f"  Iter {i+1:3d}: {obj:.4f}")
        print()

    print("=" * 70)
    print("Done!")
    print("=" * 70)

    return 0 if solution.is_feasible else 1


if __name__ == "__main__":
    sys.exit(main())
