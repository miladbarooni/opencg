"""
Crew Pairing Problem via Column Generation.

The Crew Pairing Problem (CPP) asks: given a set of flight legs, find the minimum
cost set of pairings (sequences of flights) that cover all flights exactly once,
respecting crew duty regulations.

This is a classic application of column generation where:
- Master problem: Select pairings (columns) to minimize cost while covering all flights
- Pricing problem: SPPRC to find pairings with negative reduced cost

Mathematical Formulation:
------------------------
Master Problem (Set Partitioning):
    min  sum_p c_p * x_p              (minimize total cost)
    s.t. sum_p a_fp * x_p = 1         (each flight covered exactly once)
         x_p >= 0, integer

Where:
- x_p = 1 if pairing p is selected, 0 otherwise
- c_p = cost of pairing p
- a_fp = 1 if flight f is in pairing p, 0 otherwise

Pricing Subproblem (SPPRC):
    Find path from source to sink in time-space network with:
    - Negative reduced cost: c_p - sum_f pi_f * a_fp < 0
    - Resource feasibility: duty time, flight time, rest time, etc.
    - Home base constraint: start and end at same base

Key Concepts:
------------
- **Pairing**: A sequence of flights a crew can legally operate, starting and
  ending at their home base, typically spanning 1-5 days.

- **Duty Period**: A single work period within a pairing, bounded by duty time
  limits (e.g., 14 hours max).

- **Home Base**: The crew's home airport. Pairings must start and end at the
  same base.

- **Time-Space Network**: Nodes represent (location, time) pairs. Arcs represent
  flights, connections, rests, and deadheads.

Usage:
------
    from opencg.applications.crew_pairing import (
        solve_crew_pairing,
        CrewPairingConfig,
    )
    from opencg.parsers import KasirzadehParser

    # Load instance
    parser = KasirzadehParser()
    problem = parser.parse("data/kasirzadeh/instance1")

    # Solve
    solution = solve_crew_pairing(problem)
    print(f"Objective: {solution.objective}")
    print(f"Coverage: {solution.coverage_pct:.1f}%")

Advanced Usage:
--------------
    from opencg.applications.crew_pairing import (
        FastPerSourcePricing,
        MultiBasePricing,
        HomeBaseResource,
    )

    # Use specialized pricing for better coverage
    pricing = FastPerSourcePricing(problem, max_labels_per_node=50)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time

from opencg.core.problem import Problem
from opencg.core.column import Column

# Import crew-pairing specific pricing algorithms
from opencg.applications.crew_pairing.pricing import (
    MultiBasePricingAlgorithm,
    BaseRestrictedLabelingAlgorithm,
    PerSourcePricing,
    FastPerSourcePricing,
)

# Import HomeBaseResource
from opencg.applications.crew_pairing.resources import HomeBaseResource


@dataclass
class CrewPairingConfig:
    """Configuration for crew pairing solver."""
    max_iterations: int = 100
    max_time: float = 600.0  # seconds
    pricing_max_columns: int = 200
    pricing_max_time: float = 30.0
    pricing_max_labels_per_node: int = 50
    cols_per_source: int = 5
    time_per_source: float = 0.1
    num_threads: int = 0  # Number of threads for parallel pricing (0=auto)
    solver: str = "highs"  # "highs" or "cplex"
    verbose: bool = False


@dataclass
class CrewPairingSolution:
    """Solution to a crew pairing problem."""
    objective: float
    objective_ip: Optional[float]
    num_pairings: int
    coverage_pct: float
    uncovered_flights: Set[int]
    pairings: List[Column]
    solve_time: float
    iterations: int
    num_columns: int


def solve_crew_pairing(
    problem: Problem,
    config: Optional[CrewPairingConfig] = None,
) -> CrewPairingSolution:
    """
    Solve a crew pairing problem using column generation.

    Args:
        problem: Problem instance (from KasirzadehParser or similar)
        config: Solver configuration

    Returns:
        CrewPairingSolution with results
    """
    from opencg.master import HiGHSMasterProblem, HIGHS_AVAILABLE
    from opencg.pricing import PricingConfig

    if config is None:
        config = CrewPairingConfig()

    start_time = time.time()

    # Select solver
    if config.solver == "cplex":
        try:
            from opencg.master import CPLEXMasterProblem, CPLEX_AVAILABLE
            if not CPLEX_AVAILABLE:
                raise ImportError("CPLEX not available")
            MasterClass = CPLEXMasterProblem
        except ImportError:
            if config.verbose:
                print("CPLEX not available, falling back to HiGHS")
            MasterClass = HiGHSMasterProblem
    else:
        MasterClass = HiGHSMasterProblem

    # Create master problem
    master = MasterClass(problem, verbosity=1 if config.verbose else 0)

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

    # Create pricing - use FastPerSourcePricing for best coverage
    pricing_config = PricingConfig(
        max_columns=config.pricing_max_columns,
        max_time=config.pricing_max_time,
        reduced_cost_threshold=1e10,  # Accept any column initially
        check_elementarity=True,
        use_dominance=True,
    )

    try:
        pricing = FastPerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=config.pricing_max_labels_per_node,
            cols_per_source=config.cols_per_source,
            time_per_source=config.time_per_source,
            num_threads=config.num_threads,
        )
    except ImportError:
        # Fall back to PerSourcePricing if C++ backend not available
        pricing = PerSourcePricing(
            problem,
            config=pricing_config,
            max_labels_per_node=config.pricing_max_labels_per_node,
            cols_per_source=config.cols_per_source,
            time_per_source=config.time_per_source,
        )

    # Generate initial columns
    if config.verbose:
        print("Generating initial columns...")

    pricing.set_dual_values({})
    sol = pricing.solve()

    for col in sol.columns:
        col_with_id = col.with_id(next_col_id)
        next_col_id += 1
        master.add_column(col_with_id)

    if config.verbose:
        print(f"  Found {len(sol.columns)} initial columns")

    # Switch to normal threshold for CG
    pricing._config = PricingConfig(
        max_columns=config.pricing_max_columns,
        max_time=config.pricing_max_time,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    # Column generation loop
    if config.verbose:
        print("Running Column Generation...")
        print(f"{'Iter':>5} {'Objective':>15} {'Columns':>10} {'New':>6} {'Coverage':>10}")
        print("-" * 50)

    lp_sol = None
    iterations = 0
    converged = False
    all_flights = set(range(len(problem.cover_constraints)))

    for iteration in range(config.max_iterations):
        iterations = iteration + 1

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= config.max_time:
            if config.verbose:
                print(f"Time limit reached ({elapsed:.1f}s)")
            break

        # Solve LP
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            if config.verbose:
                print(f"LP not optimal: {lp_sol.status}")
            break

        # Compute coverage and identify uncovered flights
        covered = set()
        for col_id, val in lp_sol.column_values.items():
            if val > 1e-6:
                col = master.get_column(col_id)
                if col and not col.attributes.get('artificial'):
                    covered.update(col.covered_items)

        uncovered = all_flights - covered
        coverage_pct = 100.0 * len(covered) / len(problem.cover_constraints)

        # Set uncovered flights as priority items for pricing
        if hasattr(pricing, 'set_priority_items'):
            pricing.set_priority_items(uncovered)

        # Get duals and run pricing
        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if config.verbose and (iteration % 5 == 0 or not pricing_sol.columns):
            print(f"{iteration:>5} {lp_sol.objective_value:>15.2f} "
                  f"{master.num_columns:>10} {len(pricing_sol.columns):>6} "
                  f"{coverage_pct:>9.1f}%")

        # Check convergence
        if not pricing_sol.columns:
            if config.verbose:
                print("Converged - no columns with negative reduced cost")
            converged = True
            break

        # Add new columns
        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

    solve_time = time.time() - start_time

    # Extract solution
    objective = lp_sol.objective_value if lp_sol else 0
    covered = set()
    pairings = []

    if lp_sol:
        for col_id, val in lp_sol.column_values.items():
            if val > 1e-6:
                col = master.get_column(col_id)
                if col and not col.attributes.get('artificial'):
                    covered.update(col.covered_items)
                    pairings.append(col)

    # Post-processing: Find columns covering remaining uncovered items
    uncovered = all_flights - covered
    if uncovered and config.verbose:
        print(f"\nPost-processing: {len(uncovered)} uncovered flights, checking for available columns...")

    # Iterate through all columns to find ones covering uncovered items
    for col_id in range(master.num_columns):
        col = master.get_column(col_id)
        if col and not col.attributes.get('artificial'):
            new_coverage = col.covered_items & uncovered
            if new_coverage:
                # Found a column covering some uncovered items
                pairings.append(col)
                covered.update(col.covered_items)
                uncovered = all_flights - covered
                if config.verbose:
                    print(f"  Added column {col_id} covering {len(new_coverage)} uncovered flights")
                if not uncovered:
                    break

    coverage_pct = 100.0 * len(covered) / len(problem.cover_constraints)

    return CrewPairingSolution(
        objective=objective,
        objective_ip=None,  # Could add IP solve if needed
        num_pairings=len(pairings),
        coverage_pct=coverage_pct,
        uncovered_flights=uncovered,
        pairings=pairings,
        solve_time=solve_time,
        iterations=iterations,
        num_columns=master.num_columns,
    )


__all__ = [
    # Main solver
    'solve_crew_pairing',
    'CrewPairingConfig',
    'CrewPairingSolution',
    # Pricing algorithms
    'MultiBasePricingAlgorithm',
    'BaseRestrictedLabelingAlgorithm',
    'PerSourcePricing',
    'FastPerSourcePricing',
    # Resources
    'HomeBaseResource',
]
