"""Compare AcceleratedLabelingAlgorithm vs FastPerSourcePricing for crew pairing."""

import sys
sys.stdout.reconfigure(line_buffering=True)

from opencg.parsers import KasirzadehParser
from opencg.master import HiGHSMasterProblem
from opencg.core.column import Column
from opencg.pricing import AcceleratedLabelingAlgorithm, PricingConfig
from opencg.pricing.fast_per_source import FastPerSourcePricing
import time

# Load instance
parser = KasirzadehParser()
problem = parser.parse('data/kasirzadeh/instance1')

print('='*70)
print('Crew Pairing: AcceleratedLabelingAlgorithm vs FastPerSourcePricing')
print(f'Instance: {problem.name}')
print(f'  Flights: {len(problem.cover_constraints)}')
print(f'  Network arcs: {problem.network.num_arcs}')
print('='*70)
sys.stdout.flush()


def run_cg_with_pricing(problem, pricing_class, pricing_kwargs, name, max_iter=30):
    """Run column generation with specified pricing algorithm."""
    print(f'\nRunning {name}...', flush=True)

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

    # Create pricing with relaxed threshold for initial columns
    config = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=1e10,
        check_elementarity=True,
        use_dominance=True,
    )
    pricing = pricing_class(problem, config=config, **pricing_kwargs)

    # Generate initial columns
    print('  Generating initial columns...', flush=True)
    pricing.set_dual_values({})
    sol = pricing.solve()
    print(f'  Found {len(sol.columns)} initial columns', flush=True)

    for col in sol.columns:
        col_with_id = col.with_id(next_col_id)
        next_col_id += 1
        master.add_column(col_with_id)

    # Switch to normal threshold
    pricing._config = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    # Column generation loop
    print('  Running CG iterations...', flush=True)
    start = time.time()
    lp_sol = None
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1

        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            print(f'  LP not optimal: {lp_sol.status}', flush=True)
            break

        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if iteration % 5 == 0:
            print(f'    Iter {iteration}: obj={lp_sol.objective_value:.0f}, new_cols={len(pricing_sol.columns)}', flush=True)

        if not pricing_sol.columns:
            print(f'  Converged at iteration {iteration}', flush=True)
            break

        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

    elapsed = time.time() - start

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

    all_flights = set(range(len(problem.cover_constraints)))
    uncovered = all_flights - covered
    coverage_pct = 100.0 * len(covered) / len(problem.cover_constraints)

    return {
        'name': name,
        'objective': objective,
        'pairings': len(pairings),
        'coverage': coverage_pct,
        'uncovered': len(uncovered),
        'time': elapsed,
        'iterations': iterations,
        'columns': master.num_columns,
    }


# Test 1: AcceleratedLabelingAlgorithm (single source C++ labeling)
result1 = run_cg_with_pricing(
    problem,
    AcceleratedLabelingAlgorithm,
    {},  # no extra kwargs
    'AcceleratedLabeling (single source)'
)

# Test 2: FastPerSourcePricing (per-source C++ labeling)
result2 = run_cg_with_pricing(
    problem,
    FastPerSourcePricing,
    {'max_labels_per_node': 50, 'cols_per_source': 5, 'time_per_source': 0.1},
    'FastPerSource (per-source)'
)

# Print comparison
print()
print('='*70)
print('COMPARISON RESULTS')
print('='*70)
print(f"{'Metric':<20} {'SingleSource':>20} {'PerSource':>20}")
print('-'*70)
print(f"{'LP Objective':<20} {result1['objective']:>20.2f} {result2['objective']:>20.2f}")
print(f"{'Coverage %':<20} {result1['coverage']:>19.1f}% {result2['coverage']:>19.1f}%")
print(f"{'Uncovered flights':<20} {result1['uncovered']:>20} {result2['uncovered']:>20}")
print(f"{'Pairings used':<20} {result1['pairings']:>20} {result2['pairings']:>20}")
print(f"{'CG Iterations':<20} {result1['iterations']:>20} {result2['iterations']:>20}")
print(f"{'Total columns':<20} {result1['columns']:>20} {result2['columns']:>20}")
print(f"{'Time (s)':<20} {result1['time']:>20.2f} {result2['time']:>20.2f}")
print('='*70)
