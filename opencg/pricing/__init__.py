"""
Pricing subproblem module - SPPRC algorithms for column generation.

The pricing problem finds columns with negative reduced cost.
For crew pairing and vehicle routing, this is typically a Shortest Path
Problem with Resource Constraints (SPPRC).

This module provides:
- PricingProblem: Abstract base class for custom implementations
- LabelingAlgorithm: Standard mono-directional labeling algorithm
- ElementaryLabelingAlgorithm: Enforces elementary (no-repeat) paths
- HeuristicLabelingAlgorithm: Faster heuristic with early termination
- Label: Data structure representing partial paths
- PricingSolution: Result of pricing solve

Usage:
------
Basic usage with default labeling algorithm:

    >>> from opencg.pricing import LabelingAlgorithm
    >>> pricing = LabelingAlgorithm(problem)
    >>> pricing.set_dual_values(master.get_dual_values())
    >>> solution = pricing.solve()
    >>> if solution.has_negative_reduced_cost:
    ...     for col in solution.columns:
    ...         master.add_column(col)

Using heuristic pricing for faster results:

    >>> from opencg.pricing import HeuristicLabelingAlgorithm, PricingConfig
    >>> config = PricingConfig(max_columns=10, max_time=5.0)
    >>> pricing = HeuristicLabelingAlgorithm(
    ...     problem,
    ...     config=config,
    ...     max_labels_per_node=50,
    ...     early_termination_count=5,
    ... )
    >>> solution = pricing.solve()

Creating a custom pricing solver:

    >>> from opencg.pricing import PricingProblem, PricingSolution
    >>>
    >>> class MyCustomPricing(PricingProblem):
    ...     def _solve_impl(self) -> PricingSolution:
    ...         # Custom SPPRC algorithm
    ...         ...

Algorithm Overview:
------------------
The labeling algorithm works as follows:

1. Initialize a label at the source node with initial resource values
2. Maintain a priority queue of labels ordered by reduced cost
3. For each label:
   a. Extend along all outgoing arcs
   b. Check resource feasibility for each extension
   c. Check dominance: if a new label is dominated by an existing
      label at the same node, discard it
   d. Add non-dominated labels to the queue
4. Collect labels at the sink node with negative reduced cost
5. Convert labels to columns

Dominance:
---------
Label L1 dominates L2 at the same node if:
- L1.reduced_cost <= L2.reduced_cost
- L1 dominates on all resources (resource-specific rule)
- L1.covered_items ⊆ L2.covered_items

Dominated labels can be safely discarded without losing optimal columns.

Customization Points:
--------------------
The PricingProblem ABC provides hooks for customization:

1. Required methods:
   - _solve_impl(): The pricing algorithm

2. Hooks:
   - _on_duals_updated(): Called when dual values change
   - _before_solve(): Called before solving
   - _after_solve(): Called after solving
   - _create_column_from_label(): Customize column creation
"""

# Label and pool
from opencg.pricing.label import Label, LabelPool

# Solution and config
from opencg.pricing.base import (
    PricingProblem,
    PricingConfig,
    PricingSolution,
    PricingStatus,
)

# Labeling algorithms
from opencg.pricing.labeling import (
    LabelingAlgorithm,
    ElementaryLabelingAlgorithm,
    HeuristicLabelingAlgorithm,
)

# Accelerated algorithm (uses C++ when available)
from opencg.pricing.accelerated import (
    AcceleratedLabelingAlgorithm,
    create_labeling_algorithm,
)

# Multi-base pricing for crew pairing
from opencg.pricing.multibase import (
    MultiBasePricingAlgorithm,
    BaseRestrictedLabelingAlgorithm,
)

# Multi-base C++ pricing (uses C++ backend per base)
try:
    from opencg.pricing.multibase_cpp import MultiBaseCppPricing
except ImportError:
    MultiBaseCppPricing = None

# Multi-base Boost pricing (uses Boost r_c_shortest_paths)
try:
    from opencg.pricing.multibase_boost import MultiBaseBoostPricing
except ImportError:
    MultiBaseBoostPricing = None

# Optimized multi-base pricing (uses C++ with optimizations)
try:
    from opencg.pricing.optimized_multibase import OptimizedMultiBasePricing
except ImportError:
    OptimizedMultiBasePricing = None

# Diversity-promoting multi-base pricing
try:
    from opencg.pricing.diverse_multibase import DiverseMultiBasePricing
except ImportError:
    DiverseMultiBasePricing = None

# Targeted pricing for uncovered flights
try:
    from opencg.pricing.targeted_pricing import TargetedPricing
except ImportError:
    TargetedPricing = None

# Per-source-arc pricing for comprehensive coverage
try:
    from opencg.pricing.per_source_pricing import PerSourcePricing
except ImportError:
    PerSourcePricing = None

# Per-source-arc pricing using Boost
try:
    from opencg.pricing.per_source_boost import PerSourceBoostPricing
except ImportError:
    PerSourceBoostPricing = None

# Fast per-source pricing with prebuilt networks
try:
    from opencg.pricing.fast_per_source import FastPerSourcePricing
except ImportError:
    FastPerSourcePricing = None

__all__ = [
    # Data structures
    'Label',
    'LabelPool',

    # Base class and config
    'PricingProblem',
    'PricingConfig',
    'PricingSolution',
    'PricingStatus',

    # Algorithms
    'LabelingAlgorithm',
    'ElementaryLabelingAlgorithm',
    'HeuristicLabelingAlgorithm',

    # Accelerated (C++ backend)
    'AcceleratedLabelingAlgorithm',
    'create_labeling_algorithm',

    # Multi-base pricing
    'MultiBasePricingAlgorithm',
    'BaseRestrictedLabelingAlgorithm',
    'MultiBaseCppPricing',
    'MultiBaseBoostPricing',
    'OptimizedMultiBasePricing',
    'DiverseMultiBasePricing',
    'TargetedPricing',
    'PerSourcePricing',
    'PerSourceBoostPricing',
    'FastPerSourcePricing',
]
