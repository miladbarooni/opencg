"""
Crew pairing specific pricing algorithms.

These algorithms are optimized for the crew pairing problem structure,
particularly handling multiple bases and ensuring comprehensive flight coverage.

Available Algorithms:
--------------------
- MultiBasePricingAlgorithm: Runs pricing per-base for balanced coverage
- BaseRestrictedLabelingAlgorithm: Labeling restricted to a single base
- PerSourcePricing: Per-source-arc pricing for comprehensive coverage
- FastPerSourcePricing: Optimized version with prebuilt networks
- MultiBaseCppPricing: C++ implementation of multi-base pricing
- MultiBaseBoostPricing: Boost library implementation
- OptimizedMultiBasePricing: Further optimized variant
- DiverseMultiBasePricing: Diversity-promoting pricing
- TargetedPricing: Targets specific uncovered flights
"""

# Re-export from original locations
# These will be moved to this package in the future

from opencg.pricing.multibase import (
    BaseRestrictedLabelingAlgorithm,
    MultiBasePricingAlgorithm,
)
from opencg.pricing.per_source_pricing import PerSourcePricing

try:
    from opencg.pricing.fast_per_source import FastPerSourcePricing
except ImportError:
    FastPerSourcePricing = None

try:
    from opencg.pricing.multibase_cpp import MultiBaseCppPricing
except ImportError:
    MultiBaseCppPricing = None

try:
    from opencg.pricing.multibase_boost import MultiBaseBoostPricing
except ImportError:
    MultiBaseBoostPricing = None

try:
    from opencg.pricing.optimized_multibase import OptimizedMultiBasePricing
except ImportError:
    OptimizedMultiBasePricing = None

try:
    from opencg.pricing.diverse_multibase import DiverseMultiBasePricing
except ImportError:
    DiverseMultiBasePricing = None

try:
    from opencg.pricing.targeted_pricing import TargetedPricing
except ImportError:
    TargetedPricing = None

try:
    from opencg.pricing.per_source_boost import PerSourceBoostPricing
except ImportError:
    PerSourceBoostPricing = None

__all__ = [
    'MultiBasePricingAlgorithm',
    'BaseRestrictedLabelingAlgorithm',
    'PerSourcePricing',
    'FastPerSourcePricing',
    'MultiBaseCppPricing',
    'MultiBaseBoostPricing',
    'OptimizedMultiBasePricing',
    'DiverseMultiBasePricing',
    'TargetedPricing',
    'PerSourceBoostPricing',
]
