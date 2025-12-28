"""
C++ core module for high-performance SPPRC computations.

This module provides Python bindings to optimized C++ implementations of:
- Network: Graph representation for SPPRC
- Label/LabelPool: Label data structures for labeling algorithm
- LabelingAlgorithm: High-performance SPPRC solver
- LabelingConfig/LabelingResult: Configuration and results

When the C++ backend is not available, the pure Python implementations
from opencg.pricing will be used as fallback.

Usage:
    >>> from opencg._core import Network, LabelingAlgorithm, LabelingConfig
    >>> net = Network()
    >>> source = net.add_source()
    >>> sink = net.add_sink()
    >>> # ... build network ...
    >>> algo = LabelingAlgorithm(net, num_resources=1, resource_limits=[10.0])
    >>> algo.set_dual_values({0: 5.0})
    >>> result = algo.solve()
"""

# Try to import the C++ extension
try:
    from opencg._core._core import (
        Arc,
        # Label types
        Label,
        # Labeling algorithm
        LabelingAlgorithm,
        LabelingConfig,
        LabelingResult,
        LabelPool,
        # Network types
        Network,
        Node,
    )
    HAS_CPP_BACKEND = True

except ImportError:
    # C++ extension not available, flag for fallback
    HAS_CPP_BACKEND = False

    # Import placeholders that will raise informative errors
    def _not_available(*args, **kwargs):
        raise ImportError(
            "C++ backend not available. Either:\n"
            "  1. Install from source with C++ compilation: pip install .\n"
            "  2. Use pure Python implementations from opencg.pricing instead"
        )

    Network = _not_available
    Arc = _not_available
    Node = _not_available
    Label = _not_available
    LabelPool = _not_available
    LabelingAlgorithm = _not_available
    LabelingConfig = _not_available
    LabelingResult = _not_available


__all__ = [
    'HAS_CPP_BACKEND',
    # Network
    'Network',
    'Arc',
    'Node',
    # Labels
    'Label',
    'LabelPool',
    # Algorithm
    'LabelingAlgorithm',
    'LabelingConfig',
    'LabelingResult',
]
