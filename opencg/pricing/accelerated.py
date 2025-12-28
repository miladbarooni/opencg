"""
Accelerated labeling algorithm using C++ backend.

This module provides a high-performance labeling algorithm that uses the
C++ implementation when available, with automatic fallback to pure Python.

The C++ backend provides 10-100x speedup for large instances by:
- Using contiguous memory layouts for cache efficiency
- Avoiding Python object overhead during label extension
- Using optimized priority queue implementations

Usage:
    >>> from opencg.pricing import AcceleratedLabelingAlgorithm
    >>> pricing = AcceleratedLabelingAlgorithm(problem)
    >>> pricing.set_dual_values(duals)
    >>> solution = pricing.solve()

The interface is identical to LabelingAlgorithm, so they are interchangeable.
"""

import time
from typing import Optional

from opencg.core.arc import ArcType
from opencg.core.column import Column
from opencg.core.node import NodeType
from opencg.core.problem import Problem
from opencg.pricing.base import (
    PricingConfig,
    PricingProblem,
    PricingSolution,
    PricingStatus,
)

# Try to import C++ backend
try:
    from opencg._core import HAS_CPP_BACKEND
    if HAS_CPP_BACKEND:
        from opencg._core import (
            LabelingAlgorithm as CppLabelingAlgorithm,
        )
        from opencg._core import (
            LabelingConfig as CppLabelingConfig,
        )
        from opencg._core import (
            LabelingResult,
        )
        from opencg._core import (
            Network as CppNetwork,
        )
    else:
        CppNetwork = None
        CppLabelingAlgorithm = None
        CppLabelingConfig = None
        LabelingResult = None
except ImportError:
    HAS_CPP_BACKEND = False
    CppNetwork = None
    CppLabelingAlgorithm = None
    CppLabelingConfig = None
    LabelingResult = None


class AcceleratedLabelingAlgorithm(PricingProblem):
    """
    High-performance labeling algorithm using C++ backend.

    This class provides a drop-in replacement for LabelingAlgorithm with
    significant performance improvements for large instances. It automatically
    falls back to the pure Python implementation if C++ is not available.

    The C++ backend is used when:
    1. The C++ extension is compiled and available
    2. The problem network can be converted to C++ format

    Example:
        >>> from opencg.pricing import AcceleratedLabelingAlgorithm, PricingConfig
        >>> config = PricingConfig(max_columns=50, max_time=30.0)
        >>> pricing = AcceleratedLabelingAlgorithm(problem, config)
        >>> pricing.set_dual_values(duals)
        >>> solution = pricing.solve()
        >>> print(f"Found {len(solution.columns)} columns")

    Attributes:
        uses_cpp_backend: True if using C++ acceleration
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None
    ):
        """
        Initialize the accelerated labeling algorithm.

        Args:
            problem: The Problem instance
            config: Optional configuration
        """
        super().__init__(problem, config)

        self._uses_cpp = False
        self._cpp_network: Optional[CppNetwork] = None
        self._cpp_algo: Optional[CppLabelingAlgorithm] = None
        self._py_to_cpp_node: dict[int, int] = {}
        self._resource_names: list[str] = []

        # Try to set up C++ backend
        if HAS_CPP_BACKEND:
            try:
                self._setup_cpp_backend()
                self._uses_cpp = True
            except Exception as e:
                # Fall back to Python if C++ setup fails
                import warnings
                warnings.warn(f"C++ backend setup failed, using Python: {e}")
                self._uses_cpp = False

        # If C++ not available, import Python labeling
        if not self._uses_cpp:
            from opencg.pricing.labeling import LabelingAlgorithm
            self._py_labeling = LabelingAlgorithm(problem, config)

    @property
    def uses_cpp_backend(self) -> bool:
        """Whether C++ acceleration is being used."""
        return self._uses_cpp

    def _setup_cpp_backend(self) -> None:
        """Set up the C++ network and algorithm."""
        py_network = self._problem.network

        # Get resource names for conversion
        self._resource_names = [r.name for r in self._problem.resources]

        # Check if we have time windows (look for time resource)
        self._has_time_windows = False
        self._time_resource_index = 0
        for i, name in enumerate(self._resource_names):
            if name.lower() == 'time':
                self._has_time_windows = True
                self._time_resource_index = i
                break

        # Create C++ network
        self._cpp_network = CppNetwork()
        self._py_to_cpp_node = {}

        # Add nodes (preserving source/sink designation)
        for i in range(py_network.num_nodes):
            node = py_network.get_node(i)
            if node.node_type == NodeType.SOURCE:
                cpp_idx = self._cpp_network.add_source()
            elif node.node_type == NodeType.SINK:
                cpp_idx = self._cpp_network.add_sink()
            else:
                cpp_idx = self._cpp_network.add_node()
            self._py_to_cpp_node[i] = cpp_idx

        # Add arcs with resource consumption
        # Get item IDs from cover constraints
        item_ids = {c.item_id for c in self._problem.cover_constraints}

        for arc in py_network.arcs:
            cpp_source = self._py_to_cpp_node[arc.source]
            cpp_target = self._py_to_cpp_node[arc.target]

            # Convert resource consumption from dict to list (by index)
            res_consumption = []
            for res_name in self._resource_names:
                val = arc.get_consumption(res_name, 0.0)
                res_consumption.append(val)

            # Get covered items:
            # 1. For VRP: check customer_id attribute
            # 2. For crew pairing: FLIGHT arcs cover themselves
            covered_items = []
            customer_id = arc.get_attribute('customer_id', None)
            if customer_id is not None and customer_id in item_ids:
                covered_items = [customer_id]
            elif arc.arc_type == ArcType.FLIGHT and arc.index in item_ids:
                covered_items = [arc.index]

            cpp_arc_idx = self._cpp_network.add_arc(
                cpp_source, cpp_target, arc.cost,
                res_consumption, covered_items
            )

            # Set time window attributes if present
            if self._has_time_windows:
                travel_time = arc.get_attribute('travel_time', 0.0)
                service_time = arc.get_attribute('service_time', 0.0)
                earliest = arc.get_attribute('earliest', 0.0)
                latest = arc.get_attribute('latest', 1e30)
                cpp_arc = self._cpp_network.arc(cpp_arc_idx)
                cpp_arc.travel_time = travel_time
                cpp_arc.service_time = service_time
                cpp_arc.earliest = earliest
                cpp_arc.latest = latest

        # Create C++ labeling algorithm
        resource_limits = [r.max_value for r in self._problem.resources]

        cpp_config = CppLabelingConfig()
        cpp_config.max_columns = self._config.max_columns if self._config.max_columns > 0 else 0
        cpp_config.max_time = self._config.max_time if self._config.max_time > 0 else 0.0
        cpp_config.max_labels = self._config.max_labels if self._config.max_labels > 0 else 0
        cpp_config.rc_threshold = self._config.reduced_cost_threshold
        cpp_config.check_dominance = self._config.use_dominance
        cpp_config.check_elementarity = self._config.check_elementarity

        # Enable time window processing if we have time resource
        if self._has_time_windows:
            cpp_config.use_time_windows = True
            cpp_config.time_resource_index = self._time_resource_index

        self._cpp_algo = CppLabelingAlgorithm(
            self._cpp_network,
            len(resource_limits),
            resource_limits,
            cpp_config
        )

    def _on_duals_updated(self) -> None:
        """Update dual values in the backend."""
        if self._uses_cpp and self._cpp_algo is not None:
            self._cpp_algo.set_dual_values(self._dual_values)
        elif not self._uses_cpp:
            self._py_labeling.set_dual_values(self._dual_values)

    def _solve_impl(self) -> PricingSolution:
        """
        Solve using C++ or Python backend.

        Returns:
            PricingSolution with found columns
        """
        if self._uses_cpp:
            try:
                return self._solve_cpp()
            except (OverflowError, RuntimeError) as e:
                # C++ backend failed, fall back to Python
                import warnings
                warnings.warn(f"C++ pricing failed, falling back to Python: {e}")
                # Initialize Python fallback if not already done
                if not hasattr(self, '_py_labeling') or self._py_labeling is None:
                    from opencg.pricing.labeling import LabelingAlgorithm
                    self._py_labeling = LabelingAlgorithm(self._problem, self._config)
                    self._py_labeling.set_dual_values(self._dual_values)
                return self._py_labeling.solve()
        else:
            return self._py_labeling.solve()

    def _solve_cpp(self) -> PricingSolution:
        """Solve using C++ backend."""
        time.time()

        # Run C++ algorithm
        result = self._cpp_algo.solve()

        # Convert C++ result to Python PricingSolution
        columns = []
        for cpp_label in result.columns:
            column = self._create_column_from_cpp_label(cpp_label)
            columns.append(column)

        # Map C++ status to Python status
        status_map = {
            'OPTIMAL': PricingStatus.OPTIMAL,
            'COLUMNS_FOUND': PricingStatus.COLUMNS_FOUND,
            'NO_COLUMNS': PricingStatus.NO_COLUMNS,
            'TIME_LIMIT': PricingStatus.TIME_LIMIT,
            'LABEL_LIMIT': PricingStatus.ITERATION_LIMIT,
            'ERROR': PricingStatus.NO_COLUMNS,
        }
        status = status_map.get(result.status.name, PricingStatus.NO_COLUMNS)

        return PricingSolution(
            status=status,
            columns=columns,
            best_reduced_cost=result.best_reduced_cost if columns else None,
            num_labels_created=result.labels_created,
            num_labels_dominated=result.labels_dominated,
            solve_time=result.solve_time,
            iterations=result.iterations,
        )

    def _create_column_from_cpp_label(self, cpp_label) -> Column:
        """Create a Python Column from a C++ Label."""
        # Get arc indices from C++ label
        arc_indices = tuple(cpp_label.get_arc_indices())

        # Get resource values
        resource_values = {}
        for i, name in enumerate(self._resource_names):
            resource_values[name] = cpp_label.resource(i)

        # Get covered items (may be property or method depending on binding)
        items = cpp_label.covered_items
        if callable(items):
            items = items()
        covered_items = frozenset(items)

        # Get cost and reduced_cost (may be property or method)
        cost = cpp_label.cost
        if callable(cost):
            cost = cost()

        reduced_cost = cpp_label.reduced_cost
        if callable(reduced_cost):
            reduced_cost = reduced_cost()

        return Column(
            arc_indices=arc_indices,
            cost=cost,
            resource_values=resource_values,
            covered_items=covered_items,
            reduced_cost=reduced_cost,
        )


def create_labeling_algorithm(
    problem: Problem,
    config: Optional[PricingConfig] = None,
    prefer_cpp: bool = True
) -> PricingProblem:
    """
    Factory function to create the best available labeling algorithm.

    This function automatically selects between C++ and Python implementations
    based on availability and user preference.

    Args:
        problem: The Problem instance
        config: Optional configuration
        prefer_cpp: If True (default), use C++ when available

    Returns:
        A labeling algorithm instance (either AcceleratedLabelingAlgorithm
        or LabelingAlgorithm)

    Example:
        >>> pricing = create_labeling_algorithm(problem)
        >>> print(f"Using C++: {pricing.uses_cpp_backend}")
    """
    if prefer_cpp and HAS_CPP_BACKEND:
        algo = AcceleratedLabelingAlgorithm(problem, config)
        if algo.uses_cpp_backend:
            return algo

    # Fall back to Python
    from opencg.pricing.labeling import LabelingAlgorithm
    return LabelingAlgorithm(problem, config)
