"""
Optimized multi-base pricing using C++ labeling with performance optimizations.

This module provides a pricing algorithm that:
1. Runs separate C++ labeling per crew base
2. Uses topological order processing for DAGs
3. Uses label limits per node for beam search behavior
4. Optionally uses ng-path relaxation for faster elementarity
"""

import time
from typing import Dict, List, Optional, Set

from opencg.core.arc import ArcType
from opencg.core.column import Column
from opencg.core.node import NodeType
from opencg.core.problem import Problem
from opencg.pricing.base import (
    PricingProblem,
    PricingConfig,
    PricingSolution,
    PricingStatus,
)

# Import C++ backend
try:
    from opencg._core import (
        HAS_CPP_BACKEND,
        Network as CppNetwork,
        LabelingAlgorithm as CppLabelingAlgorithm,
        LabelingConfig as CppLabelingConfig,
    )
except ImportError:
    HAS_CPP_BACKEND = False
    CppNetwork = None
    CppLabelingAlgorithm = None
    CppLabelingConfig = None


class OptimizedMultiBasePricing(PricingProblem):
    """
    Optimized multi-base pricing using C++ labeling algorithm.

    Uses performance optimizations:
    - Topological order processing (faster for DAGs)
    - Label limits per node (beam search heuristic)
    - Optional ng-path relaxation for elementarity

    Runs separate pricing for each crew base, filtering the network
    to only include source/sink arcs for that base.
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 50,
        use_topological_order: bool = True,
        use_ng_path: bool = False,
        ng_neighborhood_size: int = 8,
    ):
        if not HAS_CPP_BACKEND:
            raise ImportError("C++ backend not available")

        super().__init__(problem, config)

        # Store optimization options
        self._max_labels_per_node = max_labels_per_node
        self._use_topological_order = use_topological_order
        self._use_ng_path = use_ng_path
        self._ng_neighborhood_size = ng_neighborhood_size

        # Find source/sink indices
        self._source_idx: Optional[int] = None
        self._sink_idx: Optional[int] = None
        self._find_source_sink()

        # Find bases and their arcs
        self._bases = self._find_bases()
        self._base_source_arcs = self._find_base_source_arcs()
        self._base_sink_arcs = self._find_base_sink_arcs()

        # Get numeric resources (excluding HomeBaseResource)
        self._numeric_resources = []
        self._resource_limits = []
        for r in problem.resources:
            if hasattr(r, 'max_value'):
                self._numeric_resources.append(r.name)
                self._resource_limits.append(r.max_value)

        # Build per-base C++ networks and algorithms
        self._base_networks: Dict[str, CppNetwork] = {}
        self._base_algorithms: Dict[str, CppLabelingAlgorithm] = {}
        self._base_arc_maps: Dict[str, Dict[int, int]] = {}

        for base in self._bases:
            self._build_base_network(base)

    def _find_source_sink(self) -> None:
        for i in range(self._problem.network.num_nodes):
            node = self._problem.network.get_node(i)
            if node is None:
                continue
            if node.node_type == NodeType.SOURCE:
                self._source_idx = i
            elif node.node_type == NodeType.SINK:
                self._sink_idx = i

    def _find_bases(self) -> List[str]:
        bases = set()
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base:
                    bases.add(base)
        return sorted(bases)

    def _find_base_source_arcs(self) -> Dict[str, Set[int]]:
        result = {base: set() for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base in result:
                    result[base].add(arc.index)
        return result

    def _find_base_sink_arcs(self) -> Dict[str, Set[int]]:
        result = {base: set() for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base')
                if base in result:
                    result[base].add(arc.index)
        return result

    def _build_base_network(self, base: str) -> None:
        """Build C++ network for a specific base."""
        py_network = self._problem.network

        cpp_network = CppNetwork()
        cpp_arc_to_py: Dict[int, int] = {}

        # Add all nodes
        for i in range(py_network.num_nodes):
            node = py_network.get_node(i)
            if node.node_type == NodeType.SOURCE:
                cpp_network.add_source()
            elif node.node_type == NodeType.SINK:
                cpp_network.add_sink()
            else:
                cpp_network.add_node()

        # Add arcs (filtering source/sink by base)
        cpp_arc_idx = 0
        for arc in py_network.arcs:
            # Filter source arcs - only include this base's source arcs
            if arc.arc_type == ArcType.SOURCE_ARC:
                if arc.index not in self._base_source_arcs[base]:
                    continue

            # Filter sink arcs - only include this base's sink arcs
            if arc.arc_type == ArcType.SINK_ARC:
                if arc.index not in self._base_sink_arcs[base]:
                    continue

            # Get resource consumption (only numeric resources)
            res_consumption = []
            for res_name in self._numeric_resources:
                val = arc.get_consumption(res_name, 0.0)
                res_consumption.append(val)

            # Get covered items (only flight arcs)
            if arc.arc_type == ArcType.FLIGHT:
                covered_items = [arc.index]
            else:
                covered_items = []

            cpp_network.add_arc(
                arc.source, arc.target, arc.cost,
                res_consumption, covered_items
            )
            cpp_arc_to_py[cpp_arc_idx] = arc.index
            cpp_arc_idx += 1

        # Create config with optimizations
        cpp_config = CppLabelingConfig()
        cols_per_base = self._config.max_columns // len(self._bases) if self._config.max_columns > 0 else 0
        cpp_config.max_columns = cols_per_base if cols_per_base > 0 else 0
        time_per_base = self._config.max_time / len(self._bases) if self._config.max_time > 0 else 0.0
        cpp_config.max_time = time_per_base if time_per_base > 0 else 0.0
        cpp_config.max_labels = self._config.max_labels if self._config.max_labels > 0 else 0
        cpp_config.rc_threshold = self._config.reduced_cost_threshold
        cpp_config.check_dominance = self._config.use_dominance

        # Apply optimization options
        cpp_config.use_topological_order = self._use_topological_order
        cpp_config.max_labels_per_node = self._max_labels_per_node
        cpp_config.use_ng_path = self._use_ng_path
        cpp_config.ng_neighborhood_size = self._ng_neighborhood_size

        # Elementarity: use ng-path if enabled, otherwise use config setting
        if self._use_ng_path:
            cpp_config.check_elementarity = False  # ng-path handles this
        else:
            cpp_config.check_elementarity = self._config.check_elementarity

        # Create algorithm
        algo = CppLabelingAlgorithm(
            cpp_network,
            len(self._numeric_resources),
            self._resource_limits,
            cpp_config
        )

        self._base_networks[base] = cpp_network
        self._base_algorithms[base] = algo
        self._base_arc_maps[base] = cpp_arc_to_py

    def _on_duals_updated(self) -> None:
        """Update dual values in all base algorithms."""
        for base in self._bases:
            if base in self._base_algorithms:
                self._base_algorithms[base].set_dual_values(self._dual_values)

    def _solve_impl(self) -> PricingSolution:
        """Run pricing for each base and combine results."""
        start_time = time.time()

        all_columns: List[Column] = []
        best_rc = None
        total_labels = 0
        total_dominated = 0

        for base in self._bases:
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            if base not in self._base_algorithms:
                continue

            algo = self._base_algorithms[base]
            arc_map = self._base_arc_maps[base]

            # Solve for this base
            result = algo.solve()
            total_labels += result.labels_created
            total_dominated += result.labels_dominated

            # Convert columns
            for cpp_label in result.columns:
                column = self._convert_cpp_label(cpp_label, arc_map, base)
                if column is not None:
                    all_columns.append(column)
                    if best_rc is None or column.reduced_cost < best_rc:
                        best_rc = column.reduced_cost

        # Sort and limit
        all_columns.sort(key=lambda c: c.reduced_cost)
        if self._config.max_columns > 0 and len(all_columns) > self._config.max_columns:
            all_columns = all_columns[:self._config.max_columns]

        solve_time = time.time() - start_time

        if all_columns:
            status = PricingStatus.COLUMNS_FOUND
        else:
            status = PricingStatus.NO_COLUMNS

        return PricingSolution(
            status=status,
            columns=all_columns,
            best_reduced_cost=best_rc,
            num_labels_created=total_labels,
            num_labels_dominated=total_dominated,
            solve_time=solve_time,
            iterations=len(all_columns),
        )

    def _convert_cpp_label(
        self,
        cpp_label,
        arc_map: Dict[int, int],
        base: str
    ) -> Optional[Column]:
        """Convert C++ label to Python Column."""
        try:
            # Get arc indices from the label
            cpp_arcs = cpp_label.get_arc_indices()
            py_arcs = [arc_map.get(i, -1) for i in cpp_arcs]
            py_arcs = [a for a in py_arcs if a >= 0]

            if not py_arcs:
                return None

            # Get covered items
            items = cpp_label.covered_items
            if callable(items):
                items = items()

            # Get cost/reduced cost
            cost = cpp_label.cost
            if callable(cost):
                cost = cost()

            reduced_cost = cpp_label.reduced_cost
            if callable(reduced_cost):
                reduced_cost = reduced_cost()

            return Column(
                arc_indices=tuple(py_arcs),
                cost=cost,
                reduced_cost=reduced_cost,
                covered_items=frozenset(items),
                attributes={'base': base},
            )
        except Exception:
            return None
