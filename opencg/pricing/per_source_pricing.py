"""
Per-source-arc pricing for comprehensive coverage.

The issue with beam search is that it only explores paths from a few "best"
source arcs, missing many flights. This module provides pricing that
builds isolated networks for each source arc, ensuring all parts of
the network are explored.

Strategy:
1. For each source arc, build a restricted network containing ONLY that arc
   (plus matching sink arcs for the same base)
2. Run labeling on each isolated network
3. Combine results, selecting columns that cover new items

This ensures every source arc gets a chance to find columns, rather than
having a few dominant source arcs take all the beam budget.
"""

import time
from collections import defaultdict
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

# Import C++ backend
try:
    from opencg._core import (
        HAS_CPP_BACKEND,
    )
    from opencg._core import (
        LabelingAlgorithm as CppLabelingAlgorithm,
    )
    from opencg._core import (
        LabelingConfig as CppLabelingConfig,
    )
    from opencg._core import (
        Network as CppNetwork,
    )
except ImportError:
    HAS_CPP_BACKEND = False
    CppNetwork = None
    CppLabelingAlgorithm = None
    CppLabelingConfig = None


class PerSourcePricing(PricingProblem):
    """
    Per-source-arc pricing with isolated networks for comprehensive coverage.

    Instead of running one labeling that favors certain source arcs,
    this builds a separate network for each source arc (containing only
    that source arc) and runs labeling on each. This ensures all parts
    of the network get explored.

    Parameters:
        problem: The Problem instance
        config: Pricing configuration
        max_labels_per_node: Beam search limit
        use_topological_order: Use DAG-optimized processing
        cols_per_source: Max columns to find per source arc
        time_per_source: Time limit per source arc (seconds)
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 20,
        use_topological_order: bool = True,
        cols_per_source: int = 3,
        time_per_source: float = 0.05,
    ):
        if not HAS_CPP_BACKEND:
            raise ImportError("C++ backend not available")

        super().__init__(problem, config)

        self._max_labels_per_node = max_labels_per_node
        self._use_topological_order = use_topological_order
        self._cols_per_source = cols_per_source
        self._time_per_source = time_per_source

        # Get source arcs with their bases
        self._source_arcs: list[tuple] = []  # (arc_index, base)
        for arc in problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base:
                    self._source_arcs.append((arc.index, base))

        # Get sink arcs by base
        self._sink_arcs_by_base: dict[str, set[int]] = defaultdict(set)
        for arc in problem.network.arcs:
            if arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base')
                if base:
                    self._sink_arcs_by_base[base].add(arc.index)

        # Get numeric resources
        self._numeric_resources = []
        self._resource_limits = []
        for r in problem.resources:
            if hasattr(r, 'max_value'):
                self._numeric_resources.append(r.name)
                self._resource_limits.append(r.max_value)

    def _on_duals_updated(self) -> None:
        """No pre-update needed; duals are set per solve."""
        pass

    def _solve_impl(self) -> PricingSolution:
        """Run per-source-arc pricing with isolated networks."""
        start_time = time.time()

        all_columns: list[Column] = []
        covered_items: set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        network = self._problem.network

        # Process each source arc with its own isolated network
        for source_arc_idx, base in self._source_arcs:
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            # Build isolated network for this source arc
            cpp_network = CppNetwork()
            cpp_arc_to_py: dict[int, int] = {}

            # Add all nodes
            for i in range(network.num_nodes):
                node = network.get_node(i)
                if node.node_type == NodeType.SOURCE:
                    cpp_network.add_source()
                elif node.node_type == NodeType.SINK:
                    cpp_network.add_sink()
                else:
                    cpp_network.add_node()

            # Add arcs - ONLY this source arc, matching sink arcs, and other arcs
            cpp_idx = 0
            for arc in network.arcs:
                # Skip other source arcs
                if arc.arc_type == ArcType.SOURCE_ARC:
                    if arc.index != source_arc_idx:
                        continue

                # Skip sink arcs from other bases
                if arc.arc_type == ArcType.SINK_ARC:
                    if arc.index not in self._sink_arcs_by_base[base]:
                        continue

                res_consumption = [
                    arc.get_consumption(r, 0.0) for r in self._numeric_resources
                ]
                covered = [arc.index] if arc.arc_type == ArcType.FLIGHT else []

                cpp_network.add_arc(
                    arc.source, arc.target, arc.cost,
                    res_consumption, covered
                )
                cpp_arc_to_py[cpp_idx] = arc.index
                cpp_idx += 1

            # Create labeling config
            cpp_config = CppLabelingConfig()
            cpp_config.max_columns = self._cols_per_source
            cpp_config.max_time = self._time_per_source
            cpp_config.rc_threshold = self._config.reduced_cost_threshold
            cpp_config.check_dominance = self._config.use_dominance
            cpp_config.check_elementarity = self._config.check_elementarity
            cpp_config.use_topological_order = self._use_topological_order
            cpp_config.max_labels_per_node = self._max_labels_per_node

            # Run labeling
            algo = CppLabelingAlgorithm(
                cpp_network,
                len(self._numeric_resources),
                self._resource_limits,
                cpp_config
            )
            algo.set_dual_values(self._dual_values)
            result = algo.solve()

            total_labels += result.labels_created
            total_dominated += result.labels_dominated

            # Process columns
            for cpp_label in result.columns:
                col = self._convert_cpp_label(cpp_label, cpp_arc_to_py, base)
                if col is None:
                    continue

                if col.reduced_cost < (self._config.reduced_cost_threshold or -1e-6):
                    # Check if column covers new items
                    new_items = col.covered_items - covered_items
                    if new_items:
                        all_columns.append(col)
                        covered_items.update(col.covered_items)
                        if best_rc is None or col.reduced_cost < best_rc:
                            best_rc = col.reduced_cost

            # NOTE: We do NOT check column limit here anymore!
            # We must process all source arcs to ensure all flights have a chance
            # to be covered. The column limit is applied at the end.

        # Sort by reduced cost and limit to max_columns
        # This keeps the BEST columns, not just the first ones found
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
        arc_map: dict[int, int],
        base: str
    ) -> Optional[Column]:
        """Convert C++ label to Python Column."""
        try:
            cpp_arcs = cpp_label.get_arc_indices()
            py_arcs = [arc_map.get(i, -1) for i in cpp_arcs]
            py_arcs = [a for a in py_arcs if a >= 0]

            if not py_arcs:
                return None

            items = cpp_label.covered_items
            if callable(items):
                items = items()

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
