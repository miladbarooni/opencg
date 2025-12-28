"""
Targeted pricing for uncovered flights.

This module provides a pricing algorithm that specifically targets uncovered
flights by running separate pricing passes for each, forcing the labeling
algorithm to include that specific flight arc in the path.

Strategy:
1. First run normal pricing to find columns with best RC
2. Identify uncovered flights (those not covered by any found column)
3. For each uncovered flight, run a targeted pricing that forces
   paths through that flight's arc
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


class TargetedPricing(PricingProblem):
    """
    Targeted pricing that specifically finds columns for uncovered flights.

    This algorithm:
    1. Runs normal pricing first to get initial columns
    2. For uncovered flights, builds sub-networks that force paths through them
    3. Combines results for maximum coverage

    Parameters:
        problem: The Problem instance
        config: Pricing configuration
        max_labels_per_node: Beam search limit
        use_topological_order: Use DAG-optimized processing
        max_targeted_flights: Max number of uncovered flights to target per iteration
        time_per_target: Time limit per targeted flight (seconds)
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 50,
        use_topological_order: bool = True,
        max_targeted_flights: int = 100,
        time_per_target: float = 0.5,
    ):
        if not HAS_CPP_BACKEND:
            raise ImportError("C++ backend not available")

        super().__init__(problem, config)

        self._max_labels_per_node = max_labels_per_node
        self._use_topological_order = use_topological_order
        self._max_targeted_flights = max_targeted_flights
        self._time_per_target = time_per_target

        # Find source/sink
        self._source_idx: Optional[int] = None
        self._sink_idx: Optional[int] = None
        self._find_source_sink()

        # Find bases and their arcs
        self._bases = self._find_bases()
        self._base_source_arcs = self._find_base_source_arcs()
        self._base_sink_arcs = self._find_base_sink_arcs()

        # Get numeric resources
        self._numeric_resources = []
        self._resource_limits = []
        for r in problem.resources:
            if hasattr(r, 'max_value'):
                self._numeric_resources.append(r.name)
                self._resource_limits.append(r.max_value)

        # Build full network (for normal pricing)
        self._full_network: Optional[CppNetwork] = None
        self._full_arc_map: dict[int, int] = {}
        self._build_full_network()

        # Per-base networks for targeted pricing
        self._base_networks: dict[str, CppNetwork] = {}
        self._base_arc_maps: dict[str, dict[int, int]] = {}
        for base in self._bases:
            self._build_base_network(base)

        # Map from flight arc index to which bases can cover it
        self._flight_to_bases = self._compute_flight_bases()

        # Track uncovered flights from previous iteration
        self._uncovered_flights: set[int] = set()

    def _find_source_sink(self) -> None:
        for i in range(self._problem.network.num_nodes):
            node = self._problem.network.get_node(i)
            if node is None:
                continue
            if node.node_type == NodeType.SOURCE:
                self._source_idx = i
            elif node.node_type == NodeType.SINK:
                self._sink_idx = i

    def _find_bases(self) -> list[str]:
        bases = set()
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base:
                    bases.add(base)
        return sorted(bases)

    def _find_base_source_arcs(self) -> dict[str, set[int]]:
        result = {base: set() for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base in result:
                    result[base].add(arc.index)
        return result

    def _find_base_sink_arcs(self) -> dict[str, set[int]]:
        result = {base: set() for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base')
                if base in result:
                    result[base].add(arc.index)
        return result

    def _build_full_network(self) -> None:
        """Build full C++ network (no base filtering)."""
        py_network = self._problem.network
        cpp_network = CppNetwork()
        cpp_arc_to_py: dict[int, int] = {}

        # Add all nodes
        for i in range(py_network.num_nodes):
            node = py_network.get_node(i)
            if node.node_type == NodeType.SOURCE:
                cpp_network.add_source()
            elif node.node_type == NodeType.SINK:
                cpp_network.add_sink()
            else:
                cpp_network.add_node()

        # Add all arcs
        cpp_arc_idx = 0
        for arc in py_network.arcs:
            res_consumption = []
            for res_name in self._numeric_resources:
                val = arc.get_consumption(res_name, 0.0)
                res_consumption.append(val)

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

        self._full_network = cpp_network
        self._full_arc_map = cpp_arc_to_py

    def _build_base_network(self, base: str) -> None:
        """Build C++ network for a specific base."""
        py_network = self._problem.network
        cpp_network = CppNetwork()
        cpp_arc_to_py: dict[int, int] = {}

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
            if arc.arc_type == ArcType.SOURCE_ARC:
                if arc.index not in self._base_source_arcs[base]:
                    continue
            if arc.arc_type == ArcType.SINK_ARC:
                if arc.index not in self._base_sink_arcs[base]:
                    continue

            res_consumption = []
            for res_name in self._numeric_resources:
                val = arc.get_consumption(res_name, 0.0)
                res_consumption.append(val)

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

        self._base_networks[base] = cpp_network
        self._base_arc_maps[base] = cpp_arc_to_py

    def _compute_flight_bases(self) -> dict[int, list[str]]:
        """For each flight arc, determine which bases can cover it."""
        result = defaultdict(list)

        for base in self._bases:
            # BFS from base's source arcs to find reachable flight arcs
            # and which can reach base's sink arcs
            arc_map = self._base_arc_maps[base]
            # The arc_map contains cpp_idx -> py_idx
            # If a flight arc is in this map, the base can reach it
            for cpp_idx, py_idx in arc_map.items():
                arc = self._problem.network.get_arc(py_idx)
                if arc and arc.arc_type == ArcType.FLIGHT:
                    result[py_idx].append(base)

        return dict(result)

    def set_uncovered_flights(self, uncovered: set[int]) -> None:
        """Set the flights that need targeted pricing."""
        self._uncovered_flights = set(uncovered)

    def _on_duals_updated(self) -> None:
        """No pre-update needed; duals are set per solve."""
        pass

    def _solve_impl(self) -> PricingSolution:
        """Run targeted pricing."""
        start_time = time.time()

        all_columns: list[Column] = []
        covered_items: set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        # Phase 1: Run normal pricing per base
        for base in self._bases:
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time * 0.5:  # Use 50% for normal pricing
                    break

            cols, labels, dominated = self._run_base_pricing(base)
            total_labels += labels
            total_dominated += dominated

            for col in cols:
                if col.reduced_cost < (self._config.reduced_cost_threshold or -1e-6):
                    all_columns.append(col)
                    covered_items.update(col.covered_items)
                    if best_rc is None or col.reduced_cost < best_rc:
                        best_rc = col.reduced_cost

        # Phase 2: Targeted pricing for uncovered flights
        all_flights = {i for i, _ in enumerate(self._problem.cover_constraints)}
        uncovered = (all_flights - covered_items) | self._uncovered_flights

        # Sort by dual value (highest first - most needed)
        uncovered_by_priority = sorted(
            uncovered,
            key=lambda f: -self._dual_values.get(f, 0.0)
        )[:self._max_targeted_flights]

        for flight_idx in uncovered_by_priority:
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            # Find a column covering this specific flight
            col = self._find_column_for_flight(flight_idx)
            if col is not None:
                all_columns.append(col)
                covered_items.update(col.covered_items)
                total_labels += 1  # Approximate
                if best_rc is None or col.reduced_cost < best_rc:
                    best_rc = col.reduced_cost

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

    def _run_base_pricing(self, base: str) -> tuple[list[Column], int, int]:
        """Run normal pricing for a base."""
        if base not in self._base_networks:
            return [], 0, 0

        cpp_network = self._base_networks[base]
        arc_map = self._base_arc_maps[base]

        # Create algorithm
        cpp_config = CppLabelingConfig()
        cols_per_base = (self._config.max_columns // len(self._bases)) if self._config.max_columns > 0 else 100
        cpp_config.max_columns = cols_per_base
        cpp_config.max_time = self._config.max_time / len(self._bases) if self._config.max_time > 0 else 5.0
        cpp_config.rc_threshold = self._config.reduced_cost_threshold
        cpp_config.check_dominance = self._config.use_dominance
        cpp_config.check_elementarity = self._config.check_elementarity
        cpp_config.use_topological_order = self._use_topological_order
        cpp_config.max_labels_per_node = self._max_labels_per_node

        algo = CppLabelingAlgorithm(
            cpp_network,
            len(self._numeric_resources),
            self._resource_limits,
            cpp_config
        )
        algo.set_dual_values(self._dual_values)

        result = algo.solve()

        columns = []
        for cpp_label in result.columns:
            col = self._convert_cpp_label(cpp_label, arc_map, base)
            if col is not None:
                columns.append(col)

        return columns, result.labels_created, result.labels_dominated

    def _find_column_for_flight(self, flight_idx: int) -> Optional[Column]:
        """
        Find a column that covers a specific flight.

        Uses a modified dual structure where:
        - The target flight has a large positive dual (incentive to cover)
        - Other flights have normal duals
        """
        # Get bases that can cover this flight
        bases = self._flight_to_bases.get(flight_idx, [])
        if not bases:
            return None

        # Create modified duals: boost this flight's dual significantly
        modified_duals = dict(self._dual_values)
        original_dual = modified_duals.get(flight_idx, 0.0)
        # Set a high dual to make covering this flight very attractive
        modified_duals[flight_idx] = max(original_dual, 1000.0)

        # Try each base
        for base in bases:
            if base not in self._base_networks:
                continue

            cpp_network = self._base_networks[base]
            arc_map = self._base_arc_maps[base]

            # Create algorithm with smaller limits for speed
            cpp_config = CppLabelingConfig()
            cpp_config.max_columns = 5
            cpp_config.max_time = self._time_per_target
            cpp_config.rc_threshold = 1e10  # Accept any RC for targeted search
            cpp_config.check_dominance = True
            cpp_config.check_elementarity = self._config.check_elementarity
            cpp_config.use_topological_order = self._use_topological_order
            cpp_config.max_labels_per_node = self._max_labels_per_node

            algo = CppLabelingAlgorithm(
                cpp_network,
                len(self._numeric_resources),
                self._resource_limits,
                cpp_config
            )
            algo.set_dual_values(modified_duals)

            result = algo.solve()

            # Find a column that covers the target flight
            for cpp_label in result.columns:
                col = self._convert_cpp_label(cpp_label, arc_map, base)
                if col is not None and flight_idx in col.covered_items:
                    # Recompute reduced cost with original duals
                    true_rc = col.cost - sum(
                        self._dual_values.get(item, 0.0)
                        for item in col.covered_items
                    )
                    return Column(
                        arc_indices=col.arc_indices,
                        cost=col.cost,
                        reduced_cost=true_rc,
                        covered_items=col.covered_items,
                        attributes=col.attributes,
                    )

        return None

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
