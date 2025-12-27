"""
Diversity-promoting multi-base pricing using C++ labeling.

This module provides pricing that actively avoids generating columns
that cover the same flights repeatedly. It uses several strategies:

1. **Covered item penalties**: Items already covered in this iteration
   get reduced dual values to discourage re-covering them.

2. **Multiple passes**: Run pricing multiple times, each time penalizing
   previously covered items to find diverse columns.

3. **Greedy filtering**: After generating columns, filter to maximize
   unique item coverage rather than minimize reduced cost.
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


class DiverseMultiBasePricing(PricingProblem):
    """
    Diversity-promoting multi-base pricing using C++ labeling algorithm.

    This pricing algorithm generates columns that cover different items
    by using multiple strategies:

    1. Penalty-based diversity: After finding columns, penalize covered
       items and search again to find columns for uncovered items.

    2. Greedy column selection: Instead of taking columns with best RC,
       select columns that maximize unique coverage.

    3. Per-base diversity: Ensure each base contributes columns covering
       different items.

    Parameters:
        problem: The Problem instance
        config: Pricing configuration
        max_labels_per_node: Beam search limit (0 = unlimited)
        use_topological_order: Use DAG-optimized processing
        num_diversity_passes: Number of passes with penalties (default 3)
        coverage_penalty: Penalty multiplier for covered items (default 0.5)
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 100,
        use_topological_order: bool = True,
        num_diversity_passes: int = 3,
        coverage_penalty: float = 0.5,
    ):
        if not HAS_CPP_BACKEND:
            raise ImportError("C++ backend not available")

        super().__init__(problem, config)

        self._max_labels_per_node = max_labels_per_node
        self._use_topological_order = use_topological_order
        self._num_diversity_passes = num_diversity_passes
        self._coverage_penalty = coverage_penalty

        # Find source/sink indices
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

        # Build per-base C++ networks
        self._base_networks: Dict[str, CppNetwork] = {}
        self._base_algorithms: Dict[str, CppLabelingAlgorithm] = {}
        self._base_arc_maps: Dict[str, Dict[int, int]] = {}

        for base in self._bases:
            self._build_base_network(base)

        # Track items covered in current CG iteration
        self._items_covered_this_iteration: Set[int] = set()

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

        # Create config
        cpp_config = CppLabelingConfig()
        cols_per_base = self._config.max_columns // len(self._bases) if self._config.max_columns > 0 else 0
        cpp_config.max_columns = cols_per_base if cols_per_base > 0 else 0
        time_per_base = self._config.max_time / len(self._bases) if self._config.max_time > 0 else 0.0
        cpp_config.max_time = time_per_base if time_per_base > 0 else 0.0
        cpp_config.max_labels = self._config.max_labels if self._config.max_labels > 0 else 0
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

        self._base_networks[base] = cpp_network
        self._base_algorithms[base] = algo
        self._base_arc_maps[base] = cpp_arc_to_py

    def reset_iteration_coverage(self) -> None:
        """Reset the coverage tracking for a new CG iteration."""
        self._items_covered_this_iteration.clear()

    def _on_duals_updated(self) -> None:
        """Update dual values in all base algorithms."""
        for base in self._bases:
            if base in self._base_algorithms:
                self._base_algorithms[base].set_dual_values(self._dual_values)

    def _solve_impl(self) -> PricingSolution:
        """Run diversity-promoting pricing."""
        start_time = time.time()

        all_columns: List[Column] = []
        all_covered: Set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        # Multiple passes with increasing penalties on covered items
        for pass_idx in range(self._num_diversity_passes):
            # Modify duals to penalize already-covered items
            modified_duals = self._apply_coverage_penalty(pass_idx)

            pass_columns = []

            for base in self._bases:
                if self._config.max_time > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= self._config.max_time:
                        break

                if base not in self._base_algorithms:
                    continue

                algo = self._base_algorithms[base]
                arc_map = self._base_arc_maps[base]

                # Update duals for this pass
                algo.set_dual_values(modified_duals)

                # Solve
                result = algo.solve()
                total_labels += result.labels_created
                total_dominated += result.labels_dominated

                # Convert columns
                for cpp_label in result.columns:
                    column = self._convert_cpp_label(cpp_label, arc_map, base)
                    if column is not None:
                        # Check if this column covers any new items
                        new_items = column.covered_items - all_covered
                        if new_items:  # Only keep if covers new items
                            pass_columns.append(column)

            # Greedy selection: prefer columns covering more new items
            pass_columns = self._greedy_select(pass_columns, all_covered)

            for col in pass_columns:
                all_columns.append(col)
                all_covered.update(col.covered_items)
                if best_rc is None or col.reduced_cost < best_rc:
                    best_rc = col.reduced_cost

            # Check if we found enough columns
            if self._config.max_columns > 0 and len(all_columns) >= self._config.max_columns:
                break

        # Update iteration coverage tracking
        self._items_covered_this_iteration.update(all_covered)

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

    def _apply_coverage_penalty(self, pass_idx: int) -> Dict[int, float]:
        """Apply penalties to duals for covered items."""
        modified = dict(self._dual_values)

        if pass_idx == 0:
            # First pass: use original duals
            return modified

        # Subsequent passes: zero out duals for covered items
        # This forces pricing to find columns for uncovered items only
        for item_id in self._items_covered_this_iteration:
            if item_id in modified:
                modified[item_id] = 0.0

        return modified

    def _greedy_select(
        self,
        columns: List[Column],
        already_covered: Set[int]
    ) -> List[Column]:
        """
        Greedy selection of columns to maximize unique coverage.

        Instead of just taking columns with best RC, we select columns
        that cover the most NEW items (not already covered).
        """
        if not columns:
            return []

        selected = []
        current_covered = set(already_covered)

        # Sort by number of new items covered (descending), then by RC (ascending)
        def score(col):
            new_items = len(col.covered_items - current_covered)
            return (-new_items, col.reduced_cost)

        remaining = list(columns)

        while remaining:
            # Re-sort based on current coverage
            remaining.sort(key=score)

            # Take the best column
            best = remaining.pop(0)
            new_items = best.covered_items - current_covered

            if not new_items:
                # No new items, stop
                break

            selected.append(best)
            current_covered.update(best.covered_items)

            # Limit number of columns
            if self._config.max_columns > 0 and len(selected) >= self._config.max_columns:
                break

        return selected

    def _convert_cpp_label(
        self,
        cpp_label,
        arc_map: Dict[int, int],
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
