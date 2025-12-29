"""
Per-source-arc pricing using Boost r_c_shortest_paths.

This module combines the per-source-arc isolation strategy with Boost's
highly optimized SPPRC implementation. For each source arc, we build
an isolated network and run Boost's label-setting algorithm.

Strategy:
1. For each source arc, build a restricted network containing ONLY that arc
   (plus matching sink arcs for the same base)
2. Run Boost's r_c_shortest_paths on each isolated network
3. Combine results, selecting columns that cover new items

Advantages over PerSourcePricing:
- Boost's r_c_shortest_paths is highly optimized C++
- Returns all Pareto-optimal paths (not just beam search)
- Better dominance handling
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

# Try to import Boost SPPRC solver
try:
    from opencg._core import (
        HAS_BOOST,
        HAS_CPP_BACKEND,
        BoostSPPRCSolver,
    )
except ImportError:
    HAS_CPP_BACKEND = False
    HAS_BOOST = False
    BoostSPPRCSolver = None


class PerSourceBoostPricing(PricingProblem):
    """
    Per-source-arc pricing using Boost's r_c_shortest_paths.

    This builds a separate network for each source arc (containing only
    that source arc and matching sink arcs) and runs Boost's optimized
    SPPRC on each. This ensures comprehensive coverage while leveraging
    Boost's high-performance implementation.

    Parameters:
        problem: The Problem instance
        config: Pricing configuration
        cols_per_source: Max columns to find per source arc
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        cols_per_source: int = 3,
    ):
        if not HAS_BOOST or BoostSPPRCSolver is None:
            raise ImportError(
                "Boost SPPRC solver not available. "
                "Install Boost 1.70+ and rebuild with: pip install -e ."
            )

        super().__init__(problem, config)

        self._cols_per_source = cols_per_source

        # Find source/sink indices
        self._source_idx: Optional[int] = None
        self._sink_idx: Optional[int] = None
        self._find_source_sink()

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

        print(f"PerSourceBoostPricing: {len(self._source_arcs)} source arcs, "
              f"{len(self._numeric_resources)} resources")

    def _find_source_sink(self) -> None:
        for i in range(self._problem.network.num_nodes):
            node = self._problem.network.get_node(i)
            if node is None:
                continue
            if node.node_type == NodeType.SOURCE:
                self._source_idx = i
            elif node.node_type == NodeType.SINK:
                self._sink_idx = i

    def _on_duals_updated(self) -> None:
        """No pre-update needed; solvers rebuilt per solve."""
        pass

    def _solve_impl(self) -> PricingSolution:
        """Run Boost SPPRC for each source arc."""
        start_time = time.time()

        all_columns: list[Column] = []
        covered_items: set[int] = set()
        best_rc = None
        total_labels = 0

        py_network = self._problem.network

        for source_arc_idx, base in self._source_arcs:
            # Check time limit
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            # Check column limit
            if self._config.max_columns > 0 and len(all_columns) >= self._config.max_columns:
                break

            # Build solver for this source arc
            solver = BoostSPPRCSolver(
                num_resources=len(self._numeric_resources),
                resource_limits=self._resource_limits,
                check_elementarity=self._config.check_elementarity
            )

            # Add all nodes
            for i in range(py_network.num_nodes):
                solver.add_vertex(i)

            solver.set_source(self._source_idx)
            solver.set_sink(self._sink_idx)

            # Add arcs (filtering to this source arc and matching sink arcs)
            for arc in py_network.arcs:
                # Only include this specific source arc
                if arc.arc_type == ArcType.SOURCE_ARC:
                    if arc.index != source_arc_idx:
                        continue

                # Only include sink arcs for this base
                if arc.arc_type == ArcType.SINK_ARC:
                    if arc.index not in self._sink_arcs_by_base[base]:
                        continue

                # Get resource consumption
                res_consumption = [
                    arc.get_consumption(r, 0.0) for r in self._numeric_resources
                ]

                # Get covered items
                if arc.arc_type == ArcType.FLIGHT:
                    covered = [arc.index]
                else:
                    covered = []

                # Compute reduced cost
                reduced_cost = arc.cost
                for item in covered:
                    if item in self._dual_values:
                        reduced_cost -= self._dual_values[item]

                solver.add_arc(
                    arc.source, arc.target, arc.cost, reduced_cost,
                    res_consumption, covered, arc.index
                )

            # Solve for this source arc
            result = solver.solve(
                max_paths=self._cols_per_source,
                rc_threshold=self._config.reduced_cost_threshold
            )
            total_labels += result.num_labels

            # Convert paths to columns
            for path in result.paths:
                col = self._convert_path_to_column(path, base)
                if col is None:
                    continue

                # Check RC threshold
                if col.reduced_cost >= (self._config.reduced_cost_threshold or -1e-6):
                    continue

                # Check if column covers new items
                new_items = col.covered_items - covered_items
                if new_items:
                    all_columns.append(col)
                    covered_items.update(col.covered_items)
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
            num_labels_dominated=0,
            solve_time=solve_time,
            iterations=len(all_columns),
        )

    def _convert_path_to_column(self, path, base: str) -> Optional[Column]:
        """Convert Boost path to Python Column."""
        try:
            py_arcs = list(path.arc_indices)
            if not py_arcs:
                return None

            return Column(
                arc_indices=tuple(py_arcs),
                cost=path.cost,
                reduced_cost=path.reduced_cost,
                covered_items=frozenset(path.covered_items),
                attributes={'base': base},
            )
        except Exception:
            return None
