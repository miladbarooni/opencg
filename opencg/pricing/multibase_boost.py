"""
Multi-base pricing using Boost r_c_shortest_paths.

This module provides a pricing algorithm that:
1. Runs separate Boost SPPRC per crew base
2. Filters source/sink arcs to enforce same-base constraint
3. Combines results from all bases

Uses Boost's high-performance label-setting algorithm.
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

# Try to import Boost SPPRC solver
try:
    from opencg._core import HAS_CPP_BACKEND, BoostSPPRCSolver
except ImportError:
    HAS_CPP_BACKEND = False
    BoostSPPRCSolver = None


class MultiBaseBoostPricing(PricingProblem):
    """
    Multi-base pricing using Boost r_c_shortest_paths.

    Runs separate pricing for each crew base, filtering the network
    to only include source/sink arcs for that base. This enforces
    the same-base constraint without needing HomeBaseResource.

    Performance advantages:
    - Boost's r_c_shortest_paths is highly optimized
    - Per-base filtering reduces search space
    - Returns all Pareto-optimal paths
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None
    ):
        if not HAS_CPP_BACKEND or BoostSPPRCSolver is None:
            raise ImportError("Boost SPPRC solver not available")

        super().__init__(problem, config)

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

        # Build per-base Boost solvers
        self._base_solvers: Dict[str, BoostSPPRCSolver] = {}
        self._base_arc_maps: Dict[str, Dict[int, int]] = {}  # boost_arc -> py_arc

        for base in self._bases:
            self._build_base_solver(base)

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

    def _build_base_solver(self, base: str) -> None:
        """Build Boost SPPRC solver for a specific base."""
        py_network = self._problem.network

        # Create solver
        solver = BoostSPPRCSolver(
            num_resources=len(self._numeric_resources),
            resource_limits=self._resource_limits,
            check_elementarity=self._config.check_elementarity
        )

        # Track vertex mapping (py -> boost)
        py_to_boost_node: Dict[int, int] = {}

        # Add all nodes
        for i in range(py_network.num_nodes):
            node = py_network.get_node(i)
            solver.add_vertex(i)
            py_to_boost_node[i] = i

        solver.set_source(self._source_idx)
        solver.set_sink(self._sink_idx)

        # Add arcs (filtering source/sink by base)
        boost_arc_to_py: Dict[int, int] = {}
        boost_arc_idx = 0

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

            # Get covered items (only flight arcs cover items)
            if arc.arc_type == ArcType.FLIGHT:
                covered_items = [arc.index]
            else:
                covered_items = []

            # Compute reduced cost (will be updated with duals in _on_duals_updated)
            reduced_cost = arc.cost

            solver.add_arc(
                arc.source, arc.target, arc.cost, reduced_cost,
                res_consumption, covered_items, arc.index
            )
            boost_arc_to_py[boost_arc_idx] = arc.index
            boost_arc_idx += 1

        self._base_solvers[base] = solver
        self._base_arc_maps[base] = boost_arc_to_py

    def _on_duals_updated(self) -> None:
        """Rebuild solvers with updated reduced costs."""
        # Need to rebuild networks with new reduced costs
        for base in self._bases:
            self._rebuild_solver_with_duals(base)

    def _rebuild_solver_with_duals(self, base: str) -> None:
        """Rebuild solver for a base with current dual values."""
        py_network = self._problem.network

        # Create new solver
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

        # Add arcs with updated reduced costs
        boost_arc_to_py: Dict[int, int] = {}
        boost_arc_idx = 0

        for arc in py_network.arcs:
            # Filter source arcs
            if arc.arc_type == ArcType.SOURCE_ARC:
                if arc.index not in self._base_source_arcs[base]:
                    continue

            # Filter sink arcs
            if arc.arc_type == ArcType.SINK_ARC:
                if arc.index not in self._base_sink_arcs[base]:
                    continue

            # Get resource consumption
            res_consumption = []
            for res_name in self._numeric_resources:
                val = arc.get_consumption(res_name, 0.0)
                res_consumption.append(val)

            # Get covered items
            if arc.arc_type == ArcType.FLIGHT:
                covered_items = [arc.index]
            else:
                covered_items = []

            # Compute reduced cost: cost - sum(duals for covered items)
            reduced_cost = arc.cost
            for item in covered_items:
                if item in self._dual_values:
                    reduced_cost -= self._dual_values[item]

            solver.add_arc(
                arc.source, arc.target, arc.cost, reduced_cost,
                res_consumption, covered_items, arc.index
            )
            boost_arc_to_py[boost_arc_idx] = arc.index
            boost_arc_idx += 1

        self._base_solvers[base] = solver
        self._base_arc_maps[base] = boost_arc_to_py

    def _solve_impl(self) -> PricingSolution:
        """Run pricing for each base and combine results."""
        start_time = time.time()

        all_columns: List[Column] = []
        best_rc = None
        total_labels = 0

        for base in self._bases:
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            if base not in self._base_solvers:
                continue

            solver = self._base_solvers[base]
            arc_map = self._base_arc_maps[base]

            # Calculate max paths for this base
            cols_per_base = self._config.max_columns // len(self._bases) if self._config.max_columns > 0 else 0

            # Solve for this base
            result = solver.solve(
                max_paths=cols_per_base if cols_per_base > 0 else 0,
                rc_threshold=self._config.reduced_cost_threshold
            )
            total_labels += result.num_labels

            # Convert paths to columns
            for path in result.paths:
                column = self._convert_path_to_column(path, arc_map, base)
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
            num_labels_dominated=0,
            solve_time=solve_time,
            iterations=len(all_columns),
        )

    def _convert_path_to_column(
        self,
        path,
        arc_map: Dict[int, int],
        base: str
    ) -> Optional[Column]:
        """Convert Boost path to Python Column."""
        try:
            # Get arc indices from path
            py_arcs = []
            for boost_idx, py_idx in arc_map.items():
                if py_idx in path.arc_indices:
                    py_arcs.append(py_idx)

            # Actually, arc_indices in path are the original py arc indices
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
