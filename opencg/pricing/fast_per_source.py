"""
Fast per-source-arc pricing with prebuilt networks.

This is an optimized version of PerSourcePricing that prebuilds all
per-source networks at initialization, avoiding the overhead of building
509 networks on every solve() call.

Strategy:
1. At initialization, build one C++ network per source arc (precomputed)
2. During solve, just update dual values and run labeling
3. This gives the same high coverage as PerSourcePricing but much faster

Parallel execution:
- Set num_threads > 1 to enable parallel pricing across source arcs
- Uses concurrent.futures.ThreadPoolExecutor
- C++ labeling releases GIL, so threads run in true parallel
"""

import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

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


class FastPerSourcePricing(PricingProblem):
    """
    Fast per-source-arc pricing with prebuilt networks.

    This builds all per-source networks once at initialization, then
    only updates duals and runs labeling during solve(). This provides
    the same comprehensive coverage as PerSourcePricing but with much
    lower overhead.

    Supports parallel execution across source arcs using ThreadPoolExecutor.
    The C++ labeling algorithm releases the GIL, allowing true parallel
    execution when num_threads > 1.

    Parameters:
        problem: The Problem instance
        config: Pricing configuration
        max_labels_per_node: Beam search limit per node
        cols_per_source: Max columns per source arc
        time_per_source: Time limit per source arc (seconds)
        num_threads: Number of threads for parallel pricing (default=1, 0=auto)
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 20,
        cols_per_source: int = 3,
        time_per_source: float = 0.05,
        num_threads: int = 1,
    ):
        if not HAS_CPP_BACKEND:
            raise ImportError("C++ backend not available")

        super().__init__(problem, config)

        self._max_labels_per_node = max_labels_per_node
        self._cols_per_source = cols_per_source
        self._time_per_source = time_per_source
        self._num_threads = num_threads if num_threads > 0 else os.cpu_count() or 1

        # Get source arcs with their bases
        self._source_arcs: List[Tuple[int, str]] = []
        for arc in problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base:
                    self._source_arcs.append((arc.index, base))

        # Get sink arcs by base
        self._sink_arcs_by_base: Dict[str, Set[int]] = defaultdict(set)
        for arc in problem.network.arcs:
            if arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base')
                if base:
                    self._sink_arcs_by_base[base].add(arc.index)

        # Get numeric resources
        self._numeric_resources: List[str] = []
        self._resource_limits: List[float] = []
        for r in problem.resources:
            if hasattr(r, 'max_value'):
                self._numeric_resources.append(r.name)
                self._resource_limits.append(r.max_value)

        # Prebuild all per-source networks and algorithms
        self._source_networks: Dict[int, CppNetwork] = {}
        self._source_algorithms: Dict[int, CppLabelingAlgorithm] = {}
        self._source_arc_maps: Dict[int, Dict[int, int]] = {}

        print(f"FastPerSourcePricing: prebuilding {len(self._source_arcs)} networks...")
        build_start = time.time()
        for source_arc_idx, base in self._source_arcs:
            self._build_source_network(source_arc_idx, base)
        print(f"  Prebuilt in {time.time() - build_start:.2f}s")

    def _build_source_network(self, source_arc_idx: int, base: str) -> None:
        """Build isolated C++ network for a single source arc."""
        network = self._problem.network

        cpp_network = CppNetwork()
        cpp_arc_to_py: Dict[int, int] = {}

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
        cpp_config.use_topological_order = True
        cpp_config.max_labels_per_node = self._max_labels_per_node

        # Create algorithm
        algo = CppLabelingAlgorithm(
            cpp_network,
            len(self._numeric_resources),
            self._resource_limits,
            cpp_config
        )

        self._source_networks[source_arc_idx] = cpp_network
        self._source_algorithms[source_arc_idx] = algo
        self._source_arc_maps[source_arc_idx] = cpp_arc_to_py

    def _on_duals_updated(self) -> None:
        """Update dual values in all source algorithms."""
        for source_arc_idx, _ in self._source_arcs:
            if source_arc_idx in self._source_algorithms:
                self._source_algorithms[source_arc_idx].set_dual_values(self._dual_values)

    def _solve_impl(self) -> PricingSolution:
        """Run labeling on all prebuilt source networks."""
        if self._num_threads > 1:
            return self._solve_parallel()
        else:
            return self._solve_sequential()

    def _solve_single_source(
        self, source_arc_idx: int, base: str
    ) -> Tuple[List[Column], int, int]:
        """
        Solve labeling for a single source arc.

        Returns:
            (columns, labels_created, labels_dominated)
        """
        algo = self._source_algorithms.get(source_arc_idx)
        if algo is None:
            return [], 0, 0

        arc_map = self._source_arc_maps[source_arc_idx]

        # Run labeling (C++ releases GIL during solve)
        result = algo.solve()

        # Convert columns
        columns = []
        for cpp_label in result.columns:
            col = self._convert_cpp_label(cpp_label, arc_map, base)
            if col is not None and col.reduced_cost < (self._config.reduced_cost_threshold or -1e-6):
                columns.append(col)

        return columns, result.labels_created, result.labels_dominated

    def _solve_parallel(self) -> PricingSolution:
        """Run labeling on all sources in parallel using ThreadPoolExecutor.

        Uses batched submission to allow early termination when column limit is reached.
        """
        start_time = time.time()

        all_columns: List[Column] = []
        covered_items: Set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        # Process in batches to allow early termination
        batch_size = self._num_threads * 4  # Process 4 batches worth at a time
        source_arcs = list(self._source_arcs)

        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            idx = 0
            while idx < len(source_arcs):
                # Check if we've reached column limit
                if self._config.max_columns > 0 and len(all_columns) >= self._config.max_columns:
                    break

                # Check time limit
                if self._config.max_time > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= self._config.max_time:
                        break

                # Submit a batch of tasks
                batch_end = min(idx + batch_size, len(source_arcs))
                batch = source_arcs[idx:batch_end]

                futures = {
                    executor.submit(self._solve_single_source, source_arc_idx, base): (source_arc_idx, base)
                    for source_arc_idx, base in batch
                }

                # Collect batch results
                for future in as_completed(futures):
                    try:
                        columns, labels_created, labels_dominated = future.result()
                        total_labels += labels_created
                        total_dominated += labels_dominated

                        # Add columns that cover new items
                        for col in columns:
                            new_items = col.covered_items - covered_items
                            if new_items:
                                all_columns.append(col)
                                covered_items.update(col.covered_items)
                                if best_rc is None or col.reduced_cost < best_rc:
                                    best_rc = col.reduced_cost

                    except Exception:
                        # Skip failed sources
                        pass

                idx = batch_end

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

    def _solve_sequential(self) -> PricingSolution:
        """Run labeling on all prebuilt source networks sequentially."""
        start_time = time.time()

        all_columns: List[Column] = []
        covered_items: Set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        for source_arc_idx, base in self._source_arcs:
            # Check time limit
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            # Check column limit
            if self._config.max_columns > 0 and len(all_columns) >= self._config.max_columns:
                break

            columns, labels_created, labels_dominated = self._solve_single_source(source_arc_idx, base)
            total_labels += labels_created
            total_dominated += labels_dominated

            # Add columns that cover new items
            for col in columns:
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
