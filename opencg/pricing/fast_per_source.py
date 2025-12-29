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

import os
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

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


class NetworkCacheEntry:
    """Single entry in the network cache."""

    __slots__ = ('network', 'algorithm', 'arc_map')

    def __init__(
        self,
        network: "CppNetwork",
        algorithm: "CppLabelingAlgorithm",
        arc_map: dict[int, int],
    ):
        self.network = network
        self.algorithm = algorithm
        self.arc_map = arc_map


class NetworkCache:
    """
    Thread-safe LRU cache for per-source networks.

    Provides on-demand network building with automatic eviction of
    least-recently-used entries when the cache reaches capacity.

    Thread Safety:
    - Uses a single RLock for all operations
    - RLock allows the same thread to acquire the lock multiple times
    - All public methods are thread-safe

    Parameters:
        max_size: Maximum number of networks to cache (0 = unlimited/prebuild all)
        builder_func: Function(source_arc_idx, base) -> NetworkCacheEntry
    """

    def __init__(
        self,
        max_size: int,
        builder_func: Callable[[int, str], NetworkCacheEntry],
    ):
        self._max_size = max_size
        self._builder_func = builder_func

        # OrderedDict maintains insertion order; we move items to end on access
        self._cache: OrderedDict[int, NetworkCacheEntry] = OrderedDict()

        # Single lock for thread safety - RLock allows reentrant locking
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_or_build(
        self,
        source_arc_idx: int,
        base: str,
        current_duals: dict[int, float],
    ) -> NetworkCacheEntry:
        """
        Get a network from cache, building it if necessary.

        This is the primary interface for the cache. It:
        1. Returns cached entry if available (marking as recently used)
        2. Builds and caches new entry if not in cache
        3. Evicts LRU entry if cache is full
        4. Applies current dual values to the algorithm

        Thread-safe: Multiple threads can call this concurrently.

        Args:
            source_arc_idx: Index of the source arc
            base: Base string for this source
            current_duals: Current dual values to apply

        Returns:
            NetworkCacheEntry with network, algorithm, and arc_map
        """
        with self._lock:
            if source_arc_idx in self._cache:
                # Cache hit - move to end (most recently used)
                self._cache.move_to_end(source_arc_idx)
                self._hits += 1
                entry = self._cache[source_arc_idx]
            else:
                # Cache miss - build new entry
                self._misses += 1

                # Evict LRU entry if at capacity (max_size > 0 means limited)
                if self._max_size > 0 and len(self._cache) >= self._max_size:
                    # Pop the oldest (first) item
                    self._cache.popitem(last=False)
                    self._evictions += 1

                # Build new entry (releases lock temporarily for expensive build)
                # Note: We build outside the critical section to reduce contention
                pass  # Will build below after releasing lock check

            # Check if we need to build (cache miss case)
            if source_arc_idx not in self._cache:
                # Build the entry (this is expensive but doesn't need the lock)
                entry = self._builder_func(source_arc_idx, base)
                self._cache[source_arc_idx] = entry
            else:
                entry = self._cache[source_arc_idx]

            # Always apply current duals (they may have changed since caching)
            entry.algorithm.set_dual_values(current_duals)

            return entry

    def update_all_duals(self, dual_values: dict[int, float]) -> None:
        """
        Update dual values on all currently cached algorithms.

        Called when duals are updated to keep cached algorithms in sync.

        Args:
            dual_values: New dual values
        """
        with self._lock:
            for entry in self._cache.values():
                entry.algorithm.set_dual_values(dual_values)

    def prefill(
        self,
        sources: list[tuple[int, str]],
        initial_duals: dict[int, float],
    ) -> None:
        """
        Pre-fill the cache with networks (for prebuild mode).

        Used when the number of sources is below the threshold and
        we want to prebuild all networks at initialization.

        Args:
            sources: List of (source_arc_idx, base) tuples
            initial_duals: Initial dual values (typically empty)
        """
        with self._lock:
            for source_arc_idx, base in sources:
                if source_arc_idx not in self._cache:
                    entry = self._builder_func(source_arc_idx, base)
                    entry.algorithm.set_dual_values(initial_duals)
                    self._cache[source_arc_idx] = entry

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, source_arc_idx: int) -> bool:
        """Check if source is in cache."""
        with self._lock:
            return source_arc_idx in self._cache

    def statistics(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
            }


class FastPerSourcePricing(PricingProblem):
    """
    Fast per-source-arc pricing with optional lazy network building.

    This provides comprehensive coverage by solving separate SPPRCs for each
    source arc. Networks can be prebuilt at initialization (for small instances)
    or built lazily on-demand with LRU caching (for large instances).

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
        max_cached_networks: Network caching strategy:
            - None (default): Auto-select based on source count
              (< 500 sources: prebuild all, >= 500: lazy with LRU cache)
            - 0: Force prebuild all networks (original behavior)
            - N > 0: Force lazy mode with cache size N
    """

    # Threshold for automatic lazy mode selection
    LAZY_MODE_THRESHOLD = 500

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 20,
        cols_per_source: int = 3,
        time_per_source: float = 0.05,
        num_threads: int = 1,
        max_cached_networks: Optional[int] = None,
    ):
        if not HAS_CPP_BACKEND:
            raise ImportError("C++ backend not available")

        super().__init__(problem, config)

        self._max_labels_per_node = max_labels_per_node
        self._cols_per_source = cols_per_source
        self._time_per_source = time_per_source
        self._num_threads = num_threads if num_threads > 0 else os.cpu_count() or 1

        # Priority items - columns covering these get boosted in selection
        self._priority_items: set[int] = set()

        # Get source arcs with their bases
        self._source_arcs: list[tuple[int, str]] = []
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
        self._numeric_resources: list[str] = []
        self._resource_limits: list[float] = []
        for r in problem.resources:
            if hasattr(r, 'max_value'):
                self._numeric_resources.append(r.name)
                self._resource_limits.append(r.max_value)

        # Mapping from flight arc index to source arcs that include it
        self._flight_to_sources: dict[int, set[int]] = defaultdict(set)

        # Determine caching strategy
        num_sources = len(self._source_arcs)
        self._lazy_mode = self._determine_lazy_mode(num_sources, max_cached_networks)

        if self._lazy_mode:
            # Calculate effective cache size
            if max_cached_networks is None:
                cache_size = min(200, max(50, num_sources // 3))
            else:
                cache_size = max_cached_networks

            print(f"FastPerSourcePricing: lazy mode with cache_size={cache_size} "
                  f"for {num_sources} sources")

            # Create network cache with builder function
            self._network_cache = NetworkCache(
                max_size=cache_size,
                builder_func=self._build_network_entry,
            )

            # Build flight_to_sources mapping by scanning arcs once
            self._build_flight_to_sources_mapping()
        else:
            # Prebuild all networks (existing behavior)
            print(f"FastPerSourcePricing: prebuilding {num_sources} networks...")
            build_start = time.time()

            # Create cache with unlimited size for prebuild mode
            self._network_cache = NetworkCache(
                max_size=0,  # Unlimited
                builder_func=self._build_network_entry,
            )

            # Prefill all networks
            self._network_cache.prefill(self._source_arcs, {})

            print(f"  Prebuilt in {time.time() - build_start:.2f}s")

    def _determine_lazy_mode(
        self,
        num_sources: int,
        max_cached_networks: Optional[int],
    ) -> bool:
        """
        Determine whether to use lazy loading or prebuild all.

        Returns True for lazy mode, False for prebuild all.
        """
        # Explicit user override: 0 means prebuild all
        if max_cached_networks == 0:
            return False

        # Explicit user override: positive value means lazy mode
        if max_cached_networks is not None and max_cached_networks > 0:
            return True

        # Auto-select based on source count
        return num_sources >= self.LAZY_MODE_THRESHOLD

    def _build_flight_to_sources_mapping(self) -> None:
        """
        Build the flight-to-sources mapping without building networks.

        In lazy mode, we need this mapping upfront for priority item handling.
        This scans all arcs once to determine which flights are reachable from
        which sources based on arc filtering rules.
        """
        network = self._problem.network

        for source_arc_idx, base in self._source_arcs:
            valid_sinks = self._sink_arcs_by_base[base]

            for arc in network.arcs:
                # Skip other source arcs (same filtering as network building)
                if arc.arc_type == ArcType.SOURCE_ARC:
                    if arc.index != source_arc_idx:
                        continue

                # Skip sink arcs from other bases
                if arc.arc_type == ArcType.SINK_ARC:
                    if arc.index not in valid_sinks:
                        continue

                # Track flight reachability
                if arc.arc_type == ArcType.FLIGHT:
                    self._flight_to_sources[arc.index].add(source_arc_idx)

    def _build_network_entry(self, source_arc_idx: int, base: str) -> NetworkCacheEntry:
        """
        Build isolated C++ network for a single source arc.

        Returns a NetworkCacheEntry containing the network, algorithm, and arc map.
        This method is called by the NetworkCache when a cache miss occurs.
        """
        network = self._problem.network

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

            # Track which flights this source can reach (for prebuild mode)
            # In lazy mode, this is done by _build_flight_to_sources_mapping
            if not self._lazy_mode and arc.arc_type == ArcType.FLIGHT:
                self._flight_to_sources[arc.index].add(source_arc_idx)

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

        return NetworkCacheEntry(cpp_network, algo, cpp_arc_to_py)

    def _on_duals_updated(self) -> None:
        """Update dual values in all cached algorithms."""
        self._network_cache.update_all_duals(self._dual_values)

    def set_priority_items(self, items: set[int]) -> None:
        """Set items that should be prioritized for coverage.

        Columns covering these items will be preferred during column selection.
        This is useful for ensuring uncovered flights get columns generated.
        """
        self._priority_items = set(items)

    def _solve_impl(self) -> PricingSolution:
        """Run labeling on all prebuilt source networks."""
        if self._num_threads > 1:
            return self._solve_parallel()
        else:
            return self._solve_sequential()

    def _solve_single_source(
        self, source_arc_idx: int, base: str
    ) -> tuple[list[Column], int, int]:
        """
        Solve labeling for a single source arc.

        In lazy mode, this will build the network on-demand if not cached.

        Returns:
            (columns, labels_created, labels_dominated)
        """
        # Get network entry from cache (builds on-demand if needed)
        entry = self._network_cache.get_or_build(
            source_arc_idx,
            base,
            self._dual_values,
        )

        algo = entry.algorithm
        arc_map = entry.arc_map

        # Boost max_columns if this source can reach priority items
        boosted = False
        if self._priority_items:
            # Check if any flight in this network is a priority item
            flights_in_network = set(arc_map.values())
            if flights_in_network & self._priority_items:
                # Boost max_columns 10x for priority sources
                algo.set_max_columns(self._cols_per_source * 10)
                boosted = True

        # Run labeling (C++ releases GIL during solve)
        result = algo.solve()

        # Reset max_columns if we boosted
        if boosted:
            algo.set_max_columns(self._cols_per_source)

        # Convert columns
        columns = []
        threshold = self._config.reduced_cost_threshold or -1e-6

        for cpp_label in result.columns:
            col = self._convert_cpp_label(cpp_label, arc_map, base)
            if col is None:
                continue

            # Accept columns with negative reduced cost
            if col.reduced_cost < threshold:
                columns.append(col)
            # Also accept columns covering priority items even with near-zero reduced cost
            elif self._priority_items and col.covered_items & self._priority_items:
                # Use relaxed threshold for priority items (to handle numerical precision)
                if col.reduced_cost < 1e-3:
                    columns.append(col)

        return columns, result.labels_created, result.labels_dominated

    def _solve_parallel(self) -> PricingSolution:
        """Run labeling on all sources in parallel using ThreadPoolExecutor.

        IMPORTANT: We process ALL source arcs to ensure complete coverage.
        The max_columns limit is applied at the end by keeping the best columns,
        not by early termination which could leave flights uncovered.
        """
        start_time = time.time()

        all_columns: list[Column] = []
        covered_items: set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        # Process in batches for efficiency
        batch_size = self._num_threads * 4  # Process 4 batches worth at a time
        source_arcs = list(self._source_arcs)

        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            idx = 0
            while idx < len(source_arcs):
                # NOTE: We do NOT check column limit here anymore!
                # We must process all source arcs to ensure all flights have a chance
                # to be covered. The column limit is applied at the end.

                # Check time limit (still respect this for very long runs)
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

        # Sort columns: prioritize those covering priority items, then by reduced cost
        if self._priority_items:
            all_columns.sort(key=lambda c: (
                0 if c.covered_items & self._priority_items else 1,
                c.reduced_cost
            ))
        else:
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
        """Run labeling on all prebuilt source networks sequentially.

        IMPORTANT: We always process ALL source arcs to ensure complete coverage.
        The max_columns limit is applied at the end by keeping the best columns,
        not by early termination which could leave flights uncovered.
        """
        start_time = time.time()

        all_columns: list[Column] = []
        covered_items: set[int] = set()
        best_rc = None
        total_labels = 0
        total_dominated = 0

        for source_arc_idx, base in self._source_arcs:
            # Check time limit (still respect this for very long runs)
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            # NOTE: We do NOT check column limit here anymore!
            # We must process all source arcs to ensure all flights have a chance
            # to be covered. The column limit is applied at the end.

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

        # Sort columns: prioritize those covering priority items, then by reduced cost
        if self._priority_items:
            all_columns.sort(key=lambda c: (
                0 if c.covered_items & self._priority_items else 1,
                c.reduced_cost
            ))
        else:
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
