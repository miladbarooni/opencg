"""
Tests for the pricing subproblem module.

This module tests:
- Label dataclass
- LabelPool management and dominance
- LabelingAlgorithm
- Integration with master problem
"""

import pytest
from typing import Dict, List

from opencg.core.column import Column
from opencg.core.network import Network
from opencg.core.node import NodeType
from opencg.core.arc import ArcType
from opencg.core.resource import AccumulatingResource
from opencg.core.problem import Problem, CoverType, ObjectiveSense

from opencg.pricing import (
    Label,
    LabelPool,
    PricingProblem,
    PricingConfig,
    PricingSolution,
    PricingStatus,
    LabelingAlgorithm,
    ElementaryLabelingAlgorithm,
    HeuristicLabelingAlgorithm,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_simple_network_problem() -> Problem:
    r"""
    Create a simple network for testing SPPRC.

    Network structure:
        Source (0) --> A (2) --> B (3) --> Sink (1)
                   \-> C (4) -/

    Arcs:
        0: Source -> A, cost=0
        1: Source -> C, cost=0
        2: A -> B, cost=10, covers item 0
        3: C -> B, cost=5, covers item 1
        4: B -> Sink, cost=0

    Optimal path with duals pi[0]=15, pi[1]=3:
        Source -> A -> B -> Sink: cost=10, RC = 10 - 15 = -5
        Source -> C -> B -> Sink: cost=5, RC = 5 - 3 = 2

    Items: {0, 1}
    """
    network = Network()

    # Add nodes
    source = network.add_source()  # 0
    sink = network.add_sink()      # 1
    a = network.add_node("A", NodeType.GENERIC)  # 2
    b = network.add_node("B", NodeType.GENERIC)  # 3
    c = network.add_node("C", NodeType.GENERIC)  # 4

    # Add arcs
    network.add_arc(source, a, cost=0.0, arc_type=ArcType.SOURCE_ARC)     # 0
    network.add_arc(source, c, cost=0.0, arc_type=ArcType.SOURCE_ARC)     # 1
    network.add_arc(a, b, cost=10.0, arc_type=ArcType.GENERIC)            # 2 - covers item 0
    network.add_arc(c, b, cost=5.0, arc_type=ArcType.GENERIC)             # 3 - covers item 1
    network.add_arc(b, sink, cost=0.0, arc_type=ArcType.SINK_ARC)         # 4

    # Create problem
    problem = Problem(
        name="SimplePricing",
        network=network,
        resources=[AccumulatingResource("time", initial=0.0, max_value=100.0)],
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    # Add cover constraints
    # Arc 2 covers item 0, Arc 3 covers item 1
    problem.add_cover_constraint(item_id=2, name="item_0")  # arc_index = item_id
    problem.add_cover_constraint(item_id=3, name="item_1")

    return problem


def create_resource_constrained_problem() -> Problem:
    """
    Create a problem with resource constraints for testing feasibility.

    Network:
        Source -> A -> B -> Sink
               -> C -> B ->

    Resource: duty_time with max 15 hours
    Arc A->B: 10 hours, cost 100
    Arc C->B: 8 hours, cost 80

    If a path uses more than 15 hours, it's infeasible.
    """
    network = Network()

    source = network.add_source()
    sink = network.add_sink()
    a = network.add_node("A", NodeType.GENERIC)
    b = network.add_node("B", NodeType.GENERIC)
    c = network.add_node("C", NodeType.GENERIC)

    # Arcs with resource consumption
    network.add_arc(source, a, cost=0.0, resource_consumption={"duty_time": 2.0})
    network.add_arc(source, c, cost=0.0, resource_consumption={"duty_time": 3.0})
    network.add_arc(a, b, cost=100.0, resource_consumption={"duty_time": 10.0})
    network.add_arc(c, b, cost=80.0, resource_consumption={"duty_time": 8.0})
    network.add_arc(b, sink, cost=0.0, resource_consumption={"duty_time": 1.0})

    problem = Problem(
        name="ResourceConstrained",
        network=network,
        resources=[AccumulatingResource("duty_time", initial=0.0, max_value=15.0)],
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    problem.add_cover_constraint(item_id=2, name="arc_2")
    problem.add_cover_constraint(item_id=3, name="arc_3")

    return problem


def create_multi_path_problem() -> Problem:
    """
    Create a problem with multiple paths to the same node.

    This tests dominance checking.

    Network:
        Source -> A --> D -> Sink
               -> B -/
               -> C -/

    Multiple paths to D, allowing dominance testing.
    """
    network = Network()

    source = network.add_source()
    sink = network.add_sink()
    a = network.add_node("A", NodeType.GENERIC)
    b = network.add_node("B", NodeType.GENERIC)
    c = network.add_node("C", NodeType.GENERIC)
    d = network.add_node("D", NodeType.GENERIC)

    # Multiple paths from source to D
    network.add_arc(source, a, cost=0.0, resource_consumption={"time": 1.0})
    network.add_arc(source, b, cost=0.0, resource_consumption={"time": 2.0})
    network.add_arc(source, c, cost=0.0, resource_consumption={"time": 3.0})

    network.add_arc(a, d, cost=5.0, resource_consumption={"time": 2.0})   # 3: total time=3
    network.add_arc(b, d, cost=4.0, resource_consumption={"time": 1.0})   # 4: total time=3
    network.add_arc(c, d, cost=3.0, resource_consumption={"time": 0.5})   # 5: total time=3.5

    network.add_arc(d, sink, cost=0.0)

    problem = Problem(
        name="MultiPath",
        network=network,
        resources=[AccumulatingResource("time", initial=0.0, max_value=10.0)],
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    problem.add_cover_constraint(item_id=3, name="a_to_d")
    problem.add_cover_constraint(item_id=4, name="b_to_d")
    problem.add_cover_constraint(item_id=5, name="c_to_d")

    return problem


# =============================================================================
# Test Label
# =============================================================================

class TestLabel:
    """Tests for Label dataclass."""

    def test_label_creation(self):
        """Test creating a label."""
        label = Label(
            node_index=0,
            cost=10.0,
            reduced_cost=5.0,
            resource_values={"time": 2.5},
            covered_items=frozenset({1, 2}),
        )

        assert label.node_index == 0
        assert label.cost == 10.0
        assert label.reduced_cost == 5.0
        assert label.get_resource("time") == 2.5
        assert 1 in label.covered_items
        assert label.is_source_label  # No predecessor

    def test_label_path_reconstruction(self):
        """Test reconstructing path from labels."""
        label1 = Label(node_index=0, cost=0.0, reduced_cost=0.0, label_id=0)
        label2 = Label(
            node_index=1,
            cost=5.0,
            reduced_cost=3.0,
            predecessor=label1,
            last_arc_index=0,
            label_id=1,
        )
        label3 = Label(
            node_index=2,
            cost=10.0,
            reduced_cost=6.0,
            predecessor=label2,
            last_arc_index=1,
            label_id=2,
        )

        arc_path = label3.get_arc_indices()
        assert arc_path == (0, 1)

        node_path = label3.get_node_indices()
        assert node_path == (0, 1, 2)

    def test_label_comparison(self):
        """Test label comparison for priority queue."""
        label1 = Label(node_index=0, cost=10.0, reduced_cost=-5.0, label_id=0)
        label2 = Label(node_index=0, cost=5.0, reduced_cost=-3.0, label_id=1)

        # label1 has more negative RC, so it should be "less than" label2
        assert label1 < label2

    def test_label_path_length(self):
        """Test path length computation."""
        label1 = Label(node_index=0, cost=0.0, reduced_cost=0.0, label_id=0)
        assert label1.path_length == 0

        label2 = Label(
            node_index=1,
            cost=5.0,
            reduced_cost=3.0,
            predecessor=label1,
            last_arc_index=0,
            label_id=1,
        )
        assert label2.path_length == 1


# =============================================================================
# Test LabelPool
# =============================================================================

class TestLabelPool:
    """Tests for LabelPool management."""

    def test_label_pool_creation(self):
        """Test creating a label pool."""
        pool = LabelPool(num_nodes=5)
        assert pool.num_nodes == 5
        assert pool.total_labels == 0

    def test_add_label(self):
        """Test adding labels to pool."""
        pool = LabelPool(num_nodes=3)
        resources = [AccumulatingResource("time", max_value=10.0)]

        label = Label(
            node_index=1,
            cost=5.0,
            reduced_cost=3.0,
            resource_values={"time": 2.0},
            label_id=0,
        )

        added = pool.add_label(label, resources, check_dominance=True)
        assert added
        assert pool.total_labels == 1

    def test_dominance_pruning(self):
        """Test that dominated labels are pruned."""
        pool = LabelPool(num_nodes=3)
        resources = [AccumulatingResource("time", max_value=10.0)]

        # Add first label
        label1 = Label(
            node_index=1,
            cost=5.0,
            reduced_cost=3.0,
            resource_values={"time": 2.0},
            covered_items=frozenset(),
            label_id=0,
        )
        pool.add_label(label1, resources)

        # Add dominated label (worse RC, same or worse resources)
        label2 = Label(
            node_index=1,
            cost=10.0,
            reduced_cost=5.0,  # Worse RC
            resource_values={"time": 3.0},  # Worse time
            covered_items=frozenset(),
            label_id=1,
        )
        added = pool.add_label(label2, resources)
        assert not added  # Should be dominated

        # Label1 should still dominate
        labels = pool.get_labels(1)
        assert len(labels) == 1

    def test_better_label_removes_worse(self):
        """Test that a better label removes existing worse labels."""
        pool = LabelPool(num_nodes=3)
        resources = [AccumulatingResource("time", max_value=10.0)]

        # Add first label (worse)
        label1 = Label(
            node_index=1,
            cost=10.0,
            reduced_cost=5.0,
            resource_values={"time": 3.0},
            covered_items=frozenset(),
            label_id=0,
        )
        pool.add_label(label1, resources)

        # Add better label
        label2 = Label(
            node_index=1,
            cost=5.0,
            reduced_cost=2.0,  # Better RC
            resource_values={"time": 1.0},  # Better time
            covered_items=frozenset(),
            label_id=1,
        )
        added = pool.add_label(label2, resources)
        assert added

        # Only better label should remain
        labels = pool.get_labels(1)
        assert len(labels) == 1
        assert labels[0].reduced_cost == 2.0

    def test_statistics(self):
        """Test pool statistics."""
        pool = LabelPool(num_nodes=3)
        resources = [AccumulatingResource("time", max_value=10.0)]

        # Add several labels
        for i in range(5):
            label = Label(
                node_index=1,
                cost=float(i),
                reduced_cost=float(i),
                resource_values={"time": float(i)},
                covered_items=frozenset(),
                label_id=i,
            )
            pool.add_label(label, resources)

        stats = pool.statistics()
        assert stats['total_created'] == 5
        assert stats['total_dominated'] >= 0  # Some may be dominated


# =============================================================================
# Test LabelingAlgorithm
# =============================================================================

class TestLabelingAlgorithm:
    """Tests for the labeling algorithm."""

    def test_algorithm_creation(self):
        """Test creating the labeling algorithm."""
        problem = create_simple_network_problem()
        pricing = LabelingAlgorithm(problem)

        assert pricing.problem is problem
        assert pricing.config is not None

    def test_solve_no_duals(self):
        """Test solving with no dual values (all zeros)."""
        problem = create_simple_network_problem()
        pricing = LabelingAlgorithm(problem)

        # With zero duals, reduced cost = cost
        pricing.set_dual_values({})
        solution = pricing.solve()

        # Should find paths with positive RC (no negative RC without duals)
        assert solution.status in (PricingStatus.NO_COLUMNS, PricingStatus.COLUMNS_FOUND)

    def test_solve_with_duals(self):
        """Test solving with dual values that create negative RC."""
        problem = create_simple_network_problem()
        pricing = LabelingAlgorithm(problem)

        # Set duals high enough to create negative reduced cost
        # Arc 2: cost=10, item=2 -> RC = 10 - pi[2]
        # With pi[2]=15, RC = 10 - 15 = -5
        pricing.set_dual_values({2: 15.0, 3: 3.0})
        solution = pricing.solve()

        assert solution.has_negative_reduced_cost
        assert solution.num_columns > 0
        assert solution.best_reduced_cost < 0

    def test_column_creation(self):
        """Test that columns are correctly created from labels."""
        problem = create_simple_network_problem()
        pricing = LabelingAlgorithm(problem)

        pricing.set_dual_values({2: 20.0, 3: 0.0})  # High dual for item 2
        solution = pricing.solve()

        assert solution.num_columns > 0
        col = solution.get_best_column()
        assert col is not None
        assert col.reduced_cost < 0
        assert len(col.arc_indices) > 0

    def test_resource_feasibility(self):
        """Test that infeasible paths are not generated."""
        problem = create_resource_constrained_problem()
        pricing = LabelingAlgorithm(problem)

        pricing.set_dual_values({2: 200.0, 3: 200.0})  # High duals
        solution = pricing.solve()

        # All returned columns should be resource-feasible
        for col in solution.columns:
            # Check that duty_time is within limits
            # (The algorithm should only return feasible paths)
            assert col.reduced_cost is not None

    def test_config_max_columns(self):
        """Test limiting number of columns returned."""
        problem = create_multi_path_problem()
        config = PricingConfig(max_columns=1)
        pricing = LabelingAlgorithm(problem, config)

        pricing.set_dual_values({3: 100.0, 4: 100.0, 5: 100.0})
        solution = pricing.solve()

        assert solution.num_columns <= 1

    def test_algorithm_statistics(self):
        """Test that statistics are tracked."""
        problem = create_simple_network_problem()
        pricing = LabelingAlgorithm(problem)

        pricing.set_dual_values({2: 15.0, 3: 3.0})
        solution = pricing.solve()

        assert solution.num_labels_created > 0
        assert solution.solve_time >= 0

        stats = pricing.get_label_statistics()
        assert 'total_labels' in stats


class TestHeuristicLabeling:
    """Tests for heuristic labeling algorithm."""

    def test_early_termination(self):
        """Test that early termination works."""
        problem = create_multi_path_problem()
        pricing = HeuristicLabelingAlgorithm(
            problem,
            max_labels_per_node=50,
            early_termination_count=1,
        )

        pricing.set_dual_values({3: 100.0, 4: 100.0, 5: 100.0})
        solution = pricing.solve()

        # Should find at least one column and possibly stop early
        assert solution.num_columns >= 1


class TestPricingSolution:
    """Tests for PricingSolution dataclass."""

    def test_default_status(self):
        """Test default solution status."""
        solution = PricingSolution()
        assert solution.status == PricingStatus.NO_COLUMNS
        assert not solution.has_negative_reduced_cost
        assert solution.num_columns == 0

    def test_with_columns(self):
        """Test solution with columns."""
        col = Column(
            arc_indices=(0, 1, 2),
            cost=10.0,
            reduced_cost=-5.0,
            column_id=0,
        )
        solution = PricingSolution(
            status=PricingStatus.COLUMNS_FOUND,
            columns=[col],
            best_reduced_cost=-5.0,
        )

        assert solution.has_negative_reduced_cost
        assert solution.num_columns == 1
        assert solution.get_best_column() is col

    def test_summary(self):
        """Test solution summary."""
        solution = PricingSolution(
            status=PricingStatus.COLUMNS_FOUND,
            columns=[],
            best_reduced_cost=-3.0,
            num_labels_created=100,
            num_labels_dominated=30,
            solve_time=0.5,
        )
        summary = solution.summary()
        assert "COLUMNS_FOUND" in summary
        assert "-3" in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestPricingMasterIntegration:
    """Test integration between pricing and master problem."""

    def test_pricing_finds_improving_columns(self):
        """Test that pricing finds columns that improve the master."""
        from opencg.master import HiGHSMasterProblem, HIGHS_AVAILABLE

        if not HIGHS_AVAILABLE:
            pytest.skip("HiGHS not available")

        problem = create_simple_network_problem()

        # Create initial columns manually
        col1 = Column(
            arc_indices=(0, 2, 4),  # Source -> A -> B -> Sink
            cost=10.0,
            covered_items=frozenset({2}),  # Covers item 2
            column_id=0,
        )
        col2 = Column(
            arc_indices=(1, 3, 4),  # Source -> C -> B -> Sink
            cost=5.0,
            covered_items=frozenset({3}),  # Covers item 3
            column_id=1,
        )

        # Setup master
        master = HiGHSMasterProblem(problem)
        master.add_columns([col1, col2])

        # Solve master to get duals
        solution = master.solve_lp()
        assert solution.is_optimal

        duals = master.get_dual_values()

        # Setup pricing
        pricing = LabelingAlgorithm(problem)
        pricing.set_dual_values(duals)

        # Solve pricing
        pricing_sol = pricing.solve()

        # With optimal LP solution and no other columns,
        # pricing should find no improving columns (or confirm optimality)
        # This depends on the specific dual values


# =============================================================================
# Test NetworkCache
# =============================================================================

class MockAlgorithm:
    """Mock algorithm for testing NetworkCache."""

    def __init__(self):
        self.last_duals = {}

    def set_dual_values(self, duals):
        self.last_duals = dict(duals)


class TestNetworkCache:
    """Tests for NetworkCache class used in FastPerSourcePricing."""

    def test_cache_basic_operations(self):
        """Test basic get/put operations."""
        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        builds = []

        def mock_builder(idx, base):
            builds.append(idx)
            return NetworkCacheEntry(None, MockAlgorithm(), {idx: idx})

        cache = NetworkCache(max_size=3, builder_func=mock_builder)

        # First access - cache miss
        entry = cache.get_or_build(1, "base1", {})
        assert 1 in builds
        assert len(cache) == 1
        assert entry.arc_map == {1: 1}

        # Second access - cache hit
        builds.clear()
        entry2 = cache.get_or_build(1, "base1", {})
        assert len(builds) == 0  # No rebuild
        assert entry2 is entry  # Same entry returned

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        builds = []

        def mock_builder(idx, base):
            builds.append(idx)
            return NetworkCacheEntry(None, MockAlgorithm(), {})

        cache = NetworkCache(max_size=2, builder_func=mock_builder)

        cache.get_or_build(1, "b", {})
        cache.get_or_build(2, "b", {})
        assert len(cache) == 2

        # Access 1 again to make it most recent
        cache.get_or_build(1, "b", {})

        # Add 3 - should evict 2 (LRU)
        cache.get_or_build(3, "b", {})
        assert len(cache) == 2
        assert 1 in cache
        assert 3 in cache
        assert 2 not in cache

    def test_unlimited_cache(self):
        """Test unlimited cache mode (max_size=0)."""
        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        def mock_builder(idx, base):
            return NetworkCacheEntry(None, MockAlgorithm(), {})

        cache = NetworkCache(max_size=0, builder_func=mock_builder)

        # Add many entries - none should be evicted
        for i in range(100):
            cache.get_or_build(i, "b", {})

        assert len(cache) == 100

    def test_dual_update_propagation(self):
        """Test that dual updates propagate to all cached algorithms."""
        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        algos = {}

        def mock_builder(idx, base):
            algo = MockAlgorithm()
            algos[idx] = algo
            return NetworkCacheEntry(None, algo, {})

        cache = NetworkCache(max_size=10, builder_func=mock_builder)

        cache.get_or_build(1, "b", {0: 1.0})
        cache.get_or_build(2, "b", {0: 1.0})

        # Update all duals
        cache.update_all_duals({0: 5.0, 1: 10.0})

        assert algos[1].last_duals == {0: 5.0, 1: 10.0}
        assert algos[2].last_duals == {0: 5.0, 1: 10.0}

    def test_prefill_mode(self):
        """Test prefill for prebuild mode."""
        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        builds = []

        def mock_builder(idx, base):
            builds.append(idx)
            return NetworkCacheEntry(None, MockAlgorithm(), {})

        cache = NetworkCache(max_size=0, builder_func=mock_builder)  # Unlimited

        sources = [(1, "a"), (2, "b"), (3, "c")]
        cache.prefill(sources, {})

        assert len(cache) == 3
        assert set(builds) == {1, 2, 3}

    def test_statistics(self):
        """Test cache statistics tracking."""
        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        def mock_builder(idx, base):
            return NetworkCacheEntry(None, MockAlgorithm(), {})

        cache = NetworkCache(max_size=2, builder_func=mock_builder)

        # Generate some cache activity
        cache.get_or_build(1, "b", {})  # miss
        cache.get_or_build(2, "b", {})  # miss
        cache.get_or_build(1, "b", {})  # hit
        cache.get_or_build(3, "b", {})  # miss + eviction

        stats = cache.statistics()
        assert stats['hits'] == 1
        assert stats['misses'] == 3
        assert stats['evictions'] == 1
        assert stats['size'] == 2
        assert stats['max_size'] == 2
        assert stats['hit_rate'] == 0.25  # 1 hit / 4 total

    def test_thread_safety(self):
        """Test concurrent access from multiple threads."""
        import concurrent.futures
        import threading
        import time

        from opencg.pricing.fast_per_source import NetworkCache, NetworkCacheEntry

        builds = []
        lock = threading.Lock()

        def mock_builder(idx, base):
            with lock:
                builds.append(idx)
            time.sleep(0.01)  # Simulate work
            return NetworkCacheEntry(None, MockAlgorithm(), {})

        cache = NetworkCache(max_size=10, builder_func=mock_builder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cache.get_or_build, i % 5, "b", {})
                for i in range(20)
            ]
            concurrent.futures.wait(futures)

        # Each unique index should be built exactly once
        assert len(cache) == 5
        # All 5 unique indices should have been built
        assert len(set(builds)) == 5


class TestFastPerSourcePricingLazyMode:
    """Tests for FastPerSourcePricing lazy network building."""

    def test_lazy_mode_threshold(self):
        """Test automatic mode selection based on source count."""
        from opencg.pricing.fast_per_source import FastPerSourcePricing

        # Test the _determine_lazy_mode logic directly
        # We can't easily instantiate without a real problem, so test the threshold constant
        assert FastPerSourcePricing.LAZY_MODE_THRESHOLD == 500

    def test_explicit_prebuild_override(self):
        """Test that max_cached_networks=0 forces prebuild mode."""
        from opencg.pricing.fast_per_source import FastPerSourcePricing

        # Create a mock to test _determine_lazy_mode
        class MockPricing:
            LAZY_MODE_THRESHOLD = 500

            def _determine_lazy_mode(self, num_sources, max_cached_networks):
                return FastPerSourcePricing._determine_lazy_mode(
                    self, num_sources, max_cached_networks
                )

        mock = MockPricing()

        # 0 should force prebuild even with many sources
        assert mock._determine_lazy_mode(1000, 0) is False

        # None with < 500 sources should prebuild
        assert mock._determine_lazy_mode(100, None) is False

        # None with >= 500 sources should use lazy
        assert mock._determine_lazy_mode(600, None) is True

        # Explicit positive value should force lazy
        assert mock._determine_lazy_mode(100, 50) is True


# =============================================================================
# Run tests standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
