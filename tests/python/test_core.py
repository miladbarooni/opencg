"""
Tests for the core module.

Run with: pytest tests/python/test_core.py -v
Or without pytest: python tests/python/test_core.py
"""

# Try to import pytest, but don't require it
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
from opencg.core.resource import (
    AccumulatingResource,
    IntervalResource,
    StateResource,
    TimeWindowResource,
)
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType, make_flight_arc, make_connection_arc
from opencg.core.network import Network
from opencg.core.column import Column, ColumnPool
from opencg.core.problem import Problem, CoverType, ProblemBuilder, CoverConstraint


class TestResource:
    """Tests for Resource classes."""

    def test_accumulating_resource_basic(self):
        """Test AccumulatingResource initialization and extension."""
        duty = AccumulatingResource("duty_time", initial=0.0, max_value=10.0)

        assert duty.name == "duty_time"
        assert duty.initial_value() == 0.0
        assert duty.max_value == 10.0

    def test_accumulating_resource_extension(self):
        """Test AccumulatingResource.extend()."""
        duty = AccumulatingResource("duty_time", max_value=10.0)

        # Create a mock arc
        arc = Arc(
            index=0, source=0, target=1, cost=0.0,
            resource_consumption={"duty_time": 2.5}
        )

        # Extend from 0
        new_value = duty.extend(0.0, arc)
        assert new_value == 2.5

        # Extend from 5
        new_value = duty.extend(5.0, arc)
        assert new_value == 7.5

        # Extend past max (should be None)
        new_value = duty.extend(8.0, arc)
        assert new_value is None

    def test_accumulating_resource_dominance(self):
        """Test AccumulatingResource.dominates()."""
        duty = AccumulatingResource("duty_time", max_value=10.0)

        assert duty.dominates(2.0, 3.0) is True  # Less is better
        assert duty.dominates(3.0, 2.0) is False
        assert duty.dominates(2.0, 2.0) is True  # Equal dominates

    def test_interval_resource(self):
        """Test IntervalResource."""
        conn = IntervalResource("connection", min_value=0.5, max_value=4.0)

        arc = Arc(
            index=0, source=0, target=1, cost=0.0,
            resource_consumption={"connection": 1.0}
        )

        # Valid extension
        assert conn.extend(0.0, arc) == 1.0

        # Too short (below min)
        arc_short = Arc(
            index=1, source=0, target=1, cost=0.0,
            resource_consumption={"connection": 0.3}
        )
        assert conn.extend(0.0, arc_short) is None

        # Too long (above max)
        arc_long = Arc(
            index=2, source=0, target=1, cost=0.0,
            resource_consumption={"connection": 5.0}
        )
        assert conn.extend(0.0, arc_long) is None

    def test_state_resource(self):
        """Test StateResource for elementarity."""
        visited = StateResource("visited", forbidden_revisit=True)

        arc1 = Arc(
            index=0, source=0, target=1, cost=0.0,
            resource_consumption={"visited": 1}
        )
        arc2 = Arc(
            index=1, source=1, target=2, cost=0.0,
            resource_consumption={"visited": 2}
        )
        arc_revisit = Arc(
            index=2, source=2, target=1, cost=0.0,
            resource_consumption={"visited": 1}
        )

        # Initial state
        state = visited.initial_value()
        assert state == set()

        # Visit node 1
        state = visited.extend(state, arc1)
        assert state == {1}

        # Visit node 2
        state = visited.extend(state, arc2)
        assert state == {1, 2}

        # Try to revisit node 1 (should fail)
        state = visited.extend(state, arc_revisit)
        assert state is None


class TestNode:
    """Tests for Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(index=0, name="BASE1", node_type=NodeType.BASE)

        assert node.index == 0
        assert node.name == "BASE1"
        assert node.node_type == NodeType.BASE

    def test_node_attributes(self):
        """Test node attributes."""
        node = Node(
            index=0, name="DEP1", node_type=NodeType.FLIGHT_DEP,
            attributes={"location": "JFK", "time": 8.5}
        )

        assert node.get_attribute("location") == "JFK"
        assert node.time == 8.5
        assert node.get_attribute("missing", "default") == "default"

    def test_node_equality(self):
        """Test node equality based on index."""
        node1 = Node(index=0, name="A")
        node2 = Node(index=0, name="B")  # Different name, same index
        node3 = Node(index=1, name="A")  # Same name, different index

        assert node1 == node2
        assert node1 != node3


class TestArc:
    """Tests for Arc class."""

    def test_arc_creation(self):
        """Test basic arc creation."""
        arc = Arc(
            index=0,
            source=0,
            target=1,
            cost=100.0,
            resource_consumption={"duty_time": 2.0},
            arc_type=ArcType.FLIGHT
        )

        assert arc.index == 0
        assert arc.source == 0
        assert arc.target == 1
        assert arc.cost == 100.0
        assert arc.get_consumption("duty_time") == 2.0
        assert arc.get_consumption("missing") == 0.0

    def test_arc_factory(self):
        """Test arc factory functions."""
        flight = make_flight_arc(
            index=0, source=0, target=1,
            flight_number="AA123", duration=2.5, cost=500.0
        )

        assert flight.arc_type == ArcType.FLIGHT
        assert flight.get_consumption("duty_time") == 2.5
        assert flight.get_consumption("flight_time") == 2.5
        assert flight.flight_number == "AA123"


class TestNetwork:
    """Tests for Network class."""

    def test_empty_network(self):
        """Test empty network."""
        network = Network()

        assert network.num_nodes == 0
        assert network.num_arcs == 0
        assert network.source is None
        assert network.sink is None

    def test_add_nodes(self):
        """Test adding nodes."""
        network = Network()

        source = network.add_source()
        sink = network.add_sink()
        base = network.add_node("BASE1", NodeType.BASE)

        assert network.num_nodes == 3
        assert network.source.name == "__SOURCE__"
        assert network.sink.name == "__SINK__"
        assert network.get_node(base).name == "BASE1"

    def test_add_arcs(self):
        """Test adding arcs."""
        network = Network()

        n0 = network.add_node("A")
        n1 = network.add_node("B")

        arc_idx = network.add_arc(
            source=n0, target=n1, cost=10.0,
            resource_consumption={"time": 1.0}
        )

        assert network.num_arcs == 1
        arc = network.get_arc(arc_idx)
        assert arc.source == n0
        assert arc.target == n1

    def test_adjacency(self):
        """Test outgoing/incoming arc traversal."""
        network = Network()

        n0 = network.add_node("A")
        n1 = network.add_node("B")
        n2 = network.add_node("C")

        network.add_arc(n0, n1, cost=1.0)
        network.add_arc(n0, n2, cost=2.0)
        network.add_arc(n1, n2, cost=3.0)

        outgoing_from_0 = list(network.outgoing_arcs(n0))
        assert len(outgoing_from_0) == 2

        incoming_to_2 = list(network.incoming_arcs(n2))
        assert len(incoming_to_2) == 2


class TestColumn:
    """Tests for Column class."""

    def test_column_creation(self):
        """Test basic column creation."""
        column = Column(
            arc_indices=(0, 1, 2),
            cost=300.0,
            resource_values={"duty_time": 6.0},
            covered_items=frozenset({10, 20, 30})
        )

        assert column.num_arcs == 3
        assert column.cost == 300.0
        assert column.get_resource("duty_time") == 6.0
        assert column.covers_item(20)
        assert not column.covers_item(99)

    def test_column_immutability(self):
        """Test that columns are hashable (immutable)."""
        col1 = Column(arc_indices=(0, 1), cost=100.0)
        col2 = Column(arc_indices=(0, 1), cost=200.0)
        col3 = Column(arc_indices=(1, 2), cost=100.0)

        # Same arcs = same hash
        assert hash(col1) == hash(col2)
        assert col1 == col2

        # Different arcs = different hash
        assert hash(col1) != hash(col3)
        assert col1 != col3

        # Can be used in sets
        column_set = {col1, col2, col3}
        assert len(column_set) == 2


class TestColumnPool:
    """Tests for ColumnPool class."""

    def test_column_pool(self):
        """Test column pool operations."""
        pool = ColumnPool()

        col1 = Column(arc_indices=(0, 1), cost=100.0, covered_items=frozenset({1}))
        col2 = Column(arc_indices=(2, 3), cost=200.0, covered_items=frozenset({1, 2}))

        col1 = pool.add(col1)
        col2 = pool.add(col2)

        assert pool.size == 2
        assert col1.column_id is not None
        assert col2.column_id is not None

        # Find columns covering item 1
        covering = pool.columns_covering(1)
        assert len(covering) == 2

        # Find columns covering item 2
        covering = pool.columns_covering(2)
        assert len(covering) == 1


class TestProblem:
    """Tests for Problem class."""

    def test_problem_builder(self):
        """Test problem builder."""
        network = Network()
        network.add_source()
        network.add_sink()
        n0 = network.add_node("A")
        n1 = network.add_node("B")
        a0 = network.add_arc(n0, n1, cost=1.0)

        duty = AccumulatingResource("duty_time", max_value=10.0)

        problem = (
            ProblemBuilder("TestProblem")
            .with_network(network)
            .add_resource(duty)
            .add_cover_constraint(item_id=a0, name="arc_0")
            .with_cover_type(CoverType.SET_PARTITIONING)
            .build_unchecked()
        )

        assert problem.name == "TestProblem"
        assert problem.network.num_nodes == 4
        assert len(problem.resources) == 1
        assert problem.num_cover_constraints == 1


def run_tests():
    """Simple test runner that doesn't require pytest."""
    test_classes = [
        TestResource,
        TestNode,
        TestArc,
        TestNetwork,
        TestColumn,
        TestColumnPool,
        TestProblem,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n=== {test_class.__name__} ===")
        instance = test_class()

        for name in dir(instance):
            if name.startswith('test_'):
                try:
                    getattr(instance, name)()
                    print(f"  PASS: {name}")
                    total_passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {name} - {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"  ERROR: {name} - {e}")
                    total_failed += 1

    print(f"\n{'=' * 40}")
    print(f"SUMMARY: {total_passed} passed, {total_failed} failed")
    print('=' * 40)

    return total_failed == 0


if __name__ == "__main__":
    if HAS_PYTEST:
        import sys
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_tests()
        import sys
        sys.exit(0 if success else 1)
