"""
Tests for the column generation solver module.

This module tests:
- CGSolution dataclass
- CGConfig configuration
- ColumnGeneration algorithm
- Integration of master and pricing
"""

import pytest
from typing import Dict, List

from opencg.core.column import Column, ColumnPool
from opencg.core.network import Network
from opencg.core.node import NodeType
from opencg.core.arc import ArcType
from opencg.core.resource import AccumulatingResource
from opencg.core.problem import Problem, CoverType, ObjectiveSense

from opencg.master import HiGHSMasterProblem, HIGHS_AVAILABLE
from opencg.pricing import LabelingAlgorithm, PricingConfig

from opencg.solver import (
    ColumnGeneration,
    CGConfig,
    CGSolution,
    CGStatus,
    CGIteration,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_small_problem() -> Problem:
    """
    Create a small problem for testing column generation.

    Network:
        Source -> A -> B -> Sink
               -> C -> B ->

    Items to cover: 2 (arc A->B) and 3 (arc C->B)

    Columns:
        Path Source->A->B->Sink: cost 10, covers item 2
        Path Source->C->B->Sink: cost 5, covers item 3
    """
    network = Network()

    source = network.add_source()
    sink = network.add_sink()
    a = network.add_node("A", NodeType.GENERIC)
    b = network.add_node("B", NodeType.GENERIC)
    c = network.add_node("C", NodeType.GENERIC)

    # Add arcs
    network.add_arc(source, a, cost=0.0, resource_consumption={"time": 1.0})  # 0
    network.add_arc(source, c, cost=0.0, resource_consumption={"time": 1.0})  # 1
    network.add_arc(a, b, cost=10.0, resource_consumption={"time": 2.0})      # 2 - item to cover
    network.add_arc(c, b, cost=5.0, resource_consumption={"time": 2.0})       # 3 - item to cover
    network.add_arc(b, sink, cost=0.0, resource_consumption={"time": 1.0})    # 4

    problem = Problem(
        name="SmallProblem",
        network=network,
        resources=[AccumulatingResource("time", initial=0.0, max_value=10.0)],
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    # Each item must be covered exactly once
    problem.add_cover_constraint(item_id=2, name="arc_a_b")
    problem.add_cover_constraint(item_id=3, name="arc_c_b")

    return problem


def create_problem_with_initial_columns() -> tuple:
    """
    Create a problem with initial columns provided.

    Returns:
        (problem, initial_columns)
    """
    problem = create_small_problem()

    # Create initial columns that form a feasible solution
    col1 = Column(
        arc_indices=(0, 2, 4),  # Source -> A -> B -> Sink
        cost=10.0,
        resource_values={"time": 4.0},
        covered_items=frozenset({2}),
    )
    col2 = Column(
        arc_indices=(1, 3, 4),  # Source -> C -> B -> Sink
        cost=5.0,
        resource_values={"time": 4.0},
        covered_items=frozenset({3}),
    )

    return problem, [col1, col2]


def create_larger_problem() -> Problem:
    """
    Create a larger problem with more paths.

    Network with 4 intermediate nodes and multiple paths.
    """
    network = Network()

    source = network.add_source()
    sink = network.add_sink()

    # Intermediate nodes
    nodes = []
    for i in range(4):
        n = network.add_node(f"N{i}", NodeType.GENERIC)
        nodes.append(n)

    # Arcs from source
    for i, n in enumerate(nodes):
        network.add_arc(source, n, cost=0.0, resource_consumption={"time": 1.0})

    # Arcs between nodes (creates items to cover)
    arc_idx = network.num_arcs
    for i in range(len(nodes) - 1):
        network.add_arc(
            nodes[i], nodes[i + 1],
            cost=float(5 + i * 2),
            resource_consumption={"time": 2.0}
        )

    # Arcs to sink
    for n in nodes:
        network.add_arc(n, sink, cost=0.0, resource_consumption={"time": 1.0})

    problem = Problem(
        name="LargerProblem",
        network=network,
        resources=[AccumulatingResource("time", initial=0.0, max_value=20.0)],
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    # Add cover constraints for the arcs between intermediate nodes
    for i in range(len(nodes) - 1):
        problem.add_cover_constraint(
            item_id=arc_idx + i,
            name=f"arc_{i}_{i+1}"
        )

    return problem


# =============================================================================
# Test CGSolution
# =============================================================================

class TestCGSolution:
    """Tests for CGSolution dataclass."""

    def test_default_status(self):
        """Test default solution status."""
        solution = CGSolution()
        assert solution.status == CGStatus.NOT_SOLVED
        assert solution.objective_value is None
        assert not solution.is_optimal
        assert not solution.is_feasible

    def test_optimal_solution(self):
        """Test optimal solution properties."""
        solution = CGSolution(
            status=CGStatus.OPTIMAL,
            objective_value=100.0,
            lp_objective=100.0,
            columns=[],
            total_columns=50,
            iterations=10,
        )

        assert solution.is_optimal
        assert solution.is_feasible
        assert solution.objective_value == 100.0

    def test_integer_optimal(self):
        """Test integer optimal solution."""
        col = Column(
            arc_indices=(0, 1),
            cost=10.0,
            column_id=0,
        ).with_value(1.0)

        solution = CGSolution(
            status=CGStatus.INTEGER_OPTIMAL,
            objective_value=10.0,
            columns=[col],
        )

        assert solution.is_optimal
        assert solution.is_integer

    def test_convergence_history(self):
        """Test getting convergence history."""
        history = [
            CGIteration(i, 100.0 - i * 5, -1.0, 1, 0.1, 0.2, 10 + i)
            for i in range(5)
        ]

        solution = CGSolution(
            status=CGStatus.OPTIMAL,
            iteration_history=history,
        )

        conv = solution.get_convergence_history()
        assert len(conv) == 5
        assert conv[0] == 100.0
        assert conv[4] == 80.0

    def test_summary(self):
        """Test solution summary."""
        solution = CGSolution(
            status=CGStatus.OPTIMAL,
            objective_value=100.0,
            total_columns=50,
            iterations=10,
            total_time=5.0,
            master_time=3.0,
            pricing_time=2.0,
        )

        summary = solution.summary()
        assert "OPTIMAL" in summary
        assert "100" in summary
        assert "50" in summary


# =============================================================================
# Test CGConfig
# =============================================================================

class TestCGConfig:
    """Tests for CGConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CGConfig()
        assert config.max_iterations == 1000
        assert config.max_time == 3600.0
        assert config.solve_ip is False
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = CGConfig(
            max_iterations=50,
            max_time=60.0,
            solve_ip=True,
            verbose=True,
            use_stabilization=True,
        )

        assert config.max_iterations == 50
        assert config.solve_ip is True
        assert config.use_stabilization is True


# =============================================================================
# Test ColumnGeneration
# =============================================================================

@pytest.mark.skipif(not HIGHS_AVAILABLE, reason="HiGHS not installed")
class TestColumnGeneration:
    """Tests for ColumnGeneration algorithm."""

    def test_creation(self):
        """Test creating column generation instance."""
        problem = create_small_problem()
        cg = ColumnGeneration(problem)

        assert cg.problem is problem
        assert cg.config is not None
        assert not cg.is_solved

    def test_solve_with_artificial_columns(self):
        """Test solving when no initial columns provided."""
        problem = create_small_problem()
        config = CGConfig(verbose=False, max_iterations=100)
        cg = ColumnGeneration(problem, config)

        solution = cg.solve()

        assert cg.is_solved
        assert solution.is_feasible
        # Artificial columns have high cost, so objective should be high
        # unless real columns were found

    def test_solve_with_initial_columns(self):
        """Test solving with initial columns."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False, max_iterations=100)
        cg = ColumnGeneration(problem, config)

        cg.add_initial_columns(initial_columns)
        solution = cg.solve()

        assert solution.is_feasible
        # Optimal should be 10 + 5 = 15 (sum of two paths)
        assert solution.objective_value is not None
        assert solution.objective_value <= 15.0 + 1e-6

    def test_solve_with_ip(self):
        """Test solving with IP phase."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False, solve_ip=True)
        cg = ColumnGeneration(problem, config)

        cg.add_initial_columns(initial_columns)
        solution = cg.solve()

        assert solution.is_feasible
        if solution.status == CGStatus.INTEGER_OPTIMAL:
            assert solution.ip_objective is not None

    def test_iteration_limit(self):
        """Test that iteration limit is respected."""
        problem = create_larger_problem()
        config = CGConfig(max_iterations=2, verbose=False)
        cg = ColumnGeneration(problem, config)

        solution = cg.solve()

        assert solution.iterations <= 2
        # May or may not be optimal depending on problem

    def test_callback(self):
        """Test that callbacks are invoked."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        callback_count = [0]

        def my_callback(cg, iteration):
            callback_count[0] += 1
            return True  # Continue

        cg.add_callback(my_callback)
        cg.add_initial_columns(initial_columns)
        solution = cg.solve()

        assert callback_count[0] > 0

    def test_callback_early_stop(self):
        """Test that callback can stop iteration."""
        problem = create_larger_problem()
        config = CGConfig(verbose=False, max_iterations=100)
        cg = ColumnGeneration(problem, config)

        def stop_callback(cg, iteration):
            return iteration.iteration < 2  # Stop after 2 iterations

        cg.add_callback(stop_callback)
        solution = cg.solve()

        # Should have stopped early
        assert solution.iterations <= 2

    def test_custom_master(self):
        """Test with custom master problem."""
        problem = create_small_problem()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        custom_master = HiGHSMasterProblem(problem, verbosity=0)
        cg.set_master(custom_master)

        solution = cg.solve()
        assert solution.is_feasible

    def test_custom_pricing(self):
        """Test with custom pricing problem."""
        problem = create_small_problem()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        pricing_config = PricingConfig(max_columns=5)
        custom_pricing = LabelingAlgorithm(problem, pricing_config)
        cg.set_pricing(custom_pricing)

        solution = cg.solve()
        assert solution.is_feasible

    def test_reset(self):
        """Test resetting the solver."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        # First solve
        cg.add_initial_columns(initial_columns)
        solution1 = cg.solve()
        assert cg.is_solved

        # Reset and solve again
        cg.reset()
        assert not cg.is_solved

        cg.add_initial_columns(initial_columns)
        solution2 = cg.solve()
        assert cg.is_solved

    def test_solution_columns(self):
        """Test that solution contains correct columns."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        cg.add_initial_columns(initial_columns)
        solution = cg.solve()

        # Should have columns in solution
        if solution.is_feasible and solution.columns:
            for col in solution.columns:
                assert col.value is not None
                assert col.value > 0

    def test_summary(self):
        """Test solver summary."""
        problem = create_small_problem()
        cg = ColumnGeneration(problem)

        summary = cg.summary()
        assert "SmallProblem" in summary
        assert "Max iterations" in summary


# =============================================================================
# Test Integration
# =============================================================================

@pytest.mark.skipif(not HIGHS_AVAILABLE, reason="HiGHS not installed")
class TestCGIntegration:
    """Integration tests for the complete CG algorithm."""

    def test_optimal_solution_found(self):
        """Test that optimal solution is found for a known problem."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        cg.add_initial_columns(initial_columns)
        solution = cg.solve()

        # For this problem, optimal is 15 (col1=10 + col2=5)
        # CG should find this or better
        assert solution.is_optimal
        assert solution.objective_value is not None
        assert solution.objective_value <= 15.0 + 1e-6

    def test_iteration_history(self):
        """Test that iteration history is recorded."""
        problem, initial_columns = create_problem_with_initial_columns()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        cg.add_initial_columns(initial_columns)
        solution = cg.solve()

        history = cg.get_iteration_history()
        assert len(history) > 0

        for iter_info in history:
            assert iter_info.master_objective is not None
            assert iter_info.master_time >= 0
            assert iter_info.pricing_time >= 0

    def test_column_pool_populated(self):
        """Test that column pool contains all generated columns."""
        problem = create_small_problem()
        config = CGConfig(verbose=False)
        cg = ColumnGeneration(problem, config)

        solution = cg.solve()

        # Column pool should have columns
        assert cg.column_pool.size > 0
        assert solution.total_columns == cg.column_pool.size


# =============================================================================
# Run tests standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
