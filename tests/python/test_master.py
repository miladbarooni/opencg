"""
Tests for the master problem module.

This module tests:
- MasterSolution dataclass
- MasterProblem ABC (via HiGHSMasterProblem)
- HiGHSMasterProblem implementation
- Dual stabilization
- Branching support (column bounds)
"""

import pytest
from typing import Dict, List

from opencg.core.column import Column, ColumnPool
from opencg.core.network import Network
from opencg.core.node import NodeType
from opencg.core.arc import ArcType
from opencg.core.resource import AccumulatingResource
from opencg.core.problem import Problem, CoverType, ObjectiveSense, CoverConstraint

from opencg.master import (
    MasterSolution,
    SolutionStatus,
    MasterProblem,
    StabilizationConfig,
    HiGHSMasterProblem,
    HIGHS_AVAILABLE,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_simple_problem() -> Problem:
    """
    Create a simple set partitioning problem for testing.

    This creates a problem with 3 items to cover and we'll add columns
    that cover subsets of these items.

    Items: {0, 1, 2}
    """
    # Create a minimal network
    network = Network()
    source = network.add_source()
    sink = network.add_sink()

    # Add some task nodes
    n0 = network.add_node("task_0", NodeType.GENERIC)
    n1 = network.add_node("task_1", NodeType.GENERIC)
    n2 = network.add_node("task_2", NodeType.GENERIC)

    # Add arcs (simplified - just for creating a valid problem)
    network.add_arc(source, n0, cost=0.0)
    network.add_arc(source, n1, cost=0.0)
    network.add_arc(source, n2, cost=0.0)
    network.add_arc(n0, sink, cost=0.0)
    network.add_arc(n1, sink, cost=0.0)
    network.add_arc(n2, sink, cost=0.0)

    # Create problem
    problem = Problem(
        name="TestProblem",
        network=network,
        resources=[AccumulatingResource("time", initial=0.0, max_value=10.0)],
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    # Add cover constraints for 3 items
    problem.add_cover_constraint(item_id=0, name="item_0")
    problem.add_cover_constraint(item_id=1, name="item_1")
    problem.add_cover_constraint(item_id=2, name="item_2")

    return problem


def create_columns_for_simple_problem() -> List[Column]:
    """
    Create columns for the simple problem.

    We create columns that form a valid set partitioning:
    - Column 0: covers {0, 1}, cost=10
    - Column 1: covers {2}, cost=5
    - Column 2: covers {0}, cost=6
    - Column 3: covers {1, 2}, cost=8

    Optimal solution: Column 2 + Column 3 = cost 14
    (covers {0} + {1,2} = {0,1,2})

    Alternative: Column 0 + Column 1 = cost 15
    (covers {0,1} + {2} = {0,1,2})
    """
    columns = [
        Column(
            arc_indices=(0, 3),  # Dummy arc sequence
            cost=10.0,
            covered_items=frozenset({0, 1}),
            column_id=0,
        ),
        Column(
            arc_indices=(2, 5),
            cost=5.0,
            covered_items=frozenset({2}),
            column_id=1,
        ),
        Column(
            arc_indices=(0, 3),
            cost=6.0,
            covered_items=frozenset({0}),
            column_id=2,
        ),
        Column(
            arc_indices=(1, 4, 5),
            cost=8.0,
            covered_items=frozenset({1, 2}),
            column_id=3,
        ),
    ]
    return columns


def create_set_covering_problem() -> Problem:
    """Create a set covering problem (>= constraints)."""
    network = Network()
    source = network.add_source()
    sink = network.add_sink()
    n0 = network.add_node("task_0", NodeType.GENERIC)
    network.add_arc(source, n0, cost=0.0)
    network.add_arc(n0, sink, cost=0.0)

    problem = Problem(
        name="CoveringProblem",
        network=network,
        resources=[AccumulatingResource("time", initial=0.0, max_value=10.0)],
        cover_type=CoverType.SET_COVERING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    problem.add_cover_constraint(item_id=0, name="item_0", is_equality=False)
    problem.add_cover_constraint(item_id=1, name="item_1", is_equality=False)

    return problem


# =============================================================================
# Test MasterSolution
# =============================================================================

class TestMasterSolution:
    """Tests for MasterSolution dataclass."""

    def test_default_status(self):
        """Test default solution status."""
        solution = MasterSolution()
        assert solution.status == SolutionStatus.NOT_SOLVED
        assert solution.objective_value is None
        assert not solution.is_optimal
        assert not solution.has_solution

    def test_optimal_solution(self):
        """Test optimal solution properties."""
        solution = MasterSolution(
            status=SolutionStatus.OPTIMAL,
            objective_value=100.0,
            column_values={0: 1.0, 1: 0.5},
            dual_values={0: 10.0, 1: 15.0},
        )

        assert solution.is_optimal
        assert solution.has_solution
        assert solution.objective_value == 100.0
        assert not solution.is_integer  # 0.5 is fractional

    def test_integer_solution(self):
        """Test integer solution detection."""
        solution = MasterSolution(
            status=SolutionStatus.OPTIMAL,
            objective_value=100.0,
            column_values={0: 1.0, 1: 1.0, 2: 0.0},
        )
        assert solution.is_integer

    def test_fractional_solution(self):
        """Test fractional solution detection."""
        solution = MasterSolution(
            status=SolutionStatus.OPTIMAL,
            objective_value=100.0,
            column_values={0: 0.5, 1: 0.5},
        )
        assert not solution.is_integer
        assert solution.get_fractional_columns() == [0, 1]

    def test_get_active_columns(self):
        """Test getting active columns."""
        solution = MasterSolution(
            status=SolutionStatus.OPTIMAL,
            column_values={0: 1.0, 1: 0.0, 2: 0.5, 3: 1e-10},
        )
        active = solution.get_active_columns()
        assert 0 in active
        assert 2 in active
        assert 1 not in active
        assert 3 not in active  # Below tolerance

    def test_get_dual(self):
        """Test getting dual values."""
        solution = MasterSolution(
            dual_values={0: 10.0, 1: 15.0}
        )
        assert solution.get_dual(0) == 10.0
        assert solution.get_dual(1) == 15.0
        assert solution.get_dual(2, default=0.0) == 0.0

    def test_infeasible_status(self):
        """Test infeasible solution."""
        solution = MasterSolution(status=SolutionStatus.INFEASIBLE)
        assert solution.is_infeasible
        assert not solution.is_optimal
        assert not solution.has_solution

    def test_summary(self):
        """Test solution summary."""
        solution = MasterSolution(
            status=SolutionStatus.OPTIMAL,
            objective_value=100.0,
            column_values={0: 1.0},
            solve_time=0.5,
            iterations=10,
            num_columns=5,
        )
        summary = solution.summary()
        assert "OPTIMAL" in summary
        assert "100" in summary


# =============================================================================
# Test HiGHSMasterProblem
# =============================================================================

@pytest.mark.skipif(not HIGHS_AVAILABLE, reason="HiGHS not installed")
class TestHiGHSMasterProblem:
    """Tests for HiGHSMasterProblem."""

    def test_creation(self):
        """Test creating a master problem."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        assert master.num_columns == 0
        assert master.num_constraints == 3
        assert master.problem is problem

    def test_add_column(self):
        """Test adding columns."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        for col in columns:
            master.add_column(col)

        assert master.num_columns == 4

    def test_solve_lp(self):
        """Test solving LP relaxation."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)
        solution = master.solve_lp()

        assert solution.is_optimal
        assert solution.objective_value is not None
        # Optimal LP should be <= 14 (the IP optimal)
        assert solution.objective_value <= 14.0 + 1e-6

    def test_solve_ip(self):
        """Test solving as IP."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)
        solution = master.solve_ip()

        assert solution.is_optimal
        assert solution.is_integer
        # Optimal: Column 2 (cost 6) + Column 3 (cost 8) = 14
        assert abs(solution.objective_value - 14.0) < 1e-6

    def test_get_dual_values(self):
        """Test extracting dual values."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)
        master.solve_lp()

        duals = master.get_dual_values()

        # Should have duals for all 3 items
        assert len(duals) == 3
        assert 0 in duals
        assert 1 in duals
        assert 2 in duals

    def test_reduced_cost_computation(self):
        """Test reduced cost computation."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)
        master.solve_lp()

        duals = master.get_dual_values()

        # Compute reduced cost for column 0 (covers {0, 1}, cost=10)
        col = columns[0]
        rc = master.compute_reduced_cost(col, duals)

        # rc = c_j - sum(pi_i * a_ij) = 10 - (pi_0 + pi_1)
        expected_rc = 10.0 - duals[0] - duals[1]
        assert abs(rc - expected_rc) < 1e-6

    def test_column_bounds(self):
        """Test setting column bounds for branching."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)

        # Fix column 0 to 0 (don't use it)
        master.fix_column(0, 0.0)

        solution = master.solve_ip()

        assert solution.is_optimal
        # Column 0 should not be in solution
        assert solution.column_values.get(0, 0.0) < 1e-6

    def test_set_covering(self):
        """Test set covering formulation."""
        problem = create_set_covering_problem()
        master = HiGHSMasterProblem(problem)

        # Add columns that cover items
        col1 = Column(arc_indices=(0,), cost=5.0, covered_items=frozenset({0}), column_id=0)
        col2 = Column(arc_indices=(1,), cost=3.0, covered_items=frozenset({1}), column_id=1)
        col3 = Column(arc_indices=(0, 1), cost=6.0, covered_items=frozenset({0, 1}), column_id=2)

        master.add_columns([col1, col2, col3])
        solution = master.solve_lp()

        assert solution.is_optimal
        # Minimum cost to cover both: col1 + col2 = 8, or col3 = 6
        # Actually for set covering, col3 alone covers both
        assert solution.objective_value <= 8.0 + 1e-6

    def test_warm_starting(self):
        """Test warm starting with basis."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)

        # Solve and get basis
        solution1 = master.solve_lp()
        basis = solution1.basis

        # Add a new column
        new_col = Column(
            arc_indices=(0, 1, 2),
            cost=20.0,
            covered_items=frozenset({0, 1, 2}),
            column_id=4,
        )
        master.add_column(new_col)

        # Warm start with previous basis (basis won't include new column)
        # This tests that the solver handles partial basis
        solution2 = master.solve_lp()

        assert solution2.is_optimal

    def test_model_stats(self):
        """Test getting model statistics."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)

        stats = master.get_model_stats()

        assert stats['num_columns'] == 4
        assert stats['num_rows'] == 3

    def test_summary(self):
        """Test master problem summary."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        summary = master.summary()

        assert "TestProblem" in summary
        assert "Columns:" in summary
        assert "Constraints:" in summary


# =============================================================================
# Test Dual Stabilization
# =============================================================================

@pytest.mark.skipif(not HIGHS_AVAILABLE, reason="HiGHS not installed")
class TestDualStabilization:
    """Tests for dual stabilization."""

    def test_stabilization_config(self):
        """Test stabilization configuration."""
        config = StabilizationConfig()
        assert not config.enabled
        assert config.method == 'none'

    def test_enable_boxstep(self):
        """Test enabling boxstep stabilization."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        master.enable_stabilization('boxstep', delta=5.0, shrink=0.5)

        assert master.stabilization.enabled
        assert master.stabilization.method == 'boxstep'
        assert master.stabilization.boxstep_delta == 5.0

    def test_enable_smoothing(self):
        """Test enabling smoothing stabilization."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        master.enable_stabilization('smoothing', alpha=0.7)

        assert master.stabilization.enabled
        assert master.stabilization.method == 'smoothing'
        assert master.stabilization.smoothing_alpha == 0.7

    def test_boxstep_bounds_duals(self):
        """Test that boxstep bounds the duals."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)
        columns = create_columns_for_simple_problem()

        master.add_columns(columns)

        # Set up stabilization with a center
        master.enable_stabilization('boxstep', delta=2.0)

        # Solve to get initial duals
        master.solve_lp()
        raw_duals = master.get_raw_dual_values()

        # Set center to zeros
        master.update_stabilization_center({0: 0.0, 1: 0.0, 2: 0.0})

        # Now stabilized duals should be bounded by [-2, 2]
        master.solve_lp()
        stabilized_duals = master.get_dual_values()

        for item_id, dual in stabilized_duals.items():
            assert -2.0 <= dual <= 2.0

    def test_shrink_stabilization(self):
        """Test shrinking stabilization region."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        master.enable_stabilization('boxstep', delta=10.0, shrink=0.5)
        assert master.stabilization.boxstep_delta == 10.0

        master.shrink_stabilization()
        assert master.stabilization.boxstep_delta == 5.0

        master.shrink_stabilization()
        assert master.stabilization.boxstep_delta == 2.5

    def test_disable_stabilization(self):
        """Test disabling stabilization."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        master.enable_stabilization('boxstep')
        assert master.stabilization.enabled

        master.disable_stabilization()
        assert not master.stabilization.enabled


# =============================================================================
# Test Edge Cases
# =============================================================================

@pytest.mark.skipif(not HIGHS_AVAILABLE, reason="HiGHS not installed")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_master(self):
        """Test solving with no columns."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        # Solving with no columns should be infeasible (or error for empty model)
        solution = master.solve_lp()

        # HiGHS might return unbounded, infeasible, or error for empty problem
        assert solution.status in (
            SolutionStatus.INFEASIBLE,
            SolutionStatus.UNBOUNDED,
            SolutionStatus.INF_OR_UNBOUNDED,
            SolutionStatus.ERROR,  # HiGHS may consider empty model an error
        )

    def test_column_without_id(self):
        """Test that adding column without ID raises error."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        col = Column(
            arc_indices=(0,),
            cost=5.0,
            covered_items=frozenset({0}),
            # No column_id!
        )

        with pytest.raises(ValueError, match="column_id"):
            master.add_column(col)

    def test_get_nonexistent_column(self):
        """Test getting a column that doesn't exist."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem)

        col = master.get_column(999)
        assert col is None

    def test_verbosity(self):
        """Test setting verbosity."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem, verbosity=0)

        master.set_verbosity(1)
        # Just ensure it doesn't crash

    def test_time_limit(self):
        """Test setting time limit."""
        problem = create_simple_problem()
        master = HiGHSMasterProblem(problem, time_limit=10.0)

        master.set_time_limit(5.0)
        # Just ensure it doesn't crash


# =============================================================================
# Run tests standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
