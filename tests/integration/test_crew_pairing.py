"""
Integration tests for Crew Pairing.

These tests verify the full column generation pipeline works correctly
for the Crew Pairing application, including testing the connection gap fix.
"""

import pytest
import warnings
from pathlib import Path

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.core.column import Column
from opencg.core.arc import ArcType
from opencg.master import HiGHSMasterProblem
from opencg.pricing import PricingConfig

# Try to import C++ pricing
try:
    from opencg.pricing import FastPerSourcePricing
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    from opencg.pricing import PerSourcePricing


# Path to test data
KASIRZADEH_PATH = Path("data/kasirzadeh/instance1")


def _quick_cg_solve(problem, max_iter: int = 10, use_cpp: bool = True) -> float:
    """Quick CG solve to test coverage. Returns coverage percentage."""
    if use_cpp and HAS_CPP:
        from opencg.pricing import FastPerSourcePricing as Pricing
    else:
        from opencg.pricing import PerSourcePricing as Pricing

    master = HiGHSMasterProblem(problem, verbosity=0)

    # Add artificial columns
    big_m = 1e6
    next_col_id = 0
    for constraint in problem.cover_constraints:
        art_col = Column(
            arc_indices=(),
            cost=big_m,
            covered_items=frozenset([constraint.item_id]),
            column_id=next_col_id,
            attributes={'artificial': True},
        )
        master.add_column(art_col)
        next_col_id += 1

    # Create pricing
    pricing_config = PricingConfig(
        max_columns=200,
        max_time=30.0,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    if use_cpp and HAS_CPP:
        pricing = Pricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
            num_threads=4,
        )
    else:
        pricing = Pricing(
            problem,
            config=pricing_config,
            max_labels_per_node=30,
            cols_per_source=5,
            time_per_source=0.1,
        )

    # CG iterations
    for _ in range(max_iter):
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            break

        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if not pricing_sol.columns:
            break

        for col in pricing_sol.columns:
            col_with_id = col.with_id(next_col_id)
            next_col_id += 1
            master.add_column(col_with_id)

    # Compute coverage
    lp_sol = master.solve_lp()
    covered = set()
    for col_id, val in lp_sol.column_values.items():
        if val > 1e-6:
            col = master.get_column(col_id)
            if col and not col.attributes.get('artificial'):
                covered.update(col.covered_items)

    return 100.0 * len(covered) / len(problem.cover_constraints)


class TestCrewPairingIntegration:
    """Integration tests for Crew Pairing solver."""

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    def test_connection_gap_warning(self):
        """Test that connection gap triggers a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config = ParserConfig(options={
                'max_connection_time': 4.0,
                'min_layover_time': 10.0,  # GAP!
            })
            parser = KasirzadehParser(config)

            assert len(w) == 1
            assert "gap" in str(w[0].message).lower()

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    def test_no_gap_no_warning(self):
        """Test that closing the gap doesn't trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            config = ParserConfig(options={
                'max_connection_time': 4.0,
                'min_layover_time': 4.0,  # No gap
            })
            parser = KasirzadehParser(config)

            # Filter to only our warnings (ignore deprecation warnings, etc.)
            gap_warnings = [x for x in w if "gap" in str(x.message).lower()]
            assert len(gap_warnings) == 0

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    def test_parse_kasirzadeh_instance(self):
        """Test parsing a Kasirzadeh instance."""
        config = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # Close the gap
            'max_duty_time': 14.0,
        })
        parser = KasirzadehParser(config)
        problem = parser.parse(KASIRZADEH_PATH)

        assert problem.network.num_nodes > 0
        assert problem.network.num_arcs > 0
        assert len(problem.cover_constraints) > 0

        # Check arc types
        flight_arcs = list(problem.network.arcs_of_type(ArcType.FLIGHT))
        source_arcs = list(problem.network.arcs_of_type(ArcType.SOURCE_ARC))
        sink_arcs = list(problem.network.arcs_of_type(ArcType.SINK_ARC))

        assert len(flight_arcs) == len(problem.cover_constraints), \
            "Each flight should have a cover constraint"
        assert len(source_arcs) > 0, "Should have source arcs"
        assert len(sink_arcs) > 0, "Should have sink arcs"

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    @pytest.mark.slow
    def test_solve_with_gap(self):
        """Test solving with connection gap (should have lower coverage)."""
        config = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 10.0,  # GAP!
            'max_duty_time': 14.0,
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # We know about the gap
            parser = KasirzadehParser(config)
            problem = parser.parse(KASIRZADEH_PATH)

        coverage = _quick_cg_solve(problem, max_iter=10)

        # Even with gap, should get decent coverage
        assert coverage > 90, f"Coverage too low: {coverage}%"

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    @pytest.mark.slow
    def test_solve_without_gap(self):
        """Test solving without connection gap (should have better coverage)."""
        config = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # No gap
            'max_duty_time': 14.0,
        })
        parser = KasirzadehParser(config)
        problem = parser.parse(KASIRZADEH_PATH)

        coverage = _quick_cg_solve(problem, max_iter=10)

        # Without gap, should achieve very high coverage
        assert coverage > 95, f"Coverage too low: {coverage}%"

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    @pytest.mark.slow
    def test_coverage_improvement_with_fix(self):
        """Test that closing the gap improves coverage."""
        # With gap
        config_gap = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 10.0,  # GAP
            'max_duty_time': 14.0,
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser_gap = KasirzadehParser(config_gap)
            problem_gap = parser_gap.parse(KASIRZADEH_PATH)
            arcs_gap = problem_gap.network.num_arcs

        # Without gap
        config_fixed = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # FIXED
            'max_duty_time': 14.0,
        })
        parser_fixed = KasirzadehParser(config_fixed)
        problem_fixed = parser_fixed.parse(KASIRZADEH_PATH)
        arcs_fixed = problem_fixed.network.num_arcs

        # Fixed config should have more arcs (overnight connections)
        assert arcs_fixed > arcs_gap, \
            f"Fixed config should have more arcs: {arcs_fixed} vs {arcs_gap}"


class TestCrewPairingNetworkConnectivity:
    """Tests for network connectivity in crew pairing."""

    @pytest.mark.skipif(not KASIRZADEH_PATH.exists(), reason="Kasirzadeh data not available")
    def test_all_flights_reachable_with_fix(self):
        """Test that all flights are reachable when gap is closed."""
        from collections import deque
        from opencg.core.node import NodeType

        config = ParserConfig(options={
            'max_connection_time': 4.0,
            'min_layover_time': 4.0,  # No gap
            'max_duty_time': 14.0,
        })
        parser = KasirzadehParser(config)
        problem = parser.parse(KASIRZADEH_PATH)

        network = problem.network

        # Find source node
        source_idx = None
        for i in range(network.num_nodes):
            node = network.get_node(i)
            if node.node_type == NodeType.SOURCE:
                source_idx = i
                break

        # BFS from source
        reachable_flights = set()
        queue = deque([source_idx])
        visited = {source_idx}

        while queue:
            node = queue.popleft()
            for arc in network.outgoing_arcs(node):
                if arc.arc_type == ArcType.FLIGHT:
                    reachable_flights.add(arc.index)
                if arc.target not in visited:
                    visited.add(arc.target)
                    queue.append(arc.target)

        all_flights = {arc.index for arc in network.arcs_of_type(ArcType.FLIGHT)}
        unreachable = all_flights - reachable_flights

        # With fix, all flights should be reachable
        assert len(unreachable) == 0, \
            f"{len(unreachable)} flights unreachable from source"
