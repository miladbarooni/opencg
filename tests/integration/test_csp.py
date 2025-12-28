"""
Integration tests for Cutting Stock Problem.

These tests verify the full column generation pipeline works correctly
for the Cutting Stock Problem application.
"""

import pytest
from pathlib import Path

from opencg.applications.cutting_stock import CuttingStockInstance, solve_cutting_stock


class TestCuttingStockIntegration:
    """Integration tests for CSP solver."""

    def test_simple_instance(self):
        """Test solving a simple CSP instance."""
        instance = CuttingStockInstance(
            roll_width=100,
            item_sizes=[45, 36, 31, 14],
            item_demands=[97, 610, 395, 211],
            name="simple_csp"
        )

        solution = solve_cutting_stock(instance, max_iterations=50, verbose=False)

        # L2 lower bound
        total_volume = sum(s * d for s, d in zip(instance.item_sizes, instance.item_demands))
        l2 = int((total_volume + instance.roll_width - 1) // instance.roll_width)

        assert solution.lp_objective is not None
        assert solution.num_rolls_ip is not None
        assert solution.num_rolls_ip >= l2, f"IP solution {solution.num_rolls_ip} below L2 bound {l2}"
        assert solution.num_columns > 0

    def test_small_instance(self):
        """Test a small instance with known solution."""
        # Simple instance: 3 item types that fit exactly
        instance = CuttingStockInstance(
            roll_width=10,
            item_sizes=[5, 3, 2],
            item_demands=[10, 10, 10],
            name="small_exact"
        )

        solution = solve_cutting_stock(instance, max_iterations=30, verbose=False)

        assert solution.num_rolls_ip is not None
        # Can fit 2x5=10, or 3+3+2+2=10, or 5+3+2=10
        # With demands [10, 10, 10] and patterns, should need several rolls

    def test_single_item_type(self):
        """Test with single item type."""
        instance = CuttingStockInstance(
            roll_width=100,
            item_sizes=[30],
            item_demands=[10],
            name="single_item"
        )

        solution = solve_cutting_stock(instance, max_iterations=20, verbose=False)

        # 3 items fit per roll (3*30=90), need ceil(10/3) = 4 rolls
        assert solution.num_rolls_ip == 4

    @pytest.mark.skipif(
        not Path("data/bpplib/Instances/Benchmarks/extracted/Scholl_CSP/Scholl_1").exists(),
        reason="BPPLIB data not available"
    )
    def test_bpplib_instance(self):
        """Test loading and solving a BPPLIB instance."""
        bpplib_path = Path("data/bpplib/Instances/Benchmarks/extracted/Scholl_CSP/Scholl_1")
        instances = list(bpplib_path.glob("*.txt"))[:1]

        assert len(instances) > 0, "No BPPLIB instances found"

        inst_path = instances[0]
        instance = CuttingStockInstance.from_bpplib(str(inst_path))

        assert instance.num_items > 0
        assert instance.roll_width > 0

        solution = solve_cutting_stock(instance, max_iterations=50, verbose=False)

        assert solution.lp_objective is not None
        assert solution.num_rolls_ip is not None
