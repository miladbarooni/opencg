"""
Integration tests for Vehicle Routing with Time Windows.

These tests verify the full column generation pipeline works correctly
for the VRPTW application.
"""

import pytest

from opencg.applications.vrp import VRPTWInstance, solve_vrptw, VRPTWConfig


class TestVRPTWIntegration:
    """Integration tests for VRPTW solver."""

    def test_small_instance(self):
        """Test solving a small VRPTW instance."""
        instance = VRPTWInstance(
            depot=(0, 0),
            customers=[
                (10, 0), (0, 10), (-10, 0), (0, -10), (5, 5)
            ],
            demands=[20, 15, 25, 10, 30],
            time_windows=[
                (0, 50), (20, 80), (40, 120), (0, 100), (10, 60)
            ],
            service_times=[10, 10, 10, 10, 10],
            vehicle_capacity=50,
            depot_time_window=(0, 200),
            name="small_vrptw"
        )

        config = VRPTWConfig(verbose=False, max_iterations=50)
        solution = solve_vrptw(instance, config)

        # Check all customers covered
        all_customers = set()
        for route in solution.routes:
            all_customers.update(route)

        assert len(all_customers) == instance.num_customers, \
            f"Not all customers covered: {len(all_customers)}/{instance.num_customers}"

        assert solution.total_distance_ip > 0
        assert solution.num_vehicles > 0
        assert solution.num_vehicles <= instance.num_customers  # At most one customer per vehicle

    def test_tight_time_windows(self):
        """Test with tight time windows requiring more vehicles."""
        instance = VRPTWInstance(
            depot=(0, 0),
            customers=[
                (10, 10), (-10, 10), (10, -10), (-10, -10)
            ],
            demands=[10, 10, 10, 10],
            time_windows=[
                (0, 25), (0, 25), (50, 75), (50, 75)  # Two early, two late
            ],
            service_times=[5, 5, 5, 5],
            vehicle_capacity=100,
            depot_time_window=(0, 200),
            name="tight_tw"
        )

        config = VRPTWConfig(verbose=False, max_iterations=50)
        solution = solve_vrptw(instance, config)

        all_customers = set()
        for route in solution.routes:
            all_customers.update(route)

        assert len(all_customers) == 4, "All 4 customers must be covered"
        # Should need at least 2 vehicles due to time window separation

    def test_capacity_constraint(self):
        """Test that capacity constraints are respected."""
        instance = VRPTWInstance(
            depot=(0, 0),
            customers=[
                (10, 0), (20, 0), (30, 0)
            ],
            demands=[40, 40, 40],  # Each customer needs 40
            time_windows=[
                (0, 100), (0, 100), (0, 100)
            ],
            service_times=[0, 0, 0],
            vehicle_capacity=50,  # Can only serve one customer per vehicle
            depot_time_window=(0, 200),
            name="capacity_test"
        )

        config = VRPTWConfig(verbose=False, max_iterations=50)
        solution = solve_vrptw(instance, config)

        # With capacity 50 and demands 40, need 3 vehicles
        assert solution.num_vehicles >= 3, \
            f"Expected at least 3 vehicles, got {solution.num_vehicles}"

        # Verify each route respects capacity
        for route in solution.routes:
            route_demand = sum(instance.demands[c] for c in route)
            assert route_demand <= instance.vehicle_capacity, \
                f"Route {route} exceeds capacity: {route_demand} > {instance.vehicle_capacity}"

    def test_single_customer(self):
        """Test trivial instance with single customer."""
        instance = VRPTWInstance(
            depot=(0, 0),
            customers=[(10, 0)],
            demands=[10],
            time_windows=[(0, 100)],
            service_times=[5],
            vehicle_capacity=100,
            depot_time_window=(0, 200),
            name="single_customer"
        )

        config = VRPTWConfig(verbose=False, max_iterations=20)
        solution = solve_vrptw(instance, config)

        assert solution.num_vehicles == 1
        assert len(solution.routes) == 1
        assert solution.routes[0] == [0]  # Only customer 0
