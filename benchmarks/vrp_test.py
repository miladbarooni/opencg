#!/usr/bin/env python3
"""Test script for VRP solver."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from opencg.applications.vrp import CVRPInstance, solve_cvrp, CVRPConfig


def test_small_instance():
    """Test with a small hand-crafted instance."""
    print("=" * 60)
    print("Test: Small CVRP Instance")
    print("=" * 60)

    # Simple instance: 5 customers around a depot
    # Depot at origin, customers in a rough circle
    instance = CVRPInstance(
        depot=(0, 0),
        customers=[
            (10, 0),   # Customer 0: east
            (0, 10),   # Customer 1: north
            (-10, 0),  # Customer 2: west
            (0, -10),  # Customer 3: south
            (5, 5),    # Customer 4: northeast
        ],
        demands=[20, 15, 25, 10, 30],  # Total: 100
        vehicle_capacity=50,  # Need at least 2 vehicles
        name="small_test"
    )

    print(f"Instance: {instance}")
    print(f"Min vehicles (capacity bound): {instance.min_vehicles}")

    config = CVRPConfig(verbose=True, max_iterations=50)
    solution = solve_cvrp(instance, config)

    print(f"\nSolution:")
    print(f"  Total distance (LP): {solution.total_distance:.2f}")
    print(f"  Total distance (IP): {solution.total_distance_ip:.2f}")
    print(f"  Vehicles used: {solution.num_vehicles}")
    print(f"  Routes:")
    all_customers = set()
    for i, route in enumerate(solution.routes):
        route_demand = sum(instance.demands[c] for c in route)
        print(f"    Route {i+1}: depot -> {route} -> depot (demand: {route_demand})")
        for c in route:
            if c in all_customers:
                print(f"    WARNING: Customer {c} visited multiple times!")
            all_customers.add(c)

    # Check coverage
    missing = set(range(instance.num_customers)) - all_customers
    if missing:
        print(f"  WARNING: Customers not covered: {missing}")
    print(f"  Solve time: {solution.solve_time:.2f}s")
    print(f"  Iterations: {solution.iterations}")
    print(f"  Columns: {solution.num_columns}")


def test_medium_instance():
    """Test with a medium-sized random instance."""
    import random
    import math

    print("\n" + "=" * 60)
    print("Test: Medium CVRP Instance (15 customers)")
    print("=" * 60)

    random.seed(42)
    n_customers = 15

    # Random customers in a 100x100 grid
    customers = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_customers)]
    demands = [random.randint(5, 25) for _ in range(n_customers)]

    instance = CVRPInstance(
        depot=(50, 50),  # Center
        customers=customers,
        demands=demands,
        vehicle_capacity=100,
        name="medium_random"
    )

    print(f"Instance: {instance}")
    print(f"Total demand: {instance.total_demand}")
    print(f"Min vehicles: {instance.min_vehicles}")

    config = CVRPConfig(verbose=True, max_iterations=100)
    solution = solve_cvrp(instance, config)

    print(f"\nSolution:")
    print(f"  Total distance (LP): {solution.total_distance:.2f}")
    print(f"  Total distance (IP): {solution.total_distance_ip:.2f}")
    print(f"  Vehicles used: {solution.num_vehicles}")
    print(f"  Solve time: {solution.solve_time:.2f}s")


if __name__ == "__main__":
    test_small_instance()
    test_medium_instance()
