#!/usr/bin/env python3
"""Test script for VRPTW solver."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from opencg.applications.vrp import VRPTWInstance, solve_vrptw, VRPTWConfig


def test_small_instance():
    """Test with a small hand-crafted instance."""
    print("=" * 60)
    print("Test: Small VRPTW Instance")
    print("=" * 60)

    # Simple instance: depot at origin, customers with time windows
    # Planning horizon: 0 to 200
    instance = VRPTWInstance(
        depot=(0, 0),
        customers=[
            (10, 0),   # Customer 0: east
            (0, 10),   # Customer 1: north
            (-10, 0),  # Customer 2: west
            (0, -10),  # Customer 3: south
            (5, 5),    # Customer 4: northeast
        ],
        demands=[20, 15, 25, 10, 30],  # Total: 100
        time_windows=[
            (0, 50),    # Customer 0: early deadline
            (20, 80),   # Customer 1: medium
            (40, 120),  # Customer 2: late
            (0, 100),   # Customer 3: flexible
            (10, 60),   # Customer 4: medium
        ],
        service_times=[10, 10, 10, 10, 10],
        vehicle_capacity=50,
        depot_time_window=(0, 200),
        name="small_vrptw"
    )

    print(f"Instance: {instance}")
    print(f"Min vehicles (capacity bound): {instance.min_vehicles}")
    print()
    print("Customers:")
    for i in range(instance.num_customers):
        tw = instance.time_windows[i]
        print(f"  {i}: loc={instance.customers[i]}, demand={instance.demands[i]}, "
              f"TW=[{tw[0]}, {tw[1]}], service={instance.service_times[i]}")

    config = VRPTWConfig(verbose=True, max_iterations=50)
    solution = solve_vrptw(instance, config)

    print(f"\nSolution:")
    print(f"  Total distance (LP): {solution.total_distance:.2f}")
    print(f"  Total distance (IP): {solution.total_distance_ip:.2f}")
    print(f"  Vehicles used: {solution.num_vehicles}")
    print(f"  Routes:")

    all_customers = set()
    for i, route in enumerate(solution.routes):
        route_demand = sum(instance.demands[c] for c in route)
        print(f"    Route {i+1}: depot -> {route} -> depot (demand: {route_demand})")

        # Verify time windows
        current_time = 0
        current_loc = 0
        feasible = True
        for c in route:
            travel_time = instance.travel_time(current_loc, c + 1)
            arrival = current_time + travel_time
            earliest, latest = instance.time_windows[c]
            if arrival > latest:
                print(f"      WARNING: Late arrival at customer {c}: {arrival:.1f} > {latest}")
                feasible = False
            start_service = max(arrival, earliest)
            current_time = start_service + instance.service_times[c]
            current_loc = c + 1

            if c in all_customers:
                print(f"      WARNING: Customer {c} visited multiple times!")
            all_customers.add(c)

        # Check return to depot
        return_time = current_time + instance.travel_time(current_loc, 0)
        _, depot_latest = instance.depot_time_window
        if return_time > depot_latest:
            print(f"      WARNING: Late return to depot: {return_time:.1f} > {depot_latest}")
            feasible = False

        if feasible:
            print(f"      (Time feasible, return at {return_time:.1f})")

    # Check coverage
    missing = set(range(instance.num_customers)) - all_customers
    if missing:
        print(f"  WARNING: Customers not covered: {missing}")
    else:
        print(f"  All {instance.num_customers} customers covered")

    print(f"  Solve time: {solution.solve_time:.2f}s")
    print(f"  Iterations: {solution.iterations}")
    print(f"  Columns: {solution.num_columns}")


def test_tight_time_windows():
    """Test with tight time windows that require more vehicles."""
    print("\n" + "=" * 60)
    print("Test: Tight Time Windows")
    print("=" * 60)

    # Customers at distance ~14 from depot, with non-overlapping tight time windows
    # Two early customers (TW 0-25) and two late customers (TW 50-75)
    # The gap between TWs means early and late customers can't be combined efficiently
    instance = VRPTWInstance(
        depot=(0, 0),
        customers=[
            (10, 10),    # Customer 0: NE, ~14 from depot
            (-10, 10),   # Customer 1: NW, ~14 from depot
            (10, -10),   # Customer 2: SE, ~14 from depot
            (-10, -10),  # Customer 3: SW, ~14 from depot
        ],
        demands=[10, 10, 10, 10],  # Low demands
        time_windows=[
            (0, 25),     # Customer 0: early (can reach, travel ~14)
            (0, 25),     # Customer 1: early
            (50, 75),    # Customer 2: late
            (50, 75),    # Customer 3: late
        ],
        service_times=[5, 5, 5, 5],
        vehicle_capacity=100,  # High capacity (not the constraint)
        depot_time_window=(0, 200),
        name="tight_tw"
    )

    print(f"Instance: {instance}")
    print("Note: Two early customers (TW 0-25) and two late customers (TW 50-75)")

    config = VRPTWConfig(verbose=True, max_iterations=50)
    solution = solve_vrptw(instance, config)

    print(f"\nSolution:")
    print(f"  Vehicles used: {solution.num_vehicles}")
    print(f"  Total distance (IP): {solution.total_distance_ip:.2f}")

    for i, route in enumerate(solution.routes):
        print(f"  Route {i+1}: {route}")


def test_solomon_instance():
    """Test loading a Solomon instance if available."""
    print("\n" + "=" * 60)
    print("Test: Solomon Instance (if available)")
    print("=" * 60)

    # Try to find a Solomon instance
    data_path = Path(__file__).parent / "data" / "solomon"

    if not data_path.exists():
        print(f"Solomon data not found at {data_path}")
        print("You can download Solomon instances from:")
        print("  https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/")
        return

    instance_files = list(data_path.glob("*.txt"))
    if not instance_files:
        print("No Solomon instance files found")
        return

    # Load first instance
    instance_file = instance_files[0]
    print(f"Loading: {instance_file.name}")

    instance = VRPTWInstance.from_solomon(str(instance_file))
    print(f"Instance: {instance}")
    print(f"Customers: {instance.num_customers}")
    print(f"Vehicle capacity: {instance.vehicle_capacity}")
    print(f"Depot time window: {instance.depot_time_window}")

    config = VRPTWConfig(verbose=True, max_iterations=100)
    solution = solve_vrptw(instance, config)

    print(f"\nSolution:")
    print(f"  Total distance (LP): {solution.total_distance:.2f}")
    print(f"  Total distance (IP): {solution.total_distance_ip:.2f}")
    print(f"  Vehicles used: {solution.num_vehicles}")
    print(f"  Solve time: {solution.solve_time:.2f}s")


if __name__ == "__main__":
    test_small_instance()
    test_tight_time_windows()
    test_solomon_instance()
