"""
Vehicle Routing Problems via Column Generation.

This module provides solvers for:
- CVRP: Capacitated Vehicle Routing Problem
- VRPTW: VRP with Time Windows

CVRP
----
Given customers with demands and vehicles with capacity, find minimum cost routes.

VRPTW
-----
Extends CVRP with time windows: each customer must be served within [earliest, latest].
Vehicles can wait if arriving early but cannot arrive late.

Usage:
------
    # CVRP
    from opencg.applications.vrp import CVRPInstance, solve_cvrp

    instance = CVRPInstance(
        depot=(0, 0),
        customers=[(10, 20), (30, 40), ...],
        demands=[5, 10, ...],
        vehicle_capacity=100,
    )
    solution = solve_cvrp(instance)

    # VRPTW
    from opencg.applications.vrp import VRPTWInstance, solve_vrptw

    instance = VRPTWInstance(
        depot=(0, 0),
        customers=[(10, 20), (30, 40), ...],
        demands=[5, 10, ...],
        time_windows=[(0, 100), (50, 150), ...],  # (earliest, latest)
        service_times=[10, 10, ...],
        vehicle_capacity=100,
    )
    solution = solve_vrptw(instance)

    # Or load from Solomon benchmark
    instance = VRPTWInstance.from_solomon("path/to/C101.txt")
"""

from opencg.applications.vrp.instance import CVRPInstance, VRPTWInstance
from opencg.applications.vrp.resources import CapacityResource, TimeResource
from opencg.applications.vrp.solver import (
    solve_cvrp, CVRPSolution, CVRPConfig,
    solve_vrptw, VRPTWSolution, VRPTWConfig,
)

__all__ = [
    # CVRP
    'CVRPInstance',
    'CapacityResource',
    'solve_cvrp',
    'CVRPSolution',
    'CVRPConfig',
    # VRPTW
    'VRPTWInstance',
    'TimeResource',
    'solve_vrptw',
    'VRPTWSolution',
    'VRPTWConfig',
]
