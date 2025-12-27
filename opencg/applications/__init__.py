"""
Application-specific implementations for common optimization problems.

This package provides ready-to-use implementations for standard problems:
- Cutting Stock Problem (1D bin packing)
- Crew Pairing Problem (airline crew scheduling)
- Vehicle Routing Problem (CVRP)

Each application provides:
- Problem definition helpers
- Specialized pricing algorithms
- Easy-to-use solve functions
- Example usage

Usage:
------
Cutting Stock:
    from opencg.applications import CuttingStockInstance, solve_cutting_stock
    instance = CuttingStockInstance(roll_width=100, item_sizes=[45, 36], item_demands=[10, 20])
    solution = solve_cutting_stock(instance)

Crew Pairing:
    from opencg.applications import solve_crew_pairing, CrewPairingConfig
    from opencg.parsers import KasirzadehParser
    problem = KasirzadehParser().parse("data/kasirzadeh/instance1")
    solution = solve_crew_pairing(problem)

Vehicle Routing (CVRP):
    from opencg.applications import CVRPInstance, solve_cvrp
    instance = CVRPInstance(depot=(0,0), customers=[(10,20),...], demands=[5,...], vehicle_capacity=100)
    solution = solve_cvrp(instance)
"""

# Cutting Stock Problem
from opencg.applications.cutting_stock import (
    CuttingStockInstance,
    CuttingStockPricing,
    CuttingStockMaster,
    CuttingStockSolution,
    solve_cutting_stock,
)

# Crew Pairing Problem
from opencg.applications.crew_pairing import (
    solve_crew_pairing,
    CrewPairingConfig,
    CrewPairingSolution,
    HomeBaseResource,
)

# Vehicle Routing Problem
from opencg.applications.vrp import (
    CVRPInstance,
    CVRPSolution,
    CVRPConfig,
    CapacityResource,
    solve_cvrp,
)

__all__ = [
    # Cutting Stock
    'CuttingStockInstance',
    'CuttingStockPricing',
    'CuttingStockMaster',
    'CuttingStockSolution',
    'solve_cutting_stock',
    # Crew Pairing
    'solve_crew_pairing',
    'CrewPairingConfig',
    'CrewPairingSolution',
    'HomeBaseResource',
    # Vehicle Routing
    'CVRPInstance',
    'CVRPSolution',
    'CVRPConfig',
    'CapacityResource',
    'solve_cvrp',
]
