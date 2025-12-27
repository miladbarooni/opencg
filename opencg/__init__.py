"""
OpenCG: Open-Source Column Generation Framework

A research-grade, extensible framework for solving optimization problems
using Column Generation and Branch-and-Price.
"""

__version__ = "0.1.0"

# Configuration
from opencg.config import config, get_data_path, set_data_path, get_instance_path

# Core classes - these are the main user-facing API
from opencg.core.resource import Resource
from opencg.core.node import Node
from opencg.core.arc import Arc
from opencg.core.network import Network
from opencg.core.column import Column
from opencg.core.problem import Problem

# Master problem
from opencg.master import (
    MasterProblem,
    MasterSolution,
    SolutionStatus,
    HiGHSMasterProblem,
    HIGHS_AVAILABLE,
)

# Pricing problem
from opencg.pricing import (
    PricingProblem,
    PricingConfig,
    PricingSolution,
    PricingStatus,
    LabelingAlgorithm,
    Label,
)

# Column generation solver
from opencg.solver import (
    ColumnGeneration,
    CGConfig,
    CGSolution,
    CGStatus,
)

# Applications (ready-to-use solvers for common problems)
from opencg.applications import (
    # Cutting Stock
    CuttingStockInstance,
    CuttingStockSolution,
    solve_cutting_stock,
    # Crew Pairing
    solve_crew_pairing,
    CrewPairingConfig,
    CrewPairingSolution,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "config",
    "get_data_path",
    "set_data_path",
    "get_instance_path",
    # Core classes
    "Resource",
    "Node",
    "Arc",
    "Network",
    "Column",
    "Problem",
    # Master problem
    "MasterProblem",
    "MasterSolution",
    "SolutionStatus",
    "HiGHSMasterProblem",
    "HIGHS_AVAILABLE",
    # Pricing problem
    "PricingProblem",
    "PricingConfig",
    "PricingSolution",
    "PricingStatus",
    "LabelingAlgorithm",
    "Label",
    # Column generation solver
    "ColumnGeneration",
    "CGConfig",
    "CGSolution",
    "CGStatus",
    # Applications - Cutting Stock
    "CuttingStockInstance",
    "CuttingStockSolution",
    "solve_cutting_stock",
    # Applications - Crew Pairing
    "solve_crew_pairing",
    "CrewPairingConfig",
    "CrewPairingSolution",
]