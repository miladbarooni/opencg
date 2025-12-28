"""
Core module - fundamental data structures for column generation.

This module contains the abstract base classes and core implementations
that form the backbone of the OpenCG framework.

Design Philosophy:
-----------------
1. Everything is a contract (abstract base classes define interfaces)
2. Sensible defaults with full override capability
3. Python for flexibility, C++ for speed (when needed)

Components:
----------
- Resource: Abstract base for resource constraints in SPPRC
- Node: Represents a node in the time-space network
- Arc: Represents an arc with resource consumption
- Network: The graph structure holding nodes and arcs
- Column: A feasible path (solution to pricing subproblem)
- Problem: Container that defines a complete CG problem
"""

from opencg.core.arc import Arc, ArcType
from opencg.core.column import Column
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.problem import Problem
from opencg.core.resource import (
    AccumulatingResource,
    IntervalResource,
    Resource,
    StateResource,
    TimeWindowResource,
)

__all__ = [
    # Resource types
    "Resource",
    "AccumulatingResource",
    "IntervalResource",
    "StateResource",
    "TimeWindowResource",
    # Network components
    "Node",
    "NodeType",
    "Arc",
    "ArcType",
    "Network",
    # Solution representation
    "Column",
    # Problem definition
    "Problem",
]
