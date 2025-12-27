"""
Node module - represents nodes in the time-space network.

In column generation for routing/scheduling problems, the network is typically
a time-space network where:
- Nodes represent (location, time) pairs or abstract states
- Arcs represent transitions (flights, connections, etc.)

This module provides:
- NodeType: Enum for common node types
- Node: The node class with flexible attributes

Design Notes:
------------
- Nodes have an index (int) for fast lookup in C++
- Nodes have a name (str) for human readability
- Nodes can have arbitrary attributes (dict) for flexibility
- NodeType is optional - helps with type-specific logic

Future C++ Note:
---------------
The C++ version will store nodes in a vector by index.
Attributes will be stored separately for cache efficiency.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple


class NodeType(Enum):
    """
    Common node types in airline/vehicle routing networks.

    This enum provides semantic meaning to nodes. It's optional - you can
    use NodeType.GENERIC for any node.

    Types:
        SOURCE: Start of all paths (artificial source)
        SINK: End of all paths (artificial sink)
        BASE: Crew base / vehicle depot
        AIRPORT: Non-base airport
        STATION: Generic station/location
        FLIGHT_DEP: Flight departure event
        FLIGHT_ARR: Flight arrival event
        GENERIC: Any other node type
    """
    SOURCE = auto()      # Artificial source node
    SINK = auto()        # Artificial sink node
    BASE = auto()        # Crew base / depot
    AIRPORT = auto()     # Airport (non-base)
    STATION = auto()     # Generic station
    FLIGHT_DEP = auto()  # Flight departure event (time-space)
    FLIGHT_ARR = auto()  # Flight arrival event (time-space)
    GENERIC = auto()     # Default / other


@dataclass
class Node:
    """
    Represents a node in the network.

    A node is a point in the time-space network. It could represent:
    - A physical location (airport, station)
    - A time-expanded event (departure at time t)
    - An abstract state in the problem

    Attributes:
        index: Unique integer identifier (used for fast array indexing)
        name: Human-readable name (e.g., "JFK", "BASE1", "DEP_FL123_0800")
        node_type: Semantic type of the node (see NodeType enum)
        attributes: Flexible dictionary for additional data

    Common Attributes (stored in attributes dict):
        - time_window: Tuple[float, float] - (earliest, latest) arrival time
        - location: str - Physical location code
        - time: float - Time value for time-expanded nodes
        - capacity: int - Capacity constraint at this node

    Example:
        >>> # Create a crew base node
        >>> base = Node(
        ...     index=0,
        ...     name="BASE1",
        ...     node_type=NodeType.BASE,
        ...     attributes={"location": "JFK", "crew_count": 50}
        ... )
        >>>
        >>> # Create a time-expanded departure node
        >>> dep = Node(
        ...     index=1,
        ...     name="DEP_FL123",
        ...     node_type=NodeType.FLIGHT_DEP,
        ...     attributes={"flight": "FL123", "time": 8.5, "location": "JFK"}
        ... )

    Note:
        The index should be assigned by the Network when adding nodes.
        Don't set it manually unless you're building the network yourself.
    """
    index: int
    name: str
    node_type: NodeType = NodeType.GENERIC
    attributes: Dict[str, Any] = field(default_factory=dict)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """
        Get an attribute value with a default.

        Args:
            key: Attribute name
            default: Value to return if attribute not found

        Returns:
            The attribute value or default
        """
        return self.attributes.get(key, default)

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute value.

        Args:
            key: Attribute name
            value: Attribute value
        """
        self.attributes[key] = value

    @property
    def time_window(self) -> Optional[Tuple[float, float]]:
        """Get time window if set, else None."""
        return self.attributes.get('time_window')

    @time_window.setter
    def time_window(self, value: Tuple[float, float]) -> None:
        """Set time window as (earliest, latest)."""
        self.attributes['time_window'] = value

    @property
    def location(self) -> Optional[str]:
        """Get physical location if set, else None."""
        return self.attributes.get('location')

    @location.setter
    def location(self, value: str) -> None:
        """Set physical location."""
        self.attributes['location'] = value

    @property
    def time(self) -> Optional[float]:
        """Get time value if set, else None."""
        return self.attributes.get('time')

    @time.setter
    def time(self, value: float) -> None:
        """Set time value."""
        self.attributes['time'] = value

    def is_source(self) -> bool:
        """Check if this is a source node."""
        return self.node_type == NodeType.SOURCE

    def is_sink(self) -> bool:
        """Check if this is a sink node."""
        return self.node_type == NodeType.SINK

    def is_base(self) -> bool:
        """Check if this is a base/depot node."""
        return self.node_type == NodeType.BASE

    def __hash__(self) -> int:
        """Hash by index for use in sets/dicts."""
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        """Equality by index."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index

    def __lt__(self, other: 'Node') -> bool:
        """Comparison by index (for sorting)."""
        return self.index < other.index

    def __repr__(self) -> str:
        type_str = self.node_type.name if self.node_type != NodeType.GENERIC else ""
        if type_str:
            return f"Node({self.index}, '{self.name}', {type_str})"
        return f"Node({self.index}, '{self.name}')"
