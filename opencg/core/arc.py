"""
Arc module - represents arcs (edges) in the time-space network.

Arcs connect nodes and carry:
1. Resource consumption (how much of each resource is used)
2. Cost (contribution to objective function)
3. Type information (flight, connection, rest, etc.)
4. Additional attributes (flight number, etc.)

This is Option C from our design discussion:
- Arc is generic, not domain-specific
- Resource consumption is the core concept
- User can subclass for domain-specific arcs

Design Notes:
------------
- Arcs store source/target as indices (int) for C++ compatibility
- Resource consumption is a Dict[str, float] keyed by resource name
- Cost is separate from resource consumption (always needed)
- ArcType provides semantic meaning for type-specific logic

Future C++ Note:
---------------
C++ will store arcs in a vector with source/target as indices.
Resource consumption will be a vector of (resource_id, value) pairs
for cache efficiency, not a hash map.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class ArcType(Enum):
    """
    Common arc types in airline/vehicle routing networks.

    This enum provides semantic meaning to arcs. It's optional - you can
    use ArcType.GENERIC for any arc.

    Types:
        FLIGHT: A flight leg (revenue generating)
        DEADHEAD: Repositioning flight (non-revenue, crew positioning)
        CONNECTION: Wait at airport between activities
        OVERNIGHT: Rest period at a hotel
        BASE_START: Leaving the crew base (start of pairing)
        BASE_END: Returning to crew base (end of pairing)
        SOURCE_ARC: From artificial source to real nodes
        SINK_ARC: From real nodes to artificial sink
        GENERIC: Any other arc type
    """
    FLIGHT = auto()       # Regular flight leg
    DEADHEAD = auto()     # Deadhead (repositioning) flight
    CONNECTION = auto()   # Connection/waiting at airport
    OVERNIGHT = auto()    # Overnight rest
    BASE_START = auto()   # Start from base
    BASE_END = auto()     # Return to base
    SOURCE_ARC = auto()   # From source node
    SINK_ARC = auto()     # To sink node
    GENERIC = auto()      # Default / other


@dataclass
class Arc:
    """
    Represents an arc (directed edge) in the network.

    An arc connects two nodes and carries information about:
    - Resource consumption: How traversing this arc affects each resource
    - Cost: Contribution to the objective function
    - Type: Semantic meaning (flight, connection, etc.)
    - Attributes: Any additional domain-specific data

    Attributes:
        index: Unique integer identifier
        source: Index of the source node
        target: Index of the target node
        cost: Cost of traversing this arc (for objective function)
        resource_consumption: Dict mapping resource name -> consumption value
        arc_type: Semantic type of the arc
        attributes: Flexible dictionary for additional data

    Resource Consumption:
        The resource_consumption dict maps resource names to values.
        For example: {"duty_time": 2.5, "flight_time": 2.0, "cost": 500}

        When a Resource extends along this arc, it looks up its name
        in this dictionary to get the consumption value.

    Example:
        >>> # Create a flight arc
        >>> flight = Arc(
        ...     index=0,
        ...     source=1,  # departure node index
        ...     target=2,  # arrival node index
        ...     cost=100.0,
        ...     resource_consumption={
        ...         "duty_time": 2.5,    # 2.5 hours added to duty
        ...         "flight_time": 2.0,  # 2.0 hours of actual flying
        ...         "fuel": 5000,        # kg of fuel
        ...     },
        ...     arc_type=ArcType.FLIGHT,
        ...     attributes={
        ...         "flight_number": "AA123",
        ...         "departure_time": 8.5,
        ...         "arrival_time": 11.0,
        ...     }
        ... )
        >>>
        >>> # Create a connection arc
        >>> conn = Arc(
        ...     index=1,
        ...     source=2,  # arrival node
        ...     target=3,  # next departure node
        ...     cost=0.0,  # no cost for waiting
        ...     resource_consumption={
        ...         "duty_time": 1.5,      # 1.5 hours of waiting
        ...         "connection_time": 1.5  # for connection time resource
        ...     },
        ...     arc_type=ArcType.CONNECTION,
        ... )
    """
    index: int
    source: int  # Node index (not Node object, for C++ compatibility)
    target: int  # Node index
    cost: float
    resource_consumption: Dict[str, float] = field(default_factory=dict)
    arc_type: ArcType = ArcType.GENERIC
    attributes: Dict[str, Any] = field(default_factory=dict)

    def get_consumption(self, resource_name: str, default: float = 0.0) -> float:
        """
        Get resource consumption for a specific resource.

        This is the primary method used by Resource.extend().

        Args:
            resource_name: Name of the resource
            default: Value to return if resource not in consumption dict

        Returns:
            Consumption value for the resource
        """
        return self.resource_consumption.get(resource_name, default)

    def set_consumption(self, resource_name: str, value: float) -> None:
        """
        Set resource consumption for a specific resource.

        Args:
            resource_name: Name of the resource
            value: Consumption value
        """
        self.resource_consumption[resource_name] = value

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

    def is_flight(self) -> bool:
        """Check if this is a flight arc."""
        return self.arc_type == ArcType.FLIGHT

    def is_deadhead(self) -> bool:
        """Check if this is a deadhead arc."""
        return self.arc_type == ArcType.DEADHEAD

    def is_connection(self) -> bool:
        """Check if this is a connection arc."""
        return self.arc_type == ArcType.CONNECTION

    @property
    def duration(self) -> Optional[float]:
        """
        Get duration if stored in attributes.

        This is a convenience property. Duration can also be computed
        from resource consumption (e.g., duty_time).
        """
        return self.attributes.get('duration')

    @duration.setter
    def duration(self, value: float) -> None:
        """Set duration attribute."""
        self.attributes['duration'] = value

    @property
    def flight_number(self) -> Optional[str]:
        """Get flight number if this is a flight arc."""
        return self.attributes.get('flight_number')

    @flight_number.setter
    def flight_number(self, value: str) -> None:
        """Set flight number."""
        self.attributes['flight_number'] = value

    def __hash__(self) -> int:
        """Hash by index for use in sets/dicts."""
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        """Equality by index."""
        if not isinstance(other, Arc):
            return NotImplemented
        return self.index == other.index

    def __repr__(self) -> str:
        type_str = self.arc_type.name if self.arc_type != ArcType.GENERIC else ""
        if type_str:
            return f"Arc({self.index}, {self.source}->{self.target}, {type_str}, cost={self.cost})"
        return f"Arc({self.index}, {self.source}->{self.target}, cost={self.cost})"


# =============================================================================
# Arc Factory Functions (convenience)
# =============================================================================


def make_flight_arc(
    index: int,
    source: int,
    target: int,
    flight_number: str,
    duration: float,
    cost: float,
    **extra_consumption
) -> Arc:
    """
    Convenience factory for creating flight arcs.

    Args:
        index: Arc index
        source: Source node index
        target: Target node index
        flight_number: Flight identifier
        duration: Flight duration in hours
        cost: Arc cost
        **extra_consumption: Additional resource consumptions

    Returns:
        Arc configured as a flight
    """
    consumption = {
        "duty_time": duration,
        "flight_time": duration,
        **extra_consumption
    }
    return Arc(
        index=index,
        source=source,
        target=target,
        cost=cost,
        resource_consumption=consumption,
        arc_type=ArcType.FLIGHT,
        attributes={"flight_number": flight_number, "duration": duration}
    )


def make_connection_arc(
    index: int,
    source: int,
    target: int,
    duration: float,
    cost: float = 0.0,
) -> Arc:
    """
    Convenience factory for creating connection arcs.

    Args:
        index: Arc index
        source: Source node index
        target: Target node index
        duration: Connection time in hours
        cost: Arc cost (usually 0)

    Returns:
        Arc configured as a connection
    """
    consumption = {
        "duty_time": duration,
        "connection_time": duration,
    }
    return Arc(
        index=index,
        source=source,
        target=target,
        cost=cost,
        resource_consumption=consumption,
        arc_type=ArcType.CONNECTION,
        attributes={"duration": duration}
    )


def make_deadhead_arc(
    index: int,
    source: int,
    target: int,
    flight_number: str,
    duration: float,
    cost: float,
) -> Arc:
    """
    Convenience factory for creating deadhead arcs.

    Deadheads typically count as half credit time.

    Args:
        index: Arc index
        source: Source node index
        target: Target node index
        flight_number: Flight identifier (the flight being deadheaded on)
        duration: Deadhead duration in hours
        cost: Arc cost

    Returns:
        Arc configured as a deadhead
    """
    consumption = {
        "duty_time": duration,
        "flight_time": 0.0,  # No flight time credit
        "credit_time": duration * 0.5,  # Half credit
    }
    return Arc(
        index=index,
        source=source,
        target=target,
        cost=cost,
        resource_consumption=consumption,
        arc_type=ArcType.DEADHEAD,
        attributes={"flight_number": flight_number, "duration": duration}
    )
