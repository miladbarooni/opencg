"""
Resource module - defines resource constraints for SPPRC.

In the Shortest Path Problem with Resource Constraints (SPPRC), a "resource"
is any quantity that:
1. Has an initial value at the source node
2. Gets "extended" (updated) when traversing an arc
3. Has feasibility constraints (bounds)
4. Has a dominance rule (for pruning labels)

This module provides:
- Resource: Abstract base class defining the interface
- AccumulatingResource: Resources that accumulate (e.g., duty time)
- IntervalResource: Resources that must stay in [min, max] (e.g., connection time)
- StateResource: Binary/categorical states (e.g., visited nodes)
- TimeWindowResource: Time with node-specific windows

Design Notes:
------------
- All resources inherit from Resource ABC
- Users can create custom resources by subclassing Resource
- Resources are identified by name (string) for flexibility
- The C++ backend will mirror this interface for performance

Future C++ Note:
---------------
This class will have a C++ equivalent in src/core/resource.hpp.
The Python version will remain for:
1. Pure Python fallback (no compilation needed)
2. Custom user-defined resources (Python is more accessible)
3. Prototyping new resource types
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from opencg.core.arc import Arc
    from opencg.core.node import Node

# Type variable for resource values (float, int, set, tuple, etc.)
T = TypeVar('T')


class Resource(ABC, Generic[T]):
    """
    Abstract base class for resources in SPPRC.

    A Resource defines how a quantity changes as we traverse a path in the
    network, and when a partial path (label) can be pruned.

    Type Parameter:
        T: The type of the resource value (e.g., float for time, Set[int] for visited nodes)

    Attributes:
        name: Unique identifier for this resource

    Methods to Implement:
        initial_value(): Starting value at source node
        extend(current, arc): New value after traversing arc (None if infeasible)
        is_feasible(value, node): Check if value is feasible at node
        dominates(v1, v2): Does label with v1 dominate label with v2?

    Example:
        >>> class DutyTime(Resource[float]):
        ...     def __init__(self, max_hours: float = 10.0):
        ...         super().__init__("duty_time")
        ...         self.max_hours = max_hours
        ...
        ...     def initial_value(self) -> float:
        ...         return 0.0
        ...
        ...     def extend(self, current: float, arc) -> Optional[float]:
        ...         new_value = current + arc.get_consumption("duty_time", 0.0)
        ...         return new_value if new_value <= self.max_hours else None
        ...
        ...     def is_feasible(self, value: float, node) -> bool:
        ...         return value <= self.max_hours
        ...
        ...     def dominates(self, v1: float, v2: float) -> bool:
        ...         return v1 <= v2  # Less time used is better
    """

    def __init__(self, name: str):
        """
        Initialize a resource.

        Args:
            name: Unique identifier for this resource. Used as key in
                  arc.resource_consumption dictionaries.
        """
        self.name = name

    @abstractmethod
    def initial_value(self) -> T:
        """
        Return the initial resource value at the source node.

        This is the starting point for resource extension along a path.

        Returns:
            Initial value of type T
        """
        pass

    @abstractmethod
    def extend(self, current_value: T, arc: 'Arc') -> Optional[T]:
        """
        Extend the resource value along an arc.

        This is the core operation in SPPRC. Given the current resource value
        and an arc to traverse, compute the new value after traversal.

        Args:
            current_value: Resource value before traversing the arc
            arc: The arc being traversed (contains resource_consumption dict)

        Returns:
            New resource value after traversing arc, or None if infeasible.
            Returning None means this arc cannot be taken given the current
            resource state (e.g., would exceed maximum duty time).
        """
        pass

    @abstractmethod
    def is_feasible(self, value: T, node: 'Node') -> bool:
        """
        Check if a resource value is feasible at a node.

        This allows node-specific constraints (e.g., time windows at stations).

        Args:
            value: Current resource value
            node: The node where feasibility is being checked

        Returns:
            True if the resource value is feasible at this node
        """
        pass

    @abstractmethod
    def dominates(self, value1: T, value2: T) -> bool:
        """
        Check if value1 dominates value2 for this resource.

        Dominance is used for pruning labels in SPPRC. If label L1 dominates
        label L2, then L2 can be discarded because any path extension from L2
        could also be done from L1 with equal or better resource consumption.

        For a label L1 to dominate L2, it must dominate on ALL resources.
        This method checks dominance for a single resource.

        Args:
            value1: Resource value of the potentially dominating label
            value2: Resource value of the potentially dominated label

        Returns:
            True if value1 dominates value2 for this resource.

        Note:
            - For minimization (e.g., time): value1 dominates if value1 <= value2
            - For maximization (e.g., remaining capacity): value1 dominates if value1 >= value2
            - For sets (e.g., visited nodes): value1 dominates if value1 is subset of value2
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# Built-in Resource Types
# =============================================================================


class AccumulatingResource(Resource[float]):
    """
    A resource that accumulates along the path with a maximum bound.

    This is the most common resource type. Examples:
    - Duty time (accumulates, max 10 hours)
    - Flight time (accumulates, max 8 hours)
    - Cost (accumulates, may or may not have max)
    - Distance (accumulates)

    The resource value increases by the arc's consumption, and becomes
    infeasible if it exceeds max_value.

    Attributes:
        name: Resource identifier
        initial: Starting value (default 0.0)
        max_value: Maximum allowed value (default infinity)

    Dominance:
        value1 dominates value2 if value1 <= value2 (less accumulation is better)
    """

    def __init__(
        self,
        name: str,
        initial: float = 0.0,
        max_value: float = float('inf'),
    ):
        """
        Create an accumulating resource.

        Args:
            name: Resource identifier (used in arc.resource_consumption)
            initial: Starting value at source node
            max_value: Maximum allowed value (infeasible if exceeded)
        """
        super().__init__(name)
        self.initial = initial
        self.max_value = max_value

    def initial_value(self) -> float:
        return self.initial

    def extend(self, current_value: float, arc: 'Arc') -> Optional[float]:
        # Get consumption from arc (default 0.0 if not specified)
        consumption = arc.get_consumption(self.name, 0.0)
        new_value = current_value + consumption

        # Clamp at initial value (for negative consumption, e.g., rest periods)
        # This allows overnight layovers to "reset" accumulated duty time
        if new_value < self.initial:
            new_value = self.initial

        # Check feasibility
        if new_value > self.max_value:
            return None  # Infeasible
        return new_value

    def is_feasible(self, value: float, node: 'Node') -> bool:
        return value <= self.max_value

    def dominates(self, value1: float, value2: float) -> bool:
        # Less accumulation is better (can extend further)
        return value1 <= value2

    def __repr__(self) -> str:
        max_str = f", max={self.max_value}" if self.max_value != float('inf') else ""
        return f"AccumulatingResource('{self.name}', initial={self.initial}{max_str})"


class IntervalResource(Resource[float]):
    """
    A resource that must stay within an interval [min_value, max_value].

    Unlike AccumulatingResource, this checks both lower and upper bounds.

    Examples:
    - Connection time: must be between 30 min and 4 hours
    - Break duration: must be at least 10 hours (min only)

    Attributes:
        name: Resource identifier
        initial: Starting value
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Dominance:
        value1 dominates value2 if value1 is "more flexible" - this depends
        on whether we're closer to min or max. For simplicity, we use
        value1 <= value2 (same as accumulating).
    """

    def __init__(
        self,
        name: str,
        initial: float = 0.0,
        min_value: float = 0.0,
        max_value: float = float('inf'),
    ):
        """
        Create an interval resource.

        Args:
            name: Resource identifier
            initial: Starting value at source node
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        super().__init__(name)
        self.initial = initial
        self.min_value = min_value
        self.max_value = max_value

    def initial_value(self) -> float:
        return self.initial

    def extend(self, current_value: float, arc: 'Arc') -> Optional[float]:
        consumption = arc.get_consumption(self.name, 0.0)
        new_value = current_value + consumption

        # Check both bounds
        if new_value < self.min_value or new_value > self.max_value:
            return None
        return new_value

    def is_feasible(self, value: float, node: 'Node') -> bool:
        return self.min_value <= value <= self.max_value

    def dominates(self, value1: float, value2: float) -> bool:
        # For interval resources, dominance is more nuanced
        # We use simple comparison, but user can override for specific needs
        return value1 <= value2

    def __repr__(self) -> str:
        return f"IntervalResource('{self.name}', [{self.min_value}, {self.max_value}])"


class StateResource(Resource[set[int]]):
    """
    A resource tracking visited states (e.g., nodes, activities).

    Used for elementary path constraints where each node can be visited
    at most once.

    Examples:
    - Elementary shortest path: cannot revisit nodes
    - Task assignment: each task done at most once
    - ng-route relaxation: restricted neighborhood sets

    Attributes:
        name: Resource identifier
        forbidden_revisit: If True, revisiting a state makes path infeasible

    Dominance:
        value1 (set) dominates value2 (set) if value1 is subset of value2
        (fewer visited states = more extension options)
    """

    def __init__(
        self,
        name: str,
        forbidden_revisit: bool = True,
    ):
        """
        Create a state-tracking resource.

        Args:
            name: Resource identifier
            forbidden_revisit: If True, visiting same state twice is infeasible
        """
        super().__init__(name)
        self.forbidden_revisit = forbidden_revisit

    def initial_value(self) -> set[int]:
        return set()

    def extend(self, current_value: set[int], arc: 'Arc') -> Optional[set[int]]:
        # Get the state ID from the arc's target node
        # (This will be the node index, typically)
        state_id = arc.get_consumption(self.name, None)

        if state_id is None:
            # No state to track on this arc
            return current_value

        state_id = int(state_id)

        if self.forbidden_revisit and state_id in current_value:
            return None  # Already visited, infeasible

        # Create new set with added state
        new_value = current_value | {state_id}
        return new_value

    def is_feasible(self, value: set[int], node: 'Node') -> bool:
        # Feasibility is checked during extension
        return True

    def dominates(self, value1: set[int], value2: set[int]) -> bool:
        # Fewer visited states = more flexibility = dominates
        return value1.issubset(value2)

    def __repr__(self) -> str:
        return f"StateResource('{self.name}', forbidden_revisit={self.forbidden_revisit})"


class TimeWindowResource(Resource[float]):
    """
    A time resource with node-specific time windows.

    Each node can have an earliest and latest allowed arrival time.
    If arriving early, we wait until the window opens.
    If arriving late, the path is infeasible.

    Examples:
    - Delivery time windows: must arrive between 9am-11am
    - Flight connections: must arrive before next departure
    - Crew rest: must have minimum rest before next duty

    Attributes:
        name: Resource identifier
        initial: Starting time
        time_windows: Dict mapping node_id -> (earliest, latest)

    Dominance:
        Earlier time dominates later time (more time flexibility)
    """

    def __init__(
        self,
        name: str,
        initial: float = 0.0,
    ):
        """
        Create a time window resource.

        Args:
            name: Resource identifier
            initial: Starting time value

        Note:
            Time windows are stored on nodes, not in this object.
            The node should have 'time_window' attribute as (earliest, latest).
        """
        super().__init__(name)
        self.initial = initial

    def initial_value(self) -> float:
        return self.initial

    def extend(self, current_value: float, arc: 'Arc') -> Optional[float]:
        # Get travel time from arc
        travel_time = arc.get_consumption(self.name, 0.0)
        arrival_time = current_value + travel_time

        # Time window check is done in is_feasible (needs node info)
        return arrival_time

    def is_feasible(self, value: float, node: 'Node') -> bool:
        # Get time window from node
        time_window = node.get_attribute('time_window', None)

        if time_window is None:
            return True  # No window constraint

        earliest, latest = time_window

        # If we arrive early, we wait (time becomes earliest)
        # If we arrive late, infeasible
        if value > latest:
            return False

        return True

    def apply_waiting(self, value: float, node: 'Node') -> float:
        """
        Apply waiting if arriving before time window opens.

        Call this after extend() to adjust time for waiting.

        Args:
            value: Arrival time
            node: The node arrived at

        Returns:
            Adjusted time (earliest time if arrived early, else same)
        """
        time_window = node.get_attribute('time_window', None)

        if time_window is None:
            return value

        earliest, _ = time_window
        return max(value, earliest)

    def dominates(self, value1: float, value2: float) -> bool:
        # Earlier time dominates (more flexibility for future windows)
        return value1 <= value2

    def __repr__(self) -> str:
        return f"TimeWindowResource('{self.name}', initial={self.initial})"


# =============================================================================
# Resource Factory (for convenience)
# =============================================================================


def make_resource(
    name: str,
    resource_type: str = "accumulating",
    **kwargs
) -> Resource:
    """
    Factory function to create resources by type name.

    This is a convenience function for programmatic resource creation.

    Args:
        name: Resource identifier
        resource_type: One of "accumulating", "interval", "state", "time_window"
        **kwargs: Arguments passed to the resource constructor

    Returns:
        Resource instance of the specified type

    Example:
        >>> duty = make_resource("duty_time", "accumulating", max_value=10.0)
        >>> conn = make_resource("connection", "interval", min_value=0.5, max_value=4.0)
    """
    types = {
        "accumulating": AccumulatingResource,
        "interval": IntervalResource,
        "state": StateResource,
        "time_window": TimeWindowResource,
    }

    if resource_type not in types:
        raise ValueError(
            f"Unknown resource type '{resource_type}'. "
            f"Available types: {list(types.keys())}"
        )

    return types[resource_type](name, **kwargs)


class HomeBaseResource(Resource[Optional[str]]):
    """
    A resource tracking the crew's home base in crew pairing problems.

    This enforces that crew pairings must start and end at the same base.
    The resource value is:
    - None: Not yet assigned to a base (at source)
    - base_name: The home base (set when leaving a base, checked when returning)

    When a path traverses a SOURCE_ARC with a 'base' attribute, the home base
    is set. When traversing a SINK_ARC with a 'base' attribute, the arc is
    only feasible if the base matches the home base.

    Dominance:
        Labels with the same home base can dominate each other.
        Labels with different home bases cannot dominate each other.
    """

    def __init__(self, name: str = "home_base"):
        """Create a home base resource."""
        super().__init__(name)

    def initial_value(self) -> Optional[str]:
        return None

    def extend(self, current_value: Optional[str], arc: 'Arc') -> Optional[Optional[str]]:
        # Check if this arc sets or requires a home base
        arc_base = arc.get_attribute('base', None)

        if arc_base is None:
            # No base constraint on this arc
            return current_value

        # Check arc type to determine behavior
        from opencg.core.arc import ArcType

        if arc.arc_type == ArcType.SOURCE_ARC:
            # Setting the home base
            if current_value is None:
                # First time setting base - this is valid
                return arc_base
            elif current_value == arc_base:
                # Same base - still valid
                return current_value
            else:
                # Different base - infeasible (can't switch home base)
                return None

        elif arc.arc_type == ArcType.SINK_ARC:
            # Checking the home base
            if current_value is None:
                # No home base set - can't go to sink
                return None
            elif current_value == arc_base:
                # Returning to home base - valid
                return current_value
            else:
                # Wrong base - infeasible
                return None

        # Other arc types don't affect home base
        return current_value

    def is_feasible(self, value: Optional[str], node: 'Node') -> bool:
        # Feasibility is checked during extension
        return True

    def dominates(self, value1: Optional[str], value2: Optional[str]) -> bool:
        # Can only dominate if same home base (or both None)
        return value1 == value2

    def __repr__(self) -> str:
        return f"HomeBaseResource('{self.name}')"


