"""
Crew pairing specific resources.

These resources enforce constraints specific to airline crew scheduling.
"""

from typing import TYPE_CHECKING, Optional

from opencg.core.resource import Resource

if TYPE_CHECKING:
    from opencg.core.arc import Arc
    from opencg.core.node import Node


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

    Example:
        >>> from opencg.applications.crew_pairing import HomeBaseResource
        >>> resource = HomeBaseResource()
        >>> # Initial value at source is None
        >>> value = resource.initial_value()  # None
        >>> # When traversing SOURCE_ARC with base='JFK', value becomes 'JFK'
        >>> # When traversing SINK_ARC with base='JFK', path is valid
        >>> # When traversing SINK_ARC with base='LAX', path is infeasible
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
