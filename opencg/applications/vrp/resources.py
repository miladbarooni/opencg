"""
VRP-specific resources.

These resources enforce constraints specific to vehicle routing problems:
- CapacityResource: Tracks cumulative demand (CVRP)
- TimeResource: Tracks time and enforces time windows (VRPTW)
- VisitedResource: Enforces elementarity (each customer visited once)
"""

from typing import Optional, Tuple, TYPE_CHECKING

from opencg.core.resource import Resource

if TYPE_CHECKING:
    from opencg.core.arc import Arc
    from opencg.core.node import Node


class CapacityResource(Resource[float]):
    """
    A resource tracking vehicle load/capacity in VRP.

    The resource value represents the cumulative demand picked up so far.
    A path is infeasible if the cumulative demand exceeds vehicle capacity.

    Dominance:
        A label with less demand used dominates one with more demand,
        as it has more remaining capacity for future customers.

    Example:
        >>> resource = CapacityResource(vehicle_capacity=100)
        >>> value = resource.initial_value()  # 0.0
        >>> # After visiting customer with demand 30:
        >>> value = resource.extend(value, arc_with_demand_30)  # 30.0
        >>> # After visiting another customer with demand 50:
        >>> value = resource.extend(value, arc_with_demand_50)  # 80.0
    """

    def __init__(self, vehicle_capacity: float, name: str = "capacity"):
        """
        Create a capacity resource.

        Args:
            vehicle_capacity: Maximum vehicle capacity
            name: Resource name (default: "capacity")
        """
        super().__init__(name)
        self.vehicle_capacity = vehicle_capacity
        self.max_value = vehicle_capacity  # For C++ backend compatibility

    def initial_value(self) -> float:
        """Start with empty vehicle (0 demand picked up)."""
        return 0.0

    def extend(self, current_value: float, arc: 'Arc') -> Optional[float]:
        """
        Extend capacity by picking up demand at target node.

        The arc should have a 'demand' attribute indicating the demand
        to pick up when traversing this arc.

        Returns:
            New cumulative demand, or None if exceeds capacity
        """
        demand = arc.get_attribute('demand', 0.0)
        new_value = current_value + demand

        if new_value > self.vehicle_capacity:
            return None  # Infeasible - exceeds capacity

        return new_value

    def is_feasible(self, value: float, node: 'Node') -> bool:
        """Check if cumulative demand is within capacity."""
        return value <= self.vehicle_capacity

    def dominates(self, value1: float, value2: float) -> bool:
        """Less demand used is better (more remaining capacity)."""
        return value1 <= value2

    def __repr__(self) -> str:
        return f"CapacityResource(capacity={self.vehicle_capacity})"


class VisitedResource(Resource[frozenset]):
    """
    A resource tracking which customers have been visited.

    This enforces elementarity - each customer can only be visited once.
    The resource value is a frozenset of visited customer IDs.

    Dominance:
        A label with fewer visited customers doesn't dominate one with more,
        because they may visit different customers. Two labels can only
        dominate if they have the same visited set.
    """

    def __init__(self, name: str = "visited"):
        """Create a visited resource."""
        super().__init__(name)

    def initial_value(self) -> frozenset:
        """Start with no customers visited."""
        return frozenset()

    def extend(self, current_value: frozenset, arc: 'Arc') -> Optional[frozenset]:
        """
        Add customer to visited set.

        The arc should have a 'customer_id' attribute if it visits a customer.
        Returns None if the customer was already visited (elementarity violation).
        """
        customer_id = arc.get_attribute('customer_id', None)

        if customer_id is None:
            # Not visiting a customer (e.g., depot arc)
            return current_value

        if customer_id in current_value:
            # Already visited - infeasible (elementarity)
            return None

        return current_value | {customer_id}

    def is_feasible(self, value: frozenset, node: 'Node') -> bool:
        """Always feasible if we got here."""
        return True

    def dominates(self, value1: frozenset, value2: frozenset) -> bool:
        """
        Can only dominate if same visited set.

        Actually, v1 dominates v2 if v1 is a subset of v2 (visited fewer),
        but for set partitioning we need exact coverage, so same set required.
        """
        return value1 == value2

    def __repr__(self) -> str:
        return f"VisitedResource('{self.name}')"


class TimeResource(Resource[float]):
    """
    A resource tracking time for VRPTW (VRP with Time Windows).

    The resource value represents the current time (after service at current node).
    Each arc has:
    - travel_time: Time to traverse the arc
    - service_time: Time spent at the target node
    - earliest: Earliest allowed arrival at target (can wait)
    - latest: Latest allowed arrival at target (cannot be late)

    Time evolution:
        arrival_time = current_time + travel_time
        start_service = max(arrival_time, earliest)  # wait if early
        departure_time = start_service + service_time

    A path is infeasible if arrival_time > latest.

    Dominance:
        Earlier time dominates later time (more flexibility for future).

    Example:
        >>> resource = TimeResource(depot_latest=1000)
        >>> t = resource.initial_value()  # 0.0
        >>> # Travel to customer with TW [50, 100], service=10, travel=30
        >>> t = resource.extend(t, arc)  # max(30, 50) + 10 = 60
    """

    def __init__(self, depot_latest: float = float('inf'), name: str = "time"):
        """
        Create a time resource.

        Args:
            depot_latest: Latest return time to depot (planning horizon)
            name: Resource name (default: "time")
        """
        super().__init__(name)
        self.depot_latest = depot_latest
        self.max_value = depot_latest  # For C++ backend compatibility

    def initial_value(self) -> float:
        """Start at time 0."""
        return 0.0

    def extend(self, current_time: float, arc: 'Arc') -> Optional[float]:
        """
        Extend time by traveling and serving at target.

        Arc attributes used:
        - travel_time: Time to traverse arc
        - service_time: Service time at target
        - earliest: Earliest arrival at target (default: 0)
        - latest: Latest arrival at target (default: inf)

        Returns:
            Departure time from target, or None if infeasible
        """
        travel_time = arc.get_attribute('travel_time', 0.0)
        service_time = arc.get_attribute('service_time', 0.0)
        earliest = arc.get_attribute('earliest', 0.0)
        latest = arc.get_attribute('latest', float('inf'))

        # Compute arrival time
        arrival_time = current_time + travel_time

        # Check if we can arrive in time
        if arrival_time > latest:
            return None  # Too late - infeasible

        # Wait if arriving early
        start_service = max(arrival_time, earliest)

        # Departure time
        departure_time = start_service + service_time

        return departure_time

    def is_feasible(self, value: float, node: 'Node') -> bool:
        """
        Check if current time is feasible.

        For the depot (sink), check against depot_latest.
        """
        # Check depot return time
        if node.get_attribute('is_depot', False):
            depot_latest = node.get_attribute('latest', self.depot_latest)
            return value <= depot_latest
        return True

    def dominates(self, value1: float, value2: float) -> bool:
        """Earlier time dominates (more flexibility)."""
        return value1 <= value2

    def __repr__(self) -> str:
        return f"TimeResource(depot_latest={self.depot_latest})"
