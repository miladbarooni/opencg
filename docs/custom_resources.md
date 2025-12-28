# Custom Resources Guide

This guide explains how to create custom resource constraints for the SPPRC (Shortest Path Problem with Resource Constraints) labeling algorithm.

## Overview

In OpenCG, a **Resource** defines how a quantity changes as paths are extended through the network. Resources enable you to model constraints like:

- Time limits (duty time, flight time)
- Capacity constraints (vehicle load)
- Visit constraints (elementarity)
- Time windows (delivery windows)
- Custom business rules

## The Resource Interface

All resources inherit from the `Resource` abstract base class:

```python
from opencg.core.resource import Resource
from typing import Optional

class Resource(ABC, Generic[T]):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def initial_value(self) -> T:
        """Starting value at source node."""
        pass

    @abstractmethod
    def extend(self, current_value: T, arc: Arc) -> Optional[T]:
        """New value after traversing arc. Return None if infeasible."""
        pass

    @abstractmethod
    def is_feasible(self, value: T, node: Node) -> bool:
        """Check if value is feasible at node."""
        pass

    @abstractmethod
    def dominates(self, value1: T, value2: T) -> bool:
        """Does value1 dominate value2? (for label pruning)"""
        pass
```

## Built-in Resources

OpenCG provides several built-in resource types:

### AccumulatingResource

For quantities that accumulate along the path with a maximum bound.

```python
from opencg.core.resource import AccumulatingResource

# Duty time: accumulates, max 10 hours
duty_time = AccumulatingResource(
    name="duty_time",
    initial=0.0,
    max_value=10.0  # hours
)

# Distance: accumulates, no max
distance = AccumulatingResource(
    name="distance",
    initial=0.0,
    max_value=float('inf')
)
```

### IntervalResource

For quantities that must stay within `[min, max]`.

```python
from opencg.core.resource import IntervalResource

# Connection time: must be between 30 min and 4 hours
connection = IntervalResource(
    name="connection_time",
    initial=0.0,
    min_value=0.5,  # 30 minutes
    max_value=4.0   # 4 hours
)
```

### StateResource

For tracking visited states (elementarity constraints).

```python
from opencg.core.resource import StateResource

# Elementary path: each node visited at most once
visited = StateResource(
    name="visited_nodes",
    forbidden_revisit=True
)
```

### TimeWindowResource

For time with node-specific arrival windows.

```python
from opencg.core.resource import TimeWindowResource

# Time resource for VRPTW
time = TimeWindowResource(
    name="time",
    initial=0.0
)

# Set time windows on nodes
node.set_attribute("time_window", (earliest, latest))
```

### HomeBaseResource

For crew pairing - ensures crew returns to starting base.

```python
from opencg.core.resource import HomeBaseResource

# Crew must start and end at same base
home_base = HomeBaseResource(name="home_base")
```

---

## Creating Custom Resources

### Example 1: Overtime Resource

A resource that tracks regular vs overtime hours with different costs.

```python
from opencg.core.resource import Resource
from typing import Optional, NamedTuple

class OvertimeValue(NamedTuple):
    """Tracks regular and overtime hours separately."""
    regular: float
    overtime: float

class OvertimeResource(Resource[OvertimeValue]):
    """
    Tracks hours worked with overtime threshold.

    Regular hours: 0-8 hours (cost = base rate)
    Overtime hours: >8 hours (cost = 1.5x base rate)
    """

    def __init__(
        self,
        name: str = "overtime",
        regular_limit: float = 8.0,
        max_total: float = 12.0
    ):
        super().__init__(name)
        self.regular_limit = regular_limit
        self.max_total = max_total

    def initial_value(self) -> OvertimeValue:
        return OvertimeValue(regular=0.0, overtime=0.0)

    def extend(self, current: OvertimeValue, arc) -> Optional[OvertimeValue]:
        hours = arc.get_consumption(self.name, 0.0)
        total_regular = current.regular
        total_overtime = current.overtime

        # First fill regular hours, then overtime
        remaining_regular = self.regular_limit - total_regular

        if hours <= remaining_regular:
            # All hours are regular
            new_regular = total_regular + hours
            new_overtime = total_overtime
        else:
            # Some hours are overtime
            new_regular = self.regular_limit
            new_overtime = total_overtime + (hours - remaining_regular)

        # Check max total
        if new_regular + new_overtime > self.max_total:
            return None  # Infeasible

        return OvertimeValue(regular=new_regular, overtime=new_overtime)

    def is_feasible(self, value: OvertimeValue, node) -> bool:
        return value.regular + value.overtime <= self.max_total

    def dominates(self, v1: OvertimeValue, v2: OvertimeValue) -> bool:
        # Less total hours is better (more extension possible)
        # Prefer regular over overtime (cheaper)
        total1 = v1.regular + v1.overtime
        total2 = v2.regular + v2.overtime

        if total1 < total2:
            return True
        if total1 == total2:
            # Same total, fewer overtime is better
            return v1.overtime <= v2.overtime
        return False
```

**Usage:**

```python
from opencg.core.problem import Problem

problem = Problem(name="crew_scheduling")
problem.add_resource(OvertimeResource(
    regular_limit=8.0,
    max_total=12.0
))

# On arcs, set the hours consumption
arc.set_consumption("overtime", 2.5)  # 2.5 hours of work
```

---

### Example 2: Multi-Commodity Capacity

For problems where vehicles carry multiple commodity types with separate limits.

```python
from opencg.core.resource import Resource
from typing import Optional

class MultiCommodityCapacity(Resource[dict[str, float]]):
    """
    Tracks capacity consumption for multiple commodities.

    Each commodity type has its own limit.
    """

    def __init__(
        self,
        name: str = "capacity",
        limits: dict[str, float] = None
    ):
        super().__init__(name)
        self.limits = limits or {}

    def initial_value(self) -> dict[str, float]:
        return {k: 0.0 for k in self.limits}

    def extend(self, current: dict[str, float], arc) -> Optional[dict[str, float]]:
        new_value = current.copy()

        # Get demands from arc attributes
        demands = arc.get_attribute("demands", {})

        for commodity, demand in demands.items():
            if commodity in new_value:
                new_value[commodity] += demand

                # Check limit
                if new_value[commodity] > self.limits[commodity]:
                    return None  # Infeasible

        return new_value

    def is_feasible(self, value: dict[str, float], node) -> bool:
        return all(
            value.get(k, 0) <= limit
            for k, limit in self.limits.items()
        )

    def dominates(self, v1: dict[str, float], v2: dict[str, float]) -> bool:
        # v1 dominates if all commodities have <= consumption
        return all(
            v1.get(k, 0) <= v2.get(k, 0)
            for k in set(v1.keys()) | set(v2.keys())
        )
```

**Usage:**

```python
capacity = MultiCommodityCapacity(
    limits={
        "frozen": 100.0,   # kg of frozen goods
        "fresh": 150.0,    # kg of fresh goods
        "dry": 200.0       # kg of dry goods
    }
)

problem.add_resource(capacity)

# On arcs (delivery stops)
arc.set_attribute("demands", {"frozen": 20, "fresh": 30})
```

---

### Example 3: Skill Matching Resource

Ensure workers have required skills for tasks.

```python
from opencg.core.resource import Resource
from typing import Optional

class SkillMatchingResource(Resource[frozenset[str]]):
    """
    Tracks worker skills and ensures task requirements are met.

    The resource value is the set of available skills.
    """

    def __init__(
        self,
        name: str = "skills",
        available_skills: set[str] = None
    ):
        super().__init__(name)
        self.available_skills = frozenset(available_skills or set())

    def initial_value(self) -> frozenset[str]:
        return self.available_skills

    def extend(self, current: frozenset[str], arc) -> Optional[frozenset[str]]:
        # Get required skills for this task
        required = arc.get_attribute("required_skills", set())

        # Check if worker has all required skills
        if not required.issubset(current):
            return None  # Missing skills, infeasible

        return current  # Skills don't deplete

    def is_feasible(self, value: frozenset[str], node) -> bool:
        return True  # Feasibility checked during extension

    def dominates(self, v1: frozenset[str], v2: frozenset[str]) -> bool:
        # More skills = more flexibility = dominates
        return v1.issuperset(v2)
```

**Usage:**

```python
# Create resource for a multi-skilled worker
skills = SkillMatchingResource(
    available_skills={"welding", "electrical", "plumbing"}
)

# On task arcs
arc.set_attribute("required_skills", {"welding", "electrical"})
```

---

### Example 4: Battery/Fuel Resource with Recharging

For electric vehicles that can recharge at certain nodes.

```python
from opencg.core.resource import Resource
from typing import Optional

class BatteryResource(Resource[float]):
    """
    Battery charge that depletes during travel and recharges at stations.
    """

    def __init__(
        self,
        name: str = "battery",
        max_charge: float = 100.0,
        initial_charge: float = 100.0
    ):
        super().__init__(name)
        self.max_charge = max_charge
        self.initial_charge = initial_charge

    def initial_value(self) -> float:
        return self.initial_charge

    def extend(self, current: float, arc) -> Optional[float]:
        # Get energy consumption (negative) or recharge (positive)
        consumption = arc.get_consumption(self.name, 0.0)

        new_charge = current - consumption  # consumption is positive

        # Check if recharging at this arc
        recharge = arc.get_attribute("recharge_amount", 0.0)
        new_charge = min(new_charge + recharge, self.max_charge)

        # Check if we ran out of battery
        if new_charge < 0:
            return None  # Ran out of battery

        return new_charge

    def is_feasible(self, value: float, node) -> bool:
        return value >= 0

    def dominates(self, v1: float, v2: float) -> bool:
        # More charge is better (can travel further)
        return v1 >= v2
```

**Usage:**

```python
battery = BatteryResource(max_charge=100.0, initial_charge=100.0)

# Travel arcs consume battery
travel_arc.set_consumption("battery", 15.0)  # Uses 15 units

# Charging station arcs
charging_arc.set_attribute("recharge_amount", 80.0)  # Recharge 80 units
```

---

## Best Practices

### 1. Choose the Right Value Type

| Value Type | Use Case | Example |
|------------|----------|---------|
| `float` | Simple accumulating quantities | Time, distance, cost |
| `tuple` | Multi-dimensional quantities | (regular_hours, overtime) |
| `set/frozenset` | Visited states, skills | Visited nodes, available skills |
| `dict` | Named quantities | Multi-commodity capacity |
| Custom class | Complex state | Vehicle state with multiple fields |

### 2. Dominance Rules

The dominance rule is **critical** for performance. A good dominance rule prunes many labels, speeding up the algorithm.

**Key principle**: If label L1 dominates L2, then any feasible extension from L2 must also be feasible from L1.

```python
# For minimization resources (time, cost):
def dominates(self, v1, v2):
    return v1 <= v2  # Less is better

# For maximization resources (remaining capacity):
def dominates(self, v1, v2):
    return v1 >= v2  # More is better

# For sets (visited nodes):
def dominates(self, v1, v2):
    return v1.issubset(v2)  # Fewer visited = more options
```

### 3. Handle Infeasibility in extend()

Return `None` from `extend()` when an extension is infeasible:

```python
def extend(self, current, arc):
    new_value = current + arc.get_consumption(self.name, 0.0)

    if new_value > self.max_value:
        return None  # Infeasible - prune this path

    return new_value
```

### 4. Use Arc Attributes

Store arc-specific data in attributes:

```python
# Setting attributes
arc.set_consumption("time", 2.5)           # For standard resources
arc.set_attribute("required_skills", {...}) # For custom data

# Getting in resource
consumption = arc.get_consumption("time", 0.0)
skills = arc.get_attribute("required_skills", set())
```

### 5. Consider Performance

- Use immutable types (`frozenset`, `tuple`, `NamedTuple`) when possible
- Avoid deep copying in `extend()` unless necessary
- Keep `dominates()` simple and fast (called frequently)

---

## Integrating with the C++ Backend

For maximum performance, you can implement resources in C++. The Python resource serves as a prototype, and you can then port to C++ using the same interface.

See `src/core/resource.hpp` for the C++ resource interface.

---

## Next Steps

- [Custom Pricing Algorithms](custom_pricing.md) - Implement your own SPPRC solver
- [Custom Master Problem](custom_master.md) - Use different LP/MIP solvers
- [Custom Applications](custom_application.md) - Model new problem types
