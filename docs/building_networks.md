# Building Networks Guide

This guide explains how to build time-space networks for column generation using OpenCG's core classes. No subclassing required - just use the provided building blocks.

## Overview

A **network** in OpenCG is a directed graph representing the structure of your optimization problem:

- **Nodes**: States (locations, times, or location-time pairs)
- **Arcs**: Transitions between states (activities, travel, waiting)
- **Paths**: Sequences of arcs from source to sink = columns

## Core Classes

```python
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
```

## Step 1: Create a Network

```python
network = Network()
```

## Step 2: Add Nodes

### Node Types

| NodeType | Use |
|----------|-----|
| `SOURCE` | Starting point (one per network) |
| `SINK` | Ending point (one per network) |
| `STATION` | Locations (depots, airports, warehouses) |
| `CUSTOMER` | Customers to visit (VRP) |
| `TASK` | Tasks/activities to perform |

### Adding Nodes

```python
# Source and sink (required)
source = network.add_node(NodeType.SOURCE, time=0)
sink = network.add_node(NodeType.SINK, time=1000)

# Set as network source/sink
network.source = source
network.sink = sink

# Station nodes (e.g., depots)
depot_a = network.add_node(NodeType.STATION, time=0)
depot_a.set_attribute("name", "Depot A")
depot_a.set_attribute("location", (0, 0))

depot_b = network.add_node(NodeType.STATION, time=0)
depot_b.set_attribute("name", "Depot B")
depot_b.set_attribute("location", (100, 0))

# Customer nodes
for i, (x, y, demand) in enumerate(customers):
    node = network.add_node(NodeType.CUSTOMER, time=0)
    node.set_attribute("customer_id", i)
    node.set_attribute("location", (x, y))
    node.set_attribute("demand", demand)

# Task nodes (for scheduling)
task = network.add_node(NodeType.TASK, time=start_time)
task.set_attribute("task_id", 1)
task.set_attribute("duration", 2.5)
task.set_attribute("required_skills", {"welding", "electrical"})
```

### Node Attributes

Store any data on nodes using attributes:

```python
# Set attribute
node.set_attribute("key", value)

# Get attribute (with default)
value = node.get_attribute("key", default=None)

# Common attributes
node.set_attribute("time_window", (earliest, latest))  # For VRPTW
node.set_attribute("demand", 50)                        # For capacity
node.set_attribute("location", (x, y))                  # For distances
node.set_attribute("service_time", 10)                  # Service duration
```

## Step 3: Add Arcs

### Arc Types

| ArcType | Use |
|---------|-----|
| `SOURCE_ARC` | From source to first activity |
| `SINK_ARC` | From last activity to sink |
| `FLIGHT_ARC` | Main activity arcs (flights, deliveries) |
| `CONNECTION_ARC` | Connections between activities |
| `DEADHEAD_ARC` | Empty repositioning |
| `REST_ARC` | Rest/break periods |

### Adding Arcs

```python
# Basic arc creation
arc = network.add_arc(source_node, target_node, cost=10.0)

# Set arc type
arc.arc_type = ArcType.FLIGHT_ARC

# Arc from source (starting a route)
start_arc = network.add_arc(source, depot_a, cost=0.0)
start_arc.arc_type = ArcType.SOURCE_ARC

# Activity arc (e.g., serving a customer)
service_arc = network.add_arc(depot_a, customer_1, cost=distance)
service_arc.arc_type = ArcType.FLIGHT_ARC
service_arc.set_attribute("customer_id", customer_1_id)

# Arc to sink (ending a route)
end_arc = network.add_arc(customer_n, sink, cost=return_distance)
end_arc.arc_type = ArcType.SINK_ARC
```

### Resource Consumption

Arcs consume resources as paths traverse them:

```python
# Set consumption for a named resource
arc.set_consumption("time", 2.5)        # 2.5 hours
arc.set_consumption("distance", 50.0)   # 50 km
arc.set_consumption("capacity", 20)     # 20 units of load

# Get consumption
time = arc.get_consumption("time", default=0.0)
```

### Arc Attributes

Store additional data:

```python
arc.set_attribute("flight_number", "AC123")
arc.set_attribute("customer_id", 5)
arc.set_attribute("required_skills", {"driving"})
arc.set_attribute("is_mandatory", True)
```

## Step 4: Define the Problem

```python
from opencg.core.problem import Problem, CoverConstraint, CoverType

# Create problem
problem = Problem(
    name="my_problem",
    network=network,
    cover_type=CoverType.PARTITIONING  # or COVERING
)

# Add items to cover (customers, flights, tasks, etc.)
for item_id in items_to_cover:
    problem.add_cover_constraint(CoverConstraint(
        item_id=item_id,
        name=f"cover_{item_id}"
    ))
```

### Cover Types

| Type | Constraint | Use Case |
|------|------------|----------|
| `PARTITIONING` | Each item covered exactly once | VRP, crew pairing |
| `COVERING` | Each item covered at least once | Shift scheduling |

## Step 5: Add Resources

```python
from opencg.core.resource import (
    AccumulatingResource,
    IntervalResource,
    TimeWindowResource,
    StateResource,
    HomeBaseResource
)

# Time limit
problem.add_resource(AccumulatingResource(
    name="time",
    initial=0.0,
    max_value=8.0  # 8 hour limit
))

# Vehicle capacity
problem.add_resource(AccumulatingResource(
    name="capacity",
    initial=0.0,
    max_value=100.0  # 100 unit capacity
))

# Connection time constraints
problem.add_resource(IntervalResource(
    name="connection",
    min_value=0.5,   # Minimum 30 min connection
    max_value=4.0    # Maximum 4 hour connection
))

# Time windows (set windows on nodes)
problem.add_resource(TimeWindowResource(name="time"))

# Elementarity (visit each node at most once)
problem.add_resource(StateResource(
    name="visited",
    forbidden_revisit=True
))

# Home base (return to starting depot)
problem.add_resource(HomeBaseResource(name="base"))
```

## Step 6: Solve

```python
from opencg.solver import ColumnGeneration, CGConfig

cg = ColumnGeneration(
    problem,
    config=CGConfig(
        max_iterations=50,
        verbose=True
    )
)

solution = cg.solve()

print(f"Objective: {solution.objective_value}")
print(f"Status: {solution.status}")
```

---

## Complete Examples

### Example 1: Simple VRP Network

```python
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
from opencg.core.problem import Problem, CoverConstraint, CoverType
from opencg.core.resource import AccumulatingResource
import math

def build_vrp_network(depot, customers, vehicle_capacity):
    """
    Build a VRP network.

    Args:
        depot: (x, y) coordinates of depot
        customers: List of (x, y, demand) tuples
        vehicle_capacity: Maximum vehicle load
    """
    network = Network()

    # Source and sink
    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=0)
    network.source = source
    network.sink = sink

    # Depot node
    depot_node = network.add_node(NodeType.STATION, time=0)
    depot_node.set_attribute("location", depot)

    # Customer nodes
    customer_nodes = []
    for i, (x, y, demand) in enumerate(customers):
        node = network.add_node(NodeType.CUSTOMER, time=0)
        node.set_attribute("customer_id", i)
        node.set_attribute("location", (x, y))
        node.set_attribute("demand", demand)
        customer_nodes.append(node)

    def dist(loc1, loc2):
        return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

    # Source -> Depot arc
    arc = network.add_arc(source, depot_node, cost=0.0)
    arc.arc_type = ArcType.SOURCE_ARC

    # Depot -> Customer arcs
    for i, cust_node in enumerate(customer_nodes):
        cust_loc = cust_node.get_attribute("location")
        demand = cust_node.get_attribute("demand")
        d = dist(depot, cust_loc)

        arc = network.add_arc(depot_node, cust_node, cost=d)
        arc.arc_type = ArcType.FLIGHT_ARC
        arc.set_consumption("distance", d)
        arc.set_consumption("capacity", demand)
        arc.set_attribute("customer_id", i)

    # Customer -> Customer arcs
    for i, node_i in enumerate(customer_nodes):
        loc_i = node_i.get_attribute("location")

        for j, node_j in enumerate(customer_nodes):
            if i == j:
                continue

            loc_j = node_j.get_attribute("location")
            demand_j = node_j.get_attribute("demand")
            d = dist(loc_i, loc_j)

            arc = network.add_arc(node_i, node_j, cost=d)
            arc.arc_type = ArcType.FLIGHT_ARC
            arc.set_consumption("distance", d)
            arc.set_consumption("capacity", demand_j)
            arc.set_attribute("customer_id", j)

    # Customer -> Depot -> Sink arcs
    for i, cust_node in enumerate(customer_nodes):
        cust_loc = cust_node.get_attribute("location")
        d = dist(cust_loc, depot)

        # Return to depot then sink
        arc = network.add_arc(cust_node, depot_node, cost=d)
        arc.arc_type = ArcType.SINK_ARC
        arc.set_consumption("distance", d)

    # Depot -> Sink
    arc = network.add_arc(depot_node, sink, cost=0.0)
    arc.arc_type = ArcType.SINK_ARC

    return network, customer_nodes

def solve_vrp(depot, customers, vehicle_capacity):
    """Solve VRP instance."""
    network, customer_nodes = build_vrp_network(depot, customers, vehicle_capacity)

    problem = Problem(
        name="vrp",
        network=network,
        cover_type=CoverType.PARTITIONING
    )

    # Cover each customer
    for i in range(len(customers)):
        problem.add_cover_constraint(CoverConstraint(item_id=i))

    # Capacity constraint
    problem.add_resource(AccumulatingResource(
        name="capacity",
        initial=0.0,
        max_value=vehicle_capacity
    ))

    # Distance tracking
    problem.add_resource(AccumulatingResource(
        name="distance",
        initial=0.0
    ))

    from opencg.solver import ColumnGeneration, CGConfig
    cg = ColumnGeneration(problem, config=CGConfig(max_iterations=30, verbose=True))
    return cg.solve()

# Usage
depot = (0, 0)
customers = [
    (10, 10, 20),   # x, y, demand
    (20, 5, 15),
    (15, 20, 25),
    (5, 15, 10),
]
vehicle_capacity = 50

solution = solve_vrp(depot, customers, vehicle_capacity)
```

---

### Example 2: Time-Space Network (Crew Pairing Style)

```python
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
from opencg.core.problem import Problem, CoverConstraint, CoverType
from opencg.core.resource import AccumulatingResource, HomeBaseResource

def build_crew_network(flights, bases, max_duty_time=10.0):
    """
    Build a time-space network for crew pairing.

    Args:
        flights: List of (id, origin, dest, dep_time, arr_time, cost)
        bases: List of crew base locations
        max_duty_time: Maximum duty time in hours
    """
    network = Network()

    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=float('inf'))
    network.source = source
    network.sink = sink

    # Create station nodes for each (location, time) pair
    # We need nodes at departure and arrival times
    station_nodes = {}  # (location, time) -> node

    for fid, origin, dest, dep, arr, cost in flights:
        # Departure station
        key_dep = (origin, dep)
        if key_dep not in station_nodes:
            node = network.add_node(NodeType.STATION, time=dep)
            node.set_attribute("location", origin)
            station_nodes[key_dep] = node

        # Arrival station
        key_arr = (dest, arr)
        if key_arr not in station_nodes:
            node = network.add_node(NodeType.STATION, time=arr)
            node.set_attribute("location", dest)
            station_nodes[key_arr] = node

    # Source arcs (start duty at a base)
    for base in bases:
        # Find earliest station at this base
        base_stations = [(k, n) for k, n in station_nodes.items() if k[0] == base]
        base_stations.sort(key=lambda x: x[0][1])  # Sort by time

        for (loc, time), station in base_stations:
            arc = network.add_arc(source, station, cost=0.0)
            arc.arc_type = ArcType.SOURCE_ARC
            arc.set_attribute("base", base)

    # Flight arcs
    for fid, origin, dest, dep, arr, cost in flights:
        dep_node = station_nodes[(origin, dep)]
        arr_node = station_nodes[(dest, arr)]

        arc = network.add_arc(dep_node, arr_node, cost=cost)
        arc.arc_type = ArcType.FLIGHT_ARC
        arc.set_consumption("duty_time", arr - dep)
        arc.set_attribute("flight_id", fid)

    # Connection arcs (same location, wait for next flight)
    locations = set(k[0] for k in station_nodes.keys())
    for loc in locations:
        loc_nodes = [(k[1], n) for k, n in station_nodes.items() if k[0] == loc]
        loc_nodes.sort()  # Sort by time

        for i in range(len(loc_nodes) - 1):
            time1, node1 = loc_nodes[i]
            time2, node2 = loc_nodes[i + 1]

            wait_time = time2 - time1
            if wait_time <= 4.0:  # Max 4 hour connection
                arc = network.add_arc(node1, node2, cost=0.0)
                arc.arc_type = ArcType.CONNECTION_ARC
                arc.set_consumption("duty_time", wait_time)

    # Sink arcs (end duty at a base)
    for base in bases:
        base_stations = [(k, n) for k, n in station_nodes.items() if k[0] == base]

        for (loc, time), station in base_stations:
            arc = network.add_arc(station, sink, cost=0.0)
            arc.arc_type = ArcType.SINK_ARC
            arc.set_attribute("base", base)

    return network

def solve_crew_pairing(flights, bases):
    """Solve crew pairing problem."""
    network = build_crew_network(flights, bases)

    problem = Problem(
        name="crew_pairing",
        network=network,
        cover_type=CoverType.PARTITIONING
    )

    # Cover each flight
    for fid, *_ in flights:
        problem.add_cover_constraint(CoverConstraint(item_id=fid))

    # Duty time resource
    problem.add_resource(AccumulatingResource(
        name="duty_time",
        initial=0.0,
        max_value=10.0
    ))

    # Home base resource
    problem.add_resource(HomeBaseResource(name="base"))

    from opencg.solver import ColumnGeneration, CGConfig
    cg = ColumnGeneration(problem, config=CGConfig(max_iterations=50, verbose=True))
    return cg.solve()

# Usage
flights = [
    # (id, origin, dest, dep_time, arr_time, cost)
    (0, "YUL", "YYZ", 6.0, 7.5, 100),
    (1, "YYZ", "YUL", 9.0, 10.5, 100),
    (2, "YUL", "YOW", 8.0, 8.5, 50),
    (3, "YOW", "YUL", 12.0, 12.5, 50),
    (4, "YYZ", "YOW", 11.0, 11.5, 50),
    (5, "YOW", "YYZ", 14.0, 14.5, 50),
]
bases = ["YUL", "YYZ"]

solution = solve_crew_pairing(flights, bases)
```

---

### Example 3: Cutting Stock Network

```python
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
from opencg.core.problem import Problem, CoverConstraint, CoverType
from opencg.core.resource import AccumulatingResource

def build_cutting_stock_network(roll_width, item_sizes):
    """
    Build network for cutting stock.

    Simple structure: source -> item nodes -> sink
    Each path represents one cutting pattern.
    """
    network = Network()

    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=0)
    network.source = source
    network.sink = sink

    # One node per item type (can visit multiple times for multiple copies)
    item_nodes = []
    for i, size in enumerate(item_sizes):
        node = network.add_node(NodeType.TASK, time=0)
        node.set_attribute("item_id", i)
        node.set_attribute("size", size)
        item_nodes.append(node)

    # Source to each item
    for i, node in enumerate(item_nodes):
        size = item_sizes[i]
        arc = network.add_arc(source, node, cost=0.0)
        arc.arc_type = ArcType.SOURCE_ARC
        arc.set_consumption("width", size)
        arc.set_attribute("item_id", i)

    # Item to item arcs (can cut multiple items per roll)
    for i, node_i in enumerate(item_nodes):
        for j, node_j in enumerate(item_nodes):
            size_j = item_sizes[j]
            arc = network.add_arc(node_i, node_j, cost=0.0)
            arc.arc_type = ArcType.FLIGHT_ARC
            arc.set_consumption("width", size_j)
            arc.set_attribute("item_id", j)

    # Item to sink
    for i, node in enumerate(item_nodes):
        arc = network.add_arc(node, sink, cost=0.0)
        arc.arc_type = ArcType.SINK_ARC

    # Source directly to sink (empty pattern - should be very expensive)
    arc = network.add_arc(source, sink, cost=1.0)
    arc.arc_type = ArcType.SINK_ARC

    return network

def solve_cutting_stock(roll_width, item_sizes, item_demands):
    """Solve cutting stock problem."""
    network = build_cutting_stock_network(roll_width, item_sizes)

    problem = Problem(
        name="cutting_stock",
        network=network,
        cover_type=CoverType.COVERING  # Need at least demand of each item
    )

    # Cover constraints - one per demand unit
    item_idx = 0
    for i, demand in enumerate(item_demands):
        for d in range(demand):
            problem.add_cover_constraint(CoverConstraint(
                item_id=item_idx,
                name=f"item_{i}_copy_{d}"
            ))
            item_idx += 1

    # Width resource
    problem.add_resource(AccumulatingResource(
        name="width",
        initial=0.0,
        max_value=roll_width
    ))

    from opencg.solver import ColumnGeneration, CGConfig
    cg = ColumnGeneration(problem, config=CGConfig(max_iterations=30, verbose=True))
    return cg.solve()
```

---

## Tips and Best Practices

### 1. Index vs Attribute for Item Coverage

**Method A**: Arc index equals item ID (simple but inflexible)
```python
# Arc index 5 covers item 5
arc = network.add_arc(source, node, cost=0)  # Gets index automatically
```

**Method B**: Use attributes (recommended)
```python
arc.set_attribute("customer_id", 5)
arc.set_attribute("flight_id", 123)
```

### 2. Validate Your Network

```python
def validate_network(network):
    """Check network is properly constructed."""
    assert network.source is not None, "No source node"
    assert network.sink is not None, "No sink node"

    # Check connectivity
    reachable = set()
    stack = [network.source]
    while stack:
        node = stack.pop()
        if node.index in reachable:
            continue
        reachable.add(node.index)
        for arc in network.outgoing_arcs(node):
            stack.append(arc.target_node)

    assert network.sink.index in reachable, "Sink not reachable from source"

    print(f"Network valid: {network.num_nodes} nodes, {network.num_arcs} arcs")
```

### 3. Debug Path Generation

```python
def print_path(network, arc_indices):
    """Print a path for debugging."""
    print("Path:")
    for idx in arc_indices:
        arc = network.get_arc(idx)
        src = arc.source_node
        tgt = arc.target_node
        print(f"  {src.index}({src.node_type.name}) -> {tgt.index}({tgt.node_type.name})")
        print(f"    cost={arc.cost}, type={arc.arc_type.name}")
        for key in ["customer_id", "flight_id", "item_id"]:
            val = arc.get_attribute(key, None)
            if val is not None:
                print(f"    {key}={val}")
```

---

## Next Steps

- [Custom Resources](custom_resources.md) - Add new constraint types
- [Custom Pricing](custom_pricing.md) - Implement specialized algorithms
- [Custom Applications](custom_application.md) - Full application examples
