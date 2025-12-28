# Custom Applications Guide

This guide explains how to model new optimization problems using OpenCG's column generation framework.

## Overview

OpenCG's column generation framework can solve any problem that can be formulated as:

1. **Master Problem**: Set covering/partitioning LP
2. **Pricing Problem**: Shortest path with resource constraints (SPPRC)

This includes:
- Vehicle routing problems (VRP, VRPTW, CVRP)
- Cutting stock / Bin packing
- Crew scheduling (pairing, rostering)
- Machine scheduling
- Network design

## Modeling Steps

### Step 1: Identify the Structure

Column generation works when your problem has:

1. **Items to cover**: Flights, customers, orders, tasks
2. **Columns (patterns/routes)**: Combinations of items that can be served together
3. **Path structure**: Columns can be represented as paths in a network

### Step 2: Design the Network

The network is a directed graph where:
- **Nodes** represent states (location, time, resource state)
- **Arcs** represent activities (flights, deliveries, connections)
- **Paths** from source to sink represent valid columns

### Step 3: Define Resources

Resources constrain which paths are feasible:
- Time limits, capacity, cost budgets
- Visit constraints (elementarity)
- Time windows

### Step 4: Implement the Application

Create a solver function that:
1. Builds the network
2. Creates the Problem object
3. Runs column generation
4. Extracts the solution

---

## Example 1: Parallel Machine Scheduling

**Problem**: Assign jobs to machines to minimize makespan (maximum completion time).

**Structure**:
- Items: Jobs to be scheduled
- Columns: Job sequences on a single machine
- Resources: Time, machine capacity

```python
from dataclasses import dataclass
from opencg.core.problem import Problem, CoverConstraint, CoverType
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
from opencg.core.resource import AccumulatingResource
from opencg.solver import ColumnGeneration, CGConfig

@dataclass
class Job:
    id: int
    processing_time: float
    release_time: float = 0.0
    deadline: float = float('inf')

@dataclass
class MachineSchedulingInstance:
    jobs: list[Job]
    num_machines: int
    name: str = "scheduling"

def build_scheduling_network(instance: MachineSchedulingInstance) -> Network:
    """Build time-indexed network for machine scheduling."""
    network = Network()

    # Source and sink
    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=float('inf'))
    network.source = source
    network.sink = sink

    # Create node for each job
    job_nodes = {}
    for job in instance.jobs:
        node = network.add_node(NodeType.TASK, time=job.release_time)
        node.set_attribute("job_id", job.id)
        node.set_attribute("processing_time", job.processing_time)
        job_nodes[job.id] = node

    # Source to job arcs (start processing)
    for job in instance.jobs:
        arc = network.add_arc(source, job_nodes[job.id], cost=0.0)
        arc.arc_type = ArcType.SOURCE_ARC
        arc.set_consumption("time", job.processing_time)
        arc.set_attribute("job_id", job.id)

    # Job to job arcs (sequence)
    for job1 in instance.jobs:
        for job2 in instance.jobs:
            if job1.id == job2.id:
                continue

            # Can sequence job2 after job1
            arc = network.add_arc(
                job_nodes[job1.id],
                job_nodes[job2.id],
                cost=0.0
            )
            arc.arc_type = ArcType.FLIGHT_ARC  # Activity arc
            arc.set_consumption("time", job2.processing_time)
            arc.set_attribute("job_id", job2.id)

    # Job to sink arcs (finish)
    for job in instance.jobs:
        arc = network.add_arc(job_nodes[job.id], sink, cost=0.0)
        arc.arc_type = ArcType.SINK_ARC

    return network

def solve_scheduling(instance: MachineSchedulingInstance, **kwargs) -> dict:
    """Solve parallel machine scheduling using column generation."""

    # Build network
    network = build_scheduling_network(instance)

    # Create problem
    problem = Problem(
        name=instance.name,
        network=network,
        cover_type=CoverType.PARTITIONING  # Each job exactly once
    )

    # Add covering constraints (one per job)
    for job in instance.jobs:
        problem.add_cover_constraint(CoverConstraint(
            item_id=job.id,
            name=f"job_{job.id}"
        ))

    # Add resources
    problem.add_resource(AccumulatingResource(
        name="time",
        initial=0.0,
        max_value=float('inf')  # Makespan handled differently
    ))

    # Solve
    cg = ColumnGeneration(problem, config=CGConfig(**kwargs))
    solution = cg.solve()

    # Extract schedule
    schedule = extract_schedule(solution, instance)
    return schedule

def extract_schedule(solution, instance) -> dict:
    """Extract machine assignments from CG solution."""
    schedules = []

    for col_id, value in solution.column_values.items():
        if value > 0.5:  # Column is used
            column = solution.columns[col_id]
            jobs_in_sequence = []

            for arc_idx in column.arc_indices:
                arc = solution.network.get_arc(arc_idx)
                job_id = arc.get_attribute("job_id", None)
                if job_id is not None:
                    jobs_in_sequence.append(job_id)

            schedules.append({
                "jobs": jobs_in_sequence,
                "makespan": column.resource_values.get("time", 0)
            })

    return {
        "machine_schedules": schedules,
        "makespan": max(s["makespan"] for s in schedules) if schedules else 0
    }
```

**Usage:**

```python
instance = MachineSchedulingInstance(
    jobs=[
        Job(id=0, processing_time=3),
        Job(id=1, processing_time=2),
        Job(id=2, processing_time=4),
        Job(id=3, processing_time=1),
        Job(id=4, processing_time=2),
    ],
    num_machines=2
)

schedule = solve_scheduling(instance, max_iterations=20, verbose=True)
print(f"Makespan: {schedule['makespan']}")
for i, machine in enumerate(schedule['machine_schedules']):
    print(f"Machine {i}: Jobs {machine['jobs']}")
```

---

## Example 2: Multi-Depot Vehicle Routing

**Problem**: Route vehicles from multiple depots to serve customers.

```python
from dataclasses import dataclass
from opencg.core.problem import Problem, CoverConstraint, CoverType
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
from opencg.core.resource import AccumulatingResource, HomeBaseResource
import math

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    service_time: float = 0.0

@dataclass
class Depot:
    id: int
    x: float
    y: float
    num_vehicles: int
    vehicle_capacity: float

@dataclass
class MDVRPInstance:
    depots: list[Depot]
    customers: list[Customer]
    name: str = "mdvrp"

def distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def build_mdvrp_network(instance: MDVRPInstance) -> Network:
    """Build network for multi-depot VRP."""
    network = Network()

    # Global source and sink
    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=float('inf'))
    network.source = source
    network.sink = sink

    # Depot nodes (for leaving/returning)
    depot_out_nodes = {}
    depot_in_nodes = {}

    for depot in instance.depots:
        out_node = network.add_node(NodeType.STATION, time=0)
        out_node.set_attribute("depot_id", depot.id)
        out_node.set_attribute("location", (depot.x, depot.y))
        depot_out_nodes[depot.id] = out_node

        in_node = network.add_node(NodeType.STATION, time=0)
        in_node.set_attribute("depot_id", depot.id)
        in_node.set_attribute("location", (depot.x, depot.y))
        depot_in_nodes[depot.id] = in_node

    # Customer nodes
    customer_nodes = {}
    for cust in instance.customers:
        node = network.add_node(NodeType.CUSTOMER, time=0)
        node.set_attribute("customer_id", cust.id)
        node.set_attribute("demand", cust.demand)
        node.set_attribute("location", (cust.x, cust.y))
        customer_nodes[cust.id] = node

    # Source to depot arcs (start from depot)
    for depot in instance.depots:
        arc = network.add_arc(source, depot_out_nodes[depot.id], cost=0.0)
        arc.arc_type = ArcType.SOURCE_ARC
        arc.set_attribute("depot_id", depot.id)

    # Depot to customer arcs
    for depot in instance.depots:
        depot_loc = (depot.x, depot.y)
        for cust in instance.customers:
            cust_loc = (cust.x, cust.y)
            dist = distance(depot_loc, cust_loc)

            arc = network.add_arc(
                depot_out_nodes[depot.id],
                customer_nodes[cust.id],
                cost=dist
            )
            arc.set_consumption("distance", dist)
            arc.set_consumption("capacity", cust.demand)
            arc.set_attribute("customer_id", cust.id)

    # Customer to customer arcs
    for c1 in instance.customers:
        for c2 in instance.customers:
            if c1.id == c2.id:
                continue

            loc1 = (c1.x, c1.y)
            loc2 = (c2.x, c2.y)
            dist = distance(loc1, loc2)

            arc = network.add_arc(
                customer_nodes[c1.id],
                customer_nodes[c2.id],
                cost=dist
            )
            arc.set_consumption("distance", dist)
            arc.set_consumption("capacity", c2.demand)
            arc.set_attribute("customer_id", c2.id)

    # Customer to depot arcs (return)
    for cust in instance.customers:
        cust_loc = (cust.x, cust.y)
        for depot in instance.depots:
            depot_loc = (depot.x, depot.y)
            dist = distance(cust_loc, depot_loc)

            arc = network.add_arc(
                customer_nodes[cust.id],
                depot_in_nodes[depot.id],
                cost=dist
            )
            arc.arc_type = ArcType.SINK_ARC
            arc.set_consumption("distance", dist)
            arc.set_attribute("depot_id", depot.id)

    # Depot to sink arcs
    for depot in instance.depots:
        arc = network.add_arc(depot_in_nodes[depot.id], sink, cost=0.0)
        arc.arc_type = ArcType.SINK_ARC

    return network

def solve_mdvrp(instance: MDVRPInstance, **kwargs):
    """Solve multi-depot VRP using column generation."""

    network = build_mdvrp_network(instance)

    problem = Problem(
        name=instance.name,
        network=network,
        cover_type=CoverType.PARTITIONING
    )

    # Cover each customer
    for cust in instance.customers:
        problem.add_cover_constraint(CoverConstraint(
            item_id=cust.id,
            name=f"customer_{cust.id}"
        ))

    # Capacity resource
    max_capacity = max(d.vehicle_capacity for d in instance.depots)
    problem.add_resource(AccumulatingResource(
        name="capacity",
        initial=0.0,
        max_value=max_capacity
    ))

    # Distance resource (for objective)
    problem.add_resource(AccumulatingResource(
        name="distance",
        initial=0.0
    ))

    # Home base resource (return to same depot)
    problem.add_resource(HomeBaseResource(name="depot"))

    # Solve
    from opencg.solver import ColumnGeneration, CGConfig
    cg = ColumnGeneration(problem, config=CGConfig(**kwargs))
    solution = cg.solve()

    return extract_mdvrp_solution(solution, instance)
```

---

## Example 3: Shift Scheduling

**Problem**: Assign employees to shifts to cover demand.

```python
from dataclasses import dataclass
from opencg.core.problem import Problem, CoverConstraint, CoverType
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType
from opencg.core.resource import AccumulatingResource, IntervalResource

@dataclass
class TimeSlot:
    id: int
    start_hour: int
    end_hour: int
    demand: int  # Number of employees needed

@dataclass
class ShiftType:
    id: int
    name: str
    start_hour: int
    duration: int
    cost: float

@dataclass
class ShiftSchedulingInstance:
    time_slots: list[TimeSlot]
    shift_types: list[ShiftType]
    max_hours_per_week: float = 40.0
    min_rest_between_shifts: float = 8.0
    name: str = "shift_scheduling"

def build_shift_network(instance: ShiftSchedulingInstance) -> Network:
    """Build network for shift scheduling."""
    network = Network()

    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=168)  # Week in hours
    network.source = source
    network.sink = sink

    # Create shift nodes
    shift_nodes = []
    for shift in instance.shift_types:
        # Could have multiple instances of each shift type per week
        for day in range(7):
            node = network.add_node(NodeType.TASK, time=day * 24 + shift.start_hour)
            node.set_attribute("shift_type", shift.id)
            node.set_attribute("day", day)
            node.set_attribute("duration", shift.duration)
            shift_nodes.append((node, shift, day))

    # Source to shift arcs
    for node, shift, day in shift_nodes:
        arc = network.add_arc(source, node, cost=shift.cost)
        arc.arc_type = ArcType.SOURCE_ARC
        arc.set_consumption("hours", shift.duration)

        # Mark which time slots this shift covers
        for slot in instance.time_slots:
            shift_start = day * 24 + shift.start_hour
            shift_end = shift_start + shift.duration
            slot_start = slot.start_hour
            slot_end = slot.end_hour

            # Check overlap
            if shift_start < slot_end and shift_end > slot_start:
                arc.add_to_attribute("covers_slots", slot.id)

    # Shift to shift arcs (respecting rest time)
    for n1, s1, d1 in shift_nodes:
        end_time1 = d1 * 24 + s1.start_hour + s1.duration

        for n2, s2, d2 in shift_nodes:
            start_time2 = d2 * 24 + s2.start_hour

            # Must have minimum rest
            if start_time2 >= end_time1 + instance.min_rest_between_shifts:
                arc = network.add_arc(n1, n2, cost=s2.cost)
                arc.set_consumption("hours", s2.duration)

                for slot in instance.time_slots:
                    shift_start = d2 * 24 + s2.start_hour
                    shift_end = shift_start + s2.duration
                    if shift_start < slot.end_hour and shift_end > slot.start_hour:
                        arc.add_to_attribute("covers_slots", slot.id)

    # Shift to sink arcs
    for node, shift, day in shift_nodes:
        arc = network.add_arc(node, sink, cost=0.0)
        arc.arc_type = ArcType.SINK_ARC

    return network

def solve_shift_scheduling(instance: ShiftSchedulingInstance, **kwargs):
    """Solve shift scheduling using column generation."""

    network = build_shift_network(instance)

    problem = Problem(
        name=instance.name,
        network=network,
        cover_type=CoverType.COVERING  # At least demand employees
    )

    # Cover each time slot (need 'demand' employees)
    for slot in instance.time_slots:
        for d in range(slot.demand):  # Create 'demand' constraints
            problem.add_cover_constraint(CoverConstraint(
                item_id=slot.id * 100 + d,  # Unique ID
                name=f"slot_{slot.id}_emp_{d}"
            ))

    # Weekly hours limit
    problem.add_resource(AccumulatingResource(
        name="hours",
        initial=0.0,
        max_value=instance.max_hours_per_week
    ))

    # Solve
    from opencg.solver import ColumnGeneration, CGConfig
    cg = ColumnGeneration(problem, config=CGConfig(**kwargs))
    return cg.solve()
```

---

## Application Design Patterns

### Pattern 1: Instance + Solver Function

```python
@dataclass
class MyProblemInstance:
    """Problem data."""
    ...

def solve_my_problem(instance: MyProblemInstance, **config) -> MySolution:
    """Main entry point."""
    network = build_network(instance)
    problem = create_problem(network, instance)
    solution = run_column_generation(problem, **config)
    return extract_solution(solution, instance)
```

### Pattern 2: Separate Network Builder

```python
class MyNetworkBuilder:
    """Builds problem-specific network."""

    def __init__(self, instance):
        self.instance = instance

    def build(self) -> Network:
        network = Network()
        self._add_nodes(network)
        self._add_arcs(network)
        return network

    def _add_nodes(self, network):
        ...

    def _add_arcs(self, network):
        ...
```

### Pattern 3: Custom Pricing for Specific Structure

```python
class MyProblemPricing(PricingProblem):
    """Problem-specific pricing exploiting structure."""

    def _solve_impl(self):
        # Use problem structure for efficient solving
        ...
```

---

## Best Practices

### 1. Start Simple

Begin with a basic network and add complexity:

```python
# Start with
network = build_simple_network(instance)

# Then add time windows, capacity, etc.
network = build_network_with_time_windows(instance)
network = build_network_with_capacity(instance)
```

### 2. Test Incrementally

```python
# Test network construction
network = build_network(instance)
print(f"Nodes: {network.num_nodes}, Arcs: {network.num_arcs}")

# Test feasibility (should have paths)
paths = find_all_paths(network)
print(f"Found {len(paths)} paths")

# Test CG with small instance
solution = solve(small_instance, max_iterations=5, verbose=True)
```

### 3. Use Appropriate Cover Type

| Problem | Cover Type |
|---------|------------|
| VRP (each customer once) | PARTITIONING |
| Bin packing (item in one bin) | PARTITIONING |
| Covering (at least once) | COVERING |
| Crew pairing (cover flights) | PARTITIONING |

### 4. Choose Resources Wisely

| Resource | Use Case |
|----------|----------|
| `AccumulatingResource` | Time, distance, cost |
| `IntervalResource` | Connection time, break duration |
| `StateResource` | Visited nodes, elementarity |
| `TimeWindowResource` | VRPTW, scheduling |
| `HomeBaseResource` | Crew pairing, multi-depot |

### 5. Handle Arc-Item Mapping

Two common patterns:

**Pattern A**: Arc index = item ID (simple)
```python
arc.index = customer_id
```

**Pattern B**: Arc attribute (flexible)
```python
arc.set_attribute("customer_id", customer_id)
```

---

## Debugging Tips

### Check Network Connectivity

```python
from collections import deque

def check_connectivity(network):
    """Verify all nodes reachable from source."""
    visited = set()
    queue = deque([network.source])

    while queue:
        node = queue.popleft()
        if node.index in visited:
            continue
        visited.add(node.index)

        for arc in network.outgoing_arcs(node):
            queue.append(arc.target_node)

    unreachable = set(n.index for n in network.nodes) - visited
    if unreachable:
        print(f"Warning: {len(unreachable)} unreachable nodes")
```

### Visualize Network

```python
def visualize_network(network):
    """Create simple visualization."""
    import matplotlib.pyplot as plt

    for arc in network.arcs:
        src = arc.source_node
        tgt = arc.target_node
        plt.arrow(src.time, src.index, tgt.time - src.time, tgt.index - src.index)

    plt.xlabel("Time")
    plt.ylabel("Node")
    plt.show()
```

### Print Column Details

```python
def print_column(column, network):
    """Print column path details."""
    print(f"Column {column.column_id}:")
    print(f"  Cost: {column.cost:.2f}")
    print(f"  Reduced cost: {column.reduced_cost:.4f}")
    print(f"  Covers: {column.covered_items}")
    print("  Path:")
    for arc_idx in column.arc_indices:
        arc = network.get_arc(arc_idx)
        print(f"    {arc.source_node} -> {arc.target_node} (cost={arc.cost:.2f})")
```

---

## Next Steps

- [Custom Resources](custom_resources.md) - Define new resource constraints
- [Custom Pricing](custom_pricing.md) - Implement custom SPPRC algorithms
- [Custom Master Problem](custom_master.md) - Use different LP/MIP solvers
