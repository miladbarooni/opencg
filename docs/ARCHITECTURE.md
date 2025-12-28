# OpenCG Architecture

This document describes the architecture and design decisions of OpenCG, a hybrid Python/C++ column generation framework.

---

## The Big Picture

OpenCG solves optimization problems using **Column Generation**, which decomposes a large problem into:

1. **Master Problem**: Selects which columns (routes, patterns, schedules) to use
2. **Pricing Problem**: Finds new promising columns by solving a shortest path problem

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Column Generation Loop                          │
│                                                                        │
│   ┌─────────────┐       duals (π)       ┌─────────────────┐           │
│   │   MASTER    │ ────────────────────► │     PRICING     │           │
│   │   PROBLEM   │                       │     PROBLEM     │           │
│   │             │ ◄──────────────────── │                 │           │
│   │  (LP/MIP)   │       columns (λ)     │    (SPPRC)      │           │
│   └─────────────┘                       └─────────────────┘           │
│         │                                       │                      │
│         ▼                                       ▼                      │
│   ┌─────────────┐                       ┌─────────────────┐           │
│   │   HiGHS/    │                       │    NETWORK      │           │
│   │   Gurobi/   │                       │                 │           │
│   │   CPLEX     │                       │  Nodes + Arcs   │           │
│   └─────────────┘                       │  + Resources    │           │
│                                         └─────────────────┘           │
└────────────────────────────────────────────────────────────────────────┘
```

**See also:** [Custom Master](custom_master.md) | [Custom Pricing](custom_pricing.md) | [Building Networks](building_networks.md)

### What Each Component Does

| Component | Input | Output | Customizable? |
|-----------|-------|--------|---------------|
| **Network** | Your problem data | Directed graph | You always build this |
| **Resources** | Arc consumptions | Feasibility check | Built-in + custom |
| **Master** | Columns, constraints | LP solution, duals | HiGHS default, can use Gurobi/CPLEX |
| **Pricing** | Network, duals | New columns | Labeling default, can use DP/MIP |
| **Solver** | Problem | Final solution | CG loop, can control manually |

---

## OpenCG is Network-Agnostic

**This is the key design principle.** OpenCG doesn't care what your nodes and arcs represent - it just needs:

1. A directed graph with source and sink
2. Arc costs and resource consumptions
3. Coverage information (which arcs cover which items)

```python
# All of these are valid OpenCG networks:

# VRP: nodes = customers, arcs = travel between customers
# Crew Pairing: nodes = (airport, time), arcs = flights/connections
# Cutting Stock: nodes = items, arcs = cutting decisions
# Machine Scheduling: nodes = (machine, time), arcs = job assignments
# Shift Scheduling: nodes = (employee, time), arcs = shifts
# Any shortest-path-based problem you can define!
```

You bring the network structure. OpenCG provides the column generation machinery.

---

## How Components Connect

```
┌──────────────────────────────────────────────────────────────────────┐
│                           PROBLEM                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                         NETWORK                                 │  │
│  │   ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐             │  │
│  │   │SOURCE│────►│ NODE │────►│ NODE │────►│ SINK │             │  │
│  │   └──────┘     └──────┘     └──────┘     └──────┘             │  │
│  │       │            │            │            ▲                 │  │
│  │       └────────────┴────────────┴────────────┘                 │  │
│  │                      ARCs                                       │  │
│  │           (cost, consumptions, covered_items)                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐  │
│  │   RESOURCES     │    │         COVER CONSTRAINTS               │  │
│  │   - time        │    │   - item_0 must be covered              │  │
│  │   - capacity    │    │   - item_1 must be covered              │  │
│  │   - visited     │    │   - ...                                 │  │
│  └─────────────────┘    └─────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │           COLUMN GENERATION             │
              │                                         │
              │  ┌───────────────┐ ┌─────────────────┐ │
              │  │ MasterProblem │ │ PricingProblem  │ │
              │  │               │ │                 │ │
              │  │ - add_column  │ │ - set_duals     │ │
              │  │ - solve_lp    │ │ - solve         │ │
              │  │ - get_duals   │ │ - get_columns   │ │
              │  └───────────────┘ └─────────────────┘ │
              └─────────────────────────────────────────┘
```

---

## The Column Generation Loop (Detailed)

Here's exactly what happens in each iteration:

```python
# This is what ColumnGeneration.solve() does internally

def solve(self):
    # 1. Initialize with some feasible columns
    initial_columns = self._generate_initial_columns()
    for col in initial_columns:
        self.master.add_column(col)

    for iteration in range(max_iterations):
        # 2. Solve LP relaxation of master problem
        lp_solution = self.master.solve_lp()

        # 3. Get dual values (shadow prices)
        duals = self.master.get_dual_values()
        # duals[item_id] = marginal value of covering item_id

        # 4. Pass duals to pricing problem
        self.pricing.set_dual_values(duals)

        # 5. Solve pricing (find negative reduced cost paths)
        pricing_solution = self.pricing.solve()

        # 6. Check termination
        if pricing_solution.status == NO_COLUMNS:
            # No negative RC columns = LP optimal
            break

        # 7. Add new columns to master
        for column in pricing_solution.columns:
            self.master.add_column(column)

    # 8. Solve final IP (optional)
    ip_solution = self.master.solve_ip()

    return ip_solution
```

---

## Bringing Your Own Network (Step by Step)

### Step 1: Think About Your Problem Structure

Ask yourself:
- **What are the nodes?** (locations, times, states, tasks)
- **What are the arcs?** (transitions, activities, connections)
- **What makes a path valid?** (resource constraints)
- **What items need covering?** (customers, flights, jobs)

### Step 2: Build the Network

```python
from opencg.core.network import Network
from opencg.core.node import Node, NodeType
from opencg.core.arc import Arc, ArcType

def build_my_network(my_data):
    network = Network()

    # Always need source and sink
    source = network.add_node(NodeType.SOURCE, time=0)
    sink = network.add_node(NodeType.SINK, time=float('inf'))
    network.source = source
    network.sink = sink

    # Add your nodes (whatever makes sense for your problem)
    my_nodes = {}
    for item in my_data:
        node = network.add_node(NodeType.TASK, time=item.time)
        node.set_attribute("my_id", item.id)
        my_nodes[item.id] = node

    # Add your arcs
    for connection in my_data.connections:
        arc = network.add_arc(
            my_nodes[connection.from_id],
            my_nodes[connection.to_id],
            cost=connection.cost
        )
        # Set resource consumptions
        arc.set_consumption("time", connection.duration)
        arc.set_consumption("capacity", connection.load)

        # Mark which items this arc covers
        arc.set_attribute("covers_item", connection.item_id)

    # Connect source to starting nodes
    for start_node in starting_nodes:
        arc = network.add_arc(source, start_node, cost=0)
        arc.arc_type = ArcType.SOURCE_ARC

    # Connect ending nodes to sink
    for end_node in ending_nodes:
        arc = network.add_arc(end_node, sink, cost=0)
        arc.arc_type = ArcType.SINK_ARC

    return network
```

### Step 3: Define Resources

```python
from opencg.core.resource import AccumulatingResource, StateResource

# Time limit
time_resource = AccumulatingResource(
    name="time",
    initial=0.0,
    max_value=8.0  # 8 hour limit
)

# Capacity limit
capacity_resource = AccumulatingResource(
    name="capacity",
    initial=0.0,
    max_value=100.0
)

# Elementarity (no revisiting)
visited_resource = StateResource(
    name="visited",
    forbidden_revisit=True
)
```

### Step 4: Create Problem

```python
from opencg.core.problem import Problem, CoverConstraint, CoverType

problem = Problem(
    name="my_problem",
    network=network,
    cover_type=CoverType.PARTITIONING  # or COVERING
)

# Add resources
problem.add_resource(time_resource)
problem.add_resource(capacity_resource)

# Add cover constraints
for item_id in items_to_cover:
    problem.add_cover_constraint(CoverConstraint(item_id=item_id))

# Map arcs to items they cover
problem.set_arc_coverage_mapping(arc_to_items_map)
```

### Step 5: Solve

```python
from opencg.solver import ColumnGeneration, CGConfig

cg = ColumnGeneration(
    problem,
    config=CGConfig(
        max_iterations=100,
        verbose=True
    )
)

solution = cg.solve()
```

---

## Manual Control of CG Loop

For research or custom branching, you can control the loop manually:

```python
from opencg.master.highs import HiGHSMasterProblem
from opencg.pricing.labeling import LabelingAlgorithm
from opencg.pricing.base import PricingConfig

# Create components separately
master = HiGHSMasterProblem(problem)
pricing = LabelingAlgorithm(
    problem,
    config=PricingConfig(max_columns=50)
)

# Add initial columns
for col in generate_initial_columns(problem):
    master.add_column(col)

# Manual CG loop with full control
for iteration in range(100):
    # Solve master
    lp_sol = master.solve_lp()
    print(f"Iter {iteration}: LP = {lp_sol.objective_value:.2f}")

    # Get duals
    duals = master.get_dual_values()

    # Price
    pricing.set_dual_values(duals)
    price_sol = pricing.solve()

    # Check optimality
    if not price_sol.columns:
        print("Optimal!")
        break

    # Add columns (maybe filter, limit, etc.)
    for col in price_sol.columns[:10]:  # Add at most 10
        if col.reduced_cost < -1e-6:
            master.add_column(col)

    # Custom logic: branching, stabilization, cuts, etc.
    if iteration % 10 == 0:
        master.enable_stabilization(method='boxstep', delta=5.0)

# Final IP
ip_sol = master.solve_ip()
```

---

## Customization Points

You can customize at any level:

| Component | Default | Customize When | Guide |
|-----------|---------|----------------|-------|
| **Network** | You build it | Always | [Building Networks](building_networks.md) |
| **Resources** | Built-in types | Need special constraint logic | [Custom Resources](custom_resources.md) |
| **Master** | HiGHS LP/MIP | Want Gurobi, CPLEX, or custom | [Custom Master](custom_master.md) |
| **Pricing** | Labeling algorithm | Have problem-specific algorithm | [Custom Pricing](custom_pricing.md) |
| **Application** | Use solver directly | Want packaged solution | [Custom Applications](custom_application.md) |

---

## Design Philosophy

OpenCG follows a **layered architecture** with clear separation between:

1. **User-facing Python API** - Easy to use, extend, and experiment with
2. **Performance-critical C++ core** - Fast labeling algorithms with Python bindings
3. **Application layer** - Domain-specific solvers built on the core framework

The key insight is that column generation spends most time in the **pricing subproblem** (SPPRC labeling), so we optimize that in C++ while keeping everything else in Python for flexibility.

## Layer Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│  cutting_stock.py | crew_pairing/ | vrp/                        │
│  Ready-to-use solvers for specific problem types                │
├─────────────────────────────────────────────────────────────────┤
│                    Solver Layer                                  │
│  solver/column_generation.py                                     │
│  Orchestrates master problem + pricing subproblem iterations    │
├─────────────────────────────────────────────────────────────────┤
│        Master Problem          │       Pricing Problem          │
│   master/highs.py              │   pricing/labeling.py          │
│   master/cplex.py              │   pricing/fast_per_source.py   │
│   LP/MIP formulation           │   SPPRC algorithms             │
├─────────────────────────────────────────────────────────────────┤
│                    Core Layer (Python)                          │
│  core/network.py | core/arc.py | core/node.py | core/column.py │
│  core/resource.py | core/problem.py                             │
├─────────────────────────────────────────────────────────────────┤
│                    C++ Core (via pybind11)                      │
│  Network, Label, LabelPool, LabelingAlgorithm                   │
│  ParallelLabelingSolver                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### Network (Time-Space Graph)

The network is a directed graph where:
- **Nodes** represent locations at specific times (e.g., airport at 10:00)
- **Arcs** represent activities (flights, connections, rest periods)
- **Resources** are consumed along arcs

```python
# Python API
network = Network()
node1 = network.add_node(NodeType.STATION, time=100)
node2 = network.add_node(NodeType.STATION, time=200)
arc = network.add_arc(node1, node2, cost=50.0)
arc.set_consumption("time", 100.0)
```

```cpp
// C++ implementation (src/core/network.hpp)
struct Arc {
    int32_t source;
    int32_t target;
    double cost;
    std::vector<double> resource_consumption;
    std::vector<int32_t> covered_items;
};

class Network {
    std::vector<Arc> arcs_;
    std::vector<std::vector<int32_t>> outgoing_;  // adjacency list
};
```

### Labels (Partial Paths)

Labels represent partial paths in the SPPRC algorithm:

```cpp
// src/core/label.hpp
class Label {
    double cost_;                    // Path cost
    double reduced_cost_;            // Cost - sum of duals
    std::vector<double> resources_;  // Resource consumption
    std::bitset<1024> visited_;      // Visited nodes (elementarity)
    std::vector<int32_t> covered_;   // Covered items (flights)
    Label* predecessor_;             // For path reconstruction
    int32_t arc_index_;              // Last arc taken
};
```

### Columns

Columns represent complete paths (pairings, routes, patterns):

```python
@dataclass(frozen=True)
class Column:
    arc_indices: Tuple[int, ...]      # Arcs in the path
    cost: float                        # Total cost
    covered_items: FrozenSet[int]     # Items covered (flights, customers)
    reduced_cost: Optional[float]     # For pricing
    column_id: Optional[int]          # Unique ID in master
    attributes: Dict[str, Any]        # Extra data (base, route, etc.)
```

## Resource System

Resources are constraints that accumulate along paths. OpenCG supports:

### AccumulatingResource
Simple additive resources (time, distance, cost):
```python
class AccumulatingResource(Resource):
    def extend(self, value, arc):
        return value + arc.get_consumption(self.name)

    def is_feasible(self, value, node):
        return value <= self.max_value
```

### TimeWindowResource
For VRPTW - earliest/latest arrival times:
```python
class TimeWindowResource(Resource):
    def extend(self, value, arc):
        arrival = value + arc.get_consumption("travel_time")
        # Wait if arriving early
        return max(arrival, node.time_window[0])

    def is_feasible(self, value, node):
        return value <= node.time_window[1]
```

### StateResource
For tracking discrete states (e.g., home base):
```python
class StateResource(Resource):
    def extend(self, value, arc):
        if arc.has_attribute("changes_state"):
            return arc.get_attribute("new_state")
        return value
```

## Pricing Algorithms

### Standard Labeling (pricing/labeling.py)
- Mono-directional labeling from source to sink
- Dominance pruning to reduce label count
- Supports elementarity checking

### Per-Source Pricing (pricing/fast_per_source.py)
For crew pairing, runs separate labeling from each source arc:
- Higher coverage (reaches more flights)
- Naturally parallelizable
- Each source has its own network with only relevant sink arcs

### C++ Accelerated Labeling (src/pricing/labeling.hpp)
High-performance implementation with:
- Topological order processing (for DAGs)
- ng-path relaxation (faster than full elementarity)
- Time window propagation
- Label limits per node (beam search)

```cpp
// Key solve loop
while (!queue.empty()) {
    Label* label = queue.pop();
    for (int arc_idx : network_.outgoing(label->node())) {
        Label* extended = extend(label, arc_idx);
        if (extended && !dominated(extended)) {
            queue.push(extended);
        }
    }
}
```

### Parallel Pricing (src/pricing/parallel_labeling.hpp)
Multi-threaded execution using std::async:
```cpp
std::vector<std::future<LabelingResult>> futures;
for (auto& source : sources_) {
    futures.push_back(std::async(std::launch::async, [&]() {
        return source.algorithm.solve();
    }));
}
// Collect results
for (auto& f : futures) {
    results.push_back(f.get());
}
```

## Master Problem

The master problem is the set covering/partitioning LP:

```
minimize    sum_j c_j * x_j
subject to  sum_j a_ij * x_j >= 1   for all items i
            x_j >= 0
```

Where:
- `x_j` = selection of column j
- `c_j` = cost of column j
- `a_ij` = 1 if column j covers item i

### HiGHS Implementation (master/highs.py)
- Open-source LP solver
- Incremental column addition
- Dual value extraction for pricing

### Column Addition
```python
def add_column(self, column: Column):
    # Add variable with cost
    self._highs.addVar(0.0, inf, column.cost)

    # Add to covering constraints
    for item_id in column.covered_items:
        row_idx = self._item_to_row[item_id]
        self._highs.changeCoeff(row_idx, col_idx, 1.0)
```

## Column Generation Loop

The main algorithm in `solver/column_generation.py`:

```python
def solve(self):
    # Initialize with artificial columns
    self._add_artificial_columns()

    while iteration < max_iterations:
        # Solve LP relaxation
        lp_solution = self.master.solve_lp()

        # Get dual values
        duals = self.master.get_dual_values()

        # Solve pricing subproblem
        self.pricing.set_dual_values(duals)
        pricing_solution = self.pricing.solve()

        # Check termination
        if not pricing_solution.columns:
            break  # Optimal

        # Add new columns
        for column in pricing_solution.columns:
            self.master.add_column(column)

        iteration += 1

    # Solve IP for integer solution
    return self.master.solve_ip()
```

## C++/Python Integration

### pybind11 Bindings (src/bindings/)

We expose C++ classes to Python:

```cpp
// pricing_bindings.cpp
py::class_<LabelingAlgorithm>(m, "LabelingAlgorithm")
    .def(py::init<const Network&, size_t,
                  const std::vector<double>&,
                  const LabelingConfig&>())
    .def("set_dual_values", &LabelingAlgorithm::set_dual_values)
    .def("solve", &LabelingAlgorithm::solve,
         py::call_guard<py::gil_scoped_release>())  // Release GIL!
```

### GIL Release for Parallelism

The key to parallel performance is releasing Python's Global Interpreter Lock (GIL) during C++ computation:

```cpp
.def("solve", &LabelingAlgorithm::solve,
     py::call_guard<py::gil_scoped_release>())
```

This allows multiple Python threads to execute C++ code simultaneously.

## Application Layer

### Cutting Stock (applications/cutting_stock.py)
- Dynamic programming pricing (knapsack)
- FFD heuristic for upper bounds
- L2 lower bounds
- BPPLIB benchmark support

### Vehicle Routing (applications/vrp/)
- Time-space network construction
- Capacity resource for CVRP
- Time window resource for VRPTW
- Route extraction from columns

### Crew Pairing (applications/crew_pairing/)
- Multiple resources (duty time, rest, pairing duration)
- Home base constraints (state resource)
- Per-source pricing for high coverage
- Kasirzadeh benchmark support

## Performance Considerations

### Why Per-Source is Faster and Better for Crew Pairing

Single-source pricing:
- One labeling from global source
- May not reach all flights (coverage gaps)
- Long paths, many labels

Per-source pricing:
- Separate labeling per departure base
- Each labeling is smaller (fewer arcs)
- Higher coverage (99.9% vs 96.7%)
- Naturally parallel

### Label Management

Key optimizations in C++:
1. **Object pooling**: Reuse Label objects to avoid allocation
2. **Bucket queue**: O(1) priority queue operations
3. **Bitset visited**: Fast elementarity checking
4. **Early termination**: Stop when enough columns found

### Dominance Checking

For label l1 to dominate l2:
- l1.cost <= l2.cost
- l1.resources[i] <= l2.resources[i] for all i
- l1.visited superset of l2.visited (if elementary)

Dominated labels are discarded, dramatically reducing search space.

## Future Directions

This architecture is designed to support:

1. **Branch-and-Price**: Add branching decisions, node processing
2. **Cutting Planes**: Strengthen LP with valid inequalities
3. **Stabilization**: Dual smoothing, boxstep methods
4. **Bidirectional Labeling**: Forward + backward for long paths

The clean separation of concerns makes it straightforward to add these features without modifying existing code.

---

## Related Documentation

| Guide | Description |
|-------|-------------|
| [Building Networks](building_networks.md) | How to construct networks for your problem |
| [Custom Resources](custom_resources.md) | Create new resource constraints |
| [Custom Pricing](custom_pricing.md) | Implement your own pricing algorithms |
| [Custom Master](custom_master.md) | Use different LP/MIP solvers |
| [Custom Applications](custom_application.md) | Model new optimization problems |
