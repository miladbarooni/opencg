# OpenCG Architecture

This document describes the architecture and design decisions of OpenCG, a hybrid Python/C++ column generation framework.

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
