# Custom Pricing Algorithms Guide

This guide explains how to implement custom pricing algorithms (SPPRC solvers) for column generation.

## Overview

The **pricing problem** is the core of column generation. It finds paths (columns) with negative reduced cost that can improve the master problem objective. The default implementation uses a labeling algorithm for the Shortest Path Problem with Resource Constraints (SPPRC).

You might want a custom pricing algorithm when:

- Using a problem-specific algorithm (e.g., dynamic programming for knapsack)
- Implementing heuristic pricing for faster iterations
- Using external solvers (e.g., cspy, Boost Graph)
- Experimenting with novel algorithms

---

## Using Built-in Pricing Algorithms

Before creating custom algorithms, understand the built-in options:

```python
from opencg.pricing.labeling import (
    LabelingAlgorithm,           # Standard labeling
    ElementaryLabelingAlgorithm,  # Elementary paths only
    HeuristicLabelingAlgorithm,   # Fast heuristic with limits
)
from opencg.pricing.base import PricingConfig

# Standard labeling
pricing = LabelingAlgorithm(
    problem,
    config=PricingConfig(
        max_columns=100,           # Return at most 100 columns
        reduced_cost_threshold=-1e-6,  # Only negative RC columns
        use_dominance=True,        # Enable dominance pruning
    )
)

# Heuristic labeling (faster but may miss columns)
pricing = HeuristicLabelingAlgorithm(
    problem,
    max_labels_per_node=50,      # Limit labels per node
    early_termination_count=10,  # Stop when 10 columns found
)

# Usage in CG loop
pricing.set_dual_values(duals_from_master)
solution = pricing.solve()

for column in solution.columns:
    print(f"Found column with RC = {column.reduced_cost:.4f}")
```

---

## The PricingProblem Interface

All pricing algorithms inherit from `PricingProblem`:

```python
from opencg.pricing.base import PricingProblem, PricingSolution, PricingConfig

class PricingProblem(ABC):
    def __init__(self, problem: Problem, config: PricingConfig = None):
        self._problem = problem
        self._config = config or PricingConfig()
        self._dual_values = {}

    # Required to implement
    @abstractmethod
    def _solve_impl(self) -> PricingSolution:
        """Your pricing algorithm goes here."""
        pass

    # Public API (don't override)
    def set_dual_values(self, duals: dict[int, float]) -> None:
        """Called by CG loop with new duals from master."""
        pass

    def solve(self) -> PricingSolution:
        """Solve the pricing problem."""
        pass

    # Hooks (optional to override)
    def _on_duals_updated(self) -> None:
        """Called when duals change - precompute arc reduced costs."""
        pass

    def _before_solve(self) -> None:
        """Setup before solving."""
        pass

    def _after_solve(self, solution: PricingSolution) -> PricingSolution:
        """Post-process solution."""
        pass
```

## Built-in Pricing Algorithms

### LabelingAlgorithm

The default mono-directional labeling algorithm.

```python
from opencg.pricing.labeling import LabelingAlgorithm

pricing = LabelingAlgorithm(
    problem,
    config=PricingConfig(
        max_columns=100,
        reduced_cost_threshold=-1e-6,
        check_elementarity=False,
        use_dominance=True
    )
)
```

### FastPerSourcePricing

Per-source labeling for crew pairing (higher coverage).

```python
from opencg.pricing.fast_per_source import FastPerSourcePricing

pricing = FastPerSourcePricing(
    problem,
    config=PricingConfig(max_columns=200),
    num_threads=4  # Parallel execution
)
```

---

## Creating Custom Pricing Algorithms

### Example 0: Simplified Labeling (Understanding the Built-in)

Here's a minimal rewrite of the labeling algorithm to help you understand how it works:

```python
from opencg.pricing.base import PricingProblem, PricingSolution, PricingStatus, PricingConfig
from opencg.core.column import Column
from opencg.core.node import NodeType
import heapq
import time

class SimpleLabelingPricing(PricingProblem):
    """
    A simplified labeling algorithm for educational purposes.

    This is a minimal version of LabelingAlgorithm that you can
    modify and extend for your needs.
    """

    def __init__(self, problem, config=None):
        super().__init__(problem, config)

        # Find source and sink
        self._source_idx = None
        self._sink_idx = None
        for node in problem.network.nodes:
            if node.node_type == NodeType.SOURCE:
                self._source_idx = node.index
            elif node.node_type == NodeType.SINK:
                self._sink_idx = node.index

        # Cache for arc reduced costs (computed when duals update)
        self._arc_rc = {}

    def _on_duals_updated(self):
        """Precompute arc reduced costs when duals change."""
        self._arc_rc.clear()
        for arc in self._problem.network.arcs:
            # rc = cost - sum of duals for covered items
            self._arc_rc[arc.index] = self.get_arc_reduced_cost(arc.index)

    def _solve_impl(self) -> PricingSolution:
        """Simple BFS-style labeling."""
        start_time = time.time()

        # Labels: dict of node_index -> list of (cost, rc, path, resources, covered)
        labels = {i: [] for i in range(self._problem.network.num_nodes)}

        # Initialize source label
        init_resources = {r.name: r.initial_value() for r in self._problem.resources}
        source_label = (0.0, 0.0, [], init_resources.copy(), set())
        labels[self._source_idx].append(source_label)

        # Priority queue: (reduced_cost, node_idx, label)
        pq = [(0.0, self._source_idx, source_label)]
        columns = []

        while pq:
            _, node_idx, label = heapq.heappop(pq)
            cost, rc, path, resources, covered = label

            # Skip if at sink - collect column
            if node_idx == self._sink_idx:
                if rc < self._config.reduced_cost_threshold:
                    col = Column(
                        arc_indices=tuple(path),
                        cost=cost,
                        covered_items=frozenset(covered),
                        reduced_cost=rc
                    )
                    columns.append(col)

                    # Early termination
                    if len(columns) >= (self._config.max_columns or 100):
                        break
                continue

            # Extend along outgoing arcs
            for arc in self._problem.network.outgoing_arcs(node_idx):
                # Check resource feasibility
                new_resources = {}
                feasible = True

                for resource in self._problem.resources:
                    old_val = resources.get(resource.name, resource.initial_value())
                    new_val = resource.extend(old_val, arc)

                    if new_val is None:
                        feasible = False
                        break

                    target = self._problem.network.get_node(arc.target)
                    if not resource.is_feasible(new_val, target):
                        feasible = False
                        break

                    new_resources[resource.name] = new_val

                if not feasible:
                    continue

                # Update covered items
                new_covered = covered | set(self.get_items_covered_by_arc(arc.index))

                # Compute new costs
                new_cost = cost + arc.cost
                new_rc = rc + self._arc_rc.get(arc.index, arc.cost)
                new_path = path + [arc.index]

                new_label = (new_cost, new_rc, new_path, new_resources, new_covered)

                # Simple dominance: only keep if not dominated by existing
                dominated = False
                if self._config.use_dominance:
                    for existing in labels[arc.target]:
                        if self._dominates(existing, new_label):
                            dominated = True
                            break

                if not dominated:
                    labels[arc.target].append(new_label)
                    heapq.heappush(pq, (new_rc, arc.target, new_label))

        # Build solution
        solve_time = time.time() - start_time
        columns.sort(key=lambda c: c.reduced_cost)

        return PricingSolution(
            status=PricingStatus.COLUMNS_FOUND if columns else PricingStatus.NO_COLUMNS,
            columns=columns,
            best_reduced_cost=columns[0].reduced_cost if columns else None,
            solve_time=solve_time
        )

    def _dominates(self, label1, label2):
        """Check if label1 dominates label2."""
        cost1, rc1, _, res1, cov1 = label1
        cost2, rc2, _, res2, cov2 = label2

        # Must have better or equal reduced cost
        if rc1 > rc2:
            return False

        # Must dominate on all resources
        for resource in self._problem.resources:
            v1 = res1.get(resource.name)
            v2 = res2.get(resource.name)
            if v1 is not None and v2 is not None:
                if not resource.dominates(v1, v2):
                    return False

        return True
```

**Usage:**

```python
# Use exactly like the built-in
pricing = SimpleLabelingPricing(problem)
pricing.set_dual_values(duals)
solution = pricing.solve()
```

---

### Example 1: Knapsack Pricing (Cutting Stock)

For cutting stock, pricing is a bounded knapsack problem that can be solved efficiently with dynamic programming.

```python
from opencg.pricing.base import PricingProblem, PricingSolution, PricingStatus
from opencg.core.column import Column
import numpy as np

class KnapsackPricing(PricingProblem):
    """
    Dynamic programming pricing for cutting stock / bin packing.

    Solves: max sum_i (pi_i * x_i)
            s.t. sum_i (w_i * x_i) <= W
                 x_i >= 0, integer
    """

    def __init__(self, problem, item_sizes: list[float], capacity: float, **kwargs):
        super().__init__(problem, **kwargs)
        self.item_sizes = item_sizes
        self.capacity = int(capacity)
        self.num_items = len(item_sizes)

    def _solve_impl(self) -> PricingSolution:
        # Get dual values for each item
        duals = [self._dual_values.get(i, 0.0) for i in range(self.num_items)]
        sizes = [int(s) for s in self.item_sizes]
        W = self.capacity

        # DP table: dp[w] = (max_dual_sum, pattern)
        dp = [(0.0, [0] * self.num_items) for _ in range(W + 1)]

        # Fill DP table
        for i, (size, dual) in enumerate(zip(sizes, duals)):
            if size > W or dual <= 0:
                continue

            # Process in reverse to allow multiple copies
            for w in range(W, size - 1, -1):
                # Try adding item i
                prev_w = w - size
                new_val = dp[prev_w][0] + dual

                if new_val > dp[w][0]:
                    new_pattern = dp[prev_w][1].copy()
                    new_pattern[i] += 1
                    dp[w] = (new_val, new_pattern)

        # Find best pattern
        best_w = max(range(W + 1), key=lambda w: dp[w][0])
        best_dual_sum, pattern = dp[best_w]

        # Reduced cost = cost - dual_sum = 1 - dual_sum (one roll used)
        reduced_cost = 1.0 - best_dual_sum

        if reduced_cost >= -1e-6:
            return PricingSolution(status=PricingStatus.NO_COLUMNS)

        # Create column from pattern
        covered_items = frozenset(i for i, count in enumerate(pattern) if count > 0)

        column = Column(
            arc_indices=tuple(pattern),  # Store pattern as "arc indices"
            cost=1.0,
            covered_items=covered_items,
            reduced_cost=reduced_cost,
            attributes={"pattern": pattern}
        )

        return PricingSolution(
            status=PricingStatus.COLUMNS_FOUND,
            columns=[column],
            best_reduced_cost=reduced_cost
        )
```

**Usage:**

```python
pricing = KnapsackPricing(
    problem,
    item_sizes=[45, 36, 31, 14],
    capacity=100
)

# In column generation loop
pricing.set_dual_values(duals)
solution = pricing.solve()

if solution.columns:
    for column in solution.columns:
        master.add_column(column)
```

---

### Example 2: Heuristic Nearest Neighbor Pricing

A fast heuristic for vehicle routing that doesn't guarantee optimal columns.

```python
from opencg.pricing.base import PricingProblem, PricingSolution, PricingStatus
from opencg.core.column import Column
import heapq

class NearestNeighborPricing(PricingProblem):
    """
    Fast heuristic pricing using nearest neighbor construction.

    Builds routes greedily by always visiting the closest unvisited
    customer with negative reduced cost contribution.
    """

    def __init__(self, problem, num_routes: int = 10, **kwargs):
        super().__init__(problem, **kwargs)
        self.num_routes = num_routes  # Number of routes to generate
        self._arc_reduced_costs = {}

    def _on_duals_updated(self) -> None:
        """Precompute arc reduced costs when duals change."""
        self._arc_reduced_costs.clear()
        network = self._problem.network

        for arc in network.arcs:
            rc = arc.cost
            # Subtract duals for covered items
            for item_id in self.get_items_covered_by_arc(arc.index):
                rc -= self._dual_values.get(item_id, 0.0)
            self._arc_reduced_costs[arc.index] = rc

    def _solve_impl(self) -> PricingSolution:
        columns = []
        network = self._problem.network

        # Try multiple starting points
        depot = network.source
        customers = [n for n in network.nodes if n.node_type.name == 'CUSTOMER']

        for start_idx in range(min(self.num_routes, len(customers))):
            route = self._build_route(depot, customers, start_idx)

            if route and route.reduced_cost < -1e-6:
                columns.append(route)

        if not columns:
            return PricingSolution(status=PricingStatus.NO_COLUMNS)

        # Sort by reduced cost and return best ones
        columns.sort(key=lambda c: c.reduced_cost)
        max_cols = self._config.max_columns or len(columns)
        columns = columns[:max_cols]

        return PricingSolution(
            status=PricingStatus.COLUMNS_FOUND,
            columns=columns,
            best_reduced_cost=columns[0].reduced_cost
        )

    def _build_route(self, depot, customers, start_idx: int) -> Column | None:
        """Build a route using nearest neighbor heuristic."""
        network = self._problem.network
        capacity = self._problem.get_attribute("vehicle_capacity", float('inf'))

        visited = set()
        route_arcs = []
        current_node = depot
        total_cost = 0.0
        total_load = 0.0
        covered = set()

        # Start from a specific customer
        if start_idx < len(customers):
            first_customer = customers[start_idx]
            arc = network.get_arc_between(depot, first_customer)
            if arc:
                route_arcs.append(arc.index)
                total_cost += arc.cost
                visited.add(first_customer.index)
                current_node = first_customer

        # Greedily add nearest customers
        while True:
            best_arc = None
            best_rc = 0

            for arc in network.outgoing_arcs(current_node):
                target = arc.target_node

                # Skip visited or depot
                if target.index in visited or target == network.sink:
                    continue

                # Check capacity
                demand = target.get_attribute("demand", 0)
                if total_load + demand > capacity:
                    continue

                # Check reduced cost
                rc = self._arc_reduced_costs.get(arc.index, arc.cost)
                if rc < best_rc:
                    best_rc = rc
                    best_arc = arc

            if best_arc is None:
                break  # No more customers to add

            # Add to route
            route_arcs.append(best_arc.index)
            total_cost += best_arc.cost
            visited.add(best_arc.target_node.index)
            total_load += best_arc.target_node.get_attribute("demand", 0)
            covered.update(self.get_items_covered_by_arc(best_arc.index))
            current_node = best_arc.target_node

        # Return to depot
        return_arc = network.get_arc_between(current_node, network.sink)
        if return_arc:
            route_arcs.append(return_arc.index)
            total_cost += return_arc.cost

        if not route_arcs:
            return None

        # Compute reduced cost
        dual_sum = sum(self._dual_values.get(i, 0) for i in covered)
        reduced_cost = total_cost - dual_sum

        return Column(
            arc_indices=tuple(route_arcs),
            cost=total_cost,
            covered_items=frozenset(covered),
            reduced_cost=reduced_cost
        )
```

---

### Example 3: Hybrid Exact + Heuristic Pricing

Combine fast heuristics with occasional exact solving.

```python
from opencg.pricing.base import PricingProblem, PricingSolution, PricingStatus
from opencg.pricing.labeling import LabelingAlgorithm

class HybridPricing(PricingProblem):
    """
    Hybrid pricing that uses heuristics most of the time,
    falling back to exact labeling periodically.
    """

    def __init__(self, problem, heuristic_ratio: float = 0.8, **kwargs):
        super().__init__(problem, **kwargs)
        self.heuristic_ratio = heuristic_ratio
        self.iteration = 0

        # Create sub-solvers
        self.heuristic = NearestNeighborPricing(problem, **kwargs)
        self.exact = LabelingAlgorithm(problem, **kwargs)

    def _on_duals_updated(self) -> None:
        # Forward to both sub-solvers
        self.heuristic.set_dual_values(self._dual_values)
        self.exact.set_dual_values(self._dual_values)

    def _solve_impl(self) -> PricingSolution:
        self.iteration += 1

        # Use heuristic most of the time
        import random
        if random.random() < self.heuristic_ratio:
            solution = self.heuristic.solve()

            # If heuristic found columns, use them
            if solution.columns:
                return solution

        # Fall back to exact (or use periodically)
        return self.exact.solve()
```

---

### Example 4: External Solver Integration (cspy)

Use the cspy library for resource-constrained shortest paths.

```python
from opencg.pricing.base import PricingProblem, PricingSolution, PricingStatus
from opencg.core.column import Column

class CspyPricing(PricingProblem):
    """
    Pricing using the cspy library for SPPRC.

    cspy provides efficient implementations of various labeling algorithms.
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        self._graph = None
        self._build_cspy_graph()

    def _build_cspy_graph(self):
        """Convert OpenCG network to cspy DiGraph."""
        try:
            import networkx as nx
            from cspy import REFCallback
        except ImportError:
            raise ImportError("cspy and networkx required: pip install cspy networkx")

        self._graph = nx.DiGraph()
        network = self._problem.network

        # Add nodes
        for node in network.nodes:
            self._graph.add_node(node.index)

        # Add edges with resource consumption
        for arc in network.arcs:
            # Get resource consumptions
            res = [arc.get_consumption(r.name, 0.0)
                   for r in self._problem.resources]

            self._graph.add_edge(
                arc.source_node.index,
                arc.target_node.index,
                weight=arc.cost,
                res_cost=res,
                arc_index=arc.index
            )

    def _solve_impl(self) -> PricingSolution:
        from cspy import BiDirectional

        # Update edge weights with reduced costs
        for u, v, data in self._graph.edges(data=True):
            arc_idx = data["arc_index"]
            rc = self.get_arc_reduced_cost(arc_idx)
            data["weight"] = rc

        # Get resource bounds
        max_res = [r.max_value if hasattr(r, 'max_value') else float('inf')
                   for r in self._problem.resources]
        min_res = [0.0] * len(self._problem.resources)

        # Solve with cspy
        source = self._problem.network.source.index
        sink = self._problem.network.sink.index

        alg = BiDirectional(
            self._graph,
            max_res=max_res,
            min_res=min_res,
            direction="both"
        )
        alg.run()

        if alg.path is None:
            return PricingSolution(status=PricingStatus.NO_COLUMNS)

        # Extract column from path
        arc_indices = []
        covered = set()
        total_cost = 0.0

        path = alg.path
        for i in range(len(path) - 1):
            edge_data = self._graph[path[i]][path[i+1]]
            arc_idx = edge_data["arc_index"]
            arc_indices.append(arc_idx)
            covered.update(self.get_items_covered_by_arc(arc_idx))
            total_cost += self._problem.network.get_arc(arc_idx).cost

        dual_sum = sum(self._dual_values.get(i, 0) for i in covered)
        reduced_cost = total_cost - dual_sum

        if reduced_cost >= -1e-6:
            return PricingSolution(status=PricingStatus.NO_COLUMNS)

        column = Column(
            arc_indices=tuple(arc_indices),
            cost=total_cost,
            covered_items=frozenset(covered),
            reduced_cost=reduced_cost
        )

        return PricingSolution(
            status=PricingStatus.COLUMNS_FOUND,
            columns=[column],
            best_reduced_cost=reduced_cost
        )
```

---

## Using Custom Pricing in Column Generation

```python
from opencg.solver import ColumnGeneration, CGConfig

# Create your custom pricing
pricing = KnapsackPricing(problem, item_sizes, capacity)

# Use in column generation
cg = ColumnGeneration(
    problem,
    pricing=pricing,  # Pass custom pricing
    config=CGConfig(max_iterations=50)
)

solution = cg.solve()
```

---

## Best Practices

### 1. Precompute Arc Reduced Costs

Use `_on_duals_updated()` to precompute reduced costs when duals change:

```python
def _on_duals_updated(self) -> None:
    for arc in self.network.arcs:
        self._arc_rc[arc.index] = self.get_arc_reduced_cost(arc.index)
```

### 2. Return Multiple Columns

Finding multiple negative reduced cost columns per iteration improves convergence:

```python
def _solve_impl(self):
    columns = []
    for path in self._find_paths():
        if path.reduced_cost < -1e-6:
            columns.append(path)
            if len(columns) >= self._config.max_columns:
                break
    return PricingSolution(columns=columns, ...)
```

### 3. Use PricingConfig

Respect configuration settings:

```python
if self._config.max_columns > 0 and len(columns) >= self._config.max_columns:
    break

if self._config.max_time > 0 and elapsed > self._config.max_time:
    return PricingSolution(status=PricingStatus.TIME_LIMIT, ...)
```

### 4. Handle Edge Cases

```python
def _solve_impl(self):
    if not self._dual_values:
        return PricingSolution(status=PricingStatus.NO_COLUMNS)

    # ... solve ...

    if not columns:
        return PricingSolution(status=PricingStatus.NO_COLUMNS)
```

---

## Performance Considerations

| Approach | Speed | Optimality | Use Case |
|----------|-------|------------|----------|
| Exact labeling | Slow | Optimal | Small networks, final iterations |
| Heuristic | Fast | Suboptimal | Large networks, early iterations |
| Hybrid | Medium | Good | General purpose |
| Parallel | Fast | Optimal | Multi-core systems |

---

## Next Steps

- [Custom Resources](custom_resources.md) - Define new resource constraints
- [Custom Master Problem](custom_master.md) - Use different LP/MIP solvers
- [Custom Applications](custom_application.md) - Model new problem types
