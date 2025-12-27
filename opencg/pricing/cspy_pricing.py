"""
CSPY-based pricing algorithm for SPPRC.

Uses the cspy library's bidirectional labeling algorithm for solving
the pricing subproblem in column generation. This provides a significant
speedup over the pure Python implementation.

References:
----------
- cspy: https://github.com/torressa/cspy
- Tilk et al. (2017): "Asymmetry helps: Dynamic half-way points for
  bidirectional labeling algorithms"
"""

from typing import Dict, List, Optional, Set
import time
import networkx as nx
import numpy as np

from opencg.core.arc import ArcType
from opencg.core.column import Column
from opencg.core.problem import Problem
from opencg.core.node import NodeType
from opencg.pricing.base import (
    PricingProblem,
    PricingConfig,
    PricingSolution,
    PricingStatus,
)

try:
    from cspy import BiDirectional
    CSPY_AVAILABLE = True
except ImportError:
    CSPY_AVAILABLE = False


class CSPYBasePricingAlgorithm(PricingProblem):
    """
    Pricing algorithm using cspy's bidirectional labeling.

    This converts the problem network to a NetworkX graph and uses
    cspy's BiDirectional algorithm to find negative reduced cost paths.

    Advantages over pure Python:
    - Bidirectional search reduces label count
    - C++ backend with LEMON graph library
    - Dynamic halfway point for balanced search

    Limitations:
    - Rebuilds NetworkX graph each pricing call (overhead)
    - Limited support for complex resource types
    - Returns single optimal path per call

    Example:
        >>> pricing = CSPYPricingAlgorithm(problem)
        >>> pricing.set_dual_values(duals)
        >>> result = pricing.solve()
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None
    ):
        if not CSPY_AVAILABLE:
            raise ImportError("cspy is not installed. Run: pip install cspy")

        super().__init__(problem, config)

        # Cache network structure
        self._source_idx: Optional[int] = None
        self._sink_idx: Optional[int] = None
        self._find_source_sink()

        # Resource indices mapping
        self._resource_names = [r.name for r in problem.resources]
        self._num_resources = len(self._resource_names)

        # Build arc index mappings
        self._arc_to_edge: Dict[int, tuple] = {}  # arc_index -> (src_name, tgt_name)
        self._edge_to_arc: Dict[tuple, int] = {}  # (src_name, tgt_name) -> arc_index

    def _find_source_sink(self) -> None:
        """Find source and sink node indices."""
        for i in range(self._problem.network.num_nodes):
            node = self._problem.network.get_node(i)
            if node is None:
                continue
            if node.node_type == NodeType.SOURCE:
                self._source_idx = i
            elif node.node_type == NodeType.SINK:
                self._sink_idx = i

    def _build_networkx_graph(self) -> nx.DiGraph:
        """
        Convert problem network to NetworkX DiGraph for cspy.

        Resources are mapped to a fixed array:
        - Index 0: "critical" resource (monotonic, used for halfway point)
        - Index 1+: duty_time, flight_time, duty_days

        Edge weights are reduced costs (cost - dual).
        """
        # Use a monotonic "path_length" as critical resource
        # Plus our actual resources
        n_res = 1 + self._num_resources  # path_length + actual resources

        G = nx.DiGraph(directed=True, n_res=n_res)

        network = self._problem.network

        # Add edges
        for arc in network.arcs:
            src_name = f"n{arc.source}"
            tgt_name = f"n{arc.target}"

            # Special names for source/sink
            if arc.source == self._source_idx:
                src_name = "Source"
            if arc.target == self._sink_idx:
                tgt_name = "Sink"

            # Build resource cost array
            res_cost = np.zeros(n_res)
            res_cost[0] = 1.0  # Path length (monotonic)

            for i, res_name in enumerate(self._resource_names):
                consumption = arc.get_consumption(res_name, 0.0)
                # Handle HomeBaseResource specially - it's not numeric
                if res_name == 'home_base':
                    # Skip for now - we'll handle base constraint differently
                    res_cost[1 + i] = 0.0
                else:
                    res_cost[1 + i] = max(0.0, consumption)  # cspy needs non-negative

            # Compute reduced cost as edge weight
            items_covered = self.get_items_covered_by_arc(arc.index)
            dual_sum = sum(self._dual_values.get(item, 0.0) for item in items_covered)
            reduced_cost = arc.cost - dual_sum

            G.add_edge(
                src_name,
                tgt_name,
                res_cost=res_cost,
                weight=reduced_cost,
                arc_index=arc.index
            )

            self._arc_to_edge[arc.index] = (src_name, tgt_name)
            self._edge_to_arc[(src_name, tgt_name)] = arc.index

        return G

    def _get_resource_bounds(self) -> tuple:
        """Get min/max resource bounds for cspy."""
        n_res = 1 + self._num_resources

        # Max resources
        max_res = [float('inf')] * n_res
        max_res[0] = 100.0  # Max path length

        for i, resource in enumerate(self._problem.resources):
            if hasattr(resource, 'max_value'):
                max_res[1 + i] = resource.max_value
            else:
                max_res[1 + i] = float('inf')

        # Min resources (for backward search)
        min_res = [0.0] * n_res

        return min_res, max_res

    def _solve_impl(self) -> PricingSolution:
        """
        Run cspy bidirectional labeling.
        """
        start_time = time.time()

        # Build graph
        G = self._build_networkx_graph()
        min_res, max_res = self._get_resource_bounds()

        columns = []
        best_rc = None
        num_paths_found = 0

        # Run cspy - it finds one optimal path
        try:
            bidirec = BiDirectional(
                G,
                max_res=max_res,
                min_res=min_res,
                direction="both",
                elementary=self._config.check_elementarity,
                time_limit=self._config.max_time if self._config.max_time > 0 else None,
            )
            bidirec.run()

            if bidirec.path and len(bidirec.path) > 2:  # More than just Source-Sink
                path = bidirec.path
                total_res = bidirec.total_res
                path_cost = bidirec.path_cost if hasattr(bidirec, 'path_cost') else 0

                # Convert path to column
                arc_indices = []
                covered_items = set()
                cost = 0.0

                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in self._edge_to_arc:
                        arc_idx = self._edge_to_arc[edge]
                        arc_indices.append(arc_idx)

                        arc = self._problem.network.get_arc(arc_idx)
                        cost += arc.cost
                        covered_items.update(self.get_items_covered_by_arc(arc_idx))

                # Compute reduced cost
                dual_sum = sum(self._dual_values.get(item, 0.0) for item in covered_items)
                reduced_cost = cost - dual_sum

                if reduced_cost < self._config.reduced_cost_threshold:
                    column = Column(
                        arc_indices=tuple(arc_indices),
                        cost=cost,
                        reduced_cost=reduced_cost,
                        covered_items=frozenset(covered_items),
                    )
                    columns.append(column)
                    best_rc = reduced_cost
                    num_paths_found = 1

        except Exception as e:
            # cspy can raise exceptions for infeasible problems
            pass

        solve_time = time.time() - start_time

        if columns:
            status = PricingStatus.COLUMNS_FOUND
        else:
            status = PricingStatus.NO_COLUMNS

        return PricingSolution(
            status=status,
            columns=columns,
            best_reduced_cost=best_rc,
            num_labels_created=0,  # cspy doesn't expose this
            num_labels_dominated=0,
            solve_time=solve_time,
            iterations=num_paths_found,
        )


class CSPYMultiPathPricing(CSPYBasePricingAlgorithm):
    """
    cspy-based pricing that finds multiple paths by iteratively
    modifying edge weights.

    After finding each path, increases the weight of edges in that
    path to encourage finding different paths.
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        num_iterations: int = 10,
        weight_increment: float = 1000.0,
    ):
        super().__init__(problem, config)
        self._num_iterations = num_iterations
        self._weight_increment = weight_increment

    def _solve_impl(self) -> PricingSolution:
        """Find multiple paths by iterative weight modification."""
        start_time = time.time()

        G = self._build_networkx_graph()
        min_res, max_res = self._get_resource_bounds()

        columns = []
        all_covered = set()
        best_rc = None

        for iteration in range(self._num_iterations):
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            try:
                remaining_time = None
                if self._config.max_time > 0:
                    remaining_time = self._config.max_time - (time.time() - start_time)
                    if remaining_time <= 0:
                        break

                bidirec = BiDirectional(
                    G,
                    max_res=max_res,
                    min_res=min_res,
                    direction="both",
                    elementary=self._config.check_elementarity,
                    time_limit=remaining_time,
                )
                bidirec.run()

                if not bidirec.path or len(bidirec.path) <= 2:
                    break

                path = bidirec.path

                # Convert path to column
                arc_indices = []
                covered_items = set()
                cost = 0.0

                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    if edge in self._edge_to_arc:
                        arc_idx = self._edge_to_arc[edge]
                        arc_indices.append(arc_idx)

                        arc = self._problem.network.get_arc(arc_idx)
                        cost += arc.cost
                        covered_items.update(self.get_items_covered_by_arc(arc_idx))

                        # Increase weight to discourage reuse
                        G[path[i]][path[i + 1]]['weight'] += self._weight_increment

                dual_sum = sum(self._dual_values.get(item, 0.0) for item in covered_items)
                reduced_cost = cost - dual_sum

                if reduced_cost < self._config.reduced_cost_threshold:
                    column = Column(
                        arc_indices=tuple(arc_indices),
                        cost=cost,
                        reduced_cost=reduced_cost,
                        covered_items=frozenset(covered_items),
                    )
                    columns.append(column)
                    all_covered.update(covered_items)

                    if best_rc is None or reduced_cost < best_rc:
                        best_rc = reduced_cost

                if self._config.max_columns > 0 and len(columns) >= self._config.max_columns:
                    break

            except Exception:
                break

        solve_time = time.time() - start_time

        if columns:
            status = PricingStatus.COLUMNS_FOUND
        else:
            status = PricingStatus.NO_COLUMNS

        return PricingSolution(
            status=status,
            columns=columns,
            best_reduced_cost=best_rc,
            num_labels_created=0,
            num_labels_dominated=0,
            solve_time=solve_time,
            iterations=len(columns),
        )


class CSPYMultiBasePricing(PricingProblem):
    """
    Multi-base cspy pricing that respects HomeBaseResource constraint.

    Runs separate cspy pricing for each base, only including arcs that
    belong to that base's valid paths (start and end at same base).
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        paths_per_base: int = 50,
    ):
        if not CSPY_AVAILABLE:
            raise ImportError("cspy is not installed. Run: pip install cspy")

        super().__init__(problem, config)
        self._paths_per_base = paths_per_base

        # Find source/sink
        self._source_idx: Optional[int] = None
        self._sink_idx: Optional[int] = None
        self._find_source_sink()

        # Find bases and their arcs
        self._bases = self._find_bases()
        self._base_source_arcs = self._find_base_source_arcs()
        self._base_sink_arcs = self._find_base_sink_arcs()

    def _find_source_sink(self) -> None:
        for i in range(self._problem.network.num_nodes):
            node = self._problem.network.get_node(i)
            if node is None:
                continue
            if node.node_type == NodeType.SOURCE:
                self._source_idx = i
            elif node.node_type == NodeType.SINK:
                self._sink_idx = i

    def _find_bases(self) -> List[str]:
        bases = set()
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base:
                    bases.add(base)
        return sorted(bases)

    def _find_base_source_arcs(self) -> Dict[str, Set[int]]:
        result = {base: set() for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base in result:
                    result[base].add(arc.index)
        return result

    def _find_base_sink_arcs(self) -> Dict[str, Set[int]]:
        result = {base: set() for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base')
                if base in result:
                    result[base].add(arc.index)
        return result

    def _build_base_graph(self, base: str) -> tuple:
        """Build NetworkX graph for a specific base."""
        # Resources: path_length, duty_time, flight_time, duty_days
        n_res = 4

        G = nx.DiGraph(directed=True, n_res=n_res)
        network = self._problem.network

        arc_to_edge = {}
        edge_to_arc = {}

        for arc in network.arcs:
            # Only include arcs valid for this base
            if arc.arc_type == ArcType.SOURCE_ARC:
                if arc.index not in self._base_source_arcs[base]:
                    continue
            elif arc.arc_type == ArcType.SINK_ARC:
                if arc.index not in self._base_sink_arcs[base]:
                    continue

            src_name = f"n{arc.source}"
            tgt_name = f"n{arc.target}"

            if arc.source == self._source_idx:
                src_name = "Source"
            if arc.target == self._sink_idx:
                tgt_name = "Sink"

            # Resource costs
            res_cost = np.zeros(n_res)
            res_cost[0] = 1.0  # Path length (monotonic critical resource)

            duty = arc.get_consumption('duty_time', 0.0)
            flight = arc.get_consumption('flight_time', 0.0)
            days = arc.get_consumption('duty_days', 0.0)

            # cspy needs non-negative resources
            res_cost[1] = max(0.0, duty)
            res_cost[2] = max(0.0, flight)
            res_cost[3] = max(0.0, days)

            # Reduced cost
            items_covered = self.get_items_covered_by_arc(arc.index)
            dual_sum = sum(self._dual_values.get(item, 0.0) for item in items_covered)
            reduced_cost = arc.cost - dual_sum

            G.add_edge(src_name, tgt_name, res_cost=res_cost, weight=reduced_cost)
            arc_to_edge[arc.index] = (src_name, tgt_name)
            edge_to_arc[(src_name, tgt_name)] = arc.index

        return G, arc_to_edge, edge_to_arc

    def _solve_impl(self) -> PricingSolution:
        """Run cspy for each base and combine results."""
        start_time = time.time()

        all_columns = []
        best_rc = None

        # Resource bounds
        max_res = [100.0, 14.0, 8.0, 5.0]  # path_len, duty, flight, days
        min_res = [0.0, 0.0, 0.0, 0.0]

        num_bases = len(self._bases)
        time_per_base = (
            self._config.max_time / num_bases
            if self._config.max_time > 0 and num_bases > 0 else 60.0
        )

        for base in self._bases:
            if self._config.max_time > 0:
                elapsed = time.time() - start_time
                if elapsed >= self._config.max_time:
                    break

            G, arc_to_edge, edge_to_arc = self._build_base_graph(base)

            if G.number_of_edges() == 0:
                continue

            # Check if path exists
            if "Source" not in G or "Sink" not in G:
                continue
            if not nx.has_path(G, "Source", "Sink"):
                continue

            # Find multiple paths for this base
            for path_iter in range(self._paths_per_base):
                if self._config.max_time > 0:
                    remaining = self._config.max_time - (time.time() - start_time)
                    if remaining <= 0:
                        break

                try:
                    bidirec = BiDirectional(
                        G,
                        max_res=max_res,
                        min_res=min_res,
                        direction="both",
                        elementary=self._config.check_elementarity,
                        time_limit=min(time_per_base, remaining) if self._config.max_time > 0 else time_per_base,
                    )
                    bidirec.run()

                    if not bidirec.path or len(bidirec.path) <= 2:
                        break

                    path = bidirec.path

                    # Convert to column
                    arc_indices = []
                    covered_items = set()
                    cost = 0.0

                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        if edge in edge_to_arc:
                            arc_idx = edge_to_arc[edge]
                            arc_indices.append(arc_idx)

                            arc = self._problem.network.get_arc(arc_idx)
                            cost += arc.cost
                            covered_items.update(self.get_items_covered_by_arc(arc_idx))

                            # Increase weight to find different path next time
                            G[path[i]][path[i + 1]]['weight'] += 10000.0

                    dual_sum = sum(self._dual_values.get(item, 0.0) for item in covered_items)
                    reduced_cost = cost - dual_sum

                    if reduced_cost < self._config.reduced_cost_threshold:
                        column = Column(
                            arc_indices=tuple(arc_indices),
                            cost=cost,
                            reduced_cost=reduced_cost,
                            covered_items=frozenset(covered_items),
                            attributes={'base': base},
                        )
                        all_columns.append(column)

                        if best_rc is None or reduced_cost < best_rc:
                            best_rc = reduced_cost

                except Exception:
                    break

                if self._config.max_columns > 0 and len(all_columns) >= self._config.max_columns:
                    break

            if self._config.max_columns > 0 and len(all_columns) >= self._config.max_columns:
                break

        # Sort by reduced cost
        all_columns.sort(key=lambda c: c.reduced_cost)
        if self._config.max_columns > 0:
            all_columns = all_columns[:self._config.max_columns]

        solve_time = time.time() - start_time

        if all_columns:
            status = PricingStatus.COLUMNS_FOUND
        else:
            status = PricingStatus.NO_COLUMNS

        return PricingSolution(
            status=status,
            columns=all_columns,
            best_reduced_cost=best_rc,
            num_labels_created=0,
            num_labels_dominated=0,
            solve_time=solve_time,
            iterations=len(all_columns),
        )


# Aliases for backwards compatibility
CSPYPricingAlgorithm = CSPYBasePricingAlgorithm
