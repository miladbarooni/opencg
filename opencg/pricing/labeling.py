"""
Labeling algorithm for SPPRC (Shortest Path Problem with Resource Constraints).

This module implements the standard mono-directional labeling algorithm
for solving the pricing subproblem in column generation.

Algorithm Overview:
------------------
1. Initialize source label with initial resource values
2. While there are labels to extend:
   a. Select a label (typically from a priority queue)
   b. For each outgoing arc from the label's node:
      - Extend the label along the arc
      - Check resource feasibility
      - Check dominance against existing labels
      - Add if not dominated
3. Collect labels at sink with negative reduced cost
4. Convert to columns

Key Features:
------------
- Dominance-based pruning for efficiency
- Support for multiple resource types
- Configurable column limits and time limits
- Hook points for algorithm customization

References:
----------
- Irnich, S., & Desaulniers, G. (2005). Shortest path problems with resource
  constraints. In Column generation (pp. 33-65). Springer.
"""

import heapq
import time
from typing import Any, Optional

from opencg.core.column import Column
from opencg.core.node import NodeType
from opencg.core.problem import Problem
from opencg.pricing.base import (
    PricingConfig,
    PricingProblem,
    PricingSolution,
    PricingStatus,
)
from opencg.pricing.label import Label, LabelPool


class LabelingAlgorithm(PricingProblem):
    """
    Standard labeling algorithm for SPPRC.

    This is the default implementation of the pricing problem using a
    mono-directional labeling algorithm with dominance pruning.

    The algorithm maintains labels (partial paths) at each node and
    extends them along outgoing arcs. Dominated labels are pruned
    to keep the search space manageable.

    Algorithm:
    ---------
    1. Create source label at source node
    2. Use priority queue (by reduced cost) for label extension
    3. For each label, extend along all outgoing arcs
    4. Check feasibility and dominance for new labels
    5. Collect labels at sink with negative reduced cost
    6. Return as columns

    Example:
        >>> from opencg.pricing import LabelingAlgorithm
        >>> pricing = LabelingAlgorithm(problem)
        >>> pricing.set_dual_values(duals)
        >>> solution = pricing.solve()
        >>> for col in solution.columns:
        ...     print(f"Column with RC={col.reduced_cost:.4f}")
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None
    ):
        """
        Initialize the labeling algorithm.

        Args:
            problem: The Problem instance
            config: Optional configuration
        """
        super().__init__(problem, config)

        # Label management
        self._label_pool: Optional[LabelPool] = None
        self._next_label_id: int = 0

        # Precomputed arc reduced costs (updated when duals change)
        self._arc_reduced_costs: dict[int, float] = {}

        # Source and sink node indices
        self._source_idx: Optional[int] = None
        self._sink_idx: Optional[int] = None
        self._find_source_sink()

    def _find_source_sink(self) -> None:
        """Find source and sink node indices."""
        network = self._problem.network

        for i in range(network.num_nodes):
            node = network.get_node(i)
            if node is None:
                continue

            if node.node_type == NodeType.SOURCE:
                self._source_idx = i
            elif node.node_type == NodeType.SINK:
                self._sink_idx = i

        if self._source_idx is None:
            raise ValueError("Network has no source node")
        if self._sink_idx is None:
            raise ValueError("Network has no sink node")

    # =========================================================================
    # Hook Overrides
    # =========================================================================

    def _on_duals_updated(self) -> None:
        """Precompute arc reduced costs when duals change."""
        self._arc_reduced_costs.clear()

        for arc in self._problem.network.arcs:
            self._arc_reduced_costs[arc.index] = self.get_arc_reduced_cost(arc.index)

    # =========================================================================
    # Main Algorithm
    # =========================================================================

    def _solve_impl(self) -> PricingSolution:
        """
        Run the labeling algorithm.

        Returns:
            PricingSolution with found columns
        """
        start_time = time.time()

        # Initialize
        self._label_pool = LabelPool(self._problem.network.num_nodes)
        self._next_label_id = 0

        # Create source label
        source_label = self._create_source_label()
        self._label_pool.add_label(
            source_label,
            self._problem.resources,
            check_dominance=False
        )

        # Priority queue: (reduced_cost, label_id, label)
        # label_id is used for tie-breaking to ensure consistent ordering
        pq: list[tuple[float, int, Label]] = []
        heapq.heappush(pq, (source_label.reduced_cost, source_label.label_id, source_label))

        # Track processed labels to avoid duplicates
        processed_count = 0

        # Main loop
        while pq:
            # Check limits
            if self._config.max_labels > 0:
                if self._label_pool.total_created >= self._config.max_labels:
                    break

            if self._config.max_time > 0:
                if time.time() - start_time >= self._config.max_time:
                    break

            # Get next label
            _, _, label = heapq.heappop(pq)
            processed_count += 1

            # Skip if at sink (we'll collect these at the end)
            if label.node_index == self._sink_idx:
                continue

            # Extend along all outgoing arcs
            for arc in self._problem.network.outgoing_arcs(label.node_index):
                new_label = self._extend_label(label, arc)

                if new_label is None:
                    continue  # Extension infeasible

                # Check dominance and add to pool
                added = self._label_pool.add_label(
                    new_label,
                    self._problem.resources,
                    check_dominance=self._config.use_dominance
                )

                if added:
                    heapq.heappush(pq, (new_label.reduced_cost, new_label.label_id, new_label))

        # Collect columns from sink labels
        sink_labels = self._label_pool.get_labels(self._sink_idx)
        columns = self._collect_columns(sink_labels)

        # Build solution
        solve_time = time.time() - start_time
        stats = self._label_pool.statistics()

        if columns:
            best_rc = min(c.reduced_cost for c in columns)
            status = (
                PricingStatus.COLUMNS_FOUND
                if best_rc < self._config.reduced_cost_threshold
                else PricingStatus.NO_COLUMNS
            )
        else:
            best_rc = None
            status = PricingStatus.NO_COLUMNS

        # Check if we hit limits
        if self._config.max_labels > 0 and stats['total_created'] >= self._config.max_labels:
            status = PricingStatus.ITERATION_LIMIT
        elif self._config.max_time > 0 and solve_time >= self._config.max_time:
            status = PricingStatus.TIME_LIMIT

        return PricingSolution(
            status=status,
            columns=columns,
            best_reduced_cost=best_rc,
            num_labels_created=stats['total_created'],
            num_labels_dominated=stats['total_dominated'],
            solve_time=solve_time,
            iterations=processed_count,
        )

    def _create_source_label(self) -> Label:
        """Create the initial label at the source node."""
        # Initialize resource values
        resource_values = {}
        for resource in self._problem.resources:
            resource_values[resource.name] = resource.initial_value()

        label = Label(
            node_index=self._source_idx,
            cost=0.0,
            reduced_cost=0.0,
            resource_values=resource_values,
            covered_items=frozenset(),
            predecessor=None,
            last_arc_index=None,
            label_id=self._get_next_label_id(),
        )

        return label

    def _extend_label(self, label: Label, arc) -> Optional[Label]:
        """
        Extend a label along an arc.

        Args:
            label: The label to extend
            arc: The arc to traverse

        Returns:
            New label at target node, or None if infeasible
        """
        target_node = self._problem.network.get_node(arc.target)
        if target_node is None:
            return None

        # Extend resources
        new_resource_values = {}
        for resource in self._problem.resources:
            current_val = label.get_resource(resource.name)
            if current_val is None:
                current_val = resource.initial_value()

            new_val = resource.extend(current_val, arc)
            if new_val is None:
                return None  # Resource constraint violated

            # Check node feasibility
            if not resource.is_feasible(new_val, target_node):
                return None

            new_resource_values[resource.name] = new_val

        # Update covered items
        new_covered = set(label.covered_items)
        items_covered_by_arc = self.get_items_covered_by_arc(arc.index)
        for item_id in items_covered_by_arc:
            # Check elementarity if required
            if self._config.check_elementarity:
                if item_id in new_covered:
                    return None  # Already covered this item
            new_covered.add(item_id)

        # Compute costs
        new_cost = label.cost + arc.cost

        # Reduced cost = cost - sum of duals for newly covered items
        arc_rc = self._arc_reduced_costs.get(arc.index, arc.cost)
        new_reduced_cost = label.reduced_cost + arc_rc

        # Create new label
        new_label = Label(
            node_index=arc.target,
            cost=new_cost,
            reduced_cost=new_reduced_cost,
            resource_values=new_resource_values,
            covered_items=frozenset(new_covered),
            predecessor=label,
            last_arc_index=arc.index,
            label_id=self._get_next_label_id(),
        )

        return new_label

    def _collect_columns(self, sink_labels: list[Label]) -> list[Column]:
        """
        Collect columns from sink labels.

        Filters to those with negative reduced cost and applies limits.

        Args:
            sink_labels: Labels at the sink node

        Returns:
            List of columns with negative reduced cost
        """
        threshold = self._config.reduced_cost_threshold

        # Filter to negative reduced cost
        candidates = [
            label for label in sink_labels
            if label.reduced_cost < threshold
        ]

        # Sort by reduced cost (most negative first)
        candidates.sort(key=lambda lbl: lbl.reduced_cost)

        # Apply limit
        if self._config.max_columns > 0:
            candidates = candidates[:self._config.max_columns]

        # Convert to columns
        columns = []
        for label in candidates:
            column = self._create_column_from_label(label)
            columns.append(column)

        return columns

    def _get_next_label_id(self) -> int:
        """Get next unique label ID."""
        label_id = self._next_label_id
        self._next_label_id += 1
        return label_id

    # =========================================================================
    # Advanced Features
    # =========================================================================

    def get_all_sink_labels(self) -> list[Label]:
        """
        Get all labels at the sink (for debugging/analysis).

        Call this after solve() to inspect all paths found.

        Returns:
            List of labels at sink node
        """
        if self._label_pool is None:
            return []
        return self._label_pool.get_labels(self._sink_idx)

    def get_label_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the labeling process.

        Call this after solve() to get detailed statistics.

        Returns:
            Dictionary with statistics
        """
        if self._label_pool is None:
            return {}
        return self._label_pool.statistics()


class ElementaryLabelingAlgorithm(LabelingAlgorithm):
    """
    Labeling algorithm with elementary path constraints.

    This enforces that each node (or item) can only be visited once.
    Uses the StateResource for tracking visited items.

    Note: Elementary SPPRC is NP-hard, so this may be slow for large instances.
    Consider using ng-route relaxation for better performance.
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None
    ):
        """
        Initialize elementary labeling algorithm.

        Args:
            problem: The Problem instance
            config: Optional configuration (check_elementarity will be set True)
        """
        if config is None:
            config = PricingConfig()
        config.check_elementarity = True

        super().__init__(problem, config)


class HeuristicLabelingAlgorithm(LabelingAlgorithm):
    """
    Heuristic labeling algorithm with aggressive pruning.

    This variant uses more aggressive pruning strategies to find
    columns quickly, at the cost of potentially missing some columns.

    Strategies:
    - Limit labels per node
    - Only keep best N labels at each node
    - Early termination when enough columns found
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        max_labels_per_node: int = 100,
        early_termination_count: int = 10,
    ):
        """
        Initialize heuristic labeling algorithm.

        Args:
            problem: The Problem instance
            config: Optional configuration
            max_labels_per_node: Maximum labels to keep at each node
            early_termination_count: Stop when this many columns found
        """
        super().__init__(problem, config)
        self._max_labels_per_node = max_labels_per_node
        self._early_termination_count = early_termination_count

    def _solve_impl(self) -> PricingSolution:
        """
        Run heuristic labeling with early termination.
        """
        start_time = time.time()

        # Initialize
        self._label_pool = LabelPool(self._problem.network.num_nodes)
        self._next_label_id = 0

        # Track columns found so far (for early termination)
        columns_found: list[Column] = []

        # Create source label
        source_label = self._create_source_label()
        self._label_pool.add_label(
            source_label,
            self._problem.resources,
            check_dominance=False
        )

        # Priority queue
        pq: list[tuple[float, int, Label]] = []
        heapq.heappush(pq, (source_label.reduced_cost, source_label.label_id, source_label))

        processed_count = 0

        while pq:
            # Check early termination
            if len(columns_found) >= self._early_termination_count:
                break

            # Check limits
            if self._config.max_labels > 0:
                if self._label_pool.total_created >= self._config.max_labels:
                    break

            if self._config.max_time > 0:
                if time.time() - start_time >= self._config.max_time:
                    break

            _, _, label = heapq.heappop(pq)
            processed_count += 1

            # If at sink with negative RC, add to columns
            if label.node_index == self._sink_idx:
                if label.reduced_cost < self._config.reduced_cost_threshold:
                    column = self._create_column_from_label(label)
                    columns_found.append(column)
                continue

            # Extend along outgoing arcs
            for arc in self._problem.network.outgoing_arcs(label.node_index):
                new_label = self._extend_label(label, arc)

                if new_label is None:
                    continue

                # Check if we should add this label
                node_labels = self._label_pool.get_labels(new_label.node_index)
                if len(node_labels) >= self._max_labels_per_node:
                    # Only add if better than worst existing
                    worst_rc = max(lbl.reduced_cost for lbl in node_labels)
                    if new_label.reduced_cost >= worst_rc:
                        continue

                added = self._label_pool.add_label(
                    new_label,
                    self._problem.resources,
                    check_dominance=self._config.use_dominance
                )

                if added:
                    heapq.heappush(pq, (new_label.reduced_cost, new_label.label_id, new_label))

        # Sort columns by reduced cost
        columns_found.sort(key=lambda c: c.reduced_cost)

        if self._config.max_columns > 0:
            columns_found = columns_found[:self._config.max_columns]

        # Build solution
        solve_time = time.time() - start_time
        stats = self._label_pool.statistics()

        best_rc = min(c.reduced_cost for c in columns_found) if columns_found else None
        status = (
            PricingStatus.COLUMNS_FOUND if columns_found
            else PricingStatus.NO_COLUMNS
        )

        return PricingSolution(
            status=status,
            columns=columns_found,
            best_reduced_cost=best_rc,
            num_labels_created=stats['total_created'],
            num_labels_dominated=stats['total_dominated'],
            solve_time=solve_time,
            iterations=processed_count,
        )
