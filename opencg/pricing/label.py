"""
Label module for SPPRC labeling algorithm.

In the labeling algorithm for SPPRC, a "label" represents a partial path
from the source node to some intermediate node. Each label tracks:
- The path taken (sequence of arcs)
- The cost so far
- Resource values at the current node
- Items covered by the path

Labels are extended along arcs to create new labels. Dominated labels
are pruned to keep the algorithm efficient.

Design Notes:
------------
- Labels are immutable once created (for safe dominance checking)
- Each label knows its predecessor for path reconstruction
- Resource values are stored as a dict for flexibility
- Dominance is checked across all resources
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from opencg.core.resource import Resource


@dataclass(frozen=True)
class Label:
    """
    A label representing a partial path in SPPRC.

    Labels are the core data structure in the labeling algorithm. Each label
    represents a path from the source to a specific node, with associated
    cost and resource consumption.

    Attributes:
        node_index: Index of the node this label is at
        cost: Total cost of the path so far
        reduced_cost: Cost minus dual adjustments (for pricing)
        resource_values: Dict mapping resource name to current value
        covered_items: Set of item IDs covered by this path
        predecessor: Previous label in the path (None for source)
        last_arc_index: Index of the arc used to reach this label

    The label is frozen (immutable) because:
    1. Dominance checking requires stable values
    2. Labels may be stored in sets for efficient lookup
    3. Path reconstruction uses predecessor chain

    Example:
        >>> # Create source label
        >>> source_label = Label(
        ...     node_index=0,
        ...     cost=0.0,
        ...     reduced_cost=0.0,
        ...     resource_values={'duty_time': 0.0, 'flight_time': 0.0},
        ...     covered_items=frozenset(),
        ... )
        >>>
        >>> # Extend along an arc
        >>> new_label = Label(
        ...     node_index=5,
        ...     cost=100.0,
        ...     reduced_cost=100.0 - 50.0,  # cost - dual
        ...     resource_values={'duty_time': 2.5, 'flight_time': 2.0},
        ...     covered_items=frozenset({10}),
        ...     predecessor=source_label,
        ...     last_arc_index=3,
        ... )
    """
    # Current position in the network
    node_index: int

    # Costs
    cost: float  # Total path cost (sum of arc costs)
    reduced_cost: float  # Cost - sum of duals for covered items

    # Resource state (resource_name -> value)
    # Note: Values can be any type (float, set, tuple, etc.)
    resource_values: dict[str, Any] = field(default_factory=dict)

    # Items covered by this path
    covered_items: frozenset[int] = field(default_factory=frozenset)

    # Path reconstruction
    predecessor: Optional['Label'] = None
    last_arc_index: Optional[int] = None

    # Unique identifier (for debugging and tracking)
    label_id: Optional[int] = None

    def __post_init__(self):
        """Ensure covered_items is a frozenset."""
        if not isinstance(self.covered_items, frozenset):
            object.__setattr__(self, 'covered_items', frozenset(self.covered_items))

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_source_label(self) -> bool:
        """Check if this is the source label (no predecessor)."""
        return self.predecessor is None

    @property
    def path_length(self) -> int:
        """Number of arcs in the path."""
        if self.predecessor is None:
            return 0
        return self.predecessor.path_length + 1

    # =========================================================================
    # Resource Access
    # =========================================================================

    def get_resource(self, name: str, default: Any = None) -> Any:
        """
        Get a resource value.

        Args:
            name: Resource name
            default: Default value if not found

        Returns:
            Resource value
        """
        return self.resource_values.get(name, default)

    # =========================================================================
    # Path Reconstruction
    # =========================================================================

    def get_arc_indices(self) -> tuple[int, ...]:
        """
        Reconstruct the sequence of arc indices.

        Returns:
            Tuple of arc indices from source to this label
        """
        if self.predecessor is None:
            return ()

        # Recursively get predecessor's path
        path = list(self.predecessor.get_arc_indices())
        if self.last_arc_index is not None:
            path.append(self.last_arc_index)
        return tuple(path)

    def get_node_indices(self) -> tuple[int, ...]:
        """
        Reconstruct the sequence of node indices.

        Returns:
            Tuple of node indices from source to this label
        """
        if self.predecessor is None:
            return (self.node_index,)

        path = list(self.predecessor.get_node_indices())
        path.append(self.node_index)
        return tuple(path)

    # =========================================================================
    # Comparison (for sorting in priority queue)
    # =========================================================================

    def __lt__(self, other: 'Label') -> bool:
        """Compare by reduced cost (for min-heap)."""
        if not isinstance(other, Label):
            return NotImplemented
        return self.reduced_cost < other.reduced_cost

    def __le__(self, other: 'Label') -> bool:
        """Compare by reduced cost."""
        if not isinstance(other, Label):
            return NotImplemented
        return self.reduced_cost <= other.reduced_cost

    # =========================================================================
    # Hashing and Equality
    # =========================================================================

    def __hash__(self) -> int:
        """Hash based on node and resource values."""
        # We use node_index and a tuple of sorted resource items
        # This allows labels at same node with same resources to be compared
        resource_tuple = tuple(sorted(
            (k, v if not isinstance(v, set) else frozenset(v))
            for k, v in self.resource_values.items()
        ))
        return hash((self.node_index, resource_tuple, self.covered_items))

    def __eq__(self, other: object) -> bool:
        """Equality based on node and resource values."""
        if not isinstance(other, Label):
            return NotImplemented
        return (
            self.node_index == other.node_index and
            self.resource_values == other.resource_values and
            self.covered_items == other.covered_items
        )

    def __repr__(self) -> str:
        rc_str = f", rc={self.reduced_cost:.4f}" if self.reduced_cost else ""
        items_str = f", covers={len(self.covered_items)}" if self.covered_items else ""
        return f"Label(node={self.node_index}, cost={self.cost:.2f}{rc_str}{items_str})"


class LabelPool:
    """
    Container for managing labels at each node during SPPRC.

    The LabelPool provides:
    - Storage of labels by node
    - Dominance checking within a node
    - Efficient retrieval of non-dominated labels

    This is used by the labeling algorithm to manage the set of labels
    at each node, pruning dominated labels to keep the algorithm efficient.
    """

    def __init__(self, num_nodes: int):
        """
        Create a label pool.

        Args:
            num_nodes: Number of nodes in the network
        """
        self._num_nodes = num_nodes
        # Labels at each node: node_index -> list of non-dominated labels
        self._labels: list[list[Label]] = [[] for _ in range(num_nodes)]
        # Total labels created (for statistics)
        self._total_created: int = 0
        self._total_dominated: int = 0

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self._num_nodes

    @property
    def total_labels(self) -> int:
        """Total number of labels currently stored."""
        return sum(len(labels) for labels in self._labels)

    @property
    def total_created(self) -> int:
        """Total labels created during algorithm."""
        return self._total_created

    @property
    def total_dominated(self) -> int:
        """Total labels pruned by dominance."""
        return self._total_dominated

    def get_labels(self, node_index: int) -> list[Label]:
        """
        Get all non-dominated labels at a node.

        Args:
            node_index: Node index

        Returns:
            List of labels at the node
        """
        return self._labels[node_index].copy()

    def add_label(
        self,
        label: Label,
        resources: list['Resource'],
        check_dominance: bool = True
    ) -> bool:
        """
        Add a label to the pool, checking dominance.

        Args:
            label: The label to add
            resources: List of resources for dominance checking
            check_dominance: Whether to check/apply dominance

        Returns:
            True if label was added (not dominated), False if dominated
        """
        self._total_created += 1
        node_idx = label.node_index

        if not check_dominance:
            self._labels[node_idx].append(label)
            return True

        # Check if new label is dominated by existing labels
        existing = self._labels[node_idx]
        for existing_label in existing:
            if self._dominates(existing_label, label, resources):
                self._total_dominated += 1
                return False  # New label is dominated

        # Remove existing labels dominated by new label
        non_dominated = []
        for existing_label in existing:
            if not self._dominates(label, existing_label, resources):
                non_dominated.append(existing_label)
            else:
                self._total_dominated += 1

        non_dominated.append(label)
        self._labels[node_idx] = non_dominated

        return True

    def _dominates(
        self,
        label1: Label,
        label2: Label,
        resources: list['Resource']
    ) -> bool:
        """
        Check if label1 dominates label2.

        Label1 dominates label2 if:
        1. They are at the same node
        2. label1.reduced_cost <= label2.reduced_cost
        3. label1 dominates on all resources
        4. label1.covered_items is subset of label2.covered_items
           (covering fewer items is better - more flexibility)

        Args:
            label1: Potentially dominating label
            label2: Potentially dominated label
            resources: List of resources

        Returns:
            True if label1 dominates label2
        """
        # Must be at same node
        if label1.node_index != label2.node_index:
            return False

        # Check reduced cost (for minimization)
        if label1.reduced_cost > label2.reduced_cost:
            return False

        # Check all resources
        for resource in resources:
            val1 = label1.get_resource(resource.name)
            val2 = label2.get_resource(resource.name)

            if val1 is None or val2 is None:
                continue

            if not resource.dominates(val1, val2):
                return False

        # Check covered items (fewer is better)
        if not label1.covered_items.issubset(label2.covered_items):
            return False

        # All conditions met - label1 dominates label2
        return True

    def clear(self) -> None:
        """Clear all labels."""
        self._labels = [[] for _ in range(self._num_nodes)]
        self._total_created = 0
        self._total_dominated = 0

    def statistics(self) -> dict[str, Any]:
        """
        Get statistics about the label pool.

        Returns:
            Dictionary with statistics
        """
        labels_per_node = [len(labels) for labels in self._labels]
        return {
            'total_labels': self.total_labels,
            'total_created': self._total_created,
            'total_dominated': self._total_dominated,
            'dominance_rate': (
                self._total_dominated / self._total_created
                if self._total_created > 0 else 0.0
            ),
            'max_labels_at_node': max(labels_per_node) if labels_per_node else 0,
            'avg_labels_per_node': (
                sum(labels_per_node) / len(labels_per_node)
                if labels_per_node else 0.0
            ),
        }

    def __repr__(self) -> str:
        return f"LabelPool(nodes={self._num_nodes}, labels={self.total_labels})"
