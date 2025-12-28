"""
Network module - the graph structure for column generation problems.

The Network is the container that holds all nodes and arcs, and provides
efficient access patterns for graph traversal (needed in SPPRC).

This module provides:
- Network: The main graph class with add/get methods

Design Notes:
------------
- Nodes and arcs are stored in lists, indexed by their index attribute
- Adjacency is stored as outgoing arcs per node (for forward traversal)
- Incoming arcs can be computed if needed (for backward traversal)
- The source and sink nodes have special handling

Future C++ Note:
---------------
This is one of the first classes to migrate to C++, because:
1. Large networks consume significant memory
2. Graph traversal is in the inner loop of SPPRC
3. C++ can use cache-efficient data structures

The C++ version will use:
- std::vector<Node> for nodes
- std::vector<Arc> for arcs
- std::vector<std::vector<int>> for adjacency (arc indices)
- Optional: CSR format for even better cache performance
"""

from collections.abc import Iterator
from typing import Optional

from opencg.core.arc import Arc, ArcType
from opencg.core.node import Node, NodeType


class Network:
    """
    Graph structure for column generation problems.

    The Network stores nodes and arcs, and provides efficient access
    for graph algorithms. It supports:
    - Adding nodes and arcs
    - Forward traversal (outgoing arcs from a node)
    - Backward traversal (incoming arcs to a node)
    - Source/sink node management

    Attributes:
        nodes: List of all nodes (indexed by node.index)
        arcs: List of all arcs (indexed by arc.index)
        source: The artificial source node (or None)
        sink: The artificial sink node (or None)

    Example:
        >>> network = Network()
        >>>
        >>> # Add nodes
        >>> source = network.add_source()
        >>> sink = network.add_sink()
        >>> base1 = network.add_node("BASE1", NodeType.BASE)
        >>> air1 = network.add_node("AIR1", NodeType.AIRPORT)
        >>>
        >>> # Add arcs
        >>> network.add_arc(
        ...     source=base1,
        ...     target=air1,
        ...     cost=100.0,
        ...     resource_consumption={"duty_time": 2.0},
        ...     arc_type=ArcType.FLIGHT
        ... )
        >>>
        >>> # Traverse
        >>> for arc in network.outgoing_arcs(base1):
        ...     print(arc)

    Note:
        Node and arc indices are assigned automatically and should not
        be modified after creation.
    """

    def __init__(self):
        """Create an empty network."""
        # Storage
        self._nodes: list[Node] = []
        self._arcs: list[Arc] = []

        # Adjacency lists (node index -> list of arc indices)
        self._outgoing: list[list[int]] = []
        self._incoming: list[list[int]] = []

        # Name to index mapping for lookup
        self._node_name_to_index: dict[str, int] = {}

        # Special nodes
        self._source_index: Optional[int] = None
        self._sink_index: Optional[int] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self._nodes)

    @property
    def num_arcs(self) -> int:
        """Number of arcs in the network."""
        return len(self._arcs)

    @property
    def nodes(self) -> list[Node]:
        """List of all nodes (read-only view)."""
        return self._nodes

    @property
    def arcs(self) -> list[Arc]:
        """List of all arcs (read-only view)."""
        return self._arcs

    @property
    def source(self) -> Optional[Node]:
        """The artificial source node, or None if not set."""
        if self._source_index is None:
            return None
        return self._nodes[self._source_index]

    @property
    def sink(self) -> Optional[Node]:
        """The artificial sink node, or None if not set."""
        if self._sink_index is None:
            return None
        return self._nodes[self._sink_index]

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(
        self,
        name: str,
        node_type: NodeType = NodeType.GENERIC,
        **attributes
    ) -> int:
        """
        Add a node to the network.

        Args:
            name: Unique name for the node
            node_type: Type of the node (see NodeType enum)
            **attributes: Additional node attributes

        Returns:
            Index of the newly created node

        Raises:
            ValueError: If a node with this name already exists
        """
        if name in self._node_name_to_index:
            raise ValueError(f"Node with name '{name}' already exists")

        index = len(self._nodes)
        node = Node(
            index=index,
            name=name,
            node_type=node_type,
            attributes=dict(attributes)
        )
        self._nodes.append(node)
        self._outgoing.append([])
        self._incoming.append([])
        self._node_name_to_index[name] = index

        return index

    def add_source(self, name: str = "__SOURCE__") -> int:
        """
        Add the artificial source node.

        The source node is where all paths begin. There should be exactly
        one source node in the network.

        Args:
            name: Name for the source node

        Returns:
            Index of the source node

        Raises:
            ValueError: If source already exists
        """
        if self._source_index is not None:
            raise ValueError("Source node already exists")

        self._source_index = self.add_node(name, NodeType.SOURCE)
        return self._source_index

    def add_sink(self, name: str = "__SINK__") -> int:
        """
        Add the artificial sink node.

        The sink node is where all paths end. There should be exactly
        one sink node in the network.

        Args:
            name: Name for the sink node

        Returns:
            Index of the sink node

        Raises:
            ValueError: If sink already exists
        """
        if self._sink_index is not None:
            raise ValueError("Sink node already exists")

        self._sink_index = self.add_node(name, NodeType.SINK)
        return self._sink_index

    def get_node(self, index: int) -> Node:
        """
        Get a node by index.

        Args:
            index: Node index

        Returns:
            The node

        Raises:
            IndexError: If index is out of bounds
        """
        return self._nodes[index]

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """
        Get a node by name.

        Args:
            name: Node name

        Returns:
            The node, or None if not found
        """
        index = self._node_name_to_index.get(name)
        if index is None:
            return None
        return self._nodes[index]

    def get_node_index(self, name: str) -> Optional[int]:
        """
        Get node index by name.

        Args:
            name: Node name

        Returns:
            Node index, or None if not found
        """
        return self._node_name_to_index.get(name)

    # =========================================================================
    # Arc Operations
    # =========================================================================

    def add_arc(
        self,
        source: int,
        target: int,
        cost: float,
        resource_consumption: Optional[dict[str, float]] = None,
        arc_type: ArcType = ArcType.GENERIC,
        **attributes
    ) -> int:
        """
        Add an arc to the network.

        Args:
            source: Index of source node
            target: Index of target node
            cost: Cost of the arc (for objective function)
            resource_consumption: Dict mapping resource name -> consumption
            arc_type: Type of the arc (see ArcType enum)
            **attributes: Additional arc attributes

        Returns:
            Index of the newly created arc

        Raises:
            IndexError: If source or target node doesn't exist
        """
        # Validate node indices
        if source < 0 or source >= len(self._nodes):
            raise IndexError(f"Source node index {source} out of bounds")
        if target < 0 or target >= len(self._nodes):
            raise IndexError(f"Target node index {target} out of bounds")

        index = len(self._arcs)
        arc = Arc(
            index=index,
            source=source,
            target=target,
            cost=cost,
            resource_consumption=resource_consumption or {},
            arc_type=arc_type,
            attributes=dict(attributes)
        )
        self._arcs.append(arc)

        # Update adjacency lists
        self._outgoing[source].append(index)
        self._incoming[target].append(index)

        return index

    def get_arc(self, index: int) -> Arc:
        """
        Get an arc by index.

        Args:
            index: Arc index

        Returns:
            The arc

        Raises:
            IndexError: If index is out of bounds
        """
        return self._arcs[index]

    # =========================================================================
    # Traversal Operations
    # =========================================================================

    def outgoing_arcs(self, node: int) -> Iterator[Arc]:
        """
        Iterate over outgoing arcs from a node.

        This is the primary traversal method used in SPPRC.

        Args:
            node: Node index

        Yields:
            Arc objects leaving the node
        """
        for arc_index in self._outgoing[node]:
            yield self._arcs[arc_index]

    def incoming_arcs(self, node: int) -> Iterator[Arc]:
        """
        Iterate over incoming arcs to a node.

        Used for backward traversal algorithms.

        Args:
            node: Node index

        Yields:
            Arc objects entering the node
        """
        for arc_index in self._incoming[node]:
            yield self._arcs[arc_index]

    def outgoing_arc_indices(self, node: int) -> list[int]:
        """
        Get list of outgoing arc indices from a node.

        Args:
            node: Node index

        Returns:
            List of arc indices
        """
        return self._outgoing[node]

    def incoming_arc_indices(self, node: int) -> list[int]:
        """
        Get list of incoming arc indices to a node.

        Args:
            node: Node index

        Returns:
            List of arc indices
        """
        return self._incoming[node]

    def neighbors(self, node: int) -> Iterator[int]:
        """
        Iterate over successor node indices.

        Args:
            node: Node index

        Yields:
            Indices of nodes reachable via outgoing arcs
        """
        for arc_index in self._outgoing[node]:
            yield self._arcs[arc_index].target

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def nodes_of_type(self, node_type: NodeType) -> Iterator[Node]:
        """
        Iterate over nodes of a specific type.

        Args:
            node_type: Type of nodes to return

        Yields:
            Nodes of the specified type
        """
        for node in self._nodes:
            if node.node_type == node_type:
                yield node

    def arcs_of_type(self, arc_type: ArcType) -> Iterator[Arc]:
        """
        Iterate over arcs of a specific type.

        Args:
            arc_type: Type of arcs to return

        Yields:
            Arcs of the specified type
        """
        for arc in self._arcs:
            if arc.arc_type == arc_type:
                yield arc

    def get_bases(self) -> list[Node]:
        """Get list of base nodes."""
        return list(self.nodes_of_type(NodeType.BASE))

    def get_flights(self) -> list[Arc]:
        """Get list of flight arcs."""
        return list(self.arcs_of_type(ArcType.FLIGHT))

    def validate(self) -> list[str]:
        """
        Validate the network structure.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check source/sink
        if self._source_index is None:
            errors.append("Network has no source node")
        if self._sink_index is None:
            errors.append("Network has no sink node")

        # Check node indices
        for i, node in enumerate(self._nodes):
            if node.index != i:
                errors.append(f"Node at position {i} has index {node.index}")

        # Check arc indices and references
        for i, arc in enumerate(self._arcs):
            if arc.index != i:
                errors.append(f"Arc at position {i} has index {arc.index}")
            if arc.source < 0 or arc.source >= len(self._nodes):
                errors.append(f"Arc {i} has invalid source {arc.source}")
            if arc.target < 0 or arc.target >= len(self._nodes):
                errors.append(f"Arc {i} has invalid target {arc.target}")

        return errors

    def summary(self) -> str:
        """
        Return a summary string of the network.

        Returns:
            Human-readable summary
        """
        lines = [
            f"Network: {self.num_nodes} nodes, {self.num_arcs} arcs",
            f"  Source: {self.source.name if self.source else 'None'}",
            f"  Sink: {self.sink.name if self.sink else 'None'}",
        ]

        # Count by type
        node_type_counts = {}
        for node in self._nodes:
            t = node.node_type.name
            node_type_counts[t] = node_type_counts.get(t, 0) + 1

        arc_type_counts = {}
        for arc in self._arcs:
            t = arc.arc_type.name
            arc_type_counts[t] = arc_type_counts.get(t, 0) + 1

        lines.append("  Node types: " + ", ".join(
            f"{t}={c}" for t, c in sorted(node_type_counts.items())
        ))
        lines.append("  Arc types: " + ", ".join(
            f"{t}={c}" for t, c in sorted(arc_type_counts.items())
        ))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Network(nodes={self.num_nodes}, arcs={self.num_arcs})"
