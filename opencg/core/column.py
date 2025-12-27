"""
Column module - represents a column (path) in the column generation framework.

In column generation, a "column" is a feasible solution to the pricing
subproblem. For crew pairing, a column is a pairing (sequence of flights
and rests that a crew can legally operate).

This module provides:
- Column: The main column class storing path, cost, and resource values

Design Notes:
------------
- A column is essentially a path in the network plus aggregated information
- Columns are immutable once created (hashable for use in sets)
- We store both arc sequence and resource values at the end
- The reduced cost is computed during pricing, stored for convenience

Column Lifecycle:
----------------
1. Created by pricing subproblem (SPPRC finds path with negative reduced cost)
2. Added to master problem (becomes a variable)
3. May be part of optimal solution (variable has positive value)
4. Stored in column pool for potential reuse

Future C++ Note:
---------------
Columns will be stored in a ColumnPool (C++) for memory efficiency.
The Python Column class will wrap a C++ column reference.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class Column:
    """
    Represents a column (feasible path) in column generation.

    A Column stores:
    - The sequence of arcs forming the path
    - The cost of the path (sum of arc costs)
    - Final resource values after traversing the path
    - Which constraints this column covers
    - Computed values from the solver (reduced cost, value in solution)

    Immutability:
        Column is immutable (frozen dataclass) because:
        1. Columns are added to sets/dicts (need hashable)
        2. Once created, a column's path doesn't change
        3. Solver values are set once after solving

    Attributes:
        arc_indices: Tuple of arc indices forming the path
        cost: Total cost of the path (objective coefficient)
        resource_values: Final resource values after traversing path
        covered_items: Set of item indices this column covers (for set covering/partitioning)
        column_id: Optional unique identifier
        reduced_cost: Reduced cost (set during pricing)
        value: Value in the solution (set after solving master)
        attributes: Additional attributes (e.g., origin base, pairing type)

    Example:
        >>> # Create a column representing a crew pairing
        >>> column = Column(
        ...     arc_indices=(0, 1, 2, 3),  # Sequence of arc indices
        ...     cost=500.0,
        ...     resource_values={"duty_time": 8.5, "flight_time": 6.0},
        ...     covered_items=frozenset({10, 15, 20}),  # Flights covered
        ...     attributes={"base": "JFK", "days": 2}
        ... )
        >>>
        >>> print(column.cost)
        500.0
        >>> print(column.covers_item(15))
        True
    """
    # Path representation
    arc_indices: Tuple[int, ...]

    # Cost (objective function coefficient)
    cost: float

    # Final resource values (resource_name -> value)
    resource_values: Dict[str, Any] = field(default_factory=dict)

    # Items covered by this column (for set covering/partitioning constraints)
    # Using FrozenSet for hashability
    covered_items: FrozenSet[int] = field(default_factory=frozenset)

    # Optional identifier
    column_id: Optional[int] = None

    # Values computed during/after solving
    reduced_cost: Optional[float] = None
    value: Optional[float] = None  # Value in solution (lambda)

    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure arc_indices is a tuple and covered_items is a frozenset."""
        # frozen dataclass doesn't allow assignment, use object.__setattr__
        if not isinstance(self.arc_indices, tuple):
            object.__setattr__(self, 'arc_indices', tuple(self.arc_indices))
        if not isinstance(self.covered_items, frozenset):
            object.__setattr__(self, 'covered_items', frozenset(self.covered_items))
        # Convert resource_values dict to be hashable by making it immutable
        # Note: We keep it as dict for easy access, but hash will use arc_indices
        if not isinstance(self.resource_values, dict):
            object.__setattr__(self, 'resource_values', dict(self.resource_values))
        if not isinstance(self.attributes, dict):
            object.__setattr__(self, 'attributes', dict(self.attributes))

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_arcs(self) -> int:
        """Number of arcs in the path."""
        return len(self.arc_indices)

    @property
    def is_in_solution(self) -> bool:
        """Check if this column is part of the solution (value > 0)."""
        return self.value is not None and self.value > 1e-6

    @property
    def is_integer(self) -> bool:
        """Check if the column value is (nearly) integer."""
        if self.value is None:
            return False
        return abs(self.value - round(self.value)) < 1e-6

    # =========================================================================
    # Methods
    # =========================================================================

    def covers_item(self, item: int) -> bool:
        """
        Check if this column covers a specific item.

        Args:
            item: Item index (e.g., flight index)

        Returns:
            True if this column covers the item
        """
        return item in self.covered_items

    def get_resource(self, resource_name: str, default: Any = None) -> Any:
        """
        Get final resource value.

        Args:
            resource_name: Name of the resource
            default: Default value if not found

        Returns:
            Final resource value after traversing the path
        """
        return self.resource_values.get(resource_name, default)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """
        Get an attribute value.

        Args:
            key: Attribute name
            default: Default value if not found

        Returns:
            Attribute value
        """
        return self.attributes.get(key, default)

    def with_reduced_cost(self, reduced_cost: float) -> 'Column':
        """
        Create a copy with reduced_cost set.

        Since Column is immutable, this returns a new Column.

        Args:
            reduced_cost: The reduced cost value

        Returns:
            New Column with reduced_cost set
        """
        return Column(
            arc_indices=self.arc_indices,
            cost=self.cost,
            resource_values=self.resource_values,
            covered_items=self.covered_items,
            column_id=self.column_id,
            reduced_cost=reduced_cost,
            value=self.value,
            attributes=self.attributes
        )

    def with_value(self, value: float) -> 'Column':
        """
        Create a copy with value set.

        Args:
            value: The solution value

        Returns:
            New Column with value set
        """
        return Column(
            arc_indices=self.arc_indices,
            cost=self.cost,
            resource_values=self.resource_values,
            covered_items=self.covered_items,
            column_id=self.column_id,
            reduced_cost=self.reduced_cost,
            value=value,
            attributes=self.attributes
        )

    def with_id(self, column_id: int) -> 'Column':
        """
        Create a copy with column_id set.

        Args:
            column_id: The unique identifier

        Returns:
            New Column with column_id set
        """
        return Column(
            arc_indices=self.arc_indices,
            cost=self.cost,
            resource_values=self.resource_values,
            covered_items=self.covered_items,
            column_id=column_id,
            reduced_cost=self.reduced_cost,
            value=self.value,
            attributes=self.attributes
        )

    def __hash__(self) -> int:
        """Hash based on arc sequence (paths are unique by their arcs)."""
        return hash(self.arc_indices)

    def __eq__(self, other: object) -> bool:
        """Equality based on arc sequence."""
        if not isinstance(other, Column):
            return NotImplemented
        return self.arc_indices == other.arc_indices

    def __repr__(self) -> str:
        items_str = f", covers={len(self.covered_items)} items" if self.covered_items else ""
        value_str = f", value={self.value:.4f}" if self.value is not None else ""
        rc_str = f", rc={self.reduced_cost:.4f}" if self.reduced_cost is not None else ""
        return f"Column(arcs={len(self.arc_indices)}, cost={self.cost:.2f}{items_str}{value_str}{rc_str})"


# =============================================================================
# Column Pool
# =============================================================================


class ColumnPool:
    """
    Container for storing and managing columns.

    The ColumnPool provides:
    - Efficient storage of columns
    - Lookup by column_id
    - Filtering by coverage
    - Statistics

    This is a simple Python implementation. The C++ version will be more
    memory-efficient for large numbers of columns.

    Example:
        >>> pool = ColumnPool()
        >>> pool.add(column1)
        >>> pool.add(column2)
        >>> print(pool.size)
        2
        >>> for col in pool.columns_covering(flight_id=10):
        ...     print(col)
    """

    def __init__(self):
        """Create an empty column pool."""
        self._columns: List[Column] = []
        self._id_to_index: Dict[int, int] = {}
        self._next_id: int = 0

    @property
    def size(self) -> int:
        """Number of columns in the pool."""
        return len(self._columns)

    def add(self, column: Column) -> Column:
        """
        Add a column to the pool.

        If the column doesn't have an ID, one is assigned.

        Args:
            column: Column to add

        Returns:
            Column with ID assigned
        """
        if column.column_id is None:
            column = column.with_id(self._next_id)
            self._next_id += 1

        index = len(self._columns)
        self._columns.append(column)
        self._id_to_index[column.column_id] = index

        return column

    def get(self, column_id: int) -> Optional[Column]:
        """
        Get a column by ID.

        Args:
            column_id: Column identifier

        Returns:
            The column, or None if not found
        """
        index = self._id_to_index.get(column_id)
        if index is None:
            return None
        return self._columns[index]

    def all_columns(self) -> List[Column]:
        """Get all columns in the pool."""
        return self._columns.copy()

    def columns_covering(self, item: int) -> List[Column]:
        """
        Get columns that cover a specific item.

        Args:
            item: Item index

        Returns:
            List of columns covering the item
        """
        return [col for col in self._columns if col.covers_item(item)]

    def columns_with_positive_value(self) -> List[Column]:
        """Get columns with positive value in the solution."""
        return [col for col in self._columns if col.is_in_solution]

    def total_cost(self) -> float:
        """Total cost of columns with positive values."""
        return sum(
            col.cost * col.value
            for col in self._columns
            if col.value is not None
        )

    def clear(self) -> None:
        """Remove all columns from the pool."""
        self._columns.clear()
        self._id_to_index.clear()
        self._next_id = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self._columns)

    def __repr__(self) -> str:
        return f"ColumnPool(size={self.size})"
