"""
Master problem solution module.

This module defines the data structures for representing solutions
from the master problem solver.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class SolutionStatus(Enum):
    """
    Status of the master problem solution.

    These statuses cover both LP and IP solving outcomes.
    """
    OPTIMAL = auto()           # Optimal solution found
    INFEASIBLE = auto()        # Problem is infeasible
    UNBOUNDED = auto()         # Problem is unbounded
    INF_OR_UNBOUNDED = auto()  # Infeasible or unbounded (solver couldn't determine)
    TIME_LIMIT = auto()        # Time limit reached (may have feasible solution)
    ITERATION_LIMIT = auto()   # Iteration limit reached
    NODE_LIMIT = auto()        # Node limit reached (for IP)
    NOT_SOLVED = auto()        # Solve not called yet
    ERROR = auto()             # Solver error occurred


@dataclass
class MasterSolution:
    """
    Result of solving the master problem.

    This class holds all information about a master problem solution,
    including primal values (column values), dual values (for pricing),
    and solver statistics.

    Attributes:
        status: Solution status (OPTIMAL, INFEASIBLE, etc.)
        objective_value: Objective function value (None if not solved/infeasible)
        column_values: Mapping from column_id to its value in solution
        dual_values: Mapping from item_id to dual value (pi) for pricing
        reduced_costs: Mapping from column_id to reduced cost (optional)
        solve_time: Time spent solving in seconds
        iterations: Number of LP iterations (simplex pivots)
        nodes: Number of B&B nodes explored (for IP)
        num_columns: Number of columns in the model when solved
        gap: Optimality gap for IP (None for LP)
        basis: Solver basis for warm starting (solver-specific format)

    Example:
        >>> solution = master.solve_lp()
        >>> if solution.is_optimal:
        ...     print(f"Objective: {solution.objective_value}")
        ...     for item_id, dual in solution.dual_values.items():
        ...         print(f"  Dual[{item_id}] = {dual}")
    """
    # Solution status
    status: SolutionStatus = SolutionStatus.NOT_SOLVED

    # Objective value
    objective_value: Optional[float] = None

    # Primal solution: column_id -> value (lambda)
    column_values: Dict[int, float] = field(default_factory=dict)

    # Dual solution: item_id -> dual value (pi)
    dual_values: Dict[int, float] = field(default_factory=dict)

    # Reduced costs: column_id -> reduced cost (optional, for diagnostics)
    reduced_costs: Dict[int, float] = field(default_factory=dict)

    # Solver statistics
    solve_time: float = 0.0
    iterations: int = 0
    nodes: int = 0  # For IP solving
    num_columns: int = 0

    # Optimality gap (for IP)
    gap: Optional[float] = None

    # Basis for warm starting (solver-specific)
    basis: Optional[Any] = None

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def is_optimal(self) -> bool:
        """Check if solution is optimal."""
        return self.status == SolutionStatus.OPTIMAL

    @property
    def is_infeasible(self) -> bool:
        """Check if problem is infeasible."""
        return self.status == SolutionStatus.INFEASIBLE

    @property
    def is_unbounded(self) -> bool:
        """Check if problem is unbounded."""
        return self.status == SolutionStatus.UNBOUNDED

    @property
    def has_solution(self) -> bool:
        """Check if a feasible solution is available."""
        return self.status in (
            SolutionStatus.OPTIMAL,
            SolutionStatus.TIME_LIMIT,
            SolutionStatus.ITERATION_LIMIT,
            SolutionStatus.NODE_LIMIT,
        ) and self.objective_value is not None

    @property
    def is_integer(self) -> bool:
        """
        Check if the solution is integer.

        Returns True if all column values are (nearly) integer.
        """
        if not self.column_values:
            return True

        tol = 1e-6
        for value in self.column_values.values():
            if abs(value - round(value)) > tol:
                return False
        return True

    # =========================================================================
    # Methods
    # =========================================================================

    def get_active_columns(self, tol: float = 1e-6) -> List[int]:
        """
        Get column IDs with positive value in solution.

        Args:
            tol: Tolerance for considering a value positive

        Returns:
            List of column IDs with value > tol
        """
        return [
            col_id for col_id, value in self.column_values.items()
            if value > tol
        ]

    def get_fractional_columns(self, tol: float = 1e-6) -> List[int]:
        """
        Get column IDs with fractional value in solution.

        Useful for branching decisions in branch-and-price.

        Args:
            tol: Tolerance for integrality check

        Returns:
            List of column IDs with fractional values
        """
        fractional = []
        for col_id, value in self.column_values.items():
            if value > tol and abs(value - round(value)) > tol:
                fractional.append(col_id)
        return fractional

    def get_dual(self, item_id: int, default: float = 0.0) -> float:
        """
        Get dual value for a specific item.

        Args:
            item_id: The item identifier
            default: Default value if item not found

        Returns:
            Dual value for the item
        """
        return self.dual_values.get(item_id, default)

    def summary(self) -> str:
        """
        Return a human-readable summary of the solution.

        Returns:
            Summary string
        """
        lines = [
            f"MasterSolution:",
            f"  Status: {self.status.name}",
        ]

        if self.objective_value is not None:
            lines.append(f"  Objective: {self.objective_value:.6f}")

        if self.gap is not None:
            lines.append(f"  Gap: {self.gap:.4%}")

        active = self.get_active_columns()
        lines.append(f"  Active columns: {len(active)} / {self.num_columns}")

        if not self.is_integer:
            fractional = self.get_fractional_columns()
            lines.append(f"  Fractional columns: {len(fractional)}")
        else:
            lines.append(f"  Solution is integer")

        lines.extend([
            f"  Solve time: {self.solve_time:.3f}s",
            f"  Iterations: {self.iterations}",
        ])

        if self.nodes > 0:
            lines.append(f"  Nodes: {self.nodes}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        obj_str = f", obj={self.objective_value:.4f}" if self.objective_value else ""
        return f"MasterSolution({self.status.name}{obj_str})"
