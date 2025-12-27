"""
Column generation solution module.

This module defines the data structures for representing the results
of the column generation algorithm.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from opencg.core.column import Column


class CGStatus(Enum):
    """
    Status of the column generation algorithm.
    """
    OPTIMAL = auto()           # Converged to LP optimum
    INTEGER_OPTIMAL = auto()   # IP solution is optimal
    FEASIBLE = auto()          # Found feasible solution (not proven optimal)
    INFEASIBLE = auto()        # Master problem is infeasible
    UNBOUNDED = auto()         # Master problem is unbounded
    ITERATION_LIMIT = auto()   # Iteration limit reached
    TIME_LIMIT = auto()        # Time limit reached
    COLUMN_LIMIT = auto()      # Column limit reached
    NO_IMPROVEMENT = auto()    # No improvement in recent iterations
    NOT_SOLVED = auto()        # Not yet solved
    ERROR = auto()             # Error occurred


@dataclass
class CGIteration:
    """
    Information about a single column generation iteration.

    Attributes:
        iteration: Iteration number
        master_objective: Master problem objective value
        best_reduced_cost: Best (most negative) reduced cost from pricing
        num_columns_added: Number of columns added in this iteration
        master_time: Time spent on master problem
        pricing_time: Time spent on pricing problem
        total_columns: Total columns in master after this iteration
        gap: Lagrangian bound gap (if computed)
    """
    iteration: int
    master_objective: float
    best_reduced_cost: Optional[float]
    num_columns_added: int
    master_time: float
    pricing_time: float
    total_columns: int
    gap: Optional[float] = None


@dataclass
class CGSolution:
    """
    Result of the column generation algorithm.

    Attributes:
        status: Solution status
        objective_value: Final objective value (LP or IP)
        lp_objective: LP relaxation objective
        ip_objective: IP objective (if solved)
        columns: All columns in the solution (with positive value)
        total_columns: Total columns generated
        iterations: Number of CG iterations
        total_time: Total solve time
        master_time: Time spent on master problems
        pricing_time: Time spent on pricing problems
        iteration_history: History of each iteration
        gap: Final optimality gap

    Example:
        >>> solution = cg.solve()
        >>> if solution.is_optimal:
        ...     print(f"Optimal value: {solution.objective_value}")
        ...     for col in solution.columns:
        ...         print(f"  {col}")
    """
    # Status
    status: CGStatus = CGStatus.NOT_SOLVED

    # Objective values
    objective_value: Optional[float] = None
    lp_objective: Optional[float] = None
    ip_objective: Optional[float] = None

    # Solution columns (those with positive value)
    columns: List[Column] = field(default_factory=list)

    # Statistics
    total_columns: int = 0
    iterations: int = 0
    total_time: float = 0.0
    master_time: float = 0.0
    pricing_time: float = 0.0

    # Iteration history
    iteration_history: List[CGIteration] = field(default_factory=list)

    # Optimality gap
    gap: Optional[float] = None

    # Lower bound (from Lagrangian relaxation)
    lower_bound: Optional[float] = None

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_optimal(self) -> bool:
        """Check if solution is proven optimal."""
        return self.status in (CGStatus.OPTIMAL, CGStatus.INTEGER_OPTIMAL)

    @property
    def is_feasible(self) -> bool:
        """Check if a feasible solution was found."""
        return self.status in (
            CGStatus.OPTIMAL,
            CGStatus.INTEGER_OPTIMAL,
            CGStatus.FEASIBLE,
            CGStatus.ITERATION_LIMIT,
            CGStatus.TIME_LIMIT,
            CGStatus.COLUMN_LIMIT,
            CGStatus.NO_IMPROVEMENT,
        )

    @property
    def is_integer(self) -> bool:
        """Check if solution is integer."""
        if not self.columns:
            return True
        return all(
            col.value is not None and abs(col.value - round(col.value)) < 1e-6
            for col in self.columns
        )

    @property
    def num_active_columns(self) -> int:
        """Number of columns with positive value."""
        return len(self.columns)

    # =========================================================================
    # Methods
    # =========================================================================

    def get_column_values(self) -> Dict[int, float]:
        """
        Get mapping from column ID to value.

        Returns:
            Dict mapping column_id to value for columns with positive value
        """
        return {
            col.column_id: col.value
            for col in self.columns
            if col.column_id is not None and col.value is not None
        }

    def get_convergence_history(self) -> List[float]:
        """
        Get objective values over iterations.

        Returns:
            List of objective values, one per iteration
        """
        return [it.master_objective for it in self.iteration_history]

    def summary(self) -> str:
        """
        Return a human-readable summary.

        Returns:
            Summary string
        """
        lines = [
            "Column Generation Solution:",
            f"  Status: {self.status.name}",
        ]

        if self.objective_value is not None:
            lines.append(f"  Objective: {self.objective_value:.6f}")

        if self.lp_objective is not None and self.lp_objective != self.objective_value:
            lines.append(f"  LP Objective: {self.lp_objective:.6f}")

        if self.ip_objective is not None:
            lines.append(f"  IP Objective: {self.ip_objective:.6f}")

        if self.gap is not None:
            lines.append(f"  Gap: {self.gap:.4%}")

        if self.lower_bound is not None:
            lines.append(f"  Lower bound: {self.lower_bound:.6f}")

        lines.extend([
            "",
            f"  Iterations: {self.iterations}",
            f"  Total columns: {self.total_columns}",
            f"  Active columns: {self.num_active_columns}",
            "",
            f"  Total time: {self.total_time:.3f}s",
            f"  Master time: {self.master_time:.3f}s ({100*self.master_time/max(self.total_time, 1e-6):.1f}%)",
            f"  Pricing time: {self.pricing_time:.3f}s ({100*self.pricing_time/max(self.total_time, 1e-6):.1f}%)",
        ])

        if self.is_integer:
            lines.append("\n  Solution is integer")
        else:
            lines.append("\n  Solution is fractional")

        return "\n".join(lines)

    def __repr__(self) -> str:
        obj_str = f", obj={self.objective_value:.4f}" if self.objective_value else ""
        return f"CGSolution({self.status.name}{obj_str}, iter={self.iterations})"
