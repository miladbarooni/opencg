# Custom Master Problem Guide

This guide explains how to implement custom master problem solvers for column generation, allowing you to use different LP/MIP solvers like Gurobi, CPLEX, or custom formulations.

## Overview

The **master problem** in column generation is typically a set covering or set partitioning LP:

```
minimize    sum_j c_j * λ_j
subject to  sum_j a_ij * λ_j >= 1   for all items i  (set covering)
            λ_j >= 0
```

The dual values (π_i) from the LP relaxation are used in the pricing subproblem to find columns with negative reduced cost.

---

## Using Built-in Master Problem Solvers

Before creating custom solvers, understand the built-in options:

```python
from opencg.master.highs import HiGHSMasterProblem

# Create master problem (HiGHS is the default)
master = HiGHSMasterProblem(
    problem,
    time_limit=300.0,  # 5 minute limit
    verbosity=0        # Silent (0), normal (1), verbose (2)
)

# Add columns
for column in initial_columns:
    master.add_column(column)

# Solve LP relaxation
lp_solution = master.solve_lp()
print(f"LP Objective: {lp_solution.objective_value}")

# Get dual values for pricing
duals = master.get_dual_values()
for item_id, pi in duals.items():
    print(f"  π[{item_id}] = {pi:.4f}")

# After CG converges, solve IP
ip_solution = master.solve_ip()
print(f"IP Objective: {ip_solution.objective_value}")

# Branching support
master.fix_column(column_id, value=1.0)  # Fix to 1
master.fix_column(column_id, value=0.0)  # Fix to 0
master.reset_column_bounds(column_id)    # Reset to [0, 1]

# Dual stabilization
master.enable_stabilization(method='boxstep', delta=10.0)
master.disable_stabilization()
```

---

## The MasterProblem Interface

All master problem solvers inherit from `MasterProblem`:

```python
from opencg.master.base import MasterProblem, StabilizationConfig
from opencg.master.solution import MasterSolution

class MasterProblem(ABC):
    def __init__(self, problem: Problem):
        self._problem = problem
        self._columns = []
        self._build_model()

    # Required to implement
    @abstractmethod
    def _build_model(self) -> None:
        """Create solver model and covering constraints."""
        pass

    @abstractmethod
    def _add_column_impl(self, column: Column) -> int:
        """Add a column to the solver model."""
        pass

    @abstractmethod
    def _solve_lp_impl(self) -> MasterSolution:
        """Solve LP relaxation."""
        pass

    @abstractmethod
    def _solve_ip_impl(self) -> MasterSolution:
        """Solve as integer program."""
        pass

    @abstractmethod
    def _get_dual_values_impl(self) -> dict[int, float]:
        """Extract dual values from solver."""
        pass

    # Public API (don't override)
    def add_column(self, column: Column) -> int: ...
    def solve_lp(self) -> MasterSolution: ...
    def solve_ip(self) -> MasterSolution: ...
    def get_dual_values(self) -> dict[int, float]: ...
```

## Built-in Implementations

### HiGHSMasterProblem (Default)

Uses the open-source HiGHS solver.

```python
from opencg.master.highs import HiGHSMasterProblem

master = HiGHSMasterProblem(problem)
```

### CPLEXMasterProblem

Uses IBM CPLEX (requires license).

```python
from opencg.master.cplex import CPLEXMasterProblem

master = CPLEXMasterProblem(problem)
```

---

## Creating Custom Master Problem Solvers

### Example 0: Simplified HiGHS (Understanding the Built-in)

Here's a minimal rewrite of the HiGHS master problem to help you understand how it works:

```python
from opencg.master.base import MasterProblem
from opencg.master.solution import MasterSolution, SolutionStatus
from opencg.core.problem import CoverType
import highspy

class SimpleHiGHSMaster(MasterProblem):
    """
    A simplified HiGHS master problem for educational purposes.

    This is a minimal version showing the key methods you need to implement.
    """

    def __init__(self, problem):
        self._highs = None
        self._col_to_idx = {}  # column_id -> solver column index
        super().__init__(problem)  # Calls _build_model()

    def _build_model(self):
        """Create HiGHS model with covering constraints."""
        self._highs = highspy.Highs()
        self._highs.setOptionValue('output_flag', False)

        # Add one constraint per item to cover
        for constraint in self._problem.cover_constraints:
            if self._problem.cover_type == CoverType.SET_PARTITIONING:
                # = 1 constraint
                self._highs.addRow(1.0, 1.0, 0, [], [])
            else:
                # >= 1 constraint
                self._highs.addRow(1.0, highspy.kHighsInf, 0, [], [])

    def _add_column_impl(self, column):
        """Add a column variable to the model."""
        # Find which constraints this column covers
        indices = []
        values = []

        for item_id in column.covered_items:
            if item_id in self._item_to_constraint_idx:
                indices.append(self._item_to_constraint_idx[item_id])
                values.append(1.0)  # Coefficient = 1

        # Add variable: cost, lower=0, upper=inf, nonzeros
        self._highs.addCol(column.cost, 0.0, highspy.kHighsInf,
                          len(indices), indices, values)

        # Track the column
        solver_idx = self._highs.getNumCol() - 1
        self._col_to_idx[column.column_id] = solver_idx
        return solver_idx

    def _solve_lp_impl(self):
        """Solve LP relaxation."""
        self._highs.run()

        # Map status
        status_map = {
            highspy.HighsModelStatus.kOptimal: SolutionStatus.OPTIMAL,
            highspy.HighsModelStatus.kInfeasible: SolutionStatus.INFEASIBLE,
        }
        status = status_map.get(self._highs.getModelStatus(), SolutionStatus.ERROR)

        solution = MasterSolution(status=status)

        if status == SolutionStatus.OPTIMAL:
            solution.objective_value = self._highs.getInfo().objective_function_value

            # Get primal values
            sol = self._highs.getSolution()
            for col_id, idx in self._col_to_idx.items():
                if sol.col_value[idx] > 1e-6:
                    solution.column_values[col_id] = sol.col_value[idx]

        return solution

    def _solve_ip_impl(self):
        """Solve as integer program."""
        # Set all columns to binary
        for i in range(self._highs.getNumCol()):
            self._highs.changeColIntegrality(i, highspy.HighsVarType.kInteger)
            self._highs.changeColBounds(i, 0.0, 1.0)

        self._highs.run()

        status = SolutionStatus.OPTIMAL if self._highs.getModelStatus() == highspy.HighsModelStatus.kOptimal else SolutionStatus.ERROR
        solution = MasterSolution(status=status)

        if status == SolutionStatus.OPTIMAL:
            solution.objective_value = self._highs.getInfo().objective_function_value
            sol = self._highs.getSolution()
            for col_id, idx in self._col_to_idx.items():
                if sol.col_value[idx] > 0.5:
                    solution.column_values[col_id] = sol.col_value[idx]

        return solution

    def _get_dual_values_impl(self):
        """Get dual values from LP solution."""
        duals = {}
        sol = self._highs.getSolution()

        for constraint in self._problem.cover_constraints:
            idx = self._item_to_constraint_idx[constraint.item_id]
            duals[constraint.item_id] = sol.row_dual[idx]

        return duals
```

**Usage:**

```python
# Use exactly like the built-in
master = SimpleHiGHSMaster(problem)

for col in columns:
    master.add_column(col)

solution = master.solve_lp()
duals = master.get_dual_values()
```

---

### Example 1: Gurobi Master Problem

```python
from opencg.master.base import MasterProblem
from opencg.master.solution import MasterSolution, SolutionStatus
from opencg.core.column import Column

class GurobiMasterProblem(MasterProblem):
    """
    Master problem using Gurobi optimizer.
    """

    def __init__(self, problem):
        self._model = None
        self._vars = []  # Column variables
        self._constrs = []  # Covering constraints
        super().__init__(problem)

    def _build_model(self) -> None:
        """Create Gurobi model with covering constraints."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi not installed: pip install gurobipy")

        self._model = gp.Model("master")
        self._model.setParam("OutputFlag", 0)  # Suppress output

        # Create covering constraints (initially empty)
        for constraint in self._problem.cover_constraints:
            item_id = constraint.item_id
            sense = GRB.EQUAL if self._problem.cover_type.name == "PARTITIONING" else GRB.GREATER_EQUAL

            constr = self._model.addConstr(
                0 >= 1 if sense == GRB.GREATER_EQUAL else 0 == 1,
                name=f"cover_{item_id}"
            )
            self._constrs.append(constr)

        # Store item_id to constraint index mapping
        self._item_to_constr = {
            c.item_id: i for i, c in enumerate(self._problem.cover_constraints)
        }

        self._model.update()

    def _add_column_impl(self, column: Column) -> int:
        """Add a column variable to the model."""
        import gurobipy as gp
        from gurobipy import GRB

        col_idx = len(self._vars)

        # Create Gurobi column
        gp_col = gp.Column()

        # Add coefficients for covered items
        for item_id in column.covered_items:
            if item_id in self._item_to_constr:
                constr_idx = self._item_to_constr[item_id]
                gp_col.addTerms(1.0, self._constrs[constr_idx])

        # Add variable
        var = self._model.addVar(
            lb=0.0,
            ub=1.0,
            obj=column.cost,
            vtype=GRB.CONTINUOUS,
            column=gp_col,
            name=f"col_{column.column_id}"
        )
        self._vars.append(var)
        self._model.update()

        return col_idx

    def _solve_lp_impl(self) -> MasterSolution:
        """Solve LP relaxation."""
        from gurobipy import GRB

        # Ensure continuous variables
        for var in self._vars:
            var.vtype = GRB.CONTINUOUS

        self._model.optimize()

        status = self._convert_status(self._model.status)

        if status != SolutionStatus.OPTIMAL:
            return MasterSolution(status=status)

        # Extract solution
        objective = self._model.objVal
        column_values = {
            self._columns[i].column_id: var.x
            for i, var in enumerate(self._vars)
        }

        return MasterSolution(
            status=status,
            objective=objective,
            column_values=column_values,
            columns={c.column_id: c for c in self._columns}
        )

    def _solve_ip_impl(self) -> MasterSolution:
        """Solve as integer program."""
        from gurobipy import GRB

        # Set binary variables
        for var in self._vars:
            var.vtype = GRB.BINARY

        self._model.optimize()

        status = self._convert_status(self._model.status)

        if status not in [SolutionStatus.OPTIMAL, SolutionStatus.FEASIBLE]:
            return MasterSolution(status=status)

        objective = self._model.objVal
        column_values = {
            self._columns[i].column_id: var.x
            for i, var in enumerate(self._vars)
        }

        return MasterSolution(
            status=status,
            objective=objective,
            column_values=column_values,
            columns={c.column_id: c for c in self._columns}
        )

    def _get_dual_values_impl(self) -> dict[int, float]:
        """Extract dual values from constraints."""
        duals = {}

        for constraint in self._problem.cover_constraints:
            item_id = constraint.item_id
            constr_idx = self._item_to_constr[item_id]
            duals[item_id] = self._constrs[constr_idx].Pi

        return duals

    def _convert_status(self, gurobi_status) -> SolutionStatus:
        """Convert Gurobi status to OpenCG status."""
        from gurobipy import GRB

        mapping = {
            GRB.OPTIMAL: SolutionStatus.OPTIMAL,
            GRB.INFEASIBLE: SolutionStatus.INFEASIBLE,
            GRB.UNBOUNDED: SolutionStatus.UNBOUNDED,
            GRB.INF_OR_UNBD: SolutionStatus.INFEASIBLE,
            GRB.TIME_LIMIT: SolutionStatus.TIME_LIMIT,
            GRB.SUBOPTIMAL: SolutionStatus.FEASIBLE,
        }
        return mapping.get(gurobi_status, SolutionStatus.ERROR)

    # Optional: Column bounds for branching
    def _set_column_bounds(self, column_id: int, lower: float, upper: float) -> None:
        idx = self._column_id_to_index.get(column_id)
        if idx is not None:
            self._vars[idx].lb = lower
            self._vars[idx].ub = upper
            self._model.update()
```

---

### Example 2: OR-Tools Master Problem

Using Google OR-Tools linear solver.

```python
from opencg.master.base import MasterProblem
from opencg.master.solution import MasterSolution, SolutionStatus

class ORToolsMasterProblem(MasterProblem):
    """
    Master problem using Google OR-Tools.
    """

    def __init__(self, problem, solver_id: str = "GLOP"):
        """
        Args:
            solver_id: OR-Tools solver identifier
                       "GLOP" for LP, "CBC" or "SCIP" for MIP
        """
        self._solver_id = solver_id
        self._solver = None
        self._vars = []
        self._constrs = []
        super().__init__(problem)

    def _build_model(self) -> None:
        from ortools.linear_solver import pywraplp

        self._solver = pywraplp.Solver.CreateSolver(self._solver_id)
        if not self._solver:
            raise RuntimeError(f"Could not create solver: {self._solver_id}")

        # Create covering constraints
        for constraint in self._problem.cover_constraints:
            item_id = constraint.item_id

            if self._problem.cover_type.name == "PARTITIONING":
                constr = self._solver.Constraint(1.0, 1.0, f"cover_{item_id}")
            else:  # COVERING
                constr = self._solver.Constraint(1.0, self._solver.infinity(), f"cover_{item_id}")

            self._constrs.append(constr)

        self._item_to_constr = {
            c.item_id: i for i, c in enumerate(self._problem.cover_constraints)
        }

    def _add_column_impl(self, column) -> int:
        col_idx = len(self._vars)

        var = self._solver.NumVar(0.0, 1.0, f"col_{column.column_id}")
        self._vars.append(var)

        # Set objective coefficient
        self._solver.Objective().SetCoefficient(var, column.cost)

        # Add to covering constraints
        for item_id in column.covered_items:
            if item_id in self._item_to_constr:
                constr_idx = self._item_to_constr[item_id]
                self._constrs[constr_idx].SetCoefficient(var, 1.0)

        return col_idx

    def _solve_lp_impl(self) -> MasterSolution:
        # Ensure continuous
        for var in self._vars:
            var.SetBounds(0.0, 1.0)

        self._solver.Objective().SetMinimization()
        status = self._solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            return MasterSolution(status=SolutionStatus.INFEASIBLE)

        objective = self._solver.Objective().Value()
        column_values = {
            self._columns[i].column_id: var.solution_value()
            for i, var in enumerate(self._vars)
        }

        return MasterSolution(
            status=SolutionStatus.OPTIMAL,
            objective=objective,
            column_values=column_values,
            columns={c.column_id: c for c in self._columns}
        )

    def _solve_ip_impl(self) -> MasterSolution:
        # Switch to MIP solver if needed
        if self._solver_id == "GLOP":
            # GLOP is LP only, need to recreate with MIP solver
            self._rebuild_as_mip()

        # Set integer variables
        for var in self._vars:
            var.SetInteger(True)

        status = self._solver.Solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return MasterSolution(status=SolutionStatus.INFEASIBLE)

        objective = self._solver.Objective().Value()
        column_values = {
            self._columns[i].column_id: var.solution_value()
            for i, var in enumerate(self._vars)
        }

        return MasterSolution(
            status=SolutionStatus.OPTIMAL,
            objective=objective,
            column_values=column_values,
            columns={c.column_id: c for c in self._columns}
        )

    def _get_dual_values_impl(self) -> dict[int, float]:
        duals = {}
        for constraint in self._problem.cover_constraints:
            item_id = constraint.item_id
            constr_idx = self._item_to_constr[item_id]
            duals[item_id] = self._constrs[constr_idx].dual_value()
        return duals

    def _rebuild_as_mip(self):
        """Rebuild model with MIP solver."""
        from ortools.linear_solver import pywraplp

        old_solver = self._solver
        self._solver = pywraplp.Solver.CreateSolver("CBC")

        # Rebuild constraints
        self._constrs = []
        for constraint in self._problem.cover_constraints:
            item_id = constraint.item_id
            if self._problem.cover_type.name == "PARTITIONING":
                constr = self._solver.Constraint(1.0, 1.0, f"cover_{item_id}")
            else:
                constr = self._solver.Constraint(1.0, self._solver.infinity(), f"cover_{item_id}")
            self._constrs.append(constr)

        # Rebuild variables
        new_vars = []
        for i, col in enumerate(self._columns):
            var = self._solver.IntVar(0, 1, f"col_{col.column_id}")
            new_vars.append(var)
            self._solver.Objective().SetCoefficient(var, col.cost)

            for item_id in col.covered_items:
                if item_id in self._item_to_constr:
                    constr_idx = self._item_to_constr[item_id]
                    self._constrs[constr_idx].SetCoefficient(var, 1.0)

        self._vars = new_vars
```

---

### Example 3: Custom Formulation (Generalized Covering)

For problems where columns can cover items with different coefficients (not just 0/1).

```python
from opencg.master.highs import HiGHSMasterProblem

class GeneralizedCoveringMaster(HiGHSMasterProblem):
    """
    Master problem with generalized covering constraints.

    Allows coefficients a_ij != 1, e.g., for bin packing with
    multiple copies of items per pattern.
    """

    def _get_coefficient_matrix_entry(self, column, item_id: int) -> float:
        """
        Get coefficient for column covering item.

        For cutting stock: coefficient = number of times item appears in pattern.
        """
        if not column.covers_item(item_id):
            return 0.0

        # Get count from column attributes
        pattern = column.get_attribute("pattern", None)
        if pattern is not None:
            return float(pattern.get(item_id, 0))

        # Default: 1.0
        return 1.0

    def _add_column_impl(self, column) -> int:
        """Override to use generalized coefficients."""
        # ... similar to base implementation but uses
        # self._get_coefficient_matrix_entry() for coefficients
        pass
```

---

## Advanced Features

### Dual Stabilization

Prevent dual oscillation with built-in stabilization:

```python
master = HiGHSMasterProblem(problem)

# Enable boxstep stabilization
master.enable_stabilization(method='boxstep', delta=10.0, shrink=0.5)

# Or smoothing
master.enable_stabilization(method='smoothing', alpha=0.5)

# During CG loop
solution = master.solve_lp()
duals = master.get_dual_values()  # Returns stabilized duals

# If no improving columns found, shrink stabilization region
if not pricing_solution.columns:
    master.shrink_stabilization()

# When improving columns found, update center
if pricing_solution.columns:
    master.update_stabilization_center(master.get_raw_dual_values())
```

### Warm Starting

Use basis from previous solve:

```python
# After LP solve
basis = master.get_basis()

# For next solve (after adding columns)
master.set_basis(basis)
```

### Branching Support

For Branch-and-Price:

```python
# Fix column to 0 or 1
master.fix_column(column_id, value=1.0)  # Must use this column
master.fix_column(column_id, value=0.0)  # Cannot use this column

# Set bounds
master.set_column_bounds(column_id, lower=0.0, upper=0.5)

# Reset
master.reset_column_bounds(column_id)
```

### Cutting Planes

Add valid inequalities:

```python
# Add cut: x_1 + x_2 + x_3 <= 2
master.add_cut(
    coefficients={1: 1.0, 2: 1.0, 3: 1.0},
    sense='<=',
    rhs=2.0
)
```

---

## Using Custom Master in Column Generation

```python
from opencg.solver import ColumnGeneration, CGConfig

# Create custom master
master = GurobiMasterProblem(problem)

# Use in column generation
cg = ColumnGeneration(
    problem,
    master=master,  # Pass custom master
    config=CGConfig(max_iterations=50)
)

solution = cg.solve()
```

---

## Best Practices

### 1. Handle Infeasibility

The initial model (no columns) is infeasible. Add artificial columns or big-M variables:

```python
def _build_model(self):
    # ... create constraints ...

    # Add artificial variables with high cost
    for i, constraint in enumerate(self._problem.cover_constraints):
        art_var = self._model.addVar(
            obj=1e6,  # Big penalty
            name=f"artificial_{i}"
        )
        constraint.addTerm(art_var, 1.0)
```

### 2. Efficient Column Addition

Batch column additions when possible:

```python
def add_columns_batch(self, columns: list[Column]):
    """Add multiple columns efficiently."""
    # Some solvers support batch operations
    for col in columns:
        self.add_column(col)
    self._model.update()  # Single update
```

### 3. Track Solver Time

```python
def _solve_lp_impl(self):
    import time
    start = time.time()

    self._model.optimize()

    solve_time = time.time() - start

    return MasterSolution(
        ...,
        solve_time=solve_time
    )
```

### 4. Memory Management

For large problems, consider column management:

```python
def cleanup_columns(self, threshold: float = 1e-6):
    """Remove columns with zero or near-zero values."""
    to_remove = []
    for col in self._columns:
        value = self._get_column_value(col.column_id)
        if value < threshold:
            to_remove.append(col.column_id)

    for col_id in to_remove:
        self.remove_column(col_id)
```

---

## Next Steps

- [Custom Resources](custom_resources.md) - Define new resource constraints
- [Custom Pricing](custom_pricing.md) - Implement custom SPPRC algorithms
- [Custom Applications](custom_application.md) - Model new problem types
