# Quick Start Guide - OpenCG

**Goal**: Get from zero to solving your first optimization problem in 30 minutes.

> **Looking for OpenBP?** See the [OpenBP Quick Start Guide](https://github.com/miladbarooni/openbp/blob/main/QUICKSTART.md) for Branch-and-Price.

---

## Prerequisites (5 minutes)

### Required
- **Python 3.9+** (check: `python --version`)
- **C++ compiler** with C++17 support
  - Linux: `gcc` or `clang`
  - macOS: Xcode Command Line Tools
  - Windows: Visual Studio 2019+ or MinGW-w64
- **CMake 3.15+** (check: `cmake --version`)

### Install System Dependencies

**macOS**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake (via Homebrew)
brew install cmake
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev
```

**Windows** (PowerShell):
```powershell
# Install via Chocolatey
choco install cmake visualstudio2022buildtools
```

---

## Installation (10 minutes)

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/miladbarooni/opencg.git
cd opencg

# Create and activate conda environment
conda env create -f environment.yml
conda activate opencg

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "from opencg._core import HAS_CPP_BACKEND; print(f'C++ backend: {HAS_CPP_BACKEND}')"
```

Expected output: `C++ backend: True` ✅

### Option 2: Using pip + venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clone and install
git clone https://github.com/miladbarooni/opencg.git
cd opencg
pip install --upgrade pip
pip install -e ".[dev]"
```

### Install OpenBP (Optional)

```bash
# From the same parent directory as opencg
git clone https://github.com/miladbarooni/openbp.git
cd openbp
pip install -e ".[dev]"
```

---

## Your First Solve: Cutting Stock Problem (15 minutes)

### Problem Description

You have rolls of paper 100 cm wide. You need to cut them into smaller pieces:
- 45 cm pieces (need 97)
- 36 cm pieces (need 610)
- 31 cm pieces (need 395)
- 14 cm pieces (need 211)

**Goal**: Minimize the number of rolls used.

### Solution (Copy & Run)

Create a file `my_first_solve.py`:

```python
from opencg.applications.cutting_stock import CuttingStockInstance, solve_cutting_stock

# Define the problem
instance = CuttingStockInstance(
    roll_width=100,
    item_sizes=[45, 36, 31, 14],
    item_demands=[97, 610, 395, 211],
    name="quickstart_example"
)

# Solve it!
print("Solving cutting stock problem...")
solution = solve_cutting_stock(
    instance,
    max_iterations=20,
    verbose=True,
)

# Print results
print("\n" + "="*60)
print("SOLUTION")
print("="*60)
print(f"Rolls used: {solution.objective_value}")
print(f"Iterations: {solution.iterations}")
print(f"Solve time: {solution.solve_time:.2f}s")
print(f"Status: {solution.status}")

# Show cutting patterns
print("\nCutting Patterns:")
for col_id, quantity in solution.column_values.items():
    if quantity > 0.5:  # Only show used patterns
        column = solution.columns[col_id]
        pattern = [0] * len(instance.item_sizes)

        # Count how many of each item in this pattern
        for arc_idx in column.arc_indices:
            if arc_idx < len(instance.item_sizes):  # Item arcs
                pattern[arc_idx] += 1

        print(f"  Pattern {col_id}: use {int(quantity)} times")
        for i, count in enumerate(pattern):
            if count > 0:
                print(f"    - {count}x {instance.item_sizes[i]}cm")
```

### Run It

```bash
python my_first_solve.py
```

### Expected Output

```
Solving cutting stock problem...
Iteration   0: LP=1313.00, Columns=  4
Iteration   1: LP= 477.71, Columns= 23
Iteration   2: LP= 453.02, Columns= 57
...
Iteration   8: LP= 452.00, Columns=201
Converged at iteration 8

============================================================
SOLUTION
============================================================
Rolls used: 452
Iterations: 8
Solve time: 0.15s
Status: PricingStatus.NO_COLUMNS

Cutting Patterns:
  Pattern 3: use 97 times
    - 1x 45cm
    - 1x 36cm
    - 1x 14cm
  Pattern 7: use 355 times
    - 2x 36cm
    - 1x 14cm
  ...
```

### What Just Happened?

1. **Column Generation** iteratively generated cutting patterns
2. Started with 4 simple patterns (one item per roll)
3. Each iteration found better patterns (combinations that use less waste)
4. Converged when no better patterns exist
5. Solved the integer program to get whole number of rolls

**Key Insight**: Column generation only generated ~200 patterns instead of exploring millions of possibilities!

---

## Next Steps

### Try More Examples

**Vehicle Routing**:
```python
from opencg.applications.vehicle_routing import VRPTWInstance, solve_vrptw

# Load Solomon benchmark instance
instance = VRPTWInstance.from_solomon_file("data/solomon/C101.txt")
solution = solve_vrptw(instance, max_iterations=50)

print(f"Routes: {solution.objective_value}")
print(f"Time: {solution.solve_time:.2f}s")
```

**Crew Pairing**:
```python
from opencg.parsers import KasirzadehParser
from opencg.applications.crew_pairing import solve_crew_pairing

# Parse instance
parser = KasirzadehParser()
problem = parser.parse("data/kasirzadeh/instance1")

# Solve
solution = solve_crew_pairing(
    problem,
    max_iterations=20,
    use_fast_pricing=True,
    num_threads=4,  # Parallel pricing!
)

print(f"Pairings: {solution.objective_value}")
print(f"Coverage: {solution.coverage_pct:.1f}%")
```

### Explore Jupyter Notebooks

```bash
# Install notebook dependencies
pip install -e ".[notebooks]"

# Start Jupyter
jupyter notebook examples/notebooks/

# Open:
# - 01_cutting_stock.ipynb - Detailed cutting stock tutorial
# - 02_vehicle_routing.ipynb - VRP with time windows
# - 03_crew_pairing.ipynb - Airline crew scheduling
# - 04_parallel_pricing.ipynb - Performance optimization
```

### Customize for Your Problem

See the [User Guide](docs/user_guide.md) for:
- Defining custom networks
- Adding resource constraints
- Implementing custom pricing strategies
- Performance tuning

### Use Branch-and-Price (OpenBP)

When direct column generation doesn't give integral solutions:

```python
from openbp import BranchAndPrice, BestFirstSelection
from openbp.branching import RyanFosterBranching

# Configure B&P solver
solver = BranchAndPrice(
    problem,
    branching_strategy=RyanFosterBranching(),
    node_selection=BestFirstSelection(),
)

# Solve with optimality guarantee
solution = solver.solve(time_limit=3600)

print(f"Optimal: {solution.objective}")
print(f"Gap: {solution.gap * 100:.2f}%")
print(f"Nodes: {solution.nodes_explored}")
```

---

## Troubleshooting

### C++ Backend Not Available

**Symptom**: `HAS_CPP_BACKEND = False`

**Fix**:
```bash
# Check CMake is installed
cmake --version

# Reinstall with verbose output
pip uninstall opencg
pip install -e . -v

# Look for compilation errors in output
```

**Common Issues**:
- **macOS**: Missing Xcode Command Line Tools → `xcode-select --install`
- **Ubuntu**: Missing build tools → `sudo apt-get install build-essential cmake`
- **Windows**: Missing Visual Studio → Install VS 2019+ Build Tools

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'opencg'`

**Fix**:
```bash
# Check you're in the right environment
which python  # Should show venv or conda path

# Reinstall in editable mode
pip install -e .
```

### Tests Failing

```bash
# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/unit/test_pricing.py::test_labeling_basic -v

# Check C++ backend
python -c "from opencg._core import HAS_CPP_BACKEND; assert HAS_CPP_BACKEND"
```

### Performance Issues

**Slow Solve**:
1. Verify C++ backend is active: `HAS_CPP_BACKEND = True`
2. Use parallel pricing: `use_fast_pricing=True, num_threads=4`
3. Reduce convergence tolerance: `reduced_cost_threshold=-1e-4`
4. Limit iterations: `max_iterations=20`

**High Memory Usage**:
1. Limit columns per iteration: `max_columns=200`
2. Use column pool size limit
3. Periodic column cleanup

---

## Getting Help

- **Documentation**: https://opencg.readthedocs.io
- **Examples**: `opencg/examples/`
- **Issues**: https://github.com/miladbarooni/opencg/issues
- **Discussions**: https://github.com/miladbarooni/opencg/discussions

---

## What's Next?

### Learn More
- Read the [User Guide](docs/user_guide.md) for detailed explanations
- Study [Architecture](docs/ARCHITECTURE.md) to understand internals
- Explore [API Reference](https://opencg.readthedocs.io/api) for all classes

### Contribute
- Check [Contributing Guide](CONTRIBUTING.md)
- Look for ["good first issue"](https://github.com/miladbarooni/opencg/labels/good%20first%20issue) tags
- Join our community discussions

### Research
- Implement custom branching strategies
- Try novel pricing algorithms
- Publish your results using OpenCG!

---

**Congratulations!** 🎉 You've solved your first optimization problem with OpenCG.

**Total time**: ~30 minutes
- Installation: 10 min
- First solve: 5 min
- Understanding: 15 min

**Next**: Try the Jupyter notebooks or customize for your own problem!
