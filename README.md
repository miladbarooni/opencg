# OpenCG: Open-Source Column Generation Framework

A high-performance, extensible framework for solving optimization problems using Column Generation. Built with a Python frontend and C++ backend for optimal balance of usability and speed.

## Features

- **High-Performance C++ Backend**: SPPRC labeling algorithm with parallel execution
- **Pure Python API**: Easy to use, extend, and integrate
- **Multiple Problem Types**:
  - Cutting Stock / Bin Packing
  - Capacitated Vehicle Routing (CVRP)
  - Vehicle Routing with Time Windows (VRPTW)
  - Airline Crew Pairing
- **Extensible Resource System**: Accumulating, interval, state, and time window resources
- **Multiple LP Solvers**: HiGHS (default, open-source), CPLEX (optional)
- **Parallel Pricing**: Multi-threaded labeling for faster convergence

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Layer (User-facing API)                                 │
│  - Problem definition (Network, Resources, Constraints)         │
│  - Application solvers (Cutting Stock, VRP, Crew Pairing)       │
│  - Parsers (BPPLIB, Kasirzadeh, Solomon)                        │
│  - Column generation orchestration                               │
├─────────────────────────────────────────────────────────────────┤
│  C++ Core (Performance-critical) via pybind11                   │
│  - Network/Arc/Node data structures                             │
│  - SPPRC labeling with dominance pruning                        │
│  - Parallel labeling (std::async)                               │
│  - ng-path relaxation, time windows                             │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- C++ compiler with C++17 support
- CMake 3.15+

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/opencg.git
cd opencg

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate opencg

# Install in development mode
pip install -e .
```

### Verify Installation

```python
from opencg._core import HAS_CPP_BACKEND
print(f"C++ backend available: {HAS_CPP_BACKEND}")
```

## Quick Start

### Cutting Stock Problem

```python
from opencg.applications.cutting_stock import CuttingStockInstance, solve_cutting_stock

# Define instance
instance = CuttingStockInstance(
    roll_width=100,
    item_sizes=[45, 36, 31, 14],
    item_demands=[97, 610, 395, 211],
    name="example"
)

# Solve
solution = solve_cutting_stock(instance, max_iterations=100, verbose=True)
print(f"Rolls needed: {solution.num_rolls_ip}")
```

### Vehicle Routing (CVRP)

```python
from opencg.applications.vrp import CVRPInstance, solve_cvrp

instance = CVRPInstance(
    depot=(0, 0),
    customers=[(10, 0), (0, 10), (-10, 0), (0, -10)],
    demands=[20, 15, 25, 10],
    vehicle_capacity=50,
)

solution = solve_cvrp(instance)
print(f"Total distance: {solution.total_distance:.2f}")
for route in solution.routes:
    print(f"  Route: {route}")
```

### Vehicle Routing with Time Windows (VRPTW)

```python
from opencg.applications.vrp import VRPTWInstance, solve_vrptw

instance = VRPTWInstance(
    depot=(0, 0),
    customers=[(10, 0), (0, 10)],
    demands=[20, 15],
    time_windows=[(0, 50), (20, 80)],
    service_times=[10, 10],
    vehicle_capacity=50,
    depot_time_window=(0, 200),
)

solution = solve_vrptw(instance)
```

### Crew Pairing

```python
from opencg.parsers import KasirzadehParser
from opencg.applications.crew_pairing import solve_crew_pairing

# Load benchmark instance
parser = KasirzadehParser()
problem = parser.parse("data/kasirzadeh/instance1")

# Solve with per-source pricing (high coverage)
solution = solve_crew_pairing(problem, use_per_source=True, verbose=True)
print(f"Coverage: {solution.coverage:.1f}%")
```

## Parallel Pricing

For large problems, enable parallel pricing to speed up the labeling algorithm:

```python
from opencg.pricing.fast_per_source import FastPerSourcePricing
from opencg.pricing import PricingConfig

pricing = FastPerSourcePricing(
    problem,
    config=PricingConfig(max_columns=200),
    num_threads=8  # Use 8 threads (0 = auto-detect)
)
```

Benchmark results on Kasirzadeh instance1 (1013 flights):

| Threads | Pricing Time | Speedup |
|---------|-------------|---------|
| 1       | 0.97s       | 1.0x    |
| 4       | 0.37s       | 2.6x    |
| 8       | 0.23s       | 4.3x    |

## Examples

See the `examples/notebooks/` directory for detailed tutorials:

1. **01_cutting_stock.ipynb** - Cutting stock basics and BPPLIB benchmarks
2. **02_vehicle_routing.ipynb** - CVRP with capacity constraints
3. **03_crew_pairing.ipynb** - Airline crew scheduling with multiple resources
4. **04_parallel_pricing.ipynb** - Parallel pricing performance

## Extending OpenCG

### Custom Resource Type

```python
from opencg.core.resource import Resource

class MyResource(Resource):
    def extend(self, current_value, arc):
        return current_value + arc.get_consumption(self.name, 0.0)

    def is_feasible(self, value, node):
        return value <= self.max_value

    def dominates(self, value1, value2):
        return value1 <= value2
```

### Custom Pricing Algorithm

```python
from opencg.pricing.base import PricingProblem

class MyPricing(PricingProblem):
    def _solve_impl(self):
        # Your pricing algorithm
        columns = self._find_columns()
        return PricingSolution(columns=columns, ...)
```

## Project Structure

```
opencg/
├── opencg/                 # Python package
│   ├── core/              # Core data structures (Node, Arc, Network, Column)
│   ├── master/            # LP/MIP solvers (HiGHS, CPLEX)
│   ├── pricing/           # SPPRC algorithms (labeling, per-source)
│   ├── solver/            # Column generation coordinator
│   ├── parsers/           # File format readers
│   └── applications/      # Domain-specific solvers
│       ├── cutting_stock.py
│       ├── crew_pairing/
│       └── vrp/
├── src/                   # C++ source code
│   ├── core/             # C++ data structures
│   ├── pricing/          # C++ SPPRC implementation
│   └── bindings/         # pybind11 Python bindings
├── tests/                # Test suite
├── examples/             # Jupyter notebooks and scripts
└── data/                 # Benchmark instances
```

## Dependencies

- **numpy**: Array operations
- **highspy**: HiGHS LP solver (bundled)
- **pybind11**: C++/Python bindings (build only)
- **matplotlib** (optional): Visualization in notebooks

## Performance Notes

- The C++ backend provides 10-100x speedup over pure Python for labeling
- GIL is released during C++ solve(), enabling true parallel execution
- Per-source pricing achieves higher coverage than single-source for crew pairing
- Use `max_labels_per_node` for beam search (faster but heuristic)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use OpenCG in your research, please cite:

```bibtex
@software{opencg2024,
  title = {OpenCG: Open-Source Column Generation Framework},
  author = {Contributors},
  year = {2024},
  url = {https://github.com/yourusername/opencg}
}
```

## Acknowledgments

- Kasirzadeh benchmark instances from airline crew pairing research
- BPPLIB bin packing benchmark library
- HiGHS open-source LP solver
