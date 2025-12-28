# OpenCG Documentation

Welcome to the OpenCG documentation. This directory contains guides for using and extending the framework.

## Quick Links

- [Quick Start Guide](../QUICKSTART.md) - Get started in 30 minutes
- [Architecture Overview](ARCHITECTURE.md) - Understanding the framework design

## User Guides

| Guide | Description |
|-------|-------------|
| [Building Networks](building_networks.md) | How to construct time-space networks for your problem |

## Extension Guides

These guides explain how to customize and extend OpenCG for your specific needs:

| Guide | Description |
|-------|-------------|
| [Custom Resources](custom_resources.md) | Create new resource constraints (time, capacity, skills, etc.) |
| [Custom Pricing](custom_pricing.md) | Implement your own SPPRC pricing algorithms |
| [Custom Master Problem](custom_master.md) | Use different LP/MIP solvers (Gurobi, CPLEX, OR-Tools) |
| [Custom Applications](custom_application.md) | Model new optimization problems |

## Documentation Structure

```
docs/
├── README.md              # This file
├── ARCHITECTURE.md        # Framework design and internals
├── building_networks.md   # How to build networks (USER GUIDE)
├── custom_resources.md    # Resource constraint extension guide
├── custom_pricing.md      # Pricing algorithm customization
├── custom_master.md       # Master problem solver customization
└── custom_application.md  # New problem modeling guide
```

## Who Should Read What?

### Practitioners (Using Built-in Solvers)

1. Start with [Quick Start Guide](../QUICKSTART.md)
2. Read [Building Networks](building_networks.md) to model your problem
3. Explore example notebooks in `examples/notebooks/`
4. Read [Architecture](ARCHITECTURE.md) for deeper understanding

### Researchers (Extending the Framework)

1. Start with [Architecture](ARCHITECTURE.md)
2. Read relevant extension guides:
   - [Custom Resources](custom_resources.md) for new constraints
   - [Custom Pricing](custom_pricing.md) for algorithm development
   - [Custom Applications](custom_application.md) for new problem types

### Developers (Integrating with Other Tools)

1. Start with [Architecture](ARCHITECTURE.md)
2. Read [Custom Master Problem](custom_master.md) for solver integration
3. Check API reference (docstrings in source code)

## Examples by Use Case

### "I want to solve cutting stock"
→ See `examples/notebooks/01_cutting_stock.ipynb`

### "I want to solve vehicle routing"
→ See `examples/notebooks/02_vehicle_routing.ipynb`

### "I want to solve crew pairing"
→ See `examples/notebooks/03_crew_pairing.ipynb`

### "I want to use parallel pricing"
→ See `examples/notebooks/04_parallel_pricing.ipynb`

### "I want to model my own problem"
→ Read [Building Networks](building_networks.md)

### "I want to add a new constraint type"
→ Read [Custom Resources](custom_resources.md)

### "I want to use Gurobi instead of HiGHS"
→ Read [Custom Master Problem](custom_master.md)

### "I want to implement my own pricing algorithm"
→ Read [Custom Pricing](custom_pricing.md)

### "I want to model a scheduling problem"
→ Read [Custom Applications](custom_application.md)

## API Reference

The primary API is documented via docstrings. Key modules:

- `opencg.core` - Data structures (Network, Node, Arc, Column, Resource)
- `opencg.master` - Master problem solvers (HiGHS, CPLEX)
- `opencg.pricing` - Pricing algorithms (labeling, per-source)
- `opencg.solver` - Column generation coordinator
- `opencg.applications` - Problem-specific solvers

## Contributing

See [Contributing Guide](../CONTRIBUTING.md) for how to contribute documentation improvements.
