# OpenCG Examples

This folder contains examples demonstrating how to use the OpenCG column generation framework.

## Structure

```
examples/
├── notebooks/           # Jupyter notebooks with interactive tutorials
│   ├── 01_cutting_stock.ipynb
│   ├── 02_vehicle_routing.ipynb
│   └── 03_crew_pairing.ipynb
└── scripts/             # Python scripts for specific use cases
    └── crew_pairing/
        ├── load_kasirzadeh.py
        ├── solve_kasirzadeh.py
        └── solve_kasirzadeh_coverage.py
```

## Notebooks

The notebooks provide interactive tutorials with visualizations:

### 1. Cutting Stock Problem (`01_cutting_stock.ipynb`)

Learn the basics of column generation with the classic cutting stock problem:
- Problem formulation and column generation approach
- Creating instances programmatically
- Loading BPPLIB benchmark instances
- Visualizing cutting patterns
- Understanding the pricing problem (bounded knapsack)

### 2. Vehicle Routing Problem (`02_vehicle_routing.ipynb`)

Solve capacitated vehicle routing with resource constraints:
- CVRP problem structure (depot, customers, demands, capacity)
- Time-space network construction
- SPPRC pricing with capacity resource
- Visualizing routes on 2D maps
- Comparing greedy heuristic vs column generation

### 3. Crew Pairing Problem (`03_crew_pairing.ipynb`)

The most complex example - airline crew scheduling:
- Understanding pairings, duty periods, and rests
- Time-space network for crew operations
- Multiple resources (duty time, rest, pairing duration)
- Home base constraints
- Per-source pricing strategies for better coverage

## Running the Notebooks

1. Activate the conda environment:
   ```bash
   conda activate opencg
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook examples/notebooks/
   ```

3. Open a notebook and run the cells interactively.

## Scripts

The scripts folder contains standalone Python scripts:

### Crew Pairing Scripts

- `load_kasirzadeh.py` - Load and inspect Kasirzadeh benchmark instances
- `solve_kasirzadeh.py` - Basic solver for crew pairing
- `solve_kasirzadeh_coverage.py` - Advanced solver with coverage optimization

Run from the project root:
```bash
python examples/scripts/crew_pairing/solve_kasirzadeh.py
```

## Prerequisites

All examples require the OpenCG package and its dependencies:

```bash
pip install highspy numpy matplotlib
```

For crew pairing examples, you'll need the Kasirzadeh dataset in `data/kasirzadeh/`.

For cutting stock benchmarks, download BPPLIB instances to `data/bpplib/`.
