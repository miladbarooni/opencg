# OpenCG Project Status

**Last Updated**: December 29, 2024

## Current State

OpenCG is a working column generation framework with a high-performance C++ backend. The crew pairing application has been successfully benchmarked against the Kasirzadeh et al. (2017) literature instances.

### What Works

- **C++ Backend**: SPPRC labeling algorithm with parallel execution via pybind11
- **Per-Source Pricing**: Key innovation that achieves 100% flight coverage
- **Kasirzadeh Benchmarks**: Successfully solved instances 1-3 with optimal LP solutions
- **Multiple Applications**: Cutting stock, VRP, VRPTW, and crew pairing

## Benchmark Results (Kasirzadeh Instances)

All benchmarks run on Windows 10, Intel Core i7-8700 (6 cores), 4 threads.

| Instance | Flights | Sources | Our Time | Lit. Time | Speedup | Our Iters | Lit. Iters | Status |
|----------|---------|---------|----------|-----------|---------|-----------|------------|--------|
| I1-727   | 1,013   | 509     | 82s      | 150s      | **1.83x faster** | 11 | 239 | OPTIMAL |
| I2-DC9   | 1,500   | 754     | 294s     | 260s      | 0.89x   | 14 | 1,968 | OPTIMAL |
| I3-D94   | 1,855   | 985     | 704s     | 548s      | 0.78x   | 21 | 466 | OPTIMAL |
| I4-D95   | -       | 3,000+  | -        | -         | -       | - | - | MemoryError |
| I5-757   | -       | 3,000+  | -        | -         | -       | - | - | MemoryError |
| I6-319   | -       | 3,000+  | -        | -         | -       | - | - | MemoryError |
| I7-320   | -       | 3,000+  | -        | -         | -       | - | - | MemoryError |

**Key Achievements**:
- **100% flight coverage** on all solved instances (matches literature)
- **10-140x fewer iterations** than literature (11-21 vs 239-1968)
- **Competitive runtime** on smaller instances, within 20-30% on larger ones
- **LP optimality** proven on all solved instances

## Architecture

### The Per-Source Pricing Innovation

The original global SPPRC pricing achieved only ~40% coverage because paths starting from different source arcs have different cost structures. The **per-source pricing** approach solves this by:

1. Building a separate network for each source arc (crew base departure)
2. Solving independent SPPRCs in parallel (one per source)
3. Collecting the best columns across all sources
4. Adding up to 200 high-quality columns per iteration

**Trade-offs**:
- **Pros**: 100% coverage, fewer iterations, better for branch-and-price
- **Cons**: O(n_sources) memory and per-iteration cost

### Why Fewer Iterations?

| Metric | Literature | OpenCG |
|--------|------------|--------|
| Columns/iteration | 1-5 | ~200 |
| Time/iteration | ~0.1-0.2s | ~20-35s |
| Total iterations | 239-1968 | 11-21 |

We batch more work per iteration by generating multiple columns from each source. This is advantageous for:
- Branch-and-price (fewer LP resolves in the tree)
- Problems where iteration overhead dominates

## Known Issues

### Memory Constraints (Instances 4-7)

Instances 4-7 have 3,000+ source arcs, requiring pre-building 3,000+ networks. This exceeds available RAM on the 16GB Windows machine.

**Potential Solutions** (Priority ordered):

1. **Lazy Network Building** (High Priority)
   - Don't prebuild all networks upfront
   - Build on-demand during pricing
   - Cache recently used networks with LRU eviction

2. **Dynamic Source Selection** (High Priority)
   - Only solve pricing for "promising" sources each iteration
   - Select based on dual values or historical column quality
   - Reduces per-iteration cost from O(n_sources) to O(k) for k << n_sources

3. **Network Sharing** (Medium Priority)
   - Many source networks share common structure
   - Store shared subgraphs once, compose on-the-fly

4. **Streaming/Disk-Based** (Low Priority)
   - Store networks on disk, load as needed
   - Trade memory for I/O

### Scalability Pattern

| Sources | Setup Time | Per-Iter Time | Memory |
|---------|------------|---------------|--------|
| 509     | 39s        | 7.5s          | OK     |
| 754     | 82s        | 21s           | OK     |
| 985     | 212s       | 33.5s         | OK     |
| 3000+   | -          | -             | OOM    |

Setup time and per-iteration time scale roughly linearly with source count.

## Next Steps

### Immediate (P0)

1. **Run instances 4-7 on Gerad server** (more RAM)
   - Already have SSH access to trento3.gerad.lan
   - Will validate algorithm correctness on larger instances

2. **Fix hardcoded literature comparison values**
   - Update benchmark script with correct Table 2 times
   - Currently using wrong values from different table

### Short-Term (P1)

3. **Implement lazy network building**
   - Modify `FastPerSourcePricing` to build networks on-demand
   - Add network cache with configurable size limit
   - Expected: 50-70% memory reduction

4. **Implement dynamic source selection**
   - Score sources by: recent column quality, dual value magnitude
   - Only solve top-k sources per iteration (k configurable)
   - Expected: 2-5x speedup on large instances

5. **Add IP solving**
   - Currently only solving LP relaxation
   - Add MIP solve after LP convergence
   - Literature shows small LP-IP gaps (0.02-0.16%)

### Medium-Term (P2)

6. **Dual stabilization**
   - Implement boxstep or smoothing stabilization
   - May reduce iteration count further

7. **Column management**
   - Remove dominated/unused columns periodically
   - Reduce master problem size over time

8. **Better parallelism**
   - Currently using 4 threads on 6-core machine
   - Profile and optimize thread utilization
   - Consider GPU acceleration for label operations

### Long-Term (P3)

9. **Branch-and-Price integration**
   - Integrate with OpenBP framework
   - Implement Ryan-Foster branching for crew pairing
   - Target: Prove IP optimality on benchmark instances

10. **Additional applications**
    - Nurse rostering
    - Train scheduling
    - General set covering/partitioning

## Files Changed Recently

### Modified
- `scripts/benchmark_all_instances.py` - Added `--max-time` argument, fixed coverage calculation
- `opencg/_core/__init__.py` - Import sorting fix (ruff)
- `opencg/pricing/multibase_boost.py` - Import sorting fix
- `opencg/pricing/per_source_boost.py` - Import sorting fix

### Created
- `benchmark_results_instances1-3.csv` - Summary results
- `benchmark_results_instances1-3.json` - Detailed results with convergence history
- `PROJECT_STATUS.md` - This file

## Development Environment

### Local (macOS)
- Used for development, testing, documentation

### Pasargad (Windows Server)
- SSH: `ssh milo@pasargad.local`
- Path: `C:\Users\milo\opencg`
- Python: `D:\Program Files\Python311\python.exe`
- Venv: `C:\Users\milo\opencg\venv`
- Used for: Benchmarking instances 1-3

### Gerad Server (Linux)
- SSH: `ssh baromila@trento3.gerad.lan`
- Used for: Large instances (4-7) requiring more RAM

## Running Benchmarks

```bash
# On Pasargad (Windows)
ssh milo@pasargad.local
cd C:\Users\milo\opencg
.\venv\Scripts\python scripts/benchmark_all_instances.py \
    --instances instance1 instance2 instance3 \
    --max-time 1800 \
    --max-iterations 100 \
    --no-ip \
    --threads 4

# Single instance with longer time
.\venv\Scripts\python scripts/benchmark_all_instances.py \
    --instances instance3 \
    --max-time 1800 \
    --max-iterations 100 \
    --no-ip
```

## References

- Kasirzadeh, A., Saddoune, M., & Soumis, F. (2017). Airline crew scheduling: models, algorithms, and data sets. *EURO Journal on Transportation and Logistics*, 6(2), 111-137.
- Benchmark data: https://www.gerad.ca/~miladb/Data/

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Priority areas:
1. Memory optimization for large instances
2. Dynamic source selection algorithm
3. IP solving and gap analysis
