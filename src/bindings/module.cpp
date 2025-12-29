/**
 * @file module.cpp
 * @brief Main pybind11 module definition.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void init_label_bindings(py::module_& m);
void init_pricing_bindings(py::module_& m);
void init_parallel_pricing_bindings(py::module_& m);

#ifdef HAS_BOOST
void init_boost_spprc_bindings(py::module_& m);
#endif

PYBIND11_MODULE(_core, m) {
    m.doc() = R"doc(
OpenCG C++ Core Module

High-performance implementations of performance-critical algorithms
for column generation.

This module provides:
- Label: Fast label class for SPPRC
- LabelPool: Efficient label management with dominance checking
- Network: Graph data structure
- LabelingAlgorithm: High-performance SPPRC solver
- ParallelLabelingSolver: Multi-threaded parallel pricing

These classes are designed to be drop-in replacements for the pure
Python implementations when performance is critical.
)doc";

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("HAS_CPP_BACKEND") = true;

#ifdef HAS_BOOST
    m.attr("HAS_BOOST") = true;
#else
    m.attr("HAS_BOOST") = false;
#endif

    // Initialize submodules
    init_label_bindings(m);
    init_pricing_bindings(m);
    init_parallel_pricing_bindings(m);

#ifdef HAS_BOOST
    init_boost_spprc_bindings(m);
#endif
}
