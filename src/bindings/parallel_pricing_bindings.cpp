/**
 * @file parallel_pricing_bindings.cpp
 * @brief pybind11 bindings for parallel SPPRC labeling.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pricing/parallel_labeling.hpp"

namespace py = pybind11;

void init_parallel_pricing_bindings(py::module_& m) {
    using namespace opencg;

    // ParallelLabelingConfig
    py::class_<ParallelLabelingConfig>(m, "ParallelLabelingConfig", R"doc(
Configuration for parallel labeling solver.

Parameters:
    num_threads: Number of threads (0 = auto, uses all CPU cores)
    max_total_columns: Maximum columns to collect across all sources (0 = unlimited)
    max_total_time: Maximum solve time in seconds (0 = unlimited)
    collect_all_columns: Whether to collect all columns or just track best
)doc")
        .def(py::init<>())
        .def_readwrite("num_threads", &ParallelLabelingConfig::num_threads,
            "Number of threads (0 = auto)")
        .def_readwrite("max_total_columns", &ParallelLabelingConfig::max_total_columns,
            "Maximum total columns to collect")
        .def_readwrite("max_total_time", &ParallelLabelingConfig::max_total_time,
            "Maximum solve time in seconds")
        .def_readwrite("collect_all_columns", &ParallelLabelingConfig::collect_all_columns,
            "Whether to collect all columns");

    // ParallelLabelingResult
    py::class_<ParallelLabelingResult>(m, "ParallelLabelingResult", R"doc(
Result from parallel labeling across multiple sources.
)doc")
        .def_readonly("all_columns", &ParallelLabelingResult::all_columns,
            "All columns found across sources")
        .def_readonly("best_reduced_cost", &ParallelLabelingResult::best_reduced_cost,
            "Best reduced cost found")
        .def_readonly("total_labels_created", &ParallelLabelingResult::total_labels_created,
            "Total labels created across all sources")
        .def_readonly("total_labels_dominated", &ParallelLabelingResult::total_labels_dominated,
            "Total labels dominated across all sources")
        .def_readonly("solve_time", &ParallelLabelingResult::solve_time,
            "Total solve time in seconds")
        .def_readonly("sources_processed", &ParallelLabelingResult::sources_processed,
            "Number of sources processed")
        .def_property_readonly("num_columns",
            [](const ParallelLabelingResult& self) {
                return self.all_columns.size();
            },
            "Number of columns found")
        .def("__repr__", [](const ParallelLabelingResult& self) {
            return "<ParallelLabelingResult sources=" + std::to_string(self.sources_processed) +
                   " cols=" + std::to_string(self.all_columns.size()) +
                   " time=" + std::to_string(self.solve_time) + "s>";
        });

    // ParallelLabelingSolver
    py::class_<ParallelLabelingSolver>(m, "ParallelLabelingSolver", R"doc(
Parallel labeling solver for multiple independent networks.

This class manages multiple LabelingAlgorithm instances and solves
them in parallel using C++ threads. Each source network is solved
independently, and results are aggregated.

This is useful for per-source pricing in crew pairing problems where
we need to run labeling from many different source arcs.

Example:
    >>> from opencg._core import (
    ...     ParallelLabelingSolver, ParallelLabelingConfig,
    ...     Network, LabelingConfig
    ... )
    >>> # Configure parallel solver
    >>> pconfig = ParallelLabelingConfig()
    >>> pconfig.num_threads = 4
    >>> solver = ParallelLabelingSolver(pconfig)
    >>> # Add source networks
    >>> for source_net in source_networks:
    ...     solver.add_source(source_net, num_resources, limits, lconfig)
    >>> # Solve all in parallel
    >>> solver.set_dual_values(duals)
    >>> result = solver.solve()
    >>> print(f"Found {result.num_columns} columns in {result.solve_time:.2f}s")
)doc")
        .def(py::init<const ParallelLabelingConfig&>(),
             py::arg("config") = ParallelLabelingConfig(),
             "Create a parallel labeling solver")

        .def("add_source", &ParallelLabelingSolver::add_source,
             py::arg("network"),
             py::arg("num_resources"),
             py::arg("resource_limits"),
             py::arg("config") = LabelingConfig(),
             py::arg("source_id") = -1,
             py::keep_alive<1, 2>(),  // Keep network alive
             "Add a source network to be solved")

        .def("set_dual_values", &ParallelLabelingSolver::set_dual_values,
             py::arg("dual_values"),
             "Set dual values for all source algorithms")

        .def("solve", &ParallelLabelingSolver::solve,
             py::call_guard<py::gil_scoped_release>(),
             "Solve all sources in parallel")

        .def("clear", &ParallelLabelingSolver::clear,
             "Clear all sources")

        .def_property_readonly("num_sources", &ParallelLabelingSolver::num_sources,
             "Number of sources added")

        .def_property_readonly("num_threads", &ParallelLabelingSolver::num_threads,
             "Number of threads configured")

        .def("__repr__", [](const ParallelLabelingSolver& self) {
            return "<ParallelLabelingSolver sources=" + std::to_string(self.num_sources()) +
                   " threads=" + std::to_string(self.num_threads()) + ">";
        });
}
