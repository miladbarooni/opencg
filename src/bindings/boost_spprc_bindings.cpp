/**
 * @file boost_spprc_bindings.cpp
 * @brief pybind11 bindings for Boost SPPRC solver.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pricing/boost_spprc.hpp"

namespace py = pybind11;

void init_boost_spprc_bindings(py::module_& m) {
    using namespace opencg;

    // BoostSPPRCResult::Path
    py::class_<BoostSPPRCResult::Path>(m, "BoostSPPRCPath", R"doc(
A single path found by the Boost SPPRC solver.
)doc")
        .def_readonly("arc_indices", &BoostSPPRCResult::Path::arc_indices,
            "Arc indices forming the path")
        .def_readonly("covered_items", &BoostSPPRCResult::Path::covered_items,
            "Items covered by this path")
        .def_readonly("cost", &BoostSPPRCResult::Path::cost,
            "Total path cost")
        .def_readonly("reduced_cost", &BoostSPPRCResult::Path::reduced_cost,
            "Reduced cost of the path")
        .def_readonly("resources", &BoostSPPRCResult::Path::resources,
            "Final resource values")
        .def("__repr__", [](const BoostSPPRCResult::Path& self) {
            return "<BoostSPPRCPath arcs=" + std::to_string(self.arc_indices.size()) +
                   " rc=" + std::to_string(self.reduced_cost) + ">";
        });

    // BoostSPPRCResult
    py::class_<BoostSPPRCResult>(m, "BoostSPPRCResult", R"doc(
Result from the Boost SPPRC solver.
)doc")
        .def_readonly("paths", &BoostSPPRCResult::paths,
            "List of Pareto-optimal paths found")
        .def_readonly("solve_time", &BoostSPPRCResult::solve_time,
            "Time taken to solve in seconds")
        .def_readonly("num_labels", &BoostSPPRCResult::num_labels,
            "Number of labels explored")
        .def_property_readonly("num_paths", [](const BoostSPPRCResult& self) {
            return self.paths.size();
        })
        .def("__repr__", [](const BoostSPPRCResult& self) {
            return "<BoostSPPRCResult paths=" + std::to_string(self.paths.size()) +
                   " time=" + std::to_string(self.solve_time) + "s>";
        });

    // BoostSPPRCSolver
    py::class_<BoostSPPRCSolver>(m, "BoostSPPRCSolver", R"doc(
High-performance SPPRC solver using Boost Graph Library.

This solver uses Boost's r_c_shortest_paths algorithm to find all
Pareto-optimal paths from source to sink respecting resource constraints.

Example:
    >>> solver = BoostSPPRCSolver(num_resources=2, resource_limits=[10.0, 8.0])
    >>> solver.add_vertex(0)  # source
    >>> solver.add_vertex(1)  # intermediate
    >>> solver.add_vertex(2)  # sink
    >>> solver.set_source(0)
    >>> solver.set_sink(2)
    >>> solver.add_arc(0, 1, cost=5.0, reduced_cost=-10.0,
    ...                resource_consumption=[1.0, 2.0], covered_items=[0], arc_index=0)
    >>> solver.add_arc(1, 2, cost=3.0, reduced_cost=-5.0,
    ...                resource_consumption=[2.0, 1.0], covered_items=[], arc_index=1)
    >>> result = solver.solve(max_paths=10, rc_threshold=0.0)
    >>> for path in result.paths:
    ...     print(f"Path: arcs={path.arc_indices}, rc={path.reduced_cost}")
)doc")
        .def(py::init<size_t, const std::vector<double>&, bool>(),
             py::arg("num_resources"),
             py::arg("resource_limits"),
             py::arg("check_elementarity") = true,
             "Create a Boost SPPRC solver")

        .def("clear", &BoostSPPRCSolver::clear,
             "Clear the graph and reset")

        .def("add_vertex", &BoostSPPRCSolver::add_vertex,
             py::arg("id"),
             "Add a vertex with the given ID")

        .def("set_source", &BoostSPPRCSolver::set_source,
             py::arg("id"),
             "Set the source vertex")

        .def("set_sink", &BoostSPPRCSolver::set_sink,
             py::arg("id"),
             "Set the sink vertex")

        .def("add_arc", &BoostSPPRCSolver::add_arc,
             py::arg("source_id"),
             py::arg("target_id"),
             py::arg("cost"),
             py::arg("reduced_cost"),
             py::arg("resource_consumption"),
             py::arg("covered_items"),
             py::arg("arc_index"),
             "Add an arc to the graph")

        .def("solve", &BoostSPPRCSolver::solve,
             py::arg("max_paths") = 0,
             py::arg("rc_threshold") = -1e-6,
             "Solve the SPPRC and return Pareto-optimal paths")

        .def("__repr__", [](const BoostSPPRCSolver&) {
            return "<BoostSPPRCSolver>";
        });
}
