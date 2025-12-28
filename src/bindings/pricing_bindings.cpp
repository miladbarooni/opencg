/**
 * @file pricing_bindings.cpp
 * @brief pybind11 bindings for SPPRC labeling algorithm.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pricing/labeling.hpp"

namespace py = pybind11;

void init_pricing_bindings(py::module_& m) {
    using namespace opencg;

    // LabelingConfig
    py::class_<LabelingConfig>(m, "LabelingConfig", R"doc(
Configuration for the labeling algorithm.

Supports multiple optimization strategies:
- Topological order processing for DAGs (faster than priority queue)
- ng-path relaxation (faster alternative to full elementarity)
- Label limits per node (heuristic beam search)
)doc")
        .def(py::init<>())
        .def_readwrite("max_labels", &LabelingConfig::max_labels,
            "Maximum labels to create (0 = unlimited)")
        .def_readwrite("max_time", &LabelingConfig::max_time,
            "Maximum solve time in seconds (0 = unlimited)")
        .def_readwrite("max_columns", &LabelingConfig::max_columns,
            "Maximum columns to return (0 = all)")
        .def_readwrite("rc_threshold", &LabelingConfig::rc_threshold,
            "Reduced cost threshold for columns")
        .def_readwrite("check_dominance", &LabelingConfig::check_dominance,
            "Whether to use dominance pruning")
        .def_readwrite("check_elementarity", &LabelingConfig::check_elementarity,
            "Whether to enforce elementary paths")
        // Optimization options
        .def_readwrite("use_topological_order", &LabelingConfig::use_topological_order,
            "Process nodes in topological order (for DAGs, more efficient)")
        .def_readwrite("max_labels_per_node", &LabelingConfig::max_labels_per_node,
            "Maximum labels per node (0 = unlimited, acts as beam search)")
        .def_readwrite("use_ng_path", &LabelingConfig::use_ng_path,
            "Use ng-path relaxation for elementarity (faster than full)")
        .def_readwrite("ng_neighborhood_size", &LabelingConfig::ng_neighborhood_size,
            "Size of ng-neighborhood for ng-path relaxation")
        // Time window support
        .def_readwrite("use_time_windows", &LabelingConfig::use_time_windows,
            "Enable time window constraints for VRPTW")
        .def_readwrite("time_resource_index", &LabelingConfig::time_resource_index,
            "Index of the time resource for time window processing");

    // LabelingResult
    py::class_<LabelingResult> result_class(m, "LabelingResult", R"doc(
Result of the labeling algorithm.
)doc");

    py::enum_<LabelingResult::Status>(result_class, "Status")
        .value("OPTIMAL", LabelingResult::Status::OPTIMAL)
        .value("COLUMNS_FOUND", LabelingResult::Status::COLUMNS_FOUND)
        .value("NO_COLUMNS", LabelingResult::Status::NO_COLUMNS)
        .value("TIME_LIMIT", LabelingResult::Status::TIME_LIMIT)
        .value("LABEL_LIMIT", LabelingResult::Status::LABEL_LIMIT)
        .value("ERROR", LabelingResult::Status::ERROR)
        .export_values();

    result_class
        .def_readonly("status", &LabelingResult::status)
        .def_readonly("columns", &LabelingResult::columns)
        .def_readonly("best_reduced_cost", &LabelingResult::best_reduced_cost)
        .def_readonly("labels_created", &LabelingResult::labels_created)
        .def_readonly("labels_dominated", &LabelingResult::labels_dominated)
        .def_readonly("solve_time", &LabelingResult::solve_time)
        .def_readonly("iterations", &LabelingResult::iterations)

        .def_property_readonly("has_negative_reduced_cost",
            [](const LabelingResult& self) {
                return self.best_reduced_cost < -1e-6;
            })
        .def_property_readonly("num_columns",
            [](const LabelingResult& self) {
                return self.columns.size();
            })

        .def("__repr__", [](const LabelingResult& self) {
            std::string status_str;
            switch (self.status) {
                case LabelingResult::Status::OPTIMAL: status_str = "OPTIMAL"; break;
                case LabelingResult::Status::COLUMNS_FOUND: status_str = "COLUMNS_FOUND"; break;
                case LabelingResult::Status::NO_COLUMNS: status_str = "NO_COLUMNS"; break;
                case LabelingResult::Status::TIME_LIMIT: status_str = "TIME_LIMIT"; break;
                case LabelingResult::Status::LABEL_LIMIT: status_str = "LABEL_LIMIT"; break;
                case LabelingResult::Status::ERROR: status_str = "ERROR"; break;
            }
            return "<LabelingResult status=" + status_str +
                   " cols=" + std::to_string(self.columns.size()) +
                   " rc=" + std::to_string(self.best_reduced_cost) + ">";
        });

    // LabelingAlgorithm
    py::class_<LabelingAlgorithm>(m, "LabelingAlgorithm", R"doc(
High-performance labeling algorithm for SPPRC.

This is a C++ implementation of the labeling algorithm, providing
significant speedup over the pure Python version for large problems.

Example:
    >>> from opencg._core import Network, LabelingAlgorithm, LabelingConfig
    >>> # Build network
    >>> net = Network()
    >>> source = net.add_source()
    >>> sink = net.add_sink()
    >>> # ... add nodes and arcs ...
    >>> # Create algorithm
    >>> config = LabelingConfig()
    >>> config.max_columns = 10
    >>> algo = LabelingAlgorithm(net, num_resources=2, resource_limits=[10.0, 8.0], config=config)
    >>> # Set duals and solve
    >>> algo.set_dual_values({0: 5.0, 1: 3.0})
    >>> result = algo.solve()
    >>> for label in result.columns:
    ...     print(f"Column with RC={label.reduced_cost}")
)doc")
        .def(py::init<const Network&, size_t, const std::vector<double>&, const LabelingConfig&>(),
             py::arg("network"),
             py::arg("num_resources"),
             py::arg("resource_limits"),
             py::arg("config") = LabelingConfig(),
             py::keep_alive<1, 2>(),  // Keep network alive while algorithm exists
             "Create the labeling algorithm")

        .def("set_dual_values", &LabelingAlgorithm::set_dual_values,
             py::arg("dual_values"),
             "Set dual values for reduced cost computation")

        .def("solve", &LabelingAlgorithm::solve,
             py::call_guard<py::gil_scoped_release>(),
             "Solve the SPPRC and return columns with negative reduced cost")

        .def("set_max_columns", &LabelingAlgorithm::set_max_columns,
             py::arg("max_columns"),
             "Set the maximum number of columns to return (0 = all with negative RC)")

        .def("get_max_columns", &LabelingAlgorithm::get_max_columns,
             "Get the current max_columns setting")

        .def_property_readonly("network", &LabelingAlgorithm::network,
             py::return_value_policy::reference)

        .def("__repr__", [](const LabelingAlgorithm& self) {
            return "<LabelingAlgorithm network_size=" +
                   std::to_string(self.network().num_nodes()) + ">";
        });
}
