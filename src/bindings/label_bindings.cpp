/**
 * @file label_bindings.cpp
 * @brief pybind11 bindings for Label and LabelPool classes.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/label.hpp"
#include "core/network.hpp"

namespace py = pybind11;

void init_label_bindings(py::module_& m) {
    using namespace opencg;

    // Label class
    py::class_<Label>(m, "Label", R"doc(
A label representing a partial path in SPPRC.

This is a high-performance C++ implementation of the Label class.
It provides the same interface as the Python version but with
significantly better performance for large-scale problems.
)doc")
        .def(py::init<int32_t, size_t>(),
             py::arg("node_index"),
             py::arg("num_resources"),
             "Create a source label at a node")

        .def_property_readonly("node_index", &Label::node_index,
            "Index of the node this label is at")
        .def_property_readonly("cost", &Label::cost,
            "Total cost of the path")
        .def_property_readonly("reduced_cost", &Label::reduced_cost,
            "Reduced cost of the path")
        .def_property_readonly("label_id", &Label::label_id,
            "Unique identifier for this label")
        .def_property_readonly("num_resources", &Label::num_resources,
            "Number of resources tracked")
        .def_property_readonly("num_covered", &Label::num_covered,
            "Number of items covered")

        .def("resource", &Label::resource,
             py::arg("index"),
             "Get resource value at index")

        .def_property_readonly("resource_values", &Label::resource_values,
            "Get all resource values as a list")

        .def("covers_item", &Label::covers_item,
             py::arg("item"),
             "Check if this label covers a specific item")

        .def_property_readonly("covered_items",
            [](const Label& self) {
                return std::vector<int32_t>(
                    self.covered_items().begin(),
                    self.covered_items().end()
                );
            },
            "Get list of covered items")

        .def_property_readonly("is_source_label", &Label::is_source_label,
            "Check if this is the source label")
        .def_property_readonly("path_length", &Label::path_length,
            "Number of arcs in the path")

        .def("get_arc_indices", &Label::get_arc_indices,
             "Get arc indices forming the path")
        .def("get_node_indices", &Label::get_node_indices,
             "Get node indices forming the path")

        .def("__repr__", [](const Label& self) {
            return "<Label node=" + std::to_string(self.node_index()) +
                   " cost=" + std::to_string(self.cost()) +
                   " rc=" + std::to_string(self.reduced_cost()) + ">";
        })

        .def("__lt__", &Label::operator<)
        .def("__gt__", &Label::operator>);


    // LabelPool class
    py::class_<LabelPool>(m, "LabelPool", R"doc(
Pool for managing labels at each node during SPPRC.

Provides efficient storage and dominance checking for labels.
This is a high-performance C++ implementation.
)doc")
        .def(py::init<size_t, size_t>(),
             py::arg("num_nodes"),
             py::arg("num_resources"),
             "Create a label pool")

        .def_property_readonly("num_nodes", &LabelPool::num_nodes,
            "Number of nodes")
        .def_property_readonly("total_labels", &LabelPool::total_labels,
            "Total labels currently stored")
        .def_property_readonly("total_created", &LabelPool::total_created,
            "Total labels created")
        .def_property_readonly("total_dominated", &LabelPool::total_dominated,
            "Total labels pruned by dominance")

        .def("create_source_label", &LabelPool::create_source_label,
             py::arg("node_index"),
             py::return_value_policy::reference,
             "Create a source label at a node")

        .def("extend_label", &LabelPool::extend_label,
             py::arg("predecessor"),
             py::arg("target_node"),
             py::arg("arc_index"),
             py::arg("arc_cost"),
             py::arg("arc_reduced_cost"),
             py::arg("resource_consumption"),
             py::arg("resource_limits"),
             py::arg("check_dominance") = true,
             py::return_value_policy::reference,
             "Extend a label along an arc")

        .def("get_labels", &LabelPool::get_labels,
             py::arg("node_index"),
             py::return_value_policy::reference,
             "Get all labels at a node")

        .def("clear", &LabelPool::clear,
             "Clear all labels")

        .def("__repr__", [](const LabelPool& self) {
            return "<LabelPool nodes=" + std::to_string(self.num_nodes()) +
                   " labels=" + std::to_string(self.total_labels()) + ">";
        });


    // Network classes
    py::class_<Arc>(m, "Arc", "Arc in the network")
        .def_readonly("index", &Arc::index)
        .def_readonly("source", &Arc::source)
        .def_readonly("target", &Arc::target)
        .def_readonly("cost", &Arc::cost)
        .def_readonly("resource_consumption", &Arc::resource_consumption)
        .def_readonly("covered_items", &Arc::covered_items)
        // Time window fields
        .def_readwrite("travel_time", &Arc::travel_time, "Travel time on this arc")
        .def_readwrite("service_time", &Arc::service_time, "Service time at target")
        .def_readwrite("earliest", &Arc::earliest, "Earliest arrival at target")
        .def_readwrite("latest", &Arc::latest, "Latest arrival at target")
        .def("get_consumption", &Arc::get_consumption,
             py::arg("resource_index"),
             "Get resource consumption at index");

    py::class_<Node>(m, "Node", "Node in the network")
        .def_readonly("index", &Node::index)
        .def_readonly("is_source", &Node::is_source)
        .def_readonly("is_sink", &Node::is_sink);

    py::class_<Network>(m, "Network", R"doc(
Network graph with efficient adjacency lists.

High-performance C++ implementation optimized for SPPRC.
)doc")
        .def(py::init<>())

        .def("add_node", &Network::add_node,
             py::arg("is_source") = false,
             py::arg("is_sink") = false,
             "Add a node to the network")
        .def("add_source", &Network::add_source,
             "Add a source node")
        .def("add_sink", &Network::add_sink,
             "Add a sink node")

        .def("add_arc", &Network::add_arc,
             py::arg("source"),
             py::arg("target"),
             py::arg("cost"),
             py::arg("resource_consumption") = std::vector<double>{},
             py::arg("covered_items") = std::vector<int32_t>{},
             "Add an arc to the network")

        .def_property_readonly("num_nodes", &Network::num_nodes)
        .def_property_readonly("num_arcs", &Network::num_arcs)
        .def_property_readonly("source_index", &Network::source_index)
        .def_property_readonly("sink_index", &Network::sink_index)

        .def("node", &Network::node,
             py::arg("index"),
             py::return_value_policy::reference)
        .def("arc", &Network::arc,
             py::arg("index"),
             py::return_value_policy::reference)

        .def("outgoing_arcs", &Network::outgoing_arcs,
             py::arg("node_index"),
             py::return_value_policy::reference)
        .def("incoming_arcs", &Network::incoming_arcs,
             py::arg("node_index"),
             py::return_value_policy::reference)

        .def("__repr__", [](const Network& self) {
            return "<Network nodes=" + std::to_string(self.num_nodes()) +
                   " arcs=" + std::to_string(self.num_arcs()) + ">";
        });
}
