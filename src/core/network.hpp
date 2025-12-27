/**
 * @file network.hpp
 * @brief Network data structures for SPPRC.
 *
 * Provides efficient graph representation with adjacency lists
 * for forward and backward traversal.
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

namespace opencg {

/**
 * @brief Arc (edge) in the network.
 */
struct Arc {
    int32_t index;
    int32_t source;
    int32_t target;
    double cost;
    std::vector<double> resource_consumption;

    // Optional: items covered by this arc
    std::vector<int32_t> covered_items;

    // Time window fields for VRPTW
    double travel_time = 0.0;      // Time to traverse this arc
    double service_time = 0.0;     // Service time at target
    double earliest = 0.0;         // Earliest arrival at target
    double latest = 1e30;          // Latest arrival at target (default: very large)

    Arc(int32_t idx, int32_t src, int32_t tgt, double c)
        : index(idx), source(src), target(tgt), cost(c) {}

    double get_consumption(size_t resource_index) const {
        return resource_index < resource_consumption.size()
            ? resource_consumption[resource_index]
            : 0.0;
    }
};

/**
 * @brief Node in the network.
 */
struct Node {
    int32_t index;
    bool is_source;
    bool is_sink;

    Node(int32_t idx, bool src = false, bool snk = false)
        : index(idx), is_source(src), is_sink(snk) {}
};

/**
 * @brief Network graph with efficient adjacency lists.
 *
 * Optimized for SPPRC:
 * - Contiguous storage for cache efficiency
 * - Separate forward and backward adjacency for bidirectional algorithms
 * - Pre-computed resource consumption arrays
 */
class Network {
public:
    Network() : source_index_(-1), sink_index_(-1) {}

    // Node operations
    int32_t add_node(bool is_source = false, bool is_sink = false) {
        int32_t idx = static_cast<int32_t>(nodes_.size());
        nodes_.emplace_back(idx, is_source, is_sink);
        outgoing_.emplace_back();
        incoming_.emplace_back();

        if (is_source) source_index_ = idx;
        if (is_sink) sink_index_ = idx;

        return idx;
    }

    int32_t add_source() { return add_node(true, false); }
    int32_t add_sink() { return add_node(false, true); }

    // Arc operations
    int32_t add_arc(int32_t source, int32_t target, double cost,
                    const std::vector<double>& resource_consumption = {},
                    const std::vector<int32_t>& covered_items = {}) {
        int32_t idx = static_cast<int32_t>(arcs_.size());
        arcs_.emplace_back(idx, source, target, cost);
        arcs_.back().resource_consumption = resource_consumption;
        arcs_.back().covered_items = covered_items;

        outgoing_[source].push_back(idx);
        incoming_[target].push_back(idx);

        return idx;
    }

    // Set time window attributes on an arc
    void set_arc_time_window(int32_t arc_index, double travel_time,
                             double service_time, double earliest, double latest) {
        if (arc_index >= 0 && static_cast<size_t>(arc_index) < arcs_.size()) {
            arcs_[arc_index].travel_time = travel_time;
            arcs_[arc_index].service_time = service_time;
            arcs_[arc_index].earliest = earliest;
            arcs_[arc_index].latest = latest;
        }
    }

    // Get mutable arc (for setting attributes)
    Arc& arc_mut(int32_t index) { return arcs_[index]; }

    // Accessors
    size_t num_nodes() const { return nodes_.size(); }
    size_t num_arcs() const { return arcs_.size(); }
    int32_t source_index() const { return source_index_; }
    int32_t sink_index() const { return sink_index_; }

    const Node& node(int32_t index) const { return nodes_[index]; }
    const Arc& arc(int32_t index) const { return arcs_[index]; }

    // Adjacency
    const std::vector<int32_t>& outgoing_arcs(int32_t node_index) const {
        return outgoing_[node_index];
    }

    const std::vector<int32_t>& incoming_arcs(int32_t node_index) const {
        return incoming_[node_index];
    }

    // Iteration
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<Arc>& arcs() const { return arcs_; }

private:
    std::vector<Node> nodes_;
    std::vector<Arc> arcs_;
    std::vector<std::vector<int32_t>> outgoing_;
    std::vector<std::vector<int32_t>> incoming_;
    int32_t source_index_;
    int32_t sink_index_;
};

}  // namespace opencg
