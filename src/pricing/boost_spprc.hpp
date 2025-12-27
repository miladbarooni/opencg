/**
 * @file boost_spprc.hpp
 * @brief SPPRC solver using Boost Graph Library's r_c_shortest_paths.
 *
 * This provides a high-performance implementation of the Shortest Path Problem
 * with Resource Constraints using Boost's label-setting algorithm.
 *
 * Key features:
 * - Mono-directional label-setting algorithm
 * - Efficient resource extension and dominance checking
 * - Returns all Pareto-optimal paths
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/r_c_shortest_paths.hpp>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <limits>
#include <algorithm>

namespace opencg {

/**
 * @brief Resource container for Boost SPPRC.
 *
 * Tracks multiple numeric resources plus covered items for elementarity.
 */
struct SPPRCResource {
    std::vector<double> values;           // Resource consumption values
    std::unordered_set<int32_t> covered;  // Covered items (for elementarity)
    double cost;                          // Total path cost
    double reduced_cost;                  // Reduced cost (cost - duals)

    SPPRCResource() : cost(0.0), reduced_cost(0.0) {}

    SPPRCResource(size_t num_resources)
        : values(num_resources, 0.0), cost(0.0), reduced_cost(0.0) {}

    // Required for Boost: comparison operators
    bool operator<(const SPPRCResource& other) const {
        return reduced_cost < other.reduced_cost;
    }

    bool operator>(const SPPRCResource& other) const {
        return reduced_cost > other.reduced_cost;
    }

    bool operator==(const SPPRCResource& other) const {
        if (values.size() != other.values.size()) return false;
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::abs(values[i] - other.values[i]) > 1e-9) return false;
        }
        return covered == other.covered &&
               std::abs(reduced_cost - other.reduced_cost) < 1e-9;
    }

    bool operator<=(const SPPRCResource& other) const {
        return *this < other || *this == other;
    }

    bool operator>=(const SPPRCResource& other) const {
        return *this > other || *this == other;
    }
};

/**
 * @brief Vertex properties for Boost SPPRC.
 */
struct SPPRCVertexProperties {
    int num;  // Vertex number for Boost (contiguous integers starting from 0)

    SPPRCVertexProperties(int n = 0) : num(n) {}
};

/**
 * @brief Arc properties for Boost SPPRC.
 *
 * Uses num field for Boost's r_c_shortest_paths requirement (contiguous edge indices).
 */
struct SPPRCArcProperties {
    int num;            // Edge number for Boost (required: contiguous integers)
    double cost;
    double reduced_cost;
    std::vector<double> resource_consumption;
    std::vector<int32_t> covered_items;
    int32_t arc_index;  // Original arc index in our network

    SPPRCArcProperties(int n = 0, double c = 0.0, double rc = 0.0)
        : num(n), cost(c), reduced_cost(rc), arc_index(-1) {}
};

// Graph type matching Boost example: vecS for both vertices and edges
using SPPRCGraph = boost::adjacency_list<
    boost::vecS,        // Edge container
    boost::vecS,        // Vertex container
    boost::directedS,
    SPPRCVertexProperties,
    SPPRCArcProperties
>;

using SPPRCVertex = boost::graph_traits<SPPRCGraph>::vertex_descriptor;
using SPPRCArc = boost::graph_traits<SPPRCGraph>::edge_descriptor;

/**
 * @brief Resource extension function for Boost SPPRC.
 */
class SPPRCResourceExtension {
public:
    SPPRCResourceExtension(
        const std::vector<double>& resource_limits,
        bool check_elementarity
    ) : limits_(resource_limits), check_elem_(check_elementarity) {}

    bool operator()(
        const SPPRCGraph& g,
        SPPRCResource& new_res,
        const SPPRCResource& old_res,
        SPPRCArc arc
    ) const {
        const SPPRCArcProperties& props = g[arc];

        // Copy old resource
        new_res = old_res;

        // Add cost and reduced cost
        new_res.cost += props.cost;
        new_res.reduced_cost += props.reduced_cost;

        // Extend resources
        for (size_t i = 0; i < props.resource_consumption.size() && i < new_res.values.size(); ++i) {
            new_res.values[i] += props.resource_consumption[i];
            // Clamp at 0 (for reset arcs with negative consumption)
            if (new_res.values[i] < 0.0) {
                new_res.values[i] = 0.0;
            }
            // Check feasibility
            if (i < limits_.size() && new_res.values[i] > limits_[i]) {
                return false;  // Infeasible
            }
        }

        // Check elementarity (no repeated covered items)
        if (check_elem_) {
            for (int32_t item : props.covered_items) {
                if (new_res.covered.count(item) > 0) {
                    return false;  // Would revisit an item
                }
            }
        }

        // Add covered items
        for (int32_t item : props.covered_items) {
            new_res.covered.insert(item);
        }

        return true;  // Feasible extension
    }

private:
    std::vector<double> limits_;
    bool check_elem_;
};

/**
 * @brief Dominance function for Boost SPPRC.
 *
 * res1 dominates res2 if:
 * - reduced_cost1 <= reduced_cost2
 * - All resource values of res1 <= res2
 * - covered1 ⊆ covered2
 */
class SPPRCDominance {
public:
    bool operator()(const SPPRCResource& res1, const SPPRCResource& res2) const {
        // Check reduced cost
        if (res1.reduced_cost > res2.reduced_cost + 1e-9) {
            return false;
        }

        // Check resources
        for (size_t i = 0; i < res1.values.size(); ++i) {
            if (res1.values[i] > res2.values[i] + 1e-9) {
                return false;
            }
        }

        // Check covered items (res1 must be subset of res2)
        for (int32_t item : res1.covered) {
            if (res2.covered.count(item) == 0) {
                return false;
            }
        }

        return true;
    }
};

/**
 * @brief Result of Boost SPPRC solver.
 */
struct BoostSPPRCResult {
    struct Path {
        std::vector<int32_t> arc_indices;
        std::unordered_set<int32_t> covered_items;
        double cost;
        double reduced_cost;
        std::vector<double> resources;
    };

    std::vector<Path> paths;
    double solve_time;
    size_t num_labels;
};

/**
 * @brief High-performance SPPRC solver using Boost.
 */
class BoostSPPRCSolver {
public:
    /**
     * @brief Construct solver.
     * @param num_resources Number of resources to track
     * @param resource_limits Upper bounds on resources
     * @param check_elementarity Whether to enforce elementary paths
     */
    BoostSPPRCSolver(
        size_t num_resources,
        const std::vector<double>& resource_limits,
        bool check_elementarity = true
    ) : num_resources_(num_resources),
        resource_limits_(resource_limits),
        check_elementarity_(check_elementarity),
        source_vertex_(0),
        sink_vertex_(0),
        num_edges_(0)
    {
        // Ensure limits vector is sized correctly
        if (resource_limits_.size() < num_resources_) {
            resource_limits_.resize(num_resources_, std::numeric_limits<double>::infinity());
        }
    }

    /**
     * @brief Clear the graph and reset.
     */
    void clear() {
        graph_ = SPPRCGraph();
        vertex_map_.clear();
        source_vertex_ = 0;
        sink_vertex_ = 0;
        num_edges_ = 0;
    }

    /**
     * @brief Add a vertex to the graph.
     * @param id External vertex ID
     * @return Internal vertex descriptor
     */
    SPPRCVertex add_vertex(int32_t id) {
        int vertex_num = static_cast<int>(boost::num_vertices(graph_));
        SPPRCVertex v = boost::add_vertex(SPPRCVertexProperties(vertex_num), graph_);
        vertex_map_[id] = v;
        return v;
    }

    /**
     * @brief Set source vertex.
     */
    void set_source(int32_t id) {
        source_vertex_ = vertex_map_.at(id);
    }

    /**
     * @brief Set sink vertex.
     */
    void set_sink(int32_t id) {
        sink_vertex_ = vertex_map_.at(id);
    }

    /**
     * @brief Add an arc to the graph.
     */
    void add_arc(
        int32_t source_id,
        int32_t target_id,
        double cost,
        double reduced_cost,
        const std::vector<double>& resource_consumption,
        const std::vector<int32_t>& covered_items,
        int32_t arc_index
    ) {
        SPPRCVertex u = vertex_map_.at(source_id);
        SPPRCVertex v = vertex_map_.at(target_id);

        SPPRCArcProperties props(static_cast<int>(num_edges_), cost, reduced_cost);
        props.resource_consumption = resource_consumption;
        props.covered_items = covered_items;
        props.arc_index = arc_index;

        auto [edge, inserted] = boost::add_edge(u, v, props, graph_);
        if (inserted) {
            num_edges_++;
        }
    }

    /**
     * @brief Solve the SPPRC.
     * @param max_paths Maximum number of paths to return (0 = all)
     * @param rc_threshold Only return paths with RC < this
     * @return Result with all Pareto-optimal paths
     */
    BoostSPPRCResult solve(
        size_t max_paths = 0,
        double rc_threshold = -1e-6
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        BoostSPPRCResult result;

        // Initial resource (all zeros)
        SPPRCResource initial_res(num_resources_);

        // Output containers
        std::vector<std::vector<SPPRCArc>> opt_solutions;
        std::vector<SPPRCResource> pareto_optimal_resources;

        // Create extension and dominance functors
        SPPRCResourceExtension ref(resource_limits_, check_elementarity_);
        SPPRCDominance dom;

        // Run Boost SPPRC using property maps (matching Boost example pattern)
        try {
            boost::r_c_shortest_paths(
                graph_,
                boost::get(&SPPRCVertexProperties::num, graph_),  // vertex index map
                boost::get(&SPPRCArcProperties::num, graph_),     // edge index map
                source_vertex_,
                sink_vertex_,
                opt_solutions,
                pareto_optimal_resources,
                initial_res,
                ref,
                dom,
                std::allocator<boost::r_c_shortest_paths_label<SPPRCGraph, SPPRCResource>>(),
                boost::default_r_c_shortest_paths_visitor()
            );
        } catch (const std::exception& e) {
            // Handle any exceptions from Boost
            auto end_time = std::chrono::high_resolution_clock::now();
            result.solve_time = std::chrono::duration<double>(end_time - start_time).count();
            result.num_labels = 0;
            return result;
        }

        // Convert results
        for (size_t i = 0; i < opt_solutions.size(); ++i) {
            const auto& solution = opt_solutions[i];
            const auto& resources = pareto_optimal_resources[i];

            // Filter by reduced cost threshold
            if (resources.reduced_cost >= rc_threshold) {
                continue;
            }

            BoostSPPRCResult::Path path;
            path.cost = resources.cost;
            path.reduced_cost = resources.reduced_cost;
            path.resources = resources.values;
            path.covered_items = resources.covered;

            // Extract arc indices (note: solution is in reverse order from sink to source)
            for (auto it = solution.rbegin(); it != solution.rend(); ++it) {
                path.arc_indices.push_back(graph_[*it].arc_index);
            }

            result.paths.push_back(std::move(path));

            // Limit number of paths
            if (max_paths > 0 && result.paths.size() >= max_paths) {
                break;
            }
        }

        // Sort by reduced cost
        std::sort(result.paths.begin(), result.paths.end(),
            [](const auto& a, const auto& b) {
                return a.reduced_cost < b.reduced_cost;
            });

        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time = std::chrono::duration<double>(end_time - start_time).count();
        result.num_labels = opt_solutions.size();

        return result;
    }

private:
    size_t num_resources_;
    std::vector<double> resource_limits_;
    bool check_elementarity_;

    SPPRCGraph graph_;
    std::unordered_map<int32_t, SPPRCVertex> vertex_map_;
    SPPRCVertex source_vertex_;
    SPPRCVertex sink_vertex_;
    size_t num_edges_;
};

}  // namespace opencg
