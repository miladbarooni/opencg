/**
 * @file labeling.hpp
 * @brief SPPRC labeling algorithm implementation.
 *
 * High-performance implementation of the mono-directional labeling algorithm
 * for the Shortest Path Problem with Resource Constraints.
 *
 * Features:
 * - Priority queue (best-first) or topological order processing
 * - Full elementarity or ng-path relaxation
 * - Configurable label limits per node
 * - Early termination support
 */

#pragma once

#include "core/label.hpp"
#include "core/network.hpp"

#include <queue>
#include <vector>
#include <functional>
#include <chrono>
#include <algorithm>
#include <unordered_set>

namespace opencg {

/**
 * @brief Configuration for the labeling algorithm.
 */
struct LabelingConfig {
    size_t max_labels = 0;          // 0 = unlimited total labels
    double max_time = 0.0;          // 0 = unlimited (seconds)
    size_t max_columns = 0;         // 0 = all with negative RC
    double rc_threshold = -1e-6;    // Only keep columns with RC < this
    bool check_dominance = true;
    bool check_elementarity = false;

    // Optimization options
    bool use_topological_order = false;  // Process nodes in topological order (for DAGs)
    size_t max_labels_per_node = 0;      // 0 = unlimited, limits labels at each node (heuristic)

    // ng-path relaxation for elementarity (alternative to full elementarity)
    // When enabled, only checks elementarity within ng-neighborhood of each node
    bool use_ng_path = false;            // Use ng-path relaxation
    size_t ng_neighborhood_size = 8;     // Size of ng-neighborhood for each node

    // Time window support for VRPTW
    // When enabled, uses arc time window attributes for non-linear time resource extension
    bool use_time_windows = false;       // Enable time window constraints
    size_t time_resource_index = 0;      // Which resource index represents time
};

/**
 * @brief Result of the labeling algorithm.
 */
struct LabelingResult {
    enum class Status {
        OPTIMAL,
        COLUMNS_FOUND,
        NO_COLUMNS,
        TIME_LIMIT,
        LABEL_LIMIT,
        ERROR
    };

    Status status = Status::NO_COLUMNS;
    std::vector<Label*> columns;
    double best_reduced_cost = std::numeric_limits<double>::infinity();
    size_t labels_created = 0;
    size_t labels_dominated = 0;
    double solve_time = 0.0;
    size_t iterations = 0;
};

/**
 * @brief High-performance labeling algorithm for SPPRC.
 *
 * This class implements the core labeling algorithm used in the pricing
 * subproblem of column generation. It's optimized for:
 * - Cache-efficient memory access
 * - Fast dominance checking
 * - Minimal memory allocations during solving
 * - Optional topological order processing for DAGs
 * - Optional ng-path relaxation for faster elementarity
 * - Configurable label limits per node
 */
class LabelingAlgorithm {
public:
    /**
     * @brief Construct the labeling algorithm.
     * @param network The network to solve on
     * @param num_resources Number of resources to track
     * @param resource_limits Upper bounds on resources
     * @param config Algorithm configuration
     */
    LabelingAlgorithm(const Network& network,
                      size_t num_resources,
                      const std::vector<double>& resource_limits,
                      const LabelingConfig& config = LabelingConfig())
        : network_(network)
        , num_resources_(num_resources)
        , resource_limits_(resource_limits)
        , config_(config)
        , label_pool_(network.num_nodes(), num_resources)
    {
        // Pre-size resource limits if needed
        if (resource_limits_.size() < num_resources_) {
            resource_limits_.resize(num_resources_, std::numeric_limits<double>::infinity());
        }

        // Pre-allocate arc reduced costs
        arc_reduced_costs_.resize(network.num_arcs(), 0.0);

        // Compute topological order if needed
        if (config_.use_topological_order) {
            compute_topological_order();
        }

        // Build ng-neighborhoods if using ng-path relaxation
        if (config_.use_ng_path) {
            build_ng_neighborhoods();
        }

        // Initialize label counts per node
        labels_at_node_.resize(network.num_nodes(), 0);
    }

    /**
     * @brief Set dual values for reduced cost computation.
     * @param dual_values Map from item_id to dual value
     */
    void set_dual_values(const std::unordered_map<int32_t, double>& dual_values) {
        // Precompute arc reduced costs
        for (const auto& arc : network_.arcs()) {
            double rc = arc.cost;
            for (int32_t item : arc.covered_items) {
                auto it = dual_values.find(item);
                if (it != dual_values.end()) {
                    rc -= it->second;
                }
            }
            arc_reduced_costs_[arc.index] = rc;
        }
    }

    /**
     * @brief Solve the SPPRC.
     * @return Result with found columns and statistics
     */
    LabelingResult solve() {
        if (config_.use_topological_order) {
            return solve_topological();
        } else {
            return solve_priority_queue();
        }
    }

    // Accessors
    const Network& network() const { return network_; }
    const LabelPool& label_pool() const { return label_pool_; }

private:
    /**
     * @brief Solve using priority queue (best-first search).
     * More flexible, supports early termination based on reduced cost.
     */
    LabelingResult solve_priority_queue() {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Clear previous state
        label_pool_.clear();
        std::fill(labels_at_node_.begin(), labels_at_node_.end(), 0);
        LabelingResult result;

        // Create source label
        int32_t source = network_.source_index();
        int32_t sink = network_.sink_index();

        if (source < 0 || sink < 0) {
            result.status = LabelingResult::Status::ERROR;
            return result;
        }

        Label* source_label = label_pool_.create_source_label(source);
        labels_at_node_[source] = 1;

        // Priority queue: min-heap by reduced cost
        std::vector<Label*> pq;
        pq.push_back(source_label);

        auto cmp = [](const Label* a, const Label* b) {
            return a->reduced_cost() > b->reduced_cost();  // Min-heap
        };

        size_t iterations = 0;

        while (!pq.empty()) {
            // Check limits
            if (config_.max_labels > 0 && label_pool_.total_created() >= config_.max_labels) {
                result.status = LabelingResult::Status::LABEL_LIMIT;
                break;
            }

            if (config_.max_time > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= config_.max_time) {
                    result.status = LabelingResult::Status::TIME_LIMIT;
                    break;
                }
            }

            // Pop best label
            std::pop_heap(pq.begin(), pq.end(), cmp);
            Label* label = pq.back();
            pq.pop_back();
            ++iterations;

            // Skip dominated labels (they were marked but kept for pointer safety)
            if (label->is_dominated()) {
                continue;
            }

            // Skip if at sink (collect at end)
            if (label->node_index() == sink) {
                continue;
            }

            // Extend along all outgoing arcs
            extend_label(label, pq, cmp);
        }

        return finalize_result(result, start_time, iterations, sink);
    }

    /**
     * @brief Solve using topological order (for DAGs).
     * More efficient for acyclic networks, processes each node exactly once.
     */
    LabelingResult solve_topological() {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Clear previous state
        label_pool_.clear();
        std::fill(labels_at_node_.begin(), labels_at_node_.end(), 0);
        LabelingResult result;

        int32_t source = network_.source_index();
        int32_t sink = network_.sink_index();

        if (source < 0 || sink < 0 || topological_order_.empty()) {
            result.status = LabelingResult::Status::ERROR;
            return result;
        }

        // Create source label
        Label* source_label = label_pool_.create_source_label(source);
        labels_at_node_[source] = 1;

        size_t iterations = 0;
        bool time_limit_reached = false;

        // Process nodes in topological order
        for (int32_t node : topological_order_) {
            // Check time limit periodically
            if (config_.max_time > 0 && (iterations % 1000 == 0)) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= config_.max_time) {
                    result.status = LabelingResult::Status::TIME_LIMIT;
                    time_limit_reached = true;
                    break;
                }
            }

            // Check total label limit
            if (config_.max_labels > 0 && label_pool_.total_created() >= config_.max_labels) {
                result.status = LabelingResult::Status::LABEL_LIMIT;
                break;
            }

            // Skip sink (collect at end)
            if (node == sink) continue;

            // Get all labels at this node
            auto node_labels = label_pool_.get_labels(node);

            // Extend each label
            for (Label* label : node_labels) {
                ++iterations;
                extend_label_topological(label);
            }
        }

        return finalize_result(result, start_time, iterations, sink);
    }

    /**
     * @brief Extend a label along all outgoing arcs (priority queue version).
     */
    template<typename Comparator>
    void extend_label(Label* label, std::vector<Label*>& pq, Comparator& cmp) {
        for (int32_t arc_idx : network_.outgoing_arcs(label->node_index())) {
            const Arc& arc = network_.arc(arc_idx);

            // Check elementarity
            if (!check_elementarity_ok(label, arc)) {
                continue;
            }

            // Check label limit at target node
            if (config_.max_labels_per_node > 0 &&
                labels_at_node_[arc.target] >= config_.max_labels_per_node) {
                continue;
            }

            // Handle time window constraints if enabled
            std::vector<double> resource_consumption = arc.resource_consumption;
            if (config_.use_time_windows) {
                // Get current time from label
                double current_time = label->resource(config_.time_resource_index);

                // Calculate arrival time
                double arrival_time = current_time + arc.travel_time;

                // Check if we can arrive before the latest time
                if (arrival_time > arc.latest) {
                    continue;  // Time window violation - skip this arc
                }

                // Apply waiting if we arrive early: new_time = max(arrival, earliest) + service_time
                double new_time = std::max(arrival_time, arc.earliest) + arc.service_time;

                // Update resource consumption for time resource
                // The consumption should be (new_time - current_time) so that
                // after adding to current_time we get new_time
                if (resource_consumption.size() <= config_.time_resource_index) {
                    resource_consumption.resize(config_.time_resource_index + 1, 0.0);
                }
                resource_consumption[config_.time_resource_index] = new_time - current_time;
            }

            // Extend label
            Label* new_label = label_pool_.extend_label(
                label,
                arc.target,
                arc.index,
                arc.cost,
                arc_reduced_costs_[arc.index],
                resource_consumption,
                resource_limits_,
                config_.check_dominance
            );

            if (new_label != nullptr) {
                // Add covered items
                for (int32_t item : arc.covered_items) {
                    new_label->add_covered_item(item);
                }

                labels_at_node_[arc.target]++;

                // Add to priority queue
                pq.push_back(new_label);
                std::push_heap(pq.begin(), pq.end(), cmp);
            }
        }
    }

    /**
     * @brief Extend a label along all outgoing arcs (topological version).
     */
    void extend_label_topological(Label* label) {
        for (int32_t arc_idx : network_.outgoing_arcs(label->node_index())) {
            const Arc& arc = network_.arc(arc_idx);

            // Check elementarity
            if (!check_elementarity_ok(label, arc)) {
                continue;
            }

            // Check label limit at target node
            if (config_.max_labels_per_node > 0 &&
                labels_at_node_[arc.target] >= config_.max_labels_per_node) {
                continue;
            }

            // Handle time window constraints if enabled
            std::vector<double> resource_consumption = arc.resource_consumption;
            if (config_.use_time_windows) {
                // Get current time from label
                double current_time = label->resource(config_.time_resource_index);

                // Calculate arrival time
                double arrival_time = current_time + arc.travel_time;

                // Check if we can arrive before the latest time
                if (arrival_time > arc.latest) {
                    continue;  // Time window violation - skip this arc
                }

                // Apply waiting if we arrive early: new_time = max(arrival, earliest) + service_time
                double new_time = std::max(arrival_time, arc.earliest) + arc.service_time;

                // Update resource consumption for time resource
                if (resource_consumption.size() <= config_.time_resource_index) {
                    resource_consumption.resize(config_.time_resource_index + 1, 0.0);
                }
                resource_consumption[config_.time_resource_index] = new_time - current_time;
            }

            // Extend label
            Label* new_label = label_pool_.extend_label(
                label,
                arc.target,
                arc.index,
                arc.cost,
                arc_reduced_costs_[arc.index],
                resource_consumption,
                resource_limits_,
                config_.check_dominance
            );

            if (new_label != nullptr) {
                // Add covered items
                for (int32_t item : arc.covered_items) {
                    new_label->add_covered_item(item);
                }

                labels_at_node_[arc.target]++;
            }
        }
    }

    /**
     * @brief Check if extending a label along an arc satisfies elementarity.
     */
    bool check_elementarity_ok(const Label* label, const Arc& arc) const {
        if (config_.use_ng_path) {
            // ng-path relaxation: only check items in ng-neighborhood
            return check_ng_elementarity(label, arc);
        } else if (config_.check_elementarity) {
            // Full elementarity check
            for (int32_t item : arc.covered_items) {
                if (label->covers_item(item)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Check ng-path elementarity.
     * Only forbids revisiting items that are in the ng-neighborhood of the target node.
     */
    bool check_ng_elementarity(const Label* label, const Arc& arc) const {
        if (arc.target >= static_cast<int32_t>(ng_neighborhoods_.size())) {
            return true;  // No neighborhood defined
        }

        const auto& neighborhood = ng_neighborhoods_[arc.target];

        for (int32_t item : arc.covered_items) {
            // Only check if item is in the ng-neighborhood of target
            if (neighborhood.count(item) > 0 && label->covers_item(item)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Finalize and return the result.
     */
    LabelingResult finalize_result(LabelingResult& result,
                                    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
                                    size_t iterations,
                                    int32_t sink) {
        // Collect columns from sink
        auto sink_labels = label_pool_.get_labels(sink);
        std::vector<Label*> candidates;

        for (Label* label : sink_labels) {
            // Skip dominated labels
            if (label->is_dominated()) {
                continue;
            }
            if (label->reduced_cost() < config_.rc_threshold) {
                candidates.push_back(label);
            }
        }

        // Sort by reduced cost
        std::sort(candidates.begin(), candidates.end(),
            [](const Label* a, const Label* b) {
                return a->reduced_cost() < b->reduced_cost();
            });

        // Apply limit
        if (config_.max_columns > 0 && candidates.size() > config_.max_columns) {
            candidates.resize(config_.max_columns);
        }

        // Build result
        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time = std::chrono::duration<double>(end_time - start_time).count();
        result.labels_created = label_pool_.total_created();
        result.labels_dominated = label_pool_.total_dominated();
        result.iterations = iterations;
        result.columns = std::move(candidates);

        if (!result.columns.empty()) {
            result.best_reduced_cost = result.columns[0]->reduced_cost();
            if (result.status == LabelingResult::Status::NO_COLUMNS) {
                result.status = LabelingResult::Status::COLUMNS_FOUND;
            }
        } else if (result.status == LabelingResult::Status::NO_COLUMNS) {
            // Keep NO_COLUMNS status
        }

        return result;
    }

    /**
     * @brief Compute topological order of the network (Kahn's algorithm).
     */
    void compute_topological_order() {
        size_t n = network_.num_nodes();
        std::vector<int32_t> in_degree(n, 0);

        // Compute in-degrees
        for (const auto& arc : network_.arcs()) {
            in_degree[arc.target]++;
        }

        // Start with nodes that have no incoming arcs
        std::queue<int32_t> q;
        for (size_t i = 0; i < n; ++i) {
            if (in_degree[i] == 0) {
                q.push(static_cast<int32_t>(i));
            }
        }

        topological_order_.clear();
        topological_order_.reserve(n);

        while (!q.empty()) {
            int32_t node = q.front();
            q.pop();
            topological_order_.push_back(node);

            for (int32_t arc_idx : network_.outgoing_arcs(node)) {
                int32_t target = network_.arc(arc_idx).target;
                if (--in_degree[target] == 0) {
                    q.push(target);
                }
            }
        }

        // Check if we have a valid topological order (no cycles)
        if (topological_order_.size() != n) {
            topological_order_.clear();  // Graph has cycles
        }
    }

    /**
     * @brief Build ng-neighborhoods for each node.
     *
     * For each node, the ng-neighborhood contains the k nearest nodes
     * based on graph distance or arc connectivity.
     */
    void build_ng_neighborhoods() {
        size_t n = network_.num_nodes();
        ng_neighborhoods_.resize(n);

        // Simple heuristic: for each node, include covered items from
        // nearby arcs (direct successors and predecessors)
        for (size_t node = 0; node < n; ++node) {
            std::unordered_set<int32_t>& neighborhood = ng_neighborhoods_[node];

            // Add items from outgoing arcs
            for (int32_t arc_idx : network_.outgoing_arcs(static_cast<int32_t>(node))) {
                const Arc& arc = network_.arc(arc_idx);
                for (int32_t item : arc.covered_items) {
                    neighborhood.insert(item);
                    if (neighborhood.size() >= config_.ng_neighborhood_size) break;
                }
                if (neighborhood.size() >= config_.ng_neighborhood_size) break;
            }

            // Add items from incoming arcs
            for (int32_t arc_idx : network_.incoming_arcs(static_cast<int32_t>(node))) {
                const Arc& arc = network_.arc(arc_idx);
                for (int32_t item : arc.covered_items) {
                    neighborhood.insert(item);
                    if (neighborhood.size() >= config_.ng_neighborhood_size) break;
                }
                if (neighborhood.size() >= config_.ng_neighborhood_size) break;
            }
        }
    }

    const Network& network_;
    size_t num_resources_;
    std::vector<double> resource_limits_;
    LabelingConfig config_;
    LabelPool label_pool_;
    std::vector<double> arc_reduced_costs_;

    // Topological order (computed once if use_topological_order is true)
    std::vector<int32_t> topological_order_;

    // ng-neighborhoods for ng-path relaxation
    std::vector<std::unordered_set<int32_t>> ng_neighborhoods_;

    // Label count per node (for max_labels_per_node limit)
    std::vector<size_t> labels_at_node_;
};

}  // namespace opencg
