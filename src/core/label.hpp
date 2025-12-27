/**
 * @file label.hpp
 * @brief Label class for SPPRC labeling algorithm.
 *
 * A Label represents a partial path from the source node to some intermediate
 * node, tracking cost, reduced cost, resource consumption, and covered items.
 */

#pragma once

#include <vector>
#include <set>
#include <memory>
#include <limits>
#include <cstdint>

namespace opencg {

/**
 * @brief A label representing a partial path in SPPRC.
 *
 * Labels are the core data structure in the labeling algorithm. Each label
 * represents a path from the source to a specific node, with associated
 * cost and resource consumption.
 *
 * This is a lightweight structure optimized for performance:
 * - Uses contiguous arrays for resource values
 * - Uses bitset for covered items (up to 64 items inline, heap for more)
 * - Predecessor is stored as pointer for O(1) path reconstruction
 */
class Label {
public:
    using ResourceValue = double;
    // Use std::set instead of std::unordered_set to avoid hash table overflow
    // issues (__next_prime overflow) that can occur with certain item ID patterns
    using CoveredSet = std::set<int32_t>;

    /**
     * @brief Construct a source label.
     * @param node_index Index of the source node
     * @param num_resources Number of resources to track
     */
    Label(int32_t node_index, size_t num_resources)
        : node_index_(node_index)
        , cost_(0.0)
        , reduced_cost_(0.0)
        , predecessor_(nullptr)
        , last_arc_index_(-1)
        , label_id_(-1)
        , resource_values_(num_resources, 0.0)
    {}

    /**
     * @brief Construct a label by extending a predecessor.
     * @param node_index Target node index
     * @param predecessor The predecessor label
     * @param arc_index Index of the arc used to extend
     * @param arc_cost Cost of the arc
     * @param arc_reduced_cost Reduced cost contribution of the arc
     * @param resource_consumption Resource consumption on the arc
     */
    Label(int32_t node_index,
          const Label* predecessor,
          int32_t arc_index,
          double arc_cost,
          double arc_reduced_cost,
          const std::vector<ResourceValue>& resource_consumption)
        : node_index_(node_index)
        , cost_(predecessor->cost_ + arc_cost)
        , reduced_cost_(predecessor->reduced_cost_ + arc_reduced_cost)
        , predecessor_(predecessor)
        , last_arc_index_(arc_index)
        , label_id_(-1)
        , resource_values_(predecessor->resource_values_)
        , covered_items_(predecessor->covered_items_)
    {
        // Add resource consumption with clamping at 0
        // Negative consumption (e.g., from overnight rest arcs) resets the resource
        for (size_t i = 0; i < resource_consumption.size() && i < resource_values_.size(); ++i) {
            resource_values_[i] += resource_consumption[i];
            // Clamp at 0 (resource can't go negative - this handles rest/reset arcs)
            if (resource_values_[i] < 0.0) {
                resource_values_[i] = 0.0;
            }
        }
    }

    // Accessors
    int32_t node_index() const { return node_index_; }
    double cost() const { return cost_; }
    double reduced_cost() const { return reduced_cost_; }
    const Label* predecessor() const { return predecessor_; }
    int32_t last_arc_index() const { return last_arc_index_; }
    int32_t label_id() const { return label_id_; }

    void set_label_id(int32_t id) { label_id_ = id; }

    // Resource access
    size_t num_resources() const { return resource_values_.size(); }

    ResourceValue resource(size_t index) const {
        return index < resource_values_.size() ? resource_values_[index] : 0.0;
    }

    void set_resource(size_t index, ResourceValue value) {
        if (index < resource_values_.size()) {
            resource_values_[index] = value;
        }
    }

    const std::vector<ResourceValue>& resource_values() const {
        return resource_values_;
    }

    // Covered items
    const CoveredSet& covered_items() const { return covered_items_; }

    bool covers_item(int32_t item) const {
        return covered_items_.find(item) != covered_items_.end();
    }

    void add_covered_item(int32_t item) {
        covered_items_.insert(item);
    }

    size_t num_covered() const { return covered_items_.size(); }

    // Path reconstruction
    bool is_source_label() const { return predecessor_ == nullptr; }

    size_t path_length() const {
        if (predecessor_ == nullptr) return 0;
        return predecessor_->path_length() + 1;
    }

    /**
     * @brief Get the arc indices forming the path.
     * @return Vector of arc indices from source to this label
     */
    std::vector<int32_t> get_arc_indices() const {
        std::vector<int32_t> path;
        const Label* current = this;
        while (current->predecessor_ != nullptr) {
            path.push_back(current->last_arc_index_);
            current = current->predecessor_;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    /**
     * @brief Get the node indices forming the path.
     * @return Vector of node indices from source to this label
     */
    std::vector<int32_t> get_node_indices() const {
        std::vector<int32_t> path;
        const Label* current = this;
        while (current != nullptr) {
            path.push_back(current->node_index_);
            current = current->predecessor_;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // Comparison for priority queue (by reduced cost, lower is better)
    bool operator<(const Label& other) const {
        return reduced_cost_ > other.reduced_cost_;  // Note: reversed for min-heap
    }

    bool operator>(const Label& other) const {
        return reduced_cost_ < other.reduced_cost_;
    }

    // Dominance flag - set when this label is dominated by another
    bool is_dominated() const { return dominated_; }
    void mark_dominated() { dominated_ = true; }

private:
    int32_t node_index_;
    double cost_;
    double reduced_cost_;
    const Label* predecessor_;
    int32_t last_arc_index_;
    int32_t label_id_;
    std::vector<ResourceValue> resource_values_;
    CoveredSet covered_items_;
    bool dominated_ = false;
};


/**
 * @brief Pool for managing labels at each node during SPPRC.
 *
 * Provides efficient storage and dominance checking for labels.
 */
class LabelPool {
public:
    explicit LabelPool(size_t num_nodes, size_t num_resources)
        : num_nodes_(num_nodes)
        , num_resources_(num_resources)
        , labels_(num_nodes)
        , total_created_(0)
        , total_dominated_(0)
        , next_id_(0)
    {}

    size_t num_nodes() const { return num_nodes_; }
    size_t total_labels() const {
        size_t count = 0;
        for (const auto& node_labels : labels_) {
            count += node_labels.size();
        }
        return count;
    }
    size_t total_created() const { return total_created_; }
    size_t total_dominated() const { return total_dominated_; }

    /**
     * @brief Create a new label and add it to the pool.
     * @return Pointer to the created label, or nullptr if dominated
     */
    Label* create_source_label(int32_t node_index) {
        auto label = std::make_unique<Label>(node_index, num_resources_);
        label->set_label_id(next_id_++);
        ++total_created_;

        Label* ptr = label.get();
        labels_[node_index].push_back(std::move(label));
        return ptr;
    }

    /**
     * @brief Extend a label along an arc.
     * @return Pointer to the new label, or nullptr if infeasible/dominated
     */
    Label* extend_label(const Label* predecessor,
                        int32_t target_node,
                        int32_t arc_index,
                        double arc_cost,
                        double arc_reduced_cost,
                        const std::vector<double>& resource_consumption,
                        const std::vector<double>& resource_limits,
                        bool check_dominance = true) {
        // Create new label
        auto new_label = std::make_unique<Label>(
            target_node, predecessor, arc_index,
            arc_cost, arc_reduced_cost, resource_consumption
        );

        // Check resource feasibility
        for (size_t i = 0; i < resource_limits.size() && i < new_label->num_resources(); ++i) {
            if (new_label->resource(i) > resource_limits[i]) {
                return nullptr;  // Infeasible
            }
        }

        ++total_created_;

        if (check_dominance) {
            // Check if dominated by existing labels (skip already-dominated labels)
            auto& node_labels = labels_[target_node];
            for (const auto& existing : node_labels) {
                if (!existing->is_dominated() && dominates(existing.get(), new_label.get())) {
                    ++total_dominated_;
                    return nullptr;
                }
            }

            // Mark labels dominated by new label (don't delete - they may be in priority queue)
            for (const auto& existing : node_labels) {
                if (!existing->is_dominated() && dominates(new_label.get(), existing.get())) {
                    existing->mark_dominated();
                    ++total_dominated_;
                }
            }
        }

        new_label->set_label_id(next_id_++);
        Label* ptr = new_label.get();
        labels_[target_node].push_back(std::move(new_label));
        return ptr;
    }

    /**
     * @brief Get all labels at a node.
     */
    std::vector<Label*> get_labels(int32_t node_index) const {
        std::vector<Label*> result;
        if (node_index >= 0 && static_cast<size_t>(node_index) < labels_.size()) {
            for (const auto& label : labels_[node_index]) {
                result.push_back(label.get());
            }
        }
        return result;
    }

    void clear() {
        for (auto& node_labels : labels_) {
            node_labels.clear();
        }
        total_created_ = 0;
        total_dominated_ = 0;
        next_id_ = 0;
    }

private:
    /**
     * @brief Check if label1 dominates label2.
     *
     * Label1 dominates label2 if:
     * - reduced_cost1 <= reduced_cost2
     * - All resource values of label1 <= label2
     * - covered_items1 ⊆ covered_items2
     */
    bool dominates(const Label* label1, const Label* label2) const {
        // Check reduced cost
        if (label1->reduced_cost() > label2->reduced_cost()) {
            return false;
        }

        // Check resources (all must be <=)
        for (size_t i = 0; i < num_resources_; ++i) {
            if (label1->resource(i) > label2->resource(i)) {
                return false;
            }
        }

        // Check covered items (subset)
        const auto& items1 = label1->covered_items();
        const auto& items2 = label2->covered_items();
        for (int32_t item : items1) {
            if (items2.find(item) == items2.end()) {
                return false;
            }
        }

        return true;
    }

    size_t num_nodes_;
    size_t num_resources_;
    std::vector<std::vector<std::unique_ptr<Label>>> labels_;
    size_t total_created_;
    size_t total_dominated_;
    int32_t next_id_;
};

}  // namespace opencg
