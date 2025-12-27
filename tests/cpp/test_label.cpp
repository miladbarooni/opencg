/**
 * @file test_label.cpp
 * @brief Unit tests for Label and LabelPool classes.
 */

#include "core/label.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace opencg;

// Helper to check floating point equality
bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// =============================================================================
// Label Tests
// =============================================================================

void test_source_label_creation() {
    std::cout << "  test_source_label_creation... ";

    Label label(0, 3);  // Node 0, 3 resources

    assert(label.node_index() == 0);
    assert(approx_equal(label.cost(), 0.0));
    assert(approx_equal(label.reduced_cost(), 0.0));
    assert(label.predecessor() == nullptr);
    assert(label.is_source_label());
    assert(label.num_resources() == 3);
    assert(approx_equal(label.resource(0), 0.0));
    assert(approx_equal(label.resource(1), 0.0));
    assert(approx_equal(label.resource(2), 0.0));
    assert(label.num_covered() == 0);
    assert(label.path_length() == 0);

    std::cout << "OK" << std::endl;
}

void test_label_extension() {
    std::cout << "  test_label_extension... ";

    Label source(0, 2);

    // Extend to node 1
    std::vector<double> consumption = {1.0, 2.0};
    Label extended(1, &source, 0, 5.0, 3.0, consumption);

    assert(extended.node_index() == 1);
    assert(approx_equal(extended.cost(), 5.0));
    assert(approx_equal(extended.reduced_cost(), 3.0));
    assert(extended.predecessor() == &source);
    assert(!extended.is_source_label());
    assert(extended.last_arc_index() == 0);
    assert(approx_equal(extended.resource(0), 1.0));
    assert(approx_equal(extended.resource(1), 2.0));
    assert(extended.path_length() == 1);

    std::cout << "OK" << std::endl;
}

void test_label_chain() {
    std::cout << "  test_label_chain... ";

    Label source(0, 1);
    std::vector<double> consumption1 = {1.0};
    Label label1(1, &source, 0, 10.0, 5.0, consumption1);

    std::vector<double> consumption2 = {2.0};
    Label label2(2, &label1, 1, 15.0, 8.0, consumption2);

    // Check accumulated values
    assert(approx_equal(label2.cost(), 25.0));  // 10 + 15
    assert(approx_equal(label2.reduced_cost(), 13.0));  // 5 + 8
    assert(approx_equal(label2.resource(0), 3.0));  // 1 + 2
    assert(label2.path_length() == 2);

    // Check path reconstruction
    auto arcs = label2.get_arc_indices();
    assert(arcs.size() == 2);
    assert(arcs[0] == 0);
    assert(arcs[1] == 1);

    auto nodes = label2.get_node_indices();
    assert(nodes.size() == 3);
    assert(nodes[0] == 0);
    assert(nodes[1] == 1);
    assert(nodes[2] == 2);

    std::cout << "OK" << std::endl;
}

void test_covered_items() {
    std::cout << "  test_covered_items... ";

    Label label(0, 1);

    assert(!label.covers_item(1));
    assert(!label.covers_item(2));

    label.add_covered_item(1);
    assert(label.covers_item(1));
    assert(!label.covers_item(2));
    assert(label.num_covered() == 1);

    label.add_covered_item(2);
    label.add_covered_item(3);
    assert(label.covers_item(2));
    assert(label.covers_item(3));
    assert(label.num_covered() == 3);

    // Adding duplicate shouldn't increase count
    label.add_covered_item(1);
    assert(label.num_covered() == 3);

    std::cout << "OK" << std::endl;
}

void test_covered_items_inheritance() {
    std::cout << "  test_covered_items_inheritance... ";

    Label source(0, 1);
    source.add_covered_item(1);
    source.add_covered_item(2);

    std::vector<double> consumption = {1.0};
    Label extended(1, &source, 0, 5.0, 3.0, consumption);

    // Extended label should inherit covered items
    assert(extended.covers_item(1));
    assert(extended.covers_item(2));
    assert(extended.num_covered() == 2);

    // Adding new item to extended shouldn't affect source
    extended.add_covered_item(3);
    assert(extended.covers_item(3));
    assert(!source.covers_item(3));

    std::cout << "OK" << std::endl;
}

void test_label_comparison() {
    std::cout << "  test_label_comparison... ";

    Label label1(0, 1);
    Label label2(0, 1);

    // Modify reduced costs by extending
    std::vector<double> consumption = {0.0};
    Label ext1(1, &label1, 0, 0.0, 5.0, consumption);
    Label ext2(1, &label2, 0, 0.0, 10.0, consumption);

    // Note: operator< is reversed for min-heap (greater RC means "less than")
    assert(ext1 > ext2);  // ext1 has lower RC, so it's "greater" for min-heap
    assert(ext2 < ext1);

    std::cout << "OK" << std::endl;
}

// =============================================================================
// LabelPool Tests
// =============================================================================

void test_label_pool_creation() {
    std::cout << "  test_label_pool_creation... ";

    LabelPool pool(10, 2);

    assert(pool.num_nodes() == 10);
    assert(pool.total_labels() == 0);
    assert(pool.total_created() == 0);
    assert(pool.total_dominated() == 0);

    std::cout << "OK" << std::endl;
}

void test_label_pool_source_label() {
    std::cout << "  test_label_pool_source_label... ";

    LabelPool pool(5, 2);

    Label* source = pool.create_source_label(0);

    assert(source != nullptr);
    assert(source->node_index() == 0);
    assert(source->label_id() == 0);
    assert(pool.total_labels() == 1);
    assert(pool.total_created() == 1);

    // Create another source at different node
    Label* source2 = pool.create_source_label(3);
    assert(source2 != nullptr);
    assert(source2->node_index() == 3);
    assert(source2->label_id() == 1);
    assert(pool.total_labels() == 2);

    std::cout << "OK" << std::endl;
}

void test_label_pool_extend() {
    std::cout << "  test_label_pool_extend... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {10.0};

    Label* source = pool.create_source_label(0);

    std::vector<double> consumption = {3.0};
    Label* ext = pool.extend_label(source, 1, 0, 5.0, 2.0, consumption, limits);

    assert(ext != nullptr);
    assert(ext->node_index() == 1);
    assert(approx_equal(ext->cost(), 5.0));
    assert(approx_equal(ext->reduced_cost(), 2.0));
    assert(approx_equal(ext->resource(0), 3.0));
    assert(pool.total_labels() == 2);

    std::cout << "OK" << std::endl;
}

void test_label_pool_resource_infeasibility() {
    std::cout << "  test_label_pool_resource_infeasibility... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {5.0};

    Label* source = pool.create_source_label(0);

    // This extension exceeds resource limit
    std::vector<double> consumption = {6.0};
    Label* ext = pool.extend_label(source, 1, 0, 5.0, 2.0, consumption, limits);

    assert(ext == nullptr);  // Infeasible
    assert(pool.total_labels() == 1);  // Only source label

    std::cout << "OK" << std::endl;
}

void test_label_pool_dominance() {
    std::cout << "  test_label_pool_dominance... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {100.0};

    Label* source = pool.create_source_label(0);

    // Create first label at node 1
    std::vector<double> consumption1 = {3.0};
    Label* ext1 = pool.extend_label(source, 1, 0, 5.0, 2.0, consumption1, limits, true);
    assert(ext1 != nullptr);

    // Try to create dominated label (higher RC, higher resources)
    std::vector<double> consumption2 = {5.0};
    Label* ext2 = pool.extend_label(source, 1, 1, 7.0, 4.0, consumption2, limits, true);
    assert(ext2 == nullptr);  // Dominated by ext1
    assert(pool.total_dominated() == 1);

    // Create non-dominated label (lower RC, higher resources - trade-off)
    std::vector<double> consumption3 = {10.0};
    Label* ext3 = pool.extend_label(source, 1, 2, 3.0, 1.0, consumption3, limits, true);
    assert(ext3 != nullptr);  // Not dominated (lower RC)

    auto labels_at_1 = pool.get_labels(1);
    assert(labels_at_1.size() == 2);  // Both ext1 and ext3

    std::cout << "OK" << std::endl;
}

void test_label_pool_dominance_removal() {
    std::cout << "  test_label_pool_dominance_removal... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {100.0};

    Label* source = pool.create_source_label(0);

    // Create label with high RC and resources
    std::vector<double> consumption1 = {5.0};
    Label* ext1 = pool.extend_label(source, 1, 0, 10.0, 8.0, consumption1, limits, true);
    assert(ext1 != nullptr);

    // Create dominating label (lower RC, lower resources)
    std::vector<double> consumption2 = {2.0};
    Label* ext2 = pool.extend_label(source, 1, 1, 3.0, 2.0, consumption2, limits, true);
    assert(ext2 != nullptr);

    // ext1 should have been removed
    auto labels_at_1 = pool.get_labels(1);
    assert(labels_at_1.size() == 1);
    assert(labels_at_1[0] == ext2);

    std::cout << "OK" << std::endl;
}

void test_label_pool_no_dominance_check() {
    std::cout << "  test_label_pool_no_dominance_check... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {100.0};

    Label* source = pool.create_source_label(0);

    // Create two dominated labels with dominance check disabled
    std::vector<double> consumption1 = {3.0};
    Label* ext1 = pool.extend_label(source, 1, 0, 5.0, 2.0, consumption1, limits, false);

    std::vector<double> consumption2 = {5.0};
    Label* ext2 = pool.extend_label(source, 1, 1, 7.0, 4.0, consumption2, limits, false);

    assert(ext1 != nullptr);
    assert(ext2 != nullptr);  // Not pruned

    auto labels_at_1 = pool.get_labels(1);
    assert(labels_at_1.size() == 2);  // Both kept

    std::cout << "OK" << std::endl;
}

void test_label_pool_clear() {
    std::cout << "  test_label_pool_clear... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {100.0};

    pool.create_source_label(0);
    pool.create_source_label(1);

    assert(pool.total_labels() == 2);
    assert(pool.total_created() == 2);

    pool.clear();

    assert(pool.total_labels() == 0);
    assert(pool.total_created() == 0);
    assert(pool.total_dominated() == 0);

    std::cout << "OK" << std::endl;
}

void test_label_pool_get_labels() {
    std::cout << "  test_label_pool_get_labels... ";

    LabelPool pool(5, 1);
    std::vector<double> limits = {100.0};

    Label* source = pool.create_source_label(0);

    // Create multiple labels at same node (different resources)
    std::vector<double> consumption1 = {1.0};
    pool.extend_label(source, 2, 0, 5.0, 5.0, consumption1, limits, true);

    std::vector<double> consumption2 = {10.0};
    pool.extend_label(source, 2, 1, 2.0, 2.0, consumption2, limits, true);

    auto labels = pool.get_labels(2);
    assert(labels.size() == 2);

    // Check empty node
    auto empty_labels = pool.get_labels(3);
    assert(empty_labels.empty());

    // Check invalid node
    auto invalid_labels = pool.get_labels(100);
    assert(invalid_labels.empty());

    std::cout << "OK" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "Running Label tests..." << std::endl;

    // Label tests
    test_source_label_creation();
    test_label_extension();
    test_label_chain();
    test_covered_items();
    test_covered_items_inheritance();
    test_label_comparison();

    std::cout << "\nRunning LabelPool tests..." << std::endl;

    // LabelPool tests
    test_label_pool_creation();
    test_label_pool_source_label();
    test_label_pool_extend();
    test_label_pool_resource_infeasibility();
    test_label_pool_dominance();
    test_label_pool_dominance_removal();
    test_label_pool_no_dominance_check();
    test_label_pool_clear();
    test_label_pool_get_labels();

    std::cout << "\n✓ All Label tests passed!" << std::endl;
    return 0;
}
