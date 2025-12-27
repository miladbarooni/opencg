/**
 * @file test_pricing.cpp
 * @brief Unit tests for Network and LabelingAlgorithm classes.
 */

#include "core/network.hpp"
#include "pricing/labeling.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace opencg;

// Helper to check floating point equality
bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// =============================================================================
// Network Tests
// =============================================================================

void test_network_creation() {
    std::cout << "  test_network_creation... ";

    Network net;

    assert(net.num_nodes() == 0);
    assert(net.num_arcs() == 0);
    assert(net.source_index() == -1);
    assert(net.sink_index() == -1);

    std::cout << "OK" << std::endl;
}

void test_network_add_nodes() {
    std::cout << "  test_network_add_nodes... ";

    Network net;

    int32_t source = net.add_source();
    assert(source == 0);
    assert(net.source_index() == 0);
    assert(net.num_nodes() == 1);

    int32_t n1 = net.add_node();
    assert(n1 == 1);
    assert(net.num_nodes() == 2);

    int32_t n2 = net.add_node();
    assert(n2 == 2);

    int32_t sink = net.add_sink();
    assert(sink == 3);
    assert(net.sink_index() == 3);
    assert(net.num_nodes() == 4);

    // Check node properties
    assert(net.node(source).is_source);
    assert(!net.node(source).is_sink);
    assert(!net.node(n1).is_source);
    assert(!net.node(n1).is_sink);
    assert(net.node(sink).is_sink);

    std::cout << "OK" << std::endl;
}

void test_network_add_arcs() {
    std::cout << "  test_network_add_arcs... ";

    Network net;
    int32_t source = net.add_source();
    int32_t a = net.add_node();
    int32_t sink = net.add_sink();

    int32_t arc1 = net.add_arc(source, a, 5.0, {1.0, 2.0}, {1, 2});
    int32_t arc2 = net.add_arc(a, sink, 3.0, {2.0, 1.0}, {3});

    assert(net.num_arcs() == 2);
    assert(arc1 == 0);
    assert(arc2 == 1);

    // Check arc properties
    const Arc& a1 = net.arc(arc1);
    assert(a1.source == source);
    assert(a1.target == a);
    assert(approx_equal(a1.cost, 5.0));
    assert(a1.resource_consumption.size() == 2);
    assert(approx_equal(a1.resource_consumption[0], 1.0));
    assert(approx_equal(a1.resource_consumption[1], 2.0));
    assert(a1.covered_items.size() == 2);
    assert(a1.covered_items[0] == 1);
    assert(a1.covered_items[1] == 2);

    std::cout << "OK" << std::endl;
}

void test_network_adjacency() {
    std::cout << "  test_network_adjacency... ";

    Network net;
    int32_t source = net.add_source();
    int32_t a = net.add_node();
    int32_t b = net.add_node();
    int32_t sink = net.add_sink();

    net.add_arc(source, a, 1.0);
    net.add_arc(source, b, 2.0);
    net.add_arc(a, sink, 3.0);
    net.add_arc(b, sink, 4.0);

    // Check outgoing
    auto source_out = net.outgoing_arcs(source);
    assert(source_out.size() == 2);

    auto a_out = net.outgoing_arcs(a);
    assert(a_out.size() == 1);

    auto sink_out = net.outgoing_arcs(sink);
    assert(sink_out.empty());

    // Check incoming
    auto source_in = net.incoming_arcs(source);
    assert(source_in.empty());

    auto sink_in = net.incoming_arcs(sink);
    assert(sink_in.size() == 2);

    std::cout << "OK" << std::endl;
}

void test_arc_get_consumption() {
    std::cout << "  test_arc_get_consumption... ";

    Network net;
    int32_t source = net.add_source();
    int32_t sink = net.add_sink();

    net.add_arc(source, sink, 1.0, {1.0, 2.0, 3.0});

    const Arc& arc = net.arc(0);
    assert(approx_equal(arc.get_consumption(0), 1.0));
    assert(approx_equal(arc.get_consumption(1), 2.0));
    assert(approx_equal(arc.get_consumption(2), 3.0));
    assert(approx_equal(arc.get_consumption(5), 0.0));  // Out of bounds

    std::cout << "OK" << std::endl;
}

// =============================================================================
// LabelingConfig Tests
// =============================================================================

void test_labeling_config_defaults() {
    std::cout << "  test_labeling_config_defaults... ";

    LabelingConfig config;

    assert(config.max_labels == 0);
    assert(approx_equal(config.max_time, 0.0));
    assert(config.max_columns == 0);
    assert(config.rc_threshold < 0);
    assert(config.check_dominance == true);
    assert(config.check_elementarity == false);

    std::cout << "OK" << std::endl;
}

void test_labeling_config_custom() {
    std::cout << "  test_labeling_config_custom... ";

    LabelingConfig config;
    config.max_labels = 1000;
    config.max_time = 60.0;
    config.max_columns = 10;
    config.rc_threshold = -0.001;
    config.check_dominance = false;
    config.check_elementarity = true;

    assert(config.max_labels == 1000);
    assert(approx_equal(config.max_time, 60.0));
    assert(config.max_columns == 10);
    assert(config.check_elementarity == true);

    std::cout << "OK" << std::endl;
}

// =============================================================================
// LabelingAlgorithm Tests
// =============================================================================

Network create_simple_network() {
    // Network: Source -> A -> Sink
    //                 -> B ->
    Network net;
    int32_t source = net.add_source();
    int32_t a = net.add_node();
    int32_t b = net.add_node();
    int32_t sink = net.add_sink();

    net.add_arc(source, a, 5.0, {1.0}, {1});   // Arc 0
    net.add_arc(source, b, 3.0, {2.0}, {2});   // Arc 1
    net.add_arc(a, sink, 2.0, {1.0}, {});      // Arc 2
    net.add_arc(b, sink, 4.0, {1.0}, {});      // Arc 3

    return net;
}

Network create_diamond_network() {
    // Network:     A
    //           /     \
    //  Source         Sink
    //           \     /
    //              B
    Network net;
    int32_t source = net.add_source();
    int32_t a = net.add_node();
    int32_t b = net.add_node();
    int32_t sink = net.add_sink();

    net.add_arc(source, a, 2.0, {1.0}, {1});   // Arc 0
    net.add_arc(source, b, 3.0, {1.0}, {2});   // Arc 1
    net.add_arc(a, sink, 4.0, {2.0}, {});      // Arc 2
    net.add_arc(b, sink, 3.0, {2.0}, {});      // Arc 3

    return net;
}

void test_labeling_algorithm_creation() {
    std::cout << "  test_labeling_algorithm_creation... ";

    Network net = create_simple_network();
    std::vector<double> limits = {10.0};

    LabelingAlgorithm algo(net, 1, limits);

    assert(&algo.network() == &net);

    std::cout << "OK" << std::endl;
}

void test_labeling_algorithm_solve_no_duals() {
    std::cout << "  test_labeling_algorithm_solve_no_duals... ";

    Network net = create_simple_network();
    std::vector<double> limits = {10.0};

    LabelingAlgorithm algo(net, 1, limits);
    LabelingResult result = algo.solve();

    // Without duals, reduced cost = cost
    // Path through A: 5 + 2 = 7
    // Path through B: 3 + 4 = 7
    // Neither has negative RC, so no columns found
    assert(result.status == LabelingResult::Status::NO_COLUMNS);
    assert(result.columns.empty());
    assert(result.labels_created > 0);

    std::cout << "OK" << std::endl;
}

void test_labeling_algorithm_solve_with_duals() {
    std::cout << "  test_labeling_algorithm_solve_with_duals... ";

    Network net = create_simple_network();
    std::vector<double> limits = {10.0};

    LabelingAlgorithm algo(net, 1, limits);

    // Set dual value for item 1 (on arc 0: source -> a)
    std::unordered_map<int32_t, double> duals;
    duals[1] = 10.0;  // High dual reduces cost of path through A

    algo.set_dual_values(duals);
    LabelingResult result = algo.solve();

    // Path through A: cost 5+2=7, dual 10 => RC = -3 (negative!)
    // Path through B: cost 3+4=7, dual 0 => RC = 7
    assert(result.status == LabelingResult::Status::COLUMNS_FOUND);
    assert(!result.columns.empty());
    assert(result.best_reduced_cost < 0);

    std::cout << "OK" << std::endl;
}

void test_labeling_algorithm_resource_limit() {
    std::cout << "  test_labeling_algorithm_resource_limit... ";

    Network net = create_simple_network();
    std::vector<double> limits = {1.5};  // Tight resource limit

    LabelingAlgorithm algo(net, 1, limits);

    std::unordered_map<int32_t, double> duals;
    duals[1] = 100.0;
    duals[2] = 100.0;

    algo.set_dual_values(duals);
    LabelingResult result = algo.solve();

    // Resource limit is too tight for any path
    // Path A needs 1+1=2 resources
    // Path B needs 2+1=3 resources
    assert(result.status == LabelingResult::Status::NO_COLUMNS);

    std::cout << "OK" << std::endl;
}

void test_labeling_algorithm_max_columns() {
    std::cout << "  test_labeling_algorithm_max_columns... ";

    Network net = create_diamond_network();
    std::vector<double> limits = {10.0};

    LabelingConfig config;
    config.max_columns = 1;

    LabelingAlgorithm algo(net, 1, limits, config);

    std::unordered_map<int32_t, double> duals;
    duals[1] = 10.0;
    duals[2] = 10.0;

    algo.set_dual_values(duals);
    LabelingResult result = algo.solve();

    // Both paths have negative RC, but we limit to 1
    assert(result.columns.size() <= 1);

    std::cout << "OK" << std::endl;
}

void test_labeling_algorithm_label_limit() {
    std::cout << "  test_labeling_algorithm_label_limit... ";

    Network net = create_diamond_network();
    std::vector<double> limits = {10.0};

    LabelingConfig config;
    config.max_labels = 2;  // Very low limit

    LabelingAlgorithm algo(net, 1, limits, config);
    LabelingResult result = algo.solve();

    assert(result.status == LabelingResult::Status::LABEL_LIMIT ||
           result.status == LabelingResult::Status::NO_COLUMNS);
    assert(result.labels_created <= 3);  // Source + at most 2

    std::cout << "OK" << std::endl;
}

void test_labeling_algorithm_statistics() {
    std::cout << "  test_labeling_algorithm_statistics... ";

    Network net = create_simple_network();
    std::vector<double> limits = {10.0};

    LabelingAlgorithm algo(net, 1, limits);

    std::unordered_map<int32_t, double> duals;
    duals[1] = 10.0;
    algo.set_dual_values(duals);

    LabelingResult result = algo.solve();

    assert(result.solve_time >= 0);
    assert(result.labels_created > 0);
    assert(result.iterations > 0);

    std::cout << "OK" << std::endl;
}

void test_labeling_result_status() {
    std::cout << "  test_labeling_result_status... ";

    // Test all status values exist
    LabelingResult r1; r1.status = LabelingResult::Status::OPTIMAL;
    LabelingResult r2; r2.status = LabelingResult::Status::COLUMNS_FOUND;
    LabelingResult r3; r3.status = LabelingResult::Status::NO_COLUMNS;
    LabelingResult r4; r4.status = LabelingResult::Status::TIME_LIMIT;
    LabelingResult r5; r5.status = LabelingResult::Status::LABEL_LIMIT;
    LabelingResult r6; r6.status = LabelingResult::Status::ERROR;

    assert(r1.status != r2.status);
    assert(r3.status != r4.status);

    std::cout << "OK" << std::endl;
}

void test_labeling_no_dominance() {
    std::cout << "  test_labeling_no_dominance... ";

    Network net = create_diamond_network();
    std::vector<double> limits = {10.0};

    LabelingConfig config;
    config.check_dominance = false;

    LabelingAlgorithm algo(net, 1, limits, config);

    std::unordered_map<int32_t, double> duals;
    duals[1] = 10.0;
    duals[2] = 10.0;

    algo.set_dual_values(duals);
    LabelingResult result = algo.solve();

    // Without dominance, should still find columns
    assert(result.labels_created > 0);

    std::cout << "OK" << std::endl;
}

void test_labeling_empty_network() {
    std::cout << "  test_labeling_empty_network... ";

    Network net;  // No nodes
    std::vector<double> limits = {10.0};

    LabelingAlgorithm algo(net, 1, limits);
    LabelingResult result = algo.solve();

    assert(result.status == LabelingResult::Status::ERROR);

    std::cout << "OK" << std::endl;
}

void test_labeling_no_path() {
    std::cout << "  test_labeling_no_path... ";

    Network net;
    int32_t source = net.add_source();
    int32_t a = net.add_node();
    net.add_sink();
    // No arcs connecting source to sink

    std::vector<double> limits = {10.0};

    LabelingAlgorithm algo(net, 1, limits);

    std::unordered_map<int32_t, double> duals;
    algo.set_dual_values(duals);

    LabelingResult result = algo.solve();

    assert(result.status == LabelingResult::Status::NO_COLUMNS);

    std::cout << "OK" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "Running Network tests..." << std::endl;

    test_network_creation();
    test_network_add_nodes();
    test_network_add_arcs();
    test_network_adjacency();
    test_arc_get_consumption();

    std::cout << "\nRunning LabelingConfig tests..." << std::endl;

    test_labeling_config_defaults();
    test_labeling_config_custom();

    std::cout << "\nRunning LabelingAlgorithm tests..." << std::endl;

    test_labeling_algorithm_creation();
    test_labeling_algorithm_solve_no_duals();
    test_labeling_algorithm_solve_with_duals();
    test_labeling_algorithm_resource_limit();
    test_labeling_algorithm_max_columns();
    test_labeling_algorithm_label_limit();
    test_labeling_algorithm_statistics();
    test_labeling_result_status();
    test_labeling_no_dominance();
    test_labeling_empty_network();
    test_labeling_no_path();

    std::cout << "\n✓ All Pricing tests passed!" << std::endl;
    return 0;
}
