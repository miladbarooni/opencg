/**
 * @file parallel_labeling.hpp
 * @brief Parallel SPPRC labeling for multi-source pricing.
 *
 * This file provides a parallel implementation for solving multiple
 * independent labeling problems concurrently. This is particularly
 * useful for per-source pricing in crew pairing and similar problems
 * where we need to run labeling from many different source nodes.
 *
 * Uses C++17 parallel algorithms or manual threading for parallelism.
 */

#pragma once

#include "labeling.hpp"

#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <functional>

namespace opencg {

/**
 * @brief Result from parallel labeling across multiple sources.
 */
struct ParallelLabelingResult {
    std::vector<Label*> all_columns;
    double best_reduced_cost = std::numeric_limits<double>::infinity();
    size_t total_labels_created = 0;
    size_t total_labels_dominated = 0;
    double solve_time = 0.0;
    size_t sources_processed = 0;
};

/**
 * @brief Configuration for parallel labeling.
 */
struct ParallelLabelingConfig {
    size_t num_threads = 0;          // 0 = auto (use hardware concurrency)
    size_t max_total_columns = 0;    // 0 = unlimited
    double max_total_time = 0.0;     // 0 = unlimited (seconds)
    bool collect_all_columns = true; // If false, only track best per source
};

/**
 * @brief Parallel labeling solver for multiple independent networks.
 *
 * This class manages multiple LabelingAlgorithm instances and solves
 * them in parallel using a thread pool. Each source network is solved
 * independently, and results are aggregated.
 *
 * Usage:
 *     ParallelLabelingSolver solver(num_threads);
 *     solver.add_source(network1, resources, limits, config);
 *     solver.add_source(network2, resources, limits, config);
 *     // ... add more sources
 *     solver.set_dual_values(duals);
 *     auto result = solver.solve();
 */
class ParallelLabelingSolver {
public:
    /**
     * @brief Construct a parallel labeling solver.
     * @param config Parallel configuration
     */
    explicit ParallelLabelingSolver(const ParallelLabelingConfig& config = ParallelLabelingConfig())
        : config_(config)
    {
        if (config_.num_threads == 0) {
            config_.num_threads = std::thread::hardware_concurrency();
            if (config_.num_threads == 0) {
                config_.num_threads = 1;  // Fallback
            }
        }
    }

    /**
     * @brief Add a source network to be solved.
     * @param network The network for this source
     * @param num_resources Number of resources
     * @param resource_limits Resource upper bounds
     * @param labeling_config Configuration for labeling
     * @param source_id Optional identifier for this source
     */
    void add_source(
        const Network& network,
        size_t num_resources,
        const std::vector<double>& resource_limits,
        const LabelingConfig& labeling_config,
        int32_t source_id = -1
    ) {
        sources_.emplace_back(SourceData{
            std::make_unique<LabelingAlgorithm>(network, num_resources, resource_limits, labeling_config),
            source_id >= 0 ? source_id : static_cast<int32_t>(sources_.size())
        });
    }

    /**
     * @brief Set dual values for all source algorithms.
     * @param dual_values Map from item_id to dual value
     */
    void set_dual_values(const std::unordered_map<int32_t, double>& dual_values) {
        dual_values_ = dual_values;
        for (auto& source : sources_) {
            source.algorithm->set_dual_values(dual_values);
        }
    }

    /**
     * @brief Solve all sources in parallel.
     * @return Aggregated result from all sources
     */
    ParallelLabelingResult solve() {
        auto start_time = std::chrono::high_resolution_clock::now();

        ParallelLabelingResult result;

        if (sources_.empty()) {
            return result;
        }

        // For single thread or single source, just run sequentially
        if (config_.num_threads == 1 || sources_.size() == 1) {
            return solve_sequential();
        }

        return solve_parallel();
    }

    /**
     * @brief Get number of sources added.
     */
    size_t num_sources() const { return sources_.size(); }

    /**
     * @brief Get number of threads configured.
     */
    size_t num_threads() const { return config_.num_threads; }

    /**
     * @brief Clear all sources.
     */
    void clear() {
        sources_.clear();
    }

private:
    struct SourceData {
        std::unique_ptr<LabelingAlgorithm> algorithm;
        int32_t source_id;
    };

    struct SourceResult {
        std::vector<Label*> columns;
        size_t labels_created = 0;
        size_t labels_dominated = 0;
        double best_rc = std::numeric_limits<double>::infinity();
    };

    /**
     * @brief Solve a single source and return result.
     */
    SourceResult solve_single_source(size_t source_idx) {
        SourceResult result;

        if (source_idx >= sources_.size()) {
            return result;
        }

        auto& source = sources_[source_idx];
        LabelingResult lr = source.algorithm->solve();

        result.columns = std::move(lr.columns);
        result.labels_created = lr.labels_created;
        result.labels_dominated = lr.labels_dominated;
        result.best_rc = lr.best_reduced_cost;

        return result;
    }

    /**
     * @brief Sequential solve (single-threaded).
     */
    ParallelLabelingResult solve_sequential() {
        auto start_time = std::chrono::high_resolution_clock::now();
        ParallelLabelingResult result;

        for (size_t i = 0; i < sources_.size(); ++i) {
            // Check time limit
            if (config_.max_total_time > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= config_.max_total_time) {
                    break;
                }
            }

            // Check column limit
            if (config_.max_total_columns > 0 &&
                result.all_columns.size() >= config_.max_total_columns) {
                break;
            }

            SourceResult sr = solve_single_source(i);
            result.sources_processed++;
            result.total_labels_created += sr.labels_created;
            result.total_labels_dominated += sr.labels_dominated;

            if (sr.best_rc < result.best_reduced_cost) {
                result.best_reduced_cost = sr.best_rc;
            }

            if (config_.collect_all_columns) {
                result.all_columns.insert(
                    result.all_columns.end(),
                    sr.columns.begin(),
                    sr.columns.end()
                );
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

        return result;
    }

    /**
     * @brief Parallel solve using std::async.
     */
    ParallelLabelingResult solve_parallel() {
        auto start_time = std::chrono::high_resolution_clock::now();
        ParallelLabelingResult result;

        // Mutex for thread-safe result aggregation
        std::mutex result_mutex;
        std::atomic<bool> should_stop{false};
        std::atomic<size_t> column_count{0};

        // Process sources in batches using thread pool pattern
        const size_t batch_size = config_.num_threads * 2;
        size_t processed = 0;

        while (processed < sources_.size() && !should_stop.load()) {
            // Check time limit
            if (config_.max_total_time > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                if (elapsed >= config_.max_total_time) {
                    break;
                }
            }

            // Check column limit
            if (config_.max_total_columns > 0 &&
                column_count.load() >= config_.max_total_columns) {
                break;
            }

            // Determine batch range
            size_t batch_end = std::min(processed + batch_size, sources_.size());
            std::vector<std::future<SourceResult>> futures;
            futures.reserve(batch_end - processed);

            // Launch async tasks for this batch
            for (size_t i = processed; i < batch_end; ++i) {
                futures.push_back(std::async(
                    std::launch::async,
                    [this, i]() { return this->solve_single_source(i); }
                ));
            }

            // Collect results from this batch
            for (auto& future : futures) {
                SourceResult sr = future.get();

                std::lock_guard<std::mutex> lock(result_mutex);
                result.sources_processed++;
                result.total_labels_created += sr.labels_created;
                result.total_labels_dominated += sr.labels_dominated;

                if (sr.best_rc < result.best_reduced_cost) {
                    result.best_reduced_cost = sr.best_rc;
                }

                if (config_.collect_all_columns) {
                    result.all_columns.insert(
                        result.all_columns.end(),
                        sr.columns.begin(),
                        sr.columns.end()
                    );
                    column_count.store(result.all_columns.size());
                }
            }

            processed = batch_end;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

        return result;
    }

    ParallelLabelingConfig config_;
    std::vector<SourceData> sources_;
    std::unordered_map<int32_t, double> dual_values_;
};

}  // namespace opencg
