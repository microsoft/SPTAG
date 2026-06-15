#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/utils/hashset.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::symqg {
constexpr size_t kMaxBsIter = 5;  // max iter for binary search of pruning bar
using CandidateList = std::vector<AnnCandidate<float>>;

/**
 * @brief Builder of qg. Since we need to build the symphonyqg iteratively, which requires
 * to record a lot of temp data, we use a separate class as a builder for this purpose.
 *
 */
class QGBuilder {
   private:
    QuantizedGraph<float>& qg_;
    size_t ef_build_;      // size of search pool for indexing
    size_t num_threads_;   // number of threads used for indexing
    size_t num_nodes_;     // num of data points
    size_t dim_;           // dimension of data
    size_t degree_bound_;  // degree bound for qg, multiple of 32
    static constexpr size_t kMaxCandidatePoolSize =
        750;  // max num of candidates for indexing
    static constexpr size_t kMaxPrunedSize =
        300;                                    // max number of recorded pruned candidates
    std::vector<CandidateList> new_neighbors_;  // new neighbors for current iteration
    std::vector<CandidateList> pruned_neighbors_;    // recorded pruned neighbors
    std::vector<HashBasedBooleanSet> visited_list_;  // list of visited hash set
    std::vector<uint32_t> degrees_;                  // record degree of qg
    void random_init();
    void search_new_neighbors(bool refine);
    void heuristic_prune(PID, CandidateList&, CandidateList&, bool);
    void add_reverse_edges(bool);
    void add_pruned_edges(
        const CandidateList&, const CandidateList&, CandidateList&, float
    );
    void graph_refine();
    void iter(bool);

   public:
    explicit QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        const float* data,
        size_t num_threads = std::numeric_limits<size_t>::max()
    )
        : qg_{index}
        , ef_build_{ef_build}
        , num_threads_{std::min(num_threads, total_threads())}
        , num_nodes_{qg_.num_vertices()}
        , dim_{qg_.dimension()}
        , degree_bound_(qg_.degree_bound())
        , new_neighbors_(qg_.num_vertices())
        , pruned_neighbors_(qg_.num_vertices())
        , visited_list_(
              num_threads_,
              HashBasedBooleanSet(std::min(ef_build_ * ef_build_, num_nodes_ / 10))
          )
        , degrees_(qg_.num_vertices(), degree_bound_) {
        omp_set_num_threads(static_cast<int>(num_threads_));

        std::vector<float> centroid =
            compute_centroid(data, num_nodes_, dim_, num_threads_);

        PID entry_point = exact_nn(
            data, centroid.data(), num_nodes_, dim_, num_threads_, euclidean_sqr<float>
        );

        std::cout << "Setting entry_point to " << entry_point << '\n' << std::flush;

        qg_.set_ep(entry_point);
        qg_.copy_vectors(data);

        random_init();
    }

    void build(size_t num_iter = 3) {
        if (num_iter < 2) {
            std::cerr << "The number of iter for building qg should >= 3\n";
            exit(1);
        }
        // for first iterations, we do not need to refine the graph structure
        for (size_t i = 0; i < num_iter - 1; ++i) {
            iter(false);
        }
        iter(true);
    }

    [[nodiscard]] bool check_dup() const {
        std::atomic<bool> flag(false);
#pragma omp parallel for
        for (size_t i = 0; i < num_nodes_; ++i) {
            std::unordered_set<PID> edges;
            for (auto nei : new_neighbors_[i]) {
                if (edges.find(nei.id) != edges.end()) {
                    flag = true;
                }
                edges.emplace(nei.id);
            }
        }
        return flag;
    }

    [[nodiscard]] float avg_degree() const {
        size_t degrees = std::accumulate(degrees_.begin(), degrees_.end(), 0U);
        return static_cast<float>(degrees) / static_cast<float>(num_nodes_);
    }
};

inline void QGBuilder::add_pruned_edges(
    const CandidateList& result,
    const CandidateList& pruned_list,
    CandidateList& new_result,
    float threshold
) {
    size_t start = 0;
    new_result.clear();
    new_result = result;

    std::unordered_set<PID> nei_set;
    nei_set.reserve(degree_bound_);
    for (const auto& nei : result) {
        nei_set.emplace(nei.id);
    }

    while (new_result.size() < degree_bound_ && start < pruned_list.size()) {
        const auto& cur = pruned_list[start];
        bool occlude = false;
        const float* cur_data = qg_.get_vector(cur.id);
        float dik_sqr = cur.distance;

        if (nei_set.find(cur.id) != nei_set.end()) {
            occlude = true;
            break;
        }

        for (auto& nei : new_result) {
            float dij_sqr = nei.distance;
            if (dij_sqr > dik_sqr) {
                break;
            }
            float djk_sqr = qg_.raw_dist_func_(qg_.get_vector(nei.id), cur_data, dim_);
            float cosine =
                (dik_sqr + dij_sqr - djk_sqr) / (2 * std::sqrt(dij_sqr * dik_sqr));
            if (cosine > threshold) {
                occlude = true;
                break;
            }
        }

        if (!occlude) {
            new_result.emplace_back(cur);
            nei_set.emplace(cur.id);
            std::sort(new_result.begin(), new_result.end());
        }

        ++start;
    }
}

inline void QGBuilder::heuristic_prune(
    PID cur_id, CandidateList& pool, CandidateList& pruned_results, bool refine
) {
    if (pool.empty()) {
        return;
    }
    pruned_results.clear();
    size_t poolsize = pool.size();

    // if we dont have enough candidates, just keep all neighbors
    if (poolsize <= degree_bound_) {
        pruned_results = pool;
        return;
    }

    std::vector<bool> pruned(
        poolsize, false
    );                 // bool vector to record if this neighbor is pruned
    size_t start = 0;  // start position

    while (pruned_results.size() < degree_bound_ && start < poolsize) {
        auto candidate_id = pool[start].id;

        // if already pruned, move to next
        if (pruned[start]) {
            ++start;
            continue;
        }

        pruned_results.emplace_back(pool[start]);  // add current candidate to result
        const float* data_j = qg_.get_vector(candidate_id);

        // i : current vertex
        // j : neighbor added in this iter
        // k : remained unpruned candidate neighbor
        for (size_t k = start + 1; k < poolsize; ++k) {
            if (pruned[k]) {
                continue;
            }
            float dik = pool[k].distance;
            auto djk = qg_.raw_dist_func_(data_j, qg_.get_vector(pool[k].id), dim_);

            if (djk < dik) {
                if (refine && pruned_neighbors_[cur_id].size() < kMaxPrunedSize) {
                    pruned_neighbors_[cur_id].emplace_back(pool[k]);
                }
                pruned[k] = true;
            }
        }

        ++start;
    }
}

/**
 * @brief search for new neighbor in qg
 *
 * @param refine refine = true means recording pruned candidates
 */
inline void QGBuilder::search_new_neighbors(bool refine) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        PID cur_id = i;
        auto tid = omp_get_thread_num();
        CandidateList candidates;
        HashBasedBooleanSet& vis = visited_list_[tid];
        candidates.reserve(2 * kMaxCandidatePoolSize);
        vis.clear();
        qg_.find_candidates(cur_id, ef_build_, candidates, vis, degrees_);

        // add current neighbors
        for (auto& nei : new_neighbors_[cur_id]) {
            auto neighbor_id = nei.id;
            if (neighbor_id != cur_id && !vis.get(neighbor_id)) {
                candidates.emplace_back(nei);
            }
        }

        size_t min_size = std::min(candidates.size(), kMaxCandidatePoolSize);
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<long>(min_size),
            candidates.end()
        );
        candidates.resize(min_size);

        // prune and update qg
        heuristic_prune(cur_id, candidates, new_neighbors_[cur_id], refine);
    }
}

inline void QGBuilder::add_reverse_edges(bool refine) {
    std::vector<std::mutex> locks(num_nodes_);
    std::vector<CandidateList> reverse_buffer(num_nodes_);

#pragma omp parallel for schedule(dynamic)
    for (PID data_id = 0; data_id < num_nodes_; ++data_id) {
        for (const auto& nei : new_neighbors_[data_id]) {
            PID dst = nei.id;
            bool dup = false;
            CandidateList& dst_neighbors = new_neighbors_[dst];
            std::lock_guard lock(locks[dst]);
            for (auto& dst_nei : dst_neighbors) {
                if (dst_nei.id == data_id) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }

            if (dst_neighbors.size() < degree_bound_) {
                dst_neighbors.emplace_back(data_id, nei.distance);
            } else {
                if (reverse_buffer[dst].size() < kMaxCandidatePoolSize) {
                    reverse_buffer[dst].emplace_back(data_id, nei.distance);
                }
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (PID data_id = 0; data_id < num_nodes_; ++data_id) {
        CandidateList& tmp_pool = reverse_buffer[data_id];
        tmp_pool.reserve(tmp_pool.size() + degree_bound_);
        tmp_pool.insert(
            tmp_pool.end(), new_neighbors_[data_id].begin(), new_neighbors_[data_id].end()
        );
        std::sort(tmp_pool.begin(), tmp_pool.end());
        heuristic_prune(data_id, tmp_pool, new_neighbors_[data_id], refine);
    }
}

inline void QGBuilder::random_init() {
    const PID min_id = 0;
    const PID max_id = num_nodes_ - 1;
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        std::unordered_set<PID> neighbor_set;
        neighbor_set.reserve(degree_bound_);
        while (neighbor_set.size() < degree_bound_) {
            PID rand_id = rand_integer<PID>(min_id, max_id);
            if (rand_id != i) {
                neighbor_set.emplace(rand_id);
            }
        }

        const float* cur_data = qg_.get_vector(i);
        new_neighbors_[i].reserve(degree_bound_);
        for (PID cur_neigh : neighbor_set) {
            new_neighbors_[i].emplace_back(
                cur_neigh, qg_.raw_dist_func_(cur_data, qg_.get_vector(cur_neigh), dim_)
            );
        }

        degrees_[i] = new_neighbors_[i].size();
        qg_.update_qg(i, new_neighbors_[i]);
    }
}

/**
 * @brief refine the graph structure, make sure the degree for each vertex in qg equals the
 * degree bound (multiple of 32)
 *
 */
inline void QGBuilder::graph_refine() {
    std::cout << "Supplementing edges...\n";

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        CandidateList& cur_neighbors = new_neighbors_[i];
        size_t cur_degree = cur_neighbors.size();

        // skip vertices with enough neighbors
        if (cur_degree >= degree_bound_) {
            continue;
        }

        CandidateList& pruned_list = pruned_neighbors_[i];
        CandidateList new_result;
        new_result.reserve(degree_bound_);

        std::sort(pruned_list.begin(), pruned_list.end());

        // use binary search to get refined results
        float left = 0.5;
        float right = 1.0;
        size_t iter = 0;
        while (iter++ < kMaxBsIter) {
            float mid = (left + right) / 2;
            add_pruned_edges(cur_neighbors, pruned_list, new_result, mid);
            if (new_result.size() < degree_bound_) {
                left = mid;
            } else {
                right = mid;
            }
        }

        // update neighbors with larger cosine value since we want to retain more edges
        add_pruned_edges(cur_neighbors, pruned_list, new_result, right);

        // if the vertex still doesn't have enough neighbors, use random vertices
        if (new_result.size() < degree_bound_) {
            std::unordered_set<PID> ids;
            ids.reserve(degree_bound_);
            for (auto& neighbor : new_result) {
                ids.emplace(neighbor.id);
            }
            while (new_result.size() < degree_bound_) {
                PID rand_id = rand_integer<PID>(0, static_cast<PID>(num_nodes_) - 1);
                if (rand_id != static_cast<PID>(i) && ids.find(rand_id) == ids.end()) {
                    new_result.emplace_back(
                        rand_id,
                        qg_.raw_dist_func_(qg_.get_vector(rand_id), qg_.get_vector(i), dim_)
                    );
                    ids.emplace(rand_id);
                }
            }
        }

        cur_neighbors = new_result;
    }
    std::cout << "Supplementing finished...\n";
}

inline void QGBuilder::iter(bool refine) {
    if (refine) {
        for (size_t i = 0; i < num_nodes_; ++i) {
            pruned_neighbors_[i].clear();
            pruned_neighbors_[i].reserve(kMaxPrunedSize);
        }
    }

    search_new_neighbors(refine);

    add_reverse_edges(refine);

    // Use pruned edges to refine graph
    if (refine) {
        graph_refine();
    }

    // update qg
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
        qg_.update_qg(i, new_neighbors_[i]);
        degrees_[i] = new_neighbors_[i].size();
    }
}
}  // namespace rabitqlib::symqg
