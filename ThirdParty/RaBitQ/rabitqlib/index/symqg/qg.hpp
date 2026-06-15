#pragma once

#include <omp.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/array.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/hashset.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/visited_pool.hpp"

namespace rabitqlib::symqg {

template <typename T = float>
class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;                           // num points
    size_t degree_bound_ = 0;                         // degree bound
    size_t dim_ = 0;                                  // dimension
    size_t padded_dim_ = 0;                           // padded dimension
    T (*raw_dist_func_)(const T*, const T*, size_t);  // dist func for raw vector
    PID entry_point_ = 0;                             // Entry point of graph
    MetricType metric_type_ = MetricType::METRIC_L2;
    RotatorType rotator_type_ = RotatorType::FhtKacRotator;

    Array<
        char,
        std::vector<size_t>,
        memory::AlignedAllocator<
            char,
            1 << 22,
            true>>
        data_;                       // vectors + graph + quantization codes + factors
    Rotator<T>* rotator_ = nullptr;  // data rotator
    std::unique_ptr<VisitedListPool> visited_list_pool_ = nullptr;

    // Position of different data in each row (RawData + QuantizationCodes + Factors +
    // neighborIDs) Since we guarantee the degree for each vertex equals degree_bound
    // (multiple of 32), we do not need to store the degree for each vertex
    size_t batch_data_offset_ = 0;  // offset of qg batch data
    size_t neighbor_offset_ = 0;    // offset of neighbors
    size_t row_offset_ = 0;         // length of entire row
    size_t ef_ = 0;

    void initialize();

    void copy_vectors(const T*);

    [[nodiscard]] T* get_vector(PID data_id) {
        return reinterpret_cast<T*>(&data_.at(row_offset_ * data_id));
    }

    [[nodiscard]] const T* get_vector(PID data_id) const {
        return reinterpret_cast<const T*>(&data_.at(row_offset_ * data_id));
    }

    [[nodiscard]] char* get_batch_data(PID data_id) {
        return &data_.at((row_offset_ * data_id) + batch_data_offset_);
    }

    [[nodiscard]] const char* get_batch_data(PID data_id) const {
        return &data_.at((row_offset_ * data_id) + batch_data_offset_);
    }

    [[nodiscard]] PID* get_neighbors(PID data_id) {
        return reinterpret_cast<PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    [[nodiscard]] const PID* get_neighbors(PID data_id) const {
        return reinterpret_cast<const PID*>(
            &data_.at((row_offset_ * data_id) + neighbor_offset_)
        );
    }

    void find_candidates(
        PID,
        size_t,
        std::vector<AnnCandidate<T>>&,
        HashBasedBooleanSet&,
        const std::vector<uint32_t>&
    ) const;

    void update_qg(PID, const std::vector<AnnCandidate<T>>&);

    void update_results(buffer::SearchBuffer<T>&, HashBasedBooleanSet&, const T*);

    void scan_neighbors(
        const BatchQuery<T>&,
        PID,
        T*,
        buffer::SearchBuffer<T>&,
        HashBasedBooleanSet&,
        size_t
    ) const;

   public:
    explicit QuantizedGraph(
        size_t num,
        size_t dim,
        size_t max_deg,
        MetricType metric_type = METRIC_L2,
        RotatorType rotator_type = RotatorType::FhtKacRotator
    );

    explicit QuantizedGraph() = default;

    ~QuantizedGraph();

    [[nodiscard]] auto num_vertices() const { return this->num_points_; }

    [[nodiscard]] auto dimension() const { return this->dim_; }

    [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

    [[nodiscard]] auto entry_point() const { return this->entry_point_; }

    void set_ep(PID entry) { this->entry_point_ = entry; };

    void save(const char*) const;

    void load(const char*);

    void set_ef(size_t);

    /* search and copy results to KNN */
    void search(const T* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results);
    void search(
        const T* __restrict__ query,
        uint32_t knn,
        uint32_t* __restrict__ results,
        T* __restrict__ dists
    );
};

template <typename T>
inline QuantizedGraph<T>::QuantizedGraph(
    size_t num, size_t dim, size_t max_deg, MetricType metric_type, RotatorType rotator_type
)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dim_(dim)
    , padded_dim_(dim)
    , raw_dist_func_((metric_type == METRIC_IP) ? dot_product_dis<T> : euclidean_sqr<T>)
    , metric_type_(metric_type)
    , rotator_type_(rotator_type) {
    initialize();
}

template <typename T>
inline QuantizedGraph<T>::~QuantizedGraph() {
    ::delete this->rotator_;
}

template <typename T>
inline void QuantizedGraph<T>::copy_vectors(const T* data) {
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points_; ++i) {
        const T* src = data + (dim_ * i);
        T* dst = get_vector(i);
        std::copy(src, src + dim_, dst);
    }
    std::cout << "\tVectors Copied\n";
}

template <typename T>
inline void QuantizedGraph<T>::save(const char* filename) const {
    std::cout << "Saving quantized graph to " << filename << '\n';
    std::ofstream output(filename, std::ios::binary);
    assert(output.is_open());

    /* Basic variants */
    output.write(reinterpret_cast<const char*>(&num_points_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&degree_bound_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&entry_point_), sizeof(PID));
    output.write(reinterpret_cast<const char*>(&rotator_type_), sizeof(RotatorType));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(MetricType));

    /* Data */
    data_.save(output);

    /* Rotator */
    this->rotator_->save(output);

    output.close();
    std::cout << "\tQuantized graph saved!\n";
}

template <typename T>
inline void QuantizedGraph<T>::load(const char* filename) {
    std::cout << "loading quantized graph " << filename << '\n';

    /* Check existence */
    if (!file_exists(filename)) {
        std::cerr << "Index does not exist!\n";
        exit(1);
    }

    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Basic variants */
    input.read(reinterpret_cast<char*>(&num_points_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&degree_bound_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&padded_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&entry_point_), sizeof(PID));
    input.read(reinterpret_cast<char*>(&rotator_type_), sizeof(RotatorType));
    input.read(reinterpret_cast<char*>(&metric_type_), sizeof(MetricType));

    raw_dist_func_ = (metric_type_ == METRIC_IP) ? dot_product_dis<T> : euclidean_sqr<T>;

    initialize();

    /* Data */
    data_.load(input);

    /* Rotator */
    this->rotator_->load(input);
    if (rotator_->size() != padded_dim_) {
        std::cerr << "Bad padded_dim_ for rotator in QuantizedGraph<T>.load()\n";
        exit(1);
    }

    input.close();
    std::cout << "Quantized graph loaded!\n";
}

template <typename T>
inline void QuantizedGraph<T>::set_ef(size_t cur_ef) {
    this->ef_ = cur_ef;
}

/**
 * @brief search on qg
 *
 * @param query     unrotated query vector, dimension_ elements
 * @param knn       num of nearest neighbors
 * @param results   search result
 */
template <typename T>
inline void QuantizedGraph<T>::search(
    const T* __restrict__ query, uint32_t k, uint32_t* __restrict__ results
) {
    std::vector<T> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // init query
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    buffer::SearchBuffer<T> search_pool(ef_);
    // init search buffer
    search_pool.insert(this->entry_point_, std::numeric_limits<T>::max());

    buffer::SearchBuffer res_pool(k);  // result buffer
    auto* vis = visited_list_pool_->get_free_vislist();

    std::vector<T> est_dist(degree_bound_);  // estimated distances

    while (search_pool.has_next()) {
        PID cur_node = search_pool.pop();
        if (vis->get(cur_node)) {
            continue;
        }
        vis->set(cur_node);

        q_obj.set_g_add(raw_dist_func_(query, get_vector(cur_node), dim_));

        scan_neighbors(
            q_obj, cur_node, est_dist.data(), search_pool, *vis, this->degree_bound_
        );
        res_pool.insert(cur_node, q_obj.g_add());
    }

    update_results(res_pool, *vis, query);
    visited_list_pool_->release_vis_list(vis);
    res_pool.copy_results(results);
}

template <typename T>
inline void QuantizedGraph<T>::search(
    const T* __restrict__ query,
    uint32_t k,
    uint32_t* __restrict__ results,
    T* __restrict__ dists
) {
    std::vector<T> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // init query
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    buffer::SearchBuffer<T> search_pool(ef_);
    // init search buffer
    search_pool.insert(this->entry_point_, std::numeric_limits<T>::max());

    buffer::SearchBuffer res_pool(k);  // result buffer
    auto* vis = visited_list_pool_->get_free_vislist();

    std::vector<T> est_dist(degree_bound_);  // estimated distances

    while (search_pool.has_next()) {
        PID cur_node = search_pool.pop();
        if (vis->get(cur_node)) {
            continue;
        }
        vis->set(cur_node);

        q_obj.set_g_add(raw_dist_func_(query, get_vector(cur_node), dim_));

        scan_neighbors(
            q_obj, cur_node, est_dist.data(), search_pool, *vis, this->degree_bound_
        );
        res_pool.insert(cur_node, q_obj.g_add());
    }

    update_results(res_pool, *vis, query);
    visited_list_pool_->release_vis_list(vis);
    res_pool.copy_results(results, dists);
}

// scan a data row (including data vec and quantization codes for its neighbors)
// store estimated distance & return exact distnace for current vertex
template <typename T>
void QuantizedGraph<T>::scan_neighbors(
    const BatchQuery<T>& q_obj,
    PID data_id,
    T* est_dist,
    buffer::SearchBuffer<T>& search_pool,
    HashBasedBooleanSet& vis,
    size_t cur_degree
) const {
    const auto* batch_data = get_batch_data(data_id);
    for (size_t i = 0; i < cur_degree; i += fastscan::kBatchSize) {
        qg_batch_estdist(batch_data, q_obj, padded_dim_, est_dist + i);
        batch_data += QGBatchDataMap<T>::data_bytes(padded_dim_);
    }

    const PID* ptr_nb = get_neighbors(data_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        T dist = est_dist[i];

        if (search_pool.is_full(dist) || vis.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, dist);  // update search buffer
        memory::mem_prefetch_l2(
            reinterpret_cast<const char*>(get_vector(search_pool.next_id())), 10
        );
    }
}

template <typename T>
inline void QuantizedGraph<T>::update_results(
    buffer::SearchBuffer<T>& result_pool, HashBasedBooleanSet& vis, const T* query
) {
    if (result_pool.is_full()) {
        return;
    }

    auto data = result_pool.data();
    for (auto record : data) {
        PID* ptr_nb = get_neighbors(record.id);
        for (uint32_t i = 0; i < this->degree_bound_; ++i) {
            PID cur_neighbor = ptr_nb[i];
            if (!vis.get(cur_neighbor)) {
                vis.set(cur_neighbor);
                result_pool.insert(
                    cur_neighbor, raw_dist_func_(query, get_vector(cur_neighbor), dim_)
                );
            }
        }
        if (result_pool.is_full()) {
            break;
        }
    }
}

// initialize const offsets & data array
template <typename T>
inline void QuantizedGraph<T>::initialize() {
    ::delete rotator_;

    rotator_ = choose_rotator<float>(dim_, rotator_type_, round_up_to_multiple(dim_, 64));
    padded_dim_ = rotator_->size();

    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);

    this->batch_data_offset_ = dim_ * sizeof(T);  // pos of packed code (aligned)
    this->neighbor_offset_ =
        batch_data_offset_ + (QGBatchDataMap<T>::data_bytes(padded_dim_) *
                              (degree_bound_ / fastscan::kBatchSize));
    this->row_offset_ = neighbor_offset_ + (degree_bound_ * sizeof(PID));

    data_ = Array<char, std::vector<size_t>, memory::AlignedAllocator<char, 1 << 22, true>>(
        std::vector<size_t>{num_points_, row_offset_}
    );

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, num_points_);
}

// find candidate neighbors for cur_id, exclude the vertex itself
template <typename T>
inline void QuantizedGraph<T>::find_candidates(
    PID cur_id,
    size_t search_ef,
    std::vector<AnnCandidate<T>>& results,
    HashBasedBooleanSet& vis,
    const std::vector<uint32_t>& degrees
) const {
    const T* query = get_vector(cur_id);
    std::vector<T> rotated_query(padded_dim_);
    rotator_->rotate(query, rotated_query.data());

    // init query
    BatchQuery<T> q_obj(rotated_query.data(), padded_dim_);

    // insert entry point to initialize search buffer
    buffer::SearchBuffer tmp_pool(search_ef);
    tmp_pool.insert(this->entry_point_, 1e10);
    memory::mem_prefetch_l1(
        reinterpret_cast<const char*>(get_vector(this->entry_point_)), 10
    );

    /* Current version of fast scan compute 32 distances */
    std::vector<T> est_dist(degree_bound_);  // estimated distances
    while (tmp_pool.has_next()) {
        auto cur_candi = tmp_pool.pop();
        if (vis.get(cur_candi)) {
            continue;
        }
        vis.set(cur_candi);
        auto cur_degree = degrees[cur_candi];
        q_obj.set_g_add(raw_dist_func_(query, get_vector(cur_candi), dim_));
        scan_neighbors(q_obj, cur_candi, est_dist.data(), tmp_pool, vis, cur_degree);
        if (cur_candi != cur_id) {
            results.emplace_back(cur_candi, q_obj.g_add());
        }
    }
}

// based on new neighbor lists to update quantization code and factors
template <typename T>
inline void QuantizedGraph<T>::update_qg(
    PID cur_id, const std::vector<AnnCandidate<T>>& new_neighbors
) {
    size_t cur_degree = new_neighbors.size();

    if (cur_degree == 0) {
        return;
    }
    // copy neighbors
    PID* neighbor_ptr = get_neighbors(cur_id);
    for (size_t i = 0; i < cur_degree; ++i) {
        neighbor_ptr[i] = new_neighbors[i].id;
    }

    // rotated data
    std::vector<T> rotated_data(cur_degree * padded_dim_);
    std::vector<T> rotated_centroid(padded_dim_);
    for (size_t i = 0; i < cur_degree; ++i) {
        const T* neighbor_vec = get_vector(new_neighbors[i].id);
        this->rotator_->rotate(neighbor_vec, &rotated_data[i * padded_dim_]);
    }
    this->rotator_->rotate(get_vector(cur_id), rotated_centroid.data());

    // quantize batches for current vertex
    auto* batch_data = get_batch_data(cur_id);
    const auto* data = rotated_data.data();
    for (size_t i = 0; i < cur_degree; i += fastscan::kBatchSize) {
        quant::quantize_qg_batch(
            data,
            rotated_centroid.data(),
            std::min(cur_degree - i, fastscan::kBatchSize),
            padded_dim_,
            batch_data,
            metric_type_
        );

        data += fastscan::kBatchSize * padded_dim_;
        batch_data += QGBatchDataMap<T>::data_bytes(padded_dim_);
    }
}
}  // namespace rabitqlib::symqg
