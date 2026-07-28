#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/cluster.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/memory.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib::ivf {
class IVF {
   private:
    Initializer* initer_ = nullptr;      // initializer for find candidate cluster
    char* batch_data_ = nullptr;         // 1-bit code and factors
    char* ex_data_ = nullptr;            // code for remaining bits
    PID* ids_ = nullptr;                 // PID of vectors (orgnized by clusters)
    size_t num_;                         // num of data points
    size_t dim_;                         // dimension of data points
    size_t padded_dim_;                  // dimension after padding,
    size_t num_cluster_;                 // num of centroids (clusters)
    size_t ex_bits_;                     // total bits = ex_bits_ + 1
    RotatorType type_;                   // type of rotator
    Rotator<float>* rotator_ = nullptr;  // Data Rotator
    std::vector<Cluster> cluster_lst_;   // List of clusters in ivf
    MetricType metric_type_ = rabitqlib::METRIC_L2;  // metric type
    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;

    void quantize_cluster(
        Cluster&,
        const std::vector<PID>&,
        const float*,
        const float*,
        float*,
        const quant::RabitqConfig&
    );

    [[nodiscard]] size_t ids_bytes() const { return sizeof(PID) * num_; }

    // get num of bytes used for 1-bit code and corresponding factors
    [[nodiscard]] size_t batch_data_bytes(const std::vector<size_t>& cluster_sizes) const {
        assert(cluster_sizes.size() == num_cluster_);  // num of clusters
        size_t total_blocks = 0;
        for (auto size : cluster_sizes) {
            total_blocks += div_round_up(size, fastscan::kBatchSize);
        }
        return total_blocks * BatchDataMap<float>::data_bytes(padded_dim_);
    }

    [[nodiscard]] size_t ex_data_bytes() const {
        return ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * num_;
    }

    void allocate_memory(const std::vector<size_t>&);

    void init_clusters(const std::vector<size_t>&);

    void free_memory() {
        ::delete initer_;
        std::free(batch_data_);
        std::free(ex_data_);
        std::free(ids_);
    }

    void search_cluster(
        const Cluster&, const SplitBatchQuery<float>&, buffer::SearchBuffer<float>&, bool
    ) const;

    void scan_one_batch(
        const char* batch_data,
        const char* ex_data,
        const PID* ids,
        const SplitBatchQuery<float>& q_obj,
        buffer::SearchBuffer<float>& knns,
        size_t num_points,
        bool
    ) const;

   public:
    explicit IVF() {}
    explicit IVF(
        size_t,
        size_t,
        size_t,
        size_t,
        MetricType metric_type = rabitqlib::METRIC_L2,
        RotatorType type = RotatorType::FhtKacRotator
    );

    ~IVF();

    [[nodiscard]] size_t max_elements() const { return num_; }
    [[nodiscard]] size_t dimension() const { return dim_; }
    [[nodiscard]] size_t nbits() const { return ex_bits_ + 1; }
    [[nodiscard]] MetricType metric_type() const { return metric_type_; }
    [[nodiscard]] RotatorType rotator_type() const { return type_; }

    void construct(const float*, const float*, const PID*, bool, size_t);

    void save(const char*) const;

    void load(const char*);

    void search(const float*, size_t, size_t, PID*, bool) const;

    void search(const float*, size_t, size_t, PID*, float*, bool) const;

    [[nodiscard]] size_t padded_dim() const { return this->padded_dim_; }

    [[nodiscard]] size_t num_clusters() const { return this->num_cluster_; }
};

inline IVF::IVF(
    size_t n,
    size_t dim,
    size_t cluster_num,
    size_t bits,
    MetricType metric_type,
    RotatorType type
)
    : num_(n)
    , dim_(dim)
    , padded_dim_(dim)
    , num_cluster_(cluster_num)
    , ex_bits_(bits - 1)
    , type_(type)
    , metric_type_(metric_type) {
    if (bits < 1 || bits > 9) {
        std::cerr << "Invalid number of bits for quantization in IVF::IVF\n";
        std::cerr << "Expected: 1 to 9  Input:" << bits << '\n';
        std::cerr.flush();
        exit(1);
    };
    rotator_ = choose_rotator<float>(dim, type, round_up_to_multiple(dim_, 64));
    padded_dim_ = rotator_->size();
    /* check size */
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);
}

inline IVF::~IVF() {
    delete rotator_;
    free_memory();
}

/**
 * @brief Construct clusters in IVF
 *
 * @param data Data objects (N*DIM)
 * @param centroids Centroid vectors (K*DIM)
 * @param clustter_ids Cluster ID for each data objects
 */
inline void IVF::construct(
    const float* data, const float* centroids, const PID* cluster_ids, bool faster = false,
    size_t num_threads = std::numeric_limits<size_t>::max()
) {
    std::cout << "Start IVF construction...\n";

    // get id list for each cluster
    std::cout << "\tLoading clustering information...\n";
    std::vector<size_t> counts(num_cluster_, 0);
    std::vector<std::vector<PID>> id_lists(num_cluster_);
    for (size_t i = 0; i < num_; ++i) {
        PID cid = cluster_ids[i];
        if (cid > num_cluster_) {
            std::cerr << "Bad cluster id\n";
            exit(1);
        }
        id_lists[cid].push_back(static_cast<PID>(i));
        counts[cid] += 1;
    }

    allocate_memory(counts);

    // init the cluster list
    init_clusters(counts);

    // all rotated centroids
    std::vector<float> rotated_centroids(num_cluster_ * padded_dim_);

    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    num_threads = std::min(num_threads, rabitqlib::total_threads());
    /* Quantize each cluster */
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < num_cluster_; ++i) {
        const float* cur_centroid = centroids + (i * dim_);
        float* cur_rotated_c = &rotated_centroids[i * padded_dim_];
        Cluster& cp = cluster_lst_[i];
        quantize_cluster(cp, id_lists[i], data, cur_centroid, cur_rotated_c, config);
    }

    this->initer_->add_vectors(rotated_centroids.data(), num_threads);
}

inline void IVF::allocate_memory(const std::vector<size_t>& cluster_sizes) {
    std::cout << "Allocating memory for IVF...\n";
    if (num_cluster_ < 20000UL) {
        this->initer_ = new FlatInitializer(padded_dim_, num_cluster_);
    } else {
        this->initer_ = new HNSWInitializer(padded_dim_, num_cluster_);
    }
    this->batch_data_ =
        memory::align_allocate<64, char, true>(batch_data_bytes(cluster_sizes));
    if (ex_bits_ > 0) {
        this->ex_data_ = memory::align_allocate<64, char, true>(ex_data_bytes());
    }
    this->ids_ = memory::align_allocate<64, PID, true>(ids_bytes());

    this->ip_func_ = select_excode_ipfunc(ex_bits_);
}

/**
 * @brief intialize the cluster list: finding idx for all data
 */
inline void IVF::init_clusters(const std::vector<size_t>& cluster_sizes) {
    this->cluster_lst_.reserve(num_cluster_);
    size_t added_vectors = 0;
    size_t added_batches = 0;
    for (size_t i = 0; i < num_cluster_; ++i) {
        // find data location for current cluster
        size_t num = cluster_sizes[i];
        size_t num_batches = div_round_up(num, fastscan::kBatchSize);

        char* current_batch_data =
            batch_data_ + (BatchDataMap<float>::data_bytes(padded_dim_) * added_batches);
        char* current_ex_data =
            ex_data_ +
            (added_vectors * ExDataMap<float>::data_bytes(padded_dim_, ex_bits_));
        PID* ids = ids_ + added_vectors;

        Cluster cur_cluster(num, current_batch_data, current_ex_data, ids);
        this->cluster_lst_.push_back(std::move(cur_cluster));

        added_vectors += num;
        added_batches += num_batches;
    }
}

inline void IVF::quantize_cluster(
    Cluster& cp,
    const std::vector<PID>& IDs,
    const float* data,
    const float* cur_centroid,
    float* rotated_centroid,
    const quant::RabitqConfig& config
) {
    size_t num_points = IDs.size();
    if (cp.num() != num_points) {
        std::cerr << "Size of cluster and IDs are inequivalent\n";
        std::cerr << "Cluster: " << cp.num() << " IDs: " << num_points << '\n';
        exit(1);
    }

    // copy ids
    std::copy(IDs.begin(), IDs.end(), cp.ids());

    // rotate centroid
    this->rotator_->rotate(cur_centroid, rotated_centroid);

    // rotate vectors for this cluster
    std::vector<float> rotated_data(padded_dim_ * num_points);
    for (size_t i = 0; i < num_points; ++i) {
        rotator_->rotate(data + (IDs[i] * dim_), rotated_data.data() + (i * padded_dim_));
    }

    char* batch_data = cp.batch_data();
    char* ex_data = cp.ex_data();
    for (size_t i = 0; i < num_points; i += fastscan::kBatchSize) {
        size_t n = std::min(fastscan::kBatchSize, num_points - i);

        quant::quantize_split_batch(
            rotated_data.data() + (i * padded_dim_),
            rotated_centroid,
            n,
            padded_dim_,
            ex_bits_,
            batch_data,
            ex_data,
            metric_type_,
            config
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * n;
    }
}

inline void IVF::save(const char* filename) const {
    if (cluster_lst_.size() == 0) {
        std::cerr << "IVF not constructed\n";
        return;
    }

    std::ofstream output(filename, std::ios::binary);

    /* Save meta data */
    output.write(reinterpret_cast<const char*>(&num_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_cluster_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&type_), sizeof(type_));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    /* Save number of vectors of each cluster */
    std::vector<size_t> cluster_sizes;
    cluster_sizes.reserve(num_cluster_);
    for (const auto& cur_cluster : cluster_lst_) {
        cluster_sizes.push_back(cur_cluster.num());
    }
    output.write(
        reinterpret_cast<const char*>(cluster_sizes.data()),
        static_cast<long>(sizeof(size_t) * num_cluster_)
    );

    /* Save rotator */
    this->rotator_->save(output);

    /* Save data */
    this->initer_->save(output, filename);
    output.write(
        reinterpret_cast<const char*>(batch_data_),
        static_cast<long>(batch_data_bytes(cluster_sizes))
    );
    output.write(
        reinterpret_cast<const char*>(ex_data_), static_cast<long>(ex_data_bytes())
    );
    output.write(reinterpret_cast<const char*>(ids_), static_cast<long>(ids_bytes()));

    output.close();
}

inline void IVF::load(const char* filename) {
    std::cout << "Loading IVF...\n";
    std::ifstream input(filename, std::ios::binary);
    assert(input.is_open());

    /* Load meta data */
    std::cout << "\tLoading meta data...\n";
    input.read(reinterpret_cast<char*>(&this->num_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&this->dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&this->num_cluster_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&this->ex_bits_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&type_), sizeof(type_));
    input.read(reinterpret_cast<char*>(&metric_type_), sizeof(metric_type_));

    rotator_ = choose_rotator<float>(dim_, type_, round_up_to_multiple(dim_, 64));
    padded_dim_ = rotator_->size();

    /* Load number of vectors of each cluster */
    std::vector<size_t> cluster_sizes(num_cluster_, 0);
    input.read(
        reinterpret_cast<char*>(cluster_sizes.data()),
        static_cast<long>(sizeof(size_t) * num_cluster_)
    );

    size_t tmp =
        std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), static_cast<size_t>(0));
    if (tmp != num_) {
        std::cerr << "The sum of cluster num != total number of points\n";
        exit(1);
    }

    /* Load rotator */
    this->rotator_->load(input);

    /* Load data */
    free_memory();
    allocate_memory(cluster_sizes);
    this->initer_->load(input, filename);
    input.read(batch_data_, static_cast<long>(batch_data_bytes(cluster_sizes)));
    input.read(ex_data_, static_cast<long>(ex_data_bytes()));
    input.read(reinterpret_cast<char*>(ids_), static_cast<long>(ids_bytes()));

    /* Init each cluster */
    init_clusters(cluster_sizes);

    input.close();
    std::cout << "Index loaded\n";
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    bool use_hacc = true
) const {
    this->search(query, k, nprobe, results, nullptr, use_hacc);
}

inline void IVF::search(
    const float* __restrict__ query,
    size_t k,
    size_t nprobe,
    PID* __restrict__ results,
    float* __restrict__ dists,
    bool use_hacc
) const {
    nprobe = std::min(nprobe, num_cluster_);  // corner case
    std::vector<float> rotated_query(padded_dim_);
    this->rotator_->rotate(query, rotated_query.data());

    // use initer to get closest nprobe centroids
    std::vector<AnnCandidate<float>> centroid_dist(nprobe);
    this->initer_->centroids_distances(rotated_query.data(), nprobe, centroid_dist);

    buffer::SearchBuffer knns(k);

    SplitBatchQuery<float> q_obj(
        rotated_query.data(), padded_dim_, ex_bits_, metric_type_, use_hacc
    );

    for (size_t i = 0; i < nprobe; ++i) {
        PID cid = centroid_dist[i].id;
        float dist = centroid_dist[i].distance;
        const Cluster& cur_cluster = cluster_lst_[cid];

        if (metric_type_ == METRIC_L2) {
            q_obj.set_g_add(dist);
        } else if (metric_type_ == METRIC_IP) {
            auto g_add_ip = dot_product<float>(
                rotated_query.data(), initer_->centroid(cid), padded_dim_
            );
            q_obj.set_g_add(dist, g_add_ip);
        } else {
            // unsupported
            std::cerr << "Invalid quantize metric type, only support L2 and IP metric\n "
                      << std::flush;
            return;
        }
        // q_obj.set_g_add(dist);
        search_cluster(cur_cluster, q_obj, knns, use_hacc);
    }

    if (dists != nullptr) {
        knns.copy_results(results, dists);
    } else {
        knns.copy_results(results);
    }
}

inline void IVF::search_cluster(
    const Cluster& cur_cluster,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    bool use_hacc
) const {
    size_t iter = cur_cluster.num() / fastscan::kBatchSize;
    size_t remain = cur_cluster.num() - (iter * fastscan::kBatchSize);

    const char* batch_data = cur_cluster.batch_data();
    const char* ex_data = cur_cluster.ex_data();
    const PID* ids = cur_cluster.ids();

    /* Compute distances block by block */
    for (size_t i = 0; i < iter; ++i) {
        scan_one_batch(
            batch_data, ex_data, ids, q_obj, knns, fastscan::kBatchSize, use_hacc
        );

        batch_data += BatchDataMap<float>::data_bytes(padded_dim_);
        ex_data +=
            ExDataMap<float>::data_bytes(padded_dim_, ex_bits_) * fastscan::kBatchSize;
        ids += fastscan::kBatchSize;
    }

    if (remain > 0) {
        // scan the last block
        scan_one_batch(batch_data, ex_data, ids, q_obj, knns, remain, use_hacc);
    }
}

inline void IVF::scan_one_batch(
    const char* batch_data,
    const char* ex_data,
    const PID* ids,
    const SplitBatchQuery<float>& q_obj,
    buffer::SearchBuffer<float>& knns,
    size_t num_points,
    bool use_hacc
) const {
    std::array<float, fastscan::kBatchSize> est_distance;  // estimated distance
    std::array<float, fastscan::kBatchSize> low_distance;  // lower distance
    std::array<float, fastscan::kBatchSize> ip_x0_qr;      // inner product of the 1st bit

    split_batch_estdist(
        batch_data,
        q_obj,
        padded_dim_,
        est_distance.data(),
        low_distance.data(),
        ip_x0_qr.data(),
        use_hacc
    );

    float distk = knns.top_dist();

    // if only use 1-bit code, directly return
    if (ex_bits_ == 0) {
        for (size_t i = 0; i < num_points; ++i) {
            PID id = ids[i];
            float ex_dist = est_distance[i];
            knns.insert(id, ex_dist);
            distk = knns.top_dist();
        }
        return;
    }

    // incremental distance computation - V2
    for (size_t i = 0; i < num_points; ++i) {
        float lower_dist = low_distance[i];
        if (lower_dist < distk) {
            PID id = ids[i];
            ConstExDataMap<float> cur_ex(ex_data, padded_dim_, ex_bits_);
            float ex_dist = split_distance_boosting(
                ex_data, ip_func_, q_obj, padded_dim_, ex_bits_, ip_x0_qr[i]
            );
            knns.insert(id, ex_dist);
            distk = knns.top_dist();
        }
        ex_data += ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    }
}
}  // namespace rabitqlib::ivf
