#include "hnsw_search_avx2_kernels.hpp"
#include "hnsw_search_avx512_kernels.hpp"

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace rabitqlib::hnsw::detail {

struct HnswAvx512CoreKernel {
    static inline float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return hnsw_warmup_ip_x0_q_512_avx2(data, query, delta, vl, padded_dim, b_query);
    }

    static inline float mask_ip_x0_q(
        const float* query, const uint64_t* data, size_t padded_dim
    ) {
        return hnsw_mask_ip_x0_q_avx512(query, data, padded_dim);
    }
};

maxheap<std::pair<float, PID>> search_knn_avx512_core(
    HierarchicalNSW& index, const float* rotated_query, size_t topk
) {
    return index.search_knn_direct<HnswAvx512CoreKernel>(rotated_query, topk);
}

}  // namespace rabitqlib::hnsw::detail
