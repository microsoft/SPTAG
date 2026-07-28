#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "rabitqlib/index/query.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"

namespace rabitqlib::hnsw::detail {

static inline float hnsw_mask_ip_x0_q_avx512(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;

    //    __m512 sum0 = _mm512_setzero_ps();
    //    __m512 sum1 = _mm512_setzero_ps();
    //    __m512 sum2 = _mm512_setzero_ps();
    //    __m512 sum3 = _mm512_setzero_ps();

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = rabitqlib::reverse_bits_u64(*it_data);

        auto mask0 = static_cast<__mmask16>(bits);
        auto mask1 = static_cast<__mmask16>(bits >> 16);
        auto mask2 = static_cast<__mmask16>(bits >> 32);
        auto mask3 = static_cast<__mmask16>(bits >> 48);

        __m512 masked0 = _mm512_maskz_loadu_ps(mask0, it_query);
        __m512 masked1 = _mm512_maskz_loadu_ps(mask1, it_query + 16);
        __m512 masked2 = _mm512_maskz_loadu_ps(mask2, it_query + 32);
        __m512 masked3 = _mm512_maskz_loadu_ps(mask3, it_query + 48);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        //         _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);

        ++it_data;
        it_query += 64;
    }

    //    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
}

static inline float hnsw_warmup_ip_x0_q_512_avx512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    __m512i acc_ip = _mm512_setzero_si512();
    __m512i acc_ppc = _mm512_setzero_si512();

    size_t i = 0;
    size_t dim_end_512 = (padded_dim / 512) * 512;

    __m512i acc_bits[SplitSingleQuery<float>::kNumBits];
    for (size_t j = 0; j < b_query; ++j) {
        acc_bits[j] = _mm512_setzero_si512();
    }

    for (; i < dim_end_512; i += 512) {
        __m512i data_vec = _mm512_loadu_si512(data);
        data += 8;

        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(data_vec));

        for (size_t j = 0; j < b_query; ++j) {
            __m512i query_vec = _mm512_loadu_si512(query);
            query += 8;

            __m512i pop = _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec));
            acc_bits[j] = _mm512_add_epi64(acc_bits[j], pop);
        }
    }

    size_t remaining_dim = padded_dim - i;
    if (remaining_dim > 0) {
        size_t num_chunks = remaining_dim / 64;
        auto valid_mask = static_cast<__mmask8>((1u << num_chunks) - 1u);

        __m512i data_vec = _mm512_maskz_loadu_epi64(valid_mask, data);
        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(data_vec));

        for (size_t j = 0; j < b_query; ++j) {
            __m512i query_vec = _mm512_maskz_loadu_epi64(valid_mask, query);
            query += num_chunks;

            __m512i pop = _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec));
            acc_bits[j] = _mm512_add_epi64(acc_bits[j], pop);
        }
    }

    for (size_t j = 0; j < b_query; ++j) {
        __m128i shift = _mm_cvtsi32_si128(static_cast<int>(j));
        acc_ip = _mm512_add_epi64(acc_ip, _mm512_sll_epi64(acc_bits[j], shift));
    }

    ip_scalar += static_cast<size_t>(_mm512_reduce_add_epi64(acc_ip));
    ppc_scalar += static_cast<size_t>(_mm512_reduce_add_epi64(acc_ppc));

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}

}  // namespace rabitqlib::hnsw::detail
