#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace rabitqlib::simd {

float warmup_ip_x0_q_512_avx512(
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

    __m512i acc_bits[b_query];
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


}  // namespace rabitqlib::simd
