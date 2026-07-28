#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace rabitqlib::simd {

// Helper: AVX2 64-bit Popcount; Mula's method
static inline __m256i popcount_avx2(__m256i v) {
    // Lookup table for population count of 0-15
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    // Count low nibbles
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i cnt_lo = _mm256_shuffle_epi8(lookup, lo);

    // Count high nibbles
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i cnt_hi = _mm256_shuffle_epi8(lookup, hi);

    // Add counts (bytes)
    __m256i cnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);

    // Sum bytes horizontally into 64-bit integers (SAD against 0)
    return _mm256_sad_epu8(cnt_bytes, _mm256_setzero_si256());
}

float warmup_ip_x0_q_512_avx2(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    __m256i acc_ip = _mm256_setzero_si256();
    __m256i acc_ppc = _mm256_setzero_si256();

    size_t i = 0;
    // Step by 512 bits at a time (64 bytes = 16 elements of 32-bit integers)
    size_t dim_end_512 = (padded_dim / 512) * 512;

    __m256i acc_bits[b_query];
    for (size_t j = 0; j < b_query; ++j) {
        acc_bits[j] = _mm256_setzero_si256();
    }

    for (; i < dim_end_512; i += 512) {
        // Load 64 bytes of data using paired 32-byte loads
        __m256i data_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        __m256i data_vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + 4));
        data += 8; // Advance 8 x 64-bit ints (64 bytes)

        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_lo));
        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_hi));

        for (size_t j = 0; j < b_query; ++j) {
            // Load 64 bytes of transposed query matching the 512-bit block layout
            __m256i query_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query));
            __m256i query_vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + 4));
            query += 8; // Advance 8 x 64-bit ints (64 bytes)

            __m256i pop_lo = popcount_avx2(_mm256_and_si256(data_vec_lo, query_vec_lo));
            __m256i pop_hi = popcount_avx2(_mm256_and_si256(data_vec_hi, query_vec_hi));

            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_lo);
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_hi);
        }
    }

    // Remainder block: handles leftovers less than 512 bits wide (e.g., last 448 bits)
    size_t remaining_dim = padded_dim - i;
    if (remaining_dim > 0) {
        size_t num_chunks_64 = remaining_dim / 64;
        size_t num_chunks_32 = remaining_dim / 32;

        size_t chunks_lo = (num_chunks_32 > 8) ? 8 : num_chunks_32;
        size_t chunks_hi = (num_chunks_32 > 8) ? (num_chunks_32 - 8) : 0;

        // 1. Create a baseline sequence register
        __m256i sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        // 2. Generate masks in-register using Greater-Than comparisons
        // If chunks_lo is 3, limit will be [3,3,3,3,3,3,3,3].
        // 3 > seq results in [-1, -1, -1, 0, 0, 0, 0, 0], which is the exact mask needed.
        __m256i limit_lo = _mm256_set1_epi32(static_cast<int>(chunks_lo));
        __m256i mask_lo  = _mm256_cmpgt_epi32(limit_lo, sequence);

        __m256i limit_hi = _mm256_set1_epi32(static_cast<int>(chunks_hi));
        __m256i mask_hi  = _mm256_cmpgt_epi32(limit_hi, sequence);

        // 3. Vectorized execution continues with zero memory latency
        __m256i data_vec_lo = _mm256_maskload_epi32(reinterpret_cast<const int*>(data), mask_lo);
        __m256i data_vec_hi = _mm256_maskload_epi32(reinterpret_cast<const int*>(data + 4), mask_hi);

        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_lo));
        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_hi));

        for (size_t j = 0; j < b_query; ++j) {
            __m256i query_vec_lo = _mm256_maskload_epi32(reinterpret_cast<const int*>(query), mask_lo);
            __m256i query_vec_hi = _mm256_maskload_epi32(reinterpret_cast<const int*>(query + 4), mask_hi);
            query += num_chunks_64;

            __m256i pop_lo = popcount_avx2(_mm256_and_si256(data_vec_lo, query_vec_lo));
            __m256i pop_hi = popcount_avx2(_mm256_and_si256(data_vec_hi, query_vec_hi));

            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_lo);
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_hi);
        }
    }

    for (size_t j = 0; j < b_query; ++j) {
        __m128i shift = _mm_cvtsi32_si128(static_cast<int>(j));
        acc_ip = _mm256_add_epi64(acc_ip, _mm256_sll_epi64(acc_bits[j], shift));
    }

    // Standard reduction for a single __m256i
    auto mm256_reduce_add_epi64 = [](__m256i v) {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 1);
        __m128i sum = _mm_add_epi64(low, high);
        return _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
    };

    ip_scalar += mm256_reduce_add_epi64(acc_ip);
    ppc_scalar += mm256_reduce_add_epi64(acc_ppc);

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));

}


}  // namespace rabitqlib::simd
