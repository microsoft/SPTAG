#include <immintrin.h>

#include <cmath>
#include <cstdint>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd {

void scalar_quantize_uint8_avx2(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul8 = dim - (dim & 0b111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m256 lo256 = _mm256_set1_ps(lo);
    __m256 od256 = _mm256_set1_ps(one_over_delta);
    __m128i zero = _mm_setzero_si128();

    for (; i < mul8; i += 8) {
        __m256 cur = _mm256_loadu_ps(&vec0[i]);
        cur = _mm256_mul_ps(_mm256_sub_ps(cur, lo256), od256);
        __m256i i32 = _mm256_cvtps_epi32(cur);
        __m128i lo32 = _mm256_castsi256_si128(i32);
        __m128i hi32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(lo32, hi32);
        __m128i i8 = _mm_packus_epi16(i16, zero);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(&result[i]), i8);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void scalar_quantize_uint16_avx2(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul8 = dim - (dim & 0b111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m256 lo256 = _mm256_set1_ps(lo);
    __m256 ow256 = _mm256_set1_ps(one_over_delta);
    for (; i < mul8; i += 8) {
        __m256 cur = _mm256_loadu_ps(&vec0[i]);
        cur = _mm256_mul_ps(_mm256_sub_ps(cur, lo256), ow256);
        __m256i i32 = _mm256_cvtps_epi32(cur);
        __m128i lo32 = _mm256_castsi256_si128(i32);
        __m128i hi32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(lo32, hi32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void new_transpose_bin_avx2(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m256i vec_00_to_15 = _mm256_loadu_si256((__m256i const*)(q));
        __m256i vec_16_to_31 = _mm256_loadu_si256((__m256i const*)(q + 16));
        __m256i vec_32_to_47 = _mm256_loadu_si256((__m256i const*)(q + 32));
        __m256i vec_48_to_63 = _mm256_loadu_si256((__m256i const*)(q + 48));

        // the first (16 - b_query) bits are empty
        vec_00_to_15 = _mm256_slli_epi32(vec_00_to_15, (16 - b_query));
        vec_16_to_31 = _mm256_slli_epi32(vec_16_to_31, (16 - b_query));
        vec_32_to_47 = _mm256_slli_epi32(vec_32_to_47, (16 - b_query));
        vec_48_to_63 = _mm256_slli_epi32(vec_48_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            // pack two 16-bit vectors to 8-bit interleaved vectors
            __m256i p0 = _mm256_packs_epi16(vec_00_to_15, vec_16_to_31);
            __m256i p1 = _mm256_packs_epi16(vec_32_to_47, vec_48_to_63);

            uint32_t m0 = _mm256_movemask_epi8(p0);
            uint32_t m1 = _mm256_movemask_epi8(p1);

            // Fix AVX2 Lane Ordering of the interleaved mask
            auto fix_avx2_mask = [](uint32_t m) {
                return (m & 0xFF0000FF) | ((m & 0x00FF0000) >> 8) | ((m & 0x0000FF00) << 8);
            };

            m0 = fix_avx2_mask(m0);
            m1 = fix_avx2_mask(m1);

            m0 = reverse_bits(m0);
            m1 = reverse_bits(m1);

            uint64_t v = (static_cast<uint64_t>(m0) << 32) | m1;

            tq[b_query - j - 1] = v;

            vec_00_to_15 = _mm256_slli_epi16(vec_00_to_15, 1);
            vec_16_to_31 = _mm256_slli_epi16(vec_16_to_31, 1);
            vec_32_to_47 = _mm256_slli_epi16(vec_32_to_47, 1);
            vec_48_to_63 = _mm256_slli_epi16(vec_48_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
}

void new_transpose_bin_512_avx2(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        // Each chunk represents 64 bytes (512 bits) of dimensions
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            // Load 64 bytes using two sequential 32-byte AVX2 registers
            const uint8_t* current_q_lo = q + i + k * 64;
            const uint8_t* current_q_hi = q + i + k * 64 + 32;

            __m256i vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_lo));
            __m256i vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_hi));

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = b_query - 1 - j;
                __m256i mask_vec = _mm256_set1_epi8(static_cast<char>(1 << bit_idx));

                // Process lower 32 bytes
                __m256i res_lo = _mm256_and_si256(vec_lo, mask_vec);
                __m256i eq_lo = _mm256_cmpeq_epi8(res_lo, _mm256_setzero_si256());
                uint32_t m_lo = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_lo));

                // Process upper 32 bytes
                __m256i res_hi = _mm256_and_si256(vec_hi, mask_vec);
                __m256i eq_hi = _mm256_cmpeq_epi8(res_hi, _mm256_setzero_si256());
                uint32_t m_hi = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_hi));

                // Combine both 32-bit masks into a single 64-bit mask
                uint64_t m = (static_cast<uint64_t>(m_hi) << 32) | m_lo;

                // Write into the 64-bit structured macro-layout
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(m);
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
}

float mask_ip_x0_q_avx2(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;

    __m256 sum = _mm256_setzero_ps();

    __m256i bit_checker = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        // 64 bits / 8 floats = 8 iterations
        for (int j = 0; j < 8; ++j) {
            uint8_t current_byte = static_cast<uint8_t>(bits >> (j * 8));
            __m256i v_byte = _mm256_set1_epi32(current_byte);
            __m256i masked_bits = _mm256_and_si256(v_byte, bit_checker);
            __m256i mask = _mm256_cmpgt_epi32(masked_bits, _mm256_setzero_si256());

            __m256 q_vals = _mm256_loadu_ps(it_query);
            __m256 masked = _mm256_and_ps(q_vals, _mm256_castsi256_ps(mask));

            sum = _mm256_add_ps(sum, masked);

            it_query += 8;
        }
        ++it_data;
    }

    float result = 0.0f;
    for (int i = 0; i < 8; ++i) {
        result += reinterpret_cast<float*>(&sum)[i];
    }
    return result;
}

}  // namespace rabitqlib::simd
