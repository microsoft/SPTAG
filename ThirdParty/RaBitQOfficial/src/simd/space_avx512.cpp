#include <immintrin.h>

#include <cmath>
#include <cstdint>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd {

void scalar_quantize_uint8_avx512(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m512 lo512 = _mm512_set1_ps(lo);
    __m512 od512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        __m512 cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);
        __m128i i8 = _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(cur));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&result[i]), i8);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void scalar_quantize_uint16_avx512(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1.0F / delta;
    __m512 lo512 = _mm512_set1_ps(lo);
    __m512 ow512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        __m512 cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), ow512);
        __m256i i16 = _mm512_cvtusepi32_epi16(_mm512_cvtps_epi32(cur));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
}

void new_transpose_bin_avx512(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    // 512 / 16 = 32
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m512i vec_00_to_31 = _mm512_loadu_si512(q);
        __m512i vec_32_to_63 = _mm512_loadu_si512(q + 32);

        // the first (16 - b_query) bits are empty
        vec_00_to_31 = _mm512_slli_epi32(vec_00_to_31, (16 - b_query));
        vec_32_to_63 = _mm512_slli_epi32(vec_32_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v0 = _mm512_movepi16_mask(vec_00_to_31);  // get most significant bit
            uint32_t v1 = _mm512_movepi16_mask(vec_32_to_63);  // get most significant bit
            // [TODO: remove all reverse_bits]
            v0 = reverse_bits(v0);
            v1 = reverse_bits(v1);
            uint64_t v = (static_cast<uint64_t>(v0) << 32) + v1;

            tq[b_query - j - 1] = v;

            vec_00_to_31 = _mm512_slli_epi16(vec_00_to_31, 1);
            vec_32_to_63 = _mm512_slli_epi16(vec_32_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
}

void new_transpose_bin_512_avx512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    // Keep full 512-dim blocks as 8 chunks, but store the tail as compact
    // [b_query x num_chunks] so runtime can use maskz loads without query padding.
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            const uint8_t* current_q = q + i + k * 64;
            __m512i vec = _mm512_loadu_si512(current_q);

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = b_query - 1 - j;
                __mmask64 m = _mm512_test_epi8_mask(vec, _mm512_set1_epi8(1 << bit_idx));
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(static_cast<uint64_t>(m));
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
}

float mask_ip_x0_q_avx512(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;

    //    __m512 sum0 = _mm512_setzero_ps();
    //    __m512 sum1 = _mm512_setzero_ps();
    //    __m512 sum2 = _mm512_setzero_ps();
    //    __m512 sum3 = _mm512_setzero_ps();

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

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

}  // namespace rabitqlib::simd
