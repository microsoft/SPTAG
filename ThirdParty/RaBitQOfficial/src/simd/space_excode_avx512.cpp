#include <immintrin.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd::excode_ipimpl {

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
// avx512: only applicable for avx512
float ip16_fxu1_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0;
    __m512 sum = _mm512_setzero_ps();

    for (size_t i = 0; i < dim; i += 16) {
        __mmask16 mask = *reinterpret_cast<const __mmask16*>(compact_code);
        __m512 q = _mm512_loadu_ps(query);

        sum = _mm512_add_ps(_mm512_maskz_mov_ps(mask, q), sum);

        compact_code += 2;
        query += 16;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu2_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));

        __m128i vec_00_to_15 = _mm_and_si128(compact, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact, 6), mask);
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);

        compact_code += 16;
    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu3_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        compact_code += 16;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);

    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip16_fxu4_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0.0F;
    constexpr int64_t kMask = 0x0f0f0f0f0f0f0f0f;
    for (size_t i = 0; i < dim; i += 16) {
        int64_t compact = *reinterpret_cast<const int64_t*>(compact_code);
        int64_t code0 = compact & kMask;
        int64_t code1 = (compact >> 4) & kMask;

        __m128i c8 = _mm_set_epi64x(code1, code0);
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        sum = _mm512_fmadd_ps(cf, q, sum);

        compact_code += 8;
    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu5_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();


    float result = 0.0F;
    const __m128i mask = _mm_set1_epi8(0b1111);
    const __m128i top_mask = _mm_set1_epi8(0b10000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact4_1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i compact4_2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        compact_code += 32;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);

    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu6_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();

    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));

        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);

    }
    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip64_fxu7_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();


    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));
        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);

    }

    result = _mm512_reduce_add_ps(sum);

    return result;
}

float ip16_fxu8_avx512(
    const float* __restrict__ query, const uint8_t* __restrict__ code, size_t dim
) {
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < dim; i += 16) {
        __m128i c8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code));
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c8));
        sum = _mm512_fmadd_ps(cf, q, sum);
        code += 16;
    }
    return _mm512_reduce_add_ps(sum);
}

}  // namespace rabitqlib::simd::excode_ipimpl
