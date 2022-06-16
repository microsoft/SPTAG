// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/DistanceUtils.h"
#include <immintrin.h>

using namespace SPTAG;
using namespace SPTAG::COMMON;

#ifndef _MSC_VER
#define DIFF128 diff128
#define DIFF256 diff256
#else
#define DIFF128 diff128.m128_f32
#define DIFF256 diff256.m256_f32
#endif

inline __m128 _mm_mul_epi8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}

inline __m128 _mm_sqdf_epi8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
}

inline __m128 _mm_mul_epu8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}

inline __m128 _mm_sqdf_epu8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
}

inline __m128 _mm_mul_epi16(__m128i X, __m128i Y)
{
    return _mm_cvtepi32_ps(_mm_madd_epi16(X, Y));
}

inline __m128 _mm_sqdf_epi16(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi16(X, zero);
    __m128i sign_y = _mm_cmplt_epi16(Y, zero);

    __m128i xlo = _mm_unpacklo_epi16(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi16(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi16(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi16(Y, sign_y);

    __m128 dlo = _mm_cvtepi32_ps(_mm_sub_epi32(xlo, ylo));
    __m128 dhi = _mm_cvtepi32_ps(_mm_sub_epi32(xhi, yhi));

    return _mm_add_ps(_mm_mul_ps(dlo, dlo), _mm_mul_ps(dhi, dhi));
}

inline __m128 _mm_sqdf_ps(__m128 X, __m128 Y)
{
    __m128 d = _mm_sub_ps(X, Y);
    return _mm_mul_ps(d, d);
}

inline __m256 _mm256_mul_epi8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

inline __m256 _mm256_sqdf_epi8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
}

inline __m256 _mm256_mul_epu8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

inline __m256 _mm256_sqdf_epu8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
}

inline __m256 _mm256_mul_epi16(__m256i X, __m256i Y)
{
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(X, Y));
}

inline __m256 _mm256_sqdf_epi16(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi16(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi16(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi16(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi16(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi16(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi16(Y, sign_y);

    __m256 dlo = _mm256_cvtepi32_ps(_mm256_sub_epi32(xlo, ylo));
    __m256 dhi = _mm256_cvtepi32_ps(_mm256_sub_epi32(xhi, yhi));

    return _mm256_add_ps(_mm256_mul_ps(dlo, dlo), _mm256_mul_ps(dhi, dhi));
}

inline __m256 _mm256_sqdf_ps(__m256 X, __m256 Y)
{
    __m256 d = _mm256_sub_ps(X, Y);
    return _mm256_mul_ps(d, d);
}

// Do not use intrinsics not supported by old MS compiler version
#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
inline __m512 _mm512_mul_epi8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask (zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

inline __m512 _mm512_sqdf_epi8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask (zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
}

inline __m512 _mm512_mul_epu8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

inline __m512 _mm512_sqdf_epu8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
}

inline __m512 _mm512_mul_epi16(__m512i X, __m512i Y)
{
    return _mm512_cvtepi32_ps(_mm512_madd_epi16(X, Y));
}

inline __m512 _mm512_sqdf_epi16(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask32 sign_x_mask = _mm512_cmpgt_epi16_mask (zero, X);
    __mmask32 sign_y_mask = _mm512_cmpgt_epi16_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi16(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi16(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi16(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi16(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi16(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi16(Y, sign_y);

    __m512 dlo = _mm512_cvtepi32_ps(_mm512_sub_epi32(xlo, ylo));
    __m512 dhi = _mm512_cvtepi32_ps(_mm512_sub_epi32(xhi, yhi));

    return _mm512_add_ps(_mm512_mul_ps(dlo, dlo), _mm512_mul_ps(dhi, dhi));
}

inline __m512 _mm512_sqdf_ps(__m512 X, __m512 Y)
{
    __m512 d = _mm512_sub_ps(X, Y);
    return _mm512_mul_ps(d, d);
}
#endif


#define REPEAT(type, ctype, delta, load, exec, acc, result) \
            { \
                type c1 = load((ctype *)(pX)); \
                type c2 = load((ctype *)(pY)); \
                pX += delta; pY += delta; \
                result = acc(result, exec(c1, c2)); \
            } \

float DistanceUtils::ComputeL2Distance_SSE(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::uint8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_SSE(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_sqdf_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int16_t* pEnd32 = pX + ((length >> 5) << 5);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m512i, __m512i, 32, _mm512_loadu_si512, _mm512_sqdf_epi16, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_sqdf_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_SSE(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
            REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd8 = pX + ((length >> 3) << 3);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const float* pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_sqdf_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd8)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeCosineDistance_SSE(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_mul_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}

float DistanceUtils::ComputeCosineDistance_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::uint8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_mul_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}

float DistanceUtils::ComputeCosineDistance_SSE(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_mul_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int16_t* pEnd32 = pX + ((length >> 5) << 5);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m512i, __m512i, 32, _mm512_loadu_si512, _mm512_mul_epi16, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_mul_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}

float DistanceUtils::ComputeCosineDistance_SSE(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
            REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd8 = pX + ((length >> 3) << 3);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const float* pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_mul_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd8)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}
