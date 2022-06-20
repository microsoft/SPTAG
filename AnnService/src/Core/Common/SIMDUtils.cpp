// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/SIMDUtils.h"
#include <immintrin.h>

using namespace SPTAG;
using namespace SPTAG::COMMON;

void SIMDUtils::ComputeSum_SSE(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

     while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_SSE(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
     const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_SSE(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi16(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
     const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi16(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
     const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi16(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_SSE(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4= pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        __m128 x_part = _mm_loadu_ps(pX);
        __m128 y_part = _mm_loadu_ps(pY);
        x_part = _mm_add_ps(x_part, y_part);
        _mm_storeu_ps(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    } 
}

void SIMDUtils::ComputeSum_AVX(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4= pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        __m128 x_part = _mm_loadu_ps(pX);
        __m128 y_part = _mm_loadu_ps(pY);
        x_part = _mm_add_ps(x_part, y_part);
        _mm_storeu_ps(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4= pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        __m128 x_part = _mm_loadu_ps(pX);
        __m128 y_part = _mm_loadu_ps(pY);
        x_part = _mm_add_ps(x_part, y_part);
        _mm_storeu_ps(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}
