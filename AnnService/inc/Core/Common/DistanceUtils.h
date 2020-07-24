// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_DISTANCEUTILS_H_
#define _SPTAG_COMMON_DISTANCEUTILS_H_

#include <functional>

#include "CommonUtils.h"
#include "../../../../simde/simde/x86/avx.h"
#include "../../../../simde/simde/x86/avx2.h"

#if defined(__AVX2__)
#define AVX2
#elif defined(__SSE2__)
#define SSE2
#endif

#if defined(__AVX__)
#define AVX
#elif defined(__SSE__)
#define SSE
#endif


#ifndef _MSC_VER
#define DIFF128 diff128
#define DIFF256 diff256
#else
#define DIFF128 diff128.m128_f32
#define DIFF256 diff256.m256_f32
#endif

namespace SPTAG
{
    namespace COMMON
    {
        class DistanceUtils
        {
        public:
#if defined(SSE2) || defined(AVX2) || defined(__SIMDE_OPTIMIZATION__)

            static inline simde__m128 simde_mm_mul_epi8(simde__m128i X, simde__m128i Y)
            {
                simde__m128i zero = simde_mm_setzero_si128();

                simde__m128i sign_x = simde_mm_cmplt_epi8(X, zero);
                simde__m128i sign_y = simde_mm_cmplt_epi8(Y, zero);

                simde__m128i xlo = simde_mm_unpacklo_epi8(X, sign_x);
                simde__m128i xhi = simde_mm_unpackhi_epi8(X, sign_x);
                simde__m128i ylo = simde_mm_unpacklo_epi8(Y, sign_y);
                simde__m128i yhi = simde_mm_unpackhi_epi8(Y, sign_y);
                
                return simde_mm_cvtepi32_ps(simde_mm_add_epi32(simde_mm_madd_epi16(xlo, ylo), simde_mm_madd_epi16(xhi, yhi)));
            }

            static inline simde__m128 simde_mm_sqdf_epi8(simde__m128i X, simde__m128i Y)
            {
                simde__m128i zero = simde_mm_setzero_si128();

                simde__m128i sign_x = simde_mm_cmplt_epi8(X, zero);
                simde__m128i sign_y = simde_mm_cmplt_epi8(Y, zero);

                simde__m128i xlo = simde_mm_unpacklo_epi8(X, sign_x);
                simde__m128i xhi = simde_mm_unpackhi_epi8(X, sign_x);
                simde__m128i ylo = simde_mm_unpacklo_epi8(Y, sign_y);
                simde__m128i yhi = simde_mm_unpackhi_epi8(Y, sign_y);
                
                simde__m128i dlo = simde_mm_sub_epi16(xlo, ylo);
                simde__m128i dhi = simde_mm_sub_epi16(xhi, yhi);

                return simde_mm_cvtepi32_ps(simde_mm_add_epi32(simde_mm_madd_epi16(dlo, dlo), simde_mm_madd_epi16(dhi, dhi)));
            }

            static inline simde__m128 simde_mm_mul_epu8(simde__m128i X, simde__m128i Y)
            {
                simde__m128i zero = simde_mm_setzero_si128();

                simde__m128i xlo = simde_mm_unpacklo_epi8(X, zero);
                simde__m128i xhi = simde_mm_unpackhi_epi8(X, zero);
                simde__m128i ylo = simde_mm_unpacklo_epi8(Y, zero);
                simde__m128i yhi = simde_mm_unpackhi_epi8(Y, zero);

                return simde_mm_cvtepi32_ps(simde_mm_add_epi32(simde_mm_madd_epi16(xlo, ylo), simde_mm_madd_epi16(xhi, yhi)));
            }

            static inline simde__m128 simde_mm_sqdf_epu8(simde__m128i X, simde__m128i Y)
            {
                simde__m128i zero = simde_mm_setzero_si128();

                simde__m128i xlo = simde_mm_unpacklo_epi8(X, zero);
                simde__m128i xhi = simde_mm_unpackhi_epi8(X, zero);
                simde__m128i ylo = simde_mm_unpacklo_epi8(Y, zero);
                simde__m128i yhi = simde_mm_unpackhi_epi8(Y, zero);

                simde__m128i dlo = simde_mm_sub_epi16(xlo, ylo);
                simde__m128i dhi = simde_mm_sub_epi16(xhi, yhi);

                return simde_mm_cvtepi32_ps(simde_mm_add_epi32(simde_mm_madd_epi16(dlo, dlo), simde_mm_madd_epi16(dhi, dhi)));
            }

            static inline simde__m128 simde_mm_mul_epi16(simde__m128i X, simde__m128i Y)
            {
                return simde_mm_cvtepi32_ps(simde_mm_madd_epi16(X, Y));
            }

            static inline simde__m128 simde_mm_sqdf_epi16(simde__m128i X, simde__m128i Y)
            {
                simde__m128i zero = simde_mm_setzero_si128();

                simde__m128i sign_x = simde_mm_cmplt_epi16(X, zero);
                simde__m128i sign_y = simde_mm_cmplt_epi16(Y, zero);

                simde__m128i xlo = simde_mm_unpacklo_epi16(X, sign_x);
                simde__m128i xhi = simde_mm_unpackhi_epi16(X, sign_x);
                simde__m128i ylo = simde_mm_unpacklo_epi16(Y, sign_y);
                simde__m128i yhi = simde_mm_unpackhi_epi16(Y, sign_y);

                simde__m128 dlo = simde_mm_cvtepi32_ps(simde_mm_sub_epi32(xlo, ylo));
                simde__m128 dhi = simde_mm_cvtepi32_ps(simde_mm_sub_epi32(xhi, yhi));

                return simde_mm_add_ps(simde_mm_mul_ps(dlo, dlo), simde_mm_mul_ps(dhi, dhi));
            }
#endif
#if defined(SSE) || defined(AVX) || defined(__SIMDE_OPTIMIZATION__)
            static inline simde__m128 simde_mm_sqdf_ps(simde__m128 X, simde__m128 Y)
            {
                simde__m128 d = simde_mm_sub_ps(X, Y);
                return simde_mm_mul_ps(d, d);
            }
#endif
#if defined(AVX2)
            static inline simde__m256 simde_mm256_mul_epi8(simde__m256i X, simde__m256i Y)
            {
                simde__m256i zero = simde_mm256_setzero_si256();

                simde__m256i sign_x = simde_mm256_cmpgt_epi8(zero, X);
                simde__m256i sign_y = simde_mm256_cmpgt_epi8(zero, Y);

                simde__m256i xlo = simde_mm256_unpacklo_epi8(X, sign_x);
                simde__m256i xhi = simde_mm256_unpackhi_epi8(X, sign_x);
                simde__m256i ylo = simde_mm256_unpacklo_epi8(Y, sign_y);
                simde__m256i yhi = simde_mm256_unpackhi_epi8(Y, sign_y);

                return simde_mm256_cvtepi32_ps(simde_mm256_add_epi32(simde_mm256_madd_epi16(xlo, ylo), simde_mm256_madd_epi16(xhi, yhi))); 
            }
            static inline simde__m256 simde_mm256_sqdf_epi8(simde__m256i X, simde__m256i Y)
            {
                simde__m256i zero = simde_mm256_setzero_si256();

                simde__m256i sign_x = simde_mm256_cmpgt_epi8(zero, X);
                simde__m256i sign_y = simde_mm256_cmpgt_epi8(zero, Y);

                simde__m256i xlo = simde_mm256_unpacklo_epi8(X, sign_x);
                simde__m256i xhi = simde_mm256_unpackhi_epi8(X, sign_x);
                simde__m256i ylo = simde_mm256_unpacklo_epi8(Y, sign_y);
                simde__m256i yhi = simde_mm256_unpackhi_epi8(Y, sign_y);

                simde__m256i dlo = simde_mm256_sub_epi16(xlo, ylo);
                simde__m256i dhi = simde_mm256_sub_epi16(xhi, yhi);

                return simde_mm256_cvtepi32_ps(simde_mm256_add_epi32(simde_mm256_madd_epi16(dlo, dlo), simde_mm256_madd_epi16(dhi, dhi)));
            }
            static inline simde__m256 simde_mm256_mul_epu8(simde__m256i X, simde__m256i Y)
            {
                simde__m256i zero = simde_mm256_setzero_si256();

                simde__m256i xlo = simde_mm256_unpacklo_epi8(X, zero);
                simde__m256i xhi = simde_mm256_unpackhi_epi8(X, zero);
                simde__m256i ylo = simde_mm256_unpacklo_epi8(Y, zero);
                simde__m256i yhi = simde_mm256_unpackhi_epi8(Y, zero);

                return simde_mm256_cvtepi32_ps(simde_mm256_add_epi32(simde_mm256_madd_epi16(xlo, ylo), simde_mm256_madd_epi16(xhi, yhi)));
            }
            static inline simde__m256 simde_mm256_sqdf_epu8(simde__m256i X, simde__m256i Y)
            {
                simde__m256i zero = simde_mm256_setzero_si256();

                simde__m256i xlo = simde_mm256_unpacklo_epi8(X, zero);
                simde__m256i xhi = simde_mm256_unpackhi_epi8(X, zero);
                simde__m256i ylo = simde_mm256_unpacklo_epi8(Y, zero);
                simde__m256i yhi = simde_mm256_unpackhi_epi8(Y, zero);

                simde__m256i dlo = simde_mm256_sub_epi16(xlo, ylo);
                simde__m256i dhi = simde_mm256_sub_epi16(xhi, yhi);

                return simde_mm256_cvtepi32_ps(simde_mm256_add_epi32(simde_mm256_madd_epi16(dlo, dlo), simde_mm256_madd_epi16(dhi, dhi)));
            }
            static inline simde__m256 simde_mm256_mul_epi16(simde__m256i X, simde__m256i Y)
            {
                return simde_mm256_cvtepi32_ps(simde_mm256_madd_epi16(X, Y));
            }
            static inline simde__m256 simde_mm256_sqdf_epi16(simde__m256i X, simde__m256i Y)
            {
                simde__m256i zero = simde_mm256_setzero_si256();

                simde__m256i sign_x = simde_mm256_cmpgt_epi16(zero, X);
                simde__m256i sign_y = simde_mm256_cmpgt_epi16(zero, Y);

                simde__m256i xlo = simde_mm256_unpacklo_epi16(X, sign_x);
                simde__m256i xhi = simde_mm256_unpackhi_epi16(X, sign_x);
                simde__m256i ylo = simde_mm256_unpacklo_epi16(Y, sign_y);
                simde__m256i yhi = simde_mm256_unpackhi_epi16(Y, sign_y);

                simde__m256 dlo = simde_mm256_cvtepi32_ps(simde_mm256_sub_epi32(xlo, ylo));
                simde__m256 dhi = simde_mm256_cvtepi32_ps(simde_mm256_sub_epi32(xhi, yhi));

                return simde_mm256_add_ps(simde_mm256_mul_ps(dlo, dlo), simde_mm256_mul_ps(dhi, dhi));
            }
#endif
#if defined(AVX)
            static inline simde__m256 simde_mm256_sqdf_ps(simde__m256 X, simde__m256 Y)
            {
                simde__m256 d = simde_mm256_sub_ps(X, Y);
                return simde_mm256_mul_ps(d, d);
            }
#endif
/*
            template<typename T>
            static float ComputeL2Distance(const T *pX, const T *pY, DimensionType length)
            {
                float diff = 0;
                const T* pEnd1 = pX + length;
                while (pX < pEnd1) {
                    float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
                }
                return diff;
            }
*/
#define REPEAT(type, ctype, delta, load, exec, acc, result) \
            { \
                type c1 = load((ctype *)(pX)); \
                type c2 = load((ctype *)(pY)); \
                pX += delta; pY += delta; \
                result = acc(result, exec(c1, c2)); \
            } \

            static float ComputeL2Distance(const std::int8_t *pX, const std::int8_t *pY, DimensionType length)
            {
                const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
                const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int8_t* pEnd1 = pX + length;
#if defined(SSE2)
                simde__m128 diff128 = simde_mm_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epi8, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epi8, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epi8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX2)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m256i, simde__m256i, 32, simde_mm256_loadu_si256, simde_mm256_sqdf_epi8, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 = simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epi8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
#endif
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

            static float ComputeL2Distance(const std::uint8_t *pX, const std::uint8_t *pY, DimensionType length)
            {
                const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
                const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::uint8_t* pEnd1 = pX + length;
#if defined(SSE2)
                simde__m128 diff128 = simde_mm_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epu8, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epu8, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epu8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX2)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m256i, simde__m256i, 32, simde_mm256_loadu_si256, simde_mm256_sqdf_epu8, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 = simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_sqdf_epu8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
#endif
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

            static float ComputeL2Distance(const std::int16_t *pX, const std::int16_t *pY, DimensionType length)
            {
                const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
                const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int16_t* pEnd1 = pX + length;
#if defined(SSE2)
                simde__m128 diff128 = simde_mm_setzero_ps();
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_sqdf_epi16, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_sqdf_epi16, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd8) {
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_sqdf_epi16, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX2)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd16) {
                    REPEAT(simde__m256i, simde__m256i, 16, simde_mm256_loadu_si256, simde_mm256_sqdf_epi16, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 = simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd8) {
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_sqdf_epi16, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
#endif
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

            static float ComputeL2Distance(const float *pX, const float *pY, DimensionType length)
            {
                const float* pEnd16 = pX + ((length >> 4) << 4);
                const float* pEnd4 = pX + ((length >> 2) << 2);
                const float* pEnd1 = pX + length;
#if defined(SSE2)
                simde__m128 diff128 = simde_mm_setzero_ps();
                while (pX < pEnd16)
                {
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_sqdf_ps, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_sqdf_ps, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_sqdf_ps, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_sqdf_ps, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd4)
                {
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_sqdf_ps, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd16)
                {
                    REPEAT(simde__m256, const float, 8, simde_mm256_loadu_ps, simde_mm256_sqdf_ps, simde_mm256_add_ps, diff256)
                    REPEAT(simde__m256, const float, 8, simde_mm256_loadu_ps, simde_mm256_sqdf_ps, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 = simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd4)
                {
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_sqdf_ps, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
                while (pX < pEnd4) {
                    float c1 = (*pX++) - (*pY++); diff += c1 * c1;
                    c1 = (*pX++) - (*pY++); diff += c1 * c1;
                    c1 = (*pX++) - (*pY++); diff += c1 * c1;
                    c1 = (*pX++) - (*pY++); diff += c1 * c1;
                }
#endif
                while (pX < pEnd1) {
                    float c1 = (*pX++) - (*pY++); diff += c1 * c1;
                }
                return diff;
            }
/*
            template<typename T>
            static float ComputeCosineDistance(const T *pX, const T *pY, DimensionType length) {
                float diff = 0;
                const T* pEnd1 = pX + length;
                while (pX < pEnd1) diff += (*pX++) * (*pY++);
                return 1 - diff;
            }
*/
            static float ComputeCosineDistance(const std::int8_t *pX, const std::int8_t *pY, DimensionType length) {
                const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
                const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int8_t* pEnd1 = pX + length;
#if defined(SSE2)

                simde__m128 diff128 = simde_mm_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epi8, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epi8, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epi8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX2)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m256i, simde__m256i, 32, simde_mm256_loadu_si256, simde_mm256_mul_epi8, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 = simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epi8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
#endif
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

            static float ComputeCosineDistance(const std::uint8_t *pX, const std::uint8_t *pY, DimensionType length) {
                const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
                const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::uint8_t* pEnd1 = pX + length;
#if defined(SSE2)

                simde__m128 diff128 =simde_mm_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epu8, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epu8, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16, simde_mm_loadu_si128, simde_mm_mul_epu8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX2)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd32) {
                    REPEAT(simde__m256i, simde__m256i, 32, simde_mm256_loadu_si256, simde_mm256_mul_epu8, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 =simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 16,simde_mm_loadu_si128, simde_mm_mul_epu8, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
#endif
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

    static float ComputeCosineDistance(const std::int16_t *pX, const std::int16_t *pY, DimensionType length) {
                const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
                const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int16_t* pEnd1 = pX + length;
#if defined(SSE2)
                simde__m128 diff128 = simde_mm_setzero_ps();
                while (pX < pEnd16) {
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_mul_epi16, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_mul_epi16, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd8) {
                    REPEAT(simde__m128i, simde__m128i, 8, simde_mm_loadu_si128, simde_mm_mul_epi16, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

#elif defined(AVX2)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd16) {
                    REPEAT(simde__m256i, simde__m256i, 16, simde_mm256_loadu_si256, simde_mm256_mul_epi16, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 =simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd8) {
                    REPEAT(simde__m128i, simde__m128i, 8,simde_mm_loadu_si128, simde_mm_mul_epi16, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
#endif
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

            static float ComputeCosineDistance(const float *pX, const float *pY, DimensionType length) {
                const float* pEnd16 = pX + ((length >> 4) << 4);
                const float* pEnd4 = pX + ((length >> 2) << 2);
                const float* pEnd1 = pX + length;
#if defined(SSE2)
                simde__m128 diff128 =simde_mm_setzero_ps();
                while (pX < pEnd16)
                {
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_mul_ps, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_mul_ps, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_mul_ps, simde_mm_add_ps, diff128)
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_mul_ps, simde_mm_add_ps, diff128)
                }
                while (pX < pEnd4)
                {
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_mul_ps, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

#elif defined(AVX)
                simde__m256 diff256 = simde_mm256_setzero_ps();
                while (pX < pEnd16)
                {
                    REPEAT(simde__m256, const float, 8, simde_mm256_loadu_ps, simde_mm256_mul_ps, simde_mm256_add_ps, diff256)
                    REPEAT(simde__m256, const float, 8, simde_mm256_loadu_ps, simde_mm256_mul_ps, simde_mm256_add_ps, diff256)
                }
                simde__m128 diff128 =simde_mm_add_ps(simde_mm256_castps256_ps128(diff256), simde_mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd4)
                {
                    REPEAT(simde__m128, const float, 4, simde_mm_loadu_ps, simde_mm_mul_ps, simde_mm_add_ps, diff128)
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#else
                float diff = 0;
                while (pX < pEnd4)
                {
                    float c1 = (*pX++) * (*pY++); diff += c1;
                    c1 = (*pX++) * (*pY++); diff += c1;
                    c1 = (*pX++) * (*pY++); diff += c1;
                    c1 = (*pX++) * (*pY++); diff += c1;
                }
#endif
                while (pX < pEnd1) diff += (*pX++) * (*pY++);
                return 1 - diff;
            }

            template<typename T>
            static inline float ComputeDistance(const T *p1, const T *p2, DimensionType length, SPTAG::DistCalcMethod distCalcMethod)
            {
                if (distCalcMethod == SPTAG::DistCalcMethod::L2)
                    return ComputeL2Distance(p1, p2, length);

                return ComputeCosineDistance(p1, p2, length);
            }

            static inline float ConvertCosineSimilarityToDistance(float cs)
            {
                // Cosine similarity is in [-1, 1], the higher the value, the closer are the two vectors. 
                // However, the tree is built and searched based on "distance" between two vectors, that's >=0. The smaller the value, the closer are the two vectors.
                // So we do a linear conversion from a cosine similarity to a distance value.
                return 1 - cs; //[1, 3]
            }

            static inline float ConvertDistanceBackToCosineSimilarity(float d)
            {
                return 1 - d;
            }
        };


        template<typename T>
        float (*DistanceCalcSelector(SPTAG::DistCalcMethod p_method)) (const T*, const T*, DimensionType)
        {
            switch (p_method)
            {
            case SPTAG::DistCalcMethod::Cosine:
                return &(DistanceUtils::ComputeCosineDistance);

            case SPTAG::DistCalcMethod::L2:
                return &(DistanceUtils::ComputeL2Distance);

            default:
                break;
            }

            return nullptr;
        }
    }
}

#endif // _SPTAG_COMMON_DISTANCEUTILS_H_
