#ifndef _SPTAG_BKT_DISTANCEUTILS_H_
#define _SPTAG_BKT_DISTANCEUTILS_H_

#include <immintrin.h>
#include "CommonUtils.h"

#include <functional>

#define AVX

#ifndef _MSC_VER
#define DIFF128 diff128
#define DIFF256 diff256
#else
#define DIFF128 diff128.m128_f32
#define DIFF256 diff256.m256_f32
#endif

namespace SPTAG
{
    namespace BKT
    {

        class DistanceUtils
        {
        public:
            static inline __m128 _mm_mul_epi8(__m128i X, __m128i Y)
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
            static inline __m128 _mm_mul_epi16(__m128i X, __m128i Y)
            {
                return _mm_cvtepi32_ps(_mm_madd_epi16(X, Y));
            }

#if defined(AVX)
            static inline __m256 _mm256_mul_epi8(__m256i X, __m256i Y)
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
            static inline __m256 _mm256_mul_epi16(__m256i X, __m256i Y)
            {
                return _mm256_cvtepi32_ps(_mm256_madd_epi16(X, Y));
            }
#endif
/*
            template<typename T>
            static float ComputeL2Distance(const T *pX, const T *pY, int length)
            {
                float diff = 0;
                const T* pEnd1 = pX + length;
                while (pX < pEnd1) {
                    float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
                }
                return diff;
            }
*/
            static float ComputeL2Distance(const std::int8_t *pX, const std::int8_t *pY, int length)
            {
                const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
                const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int8_t* pEnd1 = pX + length;
#if defined(SSE)
                __m128 diff128 = _mm_setzero_ps();
                while (pX < pEnd32) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 16; pY += 16;
                    c1 = _mm_subs_epi8(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi8(c1, c1));

                    c1 = _mm_loadu_si128((__m128i *)(pX));
                    c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 16; pY += 16;
                    c1 = _mm_subs_epi8(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi8(c1, c1));
                }
                while (pX < pEnd16) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 16; pY += 16;
                    c1 = _mm_subs_epi8(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi8(c1, c1));
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX)
                __m256 diff256 = _mm256_setzero_ps();
                while (pX < pEnd32) {
                    __m256i c1 = _mm256_loadu_si256((__m256i *)(pX));
                    __m256i c2 = _mm256_loadu_si256((__m256i *)(pY));
                    pX += 32; pY += 32;
                    c1 = _mm256_subs_epi8(c1, c2);
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_epi8(c1, c1));
                }
                __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd16) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 16; pY += 16;
                    c1 = _mm_subs_epi8(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi8(c1, c1));
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

            static float ComputeL2Distance(const std::int16_t *pX, const std::int16_t *pY, int length)
            {
                const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
                const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int16_t* pEnd1 = pX + length;
#if defined(SSE)
                __m128 diff128 = _mm_setzero_ps();
                while (pX < pEnd16) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 8; pY += 8;
                    c1 = _mm_subs_epi16(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi16(c1, c1));

                    c1 = _mm_loadu_si128((__m128i *)(pX));
                    c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 8; pY += 8;
                    c1 = _mm_subs_epi16(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi16(c1, c1));
                }
                while (pX < pEnd8) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 8; pY += 8;
                    c1 = _mm_subs_epi16(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi16(c1, c1));
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX)
                __m256 diff256 = _mm256_setzero_ps();
                while (pX < pEnd16) {
                    __m256i c1 = _mm256_loadu_si256((__m256i *)(pX));
                    __m256i c2 = _mm256_loadu_si256((__m256i *)(pY));
                    pX += 16; pY += 16;
                    c1 = _mm256_subs_epi16(c1, c2);
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_epi16(c1, c1));
                }
                __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd8) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 8; pY += 8;
                    c1 = _mm_subs_epi16(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi16(c1, c1));
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

            static float ComputeL2Distance(const float *pX, const float *pY, int length)
            {
                const float* pEnd8 = pX + ((length >> 3) << 3);
                const float* pEnd4 = pX + ((length >> 2) << 2);
                const float* pEnd1 = pX + length;
#if defined(SSE)
                __m128 diff128 = _mm_setzero_ps();
                while (pX < pEnd8)
                {
                    __m128 c1 = _mm_loadu_ps(pX);
                    __m128 c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    c1 = _mm_sub_ps(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c1));

                    c1 = _mm_loadu_ps(pX);
                    c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    c1 = _mm_sub_ps(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c1));
                }
                while (pX < pEnd4)
                {
                    __m128 c1 = _mm_loadu_ps(pX);
                    __m128 c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    c1 = _mm_sub_ps(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c1));
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX)
                __m256 diff256 = _mm256_setzero_ps();
                while (pX < pEnd8)
                {
                    __m256 c1 = _mm256_loadu_ps(pX);
                    __m256 c2 = _mm256_loadu_ps(pY);
                    pX += 8; pY += 8;
                    c1 = _mm256_sub_ps(c1, c2);
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_ps(c1, c1));
                }
                __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd4)
                {
                    __m128 c1 = _mm_loadu_ps(pX);
                    __m128 c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    c1 = _mm_sub_ps(c1, c2);
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c1));
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
            static float ComputeCosineDistance(const T *pX, const T *pY, int length) {
                float diff = 0;
                const T* pEnd1 = pX + length;
                while (pX < pEnd1) diff += (*pX++) * (*pY++);
                return 1 - diff;
            }
*/
            static float ComputeCosineDistance(const std::int8_t *pX, const std::int8_t *pY, int length) {
                const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
                const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int8_t* pEnd1 = pX + length;
#if defined(SSE)

                __m128 diff128 = _mm_setzero_ps();
                while (pX < pEnd32) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 16; pY += 16;
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi8(c1, c2));

                    c1 = _mm_loadu_si128((__m128i *)(pX));
                    c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 16; pY += 16;
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi8(c1, c2));
                }
                while (pX < pEnd16) {
                    __m128 c1 = _mm_mul_epi8(_mm_loadu_si128((__m128i *)(pX)), _mm_loadu_si128((__m128i *)(pY)));
                    pX += 16; pY += 16;
                    diff128 = _mm_add_ps(diff128, c1);
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];
#elif defined(AVX)
                __m256 diff256 = _mm256_setzero_ps();
                while (pX < pEnd32) {
                    __m256i c1 = _mm256_loadu_si256((__m256i *)(pX));
                    __m256i c2 = _mm256_loadu_si256((__m256i *)(pY));
                    pX += 32; pY += 32;
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_epi8(c1, c2));
                }
                __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd16) {
                    __m128 c1 = _mm_mul_epi8(_mm_loadu_si128((__m128i *)(pX)), _mm_loadu_si128((__m128i *)(pY)));
                    pX += 16; pY += 16;
                    diff128 = _mm_add_ps(diff128, c1);
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

            static float ComputeCosineDistance(const std::int16_t *pX, const std::int16_t *pY, int length) {
                const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
                const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
                const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
                const std::int16_t* pEnd1 = pX + length;
#if defined(SSE)
                __m128 diff128 = _mm_setzero_ps();
                while (pX < pEnd16) {
                    __m128i c1 = _mm_loadu_si128((__m128i *)(pX));
                    __m128i c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 8; pY += 8;
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi16(c1, c2));

                    c1 = _mm_loadu_si128((__m128i *)(pX));
                    c2 = _mm_loadu_si128((__m128i *)(pY));
                    pX += 8; pY += 8;
                    diff128 = _mm_add_ps(diff128, _mm_mul_epi16(c1, c2));
                }
                while (pX < pEnd8) {
                    __m128 c1 = _mm_mul_epi16(_mm_loadu_si128((__m128i *)(pX)), _mm_loadu_si128((__m128i *)(pY)));
                    pX += 8; pY += 8;
                    diff128 = _mm_add_ps(diff128, c1);
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

#elif defined(AVX)
                __m256 diff256 = _mm256_setzero_ps();
                while (pX < pEnd16) {
                    __m256i c1 = _mm256_loadu_si256((__m256i *)(pX));
                    __m256i c2 = _mm256_loadu_si256((__m256i *)(pY));
                    pX += 16; pY += 16;
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_epi16(c1, c2));
                }
                __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd8) {
                    __m128 c1 = _mm_mul_epi16(_mm_loadu_si128((__m128i *)(pX)), _mm_loadu_si128((__m128i *)(pY)));
                    pX += 8; pY += 8;
                    diff128 = _mm_add_ps(diff128, c1);
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

            static float ComputeCosineDistance(const float *pX, const float *pY, int length) {
                const float* pEnd16 = pX + ((length >> 4) << 4);
                const float* pEnd4 = pX + ((length >> 2) << 2);
                const float* pEnd1 = pX + length;
#if defined(SSE)
                __m128 diff128 = _mm_setzero_ps();
                while (pX < pEnd16)
                {
                    __m128 c1 = _mm_loadu_ps(pX);
                    __m128 c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c2));

                    c1 = _mm_loadu_ps(pX);
                    c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c2));

                    c1 = _mm_loadu_ps(pX);
                    c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c2));

                    c1 = _mm_loadu_ps(pX);
                    c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c2));
                }
                while (pX < pEnd4)
                {
                    __m128 c1 = _mm_loadu_ps(pX);
                    __m128 c2 = _mm_loadu_ps(pY);
                    pX += 4; pY += 4;
                    diff128 = _mm_add_ps(diff128, _mm_mul_ps(c1, c2));
                }
                float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

#elif defined(AVX)
                __m256 diff256 = _mm256_setzero_ps();
                while (pX < pEnd16)
                {
                    __m256 c1 = _mm256_loadu_ps(pX);
                    __m256 c2 = _mm256_loadu_ps(pY);
                    pX += 8; pY += 8;
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_ps(c1, c2));

                    c1 = _mm256_loadu_ps(pX);
                    c2 = _mm256_loadu_ps(pY);
                    pX += 8; pY += 8;
                    diff256 = _mm256_add_ps(diff256, _mm256_mul_ps(c1, c2));
                }
                __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
                while (pX < pEnd4)
                {
                    __m128 c1 = _mm_mul_ps(_mm_loadu_ps(pX), _mm_loadu_ps(pY));
                    pX += 4; pY += 4;
                    diff128 = _mm_add_ps(diff128, c1);
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
            static inline float ComputeDistance(const T *p1, const T *p2, int length, SPTAG::DistCalcMethod distCalcMethod)
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
        float (*DistanceCalcSelector(SPTAG::DistCalcMethod p_method)) (const T*, const T*, int)
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

#endif // _SPTAG_BKT_DISTANCEUTILS_H_
