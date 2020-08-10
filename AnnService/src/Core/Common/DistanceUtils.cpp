#include "inc/Core/Common/DistanceUtils.h"

namespace SPTAG
{
	namespace COMMON
	{
		template<>
		float (*DistanceCalcSelector(SPTAG::DistCalcMethod p_method)) (const float*, const float*, DimensionType) {
			switch (p_method)
			{
			case SPTAG::DistCalcMethod::Cosine:
				if (InstructionSet::AVX())
				{
					return &(DistanceUtils::ComputeCosineDistance<SPTAG_AVX>);
				}
				else if (InstructionSet::SSE())
				{
					return &(DistanceUtils::ComputeCosineDistance<SPTAG_SSE>);
				}
				else {
					return &(DistanceUtils::ComputeCosineDistance<SPTAG_ELSE>);
				}

			case SPTAG::DistCalcMethod::L2:
				if (InstructionSet::AVX())
				{
					return &(DistanceUtils::ComputeL2Distance<SPTAG_AVX>);
				}
				else if (InstructionSet::SSE())
				{
					return &(DistanceUtils::ComputeL2Distance<SPTAG_SSE>);
				}
				else {
					return &(DistanceUtils::ComputeL2Distance<SPTAG_ELSE>);
				}

			default:
				break;
			}

			return nullptr;
		}

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_SSE2>(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_AVX2>(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_SSE2>(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_AVX2>(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_SSE2>(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_AVX2>(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_SSE>(const float* pX, const float* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeL2Distance<SPTAG_AVX>(const float* pX, const float* pY, DimensionType length)
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_SSE2>(const std::int8_t* pX, const std::int8_t* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_AVX2>(const std::int8_t* pX, const std::int8_t* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_SSE2>(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_AVX2>(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_SSE2>(const std::int16_t* pX, const std::int16_t* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_AVX2>(const std::int16_t* pX, const std::int16_t* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_SSE>(const float* pX, const float* pY, DimensionType length) {
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

		template<>
		float DistanceUtils::ComputeCosineDistance<SPTAG_AVX>(const float* pX, const float* pY, DimensionType length) {
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
	}
}
