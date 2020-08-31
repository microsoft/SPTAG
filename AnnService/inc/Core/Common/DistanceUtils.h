// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_DISTANCEUTILS_H_
#define _SPTAG_COMMON_DISTANCEUTILS_H_

#include <xmmintrin.h>
#include <functional>

#include "CommonUtils.h"
#include "InstructionUtils.h"

namespace SPTAG
{
	namespace COMMON
	{
		class PQQuantizer;


		template<typename T>
		float (*DistanceCalcSelector(SPTAG::DistCalcMethod p_method)) (const T*, const T*, DimensionType);

        class DistanceUtils
        {
        public:
            static std::shared_ptr<PQQuantizer> PQQuantizer;

            static float ComputeL2Distance(const std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static float ComputeL2Distance_SSE(const std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static float ComputeL2Distance_AVX(const std::int8_t* pX, const std::int8_t* pY, DimensionType length);

            static float ComputeL2Distance(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeL2Distance_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeL2Distance_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);

            static float ComputeL2Distance(const std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static float ComputeL2Distance_SSE(const std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static float ComputeL2Distance_AVX(const std::int16_t* pX, const std::int16_t* pY, DimensionType length);

            static float ComputeL2Distance(const float* pX, const float* pY, DimensionType length);
            static float ComputeL2Distance_SSE(const float* pX, const float* pY, DimensionType length);
            static float ComputeL2Distance_AVX(const float* pX, const float* pY, DimensionType length);

            static float ComputeCosineDistance(const std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static float ComputeCosineDistance_SSE(const std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static float ComputeCosineDistance_AVX(const std::int8_t* pX, const std::int8_t* pY, DimensionType length);

            static float ComputeCosineDistance(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeCosineDistance_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeCosineDistance_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);

            static float ComputeCosineDistance(const std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static float ComputeCosineDistance_SSE(const std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static float ComputeCosineDistance_AVX(const std::int16_t* pX, const std::int16_t* pY, DimensionType length);

            static float ComputeCosineDistance(const float* pX, const float* pY, DimensionType length);
            static float ComputeCosineDistance_SSE(const float* pX, const float* pY, DimensionType length);
            static float ComputeCosineDistance_AVX(const float* pX, const float* pY, DimensionType length);


            template<typename T>
            static inline float ComputeDistance(const T* p1, const T* p2, DimensionType length, SPTAG::DistCalcMethod distCalcMethod)
            {
                auto func = DistanceCalcSelector<T>(distCalcMethod);
                return func(p1, p2, length);
            }

			static float ConvertCosineSimilarityToDistance(float cs)
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

        private:
            static float ComputeCosineDistance_NonQuantized(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeCosineDistance_NonQuantized_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeCosineDistance_NonQuantized_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);

            static float ComputeL2Distance_NonQuantized(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeL2Distance_NonQuantized_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static float ComputeL2Distance_NonQuantized_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
        };

        template<typename T>
        float (*DistanceCalcSelector(SPTAG::DistCalcMethod p_method)) (const T*, const T*, DimensionType)
        {
            bool isSize4 = (sizeof(T) == 4);
            switch (p_method)
            {
            case SPTAG::DistCalcMethod::Cosine:
                if (InstructionSet::AVX2() || (isSize4 && InstructionSet::AVX()))
                {
                    return &(DistanceUtils::ComputeCosineDistance_AVX);
                }
                else if (InstructionSet::SSE2() || (isSize4 && InstructionSet::SSE()))
                {
                    return &(DistanceUtils::ComputeCosineDistance_SSE);
                }
                else {
                    return &(DistanceUtils::ComputeCosineDistance);
                }

            case SPTAG::DistCalcMethod::L2:
                if (InstructionSet::AVX2() || (isSize4 && InstructionSet::AVX()))
                {
                    return &(DistanceUtils::ComputeL2Distance_AVX);
                }
                else if (InstructionSet::SSE2() || (isSize4 && InstructionSet::SSE()))
                {
                    return &(DistanceUtils::ComputeL2Distance_SSE);
                }
                else {
                    return &(DistanceUtils::ComputeL2Distance);
                }
  
            default:
                break;
            }
            return nullptr;
        }
    }
}

#endif // _SPTAG_COMMON_DISTANCEUTILS_H_
