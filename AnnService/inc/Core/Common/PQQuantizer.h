// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_PQQUANTIZER_H_
#define _SPTAG_COMMON_PQQUANTIZER_H_

#include "CommonUtils.h"
#include "DistanceUtils.h"
#include <limits>


namespace SPTAG
{
	namespace COMMON
	{
		class PQQuantizer
		{
		public:
			PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, float*** Codebooks);		

			float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY);

			float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY);

			template<EnumInstruction ei>
			const std::uint8_t* QuantizeVector(const float* vec);

			void SaveQuantizer(std::string path);

			static void LoadQuantizer(std::string path);

		private:
			DimensionType m_NumSubvectors;
			SizeType m_KsPerSubvector;
			DimensionType m_DimPerSubvector;

			const float*** m_codebooks;
			float*** m_CosineDistanceTables;
			float*** m_L2DistanceTables;
		};
	}
}

#endif // _SPTAG_COMMON_PQQUANTIZER_H_