// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_PQQUANTIZER_H_
#define _SPTAG_COMMON_PQQUANTIZER_H_

#include "CommonUtils.h"
#include "DistanceUtils.h"
#include "Quantizer.h"
#include <limits>


namespace SPTAG
{
    namespace COMMON
    {
        class PQQuantizer : public Quantizer
        {
        public:
            PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, float*** Codebooks);	

            ~PQQuantizer();

            float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY);

            float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY);

            const std::uint8_t* QuantizeVector(const float* vec);

            void SaveQuantizer(std::string path);

            static void LoadQuantizer(std::string path);

            DimensionType GetNumSubvectors();

            SizeType GetKsPerSubvector();

            DimensionType GetDimPerSubvector();

            QuantizerType GetQuantizerType() {
                return QuantizerType::PQQuantizer;
            }

        private:
            DimensionType m_NumSubvectors;
            SizeType m_KsPerSubvector;
            DimensionType m_DimPerSubvector;
            SizeType m_BlockSize;

            inline SizeType m_DistIndexCalc(SizeType i, SizeType j, SizeType k);

            const float*** m_codebooks;
            float* m_CosineDistanceTables;
            float* m_L2DistanceTables;
        };
    }
}

#endif // _SPTAG_COMMON_PQQUANTIZER_H_