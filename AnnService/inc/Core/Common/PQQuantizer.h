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
            PQQuantizer();

            PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, float* Codebooks);

            ~PQQuantizer();

            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY);

            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY);

            virtual void QuantizeVector(const float* vec, std::uint8_t* vecout);

            virtual std::uint64_t BufferSize() const;

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const;

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in);

            virtual DimensionType GetNumSubvectors() const;

            SizeType GetKsPerSubvector() const;

            DimensionType GetDimPerSubvector() const;

            QuantizerType GetQuantizerType() {
                return QuantizerType::PQQuantizer;
            }

        private:
            DimensionType m_NumSubvectors;
            SizeType m_KsPerSubvector;
            DimensionType m_DimPerSubvector;
            SizeType m_BlockSize;

            inline SizeType m_DistIndexCalc(SizeType i, SizeType j, SizeType k);

            std::unique_ptr<float[]> m_codebooks;
            std::unique_ptr<float[]> m_CosineDistanceTables;
            std::unique_ptr<float[]> m_L2DistanceTables;
        };
    }
}

#endif // _SPTAG_COMMON_PQQUANTIZER_H_