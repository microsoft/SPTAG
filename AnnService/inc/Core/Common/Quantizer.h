// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUANTIZER_H_
#define _SPTAG_COMMON_QUANTIZER_H_

#include "CommonUtils.h"

namespace SPTAG
{
    namespace COMMON
    {
        class Quantizer
        {
        public:
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) = 0;

            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) = 0;

            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout) = 0;

            virtual SizeType QuantizeSize() = 0;

            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) = 0;

            virtual SizeType ReconstructSize() = 0;

            virtual DimensionType ReconstructDim() = 0;

            virtual std::uint64_t BufferSize() const = 0;

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const = 0;

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in) = 0;

            static ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in, QuantizerType quantizerType, VectorValueType reconstructType);

            virtual bool GetEnableADC() = 0;

            virtual void SetEnableADC(bool enableADC) = 0;

            virtual QuantizerType GetQuantizerType() = 0;

            virtual VectorValueType GetReconstructType() = 0;

            virtual DimensionType GetNumSubvectors() const = 0;

            virtual float GetBase() = 0;
        };
    }
}

#endif // _SPTAG_COMMON_QUANTIZER_H_

/*
template <typename T>
void PQQuantizer<T>::GetADCDistanceTable(T* query, float* distance_table_out) {
    int block_size = m_KsPerSubvector * m_NumSubvectors;
    float* destPtr = distance_table_out;
    T* queryPtr = query;
    auto L2Dist = COMMON::DistanceCalcSelector<T>(DistCalcMethod::L2);
    for (int i = 0; i < m_DimPerSubvector; i++) {
        //T* basec = (T*)m_codebooks + i * block_size;
        for (int j = 0; j < m_KsPerSubvector; j++) {
            destPtr[j] = L2Dist(queryPtr, &m_codebooks[i * block_size + j * m_NumSubvectors], m_NumSubvectors);
        }
        destPtr += m_KsPerSubvector;
        queryPtr += m_NumSubvectors;
    }
}
*/