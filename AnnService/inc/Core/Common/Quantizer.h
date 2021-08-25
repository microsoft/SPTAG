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

            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) = 0;

            virtual std::uint64_t BufferSize() const = 0;

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const = 0;

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in) = 0;

            static ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in, QuantizerType quantizerType, VectorValueType reconstructType);

            virtual QuantizerType GetQuantizerType() = 0;

            virtual VectorValueType GetReconstructType() = 0;

            virtual DimensionType GetNumSubvectors() const = 0;
        };
    }
}

#endif // _SPTAG_COMMON_QUANTIZER_H_ 