// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUANTIZER_H_
#define _SPTAG_COMMON_QUANTIZER_H_

#include "../Common.h"
#include <cstdint>
#include "../CommonDataStructure.h"
#include "DistanceUtils.h"

namespace SPTAG
{
    namespace COMMON
    {
        class IQuantizer
        {
        public:
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) = 0;

            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) = 0;

            template <typename T>
            std::function<float(const T*, const T*, SizeType)> DistanceCalcSelector(SPTAG::DistCalcMethod p_method)
            {
                return SPTAG::COMMON::DistanceCalcSelector<T>(p_method);
            }

            template <>
            std::function<float(const std::uint8_t*, const std::uint8_t*, SizeType)> DistanceCalcSelector<std::uint8_t>(SPTAG::DistCalcMethod p_method)
            {
                if (p_method == SPTAG::DistCalcMethod::L2)
                {
                    return ([this](const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length) {return L2Distance(pX, pY); });
                }
                else
                {
                    return ([this](const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length) {return CosineDistance(pX, pY); });
                }
            }

            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout) = 0;

            virtual SizeType QuantizeSize() = 0;

            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) = 0;

            virtual SizeType ReconstructSize() = 0;

            virtual DimensionType ReconstructDim() = 0;

            virtual std::uint64_t BufferSize() const = 0;

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const = 0;

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in) = 0;

            virtual ErrorCode LoadQuantizer(uint8_t* raw_bytes) = 0;

            static std::shared_ptr<IQuantizer> LoadIQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in);

            static std::shared_ptr<IQuantizer> LoadIQuantizer(SPTAG::ByteArray bytes);

            virtual bool GetEnableADC() = 0;

            virtual void SetEnableADC(bool enableADC) = 0;

            virtual QuantizerType GetQuantizerType() = 0;

            virtual VectorValueType GetReconstructType() = 0;

            virtual DimensionType GetNumSubvectors() const = 0;

            virtual int GetBase() = 0;
        };
    }
}

#endif // _SPTAG_COMMON_QUANTIZER_H_ 