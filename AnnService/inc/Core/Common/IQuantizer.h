// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUANTIZER_H_
#define _SPTAG_COMMON_QUANTIZER_H_

#include "inc/Core/Common.h"
#include <cstdint>
#include "inc/Core/CommonDataStructure.h"
#include "DistanceUtils.h"

namespace SPTAG
{
    namespace COMMON
    {
        class IQuantizer
        {
        public:
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const = 0;

            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const = 0;

            template <typename T>
            std::function<float(const T*, const T*, SizeType)> DistanceCalcSelector(SPTAG::DistCalcMethod p_method) const;

            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout) const = 0;

            virtual SizeType QuantizeSize() const = 0;

            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const = 0;

            virtual SizeType ReconstructSize() const = 0;

            virtual DimensionType ReconstructDim() const = 0;

            virtual std::uint64_t BufferSize() const = 0;

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const = 0;

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) = 0;

            virtual ErrorCode LoadQuantizer(uint8_t* raw_bytes) = 0;

            static std::shared_ptr<IQuantizer> LoadIQuantizer(std::shared_ptr<Helper::DiskIO> p_in);

            static std::shared_ptr<IQuantizer> LoadIQuantizer(SPTAG::ByteArray bytes);

            virtual bool GetEnableADC() const = 0;

            virtual void SetEnableADC(bool enableADC) = 0;

            virtual QuantizerType GetQuantizerType() const = 0;

            virtual VectorValueType GetReconstructType() const = 0;

            virtual DimensionType GetNumSubvectors() const = 0;

            virtual int GetBase() const = 0;

            virtual float* GetL2DistanceTables() = 0;
        };
    }
}

#endif // _SPTAG_COMMON_QUANTIZER_H_ 
