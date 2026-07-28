// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/Common.h"
#include "inc/Core/VectorSet.h"

#include "rabitqlib/utils/compiler_compat.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace SPTAG
{
namespace COMMON
{

class OfficialRaBitQ
{
public:
    struct QueryContext
    {
        std::vector<float> rotated;
        std::unique_ptr<rabitqlib::SplitSingleQuery<float>> query;
    };

    OfficialRaBitQ() = default;
    OfficialRaBitQ(DimensionType p_dimension, int p_totalBits);

    static bool IsSupportedDimension(DimensionType p_dimension);
    static bool IsSupportedPlatform();

    ErrorCode Train(const std::shared_ptr<VectorSet>& p_vectors);
    ErrorCode Save(const std::string& p_path) const;
    ErrorCode Load(const std::string& p_path);

    void Encode(const float* p_vector, std::uint8_t* p_code) const;
    QueryContext PrepareQuery(const float* p_query) const;
    float Estimate(const QueryContext& p_query, const std::uint8_t* p_code) const;

    bool Ready() const;
    DimensionType Dimension() const { return m_dimension; }
    DimensionType PaddedDimension() const { return m_paddedDimension; }
    int TotalBits() const { return m_exBits + 1; }
    int ExBits() const { return m_exBits; }
    std::size_t BinCodeBytes() const;
    std::size_t ExCodeBytes() const;
    std::size_t CodeBytes() const;

private:
    struct ModelHeader
    {
        std::uint32_t magic;
        std::uint32_t version;
        std::int32_t dimension;
        std::int32_t paddedDimension;
        std::int32_t exBits;
        std::uint32_t rotatorBytes;
    };

    static constexpr std::uint32_t kMagic = 0x3142524fU; // ORB1
    static constexpr std::uint32_t kVersion = 1U;

    ErrorCode Initialize(DimensionType p_dimension, int p_totalBits);
    void Rotate(const float* p_vector, std::vector<float>& p_rotated) const;

    DimensionType m_dimension = 0;
    DimensionType m_paddedDimension = 0;
    int m_exBits = -1;
    rabitqlib::quant::RabitqConfig m_quantizerConfig;
    rabitqlib::ex_ipfunc m_exCodeInnerProduct = nullptr;
    std::unique_ptr<rabitqlib::Rotator<float>> m_rotator;
    std::vector<float> m_centroid;
};

} // namespace COMMON
} // namespace SPTAG
