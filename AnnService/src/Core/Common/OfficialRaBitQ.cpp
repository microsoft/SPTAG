// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/OfficialRaBitQ.h"

#include "rabitqlib/utils/cpu_features.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>

namespace SPTAG
{
namespace COMMON
{

OfficialRaBitQ::OfficialRaBitQ(DimensionType p_dimension, int p_totalBits)
{
    if (Initialize(p_dimension, p_totalBits) != ErrorCode::Success) {
        throw std::invalid_argument("OfficialRaBitQ requires positive dimension and 1..8 total bits");
    }
}

bool OfficialRaBitQ::IsSupportedDimension(DimensionType p_dimension)
{
    // FhtKacRotator provides specialized kernels for floor(log2(dim)) in [6, 11].
    return p_dimension >= 64 && p_dimension < 4096;
}

bool OfficialRaBitQ::IsSupportedPlatform()
{
    return rabitqlib::cpu::has_avx2();
}

ErrorCode OfficialRaBitQ::Initialize(DimensionType p_dimension, int p_totalBits)
{
    if (!IsSupportedDimension(p_dimension) || !IsSupportedPlatform() ||
        p_totalBits < 1 || p_totalBits > 8) {
        return ErrorCode::FailedParseValue;
    }

    m_dimension = p_dimension;
    m_paddedDimension = static_cast<DimensionType>(
        rabitqlib::round_up_to_multiple(static_cast<std::size_t>(p_dimension), 64));
    m_exBits = p_totalBits - 1;
    m_rotator.reset(rabitqlib::choose_rotator<float>(
        static_cast<std::size_t>(m_dimension),
        rabitqlib::RotatorType::FhtKacRotator,
        static_cast<std::size_t>(m_paddedDimension)));
    if (!m_rotator || m_rotator->size() != static_cast<std::size_t>(m_paddedDimension)) {
        return ErrorCode::Fail;
    }

    m_centroid.assign(static_cast<std::size_t>(m_paddedDimension), 0.0F);
    m_quantizerConfig = rabitqlib::quant::faster_config(
        static_cast<std::size_t>(m_paddedDimension), static_cast<std::size_t>(p_totalBits));
    m_exCodeInnerProduct = rabitqlib::select_excode_ipfunc(static_cast<std::size_t>(m_exBits));
    return m_exCodeInnerProduct != nullptr || m_exBits == 0 ? ErrorCode::Success : ErrorCode::Fail;
}

ErrorCode OfficialRaBitQ::Train(const std::shared_ptr<VectorSet>& p_vectors)
{
    if (!p_vectors || p_vectors->GetValueType() != VectorValueType::Float ||
        p_vectors->Dimension() != m_dimension || p_vectors->Count() <= 0 || !m_rotator) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<double> accumulator(static_cast<std::size_t>(m_paddedDimension), 0.0);
    std::vector<float> rotated;
    for (SizeType i = 0; i < p_vectors->Count(); ++i) {
        Rotate(static_cast<const float*>(p_vectors->GetVector(i)), rotated);
        for (DimensionType d = 0; d < m_paddedDimension; ++d) {
            accumulator[static_cast<std::size_t>(d)] += rotated[static_cast<std::size_t>(d)];
        }
    }

    const double inverseCount = 1.0 / static_cast<double>(p_vectors->Count());
    for (DimensionType d = 0; d < m_paddedDimension; ++d) {
        m_centroid[static_cast<std::size_t>(d)] = static_cast<float>(
            accumulator[static_cast<std::size_t>(d)] * inverseCount);
    }
    return ErrorCode::Success;
}

ErrorCode OfficialRaBitQ::Save(const std::string& p_path) const
{
    if (!Ready()) {
        return ErrorCode::Fail;
    }

    std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return ErrorCode::FailedOpenFile;
    }

    const std::size_t rotatorBytes = m_rotator->dump_bytes();
    if (rotatorBytes > (std::numeric_limits<std::uint32_t>::max)()) {
        return ErrorCode::Fail;
    }
    const ModelHeader header{
        kMagic,
        kVersion,
        m_dimension,
        m_paddedDimension,
        m_exBits,
        static_cast<std::uint32_t>(rotatorBytes),
    };
    std::vector<char> rotatorData(rotatorBytes);
    m_rotator->save(rotatorData.data());

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(rotatorData.data(), static_cast<std::streamsize>(rotatorData.size()));
    output.write(reinterpret_cast<const char*>(m_centroid.data()),
                 static_cast<std::streamsize>(m_centroid.size() * sizeof(float)));
    return output ? ErrorCode::Success : ErrorCode::DiskIOFail;
}

ErrorCode OfficialRaBitQ::Load(const std::string& p_path)
{
    std::ifstream input(p_path, std::ios::binary);
    if (!input) {
        return ErrorCode::FailedOpenFile;
    }

    ModelHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.magic != kMagic || header.version != kVersion ||
        header.dimension <= 0 || header.paddedDimension < header.dimension ||
        header.exBits < 0 || header.exBits > 7) {
        return ErrorCode::FailedParseValue;
    }

    const DimensionType expectedPadded = static_cast<DimensionType>(
        rabitqlib::round_up_to_multiple(static_cast<std::size_t>(header.dimension), 64));
    if (header.paddedDimension != expectedPadded) {
        return ErrorCode::FailedParseValue;
    }

    const ErrorCode init = Initialize(header.dimension, header.exBits + 1);
    if (init != ErrorCode::Success || m_rotator->dump_bytes() != header.rotatorBytes) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<char> rotatorData(header.rotatorBytes);
    input.read(rotatorData.data(), static_cast<std::streamsize>(rotatorData.size()));
    m_centroid.resize(static_cast<std::size_t>(m_paddedDimension));
    input.read(reinterpret_cast<char*>(m_centroid.data()),
               static_cast<std::streamsize>(m_centroid.size() * sizeof(float)));
    if (!input) {
        return ErrorCode::FailedParseValue;
    }
    m_rotator->load(rotatorData.data());
    return ErrorCode::Success;
}

void OfficialRaBitQ::Encode(const float* p_vector, std::uint8_t* p_code) const
{
    std::vector<float> rotated;
    Rotate(p_vector, rotated);
    char* binCode = reinterpret_cast<char*>(p_code);
    char* exCode = binCode + BinCodeBytes();
    rabitqlib::quant::quantize_split_single(
        rotated.data(),
        m_centroid.data(),
        static_cast<std::size_t>(m_paddedDimension),
        static_cast<std::size_t>(m_exBits),
        binCode,
        m_exBits > 0 ? exCode : nullptr,
        rabitqlib::METRIC_L2,
        m_quantizerConfig);
}

OfficialRaBitQ::QueryContext OfficialRaBitQ::PrepareQuery(const float* p_query) const
{
    QueryContext context;
    Rotate(p_query, context.rotated);
    context.query = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
        context.rotated.data(),
        static_cast<std::size_t>(m_paddedDimension),
        static_cast<std::size_t>(m_exBits),
        m_quantizerConfig,
        rabitqlib::METRIC_L2);
    const float centerDistance = std::sqrt(rabitqlib::euclidean_sqr(
        context.rotated.data(), m_centroid.data(), static_cast<std::size_t>(m_paddedDimension)));
    context.query->set_g_add(centerDistance);
    return context;
}

float OfficialRaBitQ::Estimate(const QueryContext& p_query, const std::uint8_t* p_code) const
{
    float estimate = 0.0F;
    float lowerBound = 0.0F;
    float binaryInnerProduct = 0.0F;
    const char* binCode = reinterpret_cast<const char*>(p_code);
    if (m_exBits == 0) {
        rabitqlib::split_single_estdist(
            binCode,
            *p_query.query,
            static_cast<std::size_t>(m_paddedDimension),
            binaryInnerProduct,
            estimate,
            lowerBound,
            p_query.query->g_add(),
            p_query.query->g_error());
        return estimate;
    }

    const char* exCode = binCode + BinCodeBytes();
    rabitqlib::split_single_fulldist(
        binCode,
        exCode,
        m_exCodeInnerProduct,
        *p_query.query,
        static_cast<std::size_t>(m_paddedDimension),
        static_cast<std::size_t>(m_exBits),
        estimate,
        lowerBound,
        binaryInnerProduct,
        p_query.query->g_add(),
        p_query.query->g_error());
    return estimate;
}

bool OfficialRaBitQ::Ready() const
{
    return m_dimension > 0 && m_paddedDimension >= m_dimension && m_exBits >= 0 &&
        m_rotator != nullptr && m_centroid.size() == static_cast<std::size_t>(m_paddedDimension) &&
        (m_exBits == 0 || m_exCodeInnerProduct != nullptr);
}

std::size_t OfficialRaBitQ::BinCodeBytes() const
{
    return rabitqlib::BinDataMap<float>::data_bytes(static_cast<std::size_t>(m_paddedDimension));
}

std::size_t OfficialRaBitQ::ExCodeBytes() const
{
    return rabitqlib::ExDataMap<float>::data_bytes(
        static_cast<std::size_t>(m_paddedDimension), static_cast<std::size_t>(m_exBits));
}

std::size_t OfficialRaBitQ::CodeBytes() const
{
    return BinCodeBytes() + ExCodeBytes();
}

void OfficialRaBitQ::Rotate(const float* p_vector, std::vector<float>& p_rotated) const
{
    p_rotated.resize(static_cast<std::size_t>(m_paddedDimension));
    m_rotator->rotate(p_vector, p_rotated.data());
}

} // namespace COMMON
} // namespace SPTAG
