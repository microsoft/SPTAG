// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/RaBitQQuantizer.h"

#include "inc/Core/Common/DistanceUtils.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace SPTAG
{
namespace COMMON
{

RaBitQQuantizer::RaBitQQuantizer(DimensionType p_dimension, int p_bits, bool p_normalize)
{
    if (Initialize(p_dimension, p_bits, p_normalize) != ErrorCode::Success) {
        throw std::invalid_argument("RaBitQ requires a positive dimension and 1..8 bits");
    }
}

ErrorCode RaBitQQuantizer::Initialize(DimensionType p_dimension, int p_bits, bool p_normalize)
{
    if (p_dimension <= 0 || p_bits < 1 || p_bits > 8) {
        return ErrorCode::FailedParseValue;
    }

    m_dimension = p_dimension;
    m_bits = p_bits;
    m_normalize = p_normalize;
    m_enable_adc = false;
    m_centroid.assign(static_cast<std::size_t>(m_dimension), 0.0F);
    m_quantizer_config = rabitqlib::quant::faster_config(
        static_cast<std::size_t>(m_dimension), static_cast<std::size_t>(m_bits));
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::Train(const std::shared_ptr<VectorSet>& p_vectors)
{
    if (!Ready() || !p_vectors || p_vectors->GetValueType() != VectorValueType::Float ||
        p_vectors->Dimension() != m_dimension || p_vectors->Count() <= 0) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<double> accumulator(static_cast<std::size_t>(m_dimension), 0.0);
    std::vector<float> normalized(static_cast<std::size_t>(m_dimension));
    for (SizeType i = 0; i < p_vectors->Count(); ++i) {
        const auto* vector = static_cast<const float*>(p_vectors->GetVector(i));
        const float* input = vector;
        if (m_normalize) {
            CopyInput(vector, normalized.data());
            input = normalized.data();
        }
        for (DimensionType j = 0; j < m_dimension; ++j) {
            accumulator[static_cast<std::size_t>(j)] += input[j];
        }
    }

    const double inverse_count = 1.0 / static_cast<double>(p_vectors->Count());
    for (DimensionType j = 0; j < m_dimension; ++j) {
        m_centroid[static_cast<std::size_t>(j)] =
            static_cast<float>(accumulator[static_cast<std::size_t>(j)] * inverse_count);
    }
    return ErrorCode::Success;
}

float RaBitQQuantizer::L2Distance(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    thread_local std::vector<float> left;
    thread_local std::vector<float> right;
    Decode(p_y, right);
    if (m_enable_adc) {
        return DistanceUtils::ComputeL2Distance(
            reinterpret_cast<const float*>(p_x), right.data(), m_dimension);
    }

    Decode(p_x, left);
    return DistanceUtils::ComputeL2Distance(left.data(), right.data(), m_dimension);
}

float RaBitQQuantizer::CosineDistance(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    thread_local std::vector<float> left;
    thread_local std::vector<float> right;
    Decode(p_y, right);
    if (m_enable_adc) {
        left.assign(reinterpret_cast<const float*>(p_x),
                    reinterpret_cast<const float*>(p_x) + m_dimension);
    } else {
        Decode(p_x, left);
    }

    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (DimensionType i = 0; i < m_dimension; ++i) {
        dot += static_cast<double>(left[static_cast<std::size_t>(i)]) * right[static_cast<std::size_t>(i)];
        left_norm += static_cast<double>(left[static_cast<std::size_t>(i)]) * left[static_cast<std::size_t>(i)];
        right_norm += static_cast<double>(right[static_cast<std::size_t>(i)]) * right[static_cast<std::size_t>(i)];
    }
    if (left_norm <= 0.0 || right_norm <= 0.0) {
        return 1.0F;
    }
    return 1.0F - static_cast<float>(dot / std::sqrt(left_norm * right_norm));
}

void RaBitQQuantizer::QuantizeVector(const void* p_vector, std::uint8_t* p_output, bool p_adc) const
{
    const auto* input = static_cast<const float*>(p_vector);
    thread_local std::vector<float> normalized;
    if (m_normalize) {
        normalized.resize(static_cast<std::size_t>(m_dimension));
        CopyInput(input, normalized.data());
        input = normalized.data();
    }

    if (p_adc && m_enable_adc) {
        std::memcpy(p_output, input, static_cast<std::size_t>(m_dimension) * sizeof(float));
        return;
    }

    bool zero_residual = true;
    for (DimensionType i = 0; i < m_dimension; ++i) {
        if (input[i] != m_centroid[static_cast<std::size_t>(i)]) {
            zero_residual = false;
            break;
        }
    }
    std::memset(p_output, 0, static_cast<std::size_t>(m_dimension));
    float delta = 0.0F;
    float lower_value = 0.0F;
    if (zero_residual) {
        std::memcpy(p_output + m_dimension, &delta, sizeof(delta));
        std::memcpy(p_output + m_dimension + sizeof(delta), &lower_value, sizeof(lower_value));
        return;
    }

    rabitqlib::quant::quantize_scalar<float, std::uint8_t>(
        input,
        m_centroid.data(),
        static_cast<std::size_t>(m_dimension),
        static_cast<std::size_t>(m_bits),
        p_output,
        delta,
        lower_value,
        m_quantizer_config);
    std::memcpy(p_output + m_dimension, &delta, sizeof(delta));
    std::memcpy(p_output + m_dimension + sizeof(delta), &lower_value, sizeof(lower_value));
}

int RaBitQQuantizer::QuantizeSize() const
{
    return m_enable_adc ? static_cast<int>(sizeof(float) * m_dimension) : static_cast<int>(CodeBytes());
}

void RaBitQQuantizer::ReconstructVector(const std::uint8_t* p_code, void* p_output) const
{
    thread_local std::vector<float> reconstructed;
    Decode(p_code, reconstructed);
    std::memcpy(p_output, reconstructed.data(), static_cast<std::size_t>(m_dimension) * sizeof(float));
}

int RaBitQQuantizer::ReconstructSize() const
{
    return static_cast<int>(sizeof(float) * m_dimension);
}

DimensionType RaBitQQuantizer::ReconstructDim() const
{
    return m_dimension;
}

std::uint64_t RaBitQQuantizer::BufferSize() const
{
    return sizeof(QuantizerType) + sizeof(VectorValueType) + sizeof(ModelHeader) +
        m_centroid.size() * sizeof(float);
}

ErrorCode RaBitQQuantizer::SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_output) const
{
    if (!p_output || !Ready()) {
        return ErrorCode::Fail;
    }

    QuantizerType type = QuantizerType::RaBitQQuantizer;
    VectorValueType reconstruct_type = VectorValueType::Float;
    ModelHeader header{
        kModelMagic,
        kModelVersion,
        m_dimension,
        m_bits,
        m_normalize ? 1U : 0U,
    };
    if (p_output->WriteBinary(sizeof(type), reinterpret_cast<char*>(&type)) != sizeof(type) ||
        p_output->WriteBinary(sizeof(reconstruct_type), reinterpret_cast<char*>(&reconstruct_type)) != sizeof(reconstruct_type) ||
        p_output->WriteBinary(sizeof(header), reinterpret_cast<char*>(&header)) != sizeof(header) ||
        p_output->WriteBinary(m_centroid.size() * sizeof(float), reinterpret_cast<char*>(const_cast<float*>(m_centroid.data()))) !=
            m_centroid.size() * sizeof(float)) {
        return ErrorCode::DiskIOFail;
    }
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_input)
{
    if (!p_input) {
        return ErrorCode::Fail;
    }

    ModelHeader header{};
    if (p_input->ReadBinary(sizeof(header), reinterpret_cast<char*>(&header)) != sizeof(header) ||
        LoadHeader(header) != ErrorCode::Success ||
        p_input->ReadBinary(m_centroid.size() * sizeof(float), reinterpret_cast<char*>(m_centroid.data())) !=
            m_centroid.size() * sizeof(float)) {
        return ErrorCode::FailedParseValue;
    }
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::LoadQuantizer(std::uint8_t* p_raw_bytes)
{
    if (!p_raw_bytes) {
        return ErrorCode::Fail;
    }

    ModelHeader header{};
    std::memcpy(&header, p_raw_bytes, sizeof(header));
    if (LoadHeader(header) != ErrorCode::Success) {
        return ErrorCode::FailedParseValue;
    }
    p_raw_bytes += sizeof(header);
    std::memcpy(m_centroid.data(), p_raw_bytes, m_centroid.size() * sizeof(float));
    return ErrorCode::Success;
}

bool RaBitQQuantizer::GetEnableADC() const
{
    return m_enable_adc;
}

void RaBitQQuantizer::SetEnableADC(bool p_enable_adc)
{
    m_enable_adc = p_enable_adc;
}

QuantizerType RaBitQQuantizer::GetQuantizerType() const
{
    return QuantizerType::RaBitQQuantizer;
}

VectorValueType RaBitQQuantizer::GetReconstructType() const
{
    return VectorValueType::Float;
}

DimensionType RaBitQQuantizer::GetNumSubvectors() const
{
    return static_cast<DimensionType>(CodeBytes());
}

int RaBitQQuantizer::GetBase() const
{
    return COMMON::Utils::GetBase<float>();
}

float* RaBitQQuantizer::GetL2DistanceTables()
{
    return nullptr;
}

bool RaBitQQuantizer::Ready() const
{
    return m_dimension > 0 && m_bits >= 1 && m_bits <= 8 &&
        m_centroid.size() == static_cast<std::size_t>(m_dimension);
}

ErrorCode RaBitQQuantizer::LoadHeader(const ModelHeader& p_header)
{
    if (p_header.magic != kModelMagic || p_header.version != kModelVersion ||
        p_header.dimension <= 0 || p_header.bits < 1 || p_header.bits > 8) {
        return ErrorCode::FailedParseValue;
    }
    return Initialize(p_header.dimension, p_header.bits, p_header.normalize != 0);
}

void RaBitQQuantizer::Decode(const std::uint8_t* p_code, std::vector<float>& p_output) const
{
    float delta = 0.0F;
    float lower_value = 0.0F;
    ReadCodeParameters(p_code, delta, lower_value);
    p_output.resize(static_cast<std::size_t>(m_dimension));
    rabitqlib::quant::reconstruct_vec<float, std::uint8_t>(
        p_code,
        delta,
        lower_value,
        static_cast<std::size_t>(m_dimension),
        p_output.data());
    for (DimensionType i = 0; i < m_dimension; ++i) {
        p_output[static_cast<std::size_t>(i)] += m_centroid[static_cast<std::size_t>(i)];
    }
}

void RaBitQQuantizer::CopyInput(const float* p_input, float* p_output) const
{
    double norm = 0.0;
    for (DimensionType i = 0; i < m_dimension; ++i) {
        norm += static_cast<double>(p_input[i]) * p_input[i];
    }
    if (norm == 0.0) {
        std::memset(p_output, 0, static_cast<std::size_t>(m_dimension) * sizeof(float));
        return;
    }

    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
    for (DimensionType i = 0; i < m_dimension; ++i) {
        p_output[i] = p_input[i] * inverse_norm;
    }
}

void RaBitQQuantizer::ReadCodeParameters(const std::uint8_t* p_code,
                                         float& p_delta,
                                         float& p_lower_value) const
{
    std::memcpy(&p_delta, p_code + m_dimension, sizeof(p_delta));
    std::memcpy(&p_lower_value, p_code + m_dimension + sizeof(p_delta), sizeof(p_lower_value));
}

std::size_t RaBitQQuantizer::CodeBytes() const
{
    return static_cast<std::size_t>(m_dimension) + 2 * sizeof(float);
}

} // namespace COMMON
} // namespace SPTAG
