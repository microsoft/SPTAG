// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/RaBitQQuantizer.h"

#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
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
    m_padded_dimension = static_cast<DimensionType>(
        (static_cast<std::size_t>(p_dimension) + 63U) / 64U * 64U);
    m_bits = p_bits;
    m_normalize = p_normalize;
    m_enable_adc = false;
    m_centroid.assign(static_cast<std::size_t>(m_padded_dimension), 0.0F);
    m_quantizer_config = rabitqlib::quant::faster_config(
        static_cast<std::size_t>(m_padded_dimension), static_cast<std::size_t>(m_bits));
    m_ip_func = rabitqlib::select_excode_ipfunc(static_cast<std::size_t>(m_bits));
    m_trained = false;
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::Train(const std::shared_ptr<VectorSet>& p_vectors)
{
    if (!Ready() || !p_vectors || p_vectors->GetValueType() != VectorValueType::Float ||
        p_vectors->Dimension() != m_dimension || p_vectors->Count() <= 0) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<double> accumulator(static_cast<std::size_t>(m_dimension), 0.0);
    std::vector<float> prepared;
    for (SizeType i = 0; i < p_vectors->Count(); ++i) {
        const auto* vector = static_cast<const float*>(p_vectors->GetVector(i));
        PrepareInput(vector, prepared);
        for (DimensionType j = 0; j < m_dimension; ++j) {
            accumulator[static_cast<std::size_t>(j)] += prepared[static_cast<std::size_t>(j)];
        }
    }

    const double inverse_count = 1.0 / static_cast<double>(p_vectors->Count());
    for (DimensionType j = 0; j < m_dimension; ++j) {
        m_centroid[static_cast<std::size_t>(j)] =
            static_cast<float>(accumulator[static_cast<std::size_t>(j)] * inverse_count);
    }
    m_trained = true;
    return ErrorCode::Success;
}

std::shared_ptr<RaBitQQuantizer> RaBitQQuantizer::CloneWithBits(int p_bits) const
{
    if (!Ready() || !m_trained || p_bits < 1 || p_bits > 8) {
        return nullptr;
    }
    auto quantizer = std::make_shared<RaBitQQuantizer>(m_dimension, p_bits, m_normalize);
    quantizer->m_centroid = m_centroid;
    quantizer->m_trained = true;
    return quantizer;
}

float RaBitQQuantizer::L2Distance(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    thread_local std::vector<float> reconstructed_query;
    const float* query = reinterpret_cast<const float*>(p_x);
    float g_add = 0.0F;
    float k1xsumq = 0.0F;
    if (!m_enable_adc) {
        Decode(p_x, reconstructed_query);
        query = reconstructed_query.data();
        g_add = rabitqlib::euclidean_sqr(
            query, m_centroid.data(), static_cast<std::size_t>(m_padded_dimension));
        k1xsumq = -0.5F * std::accumulate(
            query, query + m_padded_dimension, 0.0F);
    } else {
        const std::uint8_t* query_factors =
            p_x + static_cast<std::size_t>(m_padded_dimension) * sizeof(float);
        std::memcpy(&g_add, query_factors, sizeof(float));
        std::memcpy(&k1xsumq, query_factors + sizeof(float), sizeof(float));
    }

    float f_add = 0.0F;
    float f_rescale = 0.0F;
    ReadDistanceFactors(p_y, f_add, f_rescale);
    return rabitqlib::quant::full_est_dist<float, std::uint8_t>(
        p_y,
        query,
        m_ip_func,
        static_cast<std::size_t>(m_padded_dimension),
        static_cast<std::size_t>(m_bits),
        f_add,
        f_rescale,
        g_add,
        k1xsumq);
}

float RaBitQQuantizer::CosineDistance(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                 "RaBitQ official full-code adapter supports L2 distance only.\n");
    return std::numeric_limits<float>::infinity();
}

void RaBitQQuantizer::QuantizeVector(const void* p_vector, std::uint8_t* p_output, bool p_adc) const
{
    thread_local std::vector<float> prepared;
    PrepareInput(static_cast<const float*>(p_vector), prepared);
    const float* input = prepared.data();

    if (p_adc && m_enable_adc) {
        const std::size_t query_bytes =
            static_cast<std::size_t>(m_padded_dimension) * sizeof(float);
        std::memcpy(p_output, input, query_bytes);
        const float g_add = rabitqlib::euclidean_sqr(
            input, m_centroid.data(), static_cast<std::size_t>(m_padded_dimension));
        const float k1xsumq = -0.5F * std::accumulate(
            input, input + m_padded_dimension, 0.0F);
        std::memcpy(p_output + query_bytes, &g_add, sizeof(float));
        std::memcpy(p_output + query_bytes + sizeof(float), &k1xsumq, sizeof(float));
        return;
    }

    bool zero_residual = true;
    for (DimensionType i = 0; i < m_padded_dimension; ++i) {
        if (input[i] != m_centroid[static_cast<std::size_t>(i)]) {
            zero_residual = false;
            break;
        }
    }
    std::memset(p_output, 0, CodeBytes());
    float f_add = 0.0F;
    float f_rescale = 0.0F;
    float f_error = 0.0F;
    float delta = 0.0F;
    float lower_value = 0.0F;
    if (!zero_residual) {
        thread_local std::vector<std::uint8_t> scalar_code;
        thread_local std::vector<std::uint8_t> full_code;
        scalar_code.assign(static_cast<std::size_t>(m_padded_dimension), 0);
        full_code.assign(static_cast<std::size_t>(m_padded_dimension), 0);

        rabitqlib::quant::quantize_scalar<float, std::uint8_t>(
            input,
            m_centroid.data(),
            static_cast<std::size_t>(m_padded_dimension),
            static_cast<std::size_t>(m_bits),
            scalar_code.data(),
            delta,
            lower_value,
            m_quantizer_config);

        if (m_bits == 1) {
            std::vector<char> bin_data(
                rabitqlib::BinDataMap<float>::data_bytes(
                    static_cast<std::size_t>(m_padded_dimension)));
            rabitqlib::quant::quantize_compact_one_bit(
                input,
                m_centroid.data(),
                static_cast<std::size_t>(m_padded_dimension),
                bin_data.data(),
                rabitqlib::METRIC_L2);
            rabitqlib::ConstBinDataMap<float> bin(
                bin_data.data(), static_cast<std::size_t>(m_padded_dimension));
            rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                scalar_code.data(),
                p_output,
                static_cast<std::size_t>(m_padded_dimension),
                1);
            f_add = bin.f_add();
            f_rescale = bin.f_rescale();
            f_error = bin.f_error();
        } else {
            rabitqlib::quant::quantize_full_single<float, std::uint8_t>(
                input,
                m_centroid.data(),
                static_cast<std::size_t>(m_padded_dimension),
                static_cast<std::size_t>(m_bits),
                full_code.data(),
                f_add,
                f_rescale,
                f_error,
                rabitqlib::METRIC_L2,
                m_quantizer_config);
            if (scalar_code != full_code) {
                throw std::runtime_error(
                    "Official RaBitQ scalar and full-code quantizers produced different codes");
            }
            rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                full_code.data(),
                p_output,
                static_cast<std::size_t>(m_padded_dimension),
                static_cast<std::size_t>(m_bits));
        }
    }

    std::uint8_t* factors = p_output + PackedCodeBytes();
    std::memcpy(factors, &f_add, sizeof(f_add));
    std::memcpy(factors + sizeof(float), &f_rescale, sizeof(f_rescale));
    std::memcpy(factors + 2 * sizeof(float), &f_error, sizeof(f_error));
    std::memcpy(factors + 3 * sizeof(float), &delta, sizeof(delta));
    std::memcpy(factors + 4 * sizeof(float), &lower_value, sizeof(lower_value));
}

int RaBitQQuantizer::QuantizeSize() const
{
    return m_enable_adc
        ? static_cast<int>(
              sizeof(float) *
              (static_cast<std::size_t>(m_padded_dimension) + kQueryFactorCount))
        : static_cast<int>(CodeBytes());
}

void RaBitQQuantizer::ReconstructVector(const std::uint8_t* p_code, void* p_output) const
{
    thread_local std::vector<float> reconstructed;
    Decode(p_code, reconstructed);
    std::memcpy(
        p_output, reconstructed.data(), static_cast<std::size_t>(m_dimension) * sizeof(float));
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
    if (!p_output || !Ready() || !m_trained) {
        return ErrorCode::Fail;
    }

    QuantizerType type = QuantizerType::RaBitQQuantizer;
    VectorValueType reconstruct_type = VectorValueType::Float;
    ModelHeader header{
        kModelMagic,
        kModelVersion,
        m_dimension,
        m_padded_dimension,
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
    m_trained = true;
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
    m_trained = true;
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
        m_padded_dimension >= m_dimension && m_padded_dimension % 64 == 0 &&
        m_centroid.size() == static_cast<std::size_t>(m_padded_dimension) &&
        m_ip_func != nullptr;
}

ErrorCode RaBitQQuantizer::LoadHeader(const ModelHeader& p_header)
{
    if (p_header.magic != kModelMagic || p_header.version != kModelVersion ||
        p_header.dimension <= 0 || p_header.paddedDimension < p_header.dimension ||
        p_header.paddedDimension % 64 != 0 || p_header.bits < 1 || p_header.bits > 8) {
        return ErrorCode::FailedParseValue;
    }
    const ErrorCode status =
        Initialize(p_header.dimension, p_header.bits, p_header.normalize != 0);
    return status == ErrorCode::Success && m_padded_dimension == p_header.paddedDimension
        ? ErrorCode::Success
        : ErrorCode::FailedParseValue;
}

void RaBitQQuantizer::Decode(const std::uint8_t* p_code, std::vector<float>& p_output) const
{
    float f_add = 0.0F;
    float f_rescale = 0.0F;
    float f_error = 0.0F;
    float delta = 0.0F;
    float lower_value = 0.0F;
    ReadCodeFactors(
        p_code, f_add, f_rescale, f_error, delta, lower_value);

    thread_local std::vector<std::uint8_t> unpacked;
    unpacked.resize(static_cast<std::size_t>(m_padded_dimension));
    UnpackCode(p_code, unpacked.data());
    p_output.resize(static_cast<std::size_t>(m_padded_dimension));
    rabitqlib::quant::reconstruct_vec<float, std::uint8_t>(
        unpacked.data(),
        delta,
        lower_value,
        static_cast<std::size_t>(m_padded_dimension),
        p_output.data());
    for (DimensionType i = 0; i < m_padded_dimension; ++i) {
        p_output[static_cast<std::size_t>(i)] += m_centroid[static_cast<std::size_t>(i)];
    }
    std::fill(
        p_output.begin() + static_cast<std::size_t>(m_dimension), p_output.end(), 0.0F);
}

void RaBitQQuantizer::PrepareInput(
    const float* p_input, std::vector<float>& p_output) const
{
    p_output.assign(static_cast<std::size_t>(m_padded_dimension), 0.0F);
    std::copy(p_input, p_input + m_dimension, p_output.begin());
    if (!m_normalize) {
        return;
    }

    double norm = 0.0;
    for (DimensionType i = 0; i < m_dimension; ++i) {
        norm += static_cast<double>(p_input[i]) * p_input[i];
    }
    if (norm == 0.0) {
        return;
    }

    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
    for (DimensionType i = 0; i < m_dimension; ++i) {
        p_output[static_cast<std::size_t>(i)] = p_input[i] * inverse_norm;
    }
}

void RaBitQQuantizer::UnpackCode(
    const std::uint8_t* p_code, std::uint8_t* p_output) const
{
    const std::size_t dim = static_cast<std::size_t>(m_padded_dimension);
    std::memset(p_output, 0, dim);
    if (m_bits == 8) {
        std::memcpy(p_output, p_code, dim);
        return;
    }
    if (m_bits == 1) {
        for (std::size_t i = 0; i < dim; ++i) {
            p_output[i] = static_cast<std::uint8_t>((p_code[i / 8] >> (i % 8)) & 1U);
        }
        return;
    }

    for (std::size_t block = 0; block < dim; block += 64) {
        const std::uint8_t* packed =
            p_code + block * static_cast<std::size_t>(m_bits) / 8;
        std::uint8_t* output = p_output + block;
        if (m_bits == 2 || m_bits == 3) {
            for (std::size_t i = 0; i < 64; ++i) {
                output[i] = static_cast<std::uint8_t>(
                    (packed[i % 16] >> (2 * (i / 16))) & 0x3U);
                if (m_bits == 3) {
                    output[i] |= static_cast<std::uint8_t>(
                        ((packed[16 + (i % 8)] >> (i / 8)) & 1U) << 2);
                }
            }
        } else if (m_bits == 4) {
            for (std::size_t i = 0; i < 64; ++i) {
                const std::size_t group = i / 16;
                const std::size_t within_group = i % 16;
                output[i] = static_cast<std::uint8_t>(
                    (packed[group * 8 + (within_group % 8)] >>
                     (4 * (within_group / 8))) &
                    0xFU);
            }
        } else if (m_bits == 5) {
            for (std::size_t i = 0; i < 64; ++i) {
                const std::size_t half = i / 32;
                const std::size_t within_half = i % 32;
                output[i] = static_cast<std::uint8_t>(
                    (packed[half * 16 + (within_half % 16)] >>
                     (4 * (within_half / 16))) &
                    0xFU);
                output[i] |= static_cast<std::uint8_t>(
                    ((packed[32 + (i % 8)] >> (i / 8)) & 1U) << 4);
            }
        } else {
            for (std::size_t i = 0; i < 48; ++i) {
                output[i] = static_cast<std::uint8_t>(packed[i] & 0x3FU);
            }
            for (std::size_t lane = 0; lane < 16; ++lane) {
                output[48 + lane] = static_cast<std::uint8_t>(
                    ((packed[lane] >> 6) & 0x3U) |
                    (((packed[16 + lane] >> 6) & 0x3U) << 2) |
                    (((packed[32 + lane] >> 6) & 0x3U) << 4));
            }
            if (m_bits == 7) {
                for (std::size_t i = 0; i < 64; ++i) {
                    output[i] |= static_cast<std::uint8_t>(
                        ((packed[48 + (i % 8)] >> (i / 8)) & 1U) << 6);
                }
            }
        }
    }
}

void RaBitQQuantizer::ReadCodeFactors(const std::uint8_t* p_code,
                                      float& p_f_add,
                                      float& p_f_rescale,
                                      float& p_f_error,
                                      float& p_delta,
                                      float& p_lower_value) const
{
    const std::uint8_t* factors = p_code + PackedCodeBytes();
    std::memcpy(&p_f_add, factors, sizeof(float));
    std::memcpy(&p_f_rescale, factors + sizeof(float), sizeof(float));
    std::memcpy(&p_f_error, factors + 2 * sizeof(float), sizeof(float));
    std::memcpy(&p_delta, factors + 3 * sizeof(float), sizeof(float));
    std::memcpy(&p_lower_value, factors + 4 * sizeof(float), sizeof(float));
}

void RaBitQQuantizer::ReadDistanceFactors(const std::uint8_t* p_code,
                                          float& p_f_add,
                                          float& p_f_rescale) const
{
    const std::uint8_t* factors = p_code + PackedCodeBytes();
    std::memcpy(&p_f_add, factors, sizeof(float));
    std::memcpy(&p_f_rescale, factors + sizeof(float), sizeof(float));
}

std::size_t RaBitQQuantizer::PackedCodeBytes() const
{
    return static_cast<std::size_t>(m_padded_dimension) *
        static_cast<std::size_t>(m_bits) / 8U;
}

std::size_t RaBitQQuantizer::CodeBytes() const
{
    return PackedCodeBytes() + kCodeFactorCount * sizeof(float);
}

} // namespace COMMON
} // namespace SPTAG
