// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_RABITQQUANTIZER_H_
#define _SPTAG_COMMON_RABITQQUANTIZER_H_

#include "IQuantizer.h"
#include "inc/Core/VectorSet.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "rabitqlib/quantization/scalar_quantizer.hpp"

namespace SPTAG
{
namespace COMMON
{

// Global RaBitQ adapter for the existing PQ/OPQ IQuantizer path. Data codes use
// RaBitQ's reconstruction-oriented total-bits layout so BKT/KDT can compare
// codes during construction, while ADC mode supplies a rotated query context.
template <typename T>
class RaBitQQuantizer : public IQuantizer
{
public:
    RaBitQQuantizer()
        : m_dim(0),
          m_paddedDim(0),
          m_bits(0),
          m_normalize(false),
          m_enableADC(false)
    {
    }

    RaBitQQuantizer(DimensionType p_dim, int p_bits, bool p_normalize)
        : m_dim(p_dim),
          m_paddedDim(p_dim > 0 ? NextPowerOfTwo(static_cast<std::size_t>(p_dim)) : 0),
          m_bits(p_bits),
          m_normalize(p_normalize),
          m_enableADC(false)
    {
        if (p_dim <= 0 || p_bits < 1 || p_bits > 8) {
            throw std::invalid_argument("RaBitQ requires positive dimension and 1..8 bits");
        }
        InitializeRotation();
        m_centroid.assign(m_paddedDim, 0.0F);
    }

    void Train(const std::shared_ptr<VectorSet>& p_vectors)
    {
        if (!p_vectors || p_vectors->Count() <= 0 || p_vectors->Dimension() != m_dim) {
            throw std::invalid_argument("RaBitQ training vectors are invalid");
        }

        std::vector<double> centroidAccumulator(m_paddedDim, 0.0);
        std::vector<float> rotated;
        for (SizeType i = 0; i < p_vectors->Count(); ++i) {
            RotateInput(static_cast<const T*>(p_vectors->GetVector(i)), rotated);
            for (std::size_t d = 0; d < m_paddedDim; ++d) {
                centroidAccumulator[d] += rotated[d];
            }
        }

        m_centroid.resize(m_paddedDim);
        const double inverseCount = 1.0 / static_cast<double>(p_vectors->Count());
        for (std::size_t d = 0; d < m_paddedDim; ++d) {
            m_centroid[d] = static_cast<float>(centroidAccumulator[d] * inverseCount);
        }
    }

    float L2Distance(const std::uint8_t* p_x, const std::uint8_t* p_y) const override
    {
        float rightDelta = 0.0F;
        float rightLowerValue = 0.0F;
        this->ReadCodeParameters(p_y, rightDelta, rightLowerValue);
        double distance = 0.0;
        if (m_enableADC) {
            const float* query = reinterpret_cast<const float*>(p_x);
            for (std::size_t d = 0; d < m_paddedDim; ++d) {
                const double difference = static_cast<double>(query[d]) -
                    static_cast<double>(this->DecodeCoordinate(p_y, d, rightDelta, rightLowerValue, true));
                distance += difference * difference;
            }
        } else {
            float leftDelta = 0.0F;
            float leftLowerValue = 0.0F;
            this->ReadCodeParameters(p_x, leftDelta, leftLowerValue);
            for (std::size_t d = 0; d < m_paddedDim; ++d) {
                const double difference = static_cast<double>(
                    this->DecodeCoordinate(p_x, d, leftDelta, leftLowerValue, false)) -
                    static_cast<double>(this->DecodeCoordinate(p_y, d, rightDelta, rightLowerValue, false));
                distance += difference * difference;
            }
        }
        return static_cast<float>(distance);
    }

    float CosineDistance(const std::uint8_t* p_x, const std::uint8_t* p_y) const override
    {
        if (m_normalize) {
            return 0.5F * L2Distance(p_x, p_y);
        }

        float rightDelta = 0.0F;
        float rightLowerValue = 0.0F;
        this->ReadCodeParameters(p_y, rightDelta, rightLowerValue);
        if (m_enableADC) {
            const float* query = reinterpret_cast<const float*>(p_x);
            double dot = 0.0;
            double leftNorm = 0.0;
            double rightNorm = 0.0;
            for (std::size_t d = 0; d < m_paddedDim; ++d) {
                const double left = query[d];
                const double right = this->DecodeCoordinate(p_y, d, rightDelta, rightLowerValue, true);
                dot += left * right;
                leftNorm += left * left;
                rightNorm += right * right;
            }
            if (leftNorm <= 0.0 || rightNorm <= 0.0) {
                return 1.0F;
            }
            return 1.0F - static_cast<float>(dot / std::sqrt(leftNorm * rightNorm));
        } else {
            float leftDelta = 0.0F;
            float leftLowerValue = 0.0F;
            this->ReadCodeParameters(p_x, leftDelta, leftLowerValue);
            double dot = 0.0;
            double leftNorm = 0.0;
            double rightNorm = 0.0;
            for (std::size_t d = 0; d < m_paddedDim; ++d) {
                const double left = this->DecodeCoordinate(p_x, d, leftDelta, leftLowerValue, true);
                const double right = this->DecodeCoordinate(p_y, d, rightDelta, rightLowerValue, true);
                dot += left * right;
                leftNorm += left * left;
                rightNorm += right * right;
            }
            if (leftNorm <= 0.0 || rightNorm <= 0.0) {
                return 1.0F;
            }
            return 1.0F - static_cast<float>(dot / std::sqrt(leftNorm * rightNorm));
        }
    }

    void QuantizeVector(const void* p_vector, std::uint8_t* p_output, bool p_adc = true) const override
    {
        std::vector<float> rotated;
        RotateInput(static_cast<const T*>(p_vector), rotated);

        if (p_adc && m_enableADC) {
            std::memcpy(p_output, rotated.data(), m_paddedDim * sizeof(float));
            return;
        }

        std::vector<std::uint8_t> scalarCodes(m_paddedDim);
        float delta = 0.0F;
        float lowerValue = 0.0F;
        rabitqlib::quant::global_scalar::QuantizeScalar<float, std::uint8_t>(
            rotated.data(), m_centroid.data(), m_paddedDim,
            static_cast<std::size_t>(m_bits), scalarCodes.data(), delta, lowerValue);
        PackCodes(scalarCodes, p_output);
        const std::size_t packedCodeBytes = PackedCodeBytes();
        std::memcpy(p_output + packedCodeBytes, &delta, sizeof(delta));
        std::memcpy(p_output + packedCodeBytes + sizeof(delta), &lowerValue, sizeof(lowerValue));
    }

    int QuantizeSize() const override
    {
        return m_enableADC
            ? static_cast<int>(m_paddedDim * sizeof(float))
            : static_cast<int>(CodeBytes());
    }

    void ReconstructVector(const std::uint8_t* p_code, void* p_output) const override
    {
        std::vector<float> rotated;
        DecodeRotated(p_code, rotated);
        InverseRotate(rotated);
        T* output = static_cast<T*>(p_output);
        for (DimensionType d = 0; d < m_dim; ++d) {
            output[d] = ConvertOutput(rotated[static_cast<std::size_t>(d)]);
        }
    }

    int ReconstructSize() const override
    {
        return static_cast<int>(sizeof(T) * static_cast<std::size_t>(m_dim));
    }

    DimensionType ReconstructDim() const override
    {
        return m_dim;
    }

    std::uint64_t BufferSize() const override
    {
        return sizeof(QuantizerType) + sizeof(VectorValueType) + sizeof(ModelHeader) +
            m_centroid.size() * sizeof(float) + m_signs.size();
    }

    ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_output) const override
    {
        if (!p_output || !ModelReady()) {
            return ErrorCode::Fail;
        }

        const QuantizerType type = QuantizerType::RaBitQQuantizer;
        const VectorValueType reconstructType = GetEnumValueType<T>();
        const ModelHeader header{
            kModelMagic,
            kModelVersion,
            static_cast<std::int32_t>(m_dim),
            static_cast<std::int32_t>(m_paddedDim),
            static_cast<std::int32_t>(m_bits),
            m_normalize ? 1U : 0U,
            static_cast<std::uint32_t>(m_signs.size())
        };
        if (!Write(p_output, &type, sizeof(type)) ||
            !Write(p_output, &reconstructType, sizeof(reconstructType)) ||
            !Write(p_output, &header, sizeof(header)) ||
            !Write(p_output, m_centroid.data(), m_centroid.size() * sizeof(float)) ||
            !Write(p_output, m_signs.data(), m_signs.size())) {
            return ErrorCode::Fail;
        }
        return ErrorCode::Success;
    }

    ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_input) override
    {
        if (!p_input) {
            return ErrorCode::Fail;
        }

        ModelHeader header{};
        if (p_input->ReadBinary(sizeof(header), reinterpret_cast<char*>(&header)) != sizeof(header) ||
            !LoadHeader(header)) {
            return ErrorCode::FailedParseValue;
        }
        m_centroid.resize(m_paddedDim);
        m_signs.resize(header.signBytes);
        if (p_input->ReadBinary(m_centroid.size() * sizeof(float),
                                reinterpret_cast<char*>(m_centroid.data())) !=
                m_centroid.size() * sizeof(float) ||
            p_input->ReadBinary(m_signs.size(), reinterpret_cast<char*>(m_signs.data())) !=
                m_signs.size()) {
            return ErrorCode::FailedParseValue;
        }
        return ErrorCode::Success;
    }

    ErrorCode LoadQuantizer(std::uint8_t* p_rawBytes) override
    {
        if (p_rawBytes == nullptr) {
            return ErrorCode::Fail;
        }

        ModelHeader header{};
        std::memcpy(&header, p_rawBytes, sizeof(header));
        if (!LoadHeader(header)) {
            return ErrorCode::FailedParseValue;
        }
        p_rawBytes += sizeof(header);
        m_centroid.resize(m_paddedDim);
        m_signs.resize(header.signBytes);
        std::memcpy(m_centroid.data(), p_rawBytes, m_centroid.size() * sizeof(float));
        p_rawBytes += m_centroid.size() * sizeof(float);
        std::memcpy(m_signs.data(), p_rawBytes, m_signs.size());
        return ErrorCode::Success;
    }

    bool GetEnableADC() const override
    {
        return m_enableADC;
    }

    void SetEnableADC(bool p_enableADC) override
    {
        m_enableADC = p_enableADC;
    }

    QuantizerType GetQuantizerType() const override
    {
        return QuantizerType::RaBitQQuantizer;
    }

    VectorValueType GetReconstructType() const override
    {
        return GetEnumValueType<T>();
    }

    DimensionType GetNumSubvectors() const override
    {
        return static_cast<DimensionType>(CodeBytes());
    }

    int GetBase() const override
    {
        return COMMON::Utils::GetBase<T>();
    }

    float* GetL2DistanceTables() override
    {
        return nullptr;
    }

private:
    static constexpr std::uint32_t kModelMagic = 0x47514252U; // RBQG
    static constexpr std::uint32_t kModelVersion = 1U;

    struct ModelHeader
    {
        std::uint32_t magic;
        std::uint32_t version;
        std::int32_t dim;
        std::int32_t paddedDim;
        std::int32_t bits;
        std::uint32_t normalize;
        std::uint32_t signBytes;
    };

    static std::size_t NextPowerOfTwo(std::size_t p_value)
    {
        std::size_t value = 1;
        while (value < (std::max)(std::size_t(64), p_value)) {
            value <<= 1;
        }
        return value;
    }

    static bool Write(const std::shared_ptr<Helper::DiskIO>& p_output,
                      const void* p_data,
                      std::size_t p_bytes)
    {
        return p_output->WriteBinary(p_bytes, const_cast<char*>(
            reinterpret_cast<const char*>(p_data))) == p_bytes;
    }

    bool LoadHeader(const ModelHeader& p_header)
    {
        if (p_header.magic != kModelMagic || p_header.version != kModelVersion ||
            p_header.dim <= 0 || p_header.paddedDim < p_header.dim ||
            (p_header.paddedDim & (p_header.paddedDim - 1)) != 0 ||
            p_header.bits < 1 || p_header.bits > 8 ||
            p_header.signBytes != static_cast<std::uint32_t>(p_header.paddedDim)) {
            return false;
        }
        m_dim = static_cast<DimensionType>(p_header.dim);
        m_paddedDim = static_cast<std::size_t>(p_header.paddedDim);
        m_bits = p_header.bits;
        m_normalize = p_header.normalize != 0;
        m_enableADC = false;
        return true;
    }

    bool ModelReady() const
    {
        return m_dim > 0 && m_paddedDim >= static_cast<std::size_t>(m_dim) &&
            m_bits >= 1 && m_bits <= 8 && m_centroid.size() == m_paddedDim &&
            m_signs.size() == m_paddedDim;
    }

    std::size_t PackedCodeBytes() const
    {
        return (m_paddedDim * static_cast<std::size_t>(m_bits) + 7U) / 8U;
    }

    std::size_t CodeBytes() const
    {
        return PackedCodeBytes() + 2U * sizeof(float);
    }

    void InitializeRotation()
    {
        m_signs.resize(m_paddedDim);
        std::mt19937 generator(0x52425147U);
        std::uniform_int_distribution<int> distribution(0, 1);
        for (auto& sign : m_signs) {
            sign = static_cast<std::uint8_t>(distribution(generator));
        }
    }

    void RotateInput(const T* p_input, std::vector<float>& p_rotated) const
    {
        p_rotated.assign(m_paddedDim, 0.0F);
        double norm = 0.0;
        for (DimensionType d = 0; d < m_dim; ++d) {
            const float value = static_cast<float>(p_input[d]);
            p_rotated[static_cast<std::size_t>(d)] = value;
            norm += static_cast<double>(value) * value;
        }
        if (m_normalize && norm > 0.0) {
            const float inverseNorm = static_cast<float>(1.0 / std::sqrt(norm));
            for (DimensionType d = 0; d < m_dim; ++d) {
                p_rotated[static_cast<std::size_t>(d)] *= inverseNorm;
            }
        }
        ApplySigns(p_rotated);
        Hadamard(p_rotated);
    }

    void InverseRotate(std::vector<float>& p_rotated) const
    {
        Hadamard(p_rotated);
        ApplySigns(p_rotated);
    }

    void ApplySigns(std::vector<float>& p_values) const
    {
        for (std::size_t d = 0; d < m_paddedDim; ++d) {
            if (m_signs[d] != 0) {
                p_values[d] = -p_values[d];
            }
        }
    }

    void Hadamard(std::vector<float>& p_values) const
    {
        for (std::size_t width = 1; width < m_paddedDim; width <<= 1) {
            for (std::size_t base = 0; base < m_paddedDim; base += width << 1) {
                for (std::size_t offset = 0; offset < width; ++offset) {
                    const float left = p_values[base + offset];
                    const float right = p_values[base + width + offset];
                    p_values[base + offset] = left + right;
                    p_values[base + width + offset] = left - right;
                }
            }
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(m_paddedDim));
        for (float& value : p_values) {
            value *= scale;
        }
    }

    void PackCodes(const std::vector<std::uint8_t>& p_codes, std::uint8_t* p_output) const
    {
        std::memset(p_output, 0, PackedCodeBytes());
        const std::uint32_t mask = (1U << m_bits) - 1U;
        for (std::size_t index = 0; index < m_paddedDim; ++index) {
            const std::uint32_t value = static_cast<std::uint32_t>(p_codes[index]) & mask;
            const std::size_t bitOffset = index * static_cast<std::size_t>(m_bits);
            for (int bit = 0; bit < m_bits; ++bit) {
                p_output[(bitOffset + static_cast<std::size_t>(bit)) >> 3U] |=
                    static_cast<std::uint8_t>(((value >> bit) & 1U) <<
                                              ((bitOffset + static_cast<std::size_t>(bit)) & 7U));
            }
        }
    }

    void UnpackCodes(const std::uint8_t* p_input, std::vector<std::uint8_t>& p_codes) const
    {
        p_codes.resize(m_paddedDim);
        for (std::size_t index = 0; index < m_paddedDim; ++index) {
            const std::size_t bitOffset = index * static_cast<std::size_t>(m_bits);
            std::uint8_t value = 0;
            for (int bit = 0; bit < m_bits; ++bit) {
                value |= static_cast<std::uint8_t>(
                    ((p_input[(bitOffset + static_cast<std::size_t>(bit)) >> 3U] >>
                      ((bitOffset + static_cast<std::size_t>(bit)) & 7U)) &
                     1U)
                    << bit);
            }
            p_codes[index] = value;
        }
    }

    void DecodeRotated(const std::uint8_t* p_code, std::vector<float>& p_output) const
    {
        std::vector<std::uint8_t> scalarCodes;
        UnpackCodes(p_code, scalarCodes);
        float delta = 0.0F;
        float lowerValue = 0.0F;
        std::memcpy(&delta, p_code + PackedCodeBytes(), sizeof(delta));
        std::memcpy(&lowerValue, p_code + PackedCodeBytes() + sizeof(delta), sizeof(lowerValue));
        p_output.resize(m_paddedDim);
        rabitqlib::quant::global_scalar::Reconstruct<float, std::uint8_t>(
            scalarCodes.data(), delta, lowerValue, m_paddedDim, p_output.data());
        for (std::size_t d = 0; d < m_paddedDim; ++d) {
            p_output[d] += m_centroid[d];
        }
    }

    std::uint8_t UnpackCode(const std::uint8_t* p_code, std::size_t p_index) const
    {
        const std::size_t bitOffset = p_index * static_cast<std::size_t>(m_bits);
        std::uint8_t value = 0;
        for (int bit = 0; bit < m_bits; ++bit) {
            value |= static_cast<std::uint8_t>(
                ((p_code[(bitOffset + static_cast<std::size_t>(bit)) >> 3U] >>
                  ((bitOffset + static_cast<std::size_t>(bit)) & 7U)) &
                 1U)
                << bit);
        }
        return value;
    }

    void ReadCodeParameters(const std::uint8_t* p_code,
                            float& p_delta,
                            float& p_lowerValue) const
    {
        const std::size_t packedCodeBytes = PackedCodeBytes();
        std::memcpy(&p_delta, p_code + packedCodeBytes, sizeof(p_delta));
        std::memcpy(&p_lowerValue, p_code + packedCodeBytes + sizeof(p_delta),
                    sizeof(p_lowerValue));
    }

    float DecodeCoordinate(const std::uint8_t* p_code,
                           std::size_t p_index,
                           float p_delta,
                           float p_lowerValue,
                           bool p_includeCentroid) const
    {
        float value = static_cast<float>(UnpackCode(p_code, p_index)) * p_delta + p_lowerValue;
        if (p_includeCentroid) {
            value += m_centroid[p_index];
        }
        return value;
    }

    void QueryRotated(const std::uint8_t* p_input, std::vector<float>& p_output) const
    {
        p_output.resize(m_paddedDim);
        std::memcpy(p_output.data(), p_input, m_paddedDim * sizeof(float));
    }

    static float SquaredL2(const std::vector<float>& p_left, const std::vector<float>& p_right)
    {
        double distance = 0.0;
        for (std::size_t d = 0; d < p_left.size(); ++d) {
            const double difference = static_cast<double>(p_left[d]) - p_right[d];
            distance += difference * difference;
        }
        return static_cast<float>(distance);
    }

    static T ConvertOutput(float p_value)
    {
        if constexpr (std::is_same_v<T, float>) {
            return p_value;
        } else {
            const float lower = static_cast<float>((std::numeric_limits<T>::lowest)());
            const float upper = static_cast<float>((std::numeric_limits<T>::max)());
            return static_cast<T>(std::round((std::max)(lower, (std::min)(upper, p_value))));
        }
    }

    DimensionType m_dim;
    std::size_t m_paddedDim;
    int m_bits;
    bool m_normalize;
    bool m_enableADC;
    std::vector<float> m_centroid;
    std::vector<std::uint8_t> m_signs;
};

} // namespace COMMON
} // namespace SPTAG

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_
