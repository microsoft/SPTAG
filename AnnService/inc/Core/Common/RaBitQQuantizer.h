// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "IQuantizer.h"
#include "inc/Core/VectorSet.h"

#include "rabitqlib/quantization/rabitq.hpp"

#include <memory>
#include <vector>

namespace SPTAG
{
namespace COMMON
{

// Global RaBitQ adapter backed by the official scalar quantizer API.
class RaBitQQuantizer : public IQuantizer
{
public:
    RaBitQQuantizer() = default;
    RaBitQQuantizer(DimensionType p_dimension, int p_bits, bool p_normalize);

    ErrorCode Train(const std::shared_ptr<VectorSet>& p_vectors);

    float L2Distance(const std::uint8_t* p_x, const std::uint8_t* p_y) const override;
    float CosineDistance(const std::uint8_t* p_x, const std::uint8_t* p_y) const override;
    void QuantizeVector(const void* p_vector, std::uint8_t* p_output, bool p_adc = true) const override;
    int QuantizeSize() const override;
    void ReconstructVector(const std::uint8_t* p_code, void* p_output) const override;
    int ReconstructSize() const override;
    DimensionType ReconstructDim() const override;
    std::uint64_t BufferSize() const override;
    ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_output) const override;
    ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_input) override;
    ErrorCode LoadQuantizer(std::uint8_t* p_raw_bytes) override;
    bool GetEnableADC() const override;
    void SetEnableADC(bool p_enable_adc) override;
    QuantizerType GetQuantizerType() const override;
    VectorValueType GetReconstructType() const override;
    DimensionType GetNumSubvectors() const override;
    int GetBase() const override;
    float* GetL2DistanceTables() override;

    DimensionType Dimension() const { return m_dimension; }
    int Bits() const { return m_bits; }
    bool Ready() const;

private:
    struct ModelHeader
    {
        std::uint32_t magic;
        std::uint32_t version;
        std::int32_t dimension;
        std::int32_t bits;
        std::uint32_t normalize;
    };

    static constexpr std::uint32_t kModelMagic = 0x31534252U; // RBS1
    static constexpr std::uint32_t kModelVersion = 1U;

    ErrorCode Initialize(DimensionType p_dimension, int p_bits, bool p_normalize);
    ErrorCode LoadHeader(const ModelHeader& p_header);
    void Decode(const std::uint8_t* p_code, std::vector<float>& p_output) const;
    void CopyInput(const float* p_input, float* p_output) const;
    void ReadCodeParameters(const std::uint8_t* p_code, float& p_delta, float& p_lower_value) const;
    std::size_t CodeBytes() const;

    DimensionType m_dimension = 0;
    int m_bits = 0;
    bool m_normalize = false;
    bool m_enable_adc = false;
    rabitqlib::quant::RabitqConfig m_quantizer_config;
    std::vector<float> m_centroid;
};

} // namespace COMMON
} // namespace SPTAG
