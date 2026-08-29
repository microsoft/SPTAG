// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "IQuantizer.h"
#include "inc/Core/VectorSet.h"

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace SPTAG
{
namespace COMMON
{

// Global RaBitQ adapter backed by the official full-code quantizer and estimator.
class RaBitQQuantizer : public IQuantizer
{
public:
    enum class QuantizationMode : std::uint32_t
    {
        Exact = 0,
        Fast = 1,
    };

    struct DistanceEstimate
    {
        float distance = 0.0F;
        float errorBound = 0.0F;
    };

    struct SplitCodeLayout
    {
        std::size_t paddedDimension = 0;
        std::size_t totalBits = 0;
        std::size_t extendedBits = 0;
        std::size_t binaryCodeBytes = 0;
        std::size_t binaryFactorBytes = 0;
        std::size_t binaryBytes = 0;
        std::size_t extendedCodeBytes = 0;
        std::size_t extendedFactorBytes = 0;
        std::size_t extendedBytes = 0;
        std::size_t totalBytes = 0;
    };

    struct SplitDistanceEstimate
    {
        float distance = 0.0F;
        float lowerBound = 0.0F;
        float upperBound = 0.0F;
        float errorBound = 0.0F;
        float intermediateInnerProduct = 0.0F;
    };

    struct SplitBatchLayout
    {
        std::size_t batchSize = 0;
        std::size_t paddedDimension = 0;
        std::size_t totalBits = 0;
        std::size_t extendedBits = 0;
        std::size_t binaryCodeBytes = 0;
        std::size_t binaryFactorBytes = 0;
        std::size_t binaryBytes = 0;
        std::size_t extendedCodeBytesPerVector = 0;
        std::size_t extendedFactorBytesPerVector = 0;
        std::size_t extendedBytesPerVector = 0;
        std::size_t extendedBytes = 0;
        std::size_t totalBytes = 0;
    };

    struct SplitBatchDistanceEstimates
    {
        std::vector<float> distances;
        std::vector<float> lowerBounds;
        std::vector<float> upperBounds;
        std::vector<float> errorBounds;
        std::vector<float> intermediateInnerProducts;

        std::size_t ValidCount() const { return distances.size(); }
    };

    class SplitQueryContext
    {
    public:
        SplitQueryContext() = default;
        SplitQueryContext(const SplitQueryContext&) = delete;
        SplitQueryContext& operator=(const SplitQueryContext&) = delete;
        SplitQueryContext(SplitQueryContext&&) noexcept = default;
        SplitQueryContext& operator=(SplitQueryContext&&) noexcept = default;

        bool Ready() const;

    private:
        friend class RaBitQQuantizer;

        std::vector<float> m_rotated_query;
        std::unique_ptr<rabitqlib::SplitSingleQuery<float>> m_query;
    };

    class SplitBatchQueryContext
    {
    public:
        SplitBatchQueryContext() = default;
        SplitBatchQueryContext(const SplitBatchQueryContext&) = delete;
        SplitBatchQueryContext& operator=(const SplitBatchQueryContext&) = delete;
        SplitBatchQueryContext(SplitBatchQueryContext&&) noexcept = default;
        SplitBatchQueryContext& operator=(SplitBatchQueryContext&&) noexcept = default;

        bool Ready() const;

    private:
        friend class RaBitQQuantizer;

        std::vector<float> m_rotated_query;
        std::unique_ptr<rabitqlib::SplitBatchQuery<float>> m_query;
        bool m_use_high_accuracy = true;
    };

    RaBitQQuantizer() = default;
    RaBitQQuantizer(DimensionType p_dimension, int p_bits, bool p_normalize);
    RaBitQQuantizer(DimensionType p_dimension,
                    int p_bits,
                    bool p_normalize,
                    DistCalcMethod p_metric,
                    QuantizationMode p_quantization_mode = QuantizationMode::Exact);

    ErrorCode Train(const std::shared_ptr<VectorSet>& p_vectors);

    float L2Distance(const std::uint8_t* p_x, const std::uint8_t* p_y) const override;
    float CosineDistance(const std::uint8_t* p_x, const std::uint8_t* p_y) const override;
    DistanceEstimate DistanceWithError(
        const std::uint8_t* p_x, const std::uint8_t* p_y) const;
    SplitCodeLayout GetSplitCodeLayout() const;
    SplitBatchLayout GetSplitBatchLayout() const;
    ErrorCode QuantizeSplitVector(const void* p_vector,
                                  const void* p_local_centroid,
                                  std::uint8_t* p_binary_output,
                                  std::uint8_t* p_extended_output) const;
    ErrorCode QuantizeSplitBatch(const float* p_vectors,
                                 std::size_t p_vector_count,
                                 const float* p_local_centroid,
                                 std::uint8_t* p_binary_output,
                                 std::uint8_t* p_extended_output,
                                 std::size_t& p_valid_count) const;
    ErrorCode PrepareSplitQueryContext(const void* p_query,
                                       const void* p_local_centroid,
                                       SplitQueryContext& p_context) const;
    ErrorCode PrepareSplitBatchQueryContext(const float* p_query,
                                            const float* p_local_centroid,
                                            SplitBatchQueryContext& p_context) const;
    SplitDistanceEstimate EstimateSplitDistance(
        const SplitQueryContext& p_context, const std::uint8_t* p_binary_code) const;
    SplitDistanceEstimate EstimateSplitDistance(const SplitQueryContext& p_context,
                                               const std::uint8_t* p_binary_code,
                                               const std::uint8_t* p_extended_code) const;
    ErrorCode EstimateSplitBatchDistances(
        const SplitBatchQueryContext& p_context,
        const std::uint8_t* p_binary_batch,
        std::size_t p_valid_count,
        SplitBatchDistanceEstimates& p_estimates) const;
    ErrorCode BoostSplitBatchDistance(
        const SplitBatchQueryContext& p_context,
        const std::uint8_t* p_binary_batch,
        const std::uint8_t* p_extended_batch,
        std::size_t p_valid_count,
        std::size_t p_index,
        SplitDistanceEstimate& p_estimate) const;
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
    DistCalcMethod GetMetric() const { return m_metric; }
    QuantizationMode GetQuantizationMode() const { return m_quantization_mode; }
    bool Ready() const;

private:
    struct ModelHeader
    {
        std::uint32_t magic;
        std::uint32_t version;
        std::int32_t dimension;
        std::int32_t paddedDimension;
        std::int32_t bits;
        std::uint32_t normalize;
        std::uint32_t metric;
        std::uint32_t quantizationMode;
        std::uint32_t rotatorType;
        std::uint32_t rotatorStateBytes;
    };

    static constexpr std::uint32_t kModelMagic = 0x32464252U; // RBF2
    static constexpr std::uint32_t kModelVersion = 3U;
    static constexpr std::size_t kCodeFactorCount = 5;
    static constexpr std::size_t kQueryFactorCount = 3;

    ErrorCode Initialize(DimensionType p_dimension,
                         int p_bits,
                         bool p_normalize,
                         DistCalcMethod p_metric,
                         QuantizationMode p_quantization_mode);
    ErrorCode LoadHeader(const ModelHeader& p_header);
    void BuildInverseProjection();
    void Decode(const std::uint8_t* p_code, std::vector<float>& p_output) const;
    void PrepareInput(const float* p_input, std::vector<float>& p_output, bool p_normalize) const;
    void PrepareInput(const float* p_input, std::vector<float>& p_output) const;
    void PrepareCentroidInput(const float* p_input, std::vector<float>& p_output) const;
    void NormalizePrepared(float* p_values) const;
    void ProjectToOriginalSpace(const float* p_rotated_input, float* p_output) const;
    void UnpackCode(const std::uint8_t* p_code, std::uint8_t* p_output) const;
    void ComputeQueryFactors(const float* p_query,
                             const float* p_centroid,
                             float& p_g_add,
                             float& p_k1xsumq,
                             float& p_g_error) const;
    void ReadQueryFactors(const std::uint8_t* p_code,
                          float& p_g_add,
                          float& p_k1xsumq,
                          float& p_g_error) const;
    void ReadCodeFactors(const std::uint8_t* p_code,
                         float& p_f_add,
                         float& p_f_rescale,
                         float& p_f_error,
                         float& p_delta,
                         float& p_lower_value) const;
    void ReadDistanceFactors(const std::uint8_t* p_code,
                             float& p_f_add,
                             float& p_f_rescale) const;
    std::size_t ExtendedBits() const;
    rabitqlib::ex_ipfunc SplitExIpFunc() const;
    rabitqlib::quant::RabitqConfig SplitQueryConfig() const;
    rabitqlib::MetricType OfficialMetric() const;
    bool UseHighAccuracyBatch() const;
    bool ShouldNormalizeInput() const;
    std::size_t PackedCodeBytes() const;
    std::size_t CodeBytes() const;

    DimensionType m_dimension = 0;
    DimensionType m_padded_dimension = 0;
    int m_bits = 0;
    bool m_normalize = false;
    bool m_enable_adc = false;
    DistCalcMethod m_metric = DistCalcMethod::L2;
    QuantizationMode m_quantization_mode = QuantizationMode::Exact;
    rabitqlib::quant::RabitqConfig m_quantizer_config;
    rabitqlib::ex_ipfunc m_ip_func = nullptr;
    std::vector<float> m_centroid;
    std::vector<float> m_inverse_projection;
    std::unique_ptr<rabitqlib::Rotator<float>> m_rotator;
};

} // namespace COMMON
} // namespace SPTAG
