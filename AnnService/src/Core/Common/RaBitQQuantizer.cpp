// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/RaBitQQuantizer.h"

#include "inc/Helper/StringConvert.h"

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace SPTAG
{
namespace COMMON
{

bool RaBitQQuantizer::SplitQueryContext::Ready() const
{
    return m_query != nullptr && !m_rotated_query.empty();
}

bool RaBitQQuantizer::SplitBatchQueryContext::Ready() const
{
    return m_query != nullptr && !m_rotated_query.empty();
}

RaBitQQuantizer::RaBitQQuantizer(DimensionType p_dimension, int p_bits, bool p_normalize)
    : RaBitQQuantizer(
          p_dimension,
          p_bits,
          p_normalize,
          DistCalcMethod::L2,
          QuantizationMode::Exact)
{
}

RaBitQQuantizer::RaBitQQuantizer(DimensionType p_dimension,
                                 int p_bits,
                                 bool p_normalize,
                                 DistCalcMethod p_metric,
                                 QuantizationMode p_quantization_mode)
{
    if (Initialize(p_dimension, p_bits, p_normalize, p_metric, p_quantization_mode) !=
        ErrorCode::Success) {
        throw std::invalid_argument(
            "RaBitQ requires a positive dimension, 1..8 bits, and L2, Cosine, or InnerProduct metric");
    }
}

ErrorCode RaBitQQuantizer::Initialize(DimensionType p_dimension,
                                      int p_bits,
                                      bool p_normalize,
                                      DistCalcMethod p_metric,
                                      QuantizationMode p_quantization_mode)
{
    if (p_dimension < 64 || p_dimension >= 4096 || p_bits < 1 || p_bits > 8 ||
        (p_metric != DistCalcMethod::L2 && p_metric != DistCalcMethod::Cosine &&
         p_metric != DistCalcMethod::InnerProduct) ||
        (p_quantization_mode != QuantizationMode::Exact &&
         p_quantization_mode != QuantizationMode::Fast)) {
        return ErrorCode::FailedParseValue;
    }

    m_dimension = p_dimension;
    m_padded_dimension = static_cast<DimensionType>(
        (static_cast<std::size_t>(p_dimension) + 63U) / 64U * 64U);
    m_bits = p_bits;
    m_normalize = p_normalize;
    m_enable_adc = false;
    m_metric = p_metric;
    m_quantization_mode = p_quantization_mode;
    m_rotator.reset(rabitqlib::choose_rotator<float>(
        static_cast<std::size_t>(m_dimension), rabitqlib::RotatorType::FhtKacRotator));
    if (!m_rotator) {
        return ErrorCode::Fail;
    }
    m_padded_dimension = static_cast<DimensionType>(m_rotator->size());
    m_centroid.assign(static_cast<std::size_t>(m_padded_dimension), 0.0F);
    m_quantizer_config =
        (m_quantization_mode == QuantizationMode::Fast)
        ? rabitqlib::quant::faster_config(
              static_cast<std::size_t>(m_padded_dimension), static_cast<std::size_t>(m_bits))
        : rabitqlib::quant::RabitqConfig();
    m_ip_func = rabitqlib::select_excode_ipfunc(static_cast<std::size_t>(m_bits));
    BuildInverseProjection();
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::Train(const std::shared_ptr<VectorSet>& p_vectors)
{
    if (!Ready() || !p_vectors || p_vectors->GetValueType() != VectorValueType::Float ||
        p_vectors->Dimension() != m_dimension || p_vectors->Count() <= 0) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<double> accumulator(static_cast<std::size_t>(m_padded_dimension), 0.0);
    std::vector<float> prepared;
    for (SizeType i = 0; i < p_vectors->Count(); ++i) {
        const auto* vector = static_cast<const float*>(p_vectors->GetVector(i));
        PrepareInput(vector, prepared);
        for (DimensionType j = 0; j < m_padded_dimension; ++j) {
            accumulator[static_cast<std::size_t>(j)] += prepared[static_cast<std::size_t>(j)];
        }
    }

    const double inverse_count = 1.0 / static_cast<double>(p_vectors->Count());
    for (DimensionType j = 0; j < m_padded_dimension; ++j) {
        m_centroid[static_cast<std::size_t>(j)] =
            static_cast<float>(accumulator[static_cast<std::size_t>(j)] * inverse_count);
    }
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::SetDeterministicRotation(std::uint64_t p_seed)
{
    if (!Ready()) {
        return ErrorCode::Fail;
    }

    std::vector<char> state(m_rotator->dump_bytes(), 0);
    std::uint64_t value = p_seed;
    for (char& byte : state) {
        value += 0x9e3779b97f4a7c15ULL;
        std::uint64_t mixed = value;
        mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
        mixed ^= mixed >> 31U;
        byte = static_cast<char>(mixed & 0xffU);
    }
    m_rotator->load(state.data());
    BuildInverseProjection();
    return ErrorCode::Success;
}

std::shared_ptr<RaBitQQuantizer> RaBitQQuantizer::CloneWithBits(int p_bits) const
{
    if (!Ready() || p_bits < 1 || p_bits > 8) {
        return nullptr;
    }

    auto clone = std::make_shared<RaBitQQuantizer>();
    clone->m_dimension = m_dimension;
    clone->m_padded_dimension = m_padded_dimension;
    clone->m_bits = p_bits;
    clone->m_normalize = m_normalize;
    clone->m_enable_adc = false;
    clone->m_metric = m_metric;
    clone->m_quantization_mode = m_quantization_mode;
    clone->m_rotator.reset(rabitqlib::choose_rotator<float>(
        static_cast<std::size_t>(m_dimension),
        rabitqlib::RotatorType::FhtKacRotator,
        static_cast<std::size_t>(m_padded_dimension)));
    if (!clone->m_rotator) {
        return nullptr;
    }
    std::vector<char> state(m_rotator->dump_bytes(), 0);
    m_rotator->save(state.data());
    clone->m_rotator->load(state.data());
    clone->m_centroid = m_centroid;
    clone->m_inverse_projection = m_inverse_projection;
    clone->m_quantizer_config =
        (m_quantization_mode == QuantizationMode::Fast)
        ? rabitqlib::quant::faster_config(
              static_cast<std::size_t>(m_padded_dimension),
              static_cast<std::size_t>(p_bits))
        : rabitqlib::quant::RabitqConfig();
    clone->m_ip_func =
        rabitqlib::select_excode_ipfunc(static_cast<std::size_t>(p_bits));
    return clone;
}

float RaBitQQuantizer::L2Distance(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    if (m_metric != DistCalcMethod::L2) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "RaBitQ quantizer was trained for %s but L2 distance was requested.\n",
            Helper::Convert::ConvertToString(m_metric).c_str());
        return std::numeric_limits<float>::infinity();
    }

    return DistanceWithError(p_x, p_y).distance;
}

float RaBitQQuantizer::CosineDistance(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    if (m_metric != DistCalcMethod::Cosine && m_metric != DistCalcMethod::InnerProduct) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "RaBitQ quantizer was trained for %s but an inner-product distance was requested.\n",
            Helper::Convert::ConvertToString(m_metric).c_str());
        return std::numeric_limits<float>::infinity();
    }

    return DistanceWithError(p_x, p_y).distance;
}

RaBitQQuantizer::DistanceEstimate
RaBitQQuantizer::DistanceWithError(const std::uint8_t* p_x, const std::uint8_t* p_y) const
{
    DistanceEstimate result;
    if (!Ready() || p_x == nullptr || p_y == nullptr) {
        result.distance = std::numeric_limits<float>::infinity();
        result.errorBound = std::numeric_limits<float>::infinity();
        return result;
    }

    thread_local std::vector<float> prepared_query;
    const float* query = reinterpret_cast<const float*>(p_x);
    float g_add = 0.0F;
    float k1xsumq = 0.0F;
    float g_error = 0.0F;

    if (!m_enable_adc) {
        Decode(p_x, prepared_query);
        if (m_metric == DistCalcMethod::Cosine) {
            NormalizePrepared(prepared_query.data());
        }
        query = prepared_query.data();
        ComputeQueryFactors(query, m_centroid.data(), g_add, k1xsumq, g_error);
    } else {
        ReadQueryFactors(p_x, g_add, k1xsumq, g_error);
    }

    float f_add = 0.0F;
    float f_rescale = 0.0F;
    float f_error = 0.0F;
    float delta = 0.0F;
    float lower_value = 0.0F;
    ReadCodeFactors(p_y, f_add, f_rescale, f_error, delta, lower_value);
    result.distance = rabitqlib::quant::full_est_dist<float, std::uint8_t>(
        p_y,
        query,
        m_ip_func,
        static_cast<std::size_t>(m_padded_dimension),
        static_cast<std::size_t>(m_bits),
        f_add,
        f_rescale,
        g_add,
        k1xsumq);
    result.errorBound = f_error * g_error;
    return result;
}

RaBitQQuantizer::SplitCodeLayout RaBitQQuantizer::GetSplitCodeLayout() const
{
    SplitCodeLayout layout;
    layout.paddedDimension = static_cast<std::size_t>(std::max<DimensionType>(m_padded_dimension, 0));
    layout.totalBits = static_cast<std::size_t>(std::max(m_bits, 0));
    layout.extendedBits = ExtendedBits();
    layout.binaryBytes =
        rabitqlib::BinDataMap<float>::data_bytes(layout.paddedDimension);
    layout.binaryCodeBytes = layout.paddedDimension / 8U;
    layout.binaryFactorBytes = layout.binaryBytes - layout.binaryCodeBytes;
    layout.extendedBytes = rabitqlib::ExDataMap<float>::data_bytes(
        layout.paddedDimension, layout.extendedBits);
    layout.extendedCodeBytes =
        layout.paddedDimension * layout.extendedBits / 8U;
    layout.extendedFactorBytes = layout.extendedBytes - layout.extendedCodeBytes;
    layout.totalBytes = layout.binaryBytes + layout.extendedBytes;
    return layout;
}

RaBitQQuantizer::SplitBatchLayout RaBitQQuantizer::GetSplitBatchLayout() const
{
    SplitBatchLayout layout;
    layout.batchSize = rabitqlib::fastscan::kBatchSize;
    layout.paddedDimension =
        static_cast<std::size_t>(std::max<DimensionType>(m_padded_dimension, 0));
    layout.totalBits = static_cast<std::size_t>(std::max(m_bits, 0));
    layout.extendedBits = ExtendedBits();
    layout.binaryCodeBytes = layout.paddedDimension * layout.batchSize / 8U;
    layout.binaryFactorBytes = sizeof(float) * layout.batchSize * 3U;
    layout.binaryBytes = rabitqlib::BatchDataMap<float>::data_bytes(layout.paddedDimension);
    layout.extendedCodeBytesPerVector =
        layout.paddedDimension * layout.extendedBits / 8U;
    layout.extendedFactorBytesPerVector =
        layout.extendedBits > 0U ? sizeof(float) * 2U : 0U;
    layout.extendedBytesPerVector = rabitqlib::ExDataMap<float>::data_bytes(
        layout.paddedDimension, layout.extendedBits);
    layout.extendedBytes = layout.extendedBytesPerVector * layout.batchSize;
    layout.totalBytes = layout.binaryBytes + layout.extendedBytes;
    return layout;
}

ErrorCode RaBitQQuantizer::QuantizeSplitVector(const void* p_vector,
                                               const void* p_local_centroid,
                                               std::uint8_t* p_binary_output,
                                               std::uint8_t* p_extended_output) const
{
    if (!Ready() || p_vector == nullptr || p_local_centroid == nullptr ||
        p_binary_output == nullptr ||
        (GetSplitCodeLayout().extendedBytes > 0 && p_extended_output == nullptr)) {
        return ErrorCode::FailedParseValue;
    }

    thread_local std::vector<float> prepared_vector;
    thread_local std::vector<float> prepared_centroid;
    PrepareInput(static_cast<const float*>(p_vector), prepared_vector);
    PrepareCentroidInput(static_cast<const float*>(p_local_centroid), prepared_centroid);
    rabitqlib::quant::quantize_split_single(
        prepared_vector.data(),
        prepared_centroid.data(),
        static_cast<std::size_t>(m_padded_dimension),
        ExtendedBits(),
        reinterpret_cast<char*>(p_binary_output),
        reinterpret_cast<char*>(p_extended_output),
        OfficialMetric(),
        m_quantizer_config);
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::QuantizeSplitBatch(const float* p_vectors,
                                              std::size_t p_vector_count,
                                              const float* p_local_centroid,
                                              std::uint8_t* p_binary_output,
                                              std::uint8_t* p_extended_output,
                                              std::size_t& p_valid_count) const
{
    p_valid_count = 0;
    const SplitBatchLayout layout = GetSplitBatchLayout();
    if (!Ready() || p_vectors == nullptr || p_local_centroid == nullptr ||
        p_binary_output == nullptr || p_vector_count == 0U ||
        p_vector_count > layout.batchSize ||
        (layout.extendedBytes > 0U && p_extended_output == nullptr)) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<float> prepared_vectors(layout.paddedDimension * layout.batchSize, 0.0F);
    std::vector<float> prepared_vector;
    std::vector<float> prepared_centroid;
    for (std::size_t i = 0; i < p_vector_count; ++i) {
        PrepareInput(p_vectors + i * static_cast<std::size_t>(m_dimension), prepared_vector);
        std::copy(
            prepared_vector.begin(),
            prepared_vector.end(),
            prepared_vectors.begin() + i * layout.paddedDimension);
    }

    const auto last_valid = prepared_vectors.begin() +
        (p_vector_count - 1U) * layout.paddedDimension;
    for (std::size_t i = p_vector_count; i < layout.batchSize; ++i) {
        std::copy(
            last_valid,
            last_valid + layout.paddedDimension,
            prepared_vectors.begin() + i * layout.paddedDimension);
    }

    PrepareCentroidInput(p_local_centroid, prepared_centroid);
    rabitqlib::quant::quantize_split_batch(
        prepared_vectors.data(),
        prepared_centroid.data(),
        layout.batchSize,
        layout.paddedDimension,
        layout.extendedBits,
        reinterpret_cast<char*>(p_binary_output),
        reinterpret_cast<char*>(p_extended_output),
        OfficialMetric(),
        m_quantizer_config);
    p_valid_count = p_vector_count;
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::PrepareSplitQueryContext(const void* p_query,
                                                    const void* p_local_centroid,
                                                    SplitQueryContext& p_context) const
{
    if (!Ready() || p_query == nullptr || p_local_centroid == nullptr) {
        return ErrorCode::FailedParseValue;
    }

    thread_local std::vector<float> prepared_centroid;
    PrepareInput(static_cast<const float*>(p_query), p_context.m_rotated_query);
    PrepareCentroidInput(static_cast<const float*>(p_local_centroid), prepared_centroid);

    p_context.m_query = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
        p_context.m_rotated_query.data(),
        static_cast<std::size_t>(m_padded_dimension),
        ExtendedBits(),
        SplitQueryConfig(),
        OfficialMetric());

    const float norm = std::sqrt(rabitqlib::euclidean_sqr(
        p_context.m_rotated_query.data(),
        prepared_centroid.data(),
        static_cast<std::size_t>(m_padded_dimension)));
    const float ip = rabitqlib::dot_product(
        p_context.m_rotated_query.data(),
        prepared_centroid.data(),
        static_cast<std::size_t>(m_padded_dimension));
    p_context.m_query->set_g_add(norm, ip);
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::PrepareSplitBatchQueryContext(
    const float* p_query,
    const float* p_local_centroid,
    SplitBatchQueryContext& p_context) const
{
    if (!Ready() || p_query == nullptr || p_local_centroid == nullptr) {
        return ErrorCode::FailedParseValue;
    }

    std::vector<float> prepared_centroid;
    PrepareInput(p_query, p_context.m_rotated_query);
    PrepareCentroidInput(p_local_centroid, prepared_centroid);
    p_context.m_use_high_accuracy = UseHighAccuracyBatch();
    p_context.m_query = std::make_unique<rabitqlib::SplitBatchQuery<float>>(
        p_context.m_rotated_query.data(),
        static_cast<std::size_t>(m_padded_dimension),
        ExtendedBits(),
        OfficialMetric(),
        p_context.m_use_high_accuracy);

    const float norm = std::sqrt(rabitqlib::euclidean_sqr(
        p_context.m_rotated_query.data(),
        prepared_centroid.data(),
        static_cast<std::size_t>(m_padded_dimension)));
    const float ip = rabitqlib::dot_product(
        p_context.m_rotated_query.data(),
        prepared_centroid.data(),
        static_cast<std::size_t>(m_padded_dimension));
    p_context.m_query->set_g_add(norm, ip);
    return ErrorCode::Success;
}

RaBitQQuantizer::SplitDistanceEstimate
RaBitQQuantizer::EstimateSplitDistance(const SplitQueryContext& p_context,
                                       const std::uint8_t* p_binary_code) const
{
    SplitDistanceEstimate result{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0.0F,
    };
    if (!Ready() || !p_context.Ready() || p_binary_code == nullptr) {
        return result;
    }

    rabitqlib::split_single_estdist(
        reinterpret_cast<const char*>(p_binary_code),
        *p_context.m_query,
        static_cast<std::size_t>(m_padded_dimension),
        result.intermediateInnerProduct,
        result.distance,
        result.lowerBound,
        p_context.m_query->g_add(),
        p_context.m_query->g_error());
    result.errorBound = std::max(0.0F, result.distance - result.lowerBound);
    result.upperBound = result.distance + result.errorBound;
    return result;
}

ErrorCode RaBitQQuantizer::EstimateSplitBatchDistances(
    const SplitBatchQueryContext& p_context,
    const std::uint8_t* p_binary_batch,
    std::size_t p_valid_count,
    SplitBatchDistanceEstimates& p_estimates) const
{
    p_estimates = SplitBatchDistanceEstimates();
    const SplitBatchLayout layout = GetSplitBatchLayout();
    if (!Ready() || !p_context.Ready() || p_binary_batch == nullptr ||
        p_valid_count == 0U || p_valid_count > layout.batchSize) {
        return ErrorCode::FailedParseValue;
    }

    std::array<float, rabitqlib::fastscan::kBatchSize> distances{};
    std::array<float, rabitqlib::fastscan::kBatchSize> lower_bounds{};
    std::array<float, rabitqlib::fastscan::kBatchSize> inner_products{};
    rabitqlib::split_batch_estdist(
        reinterpret_cast<const char*>(p_binary_batch),
        *p_context.m_query,
        layout.paddedDimension,
        distances.data(),
        lower_bounds.data(),
        inner_products.data(),
        p_context.m_use_high_accuracy);

    p_estimates.distances.assign(distances.begin(), distances.begin() + p_valid_count);
    p_estimates.lowerBounds.assign(
        lower_bounds.begin(), lower_bounds.begin() + p_valid_count);
    p_estimates.intermediateInnerProducts.assign(
        inner_products.begin(), inner_products.begin() + p_valid_count);
    p_estimates.errorBounds.resize(p_valid_count);
    p_estimates.upperBounds.resize(p_valid_count);
    for (std::size_t i = 0; i < p_valid_count; ++i) {
        p_estimates.errorBounds[i] =
            std::max(0.0F, p_estimates.distances[i] - p_estimates.lowerBounds[i]);
        p_estimates.upperBounds[i] =
            p_estimates.distances[i] + p_estimates.errorBounds[i];
    }
    return ErrorCode::Success;
}

ErrorCode RaBitQQuantizer::BoostSplitBatchDistance(
    const SplitBatchQueryContext& p_context,
    const std::uint8_t* p_binary_batch,
    const std::uint8_t* p_extended_batch,
    std::size_t p_valid_count,
    std::size_t p_index,
    SplitDistanceEstimate& p_estimate) const
{
    p_estimate = SplitDistanceEstimate{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0.0F,
    };
    const SplitBatchLayout layout = GetSplitBatchLayout();
    if (!Ready() || !p_context.Ready() || p_binary_batch == nullptr ||
        p_valid_count == 0U || p_valid_count > layout.batchSize ||
        p_index >= p_valid_count ||
        (layout.extendedBits > 0U && p_extended_batch == nullptr)) {
        return ErrorCode::FailedParseValue;
    }

    SplitBatchDistanceEstimates estimates;
    const ErrorCode status = EstimateSplitBatchDistances(
        p_context, p_binary_batch, p_valid_count, estimates);
    if (status != ErrorCode::Success) {
        return status;
    }

    p_estimate.intermediateInnerProduct =
        estimates.intermediateInnerProducts[p_index];
    if (layout.extendedBits == 0U) {
        p_estimate.distance = estimates.distances[p_index];
        p_estimate.lowerBound = estimates.lowerBounds[p_index];
        p_estimate.upperBound = estimates.upperBounds[p_index];
        p_estimate.errorBound = estimates.errorBounds[p_index];
        return ErrorCode::Success;
    }

    const std::uint8_t* extended = p_extended_batch +
        p_index * layout.extendedBytesPerVector;
    p_estimate.distance = rabitqlib::split_distance_boosting(
        reinterpret_cast<const char*>(extended),
        SplitExIpFunc(),
        *p_context.m_query,
        layout.paddedDimension,
        layout.extendedBits,
        p_estimate.intermediateInnerProduct);

    rabitqlib::ConstBatchDataMap<float> batch(
        reinterpret_cast<const char*>(p_binary_batch), layout.paddedDimension);
    p_estimate.errorBound = std::max(
        0.0F,
        batch.f_error()[p_index] * p_context.m_query->g_error() /
            static_cast<float>(std::size_t{1} << layout.extendedBits));
    p_estimate.lowerBound = p_estimate.distance - p_estimate.errorBound;
    p_estimate.upperBound = p_estimate.distance + p_estimate.errorBound;
    return ErrorCode::Success;
}

RaBitQQuantizer::SplitDistanceEstimate
RaBitQQuantizer::EstimateSplitDistance(const SplitQueryContext& p_context,
                                       const std::uint8_t* p_binary_code,
                                       const std::uint8_t* p_extended_code) const
{
    if (ExtendedBits() == 0U) {
        return EstimateSplitDistance(p_context, p_binary_code);
    }

    SplitDistanceEstimate result{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0.0F,
    };
    if (!Ready() || !p_context.Ready() || p_binary_code == nullptr ||
        p_extended_code == nullptr) {
        return result;
    }

    rabitqlib::split_single_fulldist(
        reinterpret_cast<const char*>(p_binary_code),
        reinterpret_cast<const char*>(p_extended_code),
        SplitExIpFunc(),
        *p_context.m_query,
        static_cast<std::size_t>(m_padded_dimension),
        ExtendedBits(),
        result.distance,
        result.lowerBound,
        result.intermediateInnerProduct,
        p_context.m_query->g_add(),
        p_context.m_query->g_error());
    result.errorBound = std::max(0.0F, result.distance - result.lowerBound);
    result.upperBound = result.distance + result.errorBound;
    return result;
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
        float g_add = 0.0F;
        float k1xsumq = 0.0F;
        float g_error = 0.0F;
        ComputeQueryFactors(input, m_centroid.data(), g_add, k1xsumq, g_error);
        std::uint8_t* factors = p_output + query_bytes;
        std::memcpy(factors, &g_add, sizeof(g_add));
        std::memcpy(factors + sizeof(float), &k1xsumq, sizeof(k1xsumq));
        std::memcpy(factors + 2 * sizeof(float), &g_error, sizeof(g_error));
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
                OfficialMetric());
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
                OfficialMetric(),
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
    ProjectToOriginalSpace(reconstructed.data(), static_cast<float*>(p_output));
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
        m_centroid.size() * sizeof(float) + (m_rotator ? m_rotator->dump_bytes() : 0U);
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
        m_padded_dimension,
        m_bits,
        m_normalize ? 1U : 0U,
        static_cast<std::uint32_t>(m_metric),
        static_cast<std::uint32_t>(m_quantization_mode),
        static_cast<std::uint32_t>(rabitqlib::RotatorType::FhtKacRotator),
        static_cast<std::uint32_t>(m_rotator->dump_bytes()),
    };
    std::vector<char> rotator_state(m_rotator->dump_bytes(), 0);
    m_rotator->save(rotator_state.data());
    if (p_output->WriteBinary(sizeof(type), reinterpret_cast<char*>(&type)) != sizeof(type) ||
        p_output->WriteBinary(sizeof(reconstruct_type), reinterpret_cast<char*>(&reconstruct_type)) != sizeof(reconstruct_type) ||
        p_output->WriteBinary(sizeof(header), reinterpret_cast<char*>(&header)) != sizeof(header) ||
        p_output->WriteBinary(m_centroid.size() * sizeof(float), reinterpret_cast<char*>(const_cast<float*>(m_centroid.data()))) !=
            m_centroid.size() * sizeof(float) ||
        p_output->WriteBinary(rotator_state.size(), rotator_state.data()) != rotator_state.size()) {
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
    std::vector<char> rotator_state;
    if (p_input->ReadBinary(sizeof(header), reinterpret_cast<char*>(&header)) != sizeof(header) ||
        LoadHeader(header) != ErrorCode::Success) {
        return ErrorCode::FailedParseValue;
    }
    rotator_state.resize(static_cast<std::size_t>(header.rotatorStateBytes));
    if (p_input->ReadBinary(m_centroid.size() * sizeof(float), reinterpret_cast<char*>(m_centroid.data())) !=
            m_centroid.size() * sizeof(float) ||
        p_input->ReadBinary(rotator_state.size(), rotator_state.data()) != rotator_state.size()) {
        return ErrorCode::FailedParseValue;
    }
    m_rotator->load(rotator_state.data());
    BuildInverseProjection();
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
    p_raw_bytes += m_centroid.size() * sizeof(float);
    m_rotator->load(reinterpret_cast<const char*>(p_raw_bytes));
    BuildInverseProjection();
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
        m_inverse_projection.size() ==
            static_cast<std::size_t>(m_dimension) * static_cast<std::size_t>(m_padded_dimension) &&
        m_ip_func != nullptr && m_rotator != nullptr &&
        m_rotator->size() == static_cast<std::size_t>(m_padded_dimension) &&
        (m_metric == DistCalcMethod::L2 || m_metric == DistCalcMethod::Cosine ||
         m_metric == DistCalcMethod::InnerProduct);
}

ErrorCode RaBitQQuantizer::LoadHeader(const ModelHeader& p_header)
{
    if (p_header.magic != kModelMagic) {
        return ErrorCode::FailedParseValue;
    }
    if (p_header.version == 2U) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "RaBitQ model version 2 is no longer supported. Retrain the quantizer.\n");
        return ErrorCode::FailedParseValue;
    }
    if (p_header.version != kModelVersion ||
        p_header.dimension < 64 || p_header.dimension >= 4096 ||
        p_header.paddedDimension < p_header.dimension ||
        p_header.paddedDimension % 64 != 0 || p_header.bits < 1 || p_header.bits > 8 ||
        p_header.normalize > 1U ||
        p_header.metric > static_cast<std::uint32_t>(DistCalcMethod::InnerProduct) ||
        p_header.quantizationMode > static_cast<std::uint32_t>(QuantizationMode::Fast) ||
        p_header.rotatorType !=
            static_cast<std::uint32_t>(rabitqlib::RotatorType::FhtKacRotator) ||
        p_header.rotatorStateBytes == 0U) {
        return ErrorCode::FailedParseValue;
    }
    const ErrorCode status =
        Initialize(
            p_header.dimension,
            p_header.bits,
            p_header.normalize != 0,
            static_cast<DistCalcMethod>(p_header.metric),
            static_cast<QuantizationMode>(p_header.quantizationMode));
    return status == ErrorCode::Success && m_padded_dimension == p_header.paddedDimension &&
        m_rotator != nullptr &&
        m_rotator->dump_bytes() == static_cast<std::size_t>(p_header.rotatorStateBytes)
        ? ErrorCode::Success
        : ErrorCode::FailedParseValue;
}

void RaBitQQuantizer::BuildInverseProjection()
{
    if (m_rotator == nullptr || m_dimension <= 0 || m_padded_dimension <= 0) {
        m_inverse_projection.clear();
        return;
    }

    const std::size_t dim = static_cast<std::size_t>(m_dimension);
    const std::size_t padded_dim = static_cast<std::size_t>(m_padded_dimension);
    m_inverse_projection.assign(dim * padded_dim, 0.0F);
    std::vector<float> basis(dim, 0.0F);
    for (std::size_t i = 0; i < dim; ++i) {
        std::fill(basis.begin(), basis.end(), 0.0F);
        basis[i] = 1.0F;
        m_rotator->rotate(
            basis.data(), m_inverse_projection.data() + (i * padded_dim));
    }
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
    const float* p_input, std::vector<float>& p_output, bool p_normalize) const
{
    thread_local std::vector<float> padded_input;
    padded_input.assign(static_cast<std::size_t>(m_padded_dimension), 0.0F);
    std::copy(p_input, p_input + m_dimension, padded_input.begin());
    if (p_normalize) {
        double norm = 0.0;
        for (DimensionType i = 0; i < m_dimension; ++i) {
            norm += static_cast<double>(p_input[i]) * p_input[i];
        }
        if (norm > 0.0) {
            const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
            for (DimensionType i = 0; i < m_dimension; ++i) {
                padded_input[static_cast<std::size_t>(i)] = p_input[i] * inverse_norm;
            }
        }
    }
    p_output.assign(static_cast<std::size_t>(m_padded_dimension), 0.0F);
    m_rotator->rotate(padded_input.data(), p_output.data());
}

void RaBitQQuantizer::PrepareInput(
    const float* p_input, std::vector<float>& p_output) const
{
    PrepareInput(p_input, p_output, ShouldNormalizeInput());
}

void RaBitQQuantizer::PrepareCentroidInput(
    const float* p_input, std::vector<float>& p_output) const
{
    PrepareInput(p_input, p_output, false);
}

void RaBitQQuantizer::NormalizePrepared(float* p_values) const
{
    p_output.assign(static_cast<std::size_t>(m_padded_dimension), 0.0F);
    std::copy(p_input, p_input + m_dimension, p_output.begin());
    if (!m_normalize) {
        return;
    }

    double norm = 0.0;
    for (DimensionType i = 0; i < m_padded_dimension; ++i) {
        norm += static_cast<double>(p_values[i]) * p_values[i];
    }
    if (norm <= 0.0) {
        return;
    }

    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
    for (DimensionType i = 0; i < m_padded_dimension; ++i) {
        p_values[static_cast<std::size_t>(i)] *= inverse_norm;
    }
}

void RaBitQQuantizer::ProjectToOriginalSpace(const float* p_rotated_input, float* p_output) const
{
    const std::size_t dim = static_cast<std::size_t>(m_dimension);
    const std::size_t padded_dim = static_cast<std::size_t>(m_padded_dimension);
    for (std::size_t i = 0; i < dim; ++i) {
        p_output[i] = rabitqlib::dot_product(
            p_rotated_input,
            m_inverse_projection.data() + (i * padded_dim),
            padded_dim);
    }
}

void RaBitQQuantizer::ComputeQueryFactors(const float* p_query,
                                          const float* p_centroid,
                                          float& p_g_add,
                                          float& p_k1xsumq,
                                          float& p_g_error) const
{
    p_k1xsumq = -0.5F *
        std::accumulate(p_query, p_query + m_padded_dimension, 0.0F);
    const float norm = std::sqrt(rabitqlib::euclidean_sqr(
        p_query, p_centroid, static_cast<std::size_t>(m_padded_dimension)));
    p_g_error = norm;
    if (m_metric == DistCalcMethod::L2) {
        p_g_add = norm * norm;
    } else {
        p_g_add = -rabitqlib::dot_product(
            p_query, p_centroid, static_cast<std::size_t>(m_padded_dimension));
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

void RaBitQQuantizer::ReadQueryFactors(const std::uint8_t* p_code,
                                       float& p_g_add,
                                       float& p_k1xsumq,
                                       float& p_g_error) const
{
    const std::uint8_t* factors =
        p_code + static_cast<std::size_t>(m_padded_dimension) * sizeof(float);
    std::memcpy(&p_g_add, factors, sizeof(float));
    std::memcpy(&p_k1xsumq, factors + sizeof(float), sizeof(float));
    std::memcpy(&p_g_error, factors + 2 * sizeof(float), sizeof(float));
}

void RaBitQQuantizer::ReadDistanceFactors(const std::uint8_t* p_code,
                                          float& p_f_add,
                                          float& p_f_rescale) const
{
    const std::uint8_t* factors = p_code + PackedCodeBytes();
    std::memcpy(&p_f_add, factors, sizeof(float));
    std::memcpy(&p_f_rescale, factors + sizeof(float), sizeof(float));
}

std::size_t RaBitQQuantizer::ExtendedBits() const
{
    return m_bits > 1 ? static_cast<std::size_t>(m_bits - 1) : 0U;
}

rabitqlib::ex_ipfunc RaBitQQuantizer::SplitExIpFunc() const
{
    return ExtendedBits() > 0 ? rabitqlib::select_excode_ipfunc(ExtendedBits()) : nullptr;
}

rabitqlib::quant::RabitqConfig RaBitQQuantizer::SplitQueryConfig() const
{
    if (m_quantization_mode == QuantizationMode::Fast) {
        return rabitqlib::quant::faster_config(
            static_cast<std::size_t>(m_padded_dimension),
            rabitqlib::SplitSingleQuery<float>::kNumBits);
    }

    return rabitqlib::quant::RabitqConfig();
}

rabitqlib::MetricType RaBitQQuantizer::OfficialMetric() const
{
    return m_metric == DistCalcMethod::L2 ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;
}

bool RaBitQQuantizer::UseHighAccuracyBatch() const
{
    return m_quantization_mode == QuantizationMode::Exact;
}

bool RaBitQQuantizer::ShouldNormalizeInput() const
{
    return m_normalize || m_metric == DistCalcMethod::Cosine;
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
