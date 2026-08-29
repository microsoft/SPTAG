// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"

#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Quantizer/Training.h"

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace SPTAG;

namespace
{

constexpr SizeType kVectorCount = 96;
constexpr DimensionType kDimension = 128;
constexpr int kRaBitQBits = 3;
constexpr DimensionType kRaBitQCodeBytes =
    kDimension * kRaBitQBits / 8 + 5 * sizeof(float);
constexpr const char* kQuantizerFile = "rabitq_global_quantizer_test.bin";

struct SavedRaBitQModelHeader
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

std::shared_ptr<VectorSet> MakeRawVectors(SizeType count, DimensionType dimension)
{
    ByteArray bytes = ByteArray::Alloc(
        sizeof(float) * static_cast<std::size_t>(count) * static_cast<std::size_t>(dimension));
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (SizeType vector = 0; vector < count; ++vector) {
        for (DimensionType dim = 0; dim < dimension; ++dim) {
            values[static_cast<std::size_t>(vector) * static_cast<std::size_t>(dimension) + dim] =
                static_cast<float>(vector) * 0.125F + static_cast<float>(dim) * 0.01F;
        }
    }
    return std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, dimension, count);
}

std::shared_ptr<VectorSet> MakeRawVectors()
{
    return MakeRawVectors(kVectorCount, kDimension);
}

std::vector<float> MakeLocalCentroid(float offset, float slope)
{
    std::vector<float> centroid(kDimension, 0.0F);
    for (DimensionType dim = 0; dim < kDimension; ++dim) {
        centroid[static_cast<std::size_t>(dim)] =
            offset + slope * static_cast<float>(dim);
    }
    return centroid;
}

std::vector<std::uint8_t> QuantizeSingleVector(
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer,
    const void* p_vector,
    bool p_adc)
{
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(p_adc ? p_quantizer->QuantizeSize()
                                       : p_quantizer->GetNumSubvectors()));
    p_quantizer->QuantizeVector(
        p_vector, reinterpret_cast<std::uint8_t*>(output.data()), p_adc);
    return output;
}

std::vector<char> ExtractRotatorState(
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer,
    SavedRaBitQModelHeader& p_header)
{
    auto output = f_createIO();
    BOOST_REQUIRE(output != nullptr);
    BOOST_REQUIRE(output->Initialize(kQuantizerFile, std::ios::binary | std::ios::out));
    BOOST_REQUIRE(p_quantizer->SaveQuantizer(output) == ErrorCode::Success);
    output->ShutDown();

    std::ifstream input(kQuantizerFile, std::ios::binary);
    BOOST_REQUIRE(input.good());
    input.seekg(static_cast<std::streamoff>(sizeof(QuantizerType) + sizeof(VectorValueType)));
    input.read(reinterpret_cast<char*>(&p_header), sizeof(p_header));
    BOOST_REQUIRE(input.good());
    input.seekg(
        static_cast<std::streamoff>(sizeof(float) * p_header.paddedDimension), std::ios::cur);
    std::vector<char> rotator_state(p_header.rotatorStateBytes, 0);
    input.read(rotator_state.data(), static_cast<std::streamsize>(rotator_state.size()));
    BOOST_REQUIRE(input.good());
    input.close();
    std::remove(kQuantizerFile);
    return rotator_state;
}

std::vector<float> PrepareOfficialRotatedInput(const float* p_input,
                                               DimensionType p_dimension,
                                               std::size_t p_padded_dimension,
                                               const rabitqlib::Rotator<float>& p_rotator,
                                               bool p_normalize)
{
    std::vector<float> padded(p_padded_dimension, 0.0F);
    std::copy(p_input, p_input + p_dimension, padded.begin());
    if (p_normalize) {
        double norm = 0.0;
        for (DimensionType dim = 0; dim < p_dimension; ++dim) {
            norm += static_cast<double>(p_input[dim]) * p_input[dim];
        }
        if (norm > 0.0) {
            const float inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
            for (DimensionType dim = 0; dim < p_dimension; ++dim) {
                padded[static_cast<std::size_t>(dim)] = p_input[dim] * inverse_norm;
            }
        }
    }

    std::vector<float> rotated(p_padded_dimension, 0.0F);
    p_rotator.rotate(padded.data(), rotated.data());
    return rotated;
}

float ComputeExpectedDistance(
    const float* p_left, const float* p_right, DistCalcMethod p_metric)
{
    if (p_metric == DistCalcMethod::L2) {
        return COMMON::DistanceUtils::ComputeL2Distance(p_left, p_right, kDimension);
    }

    std::vector<float> left(kDimension, 0.0F);
    std::vector<float> right(kDimension, 0.0F);
    auto normalize = [](const float* input, std::vector<float>& output) {
        double norm = 0.0;
        for (DimensionType dim = 0; dim < kDimension; ++dim) {
            norm += static_cast<double>(input[dim]) * input[dim];
        }
        if (norm <= 0.0) {
            return;
        }
        const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm));
        for (DimensionType dim = 0; dim < kDimension; ++dim) {
            output[dim] = input[dim] * inv_norm;
        }
    };

    normalize(p_left, left);
    normalize(p_right, right);
    return COMMON::DistanceUtils::ComputeCosineDistance(left.data(), right.data(), kDimension);
}

std::shared_ptr<VectorSet> QuantizeVectors(
    const std::shared_ptr<VectorSet>& p_raw,
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer)
{
    const DimensionType code_bytes = p_quantizer->GetNumSubvectors();
    ByteArray bytes = ByteArray::Alloc(
        static_cast<std::size_t>(p_raw->Count()) * static_cast<std::size_t>(code_bytes));
    auto output = std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::UInt8, code_bytes, p_raw->Count());
    p_quantizer->SetEnableADC(false);
    for (SizeType vector = 0; vector < p_raw->Count(); ++vector) {
        p_quantizer->QuantizeVector(
            p_raw->GetVector(vector),
            reinterpret_cast<std::uint8_t*>(output->GetVector(vector)),
            false);
    }
    return output;
}

std::shared_ptr<COMMON::IQuantizer> SaveAndLoad(
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer)
{
    auto output = f_createIO();
    BOOST_REQUIRE(output != nullptr);
    BOOST_REQUIRE(output->Initialize(kQuantizerFile, std::ios::binary | std::ios::out));
    BOOST_REQUIRE(p_quantizer->SaveQuantizer(output) == ErrorCode::Success);
    output->ShutDown();

    auto input = f_createIO();
    BOOST_REQUIRE(input != nullptr);
    BOOST_REQUIRE(input->Initialize(kQuantizerFile, std::ios::binary | std::ios::in));
    auto loaded = COMMON::IQuantizer::LoadIQuantizer(input);
    input->ShutDown();
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_CHECK(loaded->GetQuantizerType() == QuantizerType::RaBitQQuantizer);
    return loaded;
}

} // namespace

BOOST_AUTO_TEST_SUITE(RaBitQQuantizerTest)

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQRejectsUnsupportedRotatorDimensions)
{
    BOOST_CHECK_THROW(
        COMMON::RaBitQQuantizer(
            63,
            kRaBitQBits,
            false,
            DistCalcMethod::L2,
            COMMON::RaBitQQuantizer::QuantizationMode::Exact),
        std::invalid_argument);
    BOOST_CHECK_THROW(
        COMMON::RaBitQQuantizer(
            4096,
            kRaBitQBits,
            false,
            DistCalcMethod::L2,
            COMMON::RaBitQQuantizer::QuantizationMode::Exact),
        std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQRotationAffectsCodesAndSaveLoadStaysStable)
{
    std::remove(kQuantizerFile);
    const auto raw = MakeRawVectors();

    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const auto base_code = QuantizeSingleVector(quantizer, raw->GetVector(11), false);
    quantizer->SetEnableADC(true);
    const auto query = QuantizeSingleVector(quantizer, raw->GetVector(23), true);
    const auto estimate = quantizer->DistanceWithError(query.data(), base_code.data());

    auto loaded = std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(SaveAndLoad(quantizer));
    BOOST_REQUIRE(loaded != nullptr);
    const auto loaded_code = QuantizeSingleVector(loaded, raw->GetVector(11), false);
    loaded->SetEnableADC(true);
    const auto loaded_query = QuantizeSingleVector(loaded, raw->GetVector(23), true);
    const auto loaded_estimate = loaded->DistanceWithError(loaded_query.data(), loaded_code.data());

    BOOST_CHECK(base_code == loaded_code);
    BOOST_CHECK(query == loaded_query);
    BOOST_CHECK_EQUAL(estimate.distance, loaded_estimate.distance);
    BOOST_CHECK_EQUAL(estimate.errorBound, loaded_estimate.errorBound);

    auto other_quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(other_quantizer->Train(raw) == ErrorCode::Success);
    const auto other_code = QuantizeSingleVector(other_quantizer, raw->GetVector(11), false);
    BOOST_CHECK(base_code != other_code);

    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQReconstructsOriginalSpaceForPaddedDimensions)
{
    constexpr DimensionType odd_dimension = 65;
    ByteArray bytes = ByteArray::Alloc(sizeof(float) * odd_dimension);
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (DimensionType dim = 0; dim < odd_dimension; ++dim) {
        values[dim] = 0.2F + static_cast<float>(dim) * 0.015F;
    }

    auto raw = std::make_shared<BasicVectorSet>(bytes, VectorValueType::Float, odd_dimension, 1);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        odd_dimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    std::vector<std::uint8_t> code(quantizer->GetNumSubvectors(), 0);
    quantizer->QuantizeVector(raw->GetVector(0), code.data(), false);
    std::vector<float> reconstructed(odd_dimension, 0.0F);
    quantizer->ReconstructVector(code.data(), reconstructed.data());

    for (DimensionType dim = 0; dim < odd_dimension; ++dim) {
        BOOST_CHECK_CLOSE(reconstructed[static_cast<std::size_t>(dim)], values[dim], 1e-3F);
    }
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQNonAdcDistanceRetainsRotatedTail)
{
    constexpr DimensionType odd_dimension = 65;
    ByteArray bytes = ByteArray::Alloc(sizeof(float) * static_cast<std::size_t>(odd_dimension) * 2U);
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (DimensionType dim = 0; dim < odd_dimension; ++dim) {
        values[dim] = -0.8F + static_cast<float>(dim) * 0.02F;
        values[static_cast<std::size_t>(odd_dimension) + dim] =
            1.1F - static_cast<float>(dim) * 0.015F;
    }
    const auto raw = std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, odd_dimension, 2);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        odd_dimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    std::vector<float> centroid(odd_dimension, 0.0F);
    for (DimensionType dim = 0; dim < odd_dimension; ++dim) {
        centroid[static_cast<std::size_t>(dim)] =
            0.5F * (values[dim] + values[static_cast<std::size_t>(odd_dimension) + dim]);
    }

    const auto query_code = QuantizeSingleVector(quantizer, centroid.data(), false);
    const auto base_code = QuantizeSingleVector(quantizer, raw->GetVector(0), false);

    quantizer->SetEnableADC(false);
    const float code_distance = quantizer->L2Distance(query_code.data(), base_code.data());
    quantizer->SetEnableADC(true);
    const auto adc_query = QuantizeSingleVector(quantizer, centroid.data(), true);
    const float adc_distance = quantizer->L2Distance(adc_query.data(), base_code.data());

    BOOST_CHECK(std::isfinite(code_distance));
    BOOST_CHECK(std::isfinite(adc_distance));
    BOOST_CHECK_SMALL(std::fabs(code_distance - adc_distance), 1e-4F);
}

BOOST_AUTO_TEST_CASE(OfficialSplitRaBitQLocalCentroidsChangeCodes)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const auto layout = quantizer->GetSplitCodeLayout();
    BOOST_REQUIRE(layout.binaryBytes > 0);
    BOOST_REQUIRE(layout.extendedBits > 0);
    BOOST_REQUIRE(layout.extendedBytes > 0);

    const auto centroid_a = MakeLocalCentroid(-0.75F, 0.015F);
    const auto centroid_b = MakeLocalCentroid(1.5F, -0.02F);
    std::vector<std::uint8_t> binary_a(layout.binaryBytes, 0);
    std::vector<std::uint8_t> extended_a(layout.extendedBytes, 0);
    std::vector<std::uint8_t> binary_b(layout.binaryBytes, 0);
    std::vector<std::uint8_t> extended_b(layout.extendedBytes, 0);

    BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                      raw->GetVector(18),
                      centroid_a.data(),
                      binary_a.data(),
                      extended_a.data()) == ErrorCode::Success);
    BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                      raw->GetVector(18),
                      centroid_b.data(),
                      binary_b.data(),
                      extended_b.data()) == ErrorCode::Success);

    const bool binary_changed =
        std::memcmp(binary_a.data(), binary_b.data(), layout.binaryCodeBytes) != 0;
    const bool extended_changed =
        std::memcmp(extended_a.data(), extended_b.data(), layout.extendedCodeBytes) != 0;
    BOOST_CHECK(binary_changed || extended_changed);
}

BOOST_AUTO_TEST_CASE(OfficialSplitRaBitQEstimatesRemainFiniteForL2AndCosine)
{
    const auto raw = MakeRawVectors();
    const auto local_centroid = MakeLocalCentroid(0.25F, -0.01F);

    for (const auto metric : {DistCalcMethod::L2, DistCalcMethod::Cosine}) {
        auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
            kDimension,
            5,
            false,
            metric,
            COMMON::RaBitQQuantizer::QuantizationMode::Exact);
        BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

        const auto layout = quantizer->GetSplitCodeLayout();
        std::vector<std::uint8_t> binary(layout.binaryBytes, 0);
        std::vector<std::uint8_t> extended(layout.extendedBytes, 0);
        BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                          raw->GetVector(22),
                          local_centroid.data(),
                          binary.data(),
                          extended.data()) == ErrorCode::Success);

        COMMON::RaBitQQuantizer::SplitQueryContext query_context;
        BOOST_REQUIRE(quantizer->PrepareSplitQueryContext(
                          raw->GetVector(11),
                          local_centroid.data(),
                          query_context) == ErrorCode::Success);
        BOOST_REQUIRE(query_context.Ready());

        const auto one_bit = quantizer->EstimateSplitDistance(query_context, binary.data());
        const auto full = quantizer->EstimateSplitDistance(
            query_context, binary.data(), extended.data());
        BOOST_CHECK(std::isfinite(one_bit.distance));
        BOOST_CHECK(std::isfinite(one_bit.lowerBound));
        BOOST_CHECK(std::isfinite(full.distance));
        BOOST_CHECK(std::isfinite(full.lowerBound));
        BOOST_CHECK(one_bit.lowerBound <= one_bit.distance + 1e-5F);
        BOOST_CHECK(full.lowerBound <= full.distance + 1e-5F);
    }
}

BOOST_AUTO_TEST_CASE(OfficialSplitRaBitQFastModeUsesOfficialSplitQueryConfig)
{
    const auto raw = MakeRawVectors();
    const auto local_centroid = MakeLocalCentroid(1.7F, -0.031F);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        8,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Fast);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const auto layout = quantizer->GetSplitCodeLayout();
    std::vector<std::uint8_t> binary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> extended(layout.extendedBytes, 0);
    BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                      raw->GetVector(27),
                      local_centroid.data(),
                      binary.data(),
                      extended.data()) == ErrorCode::Success);

    COMMON::RaBitQQuantizer::SplitQueryContext query_context;
    BOOST_REQUIRE(quantizer->PrepareSplitQueryContext(
                      raw->GetVector(43),
                      local_centroid.data(),
                      query_context) == ErrorCode::Success);
    const auto actual = quantizer->EstimateSplitDistance(
        query_context, binary.data(), extended.data());

    SavedRaBitQModelHeader header{};
    const auto rotator_state = ExtractRotatorState(quantizer, header);
    std::unique_ptr<rabitqlib::Rotator<float>> rotator(rabitqlib::choose_rotator<float>(
        static_cast<std::size_t>(header.dimension),
        rabitqlib::RotatorType::FhtKacRotator,
        static_cast<std::size_t>(header.paddedDimension)));
    rotator->load(rotator_state.data());

    const auto rotated_query = PrepareOfficialRotatedInput(
        reinterpret_cast<const float*>(raw->GetVector(43)),
        kDimension,
        static_cast<std::size_t>(header.paddedDimension),
        *rotator,
        false);
    const auto rotated_centroid = PrepareOfficialRotatedInput(
        local_centroid.data(),
        kDimension,
        static_cast<std::size_t>(header.paddedDimension),
        *rotator,
        false);

    rabitqlib::SplitSingleQuery<float> reference_query(
        rotated_query.data(),
        static_cast<std::size_t>(header.paddedDimension),
        static_cast<std::size_t>(header.bits - 1),
        rabitqlib::quant::faster_config(
            static_cast<std::size_t>(header.paddedDimension),
            rabitqlib::SplitSingleQuery<float>::kNumBits),
        rabitqlib::METRIC_L2);
    const float norm = std::sqrt(rabitqlib::euclidean_sqr(
        rotated_query.data(),
        rotated_centroid.data(),
        static_cast<std::size_t>(header.paddedDimension)));
    const float ip = rabitqlib::dot_product(
        rotated_query.data(),
        rotated_centroid.data(),
        static_cast<std::size_t>(header.paddedDimension));
    reference_query.set_g_add(norm, ip);

    float reference_distance = 0.0F;
    float reference_lower_bound = 0.0F;
    float reference_ip = 0.0F;
    rabitqlib::split_single_fulldist(
        reinterpret_cast<const char*>(binary.data()),
        reinterpret_cast<const char*>(extended.data()),
        rabitqlib::select_excode_ipfunc(static_cast<std::size_t>(header.bits - 1)),
        reference_query,
        static_cast<std::size_t>(header.paddedDimension),
        static_cast<std::size_t>(header.bits - 1),
        reference_distance,
        reference_lower_bound,
        reference_ip,
        reference_query.g_add(),
        reference_query.g_error());

    BOOST_CHECK_CLOSE(actual.distance, reference_distance, 1e-4F);
    BOOST_CHECK_CLOSE(actual.lowerBound, reference_lower_bound, 1e-4F);
    BOOST_CHECK_CLOSE(actual.intermediateInnerProduct, reference_ip, 1e-4F);
}

BOOST_AUTO_TEST_CASE(OfficialSplitRaBitQCosineDoesNotNormalizeLocalCentroids)
{
    const auto raw = MakeRawVectors();
    auto centroid = MakeLocalCentroid(2.0F, 0.05F);
    std::vector<float> scaled_centroid = centroid;
    for (float& value : scaled_centroid) {
        value *= 2.5F;
    }

    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::Cosine,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const auto layout = quantizer->GetSplitCodeLayout();
    std::vector<std::uint8_t> binary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> extended(layout.extendedBytes, 0);
    std::vector<std::uint8_t> scaled_binary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> scaled_extended(layout.extendedBytes, 0);
    BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                      raw->GetVector(21),
                      centroid.data(),
                      binary.data(),
                      extended.data()) == ErrorCode::Success);
    BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                      raw->GetVector(21),
                      scaled_centroid.data(),
                      scaled_binary.data(),
                      scaled_extended.data()) == ErrorCode::Success);

    BOOST_CHECK(
        std::memcmp(binary.data(), scaled_binary.data(), layout.binaryBytes) != 0 ||
        std::memcmp(extended.data(), scaled_extended.data(), layout.extendedBytes) != 0);
}

BOOST_AUTO_TEST_CASE(OfficialSplitRaBitQFullEstimateUsesRemainingBits)
{
    const auto raw = MakeRawVectors();
    const auto local_centroid = MakeLocalCentroid(-0.4F, 0.006F);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const auto layout = quantizer->GetSplitCodeLayout();
    BOOST_REQUIRE(layout.extendedCodeBytes > 0);
    std::vector<std::uint8_t> binary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> extended(layout.extendedBytes, 0);
    BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                      raw->GetVector(27),
                      local_centroid.data(),
                      binary.data(),
                      extended.data()) == ErrorCode::Success);

    COMMON::RaBitQQuantizer::SplitQueryContext query_context;
    BOOST_REQUIRE(quantizer->PrepareSplitQueryContext(
                      raw->GetVector(43),
                      local_centroid.data(),
                      query_context) == ErrorCode::Success);

    const auto one_bit = quantizer->EstimateSplitDistance(query_context, binary.data());
    const auto full = quantizer->EstimateSplitDistance(
        query_context, binary.data(), extended.data());
    std::vector<std::uint8_t> modified_extended = extended;
    modified_extended[0] ^= 0x1U;
    const auto modified_full = quantizer->EstimateSplitDistance(
        query_context, binary.data(), modified_extended.data());

    BOOST_CHECK(std::fabs(full.distance - one_bit.distance) > 1e-5F ||
                std::fabs(full.lowerBound - one_bit.lowerBound) > 1e-5F);
    BOOST_CHECK(std::fabs(full.distance - modified_full.distance) > 1e-5F ||
                std::fabs(full.lowerBound - modified_full.lowerBound) > 1e-5F);
}

BOOST_AUTO_TEST_CASE(OfficialSplitBatchRaBitQMatchesPinnedApisAndHandlesTails)
{
    const auto raw = MakeRawVectors();
    const auto local_centroid = MakeLocalCentroid(-0.3F, 0.004F);
    constexpr std::size_t tail_count = 5;
    std::vector<float> vectors(tail_count * static_cast<std::size_t>(kDimension));
    for (std::size_t i = 0; i < tail_count; ++i) {
        std::memcpy(
            vectors.data() + i * static_cast<std::size_t>(kDimension),
            raw->GetVector(static_cast<SizeType>(17U + i)),
            sizeof(float) * static_cast<std::size_t>(kDimension));
    }

    for (const auto metric : {DistCalcMethod::L2, DistCalcMethod::Cosine}) {
        for (const auto mode : {COMMON::RaBitQQuantizer::QuantizationMode::Exact,
                                COMMON::RaBitQQuantizer::QuantizationMode::Fast}) {
            auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
                kDimension, 5, false, metric, mode);
            BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

            const auto layout = quantizer->GetSplitBatchLayout();
            BOOST_CHECK_EQUAL(layout.batchSize, rabitqlib::fastscan::kBatchSize);
            BOOST_CHECK_EQUAL(
                layout.binaryCodeBytes,
                layout.paddedDimension * layout.batchSize / 8U);
            BOOST_CHECK_EQUAL(
                layout.binaryFactorBytes,
                sizeof(float) * layout.batchSize * 3U);
            BOOST_CHECK_EQUAL(
                layout.binaryBytes,
                rabitqlib::BatchDataMap<float>::data_bytes(layout.paddedDimension));
            BOOST_CHECK_EQUAL(
                layout.extendedCodeBytesPerVector,
                layout.paddedDimension * layout.extendedBits / 8U);
            BOOST_CHECK_EQUAL(
                layout.extendedFactorBytesPerVector,
                layout.extendedBits > 0U ? sizeof(float) * 2U : 0U);
            BOOST_CHECK_EQUAL(
                layout.extendedBytesPerVector,
                rabitqlib::ExDataMap<float>::data_bytes(
                    layout.paddedDimension, layout.extendedBits));
            BOOST_CHECK_EQUAL(
                layout.extendedBytes,
                layout.extendedBytesPerVector * layout.batchSize);
            BOOST_CHECK_EQUAL(
                layout.totalBytes, layout.binaryBytes + layout.extendedBytes);

            std::vector<std::uint8_t> binary(layout.binaryBytes, 0);
            std::vector<std::uint8_t> extended(layout.extendedBytes, 0);
            std::size_t valid_count = 0;
            BOOST_REQUIRE(
                quantizer->QuantizeSplitBatch(
                    vectors.data(),
                    tail_count,
                    local_centroid.data(),
                    binary.data(),
                    extended.data(),
                    valid_count) == ErrorCode::Success);
            BOOST_CHECK_EQUAL(valid_count, tail_count);

            rabitqlib::ConstBatchDataMap<float> batch(
                reinterpret_cast<const char*>(binary.data()), layout.paddedDimension);
            for (std::size_t i = tail_count; i < layout.batchSize; ++i) {
                BOOST_CHECK_EQUAL(batch.f_add()[i], batch.f_add()[tail_count - 1U]);
                BOOST_CHECK_EQUAL(
                    batch.f_rescale()[i], batch.f_rescale()[tail_count - 1U]);
                BOOST_CHECK_EQUAL(batch.f_error()[i], batch.f_error()[tail_count - 1U]);
                BOOST_CHECK_EQUAL(
                    std::memcmp(
                        extended.data() + i * layout.extendedBytesPerVector,
                        extended.data() + (tail_count - 1U) * layout.extendedBytesPerVector,
                        layout.extendedBytesPerVector),
                    0);
            }

            COMMON::RaBitQQuantizer::SplitBatchQueryContext query_context;
            BOOST_REQUIRE(
                quantizer->PrepareSplitBatchQueryContext(
                    reinterpret_cast<const float*>(raw->GetVector(41)),
                    local_centroid.data(),
                    query_context) == ErrorCode::Success);
            BOOST_REQUIRE(query_context.Ready());

            COMMON::RaBitQQuantizer::SplitBatchDistanceEstimates estimates;
            BOOST_REQUIRE(
                quantizer->EstimateSplitBatchDistances(
                    query_context, binary.data(), valid_count, estimates) ==
                ErrorCode::Success);
            BOOST_CHECK_EQUAL(estimates.ValidCount(), tail_count);
            BOOST_CHECK_EQUAL(estimates.lowerBounds.size(), tail_count);
            BOOST_CHECK_EQUAL(estimates.upperBounds.size(), tail_count);
            BOOST_CHECK_EQUAL(estimates.errorBounds.size(), tail_count);
            BOOST_CHECK_EQUAL(estimates.intermediateInnerProducts.size(), tail_count);
            for (std::size_t i = 0; i < tail_count; ++i) {
                BOOST_CHECK(std::isfinite(estimates.distances[i]));
                BOOST_CHECK(std::isfinite(estimates.lowerBounds[i]));
                BOOST_CHECK(std::isfinite(estimates.upperBounds[i]));
                BOOST_CHECK(std::isfinite(estimates.errorBounds[i]));
                BOOST_CHECK(std::isfinite(estimates.intermediateInnerProducts[i]));
                BOOST_CHECK(estimates.lowerBounds[i] <= estimates.distances[i] + 1e-5F);
                BOOST_CHECK(estimates.distances[i] <= estimates.upperBounds[i] + 1e-5F);
            }

            SavedRaBitQModelHeader header{};
            const auto rotator_state = ExtractRotatorState(quantizer, header);
            std::unique_ptr<rabitqlib::Rotator<float>> rotator(
                rabitqlib::choose_rotator<float>(
                    static_cast<std::size_t>(header.dimension),
                    rabitqlib::RotatorType::FhtKacRotator,
                    static_cast<std::size_t>(header.paddedDimension)));
            rotator->load(rotator_state.data());
            const auto rotated_query = PrepareOfficialRotatedInput(
                reinterpret_cast<const float*>(raw->GetVector(41)),
                kDimension,
                layout.paddedDimension,
                *rotator,
                metric == DistCalcMethod::Cosine);
            const auto rotated_centroid = PrepareOfficialRotatedInput(
                local_centroid.data(),
                kDimension,
                layout.paddedDimension,
                *rotator,
                false);
            const auto official_metric =
                metric == DistCalcMethod::L2 ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;
            const bool use_high_accuracy =
                mode == COMMON::RaBitQQuantizer::QuantizationMode::Exact;
            rabitqlib::SplitBatchQuery<float> reference_query(
                rotated_query.data(),
                layout.paddedDimension,
                layout.extendedBits,
                official_metric,
                use_high_accuracy);
            const float norm = std::sqrt(rabitqlib::euclidean_sqr(
                rotated_query.data(), rotated_centroid.data(), layout.paddedDimension));
            const float ip = rabitqlib::dot_product(
                rotated_query.data(), rotated_centroid.data(), layout.paddedDimension);
            reference_query.set_g_add(norm, ip);

            std::array<float, rabitqlib::fastscan::kBatchSize> reference_distances{};
            std::array<float, rabitqlib::fastscan::kBatchSize> reference_lower_bounds{};
            std::array<float, rabitqlib::fastscan::kBatchSize> reference_inner_products{};
            rabitqlib::split_batch_estdist(
                reinterpret_cast<const char*>(binary.data()),
                reference_query,
                layout.paddedDimension,
                reference_distances.data(),
                reference_lower_bounds.data(),
                reference_inner_products.data(),
                use_high_accuracy);
            for (std::size_t i = 0; i < tail_count; ++i) {
                BOOST_CHECK_EQUAL(estimates.distances[i], reference_distances[i]);
                BOOST_CHECK_EQUAL(estimates.lowerBounds[i], reference_lower_bounds[i]);
                BOOST_CHECK_EQUAL(
                    estimates.intermediateInnerProducts[i], reference_inner_products[i]);
            }

            constexpr std::size_t boost_index = 2;
            COMMON::RaBitQQuantizer::SplitDistanceEstimate boosted;
            BOOST_REQUIRE(
                quantizer->BoostSplitBatchDistance(
                    query_context,
                    binary.data(),
                    extended.data(),
                    valid_count,
                    boost_index,
                    boosted) == ErrorCode::Success);
            const float reference_boosted = rabitqlib::split_distance_boosting(
                reinterpret_cast<const char*>(
                    extended.data() + boost_index * layout.extendedBytesPerVector),
                rabitqlib::select_excode_ipfunc(layout.extendedBits),
                reference_query,
                layout.paddedDimension,
                layout.extendedBits,
                reference_inner_products[boost_index]);
            BOOST_CHECK_EQUAL(boosted.distance, reference_boosted);
            BOOST_CHECK_EQUAL(
                boosted.intermediateInnerProduct,
                reference_inner_products[boost_index]);

            const auto split_layout = quantizer->GetSplitCodeLayout();
            std::vector<std::uint8_t> single_binary(split_layout.binaryBytes, 0);
            std::vector<std::uint8_t> single_extended(split_layout.extendedBytes, 0);
            BOOST_REQUIRE(
                quantizer->QuantizeSplitVector(
                    vectors.data() + boost_index * static_cast<std::size_t>(kDimension),
                    local_centroid.data(),
                    single_binary.data(),
                    single_extended.data()) == ErrorCode::Success);
            BOOST_CHECK_EQUAL(
                std::memcmp(
                    single_extended.data(),
                    extended.data() + boost_index * layout.extendedBytesPerVector,
                    layout.extendedBytesPerVector),
                0);

            COMMON::RaBitQQuantizer::SplitQueryContext single_query_context;
            BOOST_REQUIRE(
                quantizer->PrepareSplitQueryContext(
                    raw->GetVector(41),
                    local_centroid.data(),
                    single_query_context) == ErrorCode::Success);
            const auto single_estimate = quantizer->EstimateSplitDistance(
                single_query_context, single_binary.data(), single_extended.data());
            BOOST_CHECK(
                std::fabs(boosted.distance - single_estimate.distance) <=
                boosted.errorBound + single_estimate.errorBound + 1e-3F);

            auto loaded =
                std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(SaveAndLoad(quantizer));
            BOOST_REQUIRE(loaded != nullptr);
            std::vector<std::uint8_t> loaded_binary(layout.binaryBytes, 0);
            std::vector<std::uint8_t> loaded_extended(layout.extendedBytes, 0);
            std::size_t loaded_valid_count = 0;
            BOOST_REQUIRE(
                loaded->QuantizeSplitBatch(
                    vectors.data(),
                    tail_count,
                    local_centroid.data(),
                    loaded_binary.data(),
                    loaded_extended.data(),
                    loaded_valid_count) == ErrorCode::Success);
            BOOST_CHECK_EQUAL(loaded_valid_count, valid_count);
            BOOST_CHECK(binary == loaded_binary);
            BOOST_CHECK(extended == loaded_extended);
            std::remove(kQuantizerFile);
        }
    }
}

BOOST_AUTO_TEST_CASE(OfficialSplitBatchRaBitQSupportsPersistedUnnormalizedInnerProduct)
{
    const auto raw = MakeRawVectors();
    const auto local_centroid = MakeLocalCentroid(0.1F, -0.002F);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        6,
        false,
        DistCalcMethod::InnerProduct,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const auto compact_code = QuantizeSingleVector(quantizer, raw->GetVector(8), false);
    quantizer->SetEnableADC(true);
    const auto compact_query = QuantizeSingleVector(quantizer, raw->GetVector(12), true);
    const auto compact_estimate =
        quantizer->DistanceWithError(compact_query.data(), compact_code.data());
    const float compact_expected = COMMON::DistanceUtils::ComputeCosineDistance(
        reinterpret_cast<const float*>(raw->GetVector(12)),
        reinterpret_cast<const float*>(raw->GetVector(8)),
        kDimension);
    BOOST_CHECK(
        std::fabs(compact_expected - compact_estimate.distance) <=
        compact_estimate.errorBound + 1e-3F);

    constexpr std::size_t count = 3;
    std::vector<float> vectors(count * static_cast<std::size_t>(kDimension));
    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(
            vectors.data() + i * static_cast<std::size_t>(kDimension),
            raw->GetVector(static_cast<SizeType>(8U + i)),
            sizeof(float) * static_cast<std::size_t>(kDimension));
    }

    const auto layout = quantizer->GetSplitBatchLayout();
    std::vector<std::uint8_t> binary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> extended(layout.extendedBytes, 0);
    std::size_t valid_count = 0;
    BOOST_REQUIRE(
        quantizer->QuantizeSplitBatch(
            vectors.data(),
            count,
            local_centroid.data(),
            binary.data(),
            extended.data(),
            valid_count) == ErrorCode::Success);

    COMMON::RaBitQQuantizer::SplitBatchQueryContext query_context;
    const auto* query = reinterpret_cast<const float*>(raw->GetVector(12));
    BOOST_REQUIRE(
        quantizer->PrepareSplitBatchQueryContext(
            query, local_centroid.data(), query_context) == ErrorCode::Success);
    for (std::size_t i = 0; i < count; ++i) {
        COMMON::RaBitQQuantizer::SplitDistanceEstimate estimate;
        BOOST_REQUIRE(
            quantizer->BoostSplitBatchDistance(
                query_context,
                binary.data(),
                extended.data(),
                valid_count,
                i,
                estimate) == ErrorCode::Success);
        const float expected = COMMON::DistanceUtils::ComputeCosineDistance(
            query,
            vectors.data() + i * static_cast<std::size_t>(kDimension),
            kDimension);
        BOOST_CHECK(
            std::fabs(expected - estimate.distance) <= estimate.errorBound + 1e-3F);
    }

    auto loaded = std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(SaveAndLoad(quantizer));
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_CHECK(loaded->GetMetric() == DistCalcMethod::InnerProduct);
    BOOST_CHECK(loaded->GetSplitBatchLayout().extendedBytesPerVector > 0U);
    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQPersistsExactAndFastModes)
{
    const auto raw = MakeRawVectors();

    for (const auto mode : {COMMON::RaBitQQuantizer::QuantizationMode::Exact,
                            COMMON::RaBitQQuantizer::QuantizationMode::Fast}) {
        std::remove(kQuantizerFile);
        auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
            kDimension, 5, false, DistCalcMethod::L2, mode);
        BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
        BOOST_CHECK(quantizer->GetQuantizationMode() == mode);

        auto loaded = std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(SaveAndLoad(quantizer));
        BOOST_REQUIRE(loaded != nullptr);
        BOOST_CHECK(loaded->GetQuantizationMode() == mode);
        BOOST_CHECK(loaded->GetMetric() == DistCalcMethod::L2);

        const auto code = QuantizeSingleVector(loaded, raw->GetVector(9), false);
        loaded->SetEnableADC(true);
        const auto query = QuantizeSingleVector(loaded, raw->GetVector(10), true);
        BOOST_CHECK(std::isfinite(loaded->L2Distance(query.data(), code.data())));
    }

    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQValidationRejectsLoadedModelMismatches)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::Cosine,
        COMMON::RaBitQQuantizer::QuantizationMode::Fast);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    auto matching = std::make_shared<QuantizerOptions>(
        10000,
        true,
        0.0F,
        QuantizerType::RaBitQQuantizer,
        std::string(),
        5,
        std::string(),
        std::string(),
        DistCalcMethod::Cosine,
        "fast");
    matching->m_hasExplicitQuantizedDim = true;
    BOOST_CHECK(ValidateLoadedRaBitQQuantizer(quantizer, matching, kDimension));

    auto bits_mismatch = std::make_shared<QuantizerOptions>(
        10000,
        true,
        0.0F,
        QuantizerType::RaBitQQuantizer,
        std::string(),
        4,
        std::string(),
        std::string(),
        DistCalcMethod::Cosine,
        "fast");
    bits_mismatch->m_hasExplicitQuantizedDim = true;
    BOOST_CHECK(!ValidateLoadedRaBitQQuantizer(quantizer, bits_mismatch, kDimension));

    auto metric_mismatch = std::make_shared<QuantizerOptions>(
        10000,
        true,
        0.0F,
        QuantizerType::RaBitQQuantizer,
        std::string(),
        5,
        std::string(),
        std::string(),
        DistCalcMethod::L2,
        "fast");
    BOOST_CHECK(!ValidateLoadedRaBitQQuantizer(quantizer, metric_mismatch, kDimension));

    auto mode_mismatch = std::make_shared<QuantizerOptions>(
        10000,
        true,
        0.0F,
        QuantizerType::RaBitQQuantizer,
        std::string(),
        5,
        std::string(),
        std::string(),
        DistCalcMethod::Cosine,
        "exact");
    BOOST_CHECK(!ValidateLoadedRaBitQQuantizer(quantizer, mode_mismatch, kDimension));

    BOOST_CHECK(!ValidateLoadedRaBitQQuantizer(quantizer, matching, kDimension - 1));
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQValidationTrustsPersistedBitsWhenQdOmitted)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    auto omitted_qd = std::make_shared<QuantizerOptions>(
        10000,
        true,
        0.0F,
        QuantizerType::RaBitQQuantizer,
        std::string(),
        -1,
        std::string(),
        std::string(),
        DistCalcMethod::L2,
        "exact");
    BOOST_CHECK(!omitted_qd->m_hasExplicitQuantizedDim);
    BOOST_CHECK(ValidateLoadedRaBitQQuantizer(quantizer, omitted_qd, kDimension));
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQLoadedModelIgnoresOuterNormalizationFlag)
{
    const auto raw = MakeRawVectors();
    auto trained = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(trained->Train(raw) == ErrorCode::Success);
    auto loaded = std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(SaveAndLoad(trained));
    BOOST_REQUIRE(loaded != nullptr);

    auto make_options = [](bool normalized) {
        auto options = std::make_shared<QuantizerOptions>(
            10000,
            true,
            0.0F,
            QuantizerType::RaBitQQuantizer,
            std::string(),
            -1,
            std::string(),
            std::string(),
            DistCalcMethod::L2,
            "exact");
        options->m_normalized = normalized;
        return options;
    };

    auto encode_with_main_path =
        [&](const std::shared_ptr<QuantizerOptions>& options) -> std::vector<std::uint8_t> {
        ByteArray bytes = ByteArray::Alloc(sizeof(float) * kDimension);
        std::memcpy(bytes.Data(), raw->GetVector(7), sizeof(float) * kDimension);
        auto set = std::make_shared<BasicVectorSet>(bytes, VectorValueType::Float, kDimension, 1);
        if (ShouldOuterNormalizeForQuantizeAndSave(options, loaded)) {
            set->Normalize(options->m_threadNum);
        }

        loaded->SetEnableADC(false);
        return QuantizeSingleVector(loaded, set->GetVector(0), false);
    };

    const auto raw_code = encode_with_main_path(make_options(false));
    const auto normalized_flag_code = encode_with_main_path(make_options(true));
    BOOST_CHECK(!ShouldOuterNormalizeForQuantizeAndSave(make_options(false), loaded));
    BOOST_CHECK(!ShouldOuterNormalizeForQuantizeAndSave(make_options(true), loaded));
    BOOST_CHECK(raw_code == normalized_flag_code);
    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQCosineUsesOfficialIpEstimator)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        4,
        false,
        DistCalcMethod::Cosine,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
    BOOST_CHECK(quantizer->GetMetric() == DistCalcMethod::Cosine);

    const auto own_code = QuantizeSingleVector(quantizer, raw->GetVector(12), false);
    const auto far_code = QuantizeSingleVector(quantizer, raw->GetVector(kVectorCount - 1), false);
    quantizer->SetEnableADC(true);
    const auto query = QuantizeSingleVector(quantizer, raw->GetVector(12), true);

    const float own_distance = quantizer->CosineDistance(query.data(), own_code.data());
    const float far_distance = quantizer->CosineDistance(query.data(), far_code.data());
    BOOST_CHECK(std::isfinite(own_distance));
    BOOST_CHECK(std::isfinite(far_distance));
    BOOST_CHECK(own_distance <= far_distance);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQDistanceEstimateReturnsErrorBound)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);

    const SizeType base_id = 14;
    const SizeType query_id = 31;
    const auto code = QuantizeSingleVector(quantizer, raw->GetVector(base_id), false);
    quantizer->SetEnableADC(true);
    const auto query = QuantizeSingleVector(quantizer, raw->GetVector(query_id), true);

    const auto estimate = quantizer->DistanceWithError(query.data(), code.data());
    const float distance = quantizer->L2Distance(query.data(), code.data());
    const float expected = ComputeExpectedDistance(
        reinterpret_cast<const float*>(raw->GetVector(query_id)),
        reinterpret_cast<const float*>(raw->GetVector(base_id)),
        DistCalcMethod::L2);

    BOOST_CHECK_EQUAL(estimate.distance, distance);
    BOOST_CHECK(std::isfinite(estimate.distance));
    BOOST_CHECK(std::isfinite(estimate.errorBound));
    BOOST_CHECK(estimate.errorBound >= 0.0F);
    BOOST_CHECK(std::fabs(expected - estimate.distance) <= estimate.errorBound + 1e-3F);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQRejectsLegacyVersion2Models)
{
    std::remove(kQuantizerFile);
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        5,
        false,
        DistCalcMethod::L2,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
    BOOST_REQUIRE(SaveAndLoad(quantizer) != nullptr);

    {
        std::fstream file(kQuantizerFile, std::ios::binary | std::ios::in | std::ios::out);
        BOOST_REQUIRE(file.good());
        const auto version_offset =
            static_cast<std::streamoff>(sizeof(QuantizerType) + sizeof(VectorValueType) +
                                        sizeof(std::uint32_t));
        file.seekp(version_offset);
        const std::uint32_t legacy_version = 2U;
        file.write(reinterpret_cast<const char*>(&legacy_version), sizeof(legacy_version));
        BOOST_REQUIRE(file.good());
    }

    auto input = f_createIO();
    BOOST_REQUIRE(input != nullptr);
    BOOST_REQUIRE(input->Initialize(kQuantizerFile, std::ios::binary | std::ios::in));
    auto loaded = COMMON::IQuantizer::LoadIQuantizer(input);
    input->ShutDown();
    BOOST_CHECK(loaded == nullptr);
    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQUsesGlobalQuantizerPath)
{
    std::remove(kQuantizerFile);
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension, kRaBitQBits, false);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(quantizer->GetNumSubvectors(), kRaBitQCodeBytes);

    const auto codes = QuantizeVectors(raw, quantizer);
    BOOST_CHECK_EQUAL(codes->Dimension(), kRaBitQCodeBytes);
    auto loaded = std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(SaveAndLoad(quantizer));
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_CHECK_EQUAL(loaded->GetNumSubvectors(), codes->Dimension());
    BOOST_CHECK(loaded->GetMetric() == DistCalcMethod::L2);
    BOOST_CHECK(
        loaded->GetQuantizationMode() == COMMON::RaBitQQuantizer::QuantizationMode::Exact);

    loaded->SetEnableADC(true);
    COMMON::QueryResultSet<float> query(
        reinterpret_cast<const float*>(raw->GetVector(37)), 1);
    query.SetTarget(reinterpret_cast<const float*>(raw->GetVector(37)), loaded);
    const auto* query_code = reinterpret_cast<const std::uint8_t*>(query.GetQuantizedTarget());
    const auto* own_code = reinterpret_cast<const std::uint8_t*>(codes->GetVector(37));
    const auto* far_code = reinterpret_cast<const std::uint8_t*>(codes->GetVector(kVectorCount - 1));
    BOOST_CHECK(loaded->L2Distance(query_code, own_code) <
                loaded->L2Distance(query_code, far_code));

    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQHandlesCentroidVector)
{
    ByteArray bytes = ByteArray::Alloc(sizeof(float) * kDimension);
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (DimensionType dim = 0; dim < kDimension; ++dim) {
        values[dim] = static_cast<float>(dim) * 0.01F;
    }
    const auto vectors = std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, kDimension, 1);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension, kRaBitQBits, false);
    BOOST_REQUIRE(quantizer->Train(vectors) == ErrorCode::Success);

    std::vector<std::uint8_t> code(quantizer->GetNumSubvectors());
    quantizer->QuantizeVector(values, code.data(), false);
    std::vector<float> reconstructed(kDimension);
    quantizer->ReconstructVector(code.data(), reconstructed.data());
    for (float value : reconstructed) {
        BOOST_CHECK(std::isfinite(value));
    }

    quantizer->SetEnableADC(true);
    std::vector<std::uint8_t> query(quantizer->QuantizeSize());
    quantizer->QuantizeVector(values, query.data());
    BOOST_CHECK(std::isfinite(quantizer->L2Distance(query.data(), code.data())));
}

BOOST_AUTO_TEST_CASE(OfficialCompactRaBitQStoresRequestedBits)
{
    const auto raw = MakeRawVectors();
    for (int bits = 1; bits <= 8; ++bits) {
        auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
            kDimension, bits, false);
        BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
        BOOST_CHECK_EQUAL(
            quantizer->GetNumSubvectors(),
            kDimension * bits / 8 + 5 * sizeof(float));

        std::vector<std::uint8_t> code(quantizer->GetNumSubvectors());
        quantizer->QuantizeVector(raw->GetVector(0), code.data(), false);
        std::vector<float> reconstructed(kDimension);
        quantizer->ReconstructVector(code.data(), reconstructed.data());
        for (float value : reconstructed) {
            BOOST_CHECK(std::isfinite(value));
        }

        quantizer->SetEnableADC(true);
        std::vector<std::uint8_t> query(quantizer->QuantizeSize());
        quantizer->QuantizeVector(raw->GetVector(1), query.data());
        BOOST_CHECK(std::isfinite(quantizer->L2Distance(query.data(), code.data())));
    }
}

BOOST_AUTO_TEST_SUITE_END()
