// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"

#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/RaBitQAutoTuner.h"
#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/SSDServing/SSDIndex.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
constexpr const char* kQueryFile = "rabitq_global_query_test.fvecs";
constexpr SizeType kSearchQueryCount = 16;

std::shared_ptr<VectorSet> MakeRawVectors()
{
    ByteArray bytes = ByteArray::Alloc(
        sizeof(float) * static_cast<std::size_t>(kVectorCount) * kDimension);
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (SizeType vector = 0; vector < kVectorCount; ++vector) {
        for (DimensionType dim = 0; dim < kDimension; ++dim) {
            values[static_cast<std::size_t>(vector) * kDimension + dim] =
                static_cast<float>(vector) * 0.125F + static_cast<float>(dim) * 0.01F;
        }
    }
    return std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, kDimension, kVectorCount);
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

void VerifySearch(
    IndexAlgoType p_algorithm,
    const std::shared_ptr<VectorSet>& p_raw,
    const std::shared_ptr<VectorSet>& p_codes,
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer)
{
    p_quantizer->SetEnableADC(false);
    auto index = VectorIndex::CreateInstance(p_algorithm, VectorValueType::UInt8);
    BOOST_REQUIRE(index != nullptr);
    index->SetQuantizer(p_quantizer);
    index->SetParameter("DistCalcMethod", "L2");
    index->SetParameter("NumberOfThreads", "1");
    index->SetParameter("MaxCheck", "4096");
    if (p_algorithm == IndexAlgoType::KDT) {
        index->SetParameter("KDTNumber", "1");
    }
    BOOST_REQUIRE(index->BuildIndex(p_codes, nullptr, false, true) == ErrorCode::Success);

    p_quantizer->SetEnableADC(true);
    const SizeType expected = 37;
    COMMON::QueryResultSet<float> query(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), 8);
    BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);

    bool found = false;
    for (int rank = 0; rank < query.GetResultNum(); ++rank) {
        const auto* result = query.GetResult(rank);
        if (result != nullptr && result->VID == expected) {
            found = true;
            break;
        }
    }
    BOOST_CHECK(found);

    COMMON::QueryResultSet<float> direct_query(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), 1);
    direct_query.SetTarget(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), p_quantizer);
    const auto* query_code = reinterpret_cast<const std::uint8_t*>(direct_query.GetQuantizedTarget());
    const auto* own_code = reinterpret_cast<const std::uint8_t*>(p_codes->GetVector(expected));
    const auto* far_code = reinterpret_cast<const std::uint8_t*>(p_codes->GetVector(kVectorCount - 1));
    BOOST_CHECK(index->ComputeDistance(query_code, own_code) <
                index->ComputeDistance(query_code, far_code));
}

void ConfigureSpannIndex(const std::shared_ptr<VectorIndex>& p_index,
                         const std::string& p_index_directory,
                         const char* p_query_file,
                         const char* p_storage,
                         bool p_enable_compression)
{
    p_index->SetParameter("IndexDirectory", p_index_directory, "Base");
    p_index->SetParameter("IndexAlgoType", "BKT", "Base");
    p_index->SetParameter("DistCalcMethod", "L2", "Base");
    if (p_query_file != nullptr) {
        p_index->SetParameter("QueryPath", p_query_file, "Base");
        p_index->SetParameter("QueryType", "XVEC", "Base");
        p_index->SetParameter("WarmupPath", "", "Base");
    }

    p_index->SetParameter("isExecute", "true", "SelectHead");
    p_index->SetParameter("NumberOfThreads", "1", "SelectHead");
    p_index->SetParameter("Ratio", "0.25", "SelectHead");
    p_index->SetParameter("BKTKmeansK", "4", "SelectHead");
    p_index->SetParameter("BKTLeafSize", "2", "SelectHead");
    p_index->SetParameter("SamplesNumber", "16", "SelectHead");

    p_index->SetParameter("isExecute", "true", "BuildHead");
    p_index->SetParameter("NumberOfThreads", "1", "BuildHead");
    p_index->SetParameter("NeighborhoodSize", "8", "BuildHead");
    p_index->SetParameter("TPTNumber", "1", "BuildHead");
    p_index->SetParameter("TPTLeafSize", "64", "BuildHead");
    p_index->SetParameter("MaxCheck", "256", "BuildHead");
    p_index->SetParameter("MaxCheckForRefineGraph", "256", "BuildHead");
    p_index->SetParameter("RefineIterations", "1", "BuildHead");

    p_index->SetParameter("isExecute", "true", "BuildSSDIndex");
    p_index->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    p_index->SetParameter("Storage", p_storage, "BuildSSDIndex");
    p_index->SetParameter("NumberOfThreads", "1", "BuildSSDIndex");
    p_index->SetParameter("PostingPageLimit", "1", "BuildSSDIndex");
    p_index->SetParameter("SearchPostingPageLimit", "1", "BuildSSDIndex");
    p_index->SetParameter("InternalResultNum", "96", "BuildSSDIndex");
    p_index->SetParameter("SearchInternalResultNum", "96", "BuildSSDIndex");
    p_index->SetParameter("ReplicaCount", "1", "BuildSSDIndex");
    p_index->SetParameter("StartFileSizeGB", "1", "BuildSSDIndex");
    p_index->SetParameter("MaxFileSizeGB", "1", "BuildSSDIndex");
    p_index->SetParameter("EnableDeltaEncoding", "false", "BuildSSDIndex");
    p_index->SetParameter("EnablePostingListRearrange", "false", "BuildSSDIndex");
    p_index->SetParameter("EnableDataCompression", p_enable_compression ? "true" : "false", "BuildSSDIndex");
    p_index->SetParameter("EnableDictTraining", "false", "BuildSSDIndex");
    p_index->SetParameter("AsyncMergeInSearch", "false", "BuildSSDIndex");
    p_index->SetParameter("EnableADC", "true", "BuildSSDIndex");
}

void VerifySpannSearch(
    const std::shared_ptr<VectorSet>& p_raw,
    const std::shared_ptr<VectorSet>& p_codes,
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer,
    const char* p_storage,
    bool p_enable_compression = false)
{
    const std::string index_directory = std::string("rabitq_global_spann_test_") + p_storage +
        (p_enable_compression ? "_compressed" : "");
    std::filesystem::remove_all(index_directory);

    p_quantizer->SetEnableADC(false);
    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);
    index->SetQuantizer(p_quantizer);
    ConfigureSpannIndex(index, index_directory, nullptr, p_storage, p_enable_compression);
    BOOST_REQUIRE(index->BuildIndex(p_raw, nullptr, false, true) == ErrorCode::Success);

    p_quantizer->SetEnableADC(true);
    auto* spann_index = static_cast<SPANN::Index<float>*>(index.get());
    std::vector<SizeType> head_ids;
    BOOST_REQUIRE(spann_index->GetHeadIndexMapping(1, head_ids) == ErrorCode::Success);
    SizeType expected = 0;
    while (expected < kVectorCount &&
           std::find(head_ids.begin(), head_ids.end(), expected) != head_ids.end()) {
        ++expected;
    }
    BOOST_REQUIRE(expected < kVectorCount);
    COMMON::QueryResultSet<float> query(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), 96);
    BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
    for (int rank = 0; rank < query.GetResultNum(); ++rank) {
        const auto* result = query.GetResult(rank);
        if (result != nullptr && result->VID != -1) {
            BOOST_CHECK(std::isfinite(result->Dist));
        }
    }

    COMMON::QueryResultSet<float> direct_query(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), 1);
    direct_query.SetTarget(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), p_quantizer);
    const auto* query_code = reinterpret_cast<const std::uint8_t*>(
        direct_query.GetQuantizedTarget());
    const auto* own_code = reinterpret_cast<const std::uint8_t*>(
        p_codes->GetVector(expected));
    const auto* far_code = reinterpret_cast<const std::uint8_t*>(
        p_codes->GetVector(kVectorCount - 1));
    BOOST_CHECK(p_quantizer->L2Distance(query_code, own_code) <
                p_quantizer->L2Distance(query_code, far_code));

    index.reset();
    std::filesystem::remove_all(index_directory);
}

void VerifySSDServingSearch(
    const std::shared_ptr<VectorSet>& p_raw,
    const std::shared_ptr<VectorSet>& p_codes,
    const std::shared_ptr<COMMON::IQuantizer>& p_quantizer)
{
    const std::string index_directory = "rabitq_global_ssdserving_test";
    std::filesystem::remove_all(index_directory);
    {
        std::ofstream query_file(kQueryFile, std::ios::binary | std::ios::trunc);
        BOOST_REQUIRE(query_file.good());
        const std::int32_t dimension = kDimension;
        for (SizeType query = 0; query < kSearchQueryCount; ++query) {
            query_file.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
            query_file.write(reinterpret_cast<const char*>(p_raw->GetVector(37)),
                             sizeof(float) * kDimension);
        }
        BOOST_REQUIRE(query_file.good());
    }

    p_quantizer->SetEnableADC(false);
    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);
    index->SetQuantizer(p_quantizer);
    ConfigureSpannIndex(index, index_directory, kQueryFile, "FILEIO", false);
    index->SetParameter("EnableADC", "true", "BuildSSDIndex");
    index->SetParameter("SearchThreadNum", "1", "BuildSSDIndex");
    index->SetParameter("SearchInternalResultNum", "96", "SearchSSDIndex");
    index->SetParameter("ResultNum", "8", "SearchSSDIndex");
    index->SetParameter("QueryCountLimit", std::to_string(kSearchQueryCount), "SearchSSDIndex");
    BOOST_REQUIRE(index->BuildIndex(p_raw, nullptr, false, true) == ErrorCode::Success);

    auto* spann_index = static_cast<SPANN::Index<float>*>(index.get());
    BOOST_REQUIRE(SSDServing::SSDIndex::Search(spann_index) == ErrorCode::Success);

    index.reset();
    std::remove(kQueryFile);
    std::filesystem::remove_all(index_directory);
}

} // namespace

BOOST_AUTO_TEST_SUITE(RaBitQQuantizerTest)

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
    const auto loaded = SaveAndLoad(quantizer);
    BOOST_CHECK_EQUAL(loaded->GetNumSubvectors(), codes->Dimension());

    VerifySearch(IndexAlgoType::BKT, raw, codes, loaded);
    VerifySearch(IndexAlgoType::KDT, raw, codes, loaded);
    VerifySpannSearch(raw, codes, loaded, "FILEIO");
    VerifySpannSearch(raw, codes, loaded, "STATIC");
    VerifySpannSearch(raw, codes, loaded, "STATIC", true);
    VerifySSDServingSearch(raw, codes, loaded);

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

BOOST_AUTO_TEST_CASE(SpannAppliesConfiguredADCWhenAttachingQuantizer)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension, kRaBitQBits, false);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
    BOOST_CHECK(!quantizer->GetEnableADC());

    auto index = VectorIndex::CreateInstance(
        IndexAlgoType::SPANN, VectorValueType::UInt8);
    BOOST_REQUIRE(index != nullptr);
    index->SetParameter("EnableADC", "true", "BuildSSDIndex");
    index->SetQuantizer(quantizer);
    BOOST_CHECK(quantizer->GetEnableADC());
}

BOOST_AUTO_TEST_CASE(RaBitQAutoTuneSelectsFirstQualifyingBit)
{
    std::vector<int> evaluated;
    int selected = 0;
    float recall = 0.0F;
    BOOST_REQUIRE(
        COMMON::RaBitQAutoTuner::SelectMinimumBits(
            0.75F,
            [&](int bits, float& value) {
                evaluated.push_back(bits);
                value = bits * 0.2F;
                return ErrorCode::Success;
            },
            selected, recall) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(selected, 4);
    BOOST_CHECK_CLOSE(recall, 0.8F, 0.001F);
    const std::vector<int> expectedEvaluated = {1, 2, 3, 4};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        evaluated.begin(), evaluated.end(),
        expectedEvaluated.begin(), expectedEvaluated.end());

    BOOST_CHECK(
        COMMON::RaBitQAutoTuner::SelectMinimumBits(
            1.0F,
            [](int, float& value) {
                value = 0.99F;
                return ErrorCode::Success;
            },
            selected, recall) == ErrorCode::Fail);
    BOOST_CHECK_EQUAL(selected, 0);
}

BOOST_AUTO_TEST_CASE(RaBitQAutoTuneUsesDeeperTruthPool)
{
    const std::vector<std::vector<SizeType>> truth = {
        {10, 11, 12}, {20, 21, 22}};
    std::string error;
    BOOST_CHECK(
        COMMON::RaBitQAutoTuner::ValidateTruth(
            truth, 32, 2, 2, error) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(
        COMMON::RaBitQAutoTuner::RecallAtK(truth[0], {12, 10}, 2), 0.5F);

    BOOST_CHECK(
        COMMON::RaBitQAutoTuner::ValidateTruth(
            {{0, 1}, {2, 3}}, 4, 2, 2, error) != ErrorCode::Success);
    BOOST_CHECK(error.find("greater than ResultNum") != std::string::npos);
    BOOST_CHECK(
        COMMON::RaBitQAutoTuner::ValidateTruth(
            {{0, 1, 4}, {1, 2, 3}}, 4, 2, 2, error) != ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(RaBitQStreamingCentroidUsesEveryVector)
{
    const auto raw = MakeRawVectors();
    auto oneShot = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension, kRaBitQBits, false);
    BOOST_REQUIRE(oneShot->Train(raw) == ErrorCode::Success);

    auto streamed = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension, 1, false);
    BOOST_REQUIRE(streamed->BeginTraining() == ErrorCode::Success);
    const SizeType boundaries[] = {0, 7, 41, kVectorCount};
    for (std::size_t batch = 0; batch + 1 < std::size(boundaries); ++batch) {
        ByteArray bytes = ByteArray::Alloc(
            sizeof(float) * static_cast<std::size_t>(boundaries[batch + 1] - boundaries[batch]) *
            kDimension);
        std::memcpy(
            bytes.Data(), raw->GetVector(boundaries[batch]),
            bytes.Length());
        auto batchVectors = std::make_shared<BasicVectorSet>(
            bytes, VectorValueType::Float, kDimension,
            boundaries[batch + 1] - boundaries[batch]);
        BOOST_REQUIRE(streamed->AddTrainingBatch(batchVectors) == ErrorCode::Success);
    }
    BOOST_REQUIRE(streamed->FinishTraining() == ErrorCode::Success);
    const auto sharedCentroid = streamed->CreateWithBits(kRaBitQBits);
    BOOST_REQUIRE(sharedCentroid != nullptr);

    std::vector<std::uint8_t> expected(oneShot->GetNumSubvectors());
    std::vector<std::uint8_t> actual(sharedCentroid->GetNumSubvectors());
    oneShot->QuantizeVector(raw->GetVector(kVectorCount - 1), expected.data(), false);
    sharedCentroid->QuantizeVector(raw->GetVector(kVectorCount - 1), actual.data(), false);
    BOOST_CHECK_EQUAL_COLLECTIONS(
        expected.begin(), expected.end(), actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(RaBitQEncodedWidthMatchesSavedModel)
{
    constexpr DimensionType dimension = 70;
    constexpr int bits = 5;
    ByteArray bytes = ByteArray::Alloc(sizeof(float) * dimension * 2);
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (DimensionType i = 0; i < dimension * 2; ++i) {
        values[i] = static_cast<float>(i) / 13.0F;
    }
    auto vectors = std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, dimension, 2);
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        dimension, bits, false);
    BOOST_REQUIRE(quantizer->Train(vectors) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(
        quantizer->GetNumSubvectors(), 128 * bits / 8 + 5 * sizeof(float));

    const char* modelPath = "rabitq_width_model.bin";
    auto output = f_createIO();
    BOOST_REQUIRE(output->Initialize(modelPath, std::ios::out | std::ios::binary));
    BOOST_REQUIRE(quantizer->SaveQuantizer(output) == ErrorCode::Success);
    output->ShutDown();
    auto input = f_createIO();
    BOOST_REQUIRE(input->Initialize(modelPath, std::ios::in | std::ios::binary));
    auto loaded = COMMON::IQuantizer::LoadIQuantizer(input);
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_CHECK_EQUAL(loaded->GetNumSubvectors(), quantizer->GetNumSubvectors());
    std::remove(modelPath);
}

BOOST_AUTO_TEST_CASE(RaBitQAutoTuneProducesNativeBuildHandoff)
{
    constexpr SizeType vectorCount = 6;
    constexpr SizeType queryCount = 2;
    constexpr DimensionType dimension = 8;
    const char* basePath = "rabitq_auto_base.bin";
    const char* queryPath = "rabitq_auto_queries.bin";
    const char* truthPath = "rabitq_auto_truth.bin";
    const char* outputFolder = "rabitq_auto_handoff";
    std::filesystem::remove_all(outputFolder);

    auto writeVectors = [](const char* path, SizeType count, DimensionType dim, float offset) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        output.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        for (SizeType row = 0; row < count; ++row) {
            for (DimensionType column = 0; column < dim; ++column) {
                const float value = offset + row * 0.5F + column * 0.01F;
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
        BOOST_REQUIRE(output.good());
    };
    writeVectors(basePath, vectorCount, dimension, 0.0F);
    writeVectors(queryPath, queryCount, dimension, 0.2F);
    {
        std::ofstream truth(truthPath, std::ios::binary | std::ios::trunc);
        const DimensionType depth = 3;
        const std::int32_t truthQueryCount = queryCount;
        truth.write(
            reinterpret_cast<const char*>(&truthQueryCount), sizeof(truthQueryCount));
        truth.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
        const std::int32_t ids[] = {0, 1, 2, 1, 0, 2};
        truth.write(reinterpret_cast<const char*>(ids), sizeof(ids));
        const float distances[] = {0.0F, 1.0F, 2.0F, 0.0F, 1.0F, 2.0F};
        truth.write(reinterpret_cast<const char*>(distances), sizeof(distances));
        BOOST_REQUIRE(truth.good());
    }

    Helper::IniReader config;
    config.SetParameter("Base", "ValueType", "Float");
    config.SetParameter("Base", "DistCalcMethod", "L2");
    config.SetParameter("Base", "Dim", std::to_string(dimension));
    config.SetParameter("Base", "VectorPath", basePath);
    config.SetParameter("Base", "VectorType", "DEFAULT");
    config.SetParameter("Base", "QueryPath", queryPath);
    config.SetParameter("Base", "QueryType", "DEFAULT");
    config.SetParameter("Base", "TruthPath", truthPath);
    config.SetParameter("Base", "TruthType", "DEFAULT");
    config.SetParameter("SearchSSDIndex", "QueryCountLimit", std::to_string(queryCount));
    config.SetParameter("SearchSSDIndex", "ResultNum", "1");
    config.SetParameter("BuildSSDIndex", "NumberOfThreads", "2");
    config.SetParameter("RaBitQAutoTune", "isExecute", "true");
    config.SetParameter("RaBitQAutoTune", "TargetRecall", "0");

    COMMON::RaBitQAutoTuneResult result;
    std::string error;
    BOOST_REQUIRE_MESSAGE(
        COMMON::RaBitQAutoTuner::Run(
            config, outputFolder, result, error) == ErrorCode::Success,
        error);
    BOOST_CHECK_EQUAL(result.selectedBits, 1);
    BOOST_CHECK_EQUAL(result.vectorCount, vectorCount);
    BOOST_REQUIRE(result.quantizer != nullptr);
    BOOST_CHECK_EQUAL(
        result.codeDimension, result.quantizer->GetNumSubvectors());

    auto readerOptions = std::make_shared<Helper::ReaderOptions>(
        VectorValueType::UInt8, result.codeDimension, VectorFileType::DEFAULT);
    auto encodedReader = Helper::VectorSetReader::CreateInstance(readerOptions);
    BOOST_REQUIRE(encodedReader->LoadFile(result.vectorPath) == ErrorCode::Success);
    const auto encoded = encodedReader->GetVectorSet();
    BOOST_CHECK_EQUAL(encoded->Count(), vectorCount);
    BOOST_CHECK_EQUAL(encoded->Dimension(), result.quantizer->GetNumSubvectors());

    auto modelInput = f_createIO();
    BOOST_REQUIRE(modelInput->Initialize(
        result.quantizerPath.c_str(), std::ios::in | std::ios::binary));
    auto loaded = COMMON::IQuantizer::LoadIQuantizer(modelInput);
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_CHECK_EQUAL(loaded->GetNumSubvectors(), encoded->Dimension());

    std::remove(basePath);
    std::remove(queryPath);
    std::remove(truthPath);
    std::filesystem::remove_all(outputFolder);
}

BOOST_AUTO_TEST_SUITE_END()
