// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"

#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/PQQuantizer.h"
#include "inc/Core/Common/RaBitQAutoTuner.h"
#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/SSDServing/SSDIndex.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
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

class CheckedADCQuantizer : public COMMON::RaBitQQuantizer
{
public:
    CheckedADCQuantizer() : COMMON::RaBitQQuantizer(kDimension, 7, false) {}

    void QuantizeVector(const void* input, std::uint8_t* output, bool adc = true) const override
    {
        COMMON::RaBitQQuantizer::QuantizeVector(input, output, adc);
        if (adc && GetEnableADC()) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queries.insert(output);
        }
    }

    float L2Distance(const std::uint8_t* x, const std::uint8_t* y) const override
    {
        if (GetEnableADC()) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_queries.count(x) == 0) {
                if (invalidADCInputs.fetch_add(1) == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "Test detected a stored code passed as an ADC query; evaluating with the SDC reference.\n");
                }
                return reference->L2Distance(x, y);
            }
        }
        return COMMON::RaBitQQuantizer::L2Distance(x, y);
    }

    std::shared_ptr<COMMON::RaBitQQuantizer> reference;
    mutable std::atomic<std::size_t> invalidADCInputs{0};

private:
    mutable std::mutex m_mutex;
    mutable std::set<const std::uint8_t*> m_queries;
};

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
    p_index->SetParameter("EnableADC", "false", "BuildSSDIndex");
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
    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::UInt8);
    BOOST_REQUIRE(index != nullptr);
    index->SetQuantizer(p_quantizer);
    ConfigureSpannIndex(index, index_directory, nullptr, p_storage, p_enable_compression);
    BOOST_REQUIRE(index->BuildIndex(p_codes, nullptr, false, true) == ErrorCode::Success);

    p_quantizer->SetEnableADC(true);
    auto* spann_index = static_cast<SPANN::Index<uint8_t>*>(index.get());
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
    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::UInt8);
    BOOST_REQUIRE(index != nullptr);
    index->SetQuantizer(p_quantizer);
    ConfigureSpannIndex(index, index_directory, kQueryFile, "FILEIO", false);
    index->SetParameter("EnableADC", "false", "BuildSSDIndex");
    index->SetParameter("SearchThreadNum", "1", "BuildSSDIndex");
    index->SetParameter("SearchInternalResultNum", "96", "SearchSSDIndex");
    index->SetParameter("ResultNum", "8", "SearchSSDIndex");
    index->SetParameter("QueryCountLimit", std::to_string(kSearchQueryCount), "SearchSSDIndex");
    BOOST_REQUIRE(index->BuildIndex(p_codes, nullptr, false, true) == ErrorCode::Success);

    auto* spann_index = static_cast<SPANN::Index<uint8_t>*>(index.get());
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
    VerifySpannSearch(raw, codes, loaded, "STATIC");
    VerifySpannSearch(raw, codes, loaded, "FILEIO");
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
    for (DimensionType dim = 0; dim < kDimension; ++dim) {
        BOOST_CHECK_SMALL(reconstructed[dim] - values[dim], 0.0001F);
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
    index->SetQuantizer(quantizer);
    index->SetParameter("EnableADC", "true", "BuildSSDIndex");
    BOOST_CHECK(quantizer->GetEnableADC());
}

BOOST_AUTO_TEST_CASE(RaBitQAutoTuneSelectsFirstQualifyingBit)
{
    std::vector<int> evaluated;
    int selected = 0;
    float recall = 0.0F;
    BOOST_REQUIRE(
        COMMON::RaBitQAutoTuner<float>::SelectMinimumBits(
            0.75F,
            [&](int bits, float& value) {
                evaluated.push_back(bits);
                value = bits * 0.2F;
                return ErrorCode::Success;
            },
            selected, recall) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(selected, 4);
    BOOST_CHECK_CLOSE(recall, 0.8F, 0.001F);
    const std::vector<int> expectedEvaluated = {4, 2, 3};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        evaluated.begin(), evaluated.end(),
        expectedEvaluated.begin(), expectedEvaluated.end());

    BOOST_CHECK(
        COMMON::RaBitQAutoTuner<float>::SelectMinimumBits(
            1.0F,
            [](int, float& value) {
                value = 0.99F;
                return ErrorCode::Success;
            },
            selected, recall) == ErrorCode::Fail);
    BOOST_CHECK_EQUAL(selected, 0);
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

BOOST_AUTO_TEST_CASE(RaBitQRotationPreservesGeometryAndPersistence)
{
    for (DimensionType dimension : {3, 70, 128}) {
        for (bool normalize : {false, true}) {
            ByteArray bytes = ByteArray::Alloc(sizeof(float) * dimension * 2);
            auto* values = reinterpret_cast<float*>(bytes.Data());
            for (DimensionType i = 0; i < dimension * 2; ++i)
                values[i] = std::sin(static_cast<float>(i + 1)) + static_cast<float>(i % 7);
            auto vectors = std::make_shared<BasicVectorSet>(
                bytes, VectorValueType::Float, dimension, 2);
            auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(dimension, 3, normalize);
            BOOST_REQUIRE(quantizer->Train(vectors) == ErrorCode::Success);
            const int padded = (dimension + 63) / 64 * 64;
            BOOST_CHECK_EQUAL(quantizer->BufferSize(),
                sizeof(QuantizerType) + sizeof(VectorValueType) + 24 +
                sizeof(float) * (padded + dimension * padded));
            quantizer->SetEnableADC(true);
            std::vector<std::uint8_t> first(quantizer->QuantizeSize()), second(first.size());
            quantizer->QuantizeVector(values, first.data());
            quantizer->QuantizeVector(values + dimension, second.data());
            float originalDistance = 0, rotatedDistance = 0, firstNorm = 0, secondNorm = 0;
            for (int d = 0; d < dimension; ++d) {
                firstNorm += values[d] * values[d];
                secondNorm += values[dimension + d] * values[dimension + d];
            }
            for (int d = 0; d < dimension; ++d) {
                const float difference = values[d] / (normalize ? std::sqrt(firstNorm) : 1.0F) -
                    values[dimension + d] / (normalize ? std::sqrt(secondNorm) : 1.0F);
                originalDistance += difference * difference;
            }
            for (int d = 0; d < padded; ++d) {
                float a, b;
                std::memcpy(&a, first.data() + d * sizeof(float), sizeof(float));
                std::memcpy(&b, second.data() + d * sizeof(float), sizeof(float));
                rotatedDistance += (a - b) * (a - b);
            }
            BOOST_CHECK_CLOSE(rotatedDistance, originalDistance, 0.001F);
            BOOST_CHECK(std::memcmp(first.data(), values, dimension * sizeof(float)) != 0);

            auto diskLoaded = SaveAndLoad(quantizer);
            ByteArray saved = ByteArray::Alloc(quantizer->BufferSize());
            std::ifstream file(kQuantizerFile, std::ios::binary);
            file.read(reinterpret_cast<char*>(saved.Data()), saved.Length());
            BOOST_REQUIRE(file.good());
            auto memoryLoaded = COMMON::IQuantizer::LoadIQuantizer(saved);
            BOOST_REQUIRE(memoryLoaded != nullptr);
            auto clone = quantizer->CloneWithBits(4);
            BOOST_REQUIRE(clone != nullptr);
            for (auto loaded : {diskLoaded, memoryLoaded, std::static_pointer_cast<COMMON::IQuantizer>(clone)}) {
                loaded->SetEnableADC(true);
                std::vector<std::uint8_t> query(loaded->QuantizeSize());
                loaded->QuantizeVector(values, query.data());
                BOOST_CHECK_EQUAL_COLLECTIONS(first.begin(), first.end(), query.begin(), query.end());
            }
            std::vector<std::uint8_t> code(quantizer->GetNumSubvectors()), loadedCode(code.size());
            quantizer->QuantizeVector(values, code.data(), false);
            diskLoaded->QuantizeVector(values, loadedCode.data(), false);
            BOOST_CHECK_EQUAL_COLLECTIONS(code.begin(), code.end(), loadedCode.begin(), loadedCode.end());
            std::vector<float> reconstructed(dimension), loadedReconstruction(dimension);
            quantizer->ReconstructVector(code.data(), reconstructed.data());
            memoryLoaded->ReconstructVector(code.data(), loadedReconstruction.data());
            BOOST_CHECK_EQUAL_COLLECTIONS(reconstructed.begin(), reconstructed.end(),
                                         loadedReconstruction.begin(), loadedReconstruction.end());
            file.close();
            std::remove(kQuantizerFile);
        }
    }
}

BOOST_AUTO_TEST_CASE(RaBitQLegacyModelKeepsIdentityTransform)
{
    const std::uint32_t header[] = {0x32464252U, 2, 128, 128, 3, 0};
    std::vector<std::uint8_t> payload(sizeof(header) + 128 * sizeof(float), 0);
    std::memcpy(payload.data(), header, sizeof(header));
    auto legacy = std::make_shared<COMMON::RaBitQQuantizer>();
    BOOST_REQUIRE(legacy->LoadQuantizer(payload.data()) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(legacy->BufferSize(),
                      sizeof(QuantizerType) + sizeof(VectorValueType) + payload.size());
    const auto raw = MakeRawVectors();
    legacy->SetEnableADC(true);
    std::vector<std::uint8_t> query(legacy->QuantizeSize());
    legacy->QuantizeVector(raw->GetVector(0), query.data());
    BOOST_CHECK(std::memcmp(query.data(), raw->GetVector(0), 128 * sizeof(float)) == 0);
    auto loaded = SaveAndLoad(legacy);
    loaded->SetEnableADC(true);
    std::vector<std::uint8_t> loadedQuery(loaded->QuantizeSize());
    loaded->QuantizeVector(raw->GetVector(0), loadedQuery.data());
    BOOST_CHECK_EQUAL_COLLECTIONS(query.begin(), query.end(), loadedQuery.begin(), loadedQuery.end());
    BOOST_CHECK_EQUAL(loaded->BufferSize(), legacy->BufferSize());
    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(RaBitQLocalCentroidsPreserveADCSDCAndPersistence)
{
    constexpr SizeType count = 32, centerCount = 4;
    for (DimensionType dimension : {70, 128}) {
        for (int bits : {1, 3, 7, 8}) {
            ByteArray bytes = ByteArray::Alloc(count * dimension * sizeof(float));
            auto* values = reinterpret_cast<float*>(bytes.Data());
            for (int i = 0; i < count; ++i)
                for (int d = 0; d < dimension; ++d)
                    values[i * dimension + d] = 8.0F * (i / 8) +
                        std::sin(static_cast<float>((i + 1) * (d + 1)));
            auto raw = std::make_shared<BasicVectorSet>(bytes, VectorValueType::Float, dimension, count);
            ByteArray centerBytes = ByteArray::Alloc(centerCount * dimension * sizeof(float));
            for (int i = 0; i < centerCount; ++i)
                std::memcpy(centerBytes.Data() + i * dimension * sizeof(float), raw->GetVector(i * 8),
                            dimension * sizeof(float));
            auto centers = std::make_shared<BasicVectorSet>(
                centerBytes, VectorValueType::Float, dimension, centerCount);
            auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(dimension, bits, false);
            BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
            const auto globalSize = quantizer->BufferSize();
            BOOST_REQUIRE(quantizer->SetLocalCentroids(centers) == ErrorCode::Success);
            BOOST_CHECK_EQUAL(quantizer->LocalCentroidCount(), centerCount);
            BOOST_CHECK_EQUAL(quantizer->GetNumSubvectors(), 128 * bits / 8 + 24);
            BOOST_CHECK_EQUAL(quantizer->BufferSize(), globalSize + sizeof(std::uint32_t) +
                              centerCount * 128 * sizeof(float));
            auto codes = QuantizeVectors(raw, quantizer);
            auto loaded = SaveAndLoad(quantizer);
            ByteArray saved = ByteArray::Alloc(quantizer->BufferSize());
            std::ifstream file(kQuantizerFile, std::ios::binary);
            file.read(reinterpret_cast<char*>(saved.Data()), saved.Length());
            BOOST_REQUIRE(file.good());
            file.close();
            auto memoryLoaded = COMMON::IQuantizer::LoadIQuantizer(saved);
            BOOST_REQUIRE(memoryLoaded != nullptr);
            auto clone = quantizer->CloneWithBits(bits);
            BOOST_REQUIRE(clone != nullptr);
            auto otherBits = quantizer->CloneWithBits(bits == 7 ? 8 : 7);
            BOOST_REQUIRE(otherBits != nullptr);
            BOOST_CHECK_EQUAL(otherBits->LocalCentroidCount(), centerCount);
            for (auto model : {loaded, memoryLoaded, std::static_pointer_cast<COMMON::IQuantizer>(clone)}) {
                model->SetEnableADC(true);
                BOOST_CHECK_EQUAL(model->QuantizeSize(), (128 + 2 + centerCount) * sizeof(float));
                std::vector<std::uint8_t> query(model->QuantizeSize()), encoded(model->GetNumSubvectors());
                model->QuantizeVector(raw->GetVector(0), query.data());
                for (int i = 0; i < count; ++i) {
                    model->QuantizeVector(raw->GetVector(i), encoded.data(), false);
                    const auto* original = static_cast<const std::uint8_t*>(codes->GetVector(i));
                    BOOST_CHECK(std::memcmp(original, encoded.data(), encoded.size()) == 0);
                    BOOST_CHECK(std::isfinite(model->L2Distance(query.data(), original)));
                }
                for (int center = 0; center < centerCount; ++center) {
                    const auto* code = static_cast<const std::uint8_t*>(codes->GetVector(center * 8));
                    std::uint32_t storedId;
                    std::memcpy(&storedId, code + codes->Dimension() - sizeof(storedId), sizeof(storedId));
                    BOOST_CHECK_EQUAL(storedId, center);
                    std::vector<float> reconstructed(dimension);
                    model->ReconstructVector(code, reconstructed.data());
                    double exact = 0;
                    const auto* target = static_cast<const float*>(centers->GetVector(center));
                    for (int d = 0; d < dimension; ++d) {
                        BOOST_CHECK_SMALL(reconstructed[d] - target[d], 0.0002F);
                        exact += (values[d] - target[d]) * (values[d] - target[d]);
                    }
                    BOOST_CHECK_SMALL(model->L2Distance(query.data(), code) - static_cast<float>(exact),
                                      std::max(0.001F, static_cast<float>(exact) * 0.00001F));
                }
                // A padded residual need not lie in the original subspace; use an exact center there.
                const auto* first = static_cast<const std::uint8_t*>(codes->GetVector(dimension == 128 ? 3 : 0));
                const auto* second = static_cast<const std::uint8_t*>(codes->GetVector(25));
                std::vector<float> reconstructed(dimension);
                model->ReconstructVector(first, reconstructed.data());
                model->QuantizeVector(reconstructed.data(), query.data());
                const float adc = model->L2Distance(query.data(), second);
                model->SetEnableADC(false);
                BOOST_CHECK_CLOSE(model->L2Distance(first, second), adc, 0.001F);
                std::memcpy(encoded.data(), second, encoded.size());
                const std::uint32_t invalidId = centerCount;
                std::memcpy(encoded.data() + encoded.size() - sizeof(invalidId), &invalidId, sizeof(invalidId));
                BOOST_CHECK_THROW(model->L2Distance(first, encoded.data()), std::out_of_range);
            }
            const std::uint32_t invalidCount = 0;
            std::memcpy(saved.Data() + globalSize, &invalidCount, sizeof(invalidCount));
            auto invalid = std::make_shared<COMMON::RaBitQQuantizer>();
            BOOST_CHECK(invalid->LoadQuantizer(saved.Data() + sizeof(QuantizerType) + sizeof(VectorValueType))
                        == ErrorCode::FailedParseValue);
            std::remove(kQuantizerFile);
        }
    }
}

BOOST_AUTO_TEST_CASE(RaBitQLocalCentroidsUseExistingIndexPaths)
{
    const auto raw = MakeRawVectors();
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(kDimension, 7, false);
    BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
    BOOST_REQUIRE(quantizer->SetLocalCentroids(raw) == ErrorCode::Success);
    const auto codes = QuantizeVectors(raw, quantizer);
    VerifySearch(IndexAlgoType::BKT, raw, codes, quantizer);
    VerifySearch(IndexAlgoType::KDT, raw, codes, quantizer);
    VerifySpannSearch(raw, codes, quantizer, "STATIC");
    VerifySpannSearch(raw, codes, quantizer, "FILEIO");
    VerifySSDServingSearch(raw, codes, quantizer);
    std::remove(kQuantizerFile);
}

BOOST_AUTO_TEST_CASE(QuantizedGraphAndReplicaBuildDoNotTreatCodesAsADCQueries)
{
    const auto raw = MakeRawVectors();
    for (bool local : {false, true}) {
        auto quantizer = std::make_shared<CheckedADCQuantizer>();
        BOOST_REQUIRE(quantizer->Train(raw) == ErrorCode::Success);
        if (local) BOOST_REQUIRE(quantizer->SetLocalCentroids(raw) == ErrorCode::Success);
        const auto codes = QuantizeVectors(raw, quantizer);
        quantizer->reference = quantizer->CloneWithBits(7);
        BOOST_REQUIRE(quantizer->reference != nullptr);
        quantizer->reference->SetEnableADC(false);
        quantizer->SetEnableADC(true);
        for (SizeType x = 0; x < 8; ++x) {
            for (SizeType y = 8; y < 16; ++y) {
                const auto* a = static_cast<const std::uint8_t*>(codes->GetVector(x));
                const auto* b = static_cast<const std::uint8_t*>(codes->GetVector(y));
                BOOST_CHECK_EQUAL(quantizer->L2DistanceSDC(a, b), quantizer->reference->L2Distance(a, b));
            }
        }
        auto index = VectorIndex::CreateInstance(IndexAlgoType::BKT, VectorValueType::UInt8);
        index->SetQuantizer(quantizer);
        index->SetParameter("DistCalcMethod", "L2");
        index->SetParameter("NumberOfThreads", "1");
        index->SetParameter("NeighborhoodSize", "8");
        index->SetParameter("TPTNumber", "2");
        index->SetParameter("MaxCheck", "256");
        BOOST_REQUIRE(index->BuildIndex(codes, nullptr, false, true) == ErrorCode::Success);
        BOOST_CHECK_EQUAL(quantizer->invalidADCInputs.load(), 0);
        BOOST_CHECK(quantizer->GetEnableADC());
        quantizer->invalidADCInputs = 0;
        auto mutableCodes = codes;
        std::unordered_set<SizeType> except;
        std::vector<Edge> selections(raw->Count() * 8);
        index->ApproximateRNG(mutableCodes, except, 32, selections.data(), 8, 1, 1, 64, 1.0F, 0);
        BOOST_CHECK_EQUAL(quantizer->invalidADCInputs.load(), 0);
        BOOST_CHECK(quantizer->GetEnableADC());
    }
}

BOOST_AUTO_TEST_CASE(PQStoredDistancesAreIndependentOfADCMode)
{
    std::unique_ptr<float[]> codebooks(new float[4]{0.0F, 2.0F, 0.0F, 3.0F});
    auto quantizer = std::make_shared<COMMON::PQQuantizer<float>>(2, 2, 1, true, std::move(codebooks));
    const std::uint8_t x[2]{0, 0};
    const std::uint8_t y[2]{1, 1};
    auto index = VectorIndex::CreateInstance(IndexAlgoType::BKT, VectorValueType::UInt8);
    index->SetQuantizer(quantizer);
    index->SetParameter("DistCalcMethod", "L2");
    BOOST_CHECK_EQUAL(index->ComputeDistanceBetweenStoredVectors(x, y), 13.0F);
    BOOST_CHECK(quantizer->GetEnableADC());
    const float query[2]{0.0F, 0.0F};
    ByteArray adc = ByteArray::Alloc(quantizer->QuantizeSize());
    quantizer->QuantizeVector(query, adc.Data());
    BOOST_CHECK_EQUAL(index->ComputeDistance(adc.Data(), y), 13.0F);
    quantizer->SetEnableADC(false);
    BOOST_CHECK_EQUAL(index->ComputeDistanceBetweenStoredVectors(x, y), 13.0F);
    BOOST_CHECK_EQUAL(index->ComputeDistance(x, y), 13.0F);
}

BOOST_AUTO_TEST_CASE(RaBitQAutoTuneProducesNativeBuildHandoff)
{
    constexpr SizeType vectorCount = 6;
    constexpr SizeType queryCount = 2;
    constexpr DimensionType dimension = 8;
    const char* basePath = "rabitq_auto_base.bin";
    const char* outputFolder = "rabitq_auto_handoff";
    std::filesystem::remove_all(outputFolder);

    std::vector<float> baseData(vectorCount * dimension);
    for (SizeType row = 0; row < vectorCount; ++row) {
        for (DimensionType column = 0; column < dimension; ++column) {
            const float value = 0.0F + row * 0.5F + column * 0.01F;
            baseData[row * dimension + column] = value;
        }
    }
    std::shared_ptr<VectorSet> baseVectors = std::make_shared<BasicVectorSet>(
        ByteArray((std::uint8_t*)baseData.data(), baseData.size() * sizeof(float), false),
        VectorValueType::Float, dimension, vectorCount);

    COMMON::RaBitQAutoTuneResult result;

    BOOST_REQUIRE_MESSAGE(
        COMMON::RaBitQAutoTuner<float>::Run(
            baseVectors, queryCount, 10, 5, 0.9F, DistCalcMethod::L2, outputFolder, result) == ErrorCode::Success,
        "RaBitQ auto-tuning failed.");
    BOOST_CHECK_EQUAL(result.selectedBits, 1);
    BOOST_REQUIRE(result.quantizer != nullptr);
    BOOST_CHECK_EQUAL(
        result.codeDimension, result.quantizer->GetNumSubvectors());

    auto modelInput = f_createIO();
    BOOST_REQUIRE(modelInput->Initialize(
        result.quantizerPath.c_str(), std::ios::in | std::ios::binary));
    auto loaded = COMMON::IQuantizer::LoadIQuantizer(modelInput);
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_CHECK_EQUAL(loaded->GetNumSubvectors(), result.codeDimension);

    std::remove(basePath);
    std::filesystem::remove_all(outputFolder);
}

BOOST_AUTO_TEST_SUITE_END()
