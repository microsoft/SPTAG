// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"

#include "inc/Core/Common/OfficialRaBitQ.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace SPTAG;

namespace
{

constexpr SizeType kVectorCount = 96;
constexpr DimensionType kDimension = 128;
constexpr int kRaBitQBits = 3;
constexpr const char* kOfficialModelFile = "official_rabitq_model_test.bin";

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

void VerifyOptHashTableGrowth()
{
    COMMON::OptHashPosVector visited;
    visited.Init(1, 0);
    for (SizeType i = 0; i < 4096; ++i) {
        BOOST_CHECK(!visited.CheckAndSet(i));
    }
    for (SizeType i = 0; i < 4096; ++i) {
        BOOST_CHECK(visited.CheckAndSet(i));
    }
}

void VerifyOfficialRaBitQ(const std::shared_ptr<VectorSet>& p_raw)
{
    BOOST_CHECK(COMMON::OfficialRaBitQ::IsSupportedDimension(kDimension));
    BOOST_CHECK(!COMMON::OfficialRaBitQ::IsSupportedDimension(63));
    BOOST_CHECK(!COMMON::OfficialRaBitQ::IsSupportedDimension(4096));

    COMMON::OfficialRaBitQ model(kDimension, kRaBitQBits);
    BOOST_REQUIRE(model.Train(p_raw) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(model.CodeBytes(), 68U);

    std::vector<std::uint8_t> selfCode(model.CodeBytes());
    std::vector<std::uint8_t> farCode(model.CodeBytes());
    model.Encode(static_cast<const float*>(p_raw->GetVector(37)), selfCode.data());
    model.Encode(static_cast<const float*>(p_raw->GetVector(kVectorCount - 1)), farCode.data());

    const auto query = model.PrepareQuery(static_cast<const float*>(p_raw->GetVector(37)));
    const float selfDistance = model.Estimate(query, selfCode.data());
    const float farDistance = model.Estimate(query, farCode.data());
    BOOST_CHECK(std::isfinite(selfDistance));
    BOOST_CHECK(std::isfinite(farDistance));
    BOOST_CHECK(selfDistance < farDistance);

    BOOST_REQUIRE(model.Save(kOfficialModelFile) == ErrorCode::Success);
    COMMON::OfficialRaBitQ loaded;
    BOOST_REQUIRE(loaded.Load(kOfficialModelFile) == ErrorCode::Success);
    const auto loadedQuery = loaded.PrepareQuery(static_cast<const float*>(p_raw->GetVector(37)));
    BOOST_CHECK_CLOSE(
        loaded.Estimate(loadedQuery, selfCode.data()), selfDistance, 1e-4F);
    std::remove(kOfficialModelFile);
}

void VerifyOfficialRaBitQOneBit(const std::shared_ptr<VectorSet>& p_raw)
{
    COMMON::OfficialRaBitQ model(kDimension, 1);
    BOOST_REQUIRE(model.Train(p_raw) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(model.ExCodeBytes(), 0U);

    std::vector<std::uint8_t> selfCode(model.CodeBytes());
    std::vector<std::uint8_t> farCode(model.CodeBytes());
    model.Encode(static_cast<const float*>(p_raw->GetVector(37)), selfCode.data());
    model.Encode(static_cast<const float*>(p_raw->GetVector(kVectorCount - 1)), farCode.data());

    const auto query = model.PrepareQuery(static_cast<const float*>(p_raw->GetVector(37)));
    const float selfDistance = model.Estimate(query, selfCode.data());
    const float farDistance = model.Estimate(query, farCode.data());
    BOOST_CHECK(std::isfinite(selfDistance));
    BOOST_CHECK(std::isfinite(farDistance));
    BOOST_CHECK(selfDistance < farDistance);
}

void VerifyOfficialRaBitQPostingSearch(const std::shared_ptr<VectorSet>& p_raw, const char* p_storage)
{
    const std::string indexDirectory = std::string("official_rabitq_posting_spann_test_") + p_storage;
    const std::string modelFile = indexDirectory + ".model";
    const bool staticStorage = std::string(p_storage) == "STATIC";
    std::filesystem::remove_all(indexDirectory);
    std::remove(modelFile.c_str());

    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);
    index->SetParameter("IndexDirectory", indexDirectory, "Base");
    index->SetParameter("IndexAlgoType", "BKT", "Base");
    index->SetParameter("DistCalcMethod", "L2", "Base");

    index->SetParameter("isExecute", "true", "SelectHead");
    index->SetParameter("NumberOfThreads", "1", "SelectHead");
    index->SetParameter("Ratio", staticStorage ? "0.1" : "0.25", "SelectHead");
    index->SetParameter("BKTKmeansK", "4", "SelectHead");
    index->SetParameter("BKTLeafSize", "2", "SelectHead");
    index->SetParameter("SamplesNumber", "16", "SelectHead");

    index->SetParameter("isExecute", "true", "BuildHead");
    index->SetParameter("NumberOfThreads", "1", "BuildHead");
    index->SetParameter("NeighborhoodSize", "8", "BuildHead");
    index->SetParameter("TPTNumber", "1", "BuildHead");
    index->SetParameter("TPTLeafSize", "64", "BuildHead");
    index->SetParameter("MaxCheck", "256", "BuildHead");
    index->SetParameter("MaxCheckForRefineGraph", "256", "BuildHead");
    index->SetParameter("RefineIterations", "1", "BuildHead");

    index->SetParameter("isExecute", "true", "BuildSSDIndex");
    index->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    index->SetParameter("Storage", p_storage, "BuildSSDIndex");
    index->SetParameter("PostingQuantizer", "RaBitQOfficial", "BuildSSDIndex");
    index->SetParameter("PostingQuantizerFile", modelFile.c_str(), "BuildSSDIndex");
    index->SetParameter("PostingQuantBits", "3", "BuildSSDIndex");
    index->SetParameter("PostingQuantizerTrainingSamples", "96", "BuildSSDIndex");
    index->SetParameter("NumberOfThreads", "1", "BuildSSDIndex");
    index->SetParameter("PostingPageLimit", staticStorage ? "2" : "1", "BuildSSDIndex");
    index->SetParameter("SearchPostingPageLimit", staticStorage ? "2" : "1", "BuildSSDIndex");
    index->SetParameter("InternalResultNum", "96", "BuildSSDIndex");
    index->SetParameter("SearchInternalResultNum", "96", "BuildSSDIndex");
    index->SetParameter("ReplicaCount", staticStorage ? "8" : "1", "BuildSSDIndex");
    index->SetParameter("StartFileSizeGB", "1", "BuildSSDIndex");
    index->SetParameter("MaxFileSizeGB", "1", "BuildSSDIndex");
    index->SetParameter("EnableDeltaEncoding", "false", "BuildSSDIndex");
    index->SetParameter("EnablePostingListRearrange", "false", "BuildSSDIndex");
    index->SetParameter("EnableDataCompression", "false", "BuildSSDIndex");
    index->SetParameter("BufferLength", "0", "BuildSSDIndex");
    index->SetParameter("AsyncMergeInSearch", "false", "BuildSSDIndex");

    BOOST_REQUIRE(index->BuildIndex(p_raw, nullptr, false, true) == ErrorCode::Success);

    auto* spannIndex = static_cast<SPANN::Index<float>*>(index.get());
    std::vector<SizeType> headIDs;
    BOOST_REQUIRE(spannIndex->GetHeadIndexMapping(1, headIDs) == ErrorCode::Success);
    if (staticStorage) {
        COMMON::OfficialRaBitQ loadedModel;
        BOOST_REQUIRE(loadedModel.Load(modelFile) == ErrorCode::Success);

        std::string posting;
        bool verifiedPostingCode = false;
        const size_t recordBytes = sizeof(SizeType) + sizeof(std::uint32_t) +
            ((loadedModel.CodeBytes() + alignof(std::uint64_t) - 1) &
             ~(alignof(std::uint64_t) - 1));
        for (const SizeType headID : headIDs) {
            BOOST_REQUIRE(spannIndex->GetDiskIndex()->GetWritePosting(nullptr, headID, posting) == ErrorCode::Success);
            for (size_t offset = 0; offset + recordBytes <= posting.size(); offset += recordBytes) {
                SizeType vectorID;
                std::memcpy(&vectorID, posting.data() + offset, sizeof(vectorID));
                if (vectorID < 0 || vectorID >= kVectorCount) {
                    continue;
                }

                std::vector<std::uint8_t> expectedCode(loadedModel.CodeBytes());
                loadedModel.Encode(
                    reinterpret_cast<const float*>(p_raw->GetVector(vectorID)),
                    expectedCode.data());
                BOOST_CHECK_EQUAL_COLLECTIONS(
                    expectedCode.begin(), expectedCode.end(),
                    reinterpret_cast<const std::uint8_t*>(posting.data() + offset + sizeof(SizeType) + sizeof(std::uint32_t)),
                    reinterpret_cast<const std::uint8_t*>(posting.data() + offset + sizeof(SizeType) + sizeof(std::uint32_t)) + expectedCode.size());
                verifiedPostingCode = true;
                break;
            }
            if (verifiedPostingCode) {
                break;
            }
        }
        BOOST_CHECK(verifiedPostingCode);
    }

    const std::unordered_set<SizeType> headIDSet(headIDs.begin(), headIDs.end());
    SizeType expected = 0;
    while (expected < kVectorCount && headIDSet.count(expected) != 0) {
        ++expected;
    }
    BOOST_REQUIRE(expected < kVectorCount);

    COMMON::QueryResultSet<float> query(
        reinterpret_cast<const float*>(p_raw->GetVector(expected)), 96);
    BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);

    bool foundExpected = false;
    for (int rank = 0; rank < query.GetResultNum(); ++rank) {
        const auto* result = query.GetResult(rank);
        if (result != nullptr && result->VID == expected) {
            foundExpected = true;
            break;
        }
    }
    BOOST_CHECK(foundExpected);

    if (staticStorage) {
        BOOST_REQUIRE(!headIDs.empty());
        const SizeType headExpected = headIDs.front();
        COMMON::QueryResultSet<float> headQuery(
            reinterpret_cast<const float*>(p_raw->GetVector(headExpected)), 96);
        BOOST_REQUIRE(index->SearchIndex(headQuery) == ErrorCode::Success);

        bool foundHead = false;
        for (int rank = 0; rank < headQuery.GetResultNum(); ++rank) {
            const auto* result = headQuery.GetResult(rank);
            if (result != nullptr && result->VID == headExpected) {
                foundHead = true;
                break;
            }
        }
        BOOST_CHECK(foundHead);
    } else {
        BOOST_CHECK(index->AddIndex(
                        p_raw->GetVector(0), 1, kDimension, std::shared_ptr<MetadataSet>(), false, false) ==
                    ErrorCode::Undefined);
    }

    index.reset();
    std::remove(modelFile.c_str());
    std::filesystem::remove_all(indexDirectory);
}

} // namespace

BOOST_AUTO_TEST_SUITE(OfficialRaBitQTest)

BOOST_AUTO_TEST_CASE(OfficialRaBitQUsesPostingPath)
{
    VerifyOptHashTableGrowth();
    const auto raw = MakeRawVectors();
    VerifyOfficialRaBitQ(raw);
    VerifyOfficialRaBitQOneBit(raw);
    VerifyOfficialRaBitQPostingSearch(raw, "FILEIO");
    VerifyOfficialRaBitQPostingSearch(raw, "STATIC");
}

BOOST_AUTO_TEST_SUITE_END()
