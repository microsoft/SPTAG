// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Test.h"

#include <boost/filesystem.hpp>
#include <thread>

namespace
{
    using namespace SPTAG;

    struct FileLayerFixture
    {
        boost::filesystem::path directory =
            boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("sptag-file-layer-%%%%-%%%%");
        std::shared_ptr<SPANN::Index<uint8_t>> index = std::make_shared<SPANN::Index<uint8_t>>();
        std::vector<uint8_t> vectors;
        const int dimension;

        explicit FileLayerFixture(int layers = 1, bool shared = true, int dimensions = 2)
            : dimension(dimensions)
        {
            boost::filesystem::create_directory(directory);
            Set("IndexDirectory", directory.string(), "Base");
            Set("DistCalcMethod", "L2", "Base");
            Set("IndexAlgoType", "BKT", "Base");
            Set("ValueType", "UInt8", "Base");
            Set("isExecute", "true", "SelectHead");
            Set("SelectHeadType", "Random", "SelectHead");
            Set("Ratio", "1", "SelectHead");
            Set("NumberOfThreads", "1", "SelectHead");
            Set("isExecute", "true", "BuildHead");
            Set("NumberOfThreads", "1", "BuildHead");
            Set("NeighborhoodSize", "8", "BuildHead");
            Set("RefineIterations", "1", "BuildHead");
            Set("isExecute", "true");
            Set("BuildSsdIndex", "true");
            Set("Storage", "FILEIO");
            Set("Layers", std::to_string(layers));
            Set("ShareDB", shared ? "true" : "false");
            Set("NumberOfThreads", "1");
            Set("SearchThreadNum", "2");
            Set("AppendThreadNum", "1");
            Set("StartFileSizeGB", "1");
            Set("Update", "true");
            Set("AsyncMergeInSearch", "false");
            Set("ExcludeHead", "false");
            Set("ReplicaCount", "2");
            Set("InternalResultNum", "64");
            Set("SearchInternalResultNum", "64");
            Set("MaxCheck", "256");
            Set("MaxID", "10000");
            Set("EnableDynamicLayers", "true");
            Set("HeadMemoryHighWatermark", "1000000");
            Set("HeadMemoryTarget", "2000");
            Set("LayerResizeTimeoutSeconds", "5");
            vectors.resize(64 * dimension);
            for (int i = 0; i < 64; ++i) {
                vectors[dimension * i] = static_cast<uint8_t>(i);
                vectors[dimension * i + 1] = static_cast<uint8_t>(2 * i);
            }
            BOOST_REQUIRE(index->BuildIndex(vectors.data(), 64, dimension) == ErrorCode::Success);
            Wait();
        }

        ~FileLayerFixture()
        {
            index.reset();
            boost::filesystem::remove_all(directory);
        }

        void Set(const char* name, const std::string& value, const char* section = "BuildSSDIndex")
        {
            BOOST_REQUIRE(index->SetParameter(name, value.c_str(), section) == ErrorCode::Success);
        }

        void Wait()
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!index->AllFinished() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            BOOST_REQUIRE(index->AllFinished());
        }

        void Check()
        {
            for (int i = 0; i < 64; ++i) {
                COMMON::QueryResultSet<uint8_t> query(vectors.data() + dimension * i, 1);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
                BOOST_CHECK_EQUAL(query.GetResult(0)->VID, i);
                BOOST_CHECK_EQUAL(query.GetResult(0)->Dist, 0);
            }
        }

        std::vector<std::string> BasePostings()
        {
            std::vector<SizeType> heads;
            BOOST_REQUIRE(index->GetHeadIndexMapping(1, heads) == ErrorCode::Success);
            std::sort(heads.begin(), heads.end());
            SPANN::ExtraWorkSpace workspace;
            index->InitWorkSpace(&workspace);
            std::vector<std::string> postings;
            for (auto head : heads) {
                std::string posting;
                BOOST_REQUIRE(index->GetDiskIndex()->GetWritePosting(&workspace, head, posting) == ErrorCode::Success);
                postings.push_back(std::move(posting));
            }
            return postings;
        }

        void Reload()
        {
            index.reset();
            std::shared_ptr<VectorIndex> loaded;
            BOOST_REQUIRE(VectorIndex::LoadIndex(directory.string(), loaded) == ErrorCode::Success);
            index = std::dynamic_pointer_cast<SPANN::Index<uint8_t>>(loaded);
            BOOST_REQUIRE(index);
            Wait();
        }
    };
}

BOOST_AUTO_TEST_SUITE(FileIOLayerTest)

BOOST_AUTO_TEST_CASE(IsolatedLayersPreservePostingsAndReload)
{
    for (int layers : {1, 2}) {
        for (bool shared : {false, true}) {
            FileLayerFixture fixture(layers, shared);
            const auto original = fixture.BasePostings();
            const auto memory = fixture.index->GetHeadMemoryUsage();
            fixture.Set("Ratio", "0.1", "SelectHead");
            fixture.Set("HeadMemoryTarget", std::to_string(memory / 2));
            BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
            BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), std::to_string(layers + 1));
            BOOST_CHECK_LE(fixture.index->GetMemoryIndex()->GetNumSamples(), 6);
            BOOST_CHECK_LE(fixture.index->GetHeadMemoryUsage(), memory / 2);
            BOOST_CHECK(original == fixture.BasePostings());
            fixture.Check();
            fixture.Reload();
            fixture.Check();
            BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == ErrorCode::Success);
            fixture.Reload();
            fixture.Check();
        }
    }
}

BOOST_AUTO_TEST_CASE(RepeatedGrowthAndUpdatesSurviveDirectoryCopy)
{
    FileLayerFixture fixture;
    fixture.Set("Ratio", "0.1", "SelectHead");
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    fixture.Check();
    uint8_t inserted[2] = {70, 140};
    BOOST_REQUIRE(fixture.index->AddIndex(inserted, 1, 2, nullptr, false, true) == ErrorCode::Success);
    fixture.Wait();
    COMMON::QueryResultSet<uint8_t> query(inserted, 1);
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(query.GetResult(0)->VID, 64);
    BOOST_REQUIRE(fixture.index->DeleteIndex(64) == ErrorCode::Success);
    const boost::filesystem::path destination(fixture.directory.string() + "_copy");
    const auto saved = fixture.index->SaveIndex(destination.string());
    fixture.index.reset();
    if (saved == ErrorCode::Success) {
        boost::filesystem::remove_all(fixture.directory);
        fixture.directory = destination;
    }
    else boost::filesystem::remove_all(destination);
    BOOST_REQUIRE(saved == ErrorCode::Success);
    fixture.Reload();
    fixture.Check();
    query.Reset();
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_NE(query.GetResult(0)->VID, 64);
}

BOOST_AUTO_TEST_CASE(BKTSelectionRecomputesThresholdsForMemoryBudget)
{
    FileLayerFixture fixture;
    const auto original = fixture.BasePostings();
    const auto target = fixture.index->GetHeadMemoryUsage() / 3;
    fixture.Set("SelectHeadType", "BKT", "SelectHead");
    fixture.Set("Ratio", "0.1", "SelectHead");
    fixture.Set("SelectThreshold", "2", "SelectHead");
    fixture.Set("SplitThreshold", "4", "SelectHead");
    fixture.Set("SplitFactor", "2", "SelectHead");
    fixture.Set("HeadMemoryTarget", std::to_string(target));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    BOOST_CHECK_LE(fixture.index->GetHeadMemoryUsage(), target);
    BOOST_CHECK_LT(fixture.index->GetMemoryIndex()->GetNumSamples(), 64 / 3);
    BOOST_CHECK(original == fixture.BasePostings());
    fixture.Check();
    fixture.Reload();
    fixture.Check();
}

BOOST_AUTO_TEST_CASE(RejectedCandidateKeepsSavedIndexUsable)
{
    FileLayerFixture fixture;
    BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == ErrorCode::Success);
    const auto oldHead = fixture.index->GetMemoryIndex();
    fixture.Set("HeadMemoryTarget", "1");
    BOOST_CHECK(fixture.index->GrowTopLayer() == ErrorCode::MemoryOverFlow);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == oldHead);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "1");
    fixture.Check();
    fixture.Reload();
    fixture.Check();
}

BOOST_AUTO_TEST_CASE(FullRoutingChoicesUseNearbyCapacityWithoutTruncation)
{
    FileLayerFixture fixture(1, true, 1024);
    const auto original = fixture.BasePostings();
    const auto memory = fixture.index->GetHeadMemoryUsage();
    fixture.Set("PostingVectorLimit", "0");
    fixture.Set("PostingPageLimit", "1");
    fixture.Set("SearchPostingPageLimit", "1");
    fixture.Set("ReplicaCount", "1");
    fixture.Set("Ratio", "0.01", "SelectHead");
    fixture.Set("HeadMemoryTarget", std::to_string(memory / 2));
    const auto oldHead = fixture.index->GetMemoryIndex();
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::MemoryOverFlow);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == oldHead);
    fixture.Set("Ratio", "0.5", "SelectHead");
    fixture.Set("HeadMemoryTarget", std::to_string(memory * 3 / 4));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    BOOST_CHECK(original == fixture.BasePostings());
    fixture.Check();
    fixture.Reload();
    fixture.Check();
}

BOOST_AUTO_TEST_SUITE_END()
