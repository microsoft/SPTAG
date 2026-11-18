// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/ResultIterator.h"
#include "inc/Test.h"

#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
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

        explicit FileLayerFixture(int layers = 1, bool shared = true, int dimensions = 2, bool shrink = false)
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
            Set("EnableDynamicLayerShrink", shrink ? "true" : "false");
            Set("HeadMemoryLowWatermark", "1");
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

        void Check(int deleted = -1)
        {
            for (int i = 0; i < 64; ++i) {
                if (i == deleted) continue;
                COMMON::QueryResultSet<uint8_t> query(vectors.data() + dimension * i, 1);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
                BOOST_CHECK_EQUAL(query.GetResult(0)->VID, i);
                BOOST_CHECK_EQUAL(query.GetResult(0)->Dist, 0);
            }
        }

        void Grow()
        {
            Set("Ratio", "0.1", "SelectHead");
            Set("HeadMemoryTarget", std::to_string(index->GetHeadMemoryUsage() / 2));
            BOOST_REQUIRE(index->GrowTopLayer() == ErrorCode::Success);
            Wait();
        }

        void NotifyResize()
        {
            const int layers = std::stoi(index->GetParameter("Layers", "BuildSSDIndex"));
            std::vector<SizeType> heads;
            BOOST_REQUIRE(index->GetHeadIndexMapping(layers, heads) == ErrorCode::Success);
            BOOST_REQUIRE(!heads.empty());
            BOOST_REQUIRE(index->AddHeadIndex(vectors.data() + dimension * heads.front(), heads.front(),
                0, dimension, layers, nullptr) == ErrorCode::Success);
            Wait();
        }

        std::string Manifest()
        {
            std::ifstream input((directory / "indexloader.ini").string());
            BOOST_REQUIRE(input);
            return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
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
    void RunAutomaticLayerWorkload(int scale)
    {
        constexpr int dimension = 32, k = 10, queryCount = 64;
        const int initial = 1024 * scale, total = 16384 * scale, remaining = 256 * scale;
        const std::uint64_t high = 24000 * scale, target = 16000 * scale, low = 10000 * scale;
        const auto directory = boost::filesystem::temp_directory_path() /
            boost::filesystem::unique_path("sptag-auto-layers-%%%%-%%%%");
        boost::filesystem::create_directory(directory);
        struct Cleanup
        {
            boost::filesystem::path directory;
            ~Cleanup() { boost::filesystem::remove_all(directory); }
        } cleanup{directory};
        auto index = std::make_shared<SPANN::Index<uint8_t>>();
        auto set = [&](const char* name, const std::string& value, const char* section = "BuildSSDIndex") {
            BOOST_REQUIRE(index->SetParameter(name, value.c_str(), section) == ErrorCode::Success);
        };
        set("IndexDirectory", directory.string(), "Base");
        set("DistCalcMethod", "L2", "Base");
        set("IndexAlgoType", "BKT", "Base");
        set("ValueType", "UInt8", "Base");
        set("isExecute", "true", "SelectHead");
        set("SelectHeadType", "Random", "SelectHead");
        set("Ratio", "0.1", "SelectHead");
        set("NumberOfThreads", "1", "SelectHead");
        set("isExecute", "true", "BuildHead");
        set("NumberOfThreads", "1", "BuildHead");
        set("NeighborhoodSize", "16", "BuildHead");
        set("RefineIterations", "1", "BuildHead");
        set("isExecute", "true");
        set("BuildSsdIndex", "true");
        set("Storage", "FILEIO");
        set("Layers", "1");
        set("ShareDB", "true");
        set("NumberOfThreads", "1");
        set("SearchThreadNum", "2");
        set("AppendThreadNum", "1");
        set("ReassignThreadNum", "1");
        set("StartFileSizeGB", "1");
        set("Update", "true");
        set("AsyncMergeInSearch", "true");
        set("DisableReassign", "false");
        set("ExcludeHead", "false");
        set("ReplicaCount", "2");
        set("PostingVectorLimit", "0");
        set("PostingPageLimit", "1");
        set("SearchPostingPageLimit", "1");
        set("BufferLength", "1");
        set("MergeThreshold", "24");
        set("InternalResultNum", "64");
        set("SearchInternalResultNum", "64");
        set("MaxCheck", "512");
        set("MaxID", std::to_string(total * 2));
        set("EnableDynamicLayers", "true");
        set("EnableDynamicLayerShrink", "true");
        set("HeadMemoryHighWatermark", std::to_string(high));
        set("HeadMemoryTarget", std::to_string(target));
        set("HeadMemoryLowWatermark", std::to_string(low));
        set("LayerResizeCooldownSeconds", "1");
        set("LayerResizeTimeoutSeconds", "60");
        set("MaxLayers", "4");

        std::mt19937 random(1729);
        std::vector<uint8_t> vectors(total * dimension), queries(queryCount * dimension);
        std::array<std::array<uint8_t, dimension>, 16> centers;
        for (auto& center : centers)
            for (auto& coordinate : center) coordinate = 16 + random() % 208;
        for (int i = 0; i < total; ++i)
            for (int d = 0; d < dimension; ++d)
                vectors[i * dimension + d] = centers[i % centers.size()][d] + random() % 17;
        for (int q = 0; q < queryCount; ++q)
            memcpy(queries.data() + q * dimension, vectors.data() + (q * 251 % total) * dimension, dimension);
        BOOST_TEST_MESSAGE("AUTO-WORKLOAD config initial=" << initial << " total=" << total
            << " remaining=" << remaining << " low=" << low << " target=" << target << " high=" << high);
        BOOST_REQUIRE(index->BuildIndex(vectors.data(), initial, dimension) == ErrorCode::Success);
        auto wait = [&]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
            while (!index->AllFinished() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            BOOST_REQUIRE_MESSAGE(index->AllFinished(), "Automatic workload maintenance timed out");
            BOOST_REQUIRE(index->GetLayerResizeStatus() == ErrorCode::Success);
        };
        auto layers = [&]() { return std::stoi(index->GetParameter("Layers", "BuildSSDIndex")); };
        auto snapshot = [&](const char* stage) {
            std::vector<SizeType> routes;
            BOOST_REQUIRE(index->GetHeadIndexMapping(1, routes) == ErrorCode::Success);
            BOOST_TEST_MESSAGE("AUTO-WORKLOAD stage=" << stage << " layers=" << layers()
                << " routes=" << routes.size() << " headBytes=" << index->GetHeadMemoryUsage()
                << " headSlots=" << index->GetMemoryIndex()->GetNumSamples()
                << " headDeleted=" << index->GetMemoryIndex()->GetNumDeleted());
        };
        auto sweep = [&]() {
            for (int q = 0; q < queryCount; ++q) {
                COMMON::QueryResultSet<uint8_t> query(queries.data() + q * dimension, k);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
            }
        };
        auto verify = [&](int count, int liveStart, const char* stage) {
            std::vector<SizeType> actual;
            BOOST_REQUIRE(index->GetHeadIndexMapping(0, actual) == ErrorCode::Success);
            std::sort(actual.begin(), actual.end());
            BOOST_REQUIRE_EQUAL(actual.size(), count - liveStart);
            for (int i = liveStart; i < count; ++i) BOOST_REQUIRE_EQUAL(actual[i - liveStart], i);
            int hits = 0;
            for (int q = 0; q < queryCount; ++q) {
                const auto* queryVector = queries.data() + q * dimension;
                std::vector<float> distances;
                distances.reserve(count - liveStart);
                for (int i = liveStart; i < count; ++i) {
                    float distance = 0;
                    for (int d = 0; d < dimension; ++d) {
                        const float difference = static_cast<float>(queryVector[d]) - vectors[i * dimension + d];
                        distance += difference * difference;
                    }
                    distances.push_back(distance);
                }
                std::nth_element(distances.begin(), distances.begin() + k - 1, distances.end());
                const float boundary = distances[k - 1];
                COMMON::QueryResultSet<uint8_t> query(queryVector, k);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
                std::set<SizeType> seen;
                for (int j = 0; j < k; ++j) {
                    const auto* result = query.GetResult(j);
                    BOOST_REQUIRE_GE(result->VID, liveStart);
                    BOOST_REQUIRE_LT(result->VID, count);
                    BOOST_REQUIRE(seen.insert(result->VID).second);
                    float distance = 0;
                    for (int d = 0; d < dimension; ++d) {
                        const float difference = static_cast<float>(queryVector[d]) - vectors[result->VID * dimension + d];
                        distance += difference * difference;
                    }
                    BOOST_REQUIRE_EQUAL(result->Dist, distance);
                    if (distance <= boundary) ++hits;
                }
            }
            const double recall = static_cast<double>(hits) / (queryCount * k);
            BOOST_TEST_MESSAGE("AUTO-WORKLOAD verify=" << stage << " live=" << count - liveStart
                << " recallAt10=" << recall << " deletedHits=0");
            BOOST_REQUIRE_GE(recall, 0.95);
        };
        wait();
        BOOST_REQUIRE_EQUAL(layers(), 1);
        BOOST_REQUIRE_LT(index->GetHeadMemoryUsage(), high);
        snapshot("initial");
        verify(initial, 0, "initial");
        wait();

        std::atomic_int queryErrors{0}, queryCalls{0};
        std::atomic_bool pauseQueries{false};
        std::mutex queryMutex;
        const auto pause = [&]() {
            pauseQueries = true;
            std::lock_guard<std::mutex> barrier(queryMutex);
        };
        std::jthread reader([&](std::stop_token stop) {
            int q = 0;
            while (!stop.stop_requested()) {
                {
                    std::lock_guard<std::mutex> queryGuard(queryMutex);
                    if (!pauseQueries) {
                        COMMON::QueryResultSet<uint8_t> query(queries.data() + q * dimension, k);
                        if (index->SearchIndex(query) != ErrorCode::Success) ++queryErrors;
                        ++queryCalls;
                        q = (q + 1) % queryCount;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        for (int begin = initial; begin < total; begin += 512) {
            const int count = (std::min)(512, total - begin);
            std::vector<SizeType> assigned(count, -1);
            BOOST_REQUIRE(index->AddIndex(vectors.data() + begin * dimension, count, dimension,
                nullptr, false, true, assigned.data()) == ErrorCode::Success);
            for (int i = 0; i < count; ++i) BOOST_REQUIRE_EQUAL(assigned[i], begin + i);
            pause();
            wait();
            pauseQueries = false;
        }
        pause();
        wait();
        snapshot("inserted");
        BOOST_REQUIRE_GE(layers(), 2);
        verify(total, 0, "inserted");
        wait();
        pauseQueries = false;
        const int liveStart = total - remaining;
        for (int id = 0; id < liveStart; ++id) {
            BOOST_REQUIRE(index->DeleteIndex(id) == ErrorCode::Success);
            if (id % 1024 == 1023) {
                pause();
                wait();
                pauseQueries = false;
            }
        }
        pause();
        wait();
        snapshot("deleted");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
        int rounds = 0;
        while (layers() != 1 && std::chrono::steady_clock::now() < deadline) {
            sweep();
            wait();
            if (++rounds % 100 == 0) snapshot("converging");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        wait();
        snapshot("converged");
        BOOST_REQUIRE_EQUAL(layers(), 1);
        verify(total, liveStart, "shrunk");
        wait();
        reader.request_stop();
        reader.join();
        BOOST_REQUIRE_EQUAL(queryErrors.load(), 0);
        BOOST_REQUIRE_GT(queryCalls.load(), 0);
        BOOST_TEST_MESSAGE("AUTO-WORKLOAD concurrentQueries=" << queryCalls.load());
        snapshot("saving");
        const auto savedHeadSlots = index->GetMemoryIndex()->GetNumSamples();
        const auto savedHeadDeleted = index->GetMemoryIndex()->GetNumDeleted();
        BOOST_REQUIRE(index->SaveIndex(directory.string()) == ErrorCode::Success);
        index.reset();
        std::shared_ptr<VectorIndex> loaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(directory.string(), loaded) == ErrorCode::Success);
        index = std::dynamic_pointer_cast<SPANN::Index<uint8_t>>(loaded);
        BOOST_REQUIRE(index);
        wait();
        BOOST_REQUIRE_EQUAL(layers(), 1);
        BOOST_REQUIRE_EQUAL(index->GetMemoryIndex()->GetNumSamples(), savedHeadSlots);
        BOOST_REQUIRE_EQUAL(index->GetMemoryIndex()->GetNumDeleted(), savedHeadDeleted);
        verify(total, liveStart, "reloaded");
        wait();
        BOOST_REQUIRE(index->Check() == ErrorCode::Success);
    }
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

BOOST_AUTO_TEST_CASE(CachedWorkspaceRebindsToCurrentIndex)
{
    FileLayerFixture first;
    FileLayerFixture second;
    SPANN::ExtraWorkSpace workspace;
    first.index->InitWorkSpace(&workspace);
    auto firstOwner = workspace.m_workspaceOwner.lock();
    BOOST_REQUIRE(firstOwner);
    second.index->InitWorkSpace(&workspace, true);
    auto secondOwner = workspace.m_workspaceOwner.lock();
    BOOST_REQUIRE(secondOwner);
    BOOST_CHECK(firstOwner != secondOwner);
    BOOST_REQUIRE_GE(workspace.m_workspaceId, 0);
    BOOST_CHECK_LT(workspace.m_workspaceId, second.index->GetOptions()->GetIOThreadCount());
    for (const auto& request : workspace.m_diskRequests)
        BOOST_CHECK_EQUAL(request.m_status, workspace.m_workspaceId);
    const auto reserved = second.index->GetOptions()->GetIOThreadCount();
    second.Set("HeadMemoryTarget", std::to_string(second.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(second.index->GrowTopLayer() == ErrorCode::Success);
    BOOST_CHECK_EQUAL(second.index->GetOptions()->GetIOThreadCount(), reserved);
    first.Check();
    second.Check();
}

BOOST_AUTO_TEST_CASE(ShrinkPreservesRoutingPostingsAndReload)
{
    for (int layers : {1, 2}) {
        for (bool shared : {false, true}) {
            FileLayerFixture fixture(layers, shared);
            auto metadata = ByteArray::Alloc(64);
            auto offsets = ByteArray::Alloc(65 * sizeof(std::uint64_t));
            for (std::uint64_t i = 0; i <= 64; ++i) {
                memcpy(offsets.Data() + i * sizeof(i), &i, sizeof(i));
                if (i < 64) metadata.Data()[i] = static_cast<uint8_t>(i);
            }
            fixture.index->SetMetadata(new MemMetadataSet(metadata, offsets, 64));
            const auto checkMetadata = [&]() {
                for (int i = 0; i < 64; ++i) {
                    auto value = fixture.index->GetMetadata(i);
                    BOOST_REQUIRE_EQUAL(value.Length(), 1);
                    BOOST_CHECK_EQUAL(value.Data()[0], i);
                }
            };
            const auto postings = fixture.BasePostings();
            std::vector<SizeType> expected;
            BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(layers, expected) == ErrorCode::Success);
            std::sort(expected.begin(), expected.end());
            fixture.Grow();
            const auto retiredHead = fixture.directory / fixture.index->GetParameter("HeadIndexFolder", "Base");
            fixture.Set("HeadMemoryTarget", "750000");
            BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
            BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), std::to_string(layers));
            BOOST_CHECK(postings == fixture.BasePostings());
            std::vector<SizeType> actual;
            BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(layers, actual) == ErrorCode::Success);
            std::sort(actual.begin(), actual.end());
            BOOST_CHECK(expected == actual);
            BOOST_CHECK(boost::filesystem::exists(retiredHead));
            fixture.Check();
            checkMetadata();
            fixture.Reload();
            fixture.Check();
            checkMetadata();
            BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == ErrorCode::Success);
            fixture.Reload();
            fixture.Check();
        }
    }
}

BOOST_AUTO_TEST_CASE(ShrinkAfterMergeUsesCurrentRoutingVersions)
{
    FileLayerFixture fixture;
    fixture.Grow();
    fixture.Set("DisableReassign", "true");
    auto base = std::dynamic_pointer_cast<SPANN::ExtraDynamicSearcher<uint8_t>>(fixture.index->GetDiskIndex());
    BOOST_REQUIRE(base);
    base->MergeAsync(0);
    fixture.Wait();
    BOOST_REQUIRE(base->MaintenanceStatus() == ErrorCode::Success);
    std::vector<SizeType> expected;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(1, expected) == ErrorCode::Success);
    BOOST_REQUIRE_LT(expected.size(), 64);
    std::sort(expected.begin(), expected.end());
    const auto postings = fixture.BasePostings();
    BOOST_REQUIRE(fixture.index->DeleteIndex(0) == ErrorCode::Success);
    fixture.Wait();
    fixture.Set("HeadMemoryTarget", "750000");
    BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
    std::vector<SizeType> actual;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(1, actual) == ErrorCode::Success);
    std::sort(actual.begin(), actual.end());
    BOOST_CHECK(expected == actual);
    BOOST_CHECK(postings == fixture.BasePostings());
    fixture.Check(0);
    COMMON::QueryResultSet<uint8_t> query(fixture.vectors.data(), 1);
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_NE(query.GetResult(0)->VID, 0);
    fixture.Reload();
    fixture.Check(0);
    query.Reset();
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_NE(query.GetResult(0)->VID, 0);
}

BOOST_AUTO_TEST_CASE(ShrinkReassignmentWithoutCandidatesPreservesVersions)
{
    FileLayerFixture fixture;
    fixture.Grow();
    auto base = std::dynamic_pointer_cast<SPANN::ExtraDynamicSearcher<uint8_t>>(fixture.index->GetDiskIndex());
    auto upper = std::dynamic_pointer_cast<SPANN::ExtraDynamicSearcher<uint8_t>>(fixture.index->GetDiskIndex(1));
    BOOST_REQUIRE(base);
    BOOST_REQUIRE(upper);
    std::vector<SizeType> routes, heads;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(1, routes) == ErrorCode::Success);
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(2, heads) == ErrorCode::Success);
    BOOST_REQUIRE(!routes.empty());
    BOOST_REQUIRE(!heads.empty());
    SPANN::ExtraWorkSpace workspace;
    fixture.index->InitWorkSpace(&workspace);
    const size_t recordSize = sizeof(SizeType) + 1 + fixture.dimension;
    std::string basePosting, upperPosting;
    BOOST_REQUIRE(base->GetWritePosting(&workspace, routes.front(), basePosting) == ErrorCode::Success);
    BOOST_REQUIRE(upper->GetWritePosting(&workspace, heads.front(), upperPosting) == ErrorCode::Success);
    BOOST_REQUIRE_GE(basePosting.size(), recordSize);
    BOOST_REQUIRE_GE(upperPosting.size(), recordSize);
    auto baseRecord = std::make_shared<std::string>(basePosting.substr(0, recordSize));
    auto upperRecord = std::make_shared<std::string>(upperPosting.substr(0, recordSize));
    const auto originalBaseRecord = *baseRecord, originalUpperRecord = *upperRecord;
    for (auto head : heads)
        BOOST_REQUIRE(fixture.index->DeleteIndex(head, 2) == ErrorCode::Success);
    fixture.Wait();

    BOOST_REQUIRE(base->Reassign(&workspace, baseRecord, routes.front()) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(*baseRecord, originalBaseRecord);
    BOOST_REQUIRE(upper->Reassign(&workspace, upperRecord, heads.front()) == ErrorCode::VectorNotFound);
    BOOST_CHECK_EQUAL(*upperRecord, originalUpperRecord);
    fixture.Wait();
    for (auto head : heads)
        BOOST_REQUIRE(fixture.index->AddHeadIndex(fixture.vectors.data() + fixture.dimension * head,
            head, 0, fixture.dimension, 2, &workspace) == ErrorCode::Success);
    fixture.Wait();
    ByteArray vectors;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    BOOST_REQUIRE(upper->ExportRoutingVectors(routes, vectors, deadline) == ErrorCode::Success);
    std::vector<SizeType> users;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(0, users) == ErrorCode::Success);
    BOOST_REQUIRE(base->ExportRoutingVectors(users, vectors, deadline) == ErrorCode::Success);
    fixture.Check();
}

BOOST_AUTO_TEST_CASE(ShrinkRejectsBudgetAndIteratorWithoutPublication)
{
    FileLayerFixture fixture;
    BOOST_CHECK(fixture.index->ShrinkTopLayer() != ErrorCode::Success);
    fixture.Grow();
    const auto manifest = fixture.Manifest();
    const auto oldHead = fixture.index->GetMemoryIndex();
    fixture.Set("HeadMemoryTarget", "1");
    BOOST_CHECK(fixture.index->ShrinkTopLayer() == ErrorCode::MemoryOverFlow);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == oldHead);
    BOOST_CHECK_EQUAL(fixture.Manifest(), manifest);
    fixture.Set("HeadMemoryTarget", "2500");
    BOOST_CHECK(fixture.index->ShrinkTopLayer() == ErrorCode::MemoryOverFlow);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == oldHead);
    BOOST_CHECK_EQUAL(fixture.Manifest(), manifest);
    fixture.Check();
    fixture.Reload();
    fixture.Set("HeadMemoryTarget", "750000");
    fixture.Set("LayerResizeTimeoutSeconds", "1");
    auto iterator = fixture.index->GetIterator(fixture.vectors.data());
    BOOST_REQUIRE(iterator);
    BOOST_CHECK(fixture.index->ShrinkTopLayer() != ErrorCode::Success);
    BOOST_CHECK_EQUAL(fixture.Manifest(), manifest);
    auto result = iterator->Next(1);
    BOOST_REQUIRE(result);
    BOOST_CHECK_EQUAL(result->GetResult(0)->VID, 0);
    iterator->Close();
    BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
    BOOST_CHECK(fixture.index->GetLayerResizeStatus() == ErrorCode::Success);
    fixture.Check();
}

BOOST_AUTO_TEST_CASE(ShrinkValidatesCompleteVectorsAndDeduplicatesReplicas)
{
    FileLayerFixture fixture;
    fixture.Grow();
    fixture.Set("HeadMemoryTarget", "750000");
    std::vector<SizeType> heads;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(2, heads) == ErrorCode::Success);
    SPANN::ExtraWorkSpace workspace;
    fixture.index->InitWorkSpace(&workspace);
    auto upper = fixture.index->GetDiskIndex(1);
    std::vector<std::string> postings;
    const size_t recordSize = sizeof(SizeType) + 1 + fixture.dimension;
    for (SizeType head : heads) {
        std::string posting;
        BOOST_REQUIRE(upper->GetWritePosting(&workspace, head, posting) == ErrorCode::Success);
        postings.push_back(posting);
    }
    SizeType missing = 0;
    while (std::find(heads.begin(), heads.end(), missing) != heads.end()) ++missing;
    const auto manifest = fixture.Manifest();
    for (size_t i = 0; i < heads.size(); ++i) {
        std::string damaged;
        for (size_t offset = 0; offset < postings[i].size(); offset += recordSize) {
            SizeType vid;
            memcpy(&vid, postings[i].data() + offset, sizeof(vid));
            if (vid != missing) damaged.append(postings[i], offset, recordSize);
        }
        BOOST_REQUIRE(upper->GetWritePosting(&workspace, heads[i], damaged, true) == ErrorCode::Success);
    }
    BOOST_CHECK(fixture.index->ShrinkTopLayer() == ErrorCode::VectorNotFound);
    BOOST_CHECK_EQUAL(fixture.Manifest(), manifest);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
    for (size_t i = 0; i < heads.size(); ++i)
        BOOST_REQUIRE(upper->GetWritePosting(&workspace, heads[i], postings[i], true) == ErrorCode::Success);
    std::string posting = postings.front();
    std::string duplicate = posting.substr(0, recordSize);
    duplicate.back() ^= 1;
    posting += duplicate;
    BOOST_REQUIRE(upper->GetWritePosting(&workspace, heads.front(), posting, true) == ErrorCode::Success);
    BOOST_CHECK(fixture.index->ShrinkTopLayer() == ErrorCode::Fail);
    BOOST_CHECK_EQUAL(fixture.Manifest(), manifest);
    posting = postings.front() + '\0';
    BOOST_REQUIRE(upper->GetWritePosting(&workspace, heads.front(), posting, true) == ErrorCode::Success);
    BOOST_CHECK(fixture.index->ShrinkTopLayer() == ErrorCode::DiskIOFail);
    BOOST_CHECK_EQUAL(fixture.Manifest(), manifest);
    posting = postings.front();
    duplicate = posting.substr(0, recordSize);
    posting += duplicate;
    duplicate[sizeof(SizeType)] = 1;
    duplicate.back() ^= 1;
    posting += duplicate;
    SizeType deleted = 999;
    memcpy(duplicate.data(), &deleted, sizeof(deleted));
    duplicate[sizeof(SizeType)] = 0;
    posting += duplicate;
    BOOST_REQUIRE(upper->GetWritePosting(&workspace, heads.front(), posting, true) == ErrorCode::Success);
    BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
    BOOST_CHECK_EQUAL(fixture.index->GetMemoryIndex()->GetNumSamples(), 64);
    fixture.Check();
    fixture.Reload();
    fixture.Check();
}

BOOST_AUTO_TEST_CASE(ShrinkRepeatedTransitionsSupportQueriesUpdatesAndCopy)
{
    FileLayerFixture fixture;
    fixture.Grow();
    fixture.Grow();
    std::atomic_int failures{0}, searches{0};
    std::jthread reader([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            COMMON::QueryResultSet<uint8_t> query(fixture.vectors.data() + 10, 1);
            if (fixture.index->SearchIndex(query) != ErrorCode::Success ||
                query.GetResult(0)->VID != 5 || query.GetResult(0)->Dist != 0) ++failures;
            ++searches;
        }
    });
    fixture.Set("HeadMemoryTarget", "750000");
    BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
    BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
    fixture.Grow();
    fixture.Set("HeadMemoryTarget", "750000");
    BOOST_REQUIRE(fixture.index->ShrinkTopLayer() == ErrorCode::Success);
    reader.request_stop();
    reader.join();
    BOOST_CHECK_GT(searches.load(), 0);
    BOOST_CHECK_EQUAL(failures.load(), 0);
    uint8_t inserted[2] = {70, 140};
    BOOST_REQUIRE(fixture.index->AddIndex(inserted, 1, 2, nullptr, false, true) == ErrorCode::Success);
    fixture.Wait();
    COMMON::QueryResultSet<uint8_t> query(inserted, 1);
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(query.GetResult(0)->VID, 64);
    BOOST_REQUIRE(fixture.index->DeleteIndex(64) == ErrorCode::Success);
    const boost::filesystem::path destination(fixture.directory.string() + "_copy");
    BOOST_REQUIRE(fixture.index->SaveIndex(destination.string()) == ErrorCode::Success);
    fixture.index.reset();
    boost::filesystem::remove_all(fixture.directory);
    fixture.directory = destination;
    fixture.Reload();
    fixture.Check();
    query.Reset();
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_NE(query.GetResult(0)->VID, 64);
}

BOOST_AUTO_TEST_CASE(ShrinkAutomaticThresholdAndCooldown)
{
    FileLayerFixture fixture(1, true, 2, true);
    fixture.Grow();
    fixture.Set("LayerResizeCooldownSeconds", "0");
    fixture.NotifyResize();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
    BOOST_CHECK(fixture.index->GetLayerResizeStatus() == ErrorCode::Success);
    fixture.Set("HeadMemoryTarget", "750000");
    fixture.Set("HeadMemoryLowWatermark", "500000");
    fixture.Set("LayerResizeCooldownSeconds", "3600");
    fixture.NotifyResize();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
    fixture.Set("LayerResizeCooldownSeconds", "0");
    fixture.NotifyResize();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "1");
    BOOST_CHECK(fixture.index->GetLayerResizeStatus() == ErrorCode::Success);
    fixture.Check();
    fixture.Set("HeadMemoryLowWatermark", "1");
    fixture.Set("LayerResizeCooldownSeconds", "1");
    fixture.Grow();
    fixture.Set("HeadMemoryTarget", "750000");
    fixture.Set("HeadMemoryLowWatermark", "500000");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (fixture.index->GetParameter("Layers", "BuildSSDIndex") != "1" &&
        std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    fixture.Wait();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "1");
    fixture.Reload();
    fixture.Check();
}

BOOST_AUTO_TEST_CASE(ShrinkRejectsInvalidConfiguration)
{
    SPANN::Index<uint8_t> invalid;
    for (const auto& parameter : std::vector<std::pair<const char*, const char*>>{
        {"EnableDynamicLayerShrink", "invalid"}, {"HeadMemoryLowWatermark", "-1"},
        {"HeadMemoryLowWatermark", "18446744073709551616"}, {"HeadMemoryLowWatermark", "2KB"},
        {"LayerResizeCooldownSeconds", "soon"}}) {
        const auto previous = invalid.GetParameter(parameter.first, "BuildSSDIndex");
        BOOST_CHECK(invalid.SetParameter(parameter.first, parameter.second, "BuildSSDIndex") == ErrorCode::FailedParseValue);
        BOOST_CHECK_EQUAL(invalid.GetParameter(parameter.first, "BuildSSDIndex"), previous);
    }
    BOOST_REQUIRE(invalid.SetParameter("EnableDynamicLayerShrink", "true", "BuildSSDIndex") == ErrorCode::Success);
    uint8_t vectors[4] = {0, 0, 1, 1};
    BOOST_CHECK(invalid.BuildIndex(vectors, 2, 2) == ErrorCode::FailedParseValue);
    BOOST_REQUIRE(invalid.SetParameter("EnableDynamicLayers", "true", "BuildSSDIndex") == ErrorCode::Success);
    BOOST_REQUIRE(invalid.SetParameter("HeadMemoryHighWatermark", "10000", "BuildSSDIndex") == ErrorCode::Success);
    BOOST_REQUIRE(invalid.SetParameter("HeadMemoryTarget", "2000", "BuildSSDIndex") == ErrorCode::Success);
    BOOST_REQUIRE(invalid.SetParameter("HeadMemoryLowWatermark", "2000", "BuildSSDIndex") == ErrorCode::Success);
    BOOST_CHECK(invalid.BuildIndex(vectors, 2, 2) == ErrorCode::FailedParseValue);
    BOOST_REQUIRE(invalid.SetParameter("HeadMemoryLowWatermark", "1", "BuildSSDIndex") == ErrorCode::Success);
    BOOST_REQUIRE(invalid.SetParameter("LayerResizeCooldownSeconds", "-1", "BuildSSDIndex") == ErrorCode::Success);
    BOOST_CHECK(invalid.BuildIndex(vectors, 2, 2) == ErrorCode::FailedParseValue);
}

BOOST_AUTO_TEST_CASE(AutomaticSmallWorkload)
{
    RunAutomaticLayerWorkload(1);
}

BOOST_AUTO_TEST_CASE(AutomaticLargerWorkload, *boost::unit_test::disabled())
{
    RunAutomaticLayerWorkload(4);
}

BOOST_AUTO_TEST_CASE(Sift10MCheckpointReload, *boost::unit_test::disabled())
{
    const auto* directory = std::getenv("SPTAG_FILEIO_VERIFY_INDEX");
    const auto* queryPath = std::getenv("SPTAG_FILEIO_VERIFY_QUERIES");
    const auto* truthPath = std::getenv("SPTAG_FILEIO_VERIFY_TRUTH");
    BOOST_REQUIRE(directory);
    BOOST_REQUIRE(queryPath);
    BOOST_REQUIRE(truthPath);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex(directory, loaded) == ErrorCode::Success);
    auto index = std::dynamic_pointer_cast<SPANN::Index<uint8_t>>(loaded);
    BOOST_REQUIRE(index);
    BOOST_REQUIRE_EQUAL(index->GetNumSamples(), 10000000);
    BOOST_REQUIRE_EQUAL(index->GetFeatureDim(), 128);
    BOOST_REQUIRE_EQUAL(index->GetParameter("Layers", "BuildSSDIndex"), "2");
    BOOST_REQUIRE(index->GetLayerResizeStatus() == ErrorCode::Success);
    constexpr int queryCount = 1000;
    constexpr int k = 5;
    std::ifstream queries(queryPath, std::ios::binary);
    std::int32_t queryHeader[2] = {};
    BOOST_REQUIRE(queries.read(reinterpret_cast<char*>(queryHeader), sizeof(queryHeader)));
    BOOST_REQUIRE_GE(queryHeader[0], queryCount);
    BOOST_REQUIRE_EQUAL(queryHeader[1], 128);
    std::vector<uint8_t> vectors(queryCount * queryHeader[1]);
    BOOST_REQUIRE(queries.read(reinterpret_cast<char*>(vectors.data()), vectors.size()));
    std::ifstream truth(truthPath, std::ios::binary);
    std::int32_t truthHeader[2] = {};
    BOOST_REQUIRE(truth.read(reinterpret_cast<char*>(truthHeader), sizeof(truthHeader)));
    BOOST_REQUIRE_EQUAL(truthHeader[0], 12000);
    BOOST_REQUIRE_EQUAL(truthHeader[1], k);
    constexpr int prefixes = 6;
    std::vector<std::int32_t> ids(queryCount * k);
    std::vector<float> distances(queryCount * k);
    truth.seekg(sizeof(truthHeader) + (prefixes - 1) * queryCount * k * sizeof(std::int32_t));
    BOOST_REQUIRE(truth.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(ids[0])));
    truth.seekg(sizeof(truthHeader) + prefixes * queryCount * k * sizeof(std::int32_t) +
        (prefixes - 1) * queryCount * k * sizeof(float));
    BOOST_REQUIRE(truth.read(reinterpret_cast<char*>(distances.data()), distances.size() * sizeof(distances[0])));
    int hits = 0;
    for (int i = 0; i < queryCount; ++i) {
        COMMON::QueryResultSet<uint8_t> query(vectors.data() + i * queryHeader[1], k);
        BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
        std::vector<bool> matched(k, false);
        for (int j = 0; j < k; ++j) {
            const auto* result = query.GetResult(j);
            BOOST_REQUIRE_GE(result->VID, 0);
            BOOST_REQUIRE_LT(result->VID, 10000000);
            for (int expected = 0; expected < k; ++expected) {
                if (!matched[expected] && (result->VID == ids[i * k + expected] ||
                    std::abs(result->Dist - distances[i * k + expected]) < 0.001f)) {
                    matched[expected] = true;
                    ++hits;
                    break;
                }
            }
        }
    }
    const double recall = static_cast<double>(hits) / (queryCount * k);
    BOOST_TEST_MESSAGE("Reloaded 10M FILEIO checkpoint: layers=2, recall5@5=" << recall);
    BOOST_CHECK_GE(recall, 0.95);
}

BOOST_AUTO_TEST_SUITE_END()
