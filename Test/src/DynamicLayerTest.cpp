// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef TIKV

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/ResultIterator.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Test.h"

#include <boost/filesystem.hpp>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>

namespace
{
    using namespace SPTAG;

    class RoutingDB : public Helper::KeyValueIO
    {
        std::mutex m_mutex;
        std::map<SizeType, std::string> m_postings;
        std::map<std::string, std::string> m_versions;

    public:
        std::atomic_bool failWrites{false};
        void ShutDown() override {}
        bool Available() override { return true; }
        ErrorCode Checkpoint(std::string) override { return ErrorCode::Success; }

        std::map<SizeType, std::string> PostingSnapshot()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_postings;
        }

        ErrorCode Get(SizeType key, std::string* value, const std::chrono::microseconds&,
                      std::vector<Helper::AsyncReadRequest>*) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto found = m_postings.find(key);
            if (found == m_postings.end()) return ErrorCode::Key_NotFound;
            *value = found->second;
            return ErrorCode::Success;
        }
        ErrorCode Put(SizeType key, const std::string& value, const std::chrono::microseconds&,
                      std::vector<Helper::AsyncReadRequest>*) override
        {
            if (failWrites) return ErrorCode::DiskIOFail;
            std::lock_guard<std::mutex> lock(m_mutex);
            m_postings[key] = value;
            return ErrorCode::Success;
        }
        ErrorCode Get(const std::string& key, std::string* value, const std::chrono::microseconds&,
                      std::vector<Helper::AsyncReadRequest>*) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto found = m_versions.find(key);
            if (found == m_versions.end()) return ErrorCode::Key_NotFound;
            *value = found->second;
            return ErrorCode::Success;
        }
        ErrorCode Put(const std::string& key, const std::string& value, const std::chrono::microseconds&,
                      std::vector<Helper::AsyncReadRequest>*) override
        {
            if (failWrites) return ErrorCode::DiskIOFail;
            std::lock_guard<std::mutex> lock(m_mutex);
            m_versions[key] = value;
            return ErrorCode::Success;
        }
        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values,
                           const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* requests) override
        {
            values->clear();
            for (auto key : keys) {
                std::string value;
                auto ret = Get(key, &value, timeout, requests);
                if (ret != ErrorCode::Success && ret != ErrorCode::Key_NotFound) return ret;
                values->push_back(value);
            }
            return ErrorCode::Success;
        }
        ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values,
                           const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* requests) override
        {
            values->clear();
            for (const auto& key : keys) {
                std::string value;
                auto ret = Get(key, &value, timeout, requests);
                if (ret != ErrorCode::Success && ret != ErrorCode::Key_NotFound) return ret;
                values->push_back(value);
            }
            return ErrorCode::Success;
        }
        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<Helper::PageBuffer<uint8_t>>& buffers,
                           const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* requests) override
        {
            if (buffers.size() < keys.size()) return ErrorCode::DiskIOFail;
            std::vector<std::string> values;
            auto ret = MultiGet(keys, &values, timeout, requests);
            if (ret != ErrorCode::Success) return ret;
            for (size_t i = 0; i < values.size(); ++i) {
                buffers[i].ReservePageBuffer(values[i].size());
                if (!values[i].empty()) memcpy(buffers[i].GetBuffer(), values[i].data(), values[i].size());
                buffers[i].SetAvailableSize(values[i].size());
            }
            return ErrorCode::Success;
        }
        ErrorCode Merge(SizeType key, const std::string& value, const std::chrono::microseconds&,
                        std::vector<Helper::AsyncReadRequest>*, int& size) override
        {
            if (failWrites) return ErrorCode::DiskIOFail;
            std::lock_guard<std::mutex> lock(m_mutex);
            auto& posting = m_postings[key];
            posting += value;
            size = static_cast<int>(posting.size() / (sizeof(SizeType) + 3));
            return ErrorCode::Success;
        }
        ErrorCode MergeWithCAS(SizeType key, const std::string& value, const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* requests, int& size) override
        {
            return Merge(key, value, timeout, requests, size);
        }
        ErrorCode Delete(SizeType key) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_postings.erase(key);
            return ErrorCode::Success;
        }
        ErrorCode Check(SizeType key, std::vector<uint8_t>*) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_postings.count(key) ? ErrorCode::Success : ErrorCode::Key_NotFound;
        }
    };

    struct RoutingStorage
    {
        std::mutex mutex;
        std::map<std::string, std::shared_ptr<RoutingDB>> databases;
        bool failNewLayers = false;
    };

    class RoutingIndex : public SPANN::Index<uint8_t>
    {
        std::shared_ptr<RoutingStorage> m_storage;
    protected:
        std::shared_ptr<Helper::KeyValueIO> CreateTiKVLayerDB(const std::string& prefix) override
        {
            std::lock_guard<std::mutex> lock(m_storage->mutex);
            auto& db = m_storage->databases[prefix];
            if (!db) {
                db = std::make_shared<RoutingDB>();
                db->failWrites = m_storage->failNewLayers && prefix != "routing-test";
            }
            return db;
        }
    public:
        explicit RoutingIndex(std::shared_ptr<RoutingStorage> storage) : m_storage(std::move(storage)) {}
    };

    struct RoutingFixture
    {
        boost::filesystem::path directory =
            boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("sptag-routing-%%%%-%%%%");
        std::shared_ptr<RoutingStorage> storage = std::make_shared<RoutingStorage>();
        std::shared_ptr<RoutingIndex> index = std::make_shared<RoutingIndex>(storage);
        std::vector<uint8_t> vectors;

        explicit RoutingFixture(bool distributed = false, bool enabled = true, SizeType maxID = 10000)
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
            Set("Storage", "TIKVIO");
            Set("TiKVKeyPrefix", "routing-test");
            Set("DistributedVersionMap", distributed ? "true" : "false");
            Set("NumberOfThreads", "1");
            Set("SearchThreadNum", "2");
            Set("AppendThreadNum", "1");
            Set("Update", "true");
            Set("AsyncMergeInSearch", "false");
            Set("ExcludeHead", "false");
            Set("ReplicaCount", "2");
            Set("InternalResultNum", "64");
            Set("SearchInternalResultNum", "64");
            Set("MaxCheck", "256");
            Set("MaxID", std::to_string(maxID));
            Set("EnableDynamicLayers", enabled ? "true" : "false");
            Set("HeadMemoryHighWatermark", "1000000");
            Set("HeadMemoryTarget", "2000");
            Set("LayerResizeTimeoutSeconds", "1");
            vectors.resize(128);
            for (int i = 0; i < 64; ++i) {
                vectors[2 * i] = static_cast<uint8_t>(i);
                vectors[2 * i + 1] = static_cast<uint8_t>(2 * i);
            }
            BOOST_REQUIRE(index->BuildIndex(vectors.data(), 64, 2) == ErrorCode::Success);
            Wait();
        }
        ~RoutingFixture()
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
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!index->AllFinished() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            BOOST_REQUIRE(index->AllFinished());
        }
        void CheckQueries()
        {
            for (int i = 0; i < 64; ++i) {
                COMMON::QueryResultSet<uint8_t> query(vectors.data() + 2 * i, 1);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
                BOOST_CHECK_EQUAL(query.GetResult(0)->VID, i);
                BOOST_CHECK_EQUAL(query.GetResult(0)->Dist, 0);
            }
        }
        void Reload(bool disableAutomaticGrowth = false)
        {
            index.reset();
            index = std::make_shared<RoutingIndex>(storage);
            auto config = f_createIO();
            BOOST_REQUIRE(config->Initialize((directory / "indexloader.ini").string().c_str(), std::ios::in));
            Helper::IniReader reader;
            BOOST_REQUIRE(reader.LoadIni(config) == ErrorCode::Success);
            BOOST_REQUIRE(index->LoadConfig(reader) == ErrorCode::Success);
            if (disableAutomaticGrowth)
                Set("EnableDynamicLayers", "false");
            std::vector<std::shared_ptr<Helper::DiskIO>> streams;
            auto files = index->GetIndexFiles();
            for (const auto& file : *files) {
                auto stream = f_createIO();
                BOOST_REQUIRE(stream->Initialize((directory / file).string().c_str(), std::ios::binary | std::ios::in));
                streams.push_back(stream);
            }
            BOOST_REQUIRE(index->LoadIndexData(streams) == ErrorCode::Success);
            index->SetReady(true);
            Wait();
        }
    };
}

BOOST_AUTO_TEST_SUITE(DynamicLayerTest)

BOOST_AUTO_TEST_CASE(GrowthPreservesRoutingAndReload)
{
    for (bool distributed : {false, true}) {
        RoutingFixture fixture(distributed);
        fixture.CheckQueries();
        const auto oldMemory = fixture.index->GetHeadMemoryUsage();
        fixture.Set("HeadMemoryTarget", std::to_string(oldMemory / 2));
        auto base = fixture.storage->databases.at("routing-test")->PostingSnapshot();
        std::vector<SPTAG::SizeType> heads;
        BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(1, heads) == SPTAG::ErrorCode::Success);
        BOOST_REQUIRE(fixture.index->GrowTopLayer() == SPTAG::ErrorCode::Success);
        BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
        BOOST_CHECK_LE(fixture.index->GetHeadMemoryUsage(), oldMemory / 2);
        BOOST_CHECK(base == fixture.storage->databases.at("routing-test")->PostingSnapshot());
        std::vector<SPTAG::SizeType> demoted;
        BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(1, demoted) == SPTAG::ErrorCode::Success);
        std::sort(heads.begin(), heads.end());
        std::sort(demoted.begin(), demoted.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(heads.begin(), heads.end(), demoted.begin(), demoted.end());
        fixture.CheckQueries();
        fixture.Reload();
        fixture.CheckQueries();
        BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
        BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == SPTAG::ErrorCode::Success);
        fixture.Reload();
        fixture.CheckQueries();
        const boost::filesystem::path destination(fixture.directory.string() + "_copy");
        const auto saved = fixture.index->SaveIndex(destination.string());
        if (saved == ErrorCode::Success) {
            boost::filesystem::remove_all(fixture.directory);
            fixture.directory = destination;
        }
        else boost::filesystem::remove_all(destination);
        BOOST_REQUIRE(saved == ErrorCode::Success);
        fixture.Reload();
        fixture.CheckQueries();
    }
}

BOOST_AUTO_TEST_CASE(FailedBuildRetainsOldTopology)
{
    RoutingFixture fixture(true);
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == SPTAG::ErrorCode::Success);
    auto oldHead = fixture.index->GetMemoryIndex();
    fixture.storage->failNewLayers = true;
    BOOST_CHECK(fixture.index->GrowTopLayer() == SPTAG::ErrorCode::DiskIOFail);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == oldHead);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "1");
    fixture.CheckQueries();
    fixture.Reload();
    fixture.CheckQueries();
    fixture.storage->failNewLayers = false;
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == SPTAG::ErrorCode::Success);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
}

BOOST_AUTO_TEST_CASE(IteratorDefersGrowthWithoutInvalidation)
{
    RoutingFixture fixture;
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    auto iterator = fixture.index->GetIterator(fixture.vectors.data());
    BOOST_REQUIRE(iterator);
    BOOST_CHECK(fixture.index->SaveIndex(fixture.directory.string()) != SPTAG::ErrorCode::Success);
    BOOST_CHECK(fixture.index->GrowTopLayer() != SPTAG::ErrorCode::Success);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "1");
    auto results = iterator->Next(1);
    BOOST_REQUIRE(results);
    BOOST_CHECK_EQUAL(results->GetResult(0)->VID, 0);
    iterator->Close();
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == SPTAG::ErrorCode::Success);
    fixture.CheckQueries();
}

BOOST_AUTO_TEST_CASE(DisabledAndInvalidConfigurationsAreRejected)
{
    RoutingFixture fixture(false, false);
    BOOST_CHECK(fixture.index->GrowTopLayer() != SPTAG::ErrorCode::Success);
    fixture.CheckQueries();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "1");
    SPANN::Index<uint8_t> invalid;
    for (const auto& parameter : std::vector<std::pair<const char*, const char*>>{
        {"EnableDynamicLayers", "invalid"}, {"HeadMemoryHighWatermark", "-1"},
        {"HeadMemoryTarget", "18446744073709551616"}, {"HeadMemoryTarget", "2KB"},
        {"MaxLayers", "two"}, {"LayerResizeTimeoutSeconds", "soon"}}) {
        const auto previous = invalid.GetParameter(parameter.first, "BuildSSDIndex");
        BOOST_CHECK(invalid.SetParameter(parameter.first, parameter.second, "BuildSSDIndex") == ErrorCode::FailedParseValue);
        BOOST_CHECK_EQUAL(invalid.GetParameter(parameter.first, "BuildSSDIndex"), previous);
    }
    BOOST_REQUIRE(invalid.SetParameter("EnableDynamicLayers", "true", "BuildSSDIndex") == ErrorCode::Success);
    uint8_t vectors[4] = {0, 0, 1, 1};
    BOOST_CHECK(invalid.BuildIndex(vectors, 2, 2) == ErrorCode::FailedParseValue);
}

BOOST_AUTO_TEST_CASE(AutomaticThresholdAndRepeatedGrowth)
{
    RoutingFixture fixture;
    const auto oldMemory = fixture.index->GetHeadMemoryUsage();
    fixture.Set("HeadMemoryTarget", std::to_string(oldMemory / 2));
    fixture.Set("HeadMemoryHighWatermark", std::to_string(oldMemory));
    std::vector<SizeType> heads;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(1, heads) == ErrorCode::Success);
    BOOST_REQUIRE(!heads.empty());
    SPANN::ExtraWorkSpace workspace;
    BOOST_REQUIRE(fixture.index->AddHeadIndex(fixture.vectors.data() + 2 * heads.front(), heads.front(), 0, 2, 1, &workspace)
        == ErrorCode::Success);
    fixture.Wait();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
    BOOST_CHECK(fixture.index->GetLayerResizeStatus() == ErrorCode::Success);
    fixture.CheckQueries();
    fixture.Set("HeadMemoryHighWatermark", "1000000");
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "3");
    fixture.Reload();
    fixture.CheckQueries();
}

BOOST_AUTO_TEST_CASE(GrownIndexReloadsWithAutomaticGrowthDisabled)
{
    RoutingFixture fixture;
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    fixture.Reload(true);
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("EnableDynamicLayers", "BuildSSDIndex"), "false");
    fixture.Set("HeadMemoryHighWatermark", std::to_string(fixture.index->GetHeadMemoryUsage()));
    std::vector<SizeType> heads;
    BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(2, heads) == ErrorCode::Success);
    BOOST_REQUIRE(!heads.empty());
    SPANN::ExtraWorkSpace workspace;
    BOOST_REQUIRE(fixture.index->AddHeadIndex(fixture.vectors.data() + 2 * heads.front(), heads.front(), 0, 2, 2, &workspace)
        == ErrorCode::Success);
    fixture.Wait();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("Layers", "BuildSSDIndex"), "2");
    BOOST_CHECK(fixture.index->GrowTopLayer() != ErrorCode::Success);
    fixture.CheckQueries();
    fixture.Set("HeadMemoryTarget", "0");
    BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == ErrorCode::Success);
    fixture.Reload();
    fixture.CheckQueries();
    BOOST_CHECK_EQUAL(fixture.index->GetParameter("EnableDynamicLayers", "BuildSSDIndex"), "false");
}

BOOST_AUTO_TEST_CASE(ConcurrentQueriesAndUpdatesAfterGrowth)
{
    RoutingFixture fixture;
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    std::atomic_bool stop{false};
    std::atomic_int errors{0};
    std::atomic_int completed{0};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 2; ++thread) {
        readers.emplace_back([&]() {
            int i = 0;
            while (!stop) {
                COMMON::QueryResultSet<uint8_t> query(fixture.vectors.data() + 2 * (i % 64), 1);
                if (fixture.index->SearchIndex(query) != ErrorCode::Success || query.GetResult(0)->VID != i % 64)
                    ++errors;
                ++completed;
                ++i;
            }
        });
    }
    const auto growth = fixture.index->GrowTopLayer();
    stop = true;
    for (auto& reader : readers) reader.join();
    BOOST_REQUIRE(growth == ErrorCode::Success);
    BOOST_CHECK_EQUAL(errors.load(), 0);
    BOOST_CHECK_GT(completed.load(), 0);

    uint8_t inserted[2] = {70, 140};
    BOOST_REQUIRE(fixture.index->AddIndex(inserted, 1, 2, nullptr, false, true) == ErrorCode::Success);
    fixture.Wait();
    COMMON::QueryResultSet<uint8_t> query(inserted, 1);
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(query.GetResult(0)->VID, 64);
    BOOST_REQUIRE(fixture.index->DeleteIndex(64) == ErrorCode::Success);
    query.Reset();
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_NE(query.GetResult(0)->VID, 64);
    BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == ErrorCode::Success);
    fixture.Reload();
    query.Reset();
    BOOST_REQUIRE(fixture.index->SearchIndex(query) == ErrorCode::Success);
    BOOST_CHECK_NE(query.GetResult(0)->VID, 64);
    fixture.CheckQueries();
}

BOOST_AUTO_TEST_CASE(LimitsFailWithoutChangingTopology)
{
    RoutingFixture fixture(false, true, (std::numeric_limits<SizeType>::max)() / 2 + 1);
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_REQUIRE(fixture.index->GrowTopLayer() == ErrorCode::Success);
    auto head = fixture.index->GetMemoryIndex();
    fixture.Set("HeadMemoryTarget", std::to_string(fixture.index->GetHeadMemoryUsage() / 2));
    BOOST_CHECK(fixture.index->GrowTopLayer() == ErrorCode::Key_OverFlow);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == head);
    fixture.CheckQueries();
    fixture.Set("MaxLayers", "2");
    BOOST_CHECK(fixture.index->GrowTopLayer() != ErrorCode::Success);
    BOOST_CHECK(fixture.index->GetMemoryIndex() == head);
}

BOOST_AUTO_TEST_CASE(SparseLocalVersionBatchUsesGlobalIDs)
{
    COMMON::LocalVersionMap versions;
    SizeType vid = 12345;
    uint8_t version = 3;
    versions.SetVersion(vid, version);
    std::vector<uint8_t> results;
    versions.BatchGetVersions({vid, vid + 1, -1}, results);
    BOOST_REQUIRE_EQUAL(results.size(), 3);
    BOOST_CHECK_EQUAL(results[0], 3);
    BOOST_CHECK_EQUAL(results[1], 0xfe);
    BOOST_CHECK_EQUAL(results[2], 0xfe);
}

BOOST_AUTO_TEST_CASE(RealTiKVGrowthAndReload, *boost::unit_test::disabled())
{
    const auto* addresses = std::getenv("SPTAG_TIKV_TEST_PD");
    const auto* prefix = std::getenv("SPTAG_TIKV_TEST_PREFIX");
    const auto* root = std::getenv("SPTAG_TIKV_TEST_DIRECTORY");
    BOOST_REQUIRE(addresses && *addresses);
    BOOST_REQUIRE(prefix && *prefix);
    BOOST_REQUIRE(root && *root);

    for (bool distributed : {false, true}) {
        const auto directory = boost::filesystem::path(root) / (distributed ? "distributed" : "local");
        BOOST_REQUIRE(!boost::filesystem::exists(directory));
        boost::filesystem::create_directories(directory);
        auto index = std::make_shared<SPANN::Index<uint8_t>>();
        auto set = [&](const char* name, const std::string& value, const char* section = "BuildSSDIndex") {
            BOOST_REQUIRE(index->SetParameter(name, value.c_str(), section) == ErrorCode::Success);
        };
        auto wait = [&]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (!index->AllFinished() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            BOOST_REQUIRE(index->AllFinished());
        };
        set("IndexDirectory", directory.string(), "Base");
        set("DistCalcMethod", "L2", "Base");
        set("IndexAlgoType", "BKT", "Base");
        set("ValueType", "UInt8", "Base");
        set("isExecute", "true", "SelectHead");
        set("SelectHeadType", "Random", "SelectHead");
        set("Ratio", "1", "SelectHead");
        set("NumberOfThreads", "1", "SelectHead");
        set("isExecute", "true", "BuildHead");
        set("NumberOfThreads", "1", "BuildHead");
        set("NeighborhoodSize", "8", "BuildHead");
        set("RefineIterations", "1", "BuildHead");
        set("isExecute", "true");
        set("BuildSsdIndex", "true");
        set("Storage", "TIKVIO");
        set("TiKVPDAddresses", addresses);
        set("TiKVKeyPrefix", std::string(prefix) + (distributed ? "-distributed" : "-local"));
        set("DistributedVersionMap", distributed ? "true" : "false");
        set("ShareDB", "true");
        set("Layers", "1");
        set("NumberOfThreads", "1");
        set("SearchThreadNum", "2");
        set("AppendThreadNum", "1");
        set("Update", "true");
        set("AsyncMergeInSearch", "false");
        set("ExcludeHead", "false");
        set("ReplicaCount", "2");
        set("PostingVectorLimit", "0");
        set("PostingPageLimit", "1");
        set("SearchPostingPageLimit", "1");
        set("BufferLength", "1");
        set("InternalResultNum", "64");
        set("SearchInternalResultNum", "64");
        set("MaxCheck", "256");
        set("MaxID", "10000");
        set("EnableDynamicLayers", "true");
        set("HeadMemoryHighWatermark", "1000000");
        set("HeadMemoryTarget", "2000");
        set("MaxLayers", "2");
        set("LayerResizeTimeoutSeconds", "60");

        constexpr int dimension = 128;
        constexpr int totalCount = 136;
        std::vector<uint8_t> vectors(totalCount * dimension);
        for (int i = 0; i < totalCount; ++i) {
            vectors[dimension * i] = static_cast<uint8_t>(i);
            vectors[dimension * i + 1] = static_cast<uint8_t>(2 * i);
        }
        BOOST_REQUIRE(index->BuildIndex(vectors.data(), 64, dimension) == ErrorCode::Success);
        wait();
        const auto memory = index->GetHeadMemoryUsage();
        set("Ratio", "0.1", "SelectHead");
        set("HeadMemoryTarget", std::to_string(memory / 2));
        set("HeadMemoryHighWatermark", std::to_string(memory - 1));

        std::atomic_int errors{0};
        std::atomic_int completed{0};
        std::jthread reader([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                COMMON::QueryResultSet<uint8_t> query(vectors.data(), 1);
                if (index->SearchIndex(query) != ErrorCode::Success || query.GetResult(0)->VID != 0)
                    ++errors;
                ++completed;
            }
        });
        const auto inserted = index->AddIndex(vectors.data() + 64 * dimension, 64, dimension, nullptr, false, true);
        wait();
        reader.request_stop();
        reader.join();
        BOOST_REQUIRE(inserted == ErrorCode::Success);
        BOOST_REQUIRE_EQUAL(index->GetParameter("Layers", "BuildSSDIndex"), "2");
        BOOST_REQUIRE(index->GetLayerResizeStatus() == ErrorCode::Success);
        BOOST_CHECK_LE(index->GetHeadMemoryUsage(), memory / 2);
        BOOST_CHECK_EQUAL(errors.load(), 0);
        BOOST_CHECK_GT(completed.load(), 0);
        set("HeadMemoryHighWatermark", "1000000");
        BOOST_REQUIRE(index->AddIndex(vectors.data() + 128 * dimension, 8, dimension, nullptr, false, true) == ErrorCode::Success);
        wait();

        auto check = [&](bool deleted) {
            BOOST_CHECK_EQUAL(index->GetNumSamples(), totalCount);
            for (int i = 0; i < totalCount; ++i) {
                COMMON::QueryResultSet<uint8_t> query(vectors.data() + dimension * i, 1);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
                if (deleted && i == 64) {
                    BOOST_CHECK_NE(query.GetResult(0)->VID, 64);
                } else {
                    BOOST_CHECK_EQUAL(query.GetResult(0)->VID, i);
                    BOOST_CHECK_EQUAL(query.GetResult(0)->Dist, 0);
                }
            }
        };
        check(false);
        BOOST_REQUIRE(index->DeleteIndex(64) == ErrorCode::Success);
        check(true);
        const auto storagePrefixes = index->GetParameter("LayerStoragePrefixes", "BuildSSDIndex");
        BOOST_REQUIRE(!storagePrefixes.empty());
        BOOST_REQUIRE(index->SaveIndex(directory.string()) == ErrorCode::Success);
        index.reset();
        std::shared_ptr<VectorIndex> loaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(directory.string(), loaded) == ErrorCode::Success);
        index = std::dynamic_pointer_cast<SPANN::Index<uint8_t>>(loaded);
        BOOST_REQUIRE(index);
        wait();
        BOOST_REQUIRE_EQUAL(index->GetParameter("Layers", "BuildSSDIndex"), "2");
        BOOST_CHECK_EQUAL(index->GetParameter("LayerStoragePrefixes", "BuildSSDIndex"), storagePrefixes);
        check(true);
        BOOST_REQUIRE(index->Check() == ErrorCode::Success);
        BOOST_TEST_MESSAGE("Real TiKV automatic growth/reload passed: distributed=" << distributed
            << ", layers=2, samples=" << totalCount << ", concurrentQueries=" << completed.load());
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
