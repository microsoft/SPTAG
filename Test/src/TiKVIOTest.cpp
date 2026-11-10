// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef TIKV

#include "inc/Core/SPANN/ExtraTiKVController.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Test.h"

#include <boost/filesystem.hpp>
#include <map>
#include <mutex>
#include <optional>
#include <set>

namespace SPTAG::SPANN
{
    struct TiKVIOTestAccess
    {
        static std::weak_ptr<void> StubPool(TiKVIO& client, const std::string& address)
        {
            client.GetOrCreateStub(address);
            return client.m_storeStubs.at(address);
        }

        static std::size_t StubCount(TiKVIO& client, const std::string& address)
        {
            return client.m_storeStubs.at(address)->stubs.size();
        }
    };
}

namespace
{
    class TestPD : public pdpb::PD::Service
    {
        grpc::Status GetMembers(grpc::ServerContext*, const pdpb::GetMembersRequest*,
            pdpb::GetMembersResponse*) override
        {
            return grpc::Status::OK;
        }
    };

    class TestStore : public tikvpb::Tikv::Service
    {
        std::mutex m_peerMutex;
        std::set<std::string> m_peers;

    public:
        std::size_t ConnectionCount()
        {
            std::lock_guard<std::mutex> lock(m_peerMutex);
            return m_peers.size();
        }

        grpc::Status RawBatchGet(grpc::ServerContext* context, const kvrpcpb::RawBatchGetRequest* request,
            kvrpcpb::RawBatchGetResponse* response) override
        {
            {
                std::lock_guard<std::mutex> lock(m_peerMutex);
                m_peers.insert(context->peer());
            }
            for (const auto& key : request->keys()) {
                auto* pair = response->add_pairs();
                pair->set_key(key);
                pair->set_value("value");
            }
            return grpc::Status::OK;
        }
    };

    class VersionMapDB : public SPTAG::Helper::KeyValueIO
    {
    public:
        std::map<std::string, std::string> versions;
        std::map<SPTAG::SizeType, std::string> postings;

        void ShutDown() override {}
        bool Available() override { return true; }

        SPTAG::ErrorCode Get(const std::string& key, std::string* value,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            auto found = versions.find(key);
            if (found == versions.end()) return SPTAG::ErrorCode::Key_NotFound;
            *value = found->second;
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode Put(const std::string& key, const std::string& value,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            versions[key] = value;
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            values->clear();
            for (const auto& key : keys) {
                auto found = versions.find(key);
                values->push_back(found == versions.end() ? std::string() : found->second);
            }
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode Get(SPTAG::SizeType key, std::string* value,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            auto found = postings.find(key);
            if (found == postings.end()) return SPTAG::ErrorCode::Key_NotFound;
            *value = found->second;
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode Put(SPTAG::SizeType key, const std::string& value,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            postings[key] = value;
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode MultiGet(const std::vector<SPTAG::SizeType>& keys, std::vector<std::string>* values,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            values->clear();
            for (auto key : keys) {
                auto found = postings.find(key);
                values->push_back(found == postings.end() ? std::string() : found->second);
            }
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode Merge(SPTAG::SizeType key, const std::string& value,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*, int& size) override
        {
            auto& posting = postings[key];
            posting += value;
            size = static_cast<int>(posting.size());
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode Delete(SPTAG::SizeType key) override
        {
            return postings.erase(key) ? SPTAG::ErrorCode::Success : SPTAG::ErrorCode::Key_NotFound;
        }
    };

    struct CountFixture
    {
        boost::filesystem::path directory =
            boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("sptag-count-%%%%-%%%%");
        SPTAG::SPANN::Index<std::uint8_t> head;
        SPTAG::SPANN::Options options;
        std::shared_ptr<VersionMapDB> db = std::make_shared<VersionMapDB>();
        SPTAG::COMMON::TiKVVersionMap versions;
        SPTAG::SPANN::ExtraWorkSpace workspace;

        CountFixture()
        {
            boost::filesystem::create_directory(directory);
            auto set = [&](const char* name, const std::string& value, const char* section) {
                BOOST_REQUIRE(head.SetParameter(name, value.c_str(), section) == SPTAG::ErrorCode::Success);
            };
            set("IndexDirectory", directory.string(), "Base");
            set("DistCalcMethod", "L2", "Base");
            set("isExecute", "true", "SelectHead");
            set("SelectHeadType", "Random", "SelectHead");
            set("Ratio", "1", "SelectHead");
            set("NumberOfThreads", "1", "SelectHead");
            set("isExecute", "true", "BuildHead");
            set("NumberOfThreads", "1", "BuildHead");
            set("isExecute", "true", "BuildSSDIndex");
            set("BuildSsdIndex", "true", "BuildSSDIndex");
            set("NumberOfThreads", "1", "BuildSSDIndex");
            set("ShareDB", "false", "BuildSSDIndex");
            std::uint8_t vectors[20];
            for (int i = 0; i < 10; ++i) {
                vectors[2 * i] = static_cast<std::uint8_t>(i);
                vectors[2 * i + 1] = 0;
            }
            BOOST_REQUIRE(head.BuildIndex(vectors, 10, 2) == SPTAG::ErrorCode::Success);

            options.m_storage = SPTAG::Storage::TIKVIO;
            options.m_distributedVersionMap = true;
            options.m_dim = 2;
            options.m_replicaCount = 1;
            options.m_internalResultNum = 10;
            options.m_update = false;
            options.m_enableWAL = false;
            options.m_indexDirectory = directory.string();
            versions.SetDB(db);
            versions.SetLayer(0);
            versions.Initialize(10, 32, 1024);
            workspace.Initialize(64, 2, 10, 4 * SPTAG::PageSize, true, false);
        }

        ~CountFixture()
        {
            boost::filesystem::remove_all(directory);
        }

        std::unique_ptr<SPTAG::SPANN::ExtraDynamicSearcher<std::uint8_t>> Load()
        {
            auto searcher = std::make_unique<SPTAG::SPANN::ExtraDynamicSearcher<std::uint8_t>>(
                options, 0, &head, db);
            BOOST_REQUIRE(searcher->LoadIndex(options));
            return searcher;
        }

        void Insert(SPTAG::SPANN::ExtraDynamicSearcher<std::uint8_t>& searcher,
            SPTAG::SizeType begin, SPTAG::SizeType count)
        {
            auto bytes = SPTAG::ByteArray::Alloc(count * 2);
            for (SPTAG::SizeType i = 0; i < count; ++i) {
                bytes.Data()[2 * i] = static_cast<std::uint8_t>(begin + i);
                bytes.Data()[2 * i + 1] = 0;
            }
            std::shared_ptr<SPTAG::VectorSet> vectors = std::make_shared<SPTAG::BasicVectorSet>(
                bytes, SPTAG::VectorValueType::UInt8, 2, count);
            BOOST_REQUIRE(searcher.AddIndex(&workspace, vectors, begin, true) == SPTAG::ErrorCode::Success);
        }
    };

    struct FileIOBuildFixture
    {
        boost::filesystem::path directory =
            boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("sptag-fileio-%%%%-%%%%");
        std::shared_ptr<SPTAG::SPANN::Index<std::uint8_t>> index =
            std::make_shared<SPTAG::SPANN::Index<std::uint8_t>>();
        std::uint8_t vectors[128];

        explicit FileIOBuildFixture(int layers)
        {
            boost::filesystem::create_directory(directory);
            auto set = [&](const char* name, const std::string& value, const char* section) {
                BOOST_REQUIRE(index->SetParameter(name, value.c_str(), section) == SPTAG::ErrorCode::Success);
            };
            set("IndexDirectory", directory.string(), "Base");
            set("DistCalcMethod", "L2", "Base");
            set("IndexAlgoType", "BKT", "Base");
            set("ValueType", "UInt8", "Base");
            set("isExecute", "true", "SelectHead");
            set("SelectHeadType", "Random", "SelectHead");
            set("Ratio", "0.5", "SelectHead");
            set("NumberOfThreads", "1", "SelectHead");
            set("isExecute", "true", "BuildHead");
            set("NumberOfThreads", "1", "BuildHead");
            set("NeighborhoodSize", "8", "BuildHead");
            set("RefineIterations", "1", "BuildHead");
            set("isExecute", "true", "BuildSSDIndex");
            set("BuildSsdIndex", "true", "BuildSSDIndex");
            set("Storage", "FILEIO", "BuildSSDIndex");
            set("Layers", std::to_string(layers), "BuildSSDIndex");
            set("NumberOfThreads", "1", "BuildSSDIndex");
            set("SearchThreadNum", "1", "BuildSSDIndex");
            set("AppendThreadNum", "1", "BuildSSDIndex");
            set("StartFileSizeGB", "1", "BuildSSDIndex");
            set("Update", "true", "BuildSSDIndex");
            set("AsyncMergeInSearch", "false", "BuildSSDIndex");
            set("ExcludeHead", "false", "BuildSSDIndex");
            set("ReplicaCount", "2", "BuildSSDIndex");
            for (int i = 0; i < 64; ++i) {
                vectors[2 * i] = static_cast<std::uint8_t>(i);
                vectors[2 * i + 1] = static_cast<std::uint8_t>(2 * i);
            }
        }

        ~FileIOBuildFixture()
        {
            index.reset();
            boost::filesystem::remove_all(directory);
        }

        void Check()
        {
            BOOST_REQUIRE_EQUAL(index->GetNumSamples(), 64);
            for (int i = 0; i < 64; ++i) {
                SPTAG::COMMON::QueryResultSet<std::uint8_t> query(vectors + 2 * i, 1);
                BOOST_REQUIRE(index->SearchIndex(query) == SPTAG::ErrorCode::Success);
                BOOST_CHECK_EQUAL(query.GetResult(0)->VID, i);
                BOOST_CHECK_EQUAL(query.GetResult(0)->Dist, 0);
            }
        }
    };
}

BOOST_AUTO_TEST_SUITE(TiKVIOTest)

BOOST_AUTO_TEST_CASE(StubPoolUsesIndependentConnections)
{
    TestPD pd;
    TestStore store;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&pd);
    builder.RegisterService(&store);
    auto server = builder.BuildAndStart();
    BOOST_REQUIRE(server);
    const std::string address = "127.0.0.1:" + std::to_string(port);
    SPTAG::SPANN::TiKVIO client(address, "test", false, 1);
    BOOST_REQUIRE(client.Available());
    SPTAG::SPANN::TiKVIOTestAccess::StubPool(client, address);
    const auto count = SPTAG::SPANN::TiKVIOTestAccess::StubCount(client, address);
    for (std::size_t i = 0; i < count; ++i) {
        std::vector<std::pair<size_t, std::string>> keys = {{0, "test_key"}};
        std::vector<std::string> values(1);
        auto batch = std::make_shared<SPTAG::SPANN::TiKVIO::AsyncBatch>();
        std::atomic<int> result{0};
        batch->Add(1);
        client.AsyncRawBatchGetPrefixed(address, kvrpcpb::Context(), keys,
            &values, batch, &result, std::chrono::seconds(5));
        batch->Wait();
        BOOST_REQUIRE_EQUAL(result.load(), 1);
        BOOST_REQUIRE_EQUAL(values[0], "value");
    }
    BOOST_CHECK_EQUAL(store.ConnectionCount(), count);
}

BOOST_AUTO_TEST_CASE(AsyncReadsSurviveClientAddressReuse)
{
    TestPD pd;
    TestStore store;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&pd);
    builder.RegisterService(&store);
    auto server = builder.BuildAndStart();
    BOOST_REQUIRE(server);
    const std::string address = "127.0.0.1:" + std::to_string(port);

    std::optional<SPTAG::SPANN::TiKVIO> client;
    for (int generation = 0; generation < 8; ++generation) {
        client.emplace(address, "test", false, 1);
        BOOST_REQUIRE(client->Available());
        auto pool = SPTAG::SPANN::TiKVIOTestAccess::StubPool(*client, address);
        for (int request = 0; request < 192; ++request) {
            std::vector<std::pair<size_t, std::string>> keys = {{0, "test_key"}};
            std::vector<std::string> values(1);
            auto batch = std::make_shared<SPTAG::SPANN::TiKVIO::AsyncBatch>();
            std::atomic<int> result{0};
            batch->Add(1);
            client->AsyncRawBatchGetPrefixed(address, kvrpcpb::Context(), keys,
                &values, batch, &result, std::chrono::seconds(1));
            batch->Wait();
            BOOST_REQUIRE_EQUAL(batch->Failures(), 0);
            BOOST_REQUIRE_EQUAL(result.load(), 1);
            BOOST_REQUIRE_EQUAL(values[0], "value");
        }
        client.reset();
        BOOST_REQUIRE(pool.expired());
    }
}

BOOST_FIXTURE_TEST_CASE(InsertedVIDsGrowCountAcrossReloads, CountFixture)
{
    SPTAG::SizeType existing = 3;
    std::uint8_t existingVersion = 7;
    versions.SetVersion(existing, existingVersion);
    BOOST_REQUIRE(versions.Delete(4));
    auto searcher = Load();

    for (SPTAG::SizeType begin : {10, 12}) {
        Insert(*searcher, begin, 2);
        BOOST_CHECK_EQUAL(searcher->GetNumSamples(), begin + 2);
        std::vector<SPTAG::SizeType> contained;
        BOOST_REQUIRE(searcher->GetContainedIDs(contained) == SPTAG::ErrorCode::Success);
        BOOST_CHECK_EQUAL(contained.size(), begin + 1);
        BOOST_CHECK(std::find(contained.begin(), contained.end(), begin) != contained.end());
        BOOST_CHECK(std::find(contained.begin(), contained.end(), begin + 1) != contained.end());
        BOOST_REQUIRE(searcher->Checkpoint(directory.string()) == SPTAG::ErrorCode::Success);
        searcher.reset();
        searcher = Load();
        BOOST_CHECK_EQUAL(searcher->GetNumSamples(), begin + 2);
        BOOST_REQUIRE(searcher->GetContainedIDs(contained) == SPTAG::ErrorCode::Success);
        BOOST_CHECK_EQUAL(contained.size(), begin + 1);
    }

    Insert(*searcher, 3, 1);
    BOOST_CHECK_EQUAL(versions.GetVersion(3), existingVersion);
    Insert(*searcher, 4, 1);
    BOOST_CHECK_EQUAL(versions.GetVersion(4), 0xff);
    BOOST_CHECK_EQUAL(searcher->GetNumSamples(), 14);
    BOOST_REQUIRE(searcher->Checkpoint(directory.string()) == SPTAG::ErrorCode::Success);
    searcher.reset();
    searcher = Load();
    BOOST_CHECK_EQUAL(searcher->GetNumSamples(), 14);
    std::vector<SPTAG::SizeType> contained;
    BOOST_REQUIRE(searcher->GetContainedIDs(contained) == SPTAG::ErrorCode::Success);
    BOOST_CHECK_EQUAL(contained.size(), 14);
    BOOST_CHECK_EQUAL(versions.GetVersion(3), existingVersion);

    std::map<SPTAG::SizeType, std::uint8_t> stored;
    constexpr std::size_t recordSize = sizeof(SPTAG::SizeType) + sizeof(std::uint8_t) + 2;
    for (const auto& posting : db->postings) {
        BOOST_REQUIRE_EQUAL(posting.second.size() % recordSize, 0);
        for (std::size_t offset = 0; offset < posting.second.size(); offset += recordSize) {
            SPTAG::SizeType vid;
            std::memcpy(&vid, posting.second.data() + offset, sizeof(vid));
            stored[vid] = static_cast<std::uint8_t>(posting.second[offset + sizeof(vid)]);
            BOOST_CHECK_EQUAL(static_cast<std::uint8_t>(posting.second[offset + sizeof(vid) + 1]), vid);
        }
    }
    BOOST_CHECK_EQUAL(stored.size(), 6);
    BOOST_CHECK_EQUAL(stored.at(3), existingVersion);
    for (SPTAG::SizeType vid : {4, 10, 11, 12, 13}) BOOST_CHECK_EQUAL(stored.at(vid), 0xff);
}

BOOST_FIXTURE_TEST_CASE(LocalSparseVIDKeepsExistingVersion, CountFixture)
{
    options.m_storage = SPTAG::Storage::FILEIO;
    options.m_ssdIndex = "count-fixture-no-static-postings";
    SPTAG::COMMON::LocalVersionMap local;
    SPTAG::SizeType existing = 3, other = 8;
    std::uint8_t version = 7;
    local.SetVersion(existing, version);
    local.SetVersion(other, local.Default());
    const auto filename = (directory / (options.m_deleteIDFile + "_0")).string();
    BOOST_REQUIRE(local.Save(filename) == SPTAG::ErrorCode::Success);
    auto searcher = Load();
    BOOST_REQUIRE_EQUAL(searcher->GetNumSamples(), 2);
    Insert(*searcher, existing, 1);
    BOOST_CHECK_EQUAL(searcher->GetNumSamples(), 2);
    BOOST_REQUIRE(searcher->Checkpoint(directory.string()) == SPTAG::ErrorCode::Success);
    SPTAG::COMMON::LocalVersionMap reloaded;
    BOOST_REQUIRE(reloaded.Load(filename, 32, 1024) == SPTAG::ErrorCode::Success);
    BOOST_CHECK_EQUAL(reloaded.GetVersion(existing), version);
}

BOOST_AUTO_TEST_CASE(FileIOBuildInitializesLocalVersionsWithTiKVEnabled)
{
    for (int layers : {1, 2}) {
        FileIOBuildFixture fixture(layers);
        BOOST_REQUIRE(fixture.index->BuildIndex(fixture.vectors, 64, 2) == SPTAG::ErrorCode::Success);
        fixture.Check();
        BOOST_REQUIRE(fixture.index->SaveIndex(fixture.directory.string()) == SPTAG::ErrorCode::Success);
        for (int layer = 0; layer < layers; ++layer) {
            SPTAG::COMMON::LocalVersionMap versions;
            const auto filename = (fixture.directory / ("DeletedIDs.bin_" + std::to_string(layer))).string();
            BOOST_REQUIRE(versions.Load(filename, 32, 1024) == SPTAG::ErrorCode::Success);
            std::vector<SPTAG::SizeType> expected, actual;
            BOOST_REQUIRE(fixture.index->GetHeadIndexMapping(layer, expected) == SPTAG::ErrorCode::Success);
            BOOST_REQUIRE(versions.GetContainedIDs(actual) == SPTAG::ErrorCode::Success);
            std::sort(expected.begin(), expected.end());
            std::sort(actual.begin(), actual.end());
            BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.end(), expected.begin(), expected.end());
            BOOST_CHECK_EQUAL(versions.Count(), expected.size());
            for (auto vid : expected) BOOST_CHECK_EQUAL(versions.GetVersion(vid), 0xff);
            if (layer > 0) BOOST_CHECK_LT(expected.size(), 64);
        }
        fixture.index.reset();
        std::shared_ptr<SPTAG::VectorIndex> reloaded;
        BOOST_REQUIRE(SPTAG::VectorIndex::LoadIndex(fixture.directory.string(), reloaded) == SPTAG::ErrorCode::Success);
        fixture.index = std::dynamic_pointer_cast<SPTAG::SPANN::Index<std::uint8_t>>(reloaded);
        BOOST_REQUIRE(fixture.index);
        reloaded.reset();
        fixture.Check();
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
