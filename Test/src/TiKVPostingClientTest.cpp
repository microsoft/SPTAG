// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/AsyncFileReader.h"
#define OpenTiKVPostingBackend OpenTiKVPostingBackendForClientTest
#include "../../AnnService/src/Core/SPANN/TiKVPostingBackend.cpp"
#undef OpenTiKVPostingBackend
#include "inc/Test.h"
#include <atomic>
#include <set>

using namespace SPTAG;
using namespace SPTAG::SPANN;

namespace
{
struct FakeRawCluster : pdpb::PD::Service, tikvpb::Tikv::Service
{
    std::string endpoint;
    std::unique_ptr<grpc::Server> server;
    std::atomic<int> regions{0}, stores{0}, batches{0}, active{0}, peak{0};
    std::atomic<int> delay{0}, firstFailureDelay{0};
    std::atomic<bool> transportFailure{false}, regionFailure{false}, invalid{false};
    std::mutex peersMutex;
    std::set<std::string> peers;

    FakeRawCluster()
    {
        grpc::ServerBuilder builder;
        int port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(static_cast<pdpb::PD::Service *>(this));
        builder.RegisterService(static_cast<tikvpb::Tikv::Service *>(this));
        server = builder.BuildAndStart();
        BOOST_REQUIRE(server);
        endpoint = "127.0.0.1:" + std::to_string(port);
    }

    ~FakeRawCluster()
    {
        server->Shutdown();
    }

    grpc::Status GetMembers(grpc::ServerContext *, const pdpb::GetMembersRequest *,
                            pdpb::GetMembersResponse *response) override
    {
        response->mutable_header()->set_cluster_id(7);
        response->mutable_leader()->add_client_urls(endpoint);
        return grpc::Status::OK;
    }

    grpc::Status GetRegion(grpc::ServerContext *, const pdpb::GetRegionRequest *request,
                           pdpb::GetRegionResponse *response) override
    {
        ++regions;
        const auto n = ReadNumber(request->region_key().data());
        response->mutable_header()->set_cluster_id(7);
        response->mutable_region()->set_id(n + 1);
        response->mutable_region()->set_start_key(Number(n));
        response->mutable_region()->set_end_key(Number(n + 1));
        response->mutable_region()->mutable_region_epoch()->set_version(1);
        response->mutable_leader()->set_store_id(1);
        return grpc::Status::OK;
    }

    grpc::Status GetStore(grpc::ServerContext *, const pdpb::GetStoreRequest *,
                          pdpb::GetStoreResponse *response) override
    {
        ++stores;
        response->mutable_header()->set_cluster_id(7);
        response->mutable_store()->set_address(endpoint);
        return grpc::Status::OK;
    }

    grpc::Status RawBatchGet(grpc::ServerContext *context, const kvrpcpb::RawBatchGetRequest *request,
                             kvrpcpb::RawBatchGetResponse *response) override
    {
        {
            std::lock_guard<std::mutex> lock(peersMutex);
            peers.insert(context->peer());
        }
        ++batches;
        int running = ++active, previous = peak;
        while (previous < running && !peak.compare_exchange_weak(previous, running))
        {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay.load()));
        --active;
        if (transportFailure.exchange(false))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(firstFailureDelay.load()));
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected transient read failure");
        }
        if (regionFailure.exchange(false))
        {
            response->mutable_region_error()->mutable_region_not_found()->set_region_id(request->context().region_id());
            return grpc::Status::OK;
        }
        if (invalid)
        {
            response->add_pairs()->set_key("unexpected");
            return grpc::Status::OK;
        }
        for (int i = request->keys_size() - 1; i >= 0; --i)
        {
            if (ReadNumber(request->keys(i).data()) == 31)
                continue;
            auto *pair = response->add_pairs();
            pair->set_key(request->keys(i));
            pair->set_value("V" + request->keys(i));
        }
        return grpc::Status::OK;
    }
};

struct ClientFixture
{
    FakeRawCluster cluster;
    std::shared_ptr<ManagedRawClient> client = std::make_shared<ManagedRawClient>(cluster.endpoint);
    std::vector<std::string> values;

    ClientFixture()
    {
        BOOST_REQUIRE(client->Open() == ErrorCode::Success);
    }

    ErrorCode Read(const std::vector<std::string> &keys, int milliseconds = 1000)
    {
        return client->MultiGet(keys, values, OperationDeadline(std::chrono::milliseconds(milliseconds)));
    }
};
} // namespace

BOOST_FIXTURE_TEST_SUITE(TiKVPostingClient, ClientFixture)

BOOST_AUTO_TEST_CASE(BatchedRegionsPreserveDuplicatesMissingKeysAndOrdering)
{
    cluster.delay = 10;
    std::vector<std::string> keys;
    for (int i = 0; i < 32; ++i)
    {
        keys.push_back(Number(i) + "a");
        keys.push_back(Number(i) + "b");
        keys.push_back(Number(i) + "a");
    }
    BOOST_REQUIRE(Read(keys, 3000) == ErrorCode::Success);
    BOOST_CHECK_GT(cluster.peak.load(), 1);
    BOOST_CHECK_LE(cluster.peak.load(), 16);
    for (size_t i = 0; i < keys.size(); ++i)
        BOOST_CHECK_EQUAL(values[i], ReadNumber(keys[i].data()) == 31 ? std::string() : "V" + keys[i]);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 32);
    BOOST_CHECK_EQUAL(cluster.stores.load(), 1);
    {
        std::lock_guard<std::mutex> lock(cluster.peersMutex);
        BOOST_CHECK_GT(cluster.peers.size(), 1);
        BOOST_CHECK_LE(cluster.peers.size(), 16);
    }
    const int discovered = cluster.regions;
    BOOST_REQUIRE(Read(keys, 3000) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.regions.load(), discovered);
}

BOOST_AUTO_TEST_CASE(ShortBudgetAllowsUsefulTransientRetry)
{
    const std::vector<std::string> keys{Number(0) + "a"};
    BOOST_REQUIRE(Read(keys) == ErrorCode::Success);
    const int before = cluster.batches;
    cluster.firstFailureDelay = 30;
    cluster.transportFailure = true;
    BOOST_REQUIRE(Read(keys, 50) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.batches.load(), before + 2);
}

BOOST_AUTO_TEST_CASE(RegionErrorsRefreshRouting)
{
    const std::vector<std::string> keys{Number(0) + "a"};
    BOOST_REQUIRE(Read(keys) == ErrorCode::Success);
    const int before = cluster.regions;
    const int storesBefore = cluster.stores;
    cluster.regionFailure = true;
    BOOST_REQUIRE(Read(keys) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.regions.load(), before + 1);
    BOOST_CHECK_EQUAL(cluster.stores.load(), storesBefore + 1);
}

BOOST_AUTO_TEST_CASE(DeadlineDoesNotDiscardHealthyRoute)
{
    const std::vector<std::string> keys{Number(0) + "a"};
    BOOST_REQUIRE(Read(keys) == ErrorCode::Success);
    const int before = cluster.regions;
    cluster.delay = 100;
    const auto started = std::chrono::steady_clock::now();
    BOOST_REQUIRE(Read(keys, 30) != ErrorCode::Success);
    BOOST_CHECK_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(250));
    cluster.delay = 0;
    BOOST_REQUIRE(Read(keys) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.regions.load(), before);
}

BOOST_AUTO_TEST_CASE(ViewRpcTimeoutStartsWhenEachRegionWaveIsSubmitted)
{
    ManagedTiKVView view(client, "");
    std::vector<SizeType> keys;
    for (int i = 0; i < 32; ++i)
        keys.push_back(i * 256);
    BOOST_REQUIRE(view.MultiGet(keys, &values, std::chrono::milliseconds(1000), nullptr) == ErrorCode::Success);
    cluster.delay = 40;
    const int before = cluster.batches;
    BOOST_REQUIRE(view.MultiGet(keys, &values, std::chrono::milliseconds(70), nullptr) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.batches.load(), before + 32);
}

BOOST_AUTO_TEST_CASE(ExplicitBatchDeadlineStillBoundsQueuedRegionWaves)
{
    std::vector<std::string> keys;
    for (int i = 0; i < 32; ++i)
        keys.push_back(Number(i) + "a");
    BOOST_REQUIRE(Read(keys, 3000) == ErrorCode::Success);
    cluster.delay = 20;
    const auto started = std::chrono::steady_clock::now();
    BOOST_REQUIRE(Read(keys, 30) != ErrorCode::Success);
    BOOST_CHECK_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(250));
}

BOOST_AUTO_TEST_CASE(InvalidResponseFailsAndViewReleasesCallerBuffers)
{
    cluster.invalid = true;
    BOOST_REQUIRE(Read({Number(0) + "a"}, 50) != ErrorCode::Success);
    cluster.invalid = false;
    ManagedTiKVView view(client, "");
    BOOST_REQUIRE(view.CompletesRequestsBeforeReturn());
    std::vector<SizeType> keys{0, 1, 1};
    std::vector<Helper::PageBuffer<uint8_t>> buffers(3);
    std::vector<Helper::AsyncReadRequest> requests(2);
    requests[0].m_callback = [](bool) { BOOST_FAIL("synchronous adapter submitted caller callback"); };
    BOOST_REQUIRE(view.MultiGet(keys, buffers, std::chrono::milliseconds(300), &requests) == ErrorCode::Success);
    cluster.delay = 100;
    const int before = cluster.batches;
    const auto started = std::chrono::steady_clock::now();
    BOOST_REQUIRE(view.MultiGet(keys, buffers, std::chrono::milliseconds(30), &requests) != ErrorCode::Success);
    BOOST_CHECK_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(250));
    BOOST_CHECK_LE(cluster.batches.load(), before + 4);
    buffers.clear();
    requests.clear();
    view.ShutDown();
    BOOST_REQUIRE(view.MultiGet(keys, &values, std::chrono::milliseconds(30), nullptr) != ErrorCode::Success);
}

BOOST_AUTO_TEST_SUITE_END()
