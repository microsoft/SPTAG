// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/AsyncFileReader.h"
#define OpenTiKVPostingBackend OpenTiKVPostingBackendForClientTest
#include "../../AnnService/src/Core/SPANN/TiKVPostingBackend.cpp"
#undef OpenTiKVPostingBackend
#include "inc/Test.h"
#include <atomic>
#include <fstream>
#include <future>
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
    std::atomic<int> mutations{0}, scans{0}, batchPuts{0}, batchDeletes{0};
    std::function<void(const pdpb::GetRegionRequest &, pdpb::GetRegionResponse &)> routeHook;
    std::function<bool(const kvrpcpb::RawBatchGetRequest &, kvrpcpb::RawBatchGetResponse &)> batchHook;
    std::function<grpc::Status(kvrpcpb::RawCASResponse &)> mutationHook;
    std::function<void(const kvrpcpb::RawGetRequest &, kvrpcpb::RawGetResponse &)> getHook;
    std::function<grpc::Status(const kvrpcpb::RawCASRequest &, kvrpcpb::RawCASResponse &)> casHook;
    std::function<grpc::Status(const kvrpcpb::RawBatchPutRequest &, kvrpcpb::RawBatchPutResponse &)> batchPutHook;
    std::function<grpc::Status(const kvrpcpb::RawBatchDeleteRequest &, kvrpcpb::RawBatchDeleteResponse &)>
        batchDeleteHook;
    std::function<void(const kvrpcpb::RawScanRequest &, kvrpcpb::RawScanResponse &)> scanHook;
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
        response->mutable_region()->add_peers()->set_id(1);
        response->mutable_region()->mutable_peers(0)->set_store_id(1);
        response->mutable_leader()->set_id(1);
        response->mutable_leader()->set_store_id(1);
        if (routeHook)
            routeHook(*request, *response);
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
        if (batchHook && batchHook(*request, *response))
            return grpc::Status::OK;
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

    grpc::Status RawCompareAndSwap(grpc::ServerContext *, const kvrpcpb::RawCASRequest *request,
                                   kvrpcpb::RawCASResponse *response) override
    {
        ++mutations;
        if (casHook)
            return casHook(*request, *response);
        if (mutationHook)
            return mutationHook(*response);
        response->set_succeed(true);
        return grpc::Status::OK;
    }

    grpc::Status RawGet(grpc::ServerContext *, const kvrpcpb::RawGetRequest *request,
                        kvrpcpb::RawGetResponse *response) override
    {
        response->set_not_found(true);
        if (getHook)
            getHook(*request, *response);
        return grpc::Status::OK;
    }

    grpc::Status RawScan(grpc::ServerContext *, const kvrpcpb::RawScanRequest *request,
                         kvrpcpb::RawScanResponse *response) override
    {
        ++scans;
        if (scanHook)
            scanHook(*request, *response);
        return grpc::Status::OK;
    }

    grpc::Status RawBatchPut(grpc::ServerContext *, const kvrpcpb::RawBatchPutRequest *request,
                             kvrpcpb::RawBatchPutResponse *response) override
    {
        ++batchPuts;
        if (batchPutHook)
            return batchPutHook(*request, *response);
        return grpc::Status::OK;
    }

    grpc::Status RawBatchDelete(grpc::ServerContext *, const kvrpcpb::RawBatchDeleteRequest *request,
                                kvrpcpb::RawBatchDeleteResponse *response) override
    {
        ++batchDeletes;
        if (batchDeleteHook)
            return batchDeleteHook(*request, *response);
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
    BOOST_CHECK_EQUAL(cluster.stores.load(), storesBefore);
}

BOOST_AUTO_TEST_CASE(ReadsAndCASRecoverAcrossProlongedRegionReadiness)
{
    const auto ready = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    cluster.batchHook = [ready](const auto &, auto &response) {
        if (std::chrono::steady_clock::now() >= ready)
            return false;
        response.mutable_region_error()->mutable_stale_command();
        return true;
    };
    cluster.mutationHook = [ready](auto &response) {
        if (std::chrono::steady_clock::now() < ready)
            response.mutable_region_error()->mutable_server_is_busy();
        else
            response.set_succeed(true);
        return grpc::Status::OK;
    };
    auto mutation = std::async(std::launch::async, [&] {
        bool swapped = false, absent = false;
        std::string actual;
        return client->CAS(Number(0) + "a", "new", true, "", &swapped, &absent, &actual,
                           OperationDeadline(std::chrono::seconds(2))) == ErrorCode::Success &&
               swapped;
    });
    const auto status = Read({Number(0) + "a"}, 2000);
    BOOST_CHECK(mutation.get());
    BOOST_REQUIRE(status == ErrorCode::Success);
    BOOST_CHECK_EQUAL(values.at(0), "V" + Number(0) + "a");
    BOOST_CHECK_GT(cluster.batches.load(), 4);
    BOOST_CHECK_GT(cluster.mutations.load(), 4);
    BOOST_CHECK(!client->HasUncertainMutation());
}

BOOST_AUTO_TEST_CASE(SplitHintsRegroupKeysWithoutWaitingForPD)
{
    cluster.routeHook = [](const auto &, auto &response) {
        response.mutable_region()->set_id(10);
        response.mutable_region()->set_start_key(Number(0));
        response.mutable_region()->set_end_key(Number(2));
    };
    std::atomic<bool> invalidGroup{false};
    cluster.batchHook = [&](const auto &request, auto &response) {
        if (request.context().region_epoch().version() == 1)
        {
            for (int i = 0; i < 2; ++i)
            {
                auto *region = response.mutable_region_error()->mutable_epoch_not_match()->add_current_regions();
                region->set_id(i == 0 ? 11 : 10);
                region->set_start_key(Number(i));
                region->set_end_key(Number(i + 1));
                region->mutable_region_epoch()->set_version(2);
                region->add_peers()->set_id(i + 1);
                region->mutable_peers(0)->set_store_id(1);
            }
            return true;
        }
        for (const auto &key : request.keys())
            if (request.context().region_id() != (ReadNumber(key.data()) == 0 ? 11 : 10))
                invalidGroup = true;
        return false;
    };
    const std::vector<std::string> keys{Number(0) + "a", Number(1) + "b", Number(0) + "a"};
    BOOST_REQUIRE(Read(keys, 500) == ErrorCode::Success);
    BOOST_CHECK(!invalidGroup);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 1);
    BOOST_CHECK_EQUAL(cluster.batches.load(), 3);
    for (size_t i = 0; i < keys.size(); ++i)
        BOOST_CHECK_EQUAL(values.at(i), "V" + keys[i]);
}

BOOST_AUTO_TEST_CASE(LeaderHintsAvoidRediscoveringTheOldLeader)
{
    cluster.routeHook = [](const auto &, auto &response) {
        auto *peer = response.mutable_region()->add_peers();
        peer->set_id(2);
        peer->set_store_id(2);
    };
    cluster.batchHook = [](const auto &request, auto &response) {
        if (request.context().peer().store_id() == 2)
            return false;
        auto *error = response.mutable_region_error()->mutable_not_leader();
        error->set_region_id(request.context().region_id());
        error->mutable_leader()->set_id(2);
        error->mutable_leader()->set_store_id(2);
        return true;
    };
    BOOST_REQUIRE(Read({Number(0) + "a"}, 500) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 1);
    BOOST_CHECK_EQUAL(cluster.stores.load(), 2);
}

BOOST_AUTO_TEST_CASE(LateErrorsDoNotInvalidateANewerRoute)
{
    std::promise<void> blocked, release;
    auto released = release.get_future().share();
    std::atomic<int> oldRequests{0};
    cluster.batchHook = [&](const auto &request, auto &response) {
        if (request.context().region_epoch().version() != 1)
            return false;
        if (oldRequests.fetch_add(1) == 0)
        {
            blocked.set_value();
            released.wait_for(std::chrono::seconds(2));
            response.mutable_region_error()->mutable_region_not_found()->set_region_id(1);
        }
        else
        {
            auto *region = response.mutable_region_error()->mutable_epoch_not_match()->add_current_regions();
            region->set_id(1);
            region->set_start_key(Number(0));
            region->set_end_key(Number(1));
            region->mutable_region_epoch()->set_version(2);
            region->add_peers()->set_id(1);
            region->mutable_peers(0)->set_store_id(1);
        }
        return true;
    };
    auto first = std::async(std::launch::async, [&] {
        std::vector<std::string> output;
        return client->MultiGet({Number(0) + "a"}, output, OperationDeadline(std::chrono::seconds(3)));
    });
    const bool waiting = blocked.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    const auto second = waiting ? Read({Number(0) + "b"}, 500) : ErrorCode::Fail;
    release.set_value();
    const auto firstStatus = first.get();
    BOOST_REQUIRE(waiting);
    BOOST_REQUIRE(second == ErrorCode::Success);
    BOOST_REQUIRE(firstStatus == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 1);
}

BOOST_AUTO_TEST_CASE(PersistentRegionErrorsRespectReadAndMutationDeadlines)
{
    cluster.batchHook = [](const auto &, auto &response) {
        response.mutable_region_error()->mutable_stale_command();
        return true;
    };
    cluster.mutationHook = [](auto &response) {
        response.mutable_region_error()->mutable_server_is_busy();
        return grpc::Status::OK;
    };
    const auto started = std::chrono::steady_clock::now();
    BOOST_CHECK(Read({Number(0) + "a"}, 100) != ErrorCode::Success);
    bool swapped = false, absent = false;
    std::string actual;
    BOOST_CHECK(client->CAS(Number(0) + "a", "new", true, "", &swapped, &absent, &actual,
                            OperationDeadline(std::chrono::milliseconds(100))) != ErrorCode::Success);
    BOOST_CHECK_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(600));
    BOOST_CHECK_GT(cluster.batches.load(), 1);
    BOOST_CHECK_GT(cluster.mutations.load(), 1);
    BOOST_CHECK(!client->HasUncertainMutation());
}

BOOST_AUTO_TEST_CASE(DelayedPDResponsesCannotOverwriteNewEpochs)
{
    std::promise<void> blocked, release;
    auto released = release.get_future().share();
    std::atomic<int> discoveries{0};
    std::atomic<bool> usedOldEpoch{false};
    cluster.routeHook = [&](const auto &, auto &response) {
        if (discoveries.fetch_add(1) == 0)
        {
            blocked.set_value();
            released.wait_for(std::chrono::seconds(2));
        }
        else
            response.mutable_region()->mutable_region_epoch()->set_version(2);
    };
    cluster.batchHook = [&](const auto &request, auto &) {
        if (request.context().region_epoch().version() != 2)
            usedOldEpoch = true;
        return false;
    };
    auto first = std::async(std::launch::async, [&] {
        std::vector<std::string> output;
        return client->MultiGet({Number(0) + "a"}, output, OperationDeadline(std::chrono::seconds(3)));
    });
    const bool waiting = blocked.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    const auto second = waiting ? Read({Number(0) + "b"}, 500) : ErrorCode::Fail;
    release.set_value();
    const auto firstStatus = first.get();
    BOOST_REQUIRE(waiting);
    BOOST_REQUIRE(second == ErrorCode::Success);
    BOOST_REQUIRE(firstStatus == ErrorCode::Success);
    BOOST_CHECK(!usedOldEpoch);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 2);
}

BOOST_AUTO_TEST_CASE(MutationsProbeNewRegionsBeforeResubmission)
{
    const auto ready = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    std::atomic<bool> earlyMutation{false};
    cluster.getHook = [ready](const auto &request, auto &response) {
        if (request.context().region_epoch().version() == 2 && std::chrono::steady_clock::now() < ready)
            response.mutable_region_error()->mutable_stale_command();
    };
    cluster.casHook = [&](const auto &request, auto &response) {
        if (request.context().region_epoch().version() == 1)
        {
            auto *region = response.mutable_region_error()->mutable_epoch_not_match()->add_current_regions();
            region->set_id(1);
            region->set_start_key(Number(0));
            region->set_end_key(Number(1));
            region->mutable_region_epoch()->set_version(2);
            region->add_peers()->set_id(1);
            region->mutable_peers(0)->set_store_id(1);
        }
        else
        {
            earlyMutation = std::chrono::steady_clock::now() < ready;
            response.set_succeed(true);
        }
        return grpc::Status::OK;
    };
    bool swapped = false, absent = false;
    std::string actual;
    BOOST_REQUIRE(client->CAS(Number(0) + "a", "new", true, "", &swapped, &absent, &actual,
                              OperationDeadline(std::chrono::seconds(2))) == ErrorCode::Success);
    BOOST_CHECK(swapped);
    BOOST_CHECK(!earlyMutation);
    BOOST_CHECK_EQUAL(cluster.mutations.load(), 2);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 1);
    BOOST_CHECK(!client->HasUncertainMutation());
}

BOOST_AUTO_TEST_CASE(OlderEpochErrorsDoNotDowngradeHealthyRouting)
{
    cluster.routeHook = [&](const auto &, auto &response) {
        response.mutable_region()->mutable_region_epoch()->set_version(cluster.regions.load() == 1 ? 2 : 1);
    };
    std::atomic<bool> sentOldHint{false}, downgraded{false};
    cluster.batchHook = [&](const auto &request, auto &response) {
        if (request.context().region_epoch().version() != 2)
            downgraded = true;
        if (sentOldHint.exchange(true))
            return false;
        auto *region = response.mutable_region_error()->mutable_epoch_not_match()->add_current_regions();
        region->set_id(1);
        region->set_start_key(Number(0));
        region->set_end_key(Number(1));
        region->mutable_region_epoch()->set_version(1);
        region->add_peers()->set_id(1);
        region->mutable_peers(0)->set_store_id(1);
        return true;
    };
    BOOST_REQUIRE(Read({Number(0) + "a"}, 500) == ErrorCode::Success);
    BOOST_CHECK(!downgraded);
    BOOST_CHECK_EQUAL(cluster.regions.load(), 1);
}

BOOST_AUTO_TEST_CASE(UncertainMutationTransportFailuresAreNeverReplayed)
{
    cluster.mutationHook = [](auto &) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "reply lost after possible commit");
    };
    client->WatchNamespace(Number(0));
    bool swapped = false, absent = false;
    std::string actual;
    BOOST_CHECK(client->CAS(Number(0) + "a", "new", true, "", &swapped, &absent, &actual,
                            OperationDeadline(std::chrono::seconds(1))) != ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.mutations.load(), 1);
    BOOST_CHECK(client->HasUncertainMutation());
}

BOOST_AUTO_TEST_CASE(StaleMutationCommandsAreUncertainAndNeverReplayed)
{
    cluster.mutationHook = [](auto &response) {
        response.mutable_region_error()->mutable_stale_command();
        return grpc::Status::OK;
    };
    bool swapped = false, absent = false;
    std::string actual;
    BOOST_CHECK(client->CAS(Number(0) + "a", "new", true, "", &swapped, &absent, &actual,
                            OperationDeadline(std::chrono::seconds(1))) != ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.mutations.load(), 1);
    BOOST_CHECK(client->HasUncertainMutation());
}

BOOST_AUTO_TEST_CASE(UnsupportedRegionErrorsAreNotRetried)
{
    cluster.batchHook = [](const auto &, auto &response) {
        response.mutable_region_error()->set_message("unsupported region error");
        return true;
    };
    cluster.mutationHook = [](auto &response) {
        response.mutable_region_error()->mutable_raft_entry_too_large()->set_region_id(1);
        return grpc::Status::OK;
    };
    BOOST_CHECK(Read({Number(0) + "a"}, 500) != ErrorCode::Success);
    bool swapped = false, absent = false;
    std::string actual;
    BOOST_CHECK(client->CAS(Number(0) + "a", "new", true, "", &swapped, &absent, &actual,
                            OperationDeadline(std::chrono::seconds(1))) != ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.batches.load(), 1);
    BOOST_CHECK_EQUAL(cluster.mutations.load(), 1);
}

BOOST_AUTO_TEST_CASE(ScansRecoverAcrossRegionReadiness)
{
    const auto ready = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    cluster.scanHook = [ready](const auto &, auto &response) {
        if (std::chrono::steady_clock::now() < ready)
            response.mutable_region_error()->mutable_stale_command();
    };
    BOOST_REQUIRE(client->Scan(Number(0), Number(1), [](const auto &, const auto &) {
        BOOST_FAIL("empty scan unexpectedly returned data");
        return ErrorCode::Fail;
    }) == ErrorCode::Success);
    BOOST_CHECK_GT(cluster.scans.load(), 4);
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

BOOST_AUTO_TEST_CASE(BatchPutAndDeleteGroupByRegionAndFailClosedOnUncertainTransport)
{
    std::atomic<int> putPairs{0}, deletedKeys{0};
    cluster.batchPutHook = [&](const auto &request, auto &) {
        BOOST_CHECK(request.for_cas());
        putPairs += request.pairs_size();
        return grpc::Status::OK;
    };
    cluster.batchDeleteHook = [&](const auto &request, auto &) {
        deletedKeys += request.keys_size();
        return grpc::Status::OK;
    };
    std::vector<std::pair<std::string, std::string>> pairs;
    std::vector<std::string> delKeys;
    for (int r = 0; r < 4; ++r)
        for (int i = 0; i < 10; ++i)
        {
            const std::string k = Number(r) + std::to_string(i);
            pairs.emplace_back(k, "val");
            delKeys.push_back(k);
        }
    BOOST_REQUIRE(client->BatchPut(pairs, OperationDeadline(std::chrono::seconds(2))) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.batchPuts.load(), 4);
    BOOST_CHECK_EQUAL(putPairs.load(), 40);
    BOOST_REQUIRE(client->BatchDelete(delKeys, OperationDeadline(std::chrono::seconds(2))) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.batchDeletes.load(), 4);
    BOOST_CHECK_EQUAL(deletedKeys.load(), 40);
    BOOST_CHECK(!client->HasUncertainMutation());

    client->WatchNamespace(Number(2));
    cluster.batchDeleteHook = [](const auto &, auto &) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "lost batch delete reply");
    };
    BOOST_CHECK(client->BatchDelete({Number(2) + "x"}, OperationDeadline(std::chrono::seconds(1))) !=
                ErrorCode::Success);
    BOOST_CHECK_EQUAL(cluster.batchDeletes.load(), 5);
    BOOST_CHECK(client->HasUncertainMutation());
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TiKVPostingClientIntegration)

BOOST_AUTO_TEST_CASE(ConcurrentReadsAndCASDuringRealSplits)
{
    const char *signal = std::getenv("SPTAG_TIKV_SPLIT_SIGNAL");
    if (!signal)
    {
        BOOST_TEST_MESSAGE("SKIPPED controlled TiKV splits: set SPTAG_TIKV_SPLIT_SIGNAL and use an owned cluster.");
        return;
    }
    const char *pd = std::getenv("SPTAG_TIKV_PD"), *ownedPrefix = std::getenv("SPTAG_TIKV_PREFIX");
    BOOST_REQUIRE(pd && ownedPrefix);
    ManagedRawClient client(pd);
    BOOST_REQUIRE(client.Open() == ErrorCode::Success);
    const std::string prefix = std::string(ownedPrefix) + "split-recovery/";
    client.WatchNamespace(prefix);
    const auto key = [&](int i, char kind) { return prefix + Number(i) + kind; };
    const std::string payload(1024, 'x');
    for (int i = 0; i < 256; ++i)
        BOOST_REQUIRE(client.Put(key(i, 'r'), payload, OperationDeadline(std::chrono::seconds(3))) ==
                      ErrorCode::Success);
    for (int i = 0; i < 2; ++i)
        BOOST_REQUIRE(client.Put(key(i * 128, 'w'), Number(0), OperationDeadline(std::chrono::seconds(3))) ==
                      ErrorCode::Success);
    std::atomic<uint64_t> reads{0}, writes{0}, errors{0};
    std::array<uint64_t, 2> committed{};
    const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.push_back(std::async(std::launch::async, [&, worker] {
            std::vector<std::string> keys, values;
            for (int i = 0; i < 32; ++i)
                keys.push_back(key(i * 8 + worker, 'r'));
            while (std::chrono::steady_clock::now() < stop && errors.load() == 0)
            {
                if (client.MultiGet(keys, values, OperationDeadline(std::chrono::seconds(3))) != ErrorCode::Success ||
                    values != std::vector<std::string>(keys.size(), payload))
                    ++errors;
                else
                    ++reads;
            }
        }));
    for (int worker = 0; worker < 2; ++worker)
        workers.push_back(std::async(std::launch::async, [&, worker] {
            uint64_t sequence = 0;
            while (std::chrono::steady_clock::now() < stop && errors.load() == 0)
            {
                bool swapped = false, absent = false;
                std::string actual;
                if (client.CAS(key(worker * 128, 'w'), Number(sequence + 1), false, Number(sequence), &swapped, &absent,
                               &actual, OperationDeadline(std::chrono::seconds(3))) != ErrorCode::Success ||
                    !swapped)
                    ++errors;
                else
                {
                    committed[worker] = ++sequence;
                    ++writes;
                }
            }
        }));
    {
        std::ofstream ready(signal);
        ready << prefix;
        BOOST_REQUIRE(ready.good());
    }
    for (auto &worker : workers)
        worker.get();
    BOOST_CHECK_EQUAL(errors.load(), 0);
    BOOST_CHECK(!client.HasUncertainMutation());
    BOOST_CHECK_GT(reads.load(), 0);
    BOOST_CHECK_GT(writes.load(), 0);
    for (int worker = 0; worker < 2; ++worker)
    {
        std::string value;
        BOOST_REQUIRE(client.Get(key(worker * 128, 'w'), value, OperationDeadline(std::chrono::seconds(3))) ==
                      ErrorCode::Success);
        BOOST_CHECK_EQUAL(value, Number(committed[worker]));
    }
    BOOST_TEST_MESSAGE("Controlled split recovery: batch_reads=" << reads << " committed_cas=" << writes
                                                                 << " errors=" << errors);
    for (int i = 0; i < 256; ++i)
        BOOST_REQUIRE(client.Delete(key(i, 'r'), OperationDeadline(std::chrono::seconds(3))) == ErrorCode::Success);
    for (int worker = 0; worker < 2; ++worker)
        BOOST_REQUIRE(client.Delete(key(worker * 128, 'w'), OperationDeadline(std::chrono::seconds(3))) ==
                      ErrorCode::Success);
}

BOOST_AUTO_TEST_SUITE_END()
