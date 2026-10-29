// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef TIKV

#include "inc/Core/SPANN/ExtraTiKVController.h"
#include "inc/Test.h"

#include <optional>

namespace SPTAG::SPANN
{
    struct TiKVIOTestAccess
    {
        static std::weak_ptr<void> StubPool(TiKVIO& client, const std::string& address)
        {
            client.GetOrCreateStub(address);
            return client.m_storeStubs.at(address);
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
        grpc::Status RawBatchGet(grpc::ServerContext*, const kvrpcpb::RawBatchGetRequest* request,
            kvrpcpb::RawBatchGetResponse* response) override
        {
            for (const auto& key : request->keys()) {
                auto* pair = response->add_pairs();
                pair->set_key(key);
                pair->set_value("value");
            }
            return grpc::Status::OK;
        }
    };
}

BOOST_AUTO_TEST_SUITE(TiKVIOTest)

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

BOOST_AUTO_TEST_SUITE_END()

#endif
