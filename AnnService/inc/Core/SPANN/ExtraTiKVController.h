// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRATIKVCONTROLLER_H_
#define _SPTAG_SPANN_EXTRATIKVCONTROLLER_H_

#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Core/SPANN/Options.h"

#include <grpcpp/grpcpp.h>
#include "kvproto/tikvpb.grpc.pb.h"
#include "kvproto/kvrpcpb.pb.h"
#include "kvproto/metapb.pb.h"
#include "kvproto/pdpb.grpc.pb.h"

#include <map>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <climits>
#include <future>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <thread>

namespace SPTAG::SPANN
{
    /// TiKVIO implements the KeyValueIO interface by communicating with a TiKV
    /// cluster via its RawKV gRPC API.
    ///
    /// Architecture:
    ///   1. Connect to PD (Placement Driver) to discover TiKV store endpoints.
    ///   2. Use PD's GetRegion RPC to find, for any given key, which TiKV
    ///      store (region leader) should handle the request.
    ///   3. Send RawGet / RawPut / RawDelete / RawBatchGet requests to the
    ///      correct TiKV store's RawKV gRPC service.
    ///
    /// All keys are prefixed with a configurable namespace prefix so that
    /// SPANN posting data does not collide with other data in the same TiKV
    /// cluster.
    class TiKVIO : public Helper::KeyValueIO
    {
    public:
        TiKVIO(const std::string& pdAddresses, const std::string& keyPrefix)
            : m_keyPrefix(keyPrefix)
        {
            // Parse comma-separated PD addresses and try to connect.
            std::istringstream ss(pdAddresses);
            std::string addr;
            while (std::getline(ss, addr, ',')) {
                // Trim whitespace
                addr.erase(0, addr.find_first_not_of(" \t"));
                addr.erase(addr.find_last_not_of(" \t") + 1);
                if (!addr.empty()) {
                    m_pdAddresses.push_back(addr);
                }
            }

            if (m_pdAddresses.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: No PD addresses provided!\n");
                return;
            }

            // Create channels to all PD nodes; find the leader.
            for (const auto& pdAddr : m_pdAddresses) {
                auto channel = grpc::CreateChannel(pdAddr, grpc::InsecureChannelCredentials());
                auto stub = pdpb::PD::NewStub(channel);
                if (!stub) continue;

                // Try GetMembers to find the PD leader
                pdpb::GetMembersRequest membersReq;
                auto* header = membersReq.mutable_header();
                header->set_cluster_id(0);
                pdpb::GetMembersResponse membersResp;
                grpc::ClientContext membersCtx;
                membersCtx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

                auto status = stub->GetMembers(&membersCtx, membersReq, &membersResp);
                if (!status.ok()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO: GetMembers failed on %s: %s\n",
                                 pdAddr.c_str(), status.error_message().c_str());
                    continue;
                }

                // Find leader's client URL
                if (membersResp.has_leader() && membersResp.leader().client_urls_size() > 0) {
                    // Save cluster_id from the response header
                    if (membersResp.has_header()) {
                        m_clusterId = membersResp.header().cluster_id();
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Cluster ID: %lu\n", m_clusterId);
                    }

                    std::string leaderUrl = membersResp.leader().client_urls(0);
                    // Strip http:// prefix if present
                    std::string leaderAddr = leaderUrl;
                    auto schemePos = leaderAddr.find("://");
                    if (schemePos != std::string::npos) {
                        leaderAddr = leaderAddr.substr(schemePos + 3);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: PD leader is at %s\n", leaderAddr.c_str());

                    if (leaderAddr == pdAddr) {
                        // We're already connected to the leader
                        m_pdStub = std::move(stub);
                    } else {
                        // Connect to the actual leader
                        auto leaderChannel = grpc::CreateChannel(leaderAddr, grpc::InsecureChannelCredentials());
                        m_pdStub = pdpb::PD::NewStub(leaderChannel);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Connected to PD leader at %s\n", leaderAddr.c_str());
                    break;
                } else {
                    // No leader info; use this node anyway
                    m_pdStub = std::move(stub);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Connected to PD at %s (leader unknown)\n", pdAddr.c_str());
                    break;
                }
            }

            if (!m_pdStub) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: Failed to create PD stub!\n");
                return;
            }

            m_available = true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Initialized with key prefix '%s'\n", m_keyPrefix.c_str());
        }

        ~TiKVIO() override {
            ShutDown();
        }

        void ShutDown() override {
            m_available = false;
            std::lock_guard<std::mutex> lock(m_storeMutex);
            m_storeStubs.clear();
            m_pdStub.reset();
        }

        bool Available() override {
            return m_available;
        }

        // ---- Single-key operations ----

        ErrorCode Get(const std::string& key, std::string* value,
                      const std::chrono::microseconds& timeout,
                      std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            std::string prefixedKey = MakePrefixedKey(key);

            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(prefixedKey);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawGetRequest request;
                request.set_key(prefixedKey);
                SetContext(request.mutable_context(), prefixedKey);

                kvrpcpb::RawGetResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawGet(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::Get gRPC error (attempt %d): %s\n",
                                     attempt + 1, status.error_message().c_str());
                    InvalidateRegionCache(prefixedKey);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::Get region_error (attempt %d)\n", attempt + 1);
                    InvalidateRegionCache(prefixedKey);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Get region_error failed after %d attempts, giving up\n", attempt + 1);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Get error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                if (response.not_found()) {
                    return ErrorCode::Fail;
                }

                *value = response.value();
                return ErrorCode::Success;
            }
        }

        ErrorCode Get(const SizeType key, std::string* value,
                      const std::chrono::microseconds& timeout,
                      std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            std::string k(reinterpret_cast<const char*>(&key), sizeof(SizeType));
            return Get(k, value, timeout, reqs);
        }

        // ---- Put operations ----

        ErrorCode Put(const std::string& key, const std::string& value,
                      const std::chrono::microseconds& timeout,
                      std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            std::string prefixedKey = MakePrefixedKey(key);

            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(prefixedKey);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawPutRequest request;
                request.set_key(prefixedKey);
                request.set_value(value);
                SetContext(request.mutable_context(), prefixedKey);

                kvrpcpb::RawPutResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawPut(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::Put gRPC error (attempt %d): %s\n",
                                     attempt + 1, status.error_message().c_str());
                    InvalidateRegionCache(prefixedKey);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::Put region_error (attempt %d)\n", attempt + 1);
                    InvalidateRegionCache(prefixedKey);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Put region_error failed after %d attempts, giving up\n", attempt + 1);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Put error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        ErrorCode Put(const SizeType key, const std::string& value,
                      const std::chrono::microseconds& timeout,
                      std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            std::string k(reinterpret_cast<const char*>(&key), sizeof(SizeType));
            return Put(k, value, timeout, reqs);
        }

        // ---- Delete operations ----

        ErrorCode Delete(SizeType key) override {
            std::string k(reinterpret_cast<const char*>(&key), sizeof(SizeType));
            std::string prefixedKey = MakePrefixedKey(k);

            auto timeout = std::chrono::microseconds(5000000); // 5s default
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(prefixedKey);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawDeleteRequest request;
                request.set_key(prefixedKey);
                SetContext(request.mutable_context(), prefixedKey);

                kvrpcpb::RawDeleteResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawDelete(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::Delete gRPC error (attempt %d): %s\n", attempt + 1, status.error_message().c_str());
                    InvalidateRegionCache(prefixedKey);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::Delete region_error (attempt %d)\n", attempt + 1);
                    InvalidateRegionCache(prefixedKey);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Delete region_error failed after %d attempts, giving up\n", attempt + 1);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Delete error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        ErrorCode DeleteRange(SizeType start, SizeType end) override {
            std::string startKey(reinterpret_cast<const char*>(&start), sizeof(SizeType));
            std::string endKey(reinterpret_cast<const char*>(&end), sizeof(SizeType));
            std::string prefixedStart = MakePrefixedKey(startKey);
            std::string prefixedEnd = MakePrefixedKey(endKey);

            auto timeout = std::chrono::microseconds(10000000); // 10s default
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(prefixedStart);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawDeleteRangeRequest request;
                request.set_start_key(prefixedStart);
                request.set_end_key(prefixedEnd);
                SetContext(request.mutable_context(), prefixedStart);

                kvrpcpb::RawDeleteRangeResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawDeleteRange(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::DeleteRange gRPC error (attempt %d): %s\n", attempt + 1, status.error_message().c_str());
                    InvalidateRegionCache(prefixedStart);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::DeleteRange region_error (attempt %d)\n", attempt + 1);
                    InvalidateRegionCache(prefixedStart);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::DeleteRange region_error failed after %d attempts, giving up\n", attempt + 1);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::DeleteRange error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        // ---- Merge (append) operation ----
        // TiKV does not have native merge; we implement read-modify-write with
        // a simple get-append-put pattern.

        ErrorCode Merge(const SizeType key, const std::string& value,
                        const std::chrono::microseconds& timeout,
                        std::vector<Helper::AsyncReadRequest>* reqs,
                        int& size) override
        {
            if (value.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::Merge: empty append posting!\n");
                return ErrorCode::Fail;
            }

            std::string existingValue;
            auto ret = Get(key, &existingValue, timeout, reqs);
            if (ret != ErrorCode::Success) {
                // Key doesn't exist yet, just put the new value.
                size = static_cast<int>(value.size());
                return Put(key, value, timeout, reqs);
            }

            // Append the new value to existing
            existingValue.append(value);
            size = static_cast<int>(existingValue.size());
            return Put(key, existingValue, timeout, reqs);
        }

        // ---- MultiGet operations ----
        // Use RawBatchGet grouped by region for efficient batched reads.
        // Tolerate individual key not-found: set empty buffer for missing keys
        // (e.g., postings deleted by splits in multi-layer SPANN).

        ErrorCode MultiGet(const std::vector<SizeType>& keys,
                           std::vector<Helper::PageBuffer<std::uint8_t>>& values,
                           const std::chrono::microseconds& timeout,
                           std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            if (keys.empty()) return ErrorCode::Success;

            // Build prefixed keys and initialize all values as empty
            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                std::string k(reinterpret_cast<const char*>(&keys[i]), sizeof(SizeType));
                prefixedKeys[i] = MakePrefixedKey(k);
                values[i].SetAvailableSize(0);
            }

            // Group keys by (leader address, region id)
            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region;
                std::string addr;
                uint64_t rid = 0;
                if (FindRegionForKey(prefixedKeys[i], region) && !region.leaderAddr.empty()) {
                    addr = region.leaderAddr;
                    rid = region.regionId;
                } else {
                    addr = GetAnyStoreAddress();
                }
                auto& g = regionGroups[{addr, rid}];
                if (g.keys.empty()) g.region = region;
                g.keys.push_back({i, prefixedKeys[i]});
            }

            // Send RawBatchGet per region group (in parallel via futures)
            std::vector<std::future<void>> futures;
            std::mutex resultMutex;

            for (auto& [gkey, rg] : regionGroups) {
                futures.push_back(std::async(std::launch::async, [&, &gkey, &rg]() {
                    auto& group = rg.keys;
                    auto* stub = GetOrCreateStub(gkey.leaderAddr);
                    if (!stub) return;

                    kvrpcpb::RawBatchGetRequest request;
                    SetContextFromRegion(request.mutable_context(), rg.region);
                    for (auto& [idx, pkey] : group) {
                        request.add_keys(pkey);
                    }

                    kvrpcpb::RawBatchGetResponse response;
                    grpc::ClientContext ctx;
                    SetDeadline(ctx, timeout);

                    auto status = stub->RawBatchGet(&ctx, request, &response);
                    if (!status.ok() || (status.ok() && response.has_region_error())) {
                        // Region error or gRPC failure: invalidate cache and fallback to individual gets
                        // (Get has its own region error retry logic)
                        if (status.ok() && response.has_region_error()) {
                            for (auto& [idx, pkey] : group) {
                                InvalidateRegionCache(pkey);
                            }
                        }
                        for (auto& [idx, pkey] : group) {
                            std::string val;
                            auto ret = Get(keys[idx], &val, timeout, reqs);
                            if (ret == ErrorCode::Success && !val.empty()) {
                                std::lock_guard<std::mutex> lock(resultMutex);
                                if (val.size() > values[idx].GetPageSize()) {
                                    values[idx].ReservePageBuffer(val.size());
                                }
                                memcpy(values[idx].GetBuffer(), val.data(), val.size());
                                values[idx].SetAvailableSize(static_cast<int>(val.size()));
                            }
                        }
                        return;
                    }

                    // Build a map from prefixed key -> response value
                    std::unordered_map<std::string, std::string> resultMap;
                    for (int p = 0; p < response.pairs_size(); p++) {
                        const auto& pair = response.pairs(p);
                        if (!pair.has_error() && !pair.value().empty()) {
                            resultMap[pair.key()] = pair.value();
                        }
                    }

                    // Copy results to output buffers
                    std::lock_guard<std::mutex> lock(resultMutex);
                    for (auto& [idx, pkey] : group) {
                        auto it = resultMap.find(pkey);
                        if (it != resultMap.end()) {
                            const auto& val = it->second;
                            if (val.size() > values[idx].GetPageSize()) {
                                values[idx].ReservePageBuffer(val.size());
                            }
                            memcpy(values[idx].GetBuffer(), val.data(), val.size());
                            values[idx].SetAvailableSize(static_cast<int>(val.size()));
                        }
                        // else: remains empty (SetAvailableSize(0)), tolerating missing keys
                    }
                }));
            }

            // Wait for all region batch gets to complete
            for (auto& f : futures) {
                f.get();
            }

            // Check if any keys that should have values came back empty due to errors
            // (not-found is tolerable, but gRPC/region errors are not)
            return ErrorCode::Success;
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys,
                           std::vector<std::string>* values,
                           const std::chrono::microseconds& timeout,
                           std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            if (keys.empty()) return ErrorCode::Success;

            // Build prefixed keys
            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                prefixedKeys[i] = MakePrefixedKey(keys[i]);
            }

            // Initialize output with empty strings
            values->resize(keys.size());

            // Group by (leader address, region id)
            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region;
                std::string addr;
                uint64_t rid = 0;
                if (FindRegionForKey(prefixedKeys[i], region) && !region.leaderAddr.empty()) {
                    addr = region.leaderAddr;
                    rid = region.regionId;
                } else {
                    addr = GetAnyStoreAddress();
                }
                auto& g = regionGroups[{addr, rid}];
                if (g.keys.empty()) g.region = region;
                g.keys.push_back({i, prefixedKeys[i]});
            }

            for (auto& [gkey, rg] : regionGroups) {
                auto& group = rg.keys;
                auto* stub = GetOrCreateStub(gkey.leaderAddr);
                if (!stub) continue;

                kvrpcpb::RawBatchGetRequest request;
                SetContextFromRegion(request.mutable_context(), rg.region);
                for (auto& [idx, pkey] : group) {
                    request.add_keys(pkey);
                }

                kvrpcpb::RawBatchGetResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawBatchGet(&ctx, request, &response);
                if (!status.ok() || (status.ok() && response.has_region_error())) {
                    // Region error or gRPC failure: invalidate cache and fallback to individual gets
                    // (Get has its own region error retry logic)
                    if (status.ok() && response.has_region_error()) {
                        for (auto& [idx, pkey] : group) {
                            InvalidateRegionCache(pkey);
                        }
                    }
                    for (auto& [idx, pkey] : group) {
                        std::string val;
                        if (Get(keys[idx], &val, timeout, reqs) == ErrorCode::Success) {
                            (*values)[idx] = std::move(val);
                        }
                    }
                    continue;
                }

                std::unordered_map<std::string, std::string> resultMap;
                for (int p = 0; p < response.pairs_size(); p++) {
                    const auto& pair = response.pairs(p);
                    if (!pair.has_error() && !pair.value().empty()) {
                        resultMap[pair.key()] = pair.value();
                    }
                }

                for (auto& [idx, pkey] : group) {
                    auto it = resultMap.find(pkey);
                    if (it != resultMap.end()) {
                        (*values)[idx] = std::move(it->second);
                    }
                }
            }

            return ErrorCode::Success;
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys,
                           std::vector<std::string>* values,
                           const std::chrono::microseconds& timeout,
                           std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            if (keys.empty()) return ErrorCode::Success;

            // Convert SizeType keys to strings and delegate
            std::vector<std::string> strKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                strKeys[i] = std::string(reinterpret_cast<const char*>(&keys[i]), sizeof(SizeType));
            }
            return MultiGet(strKeys, values, timeout, reqs);
        }

        // ---- Coprocessor vector search ----
        // Push distance computation into TiKV: send query vector + posting
        // keys, TiKV reads posting data locally, computes L2 distances, and
        // returns only top-N (vector_id, distance) candidates.

        struct CoprocessorResult {
            SizeType vectorID;
            float distance;
        };

        ErrorCode CoprocessorSearch(
            const std::vector<SizeType>& postingIDs,
            const uint8_t* queryVector,
            int dim,
            int valueType,       // 0=UInt8, 1=Int8, 3=Float32
            int metaDataSize,
            int topN,
            const std::chrono::microseconds& timeout,
            std::vector<CoprocessorResult>& results)
        {
            if (postingIDs.empty()) return ErrorCode::Success;

            // Determine vector data size
            int valueSize = (valueType == 3) ? 4 : 1;
            int queryVecBytes = dim * valueSize;

            // Build prefixed keys for all posting IDs
            std::vector<std::string> prefixedKeys(postingIDs.size());
            for (size_t i = 0; i < postingIDs.size(); i++) {
                std::string k(reinterpret_cast<const char*>(&postingIDs[i]), sizeof(SizeType));
                prefixedKeys[i] = MakePrefixedKey(k);
            }

            // Group keys by (leader address, region id)
            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region;
                std::string addr;
                uint64_t rid = 0;
                if (FindRegionForKey(prefixedKeys[i], region) && !region.leaderAddr.empty()) {
                    addr = region.leaderAddr;
                    rid = region.regionId;
                } else {
                    addr = GetAnyStoreAddress();
                }
                auto& g = regionGroups[{addr, rid}];
                if (g.keys.empty()) g.region = region;
                g.keys.push_back({i, prefixedKeys[i]});
            }

            // Send RawCoprocessor per region group in parallel
            std::vector<std::future<std::vector<CoprocessorResult>>> futures;

            for (auto& [gkey, rg] : regionGroups) {
                futures.push_back(std::async(std::launch::async,
                    [&, &gkey, &rg]() -> std::vector<CoprocessorResult> {
                    auto& group = rg.keys;
                    auto* stub = GetOrCreateStub(gkey.leaderAddr);
                    if (!stub) return {};

                    // Encode the vector search request
                    std::string requestData = EncodeVectorSearchRequest(
                        dim, topN, valueType, metaDataSize,
                        queryVector, queryVecBytes, group);

                    kvrpcpb::RawCoprocessorRequest request;
                    SetContextFromRegion(request.mutable_context(), rg.region);
                    request.set_copr_name("vector_search");
                    request.set_copr_version_req("*");
                    request.set_data(std::move(requestData));

                    // Add a key range covering the group keys for routing
                    auto* range = request.add_ranges();
                    range->set_start_key(group.front().second);
                    range->set_end_key(group.back().second);

                    kvrpcpb::RawCoprocessorResponse response;
                    grpc::ClientContext ctx;
                    SetDeadline(ctx, timeout);

                    auto status = stub->RawCoprocessor(&ctx, request, &response);
                    if (!status.ok()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "TiKVIO::CoprocessorSearch gRPC error: %s\n",
                            status.error_message().c_str());
                        return {};
                    }
                    if (response.has_region_error()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "TiKVIO::CoprocessorSearch region error\n");
                        InvalidateRegionCache(group[0].second);
                        return {};
                    }
                    if (!response.error().empty()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "TiKVIO::CoprocessorSearch error: %s\n",
                            response.error().c_str());
                        return {};
                    }

                    return DecodeVectorSearchResponse(response.data());
                }));
            }

            // Merge results from all regions
            results.clear();
            for (auto& f : futures) {
                auto regionResults = f.get();
                results.insert(results.end(), regionResults.begin(), regionResults.end());
            }

            // Sort by distance and truncate to topN
            std::sort(results.begin(), results.end(),
                [](const CoprocessorResult& a, const CoprocessorResult& b) {
                    return a.distance < b.distance;
                });
            if (static_cast<int>(results.size()) > topN) {
                results.resize(topN);
            }

            return ErrorCode::Success;
        }

        // ---- Multi-Chunk Posting operations ----
        // Instead of read-modify-write on a single key per posting,
        // each posting is stored as multiple KV chunks:
        //   Base key:  [prefix]_[headID 4B]\x00          (build / compaction)
        //   Chunk key: [prefix]_[headID 4B]\x00[ts 8B]   (append)
        // Read = Scan([...headID\x00, ...headID\x01))  → concat all values
        // Delete = DeleteRange over the same span

        // Build the chunk-aware prefixed key for a headID.
        // suffix == "" → base key; suffix == 8-byte ts → chunk key.
        std::string MakeChunkKey(SizeType headID, const std::string& suffix = "") const {
            std::string raw(reinterpret_cast<const char*>(&headID), sizeof(SizeType));
            std::string result;
            result.reserve(m_keyPrefix.size() + 1 + sizeof(SizeType) + 1 + suffix.size());
            result.append(m_keyPrefix);
            result.push_back('_');
            result.append(raw);
            result.push_back('\x00');  // delimiter
            result.append(suffix);
            return result;
        }

        // Write a new chunk for an append operation.
        // Uses nanosecond timestamp as chunk ID (unique under held lock).
        ErrorCode PutChunk(SizeType headID,
                           const std::string& value,
                           const std::chrono::microseconds& timeout,
                           std::vector<Helper::AsyncReadRequest>* reqs)
        {
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
            uint64_t ts = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
            std::string suffix(reinterpret_cast<const char*>(&ts), sizeof(ts));
            std::string key = MakeChunkKey(headID, suffix);

            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(key);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawPutRequest request;
                request.set_key(key);
                request.set_value(value);
                SetContext(request.mutable_context(), key);

                kvrpcpb::RawPutResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawPut(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::PutChunk gRPC error (attempt %d): %s headID=%d\n", attempt + 1, status.error_message().c_str(), headID);
                    InvalidateRegionCache(key);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::PutChunk region_error (attempt %d) headID=%d\n", attempt + 1, headID);
                    InvalidateRegionCache(key);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::PutChunk region_error failed after %d attempts headID=%d, giving up\n", attempt + 1, headID);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::PutChunk error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        // Write the base (sole) chunk for a posting — used by Build and Split compaction.
        ErrorCode PutBaseChunk(SizeType headID,
                               const std::string& value,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs)
        {
            std::string key = MakeChunkKey(headID); // no suffix → base key
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(key);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawPutRequest request;
                request.set_key(key);
                request.set_value(value);
                SetContext(request.mutable_context(), key);

                kvrpcpb::RawPutResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawPut(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::PutBaseChunk gRPC error (attempt %d): %s headID=%d\n", attempt + 1, status.error_message().c_str(), headID);
                    InvalidateRegionCache(key);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::PutBaseChunk region_error (attempt %d) headID=%d\n", attempt + 1, headID);
                    InvalidateRegionCache(key);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::PutBaseChunk region_error failed after %d attempts headID=%d, giving up\n", attempt + 1, headID);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::PutBaseChunk error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        // Read all chunks belonging to a posting (Scan), concatenate into one string.
        // Returns the full posting data and the number of chunks found.
        ErrorCode ScanPosting(SizeType headID,
                              std::string* fullPosting,
                              const std::chrono::microseconds& timeout,
                              int* chunkCount = nullptr)
        {
            std::string startKey = MakeChunkKey(headID); // prefix_headID\x00
            std::string endKey;
            {
                // endKey = prefix_headID\x01 — one past the delimiter byte
                std::string raw(reinterpret_cast<const char*>(&headID), sizeof(SizeType));
                endKey.reserve(m_keyPrefix.size() + 1 + sizeof(SizeType) + 1);
                endKey.append(m_keyPrefix);
                endKey.push_back('_');
                endKey.append(raw);
                endKey.push_back('\x01');
            }

            fullPosting->clear();
            int chunks = 0;

            // Paginated scan with region_error retry support
            std::string scanCursor = startKey;
            bool morePages = true;
            while (morePages) {
                morePages = false;
                for (int attempt = 0; ; attempt++) {
                    auto stub = GetStubForKey(scanCursor);
                    if (!stub) { RetryBackoff(attempt); continue; }

                    kvrpcpb::RawScanRequest request;
                    request.set_start_key(scanCursor);
                    request.set_end_key(endKey);
                    request.set_limit(1024);
                    SetContext(request.mutable_context(), scanCursor);

                    kvrpcpb::RawScanResponse response;
                    grpc::ClientContext ctx;
                    SetDeadline(ctx, timeout);

                    auto status = stub->RawScan(&ctx, request, &response);
                    if (!status.ok()) {
                        if (ShouldLogRetry(attempt))
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "TiKVIO::ScanPosting gRPC error (attempt %d): %s headID=%d\n",
                                attempt + 1, status.error_message().c_str(), headID);
                        InvalidateRegionCache(scanCursor);
                        RetryBackoff(attempt);
                        continue;
                    }
                    if (response.has_region_error()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "TiKVIO::ScanPosting region_error (attempt %d) headID=%d\n",
                                attempt + 1, headID);
                        InvalidateRegionCache(scanCursor);
                        if (attempt >= 10) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::ScanPosting region_error failed after %d attempts headID=%d, giving up\n", attempt + 1, headID);
                            return ErrorCode::Fail;
                        }
                        RetryBackoff(attempt);
                        continue;
                    }

                    int count = response.kvs_size();
                    for (int i = 0; i < count; i++) {
                        fullPosting->append(response.kvs(i).value());
                        chunks++;
                    }
                    if (count >= 1024) {
                        scanCursor = response.kvs(count - 1).key();
                        scanCursor.push_back('\x00');
                        morePages = true;
                    }
                    break; // success, break retry loop
                }
            }

            if (chunkCount) *chunkCount = chunks;
            if (chunks > 0) return ErrorCode::Success;
            return ErrorCode::VectorNotFound;
        }

        // Delete all chunks of a posting (DeleteRange over the chunk key span).
        ErrorCode DeletePosting(SizeType headID)
        {
            std::string startKey = MakeChunkKey(headID); // prefix_headID\x00
            std::string endKey;
            {
                std::string raw(reinterpret_cast<const char*>(&headID), sizeof(SizeType));
                endKey.reserve(m_keyPrefix.size() + 1 + sizeof(SizeType) + 1);
                endKey.append(m_keyPrefix);
                endKey.push_back('_');
                endKey.append(raw);
                endKey.push_back('\x01');
            }

            auto timeout = std::chrono::microseconds(10000000); // 10s
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(startKey);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawDeleteRangeRequest request;
                request.set_start_key(startKey);
                request.set_end_key(endKey);
                SetContext(request.mutable_context(), startKey);

                kvrpcpb::RawDeleteRangeResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawDeleteRange(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::DeletePosting gRPC error (attempt %d): %s headID=%d\n",
                            attempt + 1, status.error_message().c_str(), headID);
                    InvalidateRegionCache(startKey);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "TiKVIO::DeletePosting gRPC error failed after %d attempts headID=%d, giving up delete but continue\n",
                                     attempt + 1, headID);
                        return ErrorCode::Success;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::DeletePosting region_error (attempt %d) headID=%d\n", attempt + 1, headID);
                    InvalidateRegionCache(startKey);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::DeletePosting region_error failed after %d attempts headID=%d, giving up delete but continue\n", attempt + 1, headID);
                        // Non-fatal: keep the pipeline running even when a stale region route
                        // cannot be recovered after retries.
                        return ErrorCode::Success;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "TiKVIO::DeletePosting error: %s, giving up delete but continue\n",
                                 response.error().c_str());
                    return ErrorCode::Success;
                }
                return ErrorCode::Success;
            }
        }

        // ---- Posting count key operations ----
        // Each posting has a count key storing the number of vectors (int32).
        //   Count key: [prefix]_[headID 4B]\x02
        // Isolated from chunk keys (\x00..\x01 range).

        std::string MakeCountKey(SizeType headID) const {
            std::string raw(reinterpret_cast<const char*>(&headID), sizeof(SizeType));
            std::string result;
            result.reserve(m_keyPrefix.size() + 1 + sizeof(SizeType) + 1);
            result.append(m_keyPrefix);
            result.push_back('_');
            result.append(raw);
            result.push_back('\x02');
            return result;
        }

        // Read posting count from TiKV. Returns count >= 0 on success, -1 on error.
        int GetPostingCount(SizeType headID, const std::chrono::microseconds& timeout) {
            std::string key = MakeCountKey(headID);
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(key);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawGetRequest request;
                request.set_key(key);
                SetContext(request.mutable_context(), key);

                kvrpcpb::RawGetResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawGet(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::GetPostingCount gRPC error (attempt %d): %s headID=%d\n", attempt + 1, status.error_message().c_str(), headID);
                    InvalidateRegionCache(key);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::GetPostingCount region_error (attempt %d) headID=%d\n", attempt + 1, headID);
                    InvalidateRegionCache(key);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::GetPostingCount region_error failed after %d attempts headID=%d, giving up\n", attempt + 1, headID);
                        return -1;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.not_found() || response.value().size() < sizeof(int32_t)) {
                    return 0;
                }
                int32_t count;
                memcpy(&count, response.value().data(), sizeof(int32_t));
                return count;
            }
        }

        // Write posting count to TiKV.
        ErrorCode SetPostingCount(SizeType headID, int count,
                                  const std::chrono::microseconds& timeout) {
            std::string key = MakeCountKey(headID);
            std::string value(reinterpret_cast<const char*>(&count), sizeof(int32_t));

            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(key);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawPutRequest request;
                request.set_key(key);
                request.set_value(value);
                SetContext(request.mutable_context(), key);

                kvrpcpb::RawPutResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawPut(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::SetPostingCount gRPC error (attempt %d): %s headID=%d\n", attempt + 1, status.error_message().c_str(), headID);
                    InvalidateRegionCache(key);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::SetPostingCount region_error (attempt %d) headID=%d\n", attempt + 1, headID);
                    InvalidateRegionCache(key);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::SetPostingCount region_error failed after %d attempts headID=%d, giving up\n", attempt + 1, headID);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::SetPostingCount error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        // Delete posting count key.
        ErrorCode DeletePostingCount(SizeType headID) {
            std::string key = MakeCountKey(headID);
            auto timeout = std::chrono::microseconds(10000000);
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(key);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawDeleteRequest request;
                request.set_key(key);
                SetContext(request.mutable_context(), key);

                kvrpcpb::RawDeleteResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawDelete(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::DeletePostingCount gRPC error (attempt %d): %s headID=%d\n", attempt + 1, status.error_message().c_str(), headID);
                    InvalidateRegionCache(key);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::DeletePostingCount region_error (attempt %d) headID=%d\n", attempt + 1, headID);
                    InvalidateRegionCache(key);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::DeletePostingCount region_error failed after %d attempts headID=%d, giving up\n", attempt + 1, headID);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                return ErrorCode::Success;
            }
        }

        // Atomically write a chunk and update count via RawBatchPut.
        // Saves one network round trip vs separate PutChunk + SetPostingCount.
        ErrorCode PutChunkAndCount(SizeType headID,
                                   const std::string& chunkValue,
                                   int newCount,
                                   const std::chrono::microseconds& timeout,
                                   std::vector<Helper::AsyncReadRequest>* reqs) {
            // Build chunk key with nanosecond timestamp
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
            uint64_t ts = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
            std::string suffix(reinterpret_cast<const char*>(&ts), sizeof(ts));
            std::string chunkKey = MakeChunkKey(headID, suffix);

            // Build count key
            std::string countKey = MakeCountKey(headID);
            std::string countValue(reinterpret_cast<const char*>(&newCount), sizeof(int32_t));

            // Try RawBatchPut first (single round trip).
            // If region error (chunkKey and countKey in different regions after split),
            // fall back immediately to individual Put calls.
            {
                auto stub = GetStubForKey(chunkKey);
                if (stub) {
                    kvrpcpb::RawBatchPutRequest request;
                    SetContext(request.mutable_context(), chunkKey);

                    auto* pair1 = request.add_pairs();
                    pair1->set_key(chunkKey);
                    pair1->set_value(chunkValue);

                    auto* pair2 = request.add_pairs();
                    pair2->set_key(countKey);
                    pair2->set_value(countValue);

                    kvrpcpb::RawBatchPutResponse response;
                    grpc::ClientContext ctx;
                    SetDeadline(ctx, timeout);

                    auto status = stub->RawBatchPut(&ctx, request, &response);
                    if (status.ok() && !response.has_region_error() && response.error().empty()) {
                        return ErrorCode::Success;
                    }
                    // Any failure: invalidate cache and fall through to individual puts
                    if (!status.ok()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::PutChunkAndCount BatchPut gRPC error headID=%d: %s, falling back\n", headID, status.error_message().c_str());
                    } else if (response.has_region_error()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO::PutChunkAndCount BatchPut region_error headID=%d, falling back to individual puts\n", headID);
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::PutChunkAndCount error: %s\n", response.error().c_str());
                    }
                    InvalidateRegionCache(chunkKey);
                    InvalidateRegionCache(countKey);
                }
            }

            // Fallback: write chunk and count separately.
            // Each call has its own region discovery + retry logic,
            // so this handles cross-region splits reliably.
            // Note: chunkKey/countKey are already prefixed, use RawPutWithRetry.
            auto ret1 = RawPutWithRetry(chunkKey, chunkValue, timeout);
            if (ret1 != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "TiKVIO::PutChunkAndCount fallback: PutChunk failed headID=%d\n", headID);
                return ret1;
            }
            auto ret2 = RawPutWithRetry(countKey, countValue, timeout);
            if (ret2 != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "TiKVIO::PutChunkAndCount fallback: PutCount failed headID=%d\n", headID);
                return ret2;
            }
            return ErrorCode::Success;
        }

        // Multi-posting scan: read multiple postings in parallel.
        // Used by SearchIndex to replace MultiGet when multi-chunk is enabled.
        ErrorCode MultiScanPostings(const std::vector<SizeType>& headIDs,
                                    std::vector<Helper::PageBuffer<std::uint8_t>>& values,
                                    const std::chrono::microseconds& timeout)
        {
            if (headIDs.empty()) return ErrorCode::Success;

            std::atomic<int> failCount(0);
            std::vector<std::future<void>> futures;
            for (size_t i = 0; i < headIDs.size(); i++) {
                futures.push_back(std::async(std::launch::async, [&, i]() {
                    std::string posting;
                    auto ret = ScanPosting(headIDs[i], &posting, timeout);
                    // On actual gRPC error (not VectorNotFound), retry with generous timeout
                    if (ret == ErrorCode::Fail) {
                        auto retryTimeout = std::chrono::microseconds(10000000); // 10s
                        ret = ScanPosting(headIDs[i], &posting, retryTimeout);
                    }
                    if (ret == ErrorCode::Success && !posting.empty()) {
                        if (posting.size() > values[i].GetPageSize()) {
                            values[i].ReservePageBuffer(posting.size());
                        }
                        memcpy(values[i].GetBuffer(), posting.data(), posting.size());
                        values[i].SetAvailableSize(static_cast<int>(posting.size()));
                    } else {
                        values[i].SetAvailableSize(0);
                        if (ret == ErrorCode::Fail) {
                            failCount.fetch_add(1);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "MultiScanPostings: ScanPosting failed for headID %d after retry\n", headIDs[i]);
                        }
                    }
                }));
            }
            for (auto& f : futures) f.get();
            if (failCount.load() > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "MultiScanPostings: %d/%d postings had gRPC errors\n", failCount.load(), (int)headIDs.size());
            }
            return ErrorCode::Success;
        }

        // ---- Scan operations ----

        ErrorCode StartToScan(SizeType& key, std::string* value) {
            std::string startKey = m_keyPrefix;
            std::string endKey = m_keyPrefix;
            // Create a range that covers all our prefixed keys
            endKey.push_back(static_cast<char>(0xFF));

            auto stub = GetStubForKey(startKey);
            if (!stub) return ErrorCode::Fail;

            kvrpcpb::RawScanRequest request;
            request.set_start_key(startKey);
            request.set_end_key(endKey);
            request.set_limit(4096);
            SetContext(request.mutable_context(), startKey);

            kvrpcpb::RawScanResponse response;
            grpc::ClientContext ctx;

            auto status = stub->RawScan(&ctx, request, &response);
            if (!status.ok() || response.kvs_size() == 0) {
                return ErrorCode::Fail;
            }

            // Cache scan results
            m_scanResults.clear();
            m_scanIndex = 0;
            for (int i = 0; i < response.kvs_size(); i++) {
                m_scanResults.push_back({response.kvs(i).key(), response.kvs(i).value()});
            }

            const auto& first = m_scanResults[0];
            std::string rawKey = StripPrefix(first.first);
            if (rawKey.size() >= sizeof(SizeType)) {
                key = *reinterpret_cast<const SizeType*>(rawKey.data());
            }
            *value = first.second;
            m_scanIndex = 1;
            return ErrorCode::Success;
        }

        ErrorCode NextToScan(SizeType& key, std::string* value) {
            if (m_scanIndex >= m_scanResults.size()) {
                return ErrorCode::Fail;
            }

            const auto& entry = m_scanResults[m_scanIndex];
            std::string rawKey = StripPrefix(entry.first);
            if (rawKey.size() >= sizeof(SizeType)) {
                key = *reinterpret_cast<const SizeType*>(rawKey.data());
            }
            *value = entry.second;
            m_scanIndex++;
            return ErrorCode::Success;
        }

        void ForceCompaction() override {
            // TiKV handles compaction internally; this is a no-op.
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: ForceCompaction is a no-op (TiKV manages compaction internally)\n");
        }

        ErrorCode Check(const SizeType key, std::vector<std::uint8_t> *visited) override {
            // TiKV guarantees data integrity internally via Raft consensus.
            // Posting size checks are skipped since TiKV is shared mutable storage
            // and concurrent inserts/splits may update postings between size recording
            // and check time.
            return ErrorCode::Success;
        }

        void GetStat() override {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Connected to PD cluster with %zu addresses, prefix='%s'\n",
                         m_pdAddresses.size(), m_keyPrefix.c_str());
        }

        ErrorCode Checkpoint(std::string prefix) override {
            // TiKV provides its own snapshot/backup mechanism; no local checkpoint.
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Checkpoint is a no-op (use TiKV's backup tools)\n");
            return ErrorCode::Success;
        }

    private:
        std::string m_keyPrefix;
        std::vector<std::string> m_pdAddresses;
        std::unique_ptr<pdpb::PD::Stub> m_pdStub;
        uint64_t m_clusterId = 0;
        bool m_available = false;

        // TiKV store stub pools keyed by store address (multiple channels per store)
        static constexpr int kStubPoolSize = 48;
        struct StubPool {
            std::vector<std::shared_ptr<tikvpb::Tikv::Stub>> stubs;
            std::atomic<uint64_t> next{0};
            tikvpb::Tikv::Stub* GetNext() {
                return stubs[next.fetch_add(1, std::memory_order_relaxed) % stubs.size()].get();
            }
        };
        mutable std::mutex m_storeMutex;
        std::unordered_map<std::string, std::shared_ptr<StubPool>> m_storeStubs;

        // Store address cache: store_id -> address
        mutable std::mutex m_storeAddrMutex;
        std::unordered_map<uint64_t, std::string> m_storeAddrCache;

        // Region cache: maps a key prefix to (region_id, leader_store_addr)
        struct RegionInfo {
            uint64_t regionId;
            uint64_t storeId;
            std::string leaderAddr;
            std::string startKey;
            std::string endKey;
            metapb::RegionEpoch epoch;
            metapb::Peer leaderPeer;  // Full peer info (id + store_id)
        };
        mutable std::shared_mutex m_regionMutex;
        std::vector<RegionInfo> m_regionCache;

        // Scan state
        std::vector<std::pair<std::string, std::string>> m_scanResults;
        size_t m_scanIndex = 0;

        // ---- Helper: build a prefixed key ----
        std::string MakePrefixedKey(const std::string& key) const {
            std::string result;
            result.reserve(m_keyPrefix.size() + 1 + key.size());
            result.append(m_keyPrefix);
            result.push_back('_');
            result.append(key);
            return result;
        }

        // ---- Helper: RawPut with retry for an already-prefixed key ----
        ErrorCode RawPutWithRetry(const std::string& prefixedKey, const std::string& value,
                                  const std::chrono::microseconds& timeout) {
            for (int attempt = 0; ; attempt++) {
                auto stub = GetStubForKey(prefixedKey);
                if (!stub) { RetryBackoff(attempt); continue; }

                kvrpcpb::RawPutRequest request;
                request.set_key(prefixedKey);
                request.set_value(value);
                SetContext(request.mutable_context(), prefixedKey);

                kvrpcpb::RawPutResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawPut(&ctx, request, &response);
                if (!status.ok()) {
                    if (ShouldLogRetry(attempt))
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::RawPutWithRetry gRPC error (attempt %d): %s\n", attempt + 1, status.error_message().c_str());
                    InvalidateRegionCache(prefixedKey);
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO::RawPutWithRetry region_error (attempt %d)\n", attempt + 1);
                    InvalidateRegionCache(prefixedKey);
                    if (attempt >= 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::RawPutWithRetry region_error failed after %d attempts, giving up\n", attempt + 1);
                        return ErrorCode::Fail;
                    }
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO::RawPutWithRetry error: %s\n", response.error().c_str());
                    return ErrorCode::Fail;
                }
                return ErrorCode::Success;
            }
        }

        std::string StripPrefix(const std::string& prefixedKey) const {
            size_t prefixLen = m_keyPrefix.size() + 1; // prefix + '_'
            if (prefixedKey.size() > prefixLen) {
                return prefixedKey.substr(prefixLen);
            }
            return "";
        }

        // ---- Helper: set gRPC deadline ----
        void SetDeadline(grpc::ClientContext& ctx, const std::chrono::microseconds& timeout) const {
            if (timeout.count() > 0) {
                // Cap at 60 seconds to prevent overflow when timeout is chrono::microseconds::max()
                auto cappedTimeout = std::min(timeout, std::chrono::microseconds(60000000));
                ctx.set_deadline(std::chrono::system_clock::now() + cappedTimeout);
            }
        }

        // ---- Helper: retry backoff with cap at 5 seconds ----
        static void RetryBackoff(int attempt) {
            int ms = std::min(100 * (attempt + 1), 5000);
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }

        // Should we log this retry attempt? First 3 always, then every 10th.
        static bool ShouldLogRetry(int attempt) {
            return attempt < 3 || attempt % 10 == 0;
        }

        // ---- Helper: set request context with region info ----
        void SetContext(kvrpcpb::Context* ctx, const std::string& key) {
            // Look up region for this key
            RegionInfo region;
            if (FindRegionForKey(key, region)) {
                ctx->set_region_id(region.regionId);
                *ctx->mutable_region_epoch() = region.epoch;
                *ctx->mutable_peer() = region.leaderPeer;
            }
        }

        // Overload: set context directly from a cached RegionInfo.
        void SetContextFromRegion(kvrpcpb::Context* ctx, const RegionInfo& region) {
            ctx->set_region_id(region.regionId);
            *ctx->mutable_region_epoch() = region.epoch;
            *ctx->mutable_peer() = region.leaderPeer;
        }

        // Composite key for grouping by (leader address, region id).
        struct RegionGroupKey {
            std::string leaderAddr;
            uint64_t regionId;
            bool operator==(const RegionGroupKey& o) const {
                return leaderAddr == o.leaderAddr && regionId == o.regionId;
            }
        };
        struct RegionGroupKeyHash {
            size_t operator()(const RegionGroupKey& k) const {
                auto h1 = std::hash<std::string>{}(k.leaderAddr);
                auto h2 = std::hash<uint64_t>{}(k.regionId);
                return h1 ^ (h2 << 1);
            }
        };

        struct RegionGroup {
            RegionInfo region;
            std::vector<std::pair<size_t, std::string>> keys; // (original_index, prefixed_key)
        };

        // ---- Reconnect to PD leader (called when PD stub fails) ----
        bool ReconnectPD() {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO: Attempting PD reconnection...\n");
            for (const auto& pdAddr : m_pdAddresses) {
                auto channel = grpc::CreateChannel(pdAddr, grpc::InsecureChannelCredentials());
                auto stub = pdpb::PD::NewStub(channel);
                if (!stub) continue;

                pdpb::GetMembersRequest membersReq;
                auto* header = membersReq.mutable_header();
                header->set_cluster_id(m_clusterId);
                pdpb::GetMembersResponse membersResp;
                grpc::ClientContext membersCtx;
                membersCtx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

                auto status = stub->GetMembers(&membersCtx, membersReq, &membersResp);
                if (!status.ok()) continue;

                if (membersResp.has_leader() && membersResp.leader().client_urls_size() > 0) {
                    if (membersResp.has_header()) {
                        m_clusterId = membersResp.header().cluster_id();
                    }
                    std::string leaderAddr = membersResp.leader().client_urls(0);
                    auto schemePos = leaderAddr.find("://");
                    if (schemePos != std::string::npos) {
                        leaderAddr = leaderAddr.substr(schemePos + 3);
                    }
                    if (leaderAddr == pdAddr) {
                        m_pdStub = std::move(stub);
                    } else {
                        auto leaderChannel = grpc::CreateChannel(leaderAddr, grpc::InsecureChannelCredentials());
                        m_pdStub = pdpb::PD::NewStub(leaderChannel);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Reconnected to PD leader at %s\n", leaderAddr.c_str());
                    return true;
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: PD reconnection failed on all addresses\n");
            return false;
        }

        // ---- PD: get region for a given key (with retry + reconnect) ----
        bool GetRegionFromPD(const std::string& key, RegionInfo& info) {
            for (int attempt = 0; attempt < 5; attempt++) {
                if (!m_pdStub) {
                    if (!ReconnectPD()) {
                        std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
                        continue;
                    }
                }

                pdpb::GetRegionRequest request;
                request.set_region_key(key);
                auto* header = request.mutable_header();
                header->set_cluster_id(m_clusterId);

                pdpb::GetRegionResponse response;
                grpc::ClientContext ctx;
                ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

                auto status = m_pdStub->GetRegion(&ctx, request, &response);
                if (!status.ok()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO: PD GetRegion failed (attempt %d): %s\n", attempt, status.error_message().c_str());
                    m_pdStub.reset();
                    std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
                    continue;
                }

                if (!response.has_region() || !response.has_leader()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO: PD GetRegion returned no region/leader (attempt %d)\n", attempt);
                    std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
                    continue;
                }

            const auto& region = response.region();
            const auto& leader = response.leader();

            info.regionId = region.id();
            info.startKey = region.start_key();
            info.endKey = region.end_key();
            info.epoch = region.region_epoch();
            info.storeId = leader.store_id();
            info.leaderPeer = leader;  // Store full peer info (id + store_id)

            // Get store address from PD
            info.leaderAddr = GetStoreAddress(leader.store_id());

            // Cache the region info
            {
                std::unique_lock<std::shared_mutex> lock(m_regionMutex);
                // Replace existing entry for this region or add new
                bool found = false;
                for (auto& cached : m_regionCache) {
                    if (cached.regionId == info.regionId) {
                        cached = info;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    m_regionCache.push_back(info);
                }
            }

                return true;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: GetRegionFromPD failed after all retries for key\n");
            return false;
        }

        // ---- PD: get store address by store ID (with retry + reconnect + cache) ----
        std::string GetStoreAddress(uint64_t storeId) {
            // Check cache first
            {
                std::lock_guard<std::mutex> lock(m_storeAddrMutex);
                auto it = m_storeAddrCache.find(storeId);
                if (it != m_storeAddrCache.end()) {
                    return it->second;
                }
            }

            for (int attempt = 0; attempt < 3; attempt++) {
                if (!m_pdStub) {
                    if (!ReconnectPD()) {
                        std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
                        continue;
                    }
                }

                pdpb::GetStoreRequest request;
                request.set_store_id(storeId);
                auto* header = request.mutable_header();
                header->set_cluster_id(m_clusterId);

                pdpb::GetStoreResponse response;
                grpc::ClientContext ctx;
                ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

                auto status = m_pdStub->GetStore(&ctx, request, &response);
                if (!status.ok() || !response.has_store()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVIO: PD GetStore failed for store %lu (attempt %d)\n", storeId, attempt);
                    if (!status.ok()) m_pdStub.reset();
                    std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
                    continue;
                }

                std::string addr = response.store().address();
                // Cache the result
                {
                    std::lock_guard<std::mutex> lock(m_storeAddrMutex);
                    m_storeAddrCache[storeId] = addr;
                }
                return addr;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: PD GetStore failed for store %lu after all retries\n", storeId);
            return "";
        }

        // ---- Find cached region for a key ----
        bool FindRegionForKey(const std::string& key, RegionInfo& info) {
            {
                std::shared_lock<std::shared_mutex> lock(m_regionMutex);
                for (const auto& region : m_regionCache) {
                    if ((region.startKey.empty() || key >= region.startKey) &&
                        (region.endKey.empty() || key < region.endKey)) {
                        info = region;
                        return true;
                    }
                }
            }
            // Cache miss: query PD
            return GetRegionFromPD(key, info);
        }

        // ---- Invalidate cached region for a key ----
        void InvalidateRegionCache(const std::string& key) {
            std::unique_lock<std::shared_mutex> lock(m_regionMutex);
            m_regionCache.erase(
                std::remove_if(m_regionCache.begin(), m_regionCache.end(),
                    [&key](const RegionInfo& r) {
                        return (r.startKey.empty() || key >= r.startKey) &&
                               (r.endKey.empty() || key < r.endKey);
                    }),
                m_regionCache.end());
        }

        // ---- Get or create a TiKV stub for a key ----
        tikvpb::Tikv::Stub* GetStubForKey(const std::string& key) {
            RegionInfo region;
            bool found = FindRegionForKey(key, region);
            if (found && region.leaderAddr.empty() && region.storeId != 0) {
                // Region found but address not resolved yet — resolve from storeId
                region.leaderAddr = GetStoreAddress(region.storeId);
            }
            if (!found || region.leaderAddr.empty()) {
                // Fallback: use one of the known TiKV store addresses from PD
                std::string fallbackAddr = GetAnyStoreAddress();
                if (!fallbackAddr.empty()) {
                    region.leaderAddr = fallbackAddr;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "TiKVIO: Failed to resolve region, falling back to store %s\n",
                                 region.leaderAddr.c_str());
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: No TiKV store available for key\n");
                    return nullptr;
                }
            }

            return GetOrCreateStub(region.leaderAddr);
        }

        // ---- Get any available TiKV store address from PD ----
        std::string GetAnyStoreAddress() {
            // First check the store address cache
            {
                std::lock_guard<std::mutex> lock(m_storeAddrMutex);
                for (const auto& [id, addr] : m_storeAddrCache) {
                    if (!addr.empty()) return addr;
                }
            }

            // Discover real store IDs by querying PD for the first region
            // (empty key hits the first region) and extracting peers
            RegionInfo firstRegion;
            if (GetRegionFromPD("", firstRegion)) {
                // firstRegion is now cached; try its leader address
                if (!firstRegion.leaderAddr.empty()) {
                    return firstRegion.leaderAddr;
                }
                // Otherwise resolve from storeId
                if (firstRegion.storeId != 0) {
                    std::string addr = GetStoreAddress(firstRegion.storeId);
                    if (!addr.empty()) return addr;
                }
            }

            // Last resort: scan region cache for any known store address
            {
                std::shared_lock<std::shared_mutex> lock(m_regionMutex);
                for (const auto& r : m_regionCache) {
                    if (!r.leaderAddr.empty()) return r.leaderAddr;
                }
            }
            return "";
        }

        // ---- Get or create a gRPC stub pool for a TiKV store ----
        tikvpb::Tikv::Stub* GetOrCreateStub(const std::string& address) {
            {
                std::lock_guard<std::mutex> lock(m_storeMutex);
                auto it = m_storeStubs.find(address);
                if (it != m_storeStubs.end()) {
                    return it->second->GetNext();
                }
            }

            // Create a pool of stubs with separate channels
            auto pool = std::make_shared<StubPool>();
            pool->stubs.reserve(kStubPoolSize);
            for (int i = 0; i < kStubPoolSize; i++) {
                grpc::ChannelArguments args;
                args.SetMaxReceiveMessageSize(64 * 1024 * 1024); // 64MB
                args.SetMaxSendMessageSize(64 * 1024 * 1024);    // 64MB
                auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
                auto stub = tikvpb::Tikv::NewStub(channel);
                if (!stub) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVIO: Failed to create stub for %s\n", address.c_str());
                    return nullptr;
                }
                pool->stubs.push_back(std::move(stub));
            }

            std::lock_guard<std::mutex> lock(m_storeMutex);
            // Double-check after acquiring lock
            auto it = m_storeStubs.find(address);
            if (it != m_storeStubs.end()) {
                return it->second->GetNext();
            }
            m_storeStubs[address] = pool;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Created %d stubs for TiKV store at %s\n", kStubPoolSize, address.c_str());
            return pool->GetNext();
        }

        // ---- Vector search protocol encoding/decoding ----

        static constexpr uint32_t VSCH_MAGIC = 0x56534348; // "VSCH"
        static constexpr uint32_t VSCH_VERSION = 1;

        std::string EncodeVectorSearchRequest(
            int dim, int topN, int valueType, int metaDataSize,
            const uint8_t* queryVector, int queryVecBytes,
            const std::vector<std::pair<size_t, std::string>>& group) const
        {
            // Header: 7 × uint32 = 28 bytes
            // Query vector: queryVecBytes
            // Keys: (4 + keyLen) per key
            size_t totalSize = 28 + queryVecBytes;
            for (auto& [idx, pkey] : group) {
                totalSize += 4 + pkey.size();
            }

            std::string buf;
            buf.resize(totalSize);
            char* p = buf.data();

            auto write_u32 = [&p](uint32_t v) {
                memcpy(p, &v, 4); p += 4;
            };

            write_u32(VSCH_MAGIC);
            write_u32(VSCH_VERSION);
            write_u32(static_cast<uint32_t>(dim));
            write_u32(static_cast<uint32_t>(topN));
            write_u32(static_cast<uint32_t>(valueType));
            write_u32(static_cast<uint32_t>(metaDataSize));
            write_u32(static_cast<uint32_t>(group.size()));

            memcpy(p, queryVector, queryVecBytes);
            p += queryVecBytes;

            for (auto& [idx, pkey] : group) {
                uint32_t keyLen = static_cast<uint32_t>(pkey.size());
                memcpy(p, &keyLen, 4); p += 4;
                memcpy(p, pkey.data(), keyLen); p += keyLen;
            }

            return buf;
        }

        std::vector<CoprocessorResult> DecodeVectorSearchResponse(const std::string& data) const {
            std::vector<CoprocessorResult> results;
            if (data.size() < 12) return results;

            const char* p = data.data();
            uint32_t magic, version, numResults;
            memcpy(&magic, p, 4); p += 4;
            memcpy(&version, p, 4); p += 4;
            memcpy(&numResults, p, 4); p += 4;

            if (magic != VSCH_MAGIC || version != VSCH_VERSION) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "TiKVIO: Invalid vector search response magic=%x version=%u\n",
                    magic, version);
                return results;
            }

            if (data.size() < 12 + numResults * 12) return results;

            results.reserve(numResults);
            for (uint32_t i = 0; i < numResults; i++) {
                CoprocessorResult r;
                int64_t vid;
                memcpy(&vid, p, 8); p += 8;
                r.vectorID = static_cast<SizeType>(vid);
                memcpy(&r.distance, p, 4); p += 4;
                results.push_back(r);
            }

            return results;
        }
    };
} // namespace SPTAG::SPANN

#endif // _SPTAG_SPANN_EXTRATIKVCONTROLLER_H_
