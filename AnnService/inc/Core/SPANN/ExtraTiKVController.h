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
#include "kvproto/pdpb.pb.h"
#include "kvproto/pdpb.grpc.pb.h"

#include <map>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <climits>
#include <future>
#include <mutex>
#include <condition_variable>
#include <shared_mutex>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <thread>

namespace SPTAG::SPANN
{
    // Simple sharded LRU cache for posting vector counts.
    // Thread-safe: each shard has its own mutex.
    class PostingCountCache {
    public:
        PostingCountCache(size_t capacity = 100000, int shards = 16)
            : m_shards(shards), m_capacity(std::max(capacity / shards, (size_t)1)) {
            m_data.resize(shards);
            m_mutexes = std::make_unique<std::mutex[]>(shards);
        }

        // Returns (count, true) on hit, (0, false) on miss.
        std::pair<int, bool> Get(SizeType headID) {
            int s = Shard(headID);
            std::lock_guard<std::mutex> lock(m_mutexes[s]);
            auto& shard = m_data[s];
            auto it = shard.map.find(headID);
            if (it == shard.map.end()) return {0, false};
            // Move to front (most recently used)
            shard.order.splice(shard.order.begin(), shard.order, it->second);
            return {it->second->second, true};
        }

        void Put(SizeType headID, int count) {
            int s = Shard(headID);
            std::lock_guard<std::mutex> lock(m_mutexes[s]);
            auto& shard = m_data[s];
            auto it = shard.map.find(headID);
            if (it != shard.map.end()) {
                it->second->second = count;
                shard.order.splice(shard.order.begin(), shard.order, it->second);
                return;
            }
            // Evict if full
            if (shard.map.size() >= m_capacity) {
                auto& back = shard.order.back();
                shard.map.erase(back.first);
                shard.order.pop_back();
            }
            shard.order.emplace_front(headID, count);
            shard.map[headID] = shard.order.begin();
        }

        void Remove(SizeType headID) {
            int s = Shard(headID);
            std::lock_guard<std::mutex> lock(m_mutexes[s]);
            auto& shard = m_data[s];
            auto it = shard.map.find(headID);
            if (it != shard.map.end()) {
                shard.order.erase(it->second);
                shard.map.erase(it);
            }
        }

    private:
        int Shard(SizeType headID) const { return static_cast<unsigned>(headID) % m_shards; }

        struct ShardData {
            std::list<std::pair<SizeType, int>> order; // front = MRU
            std::unordered_map<SizeType, std::list<std::pair<SizeType, int>>::iterator> map;
        };

        int m_shards;
        size_t m_capacity; // per shard
        std::vector<ShardData> m_data;
        std::unique_ptr<std::mutex[]> m_mutexes;
    };

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
        // ---- Async (gRPC CompletionQueue) infrastructure ----
        //
        // For high-fan-out scenarios (e.g. SPFresh AddIndex Phase 2 issuing
        // 12-30 BatchPut RPCs per call) we want to avoid spawning a fresh OS
        // thread per RPC. Instead, async RPCs are spread across a small set of
        // CompletionQueues serviced by background pump threads. The caller
        // blocks on an AsyncBatch wait-group that counts down as completions arrive.
        //
        // Lifetime: cq + pump are created lazily on first Async* call (after
        // the TiKVIO is fully constructed) and torn down in ShutDown().

        class AsyncBatch {
        public:
            // Initialize the wait-group with the EXACT number of RPCs that will
            // be submitted before any Wait() can race with completions. Caller
            // must Add(n) once up-front, then submit n RPCs each tagged with
            // this batch.
            void Add(int n) { m_pending.fetch_add(n, std::memory_order_relaxed); }

            // Called by the pump thread when a tagged RPC completes.
            // ok=false means the RPC was cancelled / cq shutdown / status not OK
            // / region_error / response.error non-empty (caller decides which).
            void Done(bool ok) {
                if (!ok) m_failures.fetch_add(1, std::memory_order_relaxed);
                if (m_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(m_mu);
                    m_cv.notify_all();
                }
            }

            // Block the calling thread until every submitted RPC has signalled.
            void Wait() {
                std::unique_lock<std::mutex> lk(m_mu);
                m_cv.wait(lk, [&] { return m_pending.load(std::memory_order_acquire) == 0; });
            }

            int Failures() const { return m_failures.load(std::memory_order_acquire); }

        private:
            std::atomic<int> m_pending{0};
            std::atomic<int> m_failures{0};
            std::mutex m_mu;
            std::condition_variable m_cv;
        };

        enum class AsyncWaitKind : int {
            MultiGetPageBuffer = 0,
            MultiGetString,
            MultiScanPostings,
            CountBatchGet,
            AddIndexMultiChunk,
            CollectReAssignMultiChunk,
            AddIndexSingleKeyGet,
            AddIndexSingleKeyPut,
            Count
        };

        void RecordAsyncWait(AsyncWaitKind kind, uint64_t batchSize, uint64_t waitUs) {
            int k = static_cast<int>(kind);
            if (k < 0 || k >= static_cast<int>(AsyncWaitKind::Count)) return;
            m_asyncWaitUs[k][AsyncWaitHistBucketOf(waitUs)].fetch_add(1, std::memory_order_relaxed);
            m_asyncWaitTotalUs[k].fetch_add(waitUs, std::memory_order_relaxed);
            m_asyncWaitBatchTotal[k].fetch_add(batchSize, std::memory_order_relaxed);
            m_asyncWaitSampleCount[k].fetch_add(1, std::memory_order_relaxed);
        }

        void LogAsyncWaitStatsAndReset(int layer) override {
            for (int k = 0; k < static_cast<int>(AsyncWaitKind::Count); k++) {
                uint64_t samples = m_asyncWaitSampleCount[k].exchange(0, std::memory_order_relaxed);
                uint64_t totalUs = m_asyncWaitTotalUs[k].exchange(0, std::memory_order_relaxed);
                uint64_t totalBatch = m_asyncWaitBatchTotal[k].exchange(0, std::memory_order_relaxed);
                uint64_t counts[kAsyncWaitHistBuckets];
                for (int i = 0; i < kAsyncWaitHistBuckets; i++) {
                    counts[i] = m_asyncWaitUs[k][i].exchange(0, std::memory_order_relaxed);
                }
                if (samples == 0) continue;

                char buf[2048];
                int n = snprintf(buf, sizeof(buf),
                    "[DIAG-ASYNC] layer %d %s waits=%lu avg=%.2fus avgBatch=%.2f histo[bucket=count]:",
                    layer, AsyncWaitKindName(static_cast<AsyncWaitKind>(k)),
                    (unsigned long)samples,
                    samples ? static_cast<double>(totalUs) / samples : 0.0,
                    samples ? static_cast<double>(totalBatch) / samples : 0.0);
                for (int i = 0; i < kAsyncWaitHistBuckets && n < static_cast<int>(sizeof(buf)); i++) {
                    uint64_t c = counts[i];
                    if (c == 0) continue;
                    uint64_t lo = (i == 0) ? 0ULL : (1ULL << i);
                    n += snprintf(buf + n, sizeof(buf) - n, " %lu%s+:%lu",
                                  (unsigned long)lo,
                                  (i == kAsyncWaitHistBuckets - 1) ? ">=" : "",
                                  (unsigned long)c);
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%s\n", buf);
            }

            if (m_asyncRpcMaxInflight > 0) {
                uint64_t waitSamples = m_asyncRpcThrottleWaitSamples.exchange(0, std::memory_order_relaxed);
                uint64_t waitTotalUs = m_asyncRpcThrottleWaitTotalUs.exchange(0, std::memory_order_relaxed);
                uint64_t maxObserved = m_asyncRpcMaxInflightObserved.exchange(0, std::memory_order_relaxed);
                uint64_t currentInflight = m_asyncRpcInflight.load(std::memory_order_acquire);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[DIAG-ASYNC] layer %d RpcThrottle limit=%d current=%lu maxObserved=%lu waitSamples=%lu avgWait=%.2fus\n",
                    layer,
                    m_asyncRpcMaxInflight,
                    (unsigned long)currentInflight,
                    (unsigned long)std::max(maxObserved, currentInflight),
                    (unsigned long)waitSamples,
                    waitSamples ? static_cast<double>(waitTotalUs) / waitSamples : 0.0);
            }
        }

        // Per-RPC heap-allocated tag base. Concrete tag types embed their own
        // grpc::ClientContext / request / response / status / Reader, then
        // implement OnComplete() to interpret the response and signal the
        // batch. The pump thread does:  raw_tag->OnComplete(ok); delete raw_tag;
        struct AsyncTagBase {
            std::shared_ptr<AsyncBatch> batch;
            std::atomic<int>* result_slot = nullptr;   // optional
            std::string keyForCacheInvalidation;
            TiKVIO* owner = nullptr;
            virtual ~AsyncTagBase() = default;
            bool releaseAsyncPermit = false;
            // Returns true on success (response was OK and parseable).
            // ok=false comes from the cq itself (cancel / shutdown / network).
            virtual bool OnComplete(bool ok) = 0;
        };

        // BatchPut tag (used by AsyncRawBatchPut / AsyncAppendChunkAndUpdateCount).
        struct AsyncBatchPutTag : AsyncTagBase {
            grpc::ClientContext ctx;
            kvrpcpb::RawBatchPutRequest request;
            kvrpcpb::RawBatchPutResponse response;
            grpc::Status status;
            std::unique_ptr<grpc::ClientAsyncResponseReader<kvrpcpb::RawBatchPutResponse>> rpc;
            bool OnComplete(bool ok) override {
                if (!ok || !status.ok()) return false;
                if (response.has_region_error()) {
                    if (owner) owner->InvalidateRegionCache(keyForCacheInvalidation);
                    return false;
                }
                return response.error().empty();
            }
        };

        // Single-key Get tag (used by AsyncRawGet for SPFresh single-key path).
        // out_value is filled iff the RPC succeeded AND the key was found.
        // not_found is reported as success with out_value->empty() AND out_found=false.
        struct AsyncGetTag : AsyncTagBase {
            grpc::ClientContext ctx;
            kvrpcpb::RawGetRequest request;
            kvrpcpb::RawGetResponse response;
            grpc::Status status;
            std::unique_ptr<grpc::ClientAsyncResponseReader<kvrpcpb::RawGetResponse>> rpc;
            std::string* out_value = nullptr;
            std::atomic<bool>* out_found = nullptr;     // optional
            bool OnComplete(bool ok) override {
                if (!ok || !status.ok()) return false;
                if (response.has_region_error()) {
                    if (owner) owner->InvalidateRegionCache(keyForCacheInvalidation);
                    return false;
                }
                if (!response.error().empty()) return false;
                if (response.not_found()) {
                    if (out_found) out_found->store(false, std::memory_order_release);
                    if (out_value) out_value->clear();
                    return true;
                }
                if (out_value) *out_value = response.value();
                if (out_found) out_found->store(true, std::memory_order_release);
                return true;
            }
        };

        // Single-key Put tag (used by AsyncRawPut).
        struct AsyncPutTag : AsyncTagBase {
            grpc::ClientContext ctx;
            kvrpcpb::RawPutRequest request;
            kvrpcpb::RawPutResponse response;
            grpc::Status status;
            std::unique_ptr<grpc::ClientAsyncResponseReader<kvrpcpb::RawPutResponse>> rpc;
            bool OnComplete(bool ok) override {
                if (!ok || !status.ok()) return false;
                if (response.has_region_error()) {
                    if (owner) owner->InvalidateRegionCache(keyForCacheInvalidation);
                    return false;
                }
                return response.error().empty();
            }
        };

        // Region-group BatchGet tag. Results are written into out_values by
        // original caller index; missing keys remain empty and are considered
        // successful. The caller handles sync fallback for failed groups after
        // AsyncBatch::Wait().
        struct AsyncBatchGetTag : AsyncTagBase {
            grpc::ClientContext ctx;
            kvrpcpb::RawBatchGetRequest request;
            kvrpcpb::RawBatchGetResponse response;
            grpc::Status status;
            std::unique_ptr<grpc::ClientAsyncResponseReader<kvrpcpb::RawBatchGetResponse>> rpc;
            std::vector<std::pair<size_t, std::string>> keys;
            std::vector<std::string>* out_values = nullptr;
            bool OnComplete(bool ok) override {
                if (!ok || !status.ok()) return false;
                if (response.has_region_error()) {
                    if (owner) {
                        for (auto& kv : keys) owner->InvalidateRegionCache(kv.second);
                    }
                    return false;
                }

                std::unordered_map<std::string, std::string> resultMap;
                resultMap.reserve(static_cast<size_t>(response.pairs_size()));
                bool hasPairError = false;
                for (int i = 0; i < response.pairs_size(); i++) {
                    const auto& pair = response.pairs(i);
                    if (pair.has_error()) {
                        hasPairError = true;
                        if (owner) owner->InvalidateRegionCache(pair.key());
                        continue;
                    }
                    if (!pair.value().empty()) {
                        resultMap[pair.key()] = pair.value();
                    }
                }
                if (hasPairError) return false;
                if (out_values) {
                    for (auto& kv : keys) {
                        auto it = resultMap.find(kv.second);
                        if (it != resultMap.end()) {
                            (*out_values)[kv.first] = std::move(it->second);
                        }
                    }
                }
                return true;
            }
        };

        // One-page RawScan tag. MultiScanPostings issues pages in rounds so the
        // CQ pump never performs a long scan loop or synchronous retry itself.
        struct AsyncScanPageTag : AsyncTagBase {
            grpc::ClientContext ctx;
            kvrpcpb::RawScanRequest request;
            kvrpcpb::RawScanResponse response;
            grpc::Status status;
            std::unique_ptr<grpc::ClientAsyncResponseReader<kvrpcpb::RawScanResponse>> rpc;
            std::string* out_posting = nullptr;
            std::string* out_next_cursor = nullptr;
            std::atomic<int>* out_count = nullptr;
            std::atomic<bool>* out_more = nullptr;
            bool OnComplete(bool ok) override {
                if (!ok || !status.ok()) return false;
                if (response.has_region_error()) {
                    if (owner) owner->InvalidateRegionCache(keyForCacheInvalidation);
                    return false;
                }

                int count = response.kvs_size();
                if (out_posting) {
                    for (int i = 0; i < count; i++) {
                        out_posting->append(response.kvs(i).value());
                    }
                }
                if (out_count) out_count->store(count, std::memory_order_release);
                bool more = count >= 1024;
                if (out_more) out_more->store(more, std::memory_order_release);
                if (more && out_next_cursor && count > 0) {
                    *out_next_cursor = response.kvs(count - 1).key();
                    out_next_cursor->push_back('\x00');
                }
                return true;
            }
        };

        TiKVIO(const std::string& pdAddresses, const std::string& keyPrefix, bool useMultiChunkPosting, int postingCountCacheCapacity, int asyncRpcMaxInflight = 0)
            : m_keyPrefix(keyPrefix), m_useMultiChunkPosting(useMultiChunkPosting),
              m_asyncRpcMaxInflight(std::max(asyncRpcMaxInflight, 0))
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

            // Initialize posting count cache for multi-chunk mode
            if (m_useMultiChunkPosting) {
                postingCountCacheCapacity = max(postingCountCacheCapacity, 1);
                m_postingCountCache = std::make_unique<PostingCountCache>(postingCountCacheCapacity, 16);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "PostingCountCache initialized (capacity=%d, shards=16)\n", postingCountCacheCapacity);
            }
            m_available = true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Initialized with key prefix '%s'\n", m_keyPrefix.c_str());
            if (m_asyncRpcMaxInflight > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "TiKVIO: Async RPC in-flight limit enabled: %d\n",
                    m_asyncRpcMaxInflight);
            }

            // Start async-RPC pump threads now that the object is fully
            // initialized. Multiple CompletionQueues prevent heavy completion
            // parsing/copying in one pump from serializing every async RPC.
            m_asyncCqs.reserve(kAsyncCompletionQueueCount);
            for (int i = 0; i < kAsyncCompletionQueueCount; i++) {
                m_asyncCqs.emplace_back(std::make_unique<grpc::CompletionQueue>());
            }
            m_asyncPumpRunning.store(true, std::memory_order_release);
            m_asyncPumpThreads.reserve(m_asyncCqs.size());
            for (size_t i = 0; i < m_asyncCqs.size(); i++) {
                m_asyncPumpThreads.emplace_back([this, i]() { AsyncPumpLoop(i); });
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "TiKVIO: Async RPC completion queues=%zu\n", m_asyncCqs.size());
        }

        ~TiKVIO() override {
            ShutDown();
        }

        void ShutDown() override {
            // Tear down the async pump first so no new completions race with
            // stub destruction. Order: stop accepting new ops -> Shutdown cq
            // (drains in-flight tags as cancelled) -> join pump -> drop stubs.
            bool wasRunning = m_asyncPumpRunning.exchange(false, std::memory_order_acq_rel);
            if (wasRunning) {
                m_asyncRpcCv.notify_all();
                for (auto& cq : m_asyncCqs) cq->Shutdown();
                for (auto& thread : m_asyncPumpThreads) {
                    if (thread.joinable()) thread.join();
                }
                m_asyncPumpThreads.clear();
                m_asyncCqs.clear();
            }
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
                    // [FIX] Distinguish a true "key absent" from an RPC/region failure.
                    // Callers (e.g. Append RMW) MUST be able to tell these apart
                    // to avoid overwriting an existing posting with an empty
                    // value when a transient Get failure was misread as "empty".
                    return ErrorCode::Key_NotFound;
                }

                *value = response.value();
                return ErrorCode::Success;
            }
        }

        ErrorCode Get(const SizeType key, std::string* value,
                      const std::chrono::microseconds& timeout,
                      std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            if (m_useMultiChunkPosting) {
                return ScanPosting(key, value, timeout);
            }
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
            if (m_useMultiChunkPosting) {
                auto delRet = DeletePosting(key);
                if (delRet != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PutPostingToDB: DeletePosting failed for key %d\n", key);
                    return delRet;
                }
                int count = static_cast<int>(value.size());
                auto ret = PutBaseChunkAndCount(key, value, count, timeout, reqs);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PutPostingToDB: PutBaseChunkAndCount failed for key %d\n", key);
                    return ret;
                }
                if (m_postingCountCache) m_postingCountCache->Put(key, count);
                return ErrorCode::Success;
            }
            
            std::string k(reinterpret_cast<const char*>(&key), sizeof(SizeType));
            return Put(k, value, timeout, reqs);
        }

        // ---- Delete operations ----

        ErrorCode Delete(SizeType key) override {
            if (m_useMultiChunkPosting) {
                auto countRet = DeletePostingCount(key);
                if (countRet != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "DeletePostingFromDB: DeletePostingCount failed for headID %d\n", key);
                }
                if (m_postingCountCache) m_postingCountCache->Remove(key);
                return DeletePosting(key);
            }

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

            if (m_useMultiChunkPosting) {
                auto [count, hit] = m_postingCountCache->Get(key);
                if (!hit) {
                    count = GetPostingCount(key, std::chrono::microseconds(5000000));
                    if (count < 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "GetCachedPostingCount: TiKV error for headID %d, returning 0\n", key);
                        return ErrorCode::Posting_SizeError;
                    }
                    m_postingCountCache->Put(key, count);
                }
                { static std::atomic<int> _logOnce{0}; if (_logOnce.fetch_add(1) == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[PATH] Append using MULTI-CHUNK AppendChunk path\n"); }

                int newCount = count + value.size();
                auto ret =PutChunkAndCount(key, value, newCount, timeout, reqs);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MultiChunkAppend failed for %lld!\n", (std::int64_t)key);
                    return ret;
                }
                if (m_postingCountCache) m_postingCountCache->Put(key, newCount);
                size = newCount;
            } else {
                { static std::atomic<int> _logOnce{0}; if (_logOnce.fetch_add(1) == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[PATH] Append using SINGLE-KEY Get+Put path (no multi-chunk)\n"); }
                std::string fullPosting;
                auto ret = Get(key, &fullPosting, MaxTimeout, reqs);
                if (ret == ErrorCode::Key_NotFound) {
                    fullPosting.clear();
                } else if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge failed to read existing posting for %lld before append.\n", (std::int64_t)key);
                    return ret;
                }

                fullPosting.append(value);
                size = static_cast<int>(fullPosting.size());
                if ((ret = Put(key, fullPosting, MaxTimeout, reqs)) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge failed for %lld! Posting Size:%d\n", (std::int64_t)key, size);
                    return ret;
                }
            }
            return ErrorCode::Success;
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
            if (m_useMultiChunkPosting) {
                return MultiScanPostings(keys, values, timeout);
            }

            if (keys.empty()) return ErrorCode::Success;

            struct PendingRegionGroup {
                std::string leaderAddr;
                RegionInfo region{};
                std::vector<std::pair<size_t, std::string>> keys;
            };

            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                std::string k(reinterpret_cast<const char*>(&keys[i]), sizeof(SizeType));
                prefixedKeys[i] = MakePrefixedKey(k);
                values[i].SetAvailableSize(0);
            }

            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region{};
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

            std::vector<PendingRegionGroup> groups;
            groups.reserve(regionGroups.size());
            for (auto& kv : regionGroups) {
                PendingRegionGroup g;
                g.leaderAddr = kv.first.leaderAddr;
                g.region = kv.second.region;
                g.keys = kv.second.keys;
                groups.push_back(std::move(g));
            }

            std::vector<std::string> tmpValues(keys.size());
            auto batch = std::make_shared<AsyncBatch>();
            batch->Add(static_cast<int>(groups.size()));
            std::vector<std::atomic<int>> okFlags(groups.size());
            for (auto& f : okFlags) f.store(0, std::memory_order_relaxed);

            for (size_t i = 0; i < groups.size(); i++) {
                kvrpcpb::Context context;
                SetContextFromRegion(&context, groups[i].region);
                AsyncRawBatchGetPrefixed(groups[i].leaderAddr, context, groups[i].keys,
                                         &tmpValues, batch, &okFlags[i], timeout);
            }
            auto waitBegin = std::chrono::high_resolution_clock::now();
            batch->Wait();
            RecordAsyncWait(AsyncWaitKind::MultiGetPageBuffer, groups.size(),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - waitBegin).count()));

            int failCount = 0;
            for (size_t i = 0; i < groups.size(); i++) {
                if (okFlags[i].load(std::memory_order_acquire) == 1) continue;
                for (auto& [idx, pkey] : groups[i].keys) {
                    InvalidateRegionCache(pkey);
                    std::string val;
                    auto ret = Get(keys[idx], &val, timeout, reqs);
                    if (ret == ErrorCode::Success) {
                        tmpValues[idx] = std::move(val);
                    } else if (ret != ErrorCode::Key_NotFound) {
                        failCount++;
                    }
                }
            }

            for (size_t i = 0; i < tmpValues.size(); i++) {
                const auto& val = tmpValues[i];
                if (val.empty()) continue;
                if (val.size() > values[i].GetPageSize()) {
                    values[i].ReservePageBuffer(val.size());
                }
                memcpy(values[i].GetBuffer(), val.data(), val.size());
                values[i].SetAvailableSize(static_cast<int>(val.size()));
            }

            return failCount == 0 ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys,
                           std::vector<std::string>* values,
                           const std::chrono::microseconds& timeout,
                           std::vector<Helper::AsyncReadRequest>* reqs) override
        {
            if (keys.empty()) return ErrorCode::Success;

            struct PendingRegionGroup {
                std::string leaderAddr;
                RegionInfo region{};
                std::vector<std::pair<size_t, std::string>> keys;
            };

            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                prefixedKeys[i] = MakePrefixedKey(keys[i]);
            }
            values->assign(keys.size(), std::string());

            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region{};
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

            std::vector<PendingRegionGroup> groups;
            groups.reserve(regionGroups.size());
            for (auto& kv : regionGroups) {
                PendingRegionGroup g;
                g.leaderAddr = kv.first.leaderAddr;
                g.region = kv.second.region;
                g.keys = kv.second.keys;
                groups.push_back(std::move(g));
            }

            auto batch = std::make_shared<AsyncBatch>();
            batch->Add(static_cast<int>(groups.size()));
            std::vector<std::atomic<int>> okFlags(groups.size());
            for (auto& f : okFlags) f.store(0, std::memory_order_relaxed);

            for (size_t i = 0; i < groups.size(); i++) {
                kvrpcpb::Context context;
                SetContextFromRegion(&context, groups[i].region);
                AsyncRawBatchGetPrefixed(groups[i].leaderAddr, context, groups[i].keys,
                                         values, batch, &okFlags[i], timeout);
            }
            auto waitBegin = std::chrono::high_resolution_clock::now();
            batch->Wait();
            RecordAsyncWait(AsyncWaitKind::MultiGetString, groups.size(),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - waitBegin).count()));

            int failCount = 0;
            for (size_t i = 0; i < groups.size(); i++) {
                if (okFlags[i].load(std::memory_order_acquire) == 1) continue;
                for (auto& [idx, pkey] : groups[i].keys) {
                    InvalidateRegionCache(pkey);
                    std::string val;
                    auto ret = Get(keys[idx], &val, timeout, reqs);
                    if (ret == ErrorCode::Success) {
                        (*values)[idx] = std::move(val);
                    } else if (ret != ErrorCode::Key_NotFound) {
                        failCount++;
                    }
                }
            }

            return failCount == 0 ? ErrorCode::Success : ErrorCode::Fail;
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

        ErrorCode MultiGetWithStatus(const std::vector<std::string>& keys,
                                     std::vector<std::string>* values,
                                     std::vector<uint8_t>* okFlags,
                                     const std::chrono::microseconds& timeout,
                                     std::vector<Helper::AsyncReadRequest>* reqs)
        {
            if (keys.empty()) {
                values->clear();
                okFlags->clear();
                return ErrorCode::Success;
            }

            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                prefixedKeys[i] = MakePrefixedKey(keys[i]);
            }
            values->assign(keys.size(), std::string());
            okFlags->assign(keys.size(), 0);

            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region{};
                std::string addr;
                uint64_t rid = 0;
                if (FindRegionForKey(prefixedKeys[i], region) && !region.leaderAddr.empty()) {
                    addr = region.leaderAddr;
                    rid = region.regionId;
                } else {
                    addr = GetAnyStoreAddress();
                }
                auto& group = regionGroups[{addr, rid}];
                if (group.keys.empty()) group.region = region;
                group.keys.push_back({i, prefixedKeys[i]});
            }

            struct PendingRegionGroup {
                std::string leaderAddr;
                RegionInfo region{};
                std::vector<std::pair<size_t, std::string>> keys;
            };

            std::vector<PendingRegionGroup> groups;
            groups.reserve(regionGroups.size());
            for (auto& kv : regionGroups) {
                PendingRegionGroup group;
                group.leaderAddr = kv.first.leaderAddr;
                group.region = kv.second.region;
                group.keys = std::move(kv.second.keys);
                groups.push_back(std::move(group));
            }

            auto batch = std::make_shared<AsyncBatch>();
            batch->Add(static_cast<int>(groups.size()));
            std::vector<std::atomic<int>> groupOk(groups.size());
            for (auto& flag : groupOk) flag.store(0, std::memory_order_relaxed);

            for (size_t i = 0; i < groups.size(); i++) {
                kvrpcpb::Context context;
                SetContextFromRegion(&context, groups[i].region);
                AsyncRawBatchGetPrefixed(groups[i].leaderAddr, context, groups[i].keys,
                                         values, batch, &groupOk[i], timeout);
            }
            batch->Wait();

            int failCount = 0;
            for (size_t i = 0; i < groups.size(); i++) {
                if (groupOk[i].load(std::memory_order_acquire) == 1) {
                    for (auto& [idx, pkey] : groups[i].keys) (*okFlags)[idx] = 1;
                    continue;
                }
                for (auto& [idx, pkey] : groups[i].keys) {
                    InvalidateRegionCache(pkey);
                    std::string val;
                    auto ret = Get(keys[idx], &val, timeout, reqs);
                    if (ret == ErrorCode::Success) {
                        (*values)[idx] = std::move(val);
                        (*okFlags)[idx] = 1;
                    } else if (ret == ErrorCode::Key_NotFound) {
                        (*okFlags)[idx] = 1;
                    } else {
                        failCount++;
                    }
                }
            }

            return failCount == 0 ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiPutWithStatus(const std::vector<std::string>& keys,
                                     const std::vector<std::string>& values,
                                     std::vector<uint8_t>* okFlags,
                                     const std::chrono::microseconds& timeout,
                                     std::vector<Helper::AsyncReadRequest>* reqs)
        {
            if (keys.empty()) {
                okFlags->clear();
                return ErrorCode::Success;
            }
            if (keys.size() != values.size()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "TiKVIO::MultiPutWithStatus size mismatch: keys=%zu values=%zu\n",
                    keys.size(), values.size());
                okFlags->assign(keys.size(), 0);
                return ErrorCode::Fail;
            }

            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                prefixedKeys[i] = MakePrefixedKey(keys[i]);
            }
            okFlags->assign(keys.size(), 0);

            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < prefixedKeys.size(); i++) {
                RegionInfo region{};
                std::string addr;
                uint64_t rid = 0;
                if (FindRegionForKey(prefixedKeys[i], region) && !region.leaderAddr.empty()) {
                    addr = region.leaderAddr;
                    rid = region.regionId;
                } else {
                    addr = GetAnyStoreAddress();
                }
                auto& group = regionGroups[{addr, rid}];
                if (group.keys.empty()) group.region = region;
                group.keys.push_back({i, prefixedKeys[i]});
            }

            struct PendingRegionGroup {
                std::string leaderAddr;
                RegionInfo region{};
                std::vector<std::pair<size_t, std::string>> keys;
            };

            std::vector<PendingRegionGroup> groups;
            groups.reserve(regionGroups.size());
            for (auto& kv : regionGroups) {
                PendingRegionGroup group;
                group.leaderAddr = kv.first.leaderAddr;
                group.region = kv.second.region;
                group.keys = std::move(kv.second.keys);
                groups.push_back(std::move(group));
            }

            auto batch = std::make_shared<AsyncBatch>();
            batch->Add(static_cast<int>(groups.size()));
            std::vector<std::atomic<int>> groupOk(groups.size());
            for (auto& flag : groupOk) flag.store(0, std::memory_order_relaxed);

            for (size_t i = 0; i < groups.size(); i++) {
                kvrpcpb::Context context;
                SetContextFromRegion(&context, groups[i].region);
                AsyncRawBatchPutPrefixed(groups[i].leaderAddr, context, groups[i].keys,
                                         values, batch, &groupOk[i], timeout);
            }
            batch->Wait();

            int failCount = 0;
            for (size_t i = 0; i < groups.size(); i++) {
                if (groupOk[i].load(std::memory_order_acquire) == 1) {
                    for (auto& [idx, pkey] : groups[i].keys) (*okFlags)[idx] = 1;
                    continue;
                }
                for (auto& [idx, pkey] : groups[i].keys) {
                    InvalidateRegionCache(pkey);
                    auto ret = RawPutWithRetry(pkey, values[idx], timeout);
                    if (ret == ErrorCode::Success) {
                        (*okFlags)[idx] = 1;
                    } else {
                        failCount++;
                    }
                }
            }

            return failCount == 0 ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiMerge(const std::vector<SizeType>& keys, const std::vector<std::string>& values, 
                                         const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs, std::vector<int>& sizes) override
        {
            if (keys.empty()) {
                sizes.clear();
                return ErrorCode::Success;
            }

            ErrorCode firstErr = ErrorCode::Success;
            sizes.resize(keys.size());
            if (m_useMultiChunkPosting) {
                std::vector<int> fetchedCounts;
                ErrorCode countRet = AsyncGetPostingCounts(keys, &fetchedCounts,
                                            std::chrono::microseconds(5000000));
                if (countRet != ErrorCode::Success || fetchedCounts.size() != keys.size()) {
                    if (fetchedCounts.size() != keys.size()) fetchedCounts.assign(keys.size(), -1);
                    for (size_t i = 0; i < keys.size(); i++) {
                        if (fetchedCounts[i] >= 0) continue;
                        fetchedCounts[i] = GetPostingCount(keys[i], MaxTimeout);
                    }
                }
                for (size_t i = 0; i < keys.size(); i++) {
                    if (fetchedCounts[i] < 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                            "TiKVIO::MultiMerge failed to fetch posting count headID=%d\n", keys[i]);
                        return ErrorCode::Fail;
                    }
                }

                auto batch = std::make_shared<TiKVIO::AsyncBatch>();
                batch->Add(static_cast<int>(keys.size()));

                // std::vector<std::atomic<int>> is non-resizable but in-place
                // constructible to size N with zero-initialization.
                std::vector<std::atomic<int>> okFlags(keys.size());
                for (auto& f : okFlags) f.store(0, std::memory_order_relaxed);

                for (size_t i = 0; i < keys.size(); i++) {
                    int newCount = fetchedCounts[i] + static_cast<int>(values[i].size());
                    AsyncAppendChunkAndUpdateCount(
                        keys[i], values[i], newCount,
                        batch, &okFlags[i], MaxTimeout);
                }

                // ---- Pass 3: single thread blocks on wait-group ----
                auto _waitBegin = std::chrono::high_resolution_clock::now();
                batch->Wait();
                auto _waitEnd = std::chrono::high_resolution_clock::now();
                RecordAsyncWait(TiKVIO::AsyncWaitKind::CollectReAssignMultiChunk,
                    keys.size(),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                        _waitEnd - _waitBegin).count()));

                // ---- Pass 4: process results, sync-retry failures ----   
                int retryCount = 0;
                for (size_t i = 0; i < keys.size(); i++) {
                    sizes[i] = fetchedCounts[i] + static_cast<int>(values[i].size());
                    if (okFlags[i].load(std::memory_order_acquire) == 1) {
                        if (m_postingCountCache) m_postingCountCache->Put(keys[i], sizes[i]);
                    } else {
                        // Sync retry preserves the existing region-cache invalidation
                        // + retry semantics already battle-tested in PutChunkAndCount.
                        retryCount++;
                        auto ret = PutChunkAndCount(keys[i], values[i], sizes[i], MaxTimeout, reqs);
                        if (ret == ErrorCode::Success && m_postingCountCache) {
                            m_postingCountCache->Put(keys[i], sizes[i]);
                        }
                        if (ret != ErrorCode::Success && firstErr == ErrorCode::Success) {
                            firstErr = ret;
                        }
                    }
                }
            }
            else {
                std::vector<std::string> getKeys(keys.size());
                std::vector<std::string> getValues;
                std::vector<uint8_t> getOk;
                for (int i = 0; i < keys.size(); i++) getKeys[i] = std::string(reinterpret_cast<const char*>(&keys[i]), sizeof(SizeType));

                auto _getWaitBegin = std::chrono::high_resolution_clock::now();
                MultiGetWithStatus(getKeys, &getValues, &getOk, MaxTimeout, reqs);
                RecordAsyncWait(TiKVIO::AsyncWaitKind::AddIndexSingleKeyGet,
                    keys.size(),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - _getWaitBegin).count()));

                int activeCount = 0;
                std::vector<std::string> putKeys;
                std::vector<std::string> putValues;
                std::vector<int> putIndexByPending(keys.size(), -1);
                putValues.reserve(keys.size());
                for (size_t i = 0; i < keys.size(); i++) {
                    if (i >= getOk.size() || getOk[i] == 0) {
                        continue;
                    }
                    getValues[i].append(values[i]);
                    sizes[i] = static_cast<int>(getValues[i].size());
                    putIndexByPending[i] = static_cast<int>(putKeys.size());
                    putKeys.push_back(getKeys[i]);
                    putValues.push_back(std::move(getValues[i]));
                    activeCount++;
                }

                std::vector<uint8_t> putOk;
                if (activeCount > 0) {
                    auto _putWaitBegin = std::chrono::high_resolution_clock::now();
                    MultiPutWithStatus(putKeys, putValues, &putOk, MaxTimeout, reqs);
                    RecordAsyncWait(TiKVIO::AsyncWaitKind::AddIndexSingleKeyPut,
                        activeCount,
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - _putWaitBegin).count()));
                }

                // ---- Pass 4: process results + post-merge SplitAsync ----
                int retryCount = 0;
                for (size_t i = 0; i < keys.size(); i++) {
                    int putIndex = putIndexByPending[i];
                    bool needRetry = putIndex < 0 ||
                                    static_cast<size_t>(putIndex) >= putOk.size() ||
                                    putOk[putIndex] == 0;
                    if (needRetry) {
                        retryCount++;
                        auto ret = Merge(keys[i], values[i], MaxTimeout, reqs, sizes[i]);
                        if (ret != ErrorCode::Success && firstErr == ErrorCode::Success) {
                            firstErr = ret;
                        }
                    }
                    if (m_postingCountCache) {
                        m_postingCountCache->Put(keys[i], sizes[i]);
                    }
                }
                if (retryCount > 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "AddIndexAsyncSingleKey: %zu heads, %d sync-retries\n",
                        keys.size(), retryCount);
                }
            }
            return firstErr;
        }

        // ---- MultiPut / MultiDelete operations ----
        // Group keys by (leader address, region id) and issue one RawBatchPut /
        // RawBatchDelete per region. Region groups run in parallel via std::async,
        // mirroring the MultiGet implementation. On region/RPC error we
        // invalidate the region cache and fall back to per-key Put/Delete which
        // each have their own retry loop. The caller blocks on all futures.
        //
        // The keys passed in are RAW caller keys (not yet prefixed with
        // m_keyPrefix). Inputs containing keys already constructed by helpers
        // such as MakeChunkKey/MakeCountKey (which already include the prefix)
        // must use the *Prefixed variants below.

        ErrorCode MultiPut(const std::vector<std::string>& keys,
                           const std::vector<std::string>& values,
                           const std::chrono::microseconds& timeout,
                           std::vector<Helper::AsyncReadRequest>* reqs)
        {
            if (keys.empty()) return ErrorCode::Success;
            if (keys.size() != values.size()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "TiKVIO::MultiPut size mismatch: keys=%zu values=%zu\n",
                    keys.size(), values.size());
                return ErrorCode::Fail;
            }
            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                prefixedKeys[i] = MakePrefixedKey(keys[i]);
            }
            return MultiPutPrefixed(prefixedKeys, values, timeout, reqs);
        }

        ErrorCode MultiDelete(const std::vector<std::string>& keys,
                              const std::chrono::microseconds& timeout)
        {
            if (keys.empty()) return ErrorCode::Success;
            std::vector<std::string> prefixedKeys(keys.size());
            for (size_t i = 0; i < keys.size(); i++) {
                prefixedKeys[i] = MakePrefixedKey(keys[i]);
            }
            return MultiDeletePrefixed(prefixedKeys, timeout);
        }

        // ScanPrefix: walks `prefix` using paged RawScan and returns logical
        // (key, value) pairs with the TiKVIO physical prefix stripped off.
        // Used by durable WALs (e.g. BatchAppendWAL) to recover entries
        // persisted before a crash.
        ErrorCode ScanPrefix(const std::string& prefix,
                             std::vector<std::pair<std::string, std::string>>& out,
                             std::size_t maxEntries) override
        {
            const auto timeout = std::chrono::microseconds(5'000'000);
            std::string physicalPrefix = MakePrefixedKey(prefix);
            // RawScan end_key: a key strictly greater than every key in the
            // prefix. Increment last byte; if it overflows append 0xff.
            std::string endKey = physicalPrefix;
            while (!endKey.empty() && static_cast<unsigned char>(endKey.back()) == 0xFF) {
                endKey.pop_back();
            }
            if (endKey.empty()) {
                endKey = physicalPrefix + std::string(1, '\xFF');
            } else {
                endKey.back() = static_cast<char>(static_cast<unsigned char>(endKey.back()) + 1);
            }

            std::string cursor = physicalPrefix;
            const int pageLimit = 1024;
            for (;;) {
                int attempt = 0;
                bool advanced = false;
                std::string lastKey;
                int count = 0;
                for (; attempt < 10; attempt++) {
                    auto stub = GetStubForKey(cursor);
                    if (!stub) { RetryBackoff(attempt); continue; }

                    kvrpcpb::RawScanRequest request;
                    request.set_start_key(cursor);
                    request.set_end_key(endKey);
                    request.set_limit(pageLimit);
                    SetContext(request.mutable_context(), cursor);

                    kvrpcpb::RawScanResponse response;
                    grpc::ClientContext ctx;
                    SetDeadline(ctx, timeout);

                    auto status = stub->RawScan(&ctx, request, &response);
                    if (!status.ok()) {
                        if (ShouldLogRetry(attempt))
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "TiKVIO::ScanPrefix gRPC error (attempt %d): %s\n",
                                attempt + 1, status.error_message().c_str());
                        InvalidateRegionCache(cursor);
                        RetryBackoff(attempt);
                        continue;
                    }
                    if (response.has_region_error()) {
                        InvalidateRegionCache(cursor);
                        RetryBackoff(attempt);
                        continue;
                    }
                    count = response.kvs_size();
                    for (int i = 0; i < count; i++) {
                        const auto& kv = response.kvs(i);
                        const std::string& k = kv.key();
                        if (k.size() < physicalPrefix.size()) continue;
                        out.emplace_back(k.substr(physicalPrefix.size() - prefix.size()), kv.value());
                        if (maxEntries > 0 && out.size() >= maxEntries) {
                            return ErrorCode::Success;
                        }
                    }
                    if (count > 0) {
                        lastKey = response.kvs(count - 1).key();
                        advanced = true;
                    }
                    break;
                }
                if (attempt >= 10) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "TiKVIO::ScanPrefix exhausted retries\n");
                    return ErrorCode::Fail;
                }
                if (!advanced || count < pageLimit) {
                    return ErrorCode::Success;
                }
                // Advance cursor past the last seen key.
                cursor = lastKey + std::string(1, '\0');
            }
        }

        // Variants that accept already-prefixed keys (used by chunk/count helpers
        // that produce keys via MakeChunkKey / MakeCountKey).
        ErrorCode MultiPutPrefixed(const std::vector<std::string>& prefixedKeys,
                                   const std::vector<std::string>& values,
                                   const std::chrono::microseconds& timeout,
                                   std::vector<Helper::AsyncReadRequest>* reqs)
        {
            if (prefixedKeys.empty()) return ErrorCode::Success;

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

            std::atomic<int> failCount{0};
            std::vector<std::future<void>> futures;
            futures.reserve(regionGroups.size());

            for (auto& kv : regionGroups) {
                auto* gkey = &kv.first;
                auto* rg = &kv.second;
                futures.push_back(std::async(std::launch::async,
                                             [this, gkey, rg, &prefixedKeys, &values, &timeout, &failCount]() {
                    auto& group = rg->keys;
                    auto* stub = GetOrCreateStub(gkey->leaderAddr);
                    bool needFallback = false;

                    if (stub) {
                        kvrpcpb::RawBatchPutRequest request;
                        SetContextFromRegion(request.mutable_context(), rg->region);
                        for (auto& [idx, pkey] : group) {
                            auto* p = request.add_pairs();
                            p->set_key(pkey);
                            p->set_value(values[idx]);
                        }
                        kvrpcpb::RawBatchPutResponse response;
                        grpc::ClientContext ctx;
                        SetDeadline(ctx, timeout);
                        auto status = stub->RawBatchPut(&ctx, request, &response);
                        if (!status.ok() || response.has_region_error() || !response.error().empty()) {
                            for (auto& [idx, pkey] : group) InvalidateRegionCache(pkey);
                            needFallback = true;
                        }
                    } else {
                        needFallback = true;
                    }

                    if (needFallback) {
                        // Per-key Put has its own retry loop and will refresh
                        // region routing on each attempt. Reuse the existing
                        // RawPutWithRetry helper for already-prefixed keys.
                        for (auto& [idx, pkey] : group) {
                            auto ret = RawPutWithRetry(pkey, values[idx], timeout);
                            if (ret != ErrorCode::Success) {
                                failCount.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }));
            }
            for (auto& f : futures) f.get();
            return failCount.load() == 0 ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiDeletePrefixed(const std::vector<std::string>& prefixedKeys,
                                      const std::chrono::microseconds& timeout)
        {
            if (prefixedKeys.empty()) return ErrorCode::Success;

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

            std::atomic<int> failCount{0};
            std::vector<std::future<void>> futures;
            futures.reserve(regionGroups.size());

            for (auto& kv : regionGroups) {
                auto* gkey = &kv.first;
                auto* rg = &kv.second;
                futures.push_back(std::async(std::launch::async,
                                             [this, gkey, rg, &timeout, &failCount]() {
                    auto& group = rg->keys;
                    auto* stub = GetOrCreateStub(gkey->leaderAddr);
                    bool needFallback = false;

                    if (stub) {
                        kvrpcpb::RawBatchDeleteRequest request;
                        SetContextFromRegion(request.mutable_context(), rg->region);
                        for (auto& [idx, pkey] : group) {
                            request.add_keys(pkey);
                        }
                        kvrpcpb::RawBatchDeleteResponse response;
                        grpc::ClientContext ctx;
                        SetDeadline(ctx, timeout);
                        auto status = stub->RawBatchDelete(&ctx, request, &response);
                        if (!status.ok() || response.has_region_error() || !response.error().empty()) {
                            for (auto& [idx, pkey] : group) InvalidateRegionCache(pkey);
                            needFallback = true;
                        }
                    } else {
                        needFallback = true;
                    }

                    if (needFallback) {
                        for (auto& [idx, pkey] : group) {
                            auto ret = DeletePrefixed(pkey, timeout);
                            if (ret != ErrorCode::Success) {
                                failCount.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }));
            }
            for (auto& f : futures) f.get();
            return failCount.load() == 0 ? ErrorCode::Success : ErrorCode::Fail;
        }

        // Low-level Delete on an already-prefixed key (for MultiDelete fallback).
        // The public Delete() builds its own prefix; we need a variant that
        // accepts pre-prefixed keys produced by MakeChunkKey/MakeCountKey.
        // (For Put we already have RawPutWithRetry; no need to duplicate.)
        ErrorCode DeletePrefixed(const std::string& prefixedKey,
                                 const std::chrono::microseconds& timeout)
        {
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
                    InvalidateRegionCache(prefixedKey);
                    if (attempt >= 10) return ErrorCode::Fail;
                    RetryBackoff(attempt);
                    continue;
                }
                if (response.has_region_error()) {
                    InvalidateRegionCache(prefixedKey);
                    if (attempt >= 10) return ErrorCode::Fail;
                    RetryBackoff(attempt);
                    continue;
                }
                if (!response.error().empty()) return ErrorCode::Fail;
                return ErrorCode::Success;
            }
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

            for (auto& kv : regionGroups) {
                RegionGroupKey gkey = kv.first;
                RegionGroup rg = kv.second;
                futures.push_back(std::async(std::launch::async,
                    [this, gkey = std::move(gkey), rg = std::move(rg), queryVector,
                     queryVecBytes, dim, topN, valueType, metaDataSize,
                     timeout]() mutable -> std::vector<CoprocessorResult> {
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

        ErrorCode AsyncGetPostingCounts(const std::vector<SizeType>& headIDs,
                                        std::vector<int>* counts,
                                        const std::chrono::microseconds& timeout)
        {
            if (!counts) return ErrorCode::Fail;
            counts->assign(headIDs.size(), -1);
            if (headIDs.empty()) return ErrorCode::Success;

            struct PendingRegionGroup {
                std::string leaderAddr;
                RegionInfo region{};
                std::vector<std::pair<size_t, std::string>> keys;
            };

            std::unordered_map<RegionGroupKey, RegionGroup, RegionGroupKeyHash> regionGroups;
            for (size_t i = 0; i < headIDs.size(); i++) {
                std::string key = MakeCountKey(headIDs[i]);
                RegionInfo region{};
                std::string addr;
                uint64_t rid = 0;
                if (FindRegionForKey(key, region) && !region.leaderAddr.empty()) {
                    addr = region.leaderAddr;
                    rid = region.regionId;
                } else {
                    addr = GetAnyStoreAddress();
                }
                auto& group = regionGroups[{addr, rid}];
                if (group.keys.empty()) group.region = region;
                group.keys.push_back({i, key});
            }

            std::vector<PendingRegionGroup> groups;
            groups.reserve(regionGroups.size());
            for (auto& kv : regionGroups) {
                PendingRegionGroup group;
                group.leaderAddr = kv.first.leaderAddr;
                group.region = kv.second.region;
                group.keys = kv.second.keys;
                groups.push_back(std::move(group));
            }

            std::vector<std::string> values(headIDs.size());
            auto batch = std::make_shared<AsyncBatch>();
            batch->Add(static_cast<int>(groups.size()));
            std::vector<std::atomic<int>> okFlags(groups.size());
            for (auto& flag : okFlags) flag.store(0, std::memory_order_relaxed);

            for (size_t i = 0; i < groups.size(); i++) {
                kvrpcpb::Context context;
                SetContextFromRegion(&context, groups[i].region);
                AsyncRawBatchGetPrefixed(groups[i].leaderAddr, context, groups[i].keys,
                                         &values, batch, &okFlags[i], timeout);
            }
            auto waitBegin = std::chrono::high_resolution_clock::now();
            batch->Wait();
            RecordAsyncWait(AsyncWaitKind::CountBatchGet, headIDs.size(),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - waitBegin).count()));

            int failCount = 0;
            for (size_t i = 0; i < groups.size(); i++) {
                if (okFlags[i].load(std::memory_order_acquire) == 1) {
                    for (auto& kv : groups[i].keys) {
                        (*counts)[kv.first] = 0;
                    }
                    continue;
                }
                for (auto& kv : groups[i].keys) {
                    size_t idx = kv.first;
                    InvalidateRegionCache(kv.second);
                    int count = GetPostingCount(headIDs[idx], timeout);
                    if (count < 0) {
                        failCount++;
                    } else {
                        (*counts)[idx] = count;
                    }
                }
            }

            for (size_t i = 0; i < values.size(); i++) {
                if (values[i].empty()) continue;
                if (values[i].size() < sizeof(int32_t)) {
                    (*counts)[i] = -1;
                    failCount++;
                    continue;
                }
                int32_t count = 0;
                memcpy(&count, values[i].data(), sizeof(int32_t));
                (*counts)[i] = count;
            }

            return failCount == 0 ? ErrorCode::Success : ErrorCode::Fail;
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

        // Same as PutChunkAndCount but writes the BASE chunk (no timestamp suffix).
        // Used by PutPostingToDB compaction path: replaces (overwrites) the base
        // chunk and updates the count in a single RawBatchPut RPC. Do not fall
        // back to separate writes here: count is required metadata for
        // multi-chunk postings, so partial base/count updates must surface as
        // failures instead of silently corrupting future append counts.
        ErrorCode PutBaseChunkAndCount(SizeType headID,
                                       const std::string& chunkValue,
                                       int newCount,
                                       const std::chrono::microseconds& timeout,
                                       std::vector<Helper::AsyncReadRequest>* reqs) {
            std::string chunkKey = MakeChunkKey(headID); // base key, no suffix
            std::string countKey = MakeCountKey(headID);
            std::string countValue(reinterpret_cast<const char*>(&newCount), sizeof(int32_t));

            {
                auto stub = GetStubForKey(chunkKey);
                if (!stub) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "TiKVIO::PutBaseChunkAndCount missing TiKV stub headID=%d\n", headID);
                    return ErrorCode::Fail;
                }

                kvrpcpb::RawBatchPutRequest request;
                SetContext(request.mutable_context(), chunkKey);

                auto* p1 = request.add_pairs();
                p1->set_key(chunkKey);
                p1->set_value(chunkValue);

                auto* p2 = request.add_pairs();
                p2->set_key(countKey);
                p2->set_value(countValue);

                kvrpcpb::RawBatchPutResponse response;
                grpc::ClientContext ctx;
                SetDeadline(ctx, timeout);

                auto status = stub->RawBatchPut(&ctx, request, &response);
                if (status.ok() && !response.has_region_error() && response.error().empty()) {
                    return ErrorCode::Success;
                }
                if (!status.ok()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "TiKVIO::PutBaseChunkAndCount BatchPut gRPC error headID=%d: %s\n",
                        headID, status.error_message().c_str());
                } else if (response.has_region_error()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVIO::PutBaseChunkAndCount BatchPut region_error headID=%d\n", headID);
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "TiKVIO::PutBaseChunkAndCount error: %s\n", response.error().c_str());
                }
                InvalidateRegionCache(chunkKey);
                InvalidateRegionCache(countKey);
            }
            return ErrorCode::Fail;
        }

        // Multi-posting scan: read multiple postings in parallel.
        // Used by SearchIndex to replace MultiGet when multi-chunk is enabled.
        ErrorCode MultiScanPostings(const std::vector<SizeType>& headIDs,
                                    std::vector<Helper::PageBuffer<std::uint8_t>>& values,
                                    const std::chrono::microseconds& timeout)
        {
            if (headIDs.empty()) return ErrorCode::Success;

            size_t n = headIDs.size();
            std::vector<std::string> postings(n);
            std::vector<std::string> cursors(n);
            std::vector<std::string> nextCursors(n);
            std::vector<std::string> endKeys(n);
            std::vector<char> active(n, 1);
            for (size_t i = 0; i < n; i++) {
                cursors[i] = MakeChunkKey(headIDs[i]);
                std::string raw(reinterpret_cast<const char*>(&headIDs[i]), sizeof(SizeType));
                endKeys[i].reserve(m_keyPrefix.size() + 1 + sizeof(SizeType) + 1);
                endKeys[i].append(m_keyPrefix);
                endKeys[i].push_back('_');
                endKeys[i].append(raw);
                endKeys[i].push_back('\x01');
                values[i].SetAvailableSize(0);
            }

            int failCount = 0;
            while (true) {
                int activeCount = 0;
                for (auto a : active) if (a) activeCount++;
                if (activeCount == 0) break;

                auto batch = std::make_shared<AsyncBatch>();
                batch->Add(activeCount);
                std::vector<std::atomic<int>> okFlags(n);
                std::vector<std::atomic<int>> pageCounts(n);
                std::vector<std::atomic<bool>> moreFlags(n);
                for (size_t i = 0; i < n; i++) {
                    okFlags[i].store(0, std::memory_order_relaxed);
                    pageCounts[i].store(0, std::memory_order_relaxed);
                    moreFlags[i].store(false, std::memory_order_relaxed);
                }

                for (size_t i = 0; i < n; i++) {
                    if (!active[i]) continue;
                    nextCursors[i].clear();
                    AsyncRawScanPagePrefixed(cursors[i], endKeys[i], &postings[i], &nextCursors[i],
                                             &pageCounts[i], &moreFlags[i], batch, &okFlags[i], timeout);
                }
                auto waitBegin = std::chrono::high_resolution_clock::now();
                batch->Wait();
                RecordAsyncWait(AsyncWaitKind::MultiScanPostings, activeCount,
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - waitBegin).count()));

                for (size_t i = 0; i < n; i++) {
                    if (!active[i]) continue;
                    if (okFlags[i].load(std::memory_order_acquire) != 1) {
                        std::string posting;
                        auto ret = ScanPosting(headIDs[i], &posting, timeout);
                        if (ret == ErrorCode::Fail) {
                            auto retryTimeout = std::chrono::microseconds(10000000); // 10s
                            ret = ScanPosting(headIDs[i], &posting, retryTimeout);
                        }
                        if (ret == ErrorCode::Success) {
                            postings[i] = std::move(posting);
                        } else if (ret == ErrorCode::Fail) {
                            failCount++;
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "MultiScanPostings: ScanPosting failed for headID %d after retry\n",
                                headIDs[i]);
                        }
                        active[i] = 0;
                        continue;
                    }

                    int pageCount = pageCounts[i].load(std::memory_order_acquire);
                    if (pageCount == 0) {
                        active[i] = 0;
                    } else if (moreFlags[i].load(std::memory_order_acquire)) {
                        cursors[i] = nextCursors[i];
                    } else {
                        active[i] = 0;
                    }
                }
            }

            for (size_t i = 0; i < n; i++) {
                if (postings[i].empty()) continue;
                if (postings[i].size() > values[i].GetPageSize()) {
                    values[i].ReservePageBuffer(postings[i].size());
                }
                memcpy(values[i].GetBuffer(), postings[i].data(), postings[i].size());
                values[i].SetAvailableSize(static_cast<int>(postings[i].size()));
            }

            if (failCount > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "MultiScanPostings: %d/%d postings had gRPC errors\n",
                    failCount, (int)headIDs.size());
            }
            return failCount == 0 ? ErrorCode::Success : ErrorCode::Fail;
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

        // ---- Async fan-out: submit one BatchPut RPC without blocking ----
        //
        // Issues an async RawBatchPut for {chunkKey,chunkValue}+{countKey,countValue}
        // (the canonical "append a chunk and update the count" pair used by
        // SPFresh multi-chunk Append). The caller must have called
        // batch->Add(N) once before submitting N RPCs against the same batch,
        // and must Wait() on the batch before reading result_slot[i] or
        // destroying any inputs.
        //
        // result_slot stores: 1 = success, 0 = failed (caller should sync-retry).
        // Failure modes (any one marks the slot as 0):
        //   * GetStubForKey returned null
        //   * cq delivered ok=false (cancelled / shutdown)
        //   * gRPC status not OK
        //   * response.has_region_error()  (region cache also invalidated)
        //   * !response.error().empty()
        //
        // The keys themselves and the values are COPIED into the per-RPC tag so
        // that the caller may free its inputs as soon as the call returns.
        void AsyncRawBatchPut(const std::string& chunkKey,
                              const std::string& chunkValue,
                              const std::string& countKey,
                              const std::string& countValue,
                              std::shared_ptr<AsyncBatch> batch,
                              std::atomic<int>* result_slot,
                              const std::chrono::microseconds& timeout)
        {
            if (!m_asyncPumpRunning.load(std::memory_order_acquire)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }
            auto* stub = GetStubForKey(chunkKey);
            if (!stub) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            bool acquiredPermit = false;
            if (!AcquireAsyncRpcPermit(acquiredPermit)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            auto* tag = new AsyncBatchPutTag();
            tag->batch = batch;
            tag->result_slot = result_slot;
            tag->keyForCacheInvalidation = chunkKey;
            tag->owner = this;
            tag->releaseAsyncPermit = acquiredPermit;
            SetContext(tag->request.mutable_context(), chunkKey);
            auto* p1 = tag->request.add_pairs();
            p1->set_key(chunkKey);
            p1->set_value(chunkValue);
            auto* p2 = tag->request.add_pairs();
            p2->set_key(countKey);
            p2->set_value(countValue);
            SetDeadline(tag->ctx, timeout);

            auto* cq = PickAsyncCompletionQueue();
            if (!cq) {
                if (tag->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                delete tag;
                return;
            }
            tag->rpc = stub->AsyncRawBatchPut(&tag->ctx, tag->request, cq);
            tag->rpc->Finish(&tag->response, &tag->status, static_cast<void*>(static_cast<AsyncTagBase*>(tag)));
        }

        // Async single-key Get. The caller key is the RAW key (will be prefixed
        // here). out_value receives the value on success; out_found (optional)
        // distinguishes "found" vs "not_found" — both yield slot=1, but found=false.
        // Caller is responsible for not destroying out_value/out_found until Wait().
        void AsyncRawGet(const std::string& key,
                         std::string* out_value,
                         std::atomic<bool>* out_found,
                         std::shared_ptr<AsyncBatch> batch,
                         std::atomic<int>* result_slot,
                         const std::chrono::microseconds& timeout)
        {
            if (!m_asyncPumpRunning.load(std::memory_order_acquire)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }
            std::string prefixedKey = MakePrefixedKey(key);
            auto* stub = GetStubForKey(prefixedKey);
            if (!stub) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            bool acquiredPermit = false;
            if (!AcquireAsyncRpcPermit(acquiredPermit)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            auto* tag = new AsyncGetTag();
            tag->batch = batch;
            tag->result_slot = result_slot;
            tag->keyForCacheInvalidation = prefixedKey;
            tag->owner = this;
            tag->releaseAsyncPermit = acquiredPermit;
            tag->out_value = out_value;
            tag->out_found = out_found;
            tag->request.set_key(prefixedKey);
            SetContext(tag->request.mutable_context(), prefixedKey);
            SetDeadline(tag->ctx, timeout);

            auto* cq = PickAsyncCompletionQueue();
            if (!cq) {
                if (tag->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                delete tag;
                return;
            }
            tag->rpc = stub->AsyncRawGet(&tag->ctx, tag->request, cq);
            tag->rpc->Finish(&tag->response, &tag->status, static_cast<void*>(static_cast<AsyncTagBase*>(tag)));
        }

        // Async single-key Put. The key is the RAW key (will be prefixed here).
        // The value is COPIED into the tag — caller may free as soon as call returns.
        void AsyncRawPut(const std::string& key,
                         const std::string& value,
                         std::shared_ptr<AsyncBatch> batch,
                         std::atomic<int>* result_slot,
                         const std::chrono::microseconds& timeout)
        {
            if (!m_asyncPumpRunning.load(std::memory_order_acquire)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }
            std::string prefixedKey = MakePrefixedKey(key);
            auto* stub = GetStubForKey(prefixedKey);
            if (!stub) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            bool acquiredPermit = false;
            if (!AcquireAsyncRpcPermit(acquiredPermit)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            auto* tag = new AsyncPutTag();
            tag->batch = batch;
            tag->result_slot = result_slot;
            tag->keyForCacheInvalidation = prefixedKey;
            tag->owner = this;
            tag->releaseAsyncPermit = acquiredPermit;
            tag->request.set_key(prefixedKey);
            tag->request.set_value(value);
            SetContext(tag->request.mutable_context(), prefixedKey);
            SetDeadline(tag->ctx, timeout);

            auto* cq = PickAsyncCompletionQueue();
            if (!cq) {
                if (tag->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                delete tag;
                return;
            }
            tag->rpc = stub->AsyncRawPut(&tag->ctx, tag->request, cq);
            tag->rpc->Finish(&tag->response, &tag->status, static_cast<void*>(static_cast<AsyncTagBase*>(tag)));
        }

        void AsyncRawBatchGetPrefixed(const std::string& leaderAddr,
                          const kvrpcpb::Context& context,
                                      const std::vector<std::pair<size_t, std::string>>& group,
                                      std::vector<std::string>* out_values,
                                      std::shared_ptr<AsyncBatch> batch,
                                      std::atomic<int>* result_slot,
                                      const std::chrono::microseconds& timeout)
        {
            if (!m_asyncPumpRunning.load(std::memory_order_acquire) || group.empty()) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }

            auto* stub = GetOrCreateStub(leaderAddr);
            if (!stub) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }
            bool acquiredPermit = false;
            if (!AcquireAsyncRpcPermit(acquiredPermit)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }

            auto* tag = new AsyncBatchGetTag();
            tag->batch = batch;
            tag->result_slot = result_slot;
            tag->keyForCacheInvalidation = group.front().second;
            tag->owner = this;
            tag->releaseAsyncPermit = acquiredPermit;
            tag->keys = group;
            tag->out_values = out_values;
            *tag->request.mutable_context() = context;
            for (auto& kv : tag->keys) {
                tag->request.add_keys(kv.second);
            }
            SetDeadline(tag->ctx, timeout);

            auto* cq = PickAsyncCompletionQueue();
            if (!cq) {
                if (tag->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                delete tag;
                return;
            }
            tag->rpc = stub->AsyncRawBatchGet(&tag->ctx, tag->request, cq);
            tag->rpc->Finish(&tag->response, &tag->status, static_cast<void*>(static_cast<AsyncTagBase*>(tag)));
        }

        void AsyncRawBatchPutPrefixed(const std::string& leaderAddr,
                                      const kvrpcpb::Context& context,
                                      const std::vector<std::pair<size_t, std::string>>& group,
                                      const std::vector<std::string>& values,
                                      std::shared_ptr<AsyncBatch> batch,
                                      std::atomic<int>* result_slot,
                                      const std::chrono::microseconds& timeout)
        {
            if (!m_asyncPumpRunning.load(std::memory_order_acquire) || group.empty()) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }

            auto* stub = GetOrCreateStub(leaderAddr);
            if (!stub) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }
            bool acquiredPermit = false;
            if (!AcquireAsyncRpcPermit(acquiredPermit)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                batch->Done(false);
                return;
            }

            auto* tag = new AsyncBatchPutTag();
            tag->batch = batch;
            tag->result_slot = result_slot;
            tag->keyForCacheInvalidation = group.front().second;
            tag->owner = this;
            tag->releaseAsyncPermit = acquiredPermit;
            *tag->request.mutable_context() = context;
            for (auto& kv : group) {
                auto* pair = tag->request.add_pairs();
                pair->set_key(kv.second);
                pair->set_value(values[kv.first]);
            }
            SetDeadline(tag->ctx, timeout);

            auto* cq = PickAsyncCompletionQueue();
            if (!cq) {
                if (tag->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                delete tag;
                return;
            }
            tag->rpc = stub->AsyncRawBatchPut(&tag->ctx, tag->request, cq);
            tag->rpc->Finish(&tag->response, &tag->status, static_cast<void*>(static_cast<AsyncTagBase*>(tag)));
        }

        void AsyncRawScanPagePrefixed(const std::string& startKey,
                                      const std::string& endKey,
                                      std::string* out_posting,
                                      std::string* out_next_cursor,
                                      std::atomic<int>* out_count,
                                      std::atomic<bool>* out_more,
                                      std::shared_ptr<AsyncBatch> batch,
                                      std::atomic<int>* result_slot,
                                      const std::chrono::microseconds& timeout)
        {
            if (!m_asyncPumpRunning.load(std::memory_order_acquire)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }

            auto* stub = GetStubForKey(startKey);
            if (!stub) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }
            bool acquiredPermit = false;
            if (!AcquireAsyncRpcPermit(acquiredPermit)) {
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                return;
            }

            auto* tag = new AsyncScanPageTag();
            tag->batch = batch;
            tag->result_slot = result_slot;
            tag->keyForCacheInvalidation = startKey;
            tag->owner = this;
            tag->releaseAsyncPermit = acquiredPermit;
            tag->out_posting = out_posting;
            tag->out_next_cursor = out_next_cursor;
            tag->out_count = out_count;
            tag->out_more = out_more;
            tag->request.set_start_key(startKey);
            tag->request.set_end_key(endKey);
            tag->request.set_limit(1024);
            SetContext(tag->request.mutable_context(), startKey);
            SetDeadline(tag->ctx, timeout);

            auto* cq = PickAsyncCompletionQueue();
            if (!cq) {
                if (tag->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (result_slot) result_slot->store(0, std::memory_order_release);
                if (batch) batch->Done(false);
                delete tag;
                return;
            }
            tag->rpc = stub->AsyncRawScan(&tag->ctx, tag->request, cq);
            tag->rpc->Finish(&tag->response, &tag->status, static_cast<void*>(static_cast<AsyncTagBase*>(tag)));
        }

        // Convenience wrapper mirroring PutChunkAndCount but async: builds the
        // (timestamped chunk key, count key) pair internally and dispatches via
        // AsyncRawBatchPut. Used by SPFresh AddIndex Phase 2 multi-chunk fast
        // path to fan out 12-30 RPCs without spawning OS threads.
        void AsyncAppendChunkAndUpdateCount(SizeType headID,
                                            const std::string& appendPosting,
                                            int newCount,
                                            std::shared_ptr<AsyncBatch> batch,
                                            std::atomic<int>* result_slot,
                                            const std::chrono::microseconds& timeout)
        {
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
            uint64_t ts = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
            std::string suffix(reinterpret_cast<const char*>(&ts), sizeof(ts));
            std::string chunkKey = MakeChunkKey(headID, suffix);
            std::string countKey = MakeCountKey(headID);
            std::string countValue(reinterpret_cast<const char*>(&newCount), sizeof(int32_t));
            AsyncRawBatchPut(chunkKey, appendPosting, countKey, countValue,
                             std::move(batch), result_slot, timeout);
        }

    private:
        bool AcquireAsyncRpcPermit(bool& acquiredPermit) {
            acquiredPermit = false;
            if (m_asyncRpcMaxInflight <= 0) return true;

            auto waitBegin = std::chrono::high_resolution_clock::now();
            const uint64_t limit = static_cast<uint64_t>(m_asyncRpcMaxInflight);
            bool waited = false;

            while (m_asyncPumpRunning.load(std::memory_order_acquire)) {
                uint64_t current = m_asyncRpcInflight.load(std::memory_order_relaxed);
                while (current < limit) {
                    if (m_asyncRpcInflight.compare_exchange_weak(
                            current, current + 1,
                            std::memory_order_acq_rel,
                            std::memory_order_relaxed)) {
                        acquiredPermit = true;
                        ObserveAsyncRpcInflight(current + 1);
                        if (waited) {
                            uint64_t waitUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::high_resolution_clock::now() - waitBegin).count());
                            m_asyncRpcThrottleWaitSamples.fetch_add(1, std::memory_order_relaxed);
                            m_asyncRpcThrottleWaitTotalUs.fetch_add(waitUs, std::memory_order_relaxed);
                        }
                        return true;
                    }
                }

                waited = true;
                std::unique_lock<std::mutex> lock(m_asyncRpcMutex);
                m_asyncRpcCv.wait(lock, [&] {
                    return !m_asyncPumpRunning.load(std::memory_order_acquire) ||
                           m_asyncRpcInflight.load(std::memory_order_acquire) < limit;
                });
            }
            return false;
        }

        void ReleaseAsyncRpcPermit() {
            if (m_asyncRpcMaxInflight <= 0) return;
            uint64_t previous = m_asyncRpcInflight.fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 0) {
                m_asyncRpcInflight.store(0, std::memory_order_release);
            }
            m_asyncRpcCv.notify_one();
        }

        void ObserveAsyncRpcInflight(uint64_t current) {
            uint64_t observed = m_asyncRpcMaxInflightObserved.load(std::memory_order_relaxed);
            while (current > observed &&
                   !m_asyncRpcMaxInflightObserved.compare_exchange_weak(
                       observed, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
            }
        }

        grpc::CompletionQueue* PickAsyncCompletionQueue() {
            if (m_asyncCqs.empty()) return nullptr;
            size_t idx = m_asyncCqNext.fetch_add(1, std::memory_order_relaxed) % m_asyncCqs.size();
            return m_asyncCqs[idx].get();
        }

        // Pump loop: blocks on one CompletionQueue::Next, dispatches each completion via
        // the tag's virtual OnComplete. Runs until ShutDown() calls
        // CompletionQueue::Shutdown(), at which point Next() drains any remaining
        // tags (delivering ok=false for cancelled) and finally returns false.
        // Every submission path MUST produce exactly one Finish() call so that
        // every tag is freed here.
        void AsyncPumpLoop(size_t cqIndex) {
            grpc::CompletionQueue* cq = cqIndex < m_asyncCqs.size() ? m_asyncCqs[cqIndex].get() : nullptr;
            if (!cq) return;
            void* raw_tag = nullptr;
            bool ok = false;
            while (cq->Next(&raw_tag, &ok)) {
                auto* t = static_cast<AsyncTagBase*>(raw_tag);
                bool success = t->OnComplete(ok);
                if (t->releaseAsyncPermit) ReleaseAsyncRpcPermit();
                if (t->result_slot) {
                    t->result_slot->store(success ? 1 : 0, std::memory_order_release);
                }
                if (t->batch) t->batch->Done(success);
                delete t;
            }
        }

    private:
        static constexpr int kAsyncWaitHistBuckets = 22;

        static int AsyncWaitHistBucketOf(uint64_t v) {
            if (v == 0) return 0;
            int b = 0;
            uint64_t x = v;
            while (x > 1) { x >>= 1; ++b; }
            if (b >= kAsyncWaitHistBuckets) b = kAsyncWaitHistBuckets - 1;
            return b;
        }

        static const char* AsyncWaitKindName(AsyncWaitKind kind) {
            switch (kind) {
            case AsyncWaitKind::MultiGetPageBuffer: return "MultiGetPageBuffer";
            case AsyncWaitKind::MultiGetString: return "MultiGetString";
            case AsyncWaitKind::MultiScanPostings: return "MultiScanPostings";
            case AsyncWaitKind::CountBatchGet: return "CountBatchGet";
            case AsyncWaitKind::AddIndexMultiChunk: return "AddIndexMultiChunk";
            case AsyncWaitKind::CollectReAssignMultiChunk: return "CollectReAssignMultiChunk";
            case AsyncWaitKind::AddIndexSingleKeyGet: return "AddIndexSingleKeyGet";
            case AsyncWaitKind::AddIndexSingleKeyPut: return "AddIndexSingleKeyPut";
            case AsyncWaitKind::Count: break;
            }
            return "Unknown";
        }

        std::string m_keyPrefix;
        std::vector<std::string> m_pdAddresses;
        std::unique_ptr<pdpb::PD::Stub> m_pdStub;
        uint64_t m_clusterId = 0;
        bool m_available = false;

        // Async RPC pump (see AsyncRawBatchPut above).
        static constexpr int kAsyncCompletionQueueCount = 4;
        std::vector<std::unique_ptr<grpc::CompletionQueue>> m_asyncCqs;
        std::vector<std::thread> m_asyncPumpThreads;
        std::atomic<uint64_t> m_asyncCqNext{0};
        std::atomic<bool> m_asyncPumpRunning{false};
        int m_asyncRpcMaxInflight = 0;
        std::mutex m_asyncRpcMutex;
        std::condition_variable m_asyncRpcCv;
        std::atomic<uint64_t> m_asyncRpcInflight{0};
        std::atomic<uint64_t> m_asyncRpcThrottleWaitSamples{0};
        std::atomic<uint64_t> m_asyncRpcThrottleWaitTotalUs{0};
        std::atomic<uint64_t> m_asyncRpcMaxInflightObserved{0};

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
        // Bumped whenever an entry in m_storeStubs is invalidated (currently never;
        // reserved for future PD topology changes). Threads compare this against a
        // thread-local cached value to know when to refresh their TLS stub map.
        std::atomic<uint64_t> m_stubEpoch{0};

        std::atomic_uint64_t m_asyncWaitUs[static_cast<int>(AsyncWaitKind::Count)][kAsyncWaitHistBuckets]{};
        std::atomic_uint64_t m_asyncWaitTotalUs[static_cast<int>(AsyncWaitKind::Count)]{};
        std::atomic_uint64_t m_asyncWaitBatchTotal[static_cast<int>(AsyncWaitKind::Count)]{};
        std::atomic_uint64_t m_asyncWaitSampleCount[static_cast<int>(AsyncWaitKind::Count)]{};

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

        // Posting count cache for multi-chunk mode.
        // Tracks approximate vector count per posting to decide when to split.
        bool m_useMultiChunkPosting = false;
        std::unique_ptr<PostingCountCache> m_postingCountCache;

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
        // Hot path is lock-free: every thread caches a shared_ptr<StubPool> per
        // address in thread_local storage and still round-robins across the pool's
        // 48 stubs on every call (preserving channel fan-out). The global
        // m_storeStubs map is only consulted on first-touch (per-thread,
        // per-address) and on epoch bump.
        //
        // The TLS cache is keyed by (TiKVIO*, address) so that multiple TiKVIO
        // instances created on the same thread (e.g. the rebuild flow) do not
        // share dangling pool pointers from a previously destroyed instance.
        // The cached shared_ptr<StubPool> also keeps the underlying gRPC
        // channels alive past instance destruction in case the address is reused.
        tikvpb::Tikv::Stub* GetOrCreateStub(const std::string& address) {
            struct TlsEntry {
                uint64_t epoch;
                std::shared_ptr<StubPool> pool;
            };
            using PerInstance = std::unordered_map<std::string, TlsEntry>;
            thread_local std::unordered_map<const void*, PerInstance> tlsStubCache;

            PerInstance& perInstance = tlsStubCache[this];
            const uint64_t curEpoch = m_stubEpoch.load(std::memory_order_acquire);
            auto tit = perInstance.find(address);
            if (tit != perInstance.end() && tit->second.epoch == curEpoch) {
                return tit->second.pool->GetNext();
            }
            if (tit != perInstance.end()) {
                perInstance.erase(tit);  // stale — refresh below
            }

            // Slow path: locate or create the global pool, then cache it for this thread.
            std::shared_ptr<StubPool> pool;
            {
                std::lock_guard<std::mutex> lock(m_storeMutex);
                auto it = m_storeStubs.find(address);
                if (it != m_storeStubs.end()) {
                    pool = it->second;
                }
            }

            if (!pool) {
                // Create a pool of stubs with separate channels
                pool = std::make_shared<StubPool>();
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
                // Double-check after acquiring lock — another thread may have won the race
                auto it = m_storeStubs.find(address);
                if (it != m_storeStubs.end()) {
                    pool = it->second;
                } else {
                    m_storeStubs[address] = pool;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVIO: Created %d stubs for TiKV store at %s\n", kStubPoolSize, address.c_str());
                }
            }

            perInstance[address] = TlsEntry{curEpoch, pool};
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
