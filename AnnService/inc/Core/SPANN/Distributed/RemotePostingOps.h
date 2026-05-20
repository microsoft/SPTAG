// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/SPANN/Distributed/DistributedProtocol.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Socket/Client.h"
#include "inc/Socket/Server.h"
#include "inc/Socket/Packet.h"
#include "inc/Socket/SimpleSerialization.h"
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace SPTAG::SPANN {

    // Per-thread hook so the SPDKThreadPool's pre-allocated ExtraWorkSpace
    // (initialised once per worker thread, see SPDKThreadPool::initSPDK) can
    // be reached from inside the AppendCallback lambda without changing the
    // callback signature. BatchAppendItemJob::exec(workspace*, abort*) sets
    // this before invoking the callback so the callback skips the per-item
    // InitWorkSpace allocation / m_freeWorkSpaceIds churn that otherwise
    // serialises 10k-item batches into ~130s on the receiver.
    inline thread_local void* tls_preallocAppendWorkSpace = nullptr;

    /// Handles all node-to-node RPC mechanics for internal posting operations:
    ///   - Append / BatchAppend (forward writes to the correct owner node)
    ///   - HeadSync (broadcast head index changes to peers)
    ///   - RemoteLock (cross-node locking for merge/split)
    ///
    /// This class owns the request/response matching state and serialization
    /// logic. It is independent of routing decisions — WorkerNode decides
    /// *where* to send, RemotePostingOps handles *how*.
    class RemotePostingOps {
    public:
        using AppendCallback = std::function<ErrorCode(
            SizeType headID,
            std::shared_ptr<std::string> headVec,
            int appendNum,
            std::string& appendPosting)>;

        using HeadSyncCallback = std::function<void(const HeadSyncEntry& entry)>;
        using RemoteLockCallback = std::function<bool(SizeType headID, bool lock)>;

        /// Callback for cross-node merge: search on a peer node observed
        /// that posting `headID` (which we own) looks underfull. The peer
        /// sent a fire-and-forget MergeRequest to us; we just schedule the
        /// local MergeAsync. Returns nothing; receiver-side m_mergeList
        /// already dedupes repeated triggers, so dropped notifications
        /// are recoverable on the next observation.
        using MergeCallback = std::function<void(SizeType headID)>;

        /// Abstract interface for network access (implemented by NetworkNode).
        class NetworkAccess {
        public:
            virtual ~NetworkAccess() = default;
            virtual Socket::ConnectionID GetPeerConnection(int nodeIndex) = 0;
            virtual void InvalidatePeerConnection(int nodeIndex) = 0;
            virtual int GetLocalNodeIndex() const = 0;
            virtual int GetNumNodes() const = 0;
            virtual Socket::Client* GetClient() = 0;
            virtual Socket::Server* GetServer() = 0;
        };

        RemotePostingOps() {
            StartHeadSyncRetryThread();
        }

        ~RemotePostingOps() {
            StopHeadSyncRetryThread();
        }

        RemotePostingOps(const RemotePostingOps&) = delete;
        RemotePostingOps& operator=(const RemotePostingOps&) = delete;

        void SetNetwork(NetworkAccess* net) { m_net = net; }

        // Inject the searcher's shared compute pool. Receiver-side BatchAppend
        // work runs as Jobs on this pool so it shares a single bounded-
        // concurrency budget with local Append/Split/Merge/Reassign (instead
        // of a separate bg executor + transient std::threads which over-
        // subscribed TiKV). Per-layer: each layer's ExtraDynamicSearcher owns
        // its own m_splitThreadPool, so BatchAppend items dispatch by the
        // request's m_layer to the matching pool. A single submitter would
        // pile both layers' remote appends into whichever pool wired last.
        using JobSubmitter = std::function<void(Helper::ThreadPool::Job*)>;
        void SetJobSubmitter(int layer, JobSubmitter submitter) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            EnsureLayerSlot_NoLock(layer);
            if (m_jobSubmitters.size() <= static_cast<size_t>(layer)) {
                m_jobSubmitters.resize(static_cast<size_t>(layer) + 1);
            }
            m_jobSubmitters[layer] = std::move(submitter);
        }

        // Helper: ensure the per-layer registries are wide enough for `layer`.
        // Caller must hold m_callbackLifetimeMutex in exclusive mode.
        void EnsureLayerSlot_NoLock(int layer) {
            if (layer < 0) return;
            const size_t needed = static_cast<size_t>(layer) + 1;
            if (m_appendCallbacks.size() < needed) m_appendCallbacks.resize(needed);
            if (m_headSyncCallbacks.size() < needed) m_headSyncCallbacks.resize(needed);
            if (m_remoteLockCallbacks.size() < needed) m_remoteLockCallbacks.resize(needed);
            if (m_mergeCallbacks.size() < needed) m_mergeCallbacks.resize(needed);
            if (m_callbackOwners.size() < needed) {
                std::vector<std::atomic<const void*>> grown(needed);
                for (size_t i = 0; i < m_callbackOwners.size(); ++i) {
                    grown[i].store(
                        m_callbackOwners[i].load(std::memory_order_acquire),
                        std::memory_order_release);
                }
                m_callbackOwners = std::move(grown);
            }
        }

        void SetAppendCallback(int layer, AppendCallback cb) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            EnsureLayerSlot_NoLock(layer);
            m_appendCallbacks[layer] = std::move(cb);
        }
        void SetHeadSyncCallback(int layer, HeadSyncCallback cb) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            EnsureLayerSlot_NoLock(layer);
            m_headSyncCallbacks[layer] = std::move(cb);
        }
        void SetRemoteLockCallback(int layer, RemoteLockCallback cb) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            EnsureLayerSlot_NoLock(layer);
            m_remoteLockCallbacks[layer] = std::move(cb);
        }
        void SetMergeCallback(int layer, MergeCallback cb) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            EnsureLayerSlot_NoLock(layer);
            m_mergeCallbacks[layer] = std::move(cb);
        }

        /// Atomically clear ALL callbacks (every layer) and wait for any in-flight
        /// callback invocation to finish. Required before the owner of the captured
        /// `this` pointer (e.g. ExtraDynamicSearcher) is destroyed, otherwise
        /// the lambdas registered via SetXxxCallback would dereference a dangling
        /// pointer.
        void ClearCallbacks() {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            m_appendCallbacks.clear();
            m_headSyncCallbacks.clear();
            m_remoteLockCallbacks.clear();
            m_mergeCallbacks.clear();
            m_callbackOwners = std::vector<std::atomic<const void*>>();
        }

        /// Claim ownership of the registered callbacks for a SPECIFIC layer.
        /// Each ExtraDynamicSearcher owns its own layer slot; per-layer
        /// ownership prevents one layer's destructor from wiping another
        /// layer's still-valid callbacks (the original 1-layer design used a
        /// single ownership token; with Layers>=2 each layer needs its own).
        void ClaimCallbackOwnership(int layer, const void* owner) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            EnsureLayerSlot_NoLock(layer);
            m_callbackOwners[layer].store(owner, std::memory_order_release);
        }

        /// Clear callbacks for `layer` ONLY if `owner` is the current registered
        /// owner of that layer. Used by ExtraDynamicSearcher destructor: each
        /// layer's destructor only clears its own slot. Returns true if cleared.
        bool ClearCallbacksIfOwner(int layer, const void* owner) {
            std::unique_lock<std::shared_timed_mutex> lk(m_callbackLifetimeMutex);
            if (layer < 0 || static_cast<size_t>(layer) >= m_callbackOwners.size()) {
                return false;
            }
            if (m_callbackOwners[layer].load(std::memory_order_acquire) != owner) {
                return false;
            }
            m_appendCallbacks[layer] = nullptr;
            m_headSyncCallbacks[layer] = nullptr;
            m_remoteLockCallbacks[layer] = nullptr;
            if (layer >= 0 && static_cast<size_t>(layer) < m_mergeCallbacks.size()) {
                m_mergeCallbacks[layer] = nullptr;
            }
            m_callbackOwners[layer].store(nullptr, std::memory_order_release);
            return true;
        }

        // ----- internal callback lookup helpers (caller holds shared lock) -----
        const AppendCallback* LookupAppendCallback_Locked(int layer) const {
            if (layer < 0 || static_cast<size_t>(layer) >= m_appendCallbacks.size()) return nullptr;
            const auto& cb = m_appendCallbacks[layer];
            return cb ? &cb : nullptr;
        }
        const HeadSyncCallback* LookupHeadSyncCallback_Locked(int layer) const {
            if (layer < 0 || static_cast<size_t>(layer) >= m_headSyncCallbacks.size()) return nullptr;
            const auto& cb = m_headSyncCallbacks[layer];
            return cb ? &cb : nullptr;
        }
        const RemoteLockCallback* LookupRemoteLockCallback_Locked(int layer) const {
            if (layer < 0 || static_cast<size_t>(layer) >= m_remoteLockCallbacks.size()) return nullptr;
            const auto& cb = m_remoteLockCallbacks[layer];
            return cb ? &cb : nullptr;
        }
        // PutPosting/FetchPosting/DeletePosting RPCs lived here historically.
        // With shared TiKV every node reads and writes the posting store
        // directly (PD routes the key), so the cross-node scatter-gather
        // and owner-callback round-trips are unnecessary.
        const MergeCallback* LookupMergeCallback_Locked(int layer) const {
            if (layer < 0 || static_cast<size_t>(layer) >= m_mergeCallbacks.size()) return nullptr;
            const auto& cb = m_mergeCallbacks[layer];
            return cb ? &cb : nullptr;
        }

        // ==================================================================
        //  Append — single item, synchronous (waits for response)
        // ==================================================================

        ErrorCode SendRemoteAppend(
            int targetNodeIndex,
            int layer,
            SizeType headID,
            const std::shared_ptr<std::string>& headVec,
            int appendNum,
            std::string& appendPosting)
        {
            Socket::ConnectionID connID = m_net->GetPeerConnection(targetNodeIndex);
            if (connID == Socket::c_invalidConnectionID) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: Cannot connect to node %d for headID %lld\n",
                    targetNodeIndex, (std::int64_t)headID);
                return ErrorCode::Fail;
            }

            RemoteAppendRequest req;
            req.m_layer = layer;
            req.m_headID = headID;
            req.m_headVec = *headVec;
            req.m_appendNum = appendNum;
            req.m_appendPosting = appendPosting;

            Socket::ResourceID resID = m_nextResourceId.fetch_add(1);
            auto [future, _] = CreatePendingResponse(resID);
            (void)_;

            Socket::Packet packet;
            packet.Header().m_packetType = Socket::PacketType::AppendRequest;
            packet.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            packet.Header().m_connectionID = Socket::c_invalidConnectionID;
            packet.Header().m_resourceID = resID;

            auto bodySize = static_cast<std::uint32_t>(req.EstimateBufferSize());
            packet.Header().m_bodyLength = bodySize;
            packet.AllocateBuffer(bodySize);
            req.Write(packet.Body());
            packet.Header().WriteBuffer(packet.HeaderBuffer());

            m_net->GetClient()->SendPacket(connID, std::move(packet),
                MakeSendFailHandler(resID));

            auto status = future.wait_for(std::chrono::seconds(30));
            if (status == std::future_status::timeout) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: Timeout waiting for append response for headID %lld from node %d\n",
                    (std::int64_t)headID, targetNodeIndex);
                ErasePending(resID);
                return ErrorCode::Fail;
            }
            return future.get();
        }

        // ==================================================================
        //  Append — batch, synchronous with retry
        // ==================================================================

        ErrorCode SendBatchRemoteAppend(
            int targetNodeIndex,
            std::vector<RemoteAppendRequest>& items)
        {
            if (items.empty()) return ErrorCode::Success;

            // Chunk the batch so a single RPC never exceeds kChunkSize items.
            // Large batches (millions of items) cannot be processed by the
            // receiver within a single timeout window, causing data loss
            // when the request is dropped. Chunking keeps each RPC bounded.
            // [v38] Reduced 50000 → 10000 to (a) shrink end-of-batch drain
            // tail (final chunk no longer 14s wide) and (b) let multiple
            // chunks pipeline on the receiver pool.
            // [v43] Back to 50000 — v42 (10k) was throughput-best (906/s)
            // but during-insert p50 was 222ms; v43 (50k) trades throughput
            // (-22% → 704/s) for during-insert p50 (-36% → 141ms) and big
            // recovery in post-insert r1 QPS (47→85). v44 (100k) blew up
            // tail drain: a single 100k chunk took 116s on the receiver,
            // making end-of-batch drain run 40+ min (vs 8 min at 50k).
            // 50k is the sweet spot.
            // [v47] With shared-pool receiver (BatchAppendItemJob on
            // m_splitThreadPool), 50k chunks still occasionally exceed the
            // 180s wait_for window under contention → "Timeout waiting for
            // batch response" + retries. Drop to 10k so each RPC's worst-case
            // receiver wall-clock is ~6× smaller and stays under the timeout.
            constexpr size_t kChunkSize = 3000;
            const size_t total = items.size();
            size_t offset = 0;
            std::vector<RemoteAppendRequest> chunk;
            chunk.reserve(std::min(kChunkSize, total));

            while (offset < total) {
                size_t end = std::min(offset + kChunkSize, total);
                chunk.clear();
                chunk.reserve(end - offset);
                for (size_t i = offset; i < end; ++i) {
                    chunk.push_back(std::move(items[i]));
                }

                ErrorCode chunkRet = SendBatchRemoteAppendChunk(targetNodeIndex, chunk);
                if (chunkRet != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "RemotePostingOps: Chunk send failed to node %d (offset=%zu/%zu, chunk=%zu items)\n",
                        targetNodeIndex, offset, total, end - offset);
                    return chunkRet;
                }
                offset = end;
            }
            return ErrorCode::Success;
        }

    private:
        ErrorCode SendBatchRemoteAppendChunk(
            int targetNodeIndex,
            std::vector<RemoteAppendRequest>& items)
        {
            if (items.empty()) return ErrorCode::Success;

            for (int attempt = 0; attempt < 3; attempt++) {
                Socket::ConnectionID connID = m_net->GetPeerConnection(targetNodeIndex);
                if (connID == Socket::c_invalidConnectionID) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "RemotePostingOps: Cannot connect to node %d for batch (%d items, attempt %d)\n",
                        targetNodeIndex, (int)items.size(), attempt + 1);
                    if (attempt < 2) continue;
                    return ErrorCode::Fail;
                }

                BatchRemoteAppendRequest batchReq;
                batchReq.m_count = static_cast<std::uint32_t>(items.size());
                batchReq.m_items = std::move(items);

                Socket::ResourceID resID = m_nextResourceId.fetch_add(1);
                auto [future, _] = CreatePendingResponse(resID);
                (void)_;

                Socket::Packet packet;
                packet.Header().m_packetType = Socket::PacketType::BatchAppendRequest;
                packet.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
                packet.Header().m_connectionID = Socket::c_invalidConnectionID;
                packet.Header().m_resourceID = resID;

                auto bodySize = static_cast<std::uint32_t>(batchReq.EstimateBufferSize());
                packet.Header().m_bodyLength = bodySize;
                packet.AllocateBuffer(bodySize);
                batchReq.Write(packet.Body());
                items = std::move(batchReq.m_items); // restore for retry

                packet.Header().WriteBuffer(packet.HeaderBuffer());

                SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                    "RemotePostingOps: Sending batch of %u appends to node %d (resID=%u, attempt=%d)\n",
                    batchReq.m_count, targetNodeIndex, resID, attempt + 1);

                auto waitStart = std::chrono::steady_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "RemotePostingOps: BatchAppendChunk -> node %d (resID=%u, attempt=%d, items=%u) wait_start\n",
                    targetNodeIndex, resID, attempt + 1, batchReq.m_count);

                m_net->GetClient()->SendPacket(connID, std::move(packet),
                    MakeSendFailHandler(resID));

                // Generous timeout: 50k items * (~10ms TiKV roundtrip / 16 worker threads)
                // = ~31s typical; cap at 180s to allow for lock contention with merges/splits.
                auto status = future.wait_for(std::chrono::seconds(180));
                auto waitMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - waitStart).count();
                if (status == std::future_status::timeout) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "RemotePostingOps: Timeout waiting for batch response from node %d (chunk=%u items, attempt=%d, waited=%lldms)\n",
                        targetNodeIndex, batchReq.m_count, attempt + 1, (long long)waitMs);
                    ErasePending(resID);
                    // Do NOT invalidate the connection on timeout — a slow
                    // response is not a broken connection, and reconnecting
                    // floods the worker's accept loop. Real connection errors
                    // are signalled via MakeSendFailHandler (which sets the
                    // promise to Fail, taking the "result != Success" path
                    // below).
                    if (attempt < 2) continue;
                    return ErrorCode::Fail;
                }

                ErrorCode result = future.get();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "RemotePostingOps: BatchAppendChunk <- node %d (resID=%u, attempt=%d, items=%u, waited=%lldms, result=%d)\n",
                    targetNodeIndex, resID, attempt + 1, batchReq.m_count, (long long)waitMs, (int)result);
                if (result == ErrorCode::Success) return ErrorCode::Success;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: Batch to node %d failed (attempt %d), reconnecting...\n",
                    targetNodeIndex, attempt + 1);
                m_net->InvalidatePeerConnection(targetNodeIndex);
            }
            return ErrorCode::Fail;
        }

    public:

        // ==================================================================
        //  HeadSync — fire-and-forget broadcast
        // ==================================================================

        void BroadcastHeadSync(const std::vector<HeadSyncEntry>& entries) {
            if (entries.empty()) return;

            int numNodes = m_net->GetNumNodes();
            int localIdx = m_net->GetLocalNodeIndex();

            // Count once per peer for sent-entry totals.
            std::uint64_t targetCount = 0;
            for (int i = 0; i < numNodes; i++) {
                if (i != localIdx) targetCount++;
            }
            m_headSyncBroadcastEntries.fetch_add(entries.size() * targetCount,
                                                  std::memory_order_relaxed);

            for (int i = 0; i < numNodes; i++) {
                if (i == localIdx) continue;
                // Pass a copy of `entries` per peer so each can be re-enqueued
                // into its own retry backlog independently on send failure.
                SendOneHeadSync(i, std::vector<HeadSyncEntry>(entries),
                                /*isRetry=*/false);
            }
        }

        // Send a HeadSync packet to a single peer. On TCP-level send failure
        // (success=false reported by the network stack), the entries are
        // appended to the per-peer retry backlog so the background retry
        // thread can re-attempt delivery. Counter increments are done
        // best-effort once the SendPacket completion lambda fires.
        void SendOneHeadSync(int nodeIdx,
                             std::vector<HeadSyncEntry> entries,
                             bool isRetry)
        {
            if (entries.empty()) return;

            Socket::ConnectionID connID = m_net->GetPeerConnection(nodeIdx);
            if (connID == Socket::c_invalidConnectionID) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: HeadSync no connection to node %d (count=%zu, isRetry=%d)\n",
                    nodeIdx, entries.size(), isRetry ? 1 : 0);
                EnqueueHeadSyncRetry(nodeIdx, std::move(entries));
                return;
            }

            size_t bodySize = sizeof(std::uint32_t);
            for (const auto& e : entries) bodySize += e.EstimateBufferSize();

            Socket::Packet pkt;
            pkt.Header().m_packetType = Socket::PacketType::HeadSyncRequest;
            pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
            pkt.Header().m_resourceID = 0;
            pkt.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
            pkt.AllocateBuffer(static_cast<std::uint32_t>(bodySize));

            std::uint8_t* buf = pkt.Body();
            buf = Socket::SimpleSerialization::SimpleWriteBuffer(
                static_cast<std::uint32_t>(entries.size()), buf);
            for (const auto& e : entries) buf = e.Write(buf);
            pkt.Header().WriteBuffer(pkt.HeaderBuffer());

            const std::uint64_t sentCount = entries.size();
            std::shared_ptr<std::vector<HeadSyncEntry>> entriesShared =
                std::make_shared<std::vector<HeadSyncEntry>>(std::move(entries));
            const bool wasRetry = isRetry;

            m_net->GetClient()->SendPacket(connID, std::move(pkt),
                [this, nodeIdx, entriesShared, sentCount, wasRetry](bool success) {
                    if (success) {
                        m_headSyncBroadcastSendOK.fetch_add(sentCount,
                            std::memory_order_relaxed);
                        if (wasRetry) {
                            m_headSyncRetrySucceeded.fetch_add(sentCount,
                                std::memory_order_relaxed);
                        }
                    } else {
                        m_headSyncBroadcastSendFail.fetch_add(sentCount,
                            std::memory_order_relaxed);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "RemotePostingOps: HeadSync send to node %d FAILED "
                            "(count=%llu, isRetry=%d) — enqueueing for retry\n",
                            nodeIdx,
                            (unsigned long long)sentCount,
                            wasRetry ? 1 : 0);
                        m_net->InvalidatePeerConnection(nodeIdx);
                        EnqueueHeadSyncRetry(nodeIdx, std::move(*entriesShared));
                    }
                });
        }

        void EnqueueHeadSyncRetry(int nodeIdx, std::vector<HeadSyncEntry> entries) {
            if (entries.empty()) return;
            auto backlog = GetOrCreateBacklog(nodeIdx);
            std::lock_guard<std::mutex> g(backlog->mu);
            if (backlog->queue.size() + entries.size() > HeadSyncBacklog::kMaxEntries) {
                std::uint64_t dropped = entries.size();
                m_headSyncRetryDropped.fetch_add(dropped, std::memory_order_relaxed);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: HeadSync retry queue full for node %d "
                    "(queue=%zu, dropping=%llu) — index will diverge!\n",
                    nodeIdx, backlog->queue.size(),
                    (unsigned long long)dropped);
                return;
            }
            for (auto& e : entries) backlog->queue.push_back(std::move(e));
            m_headSyncRetryEnqueued.fetch_add(entries.size(),
                std::memory_order_relaxed);
        }

        // Pull up to maxBatch entries from the per-peer backlog and re-send
        // them. Called from the retry thread and on demand. Returns the
        // total number of entries dispatched (including for retry-of-retry).
        size_t DrainHeadSyncBacklog(size_t maxBatch = 1024) {
            if (!m_net) return 0;
            std::vector<int> nodeIdxs;
            {
                std::shared_lock<std::shared_timed_mutex> lk(m_headSyncBacklogsMu);
                nodeIdxs.reserve(m_headSyncBacklogs.size());
                for (auto& kv : m_headSyncBacklogs) nodeIdxs.push_back(kv.first);
            }
            size_t dispatched = 0;
            for (int nodeIdx : nodeIdxs) {
                auto backlog = GetOrCreateBacklog(nodeIdx);
                std::vector<HeadSyncEntry> batch;
                {
                    std::lock_guard<std::mutex> g(backlog->mu);
                    if (backlog->queue.empty()) continue;
                    size_t bs = std::min<size_t>(backlog->queue.size(), maxBatch);
                    batch.reserve(bs);
                    for (size_t i = 0; i < bs; i++) {
                        batch.push_back(std::move(backlog->queue.front()));
                        backlog->queue.pop_front();
                    }
                }
                size_t bs = batch.size();
                SendOneHeadSync(nodeIdx, std::move(batch), /*isRetry=*/true);
                dispatched += bs;
            }
            return dispatched;
        }

        size_t GetHeadSyncBacklogSize() const {
            size_t total = 0;
            std::vector<std::shared_ptr<HeadSyncBacklog>> snapshot;
            {
                std::shared_lock<std::shared_timed_mutex> lk(m_headSyncBacklogsMu);
                snapshot.reserve(m_headSyncBacklogs.size());
                for (auto& kv : m_headSyncBacklogs) snapshot.push_back(kv.second);
            }
            for (auto& b : snapshot) {
                std::lock_guard<std::mutex> g(b->mu);
                total += b->queue.size();
            }
            return total;
        }

        // Best-effort log dump of HeadSync delivery counters. Use whenever a
        // checkpoint is needed (start/end of insert phase, before query, on
        // SaveIndex, etc.).
        void DumpHeadSyncStats(const char* label) const {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[HeadSync stats %s] broadcast_entries=%llu send_ok=%llu send_fail=%llu "
                "recv_entries=%llu apply_add=%llu apply_del=%llu "
                "retry_enqueued=%llu retry_succeeded=%llu retry_dropped=%llu "
                "backlog_now=%zu\n",
                label ? label : "",
                (unsigned long long)m_headSyncBroadcastEntries.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncBroadcastSendOK.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncBroadcastSendFail.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncRecvEntries.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncApplyAdd.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncApplyDelete.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncRetryEnqueued.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncRetrySucceeded.load(std::memory_order_relaxed),
                (unsigned long long)m_headSyncRetryDropped.load(std::memory_order_relaxed),
                GetHeadSyncBacklogSize());
        }

        // Counters incremented by the receiver-side HandleHeadSyncRequest /
        // AddHeadIndex callback. Public so the ExtraDynamicSearcher
        // HeadSyncCallback lambda can bump them after applying each entry.
        void NoteHeadSyncApplyAdd() {
            m_headSyncApplyAdd.fetch_add(1, std::memory_order_relaxed);
        }
        void NoteHeadSyncApplyDelete() {
            m_headSyncApplyDelete.fetch_add(1, std::memory_order_relaxed);
        }

        // Best-effort log dump of cross-node merge-hint channel counters.
        // Mirrors DumpHeadSyncStats: sender side tracks how many hints we
        // broadcast (send_ok / send_fail); receiver side tracks how many
        // hints we got and how many were dropped (callback missing).
        void DumpMergeRequestStats(const char* label) const {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[MergeHint stats %s] send_ok=%llu send_fail=%llu "
                "recv_hints=%llu recv_dropped=%llu\n",
                label ? label : "",
                (unsigned long long)m_mergeBroadcastSendOK.load(std::memory_order_relaxed),
                (unsigned long long)m_mergeBroadcastSendFail.load(std::memory_order_relaxed),
                (unsigned long long)m_mergeRecvHints.load(std::memory_order_relaxed),
                (unsigned long long)m_mergeRecvDropped.load(std::memory_order_relaxed));
        }

        // ==================================================================
        //  RemoteLock — synchronous request/response
        // ==================================================================

        bool SendRemoteLock(int nodeIndex, int layer, SizeType headID, bool lock) {
            Socket::ConnectionID connID = m_net->GetPeerConnection(nodeIndex);
            if (connID == Socket::c_invalidConnectionID) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: Cannot send remote lock to node %d\n", nodeIndex);
                return false;
            }

            RemoteLockRequest req;
            req.m_op = lock ? RemoteLockRequest::Op::Lock : RemoteLockRequest::Op::Unlock;
            req.m_headID = headID;
            req.m_layer = layer;

            Socket::ResourceID rid = m_nextResourceId.fetch_add(1);
            auto [future, _] = CreatePendingResponse(rid);
            (void)_;

            Socket::Packet pkt;
            auto bodySize = req.EstimateBufferSize();
            pkt.Header().m_packetType = Socket::PacketType::RemoteLockRequest;
            pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
            pkt.Header().m_resourceID = rid;
            pkt.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
            pkt.AllocateBuffer(static_cast<std::uint32_t>(bodySize));
            req.Write(pkt.Body());
            pkt.Header().WriteBuffer(pkt.HeaderBuffer());

            m_net->GetClient()->SendPacket(connID, std::move(pkt),
                MakeSendFailHandler(rid));

            auto status = future.wait_for(std::chrono::milliseconds(5000));
            if (status != std::future_status::ready) {
                ErasePending(rid);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: Lock timeout for headID %lld on node %d\n",
                    (std::int64_t)headID, nodeIndex);
                return false;
            }
            return future.get() == ErrorCode::Success;
        }

        // ==================================================================
        //  Inbound packet handlers (called by WorkerNode's server/client)
        // ==================================================================

        void HandleAppendRequest(Socket::ConnectionID connID, Socket::Packet packet) {
            if (packet.Header().m_bodyLength == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: Empty AppendRequest\n");
                return;
            }

            if (Socket::c_invalidConnectionID == packet.Header().m_connectionID)
                packet.Header().m_connectionID = connID;

            RemoteAppendRequest req;
            const std::uint8_t* body = packet.Body();
            const std::uint8_t* bodyEnd = body + packet.Header().m_bodyLength;
            if (req.Read(body, bodyEnd) == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: AppendRequest version mismatch\n");
                SendAppendResponse(packet, RemoteAppendResponse::Status::Failed);
                return;
            }

            ErrorCode result = ErrorCode::Fail;
            {
                std::shared_lock<std::shared_timed_mutex> cbLock(m_callbackLifetimeMutex);
                const auto* cb = LookupAppendCallback_Locked(req.m_layer);
                if (cb) {
                    auto headVec = std::make_shared<std::string>(std::move(req.m_headVec));
                    result = (*cb)(
                        req.m_headID, headVec, req.m_appendNum, req.m_appendPosting);
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "RemotePostingOps: AppendRequest layer=%d has no callback registered\n",
                        req.m_layer);
                }
            }

            auto status = (result == ErrorCode::Success)
                ? RemoteAppendResponse::Status::Success
                : RemoteAppendResponse::Status::Failed;
            SendAppendResponse(packet, status);
        }

        void HandleAppendResponse(Socket::ConnectionID connID, Socket::Packet packet) {
            Socket::ResourceID resID = packet.Header().m_resourceID;
            auto promise = TakePendingResponse(resID);
            if (!promise) return;

            if (packet.Header().m_processStatus != Socket::PacketProcessStatus::Ok) {
                promise->set_value(ErrorCode::Fail);
                return;
            }

            RemoteAppendResponse resp;
            if (resp.Read(packet.Body()) == nullptr) {
                promise->set_value(ErrorCode::Fail);
                return;
            }

            promise->set_value(
                resp.m_status == RemoteAppendResponse::Status::Success
                    ? ErrorCode::Success : ErrorCode::Fail);
        }

        void HandleBatchAppendRequest(Socket::ConnectionID connID, Socket::Packet packet) {
            if (packet.Header().m_bodyLength == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: Empty BatchAppendRequest\n");
                return;
            }

            if (Socket::c_invalidConnectionID == packet.Header().m_connectionID)
                packet.Header().m_connectionID = connID;

            auto batchReq = std::make_shared<BatchRemoteAppendRequest>();
            if (batchReq->Read(packet.Body(), packet.Header().m_bodyLength) == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: BatchAppendRequest parse failed\n");
                SendBatchAppendResponse(packet, 0, 1);
                return;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                "RemotePostingOps: Received batch of %u appends\n", batchReq->m_count);

            // Submit each item as a Job to the searcher's shared compute pool.
            // Pool workers run the local Append callback exactly like a local
            // insert would. Last completion ACKs the sender. This puts remote
            // work on the SAME concurrency budget as local Split/Merge/Reassign
            // — eliminating the over-subscribed TiKV behaviour of the old
            // separate bg executor + transient sub-worker threads.
            auto packetPtr = std::make_shared<Socket::Packet>(std::move(packet));
            const size_t total = batchReq->m_items.size();
            if (total == 0) {
                SendBatchAppendResponse(*packetPtr, 0, 0);
                return;
            }
            auto remaining    = std::make_shared<std::atomic<size_t>>(total);
            auto successCount = std::make_shared<std::atomic<std::uint32_t>>(0);
            auto failCount    = std::make_shared<std::atomic<std::uint32_t>>(0);

            if (m_jobSubmitters.empty()) {
                // Fallback: process inline on the network thread. Should not
                // happen once ExtraDynamicSearcher has wired its pool.
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: no job submitter wired; running BatchAppend synchronously\n");
                std::shared_lock<std::shared_timed_mutex> cbLock(m_callbackLifetimeMutex);
                for (auto& req : batchReq->m_items) {
                    ErrorCode r = ErrorCode::Fail;
                    const auto* cb = LookupAppendCallback_Locked(req.m_layer);
                    if (cb) {
                        auto hv = std::make_shared<std::string>(std::move(req.m_headVec));
                        r = (*cb)(req.m_headID, hv, req.m_appendNum, req.m_appendPosting);
                    }
                    (r == ErrorCode::Success ? *successCount : *failCount).fetch_add(1);
                }
                SendBatchAppendResponse(*packetPtr, successCount->load(), failCount->load());
                return;
            }

            for (size_t i = 0; i < total; i++) {
                auto* job = new BatchAppendItemJob(
                    this, batchReq, i, remaining, successCount, failCount, packetPtr);
                // Route to the per-layer searcher pool matching this item's
                // m_layer so local Append/Split/Merge on layer N and remote
                // appends targeting layer N share the same 16-thread budget.
                // A single global submitter sent both layers' work into one
                // pool, causing 35k+ queue depth on the receiver side.
                int layer = batchReq->m_items[i].m_layer;
                const JobSubmitter* sub = nullptr;
                if (layer >= 0 && static_cast<size_t>(layer) < m_jobSubmitters.size()
                    && m_jobSubmitters[layer]) {
                    sub = &m_jobSubmitters[layer];
                } else {
                    // Layer's pool not yet wired — fall back to whichever
                    // submitter we have.
                    for (auto& s : m_jobSubmitters) { if (s) { sub = &s; break; } }
                }
                // Per-layer routing (m_jobSubmitters[layer]) isolates layer-N
                // append items from other layers' pools.
                if (sub) (*sub)(job);
                else     { delete job; failCount->fetch_add(1); remaining->fetch_sub(1); }
            }
        }

        void HandleBatchAppendResponse(Socket::ConnectionID connID, Socket::Packet packet) {
            Socket::ResourceID resID = packet.Header().m_resourceID;
            auto promise = TakePendingResponse(resID);
            if (!promise) return;

            if (packet.Header().m_processStatus != Socket::PacketProcessStatus::Ok) {
                promise->set_value(ErrorCode::Fail);
                return;
            }

            BatchRemoteAppendResponse resp;
            if (resp.Read(packet.Body()) == nullptr) {
                promise->set_value(ErrorCode::Fail);
                return;
            }

            promise->set_value(resp.m_failCount == 0 ? ErrorCode::Success : ErrorCode::Fail);
        }

        void HandleHeadSyncRequest(Socket::ConnectionID connID, Socket::Packet packet) {
            std::shared_lock<std::shared_timed_mutex> cbLock(m_callbackLifetimeMutex);
            if (m_headSyncCallbacks.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: HeadSyncRequest but no callbacks registered\n");
                return;
            }

            const std::uint8_t* buf = packet.Body();
            const std::uint8_t* bufEnd = buf + packet.Header().m_bodyLength;
            std::uint32_t entryCount = 0;
            buf = Socket::SimpleSerialization::SimpleReadBuffer(buf, entryCount);

            std::uint32_t bodyLength = packet.Header().m_bodyLength;
            if (bodyLength < sizeof(std::uint32_t) ||
                entryCount > (bodyLength - sizeof(std::uint32_t)) / 8) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: HeadSyncRequest entryCount=%u exceeds bodyLength=%u\n",
                    entryCount, bodyLength);
                return;
            }

            for (std::uint32_t i = 0; i < entryCount; i++) {
                if (buf >= bufEnd) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "RemotePostingOps: HeadSync buffer overrun at entry %u/%u\n", i, entryCount);
                    break;
                }
                HeadSyncEntry entry;
                buf = entry.Read(buf);
                if (!buf || buf > bufEnd) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "RemotePostingOps: HeadSync parse error at entry %u/%u\n", i, entryCount);
                    break;
                }
                m_headSyncRecvEntries.fetch_add(1, std::memory_order_relaxed);
                const auto* cb = LookupHeadSyncCallback_Locked(entry.m_layer);
                if (cb) {
                    (*cb)(entry);
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "RemotePostingOps: HeadSyncEntry layer=%d has no callback registered (op=%d, vid=%d)\n",
                        entry.m_layer, static_cast<int>(entry.op), (int)entry.headVID);
                }
            }
        }

        // ==================================================================
        //  Merge — fire-and-forget cross-node hint
        // ==================================================================

        /// Send a batch of merge hints to one peer. Fire-and-forget: no
        /// response is expected and no retry queue is maintained. Receiver-
        /// side m_mergeList dedups, and the owner discovers underfull
        /// postings through its own paths (own search, own Append) if any
        /// notification is dropped.
        void SendBatchRemoteMerge(int targetNodeIndex,
                                  const std::vector<RemoteMergeRequest>& items)
        {
            if (items.empty()) return;

            Socket::ConnectionID connID = m_net->GetPeerConnection(targetNodeIndex);
            if (connID == Socket::c_invalidConnectionID) {
                m_mergeBroadcastSendFail.fetch_add(items.size(), std::memory_order_relaxed);
                return;
            }

            BatchRemoteMergeRequest batch;
            batch.m_count = static_cast<std::uint32_t>(items.size());
            batch.m_items = items;

            Socket::Packet pkt;
            pkt.Header().m_packetType = Socket::PacketType::MergeRequest;
            pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
            pkt.Header().m_resourceID = 0;

            auto bodySize = static_cast<std::uint32_t>(batch.EstimateBufferSize());
            pkt.Header().m_bodyLength = bodySize;
            pkt.AllocateBuffer(bodySize);
            batch.Write(pkt.Body());
            pkt.Header().WriteBuffer(pkt.HeaderBuffer());

            const std::uint64_t sentCount = items.size();
            m_net->GetClient()->SendPacket(connID, std::move(pkt),
                [this, targetNodeIndex, sentCount](bool success) {
                    if (success) {
                        m_mergeBroadcastSendOK.fetch_add(sentCount, std::memory_order_relaxed);
                    } else {
                        m_mergeBroadcastSendFail.fetch_add(sentCount, std::memory_order_relaxed);
                        m_net->InvalidatePeerConnection(targetNodeIndex);
                    }
                });
        }

        void HandleMergeRequest(Socket::ConnectionID connID, Socket::Packet packet) {
            (void)connID;
            BatchRemoteMergeRequest batch;
            if (batch.Read(packet.Body(), packet.Header().m_bodyLength) == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: MergeRequest parse failed (bodyLength=%u)\n",
                    packet.Header().m_bodyLength);
                return;
            }

            std::shared_lock<std::shared_timed_mutex> cbLock(m_callbackLifetimeMutex);
            for (const auto& item : batch.m_items) {
                const auto* cb = LookupMergeCallback_Locked(item.m_layer);
                if (cb) {
                    (*cb)(item.m_headID);
                    m_mergeRecvHints.fetch_add(1, std::memory_order_relaxed);
                } else {
                    m_mergeRecvDropped.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        void HandleRemoteLockRequest(Socket::ConnectionID connID, Socket::Packet packet) {
            RemoteLockRequest req;
            if (req.Read(packet.Body()) == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "RemotePostingOps: Failed to parse RemoteLockRequest\n");
                return;
            }

            RemoteLockResponse resp;
            resp.m_status = RemoteLockResponse::Status::Denied;

            {
                std::shared_lock<std::shared_timed_mutex> cbLock(m_callbackLifetimeMutex);
                const auto* cb = LookupRemoteLockCallback_Locked(req.m_layer);
                if (cb) {
                    bool isLock = (req.m_op == RemoteLockRequest::Op::Lock);
                    bool success = (*cb)(req.m_headID, isLock);
                    if (success) resp.m_status = RemoteLockResponse::Status::Granted;
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "RemotePostingOps: RemoteLockRequest layer=%d has no callback registered\n",
                        req.m_layer);
                }
            }

            Socket::Packet ret;
            auto bodySize = resp.EstimateBufferSize();
            ret.Header().m_packetType = Socket::PacketType::RemoteLockResponse;
            ret.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            ret.Header().m_connectionID = connID;
            ret.Header().m_resourceID = packet.Header().m_resourceID;
            ret.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
            ret.AllocateBuffer(static_cast<std::uint32_t>(bodySize));
            resp.Write(ret.Body());
            ret.Header().WriteBuffer(ret.HeaderBuffer());

            m_net->GetServer()->SendPacket(connID, std::move(ret), nullptr);
        }

        void HandleRemoteLockResponse(Socket::ConnectionID connID, Socket::Packet packet) {
            Socket::ResourceID rid = packet.Header().m_resourceID;
            auto promise = TakePendingResponse(rid);
            if (!promise) return;

            RemoteLockResponse resp;
            if (resp.Read(packet.Body()) == nullptr) {
                promise->set_value(ErrorCode::Fail);
                return;
            }

            promise->set_value(resp.m_status == RemoteLockResponse::Status::Granted
                ? ErrorCode::Success : ErrorCode::Fail);
        }

        // ---- Response matching helpers ----

        std::pair<std::future<ErrorCode>, bool> CreatePendingResponse(Socket::ResourceID resID) {
            std::promise<ErrorCode> promise;
            auto future = promise.get_future();
            std::lock_guard<std::mutex> lock(m_pendingMutex);
            m_pendingResponses.emplace(resID, std::move(promise));
            return {std::move(future), true};
        }

        void ErasePending(Socket::ResourceID resID) {
            std::lock_guard<std::mutex> lock(m_pendingMutex);
            m_pendingResponses.erase(resID);
        }

        /// Take a pending promise out of the map (returns nullptr if not found).
        std::unique_ptr<std::promise<ErrorCode>> TakePendingResponse(Socket::ResourceID resID) {
            std::lock_guard<std::mutex> lock(m_pendingMutex);
            auto it = m_pendingResponses.find(resID);
            if (it == m_pendingResponses.end()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "RemotePostingOps: Response for unknown resourceID %u\n", resID);
                return nullptr;
            }
            auto p = std::make_unique<std::promise<ErrorCode>>(std::move(it->second));
            m_pendingResponses.erase(it);
            return p;
        }

        /// Create a send-failure callback that resolves the pending promise.
        std::function<void(bool)> MakeSendFailHandler(Socket::ResourceID resID) {
            return [resID, this](bool success) {
                if (!success) {
                    std::lock_guard<std::mutex> lock(m_pendingMutex);
                    auto it = m_pendingResponses.find(resID);
                    if (it != m_pendingResponses.end()) {
                        it->second.set_value(ErrorCode::Fail);
                        m_pendingResponses.erase(it);
                    }
                }
            };
        }

        void SendAppendResponse(Socket::Packet& srcPacket, RemoteAppendResponse::Status status) {
            RemoteAppendResponse resp;
            resp.m_status = status;

            Socket::Packet ret;
            ret.Header().m_packetType = Socket::PacketType::AppendResponse;
            ret.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            ret.Header().m_connectionID = srcPacket.Header().m_connectionID;
            ret.Header().m_resourceID = srcPacket.Header().m_resourceID;

            auto bodySize = static_cast<std::uint32_t>(resp.EstimateBufferSize());
            ret.Header().m_bodyLength = bodySize;
            ret.AllocateBuffer(bodySize);
            resp.Write(ret.Body());
            ret.Header().WriteBuffer(ret.HeaderBuffer());

            m_net->GetServer()->SendPacket(srcPacket.Header().m_connectionID, std::move(ret), nullptr);
        }

        void SendBatchAppendResponse(Socket::Packet& srcPacket,
            std::uint32_t successCount, std::uint32_t failCount) {
            BatchRemoteAppendResponse resp;
            resp.m_successCount = successCount;
            resp.m_failCount = failCount;

            Socket::Packet ret;
            ret.Header().m_packetType = Socket::PacketType::BatchAppendResponse;
            ret.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            ret.Header().m_connectionID = srcPacket.Header().m_connectionID;
            ret.Header().m_resourceID = srcPacket.Header().m_resourceID;

            auto bodySize = static_cast<std::uint32_t>(resp.EstimateBufferSize());
            ret.Header().m_bodyLength = bodySize;
            ret.AllocateBuffer(bodySize);
            resp.Write(ret.Body());
            ret.Header().WriteBuffer(ret.HeaderBuffer());

            m_net->GetServer()->SendPacket(srcPacket.Header().m_connectionID, std::move(ret), nullptr);
        }

        // ==================================================================
        //  [Bug 26] Background executor — slow-lane for batch RPC handlers
        // ==================================================================
        //
        // Why: the network server thread pool has only 8 threads
        // (NetworkNode.h). HandleBatchAppendRequest does heavy TiKV work
        // (fan out to 4 sub-workers and join), each call tying up its
        // network thread for tens of seconds during inserts.
        // Once 4–8 such handlers run concurrently, every network thread is
        // blocked and latency-sensitive RPCs (HeadSync, RemoteLock) cannot be
        // serviced.
        //
        // Fix: parse on the network thread (fast), then enqueue the heavy
        // work onto a dedicated background thread pool and return. The
        // network thread immediately becomes available for other RPCs.
        // The background worker eventually sends the response itself.
        //
        // Sizing rationale:
        //   - Threads default to 8: matches the network pool so we never
        //     under-utilize CPU even if every network thread is parsing a
        //     batch. Tunable via env SPTAG_BG_EXEC_THREADS.
        //   - Queue cap default 256: plenty of headroom for typical bursts;
        //     when full, falls back to synchronous execution to preserve
        //     correctness rather than dropping requests.

        // Background executor removed: BatchAppend now runs as sub-Jobs on
        // the searcher's shared compute pool via SetJobSubmitter() so it
        // shares a single concurrency budget with local Split/Merge/Reassign
        // (with high-priority jumping the queue). See HandleBatchAppendRequest.

        // ==================================================================
        //  HeadSync retry thread — periodic best-effort drain of per-peer
        //  backlogs that were populated by failed BroadcastHeadSync sends.
        //
        //  Why: BroadcastHeadSync is fire-and-forget by design (we don't
        //  want to block the layer-1 split path on a slow peer). When the
        //  TCP send completion reports failure, we previously dropped the
        //  entries forever and the peer's headIndex / m_pSamples diverged,
        //  causing the receiver's BKTree to miss heads at search time and
        //  recall to collapse on later batches. The retry queue + this
        //  thread make HeadSync delivery reliable best-effort.
        // ==================================================================

        struct HeadSyncBacklog {
            std::mutex mu;
            std::deque<HeadSyncEntry> queue;
            // Matches m_addCountForRebuild scale per peer. If we ever hit
            // this we log + drop (fall back to manual reconcile).
            static constexpr size_t kMaxEntries = 1u << 18;  // 262144
        };

        void StartHeadSyncRetryThread() {
            const char* envIntervalMs = std::getenv("SPTAG_HEADSYNC_RETRY_INTERVAL_MS");
            int intervalMs = 500;
            if (envIntervalMs) {
                try { intervalMs = std::max(50, std::stoi(envIntervalMs)); } catch (...) {}
            }
            m_headSyncRetryIntervalMs = intervalMs;
            m_headSyncRetryStop.store(false, std::memory_order_release);
            m_headSyncRetryThread = std::thread([this]() { HeadSyncRetryLoop(); });
        }

        void StopHeadSyncRetryThread() {
            m_headSyncRetryStop.store(true, std::memory_order_release);
            if (m_headSyncRetryThread.joinable()) m_headSyncRetryThread.join();
        }

        void HeadSyncRetryLoop() {
            using namespace std::chrono;
            while (!m_headSyncRetryStop.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(milliseconds(m_headSyncRetryIntervalMs));
                if (m_net) DrainHeadSyncBacklog();
            }
            // Final drain pass to give the network a chance to flush.
            for (int i = 0; i < 5 && m_net; i++) {
                size_t dispatched = DrainHeadSyncBacklog();
                if (dispatched == 0) break;
                std::this_thread::sleep_for(milliseconds(200));
            }
            if (m_headSyncBroadcastEntries.load(std::memory_order_relaxed) > 0
                || m_headSyncRecvEntries.load(std::memory_order_relaxed) > 0) {
                DumpHeadSyncStats("shutdown");
            }
            if (m_mergeBroadcastSendOK.load(std::memory_order_relaxed) > 0
                || m_mergeRecvHints.load(std::memory_order_relaxed) > 0) {
                DumpMergeRequestStats("shutdown");
            }
        }

        std::shared_ptr<HeadSyncBacklog> GetOrCreateBacklog(int nodeIdx) {
            {
                std::shared_lock<std::shared_timed_mutex> lk(m_headSyncBacklogsMu);
                auto it = m_headSyncBacklogs.find(nodeIdx);
                if (it != m_headSyncBacklogs.end()) return it->second;
            }
            std::unique_lock<std::shared_timed_mutex> lk(m_headSyncBacklogsMu);
            auto& slot = m_headSyncBacklogs[nodeIdx];
            if (!slot) slot = std::make_shared<HeadSyncBacklog>();
            return slot;
        }

        // ---- State ----

        NetworkAccess* m_net = nullptr;

        // Per-layer callback registries. Indexed by ExtraDynamicSearcher layer
        // (m_layer at the call site). Resized lazily by SetXxxCallback. The
        // empty/null entry at layer 0 is preserved so a single-layer caller
        // (legacy or test) without explicit Set keeps the no-op default.
        //
        // The shared-callback design existed because the original SPANN had
        // a single ExtraDynamicSearcher (Layers=1). With Layers>=2, each
        // layer's lambda captures its own `this` (hence m_layer) and dispatch
        // by request.m_layer is required to avoid routing layer-0 events to
        // layer-1's storage and vice versa.
        std::vector<AppendCallback> m_appendCallbacks;
        std::vector<HeadSyncCallback> m_headSyncCallbacks;
        std::vector<RemoteLockCallback> m_remoteLockCallbacks;
        std::vector<MergeCallback> m_mergeCallbacks;

        // Per-layer ownership tokens. Each ExtraDynamicSearcher claims its
        // layer slot at SetWorker time and releases it on destruction; this
        // prevents earlier-layer destructors from wiping a later-layer's
        // callbacks (the original ClaimCallbackOwnership purpose, now
        // applied per-layer instead of globally).
        std::vector<std::atomic<const void*>> m_callbackOwners;

        // Guards the lifetime of the captured `this` inside the callbacks.
        // Held in shared mode by every callback invocation site, and in
        // exclusive mode by ClearCallbacks() / SetXxxCallback() so that
        // (re)assigning a callback can never race with an in-flight invocation.
        mutable std::shared_timed_mutex m_callbackLifetimeMutex;

        std::atomic<Socket::ResourceID> m_nextResourceId{1};
        std::mutex m_pendingMutex;
        std::unordered_map<Socket::ResourceID, std::promise<ErrorCode>> m_pendingResponses;

        // Per-item Job: each remote append request becomes one Job submitted
        // to the searcher's shared SPDKThreadPool. The last completing Job
        // ACKs the sender. Identical to how a local insert thread would call
        // Append; the only difference is the request originated on a peer.
        class BatchAppendItemJob : public Helper::ThreadPool::Job {
        public:
            BatchAppendItemJob(RemotePostingOps* ops,
                               std::shared_ptr<BatchRemoteAppendRequest> batchReq,
                               size_t index,
                               std::shared_ptr<std::atomic<size_t>> remaining,
                               std::shared_ptr<std::atomic<std::uint32_t>> successCount,
                               std::shared_ptr<std::atomic<std::uint32_t>> failCount,
                               std::shared_ptr<Socket::Packet> replyPacket)
                : m_ops(ops), m_batchReq(std::move(batchReq)), m_index(index),
                  m_remaining(std::move(remaining)),
                  m_success(std::move(successCount)),
                  m_fail(std::move(failCount)),
                  m_replyPacket(std::move(replyPacket)) {}

            void exec(IAbortOperation*) override { run(); }
            void exec(void* workspace, IAbortOperation*) override {
                void* prev = tls_preallocAppendWorkSpace;
                tls_preallocAppendWorkSpace = workspace;
                run();
                tls_preallocAppendWorkSpace = prev;
            }

        private:
            void run() {
                {
                    std::shared_lock<std::shared_timed_mutex> cbLock(m_ops->m_callbackLifetimeMutex);
                    auto& req = m_batchReq->m_items[m_index];
                    ErrorCode r = ErrorCode::Fail;
                    const auto* cb = m_ops->LookupAppendCallback_Locked(req.m_layer);
                    if (cb) {
                        auto hv = std::make_shared<std::string>(std::move(req.m_headVec));
                        r = (*cb)(req.m_headID, hv, req.m_appendNum, req.m_appendPosting);
                    }
                    if (r == ErrorCode::Success) m_success->fetch_add(1);
                    else                         m_fail->fetch_add(1);
                }
                if (m_remaining->fetch_sub(1) == 1) {
                    m_ops->SendBatchAppendResponse(
                        *m_replyPacket, m_success->load(), m_fail->load());
                }
            }

            RemotePostingOps* m_ops;
            std::shared_ptr<BatchRemoteAppendRequest> m_batchReq;
            size_t m_index;
            std::shared_ptr<std::atomic<size_t>> m_remaining;
            std::shared_ptr<std::atomic<std::uint32_t>> m_success;
            std::shared_ptr<std::atomic<std::uint32_t>> m_fail;
            std::shared_ptr<Socket::Packet> m_replyPacket;
        };

        // [Bug 26 retired] bg executor removed — see HandleBatchAppendRequest.
        // m_bgWorkers etc were replaced by per-layer job submission into the
        // searcher's shared SPDKThreadPool via m_jobSubmitters[layer].
        std::vector<JobSubmitter> m_jobSubmitters;

        // HeadSync delivery diagnostics + retry queue (v33). Counters give
        // observability for sender/receiver gaps; per-peer backlogs +
        // retry thread make broadcast reliable best-effort.
        std::atomic<std::uint64_t> m_headSyncBroadcastEntries{0};
        std::atomic<std::uint64_t> m_headSyncBroadcastSendOK{0};
        std::atomic<std::uint64_t> m_headSyncBroadcastSendFail{0};
        std::atomic<std::uint64_t> m_headSyncRecvEntries{0};
        std::atomic<std::uint64_t> m_headSyncApplyAdd{0};
        std::atomic<std::uint64_t> m_headSyncApplyDelete{0};
        std::atomic<std::uint64_t> m_headSyncRetryEnqueued{0};
        std::atomic<std::uint64_t> m_headSyncRetrySucceeded{0};
        std::atomic<std::uint64_t> m_headSyncRetryDropped{0};

        // Cross-node merge hint counters. No retry queue: dropped
        // notifications are recoverable since the owner discovers underfull
        // postings via its own paths too.
        std::atomic<std::uint64_t> m_mergeBroadcastSendOK{0};
        std::atomic<std::uint64_t> m_mergeBroadcastSendFail{0};
        std::atomic<std::uint64_t> m_mergeRecvHints{0};
        std::atomic<std::uint64_t> m_mergeRecvDropped{0};

        mutable std::shared_timed_mutex m_headSyncBacklogsMu;
        std::unordered_map<int, std::shared_ptr<HeadSyncBacklog>> m_headSyncBacklogs;
        std::thread m_headSyncRetryThread;
        std::atomic<bool> m_headSyncRetryStop{false};
        int m_headSyncRetryIntervalMs{500};
    };

} // namespace SPTAG::SPANN
