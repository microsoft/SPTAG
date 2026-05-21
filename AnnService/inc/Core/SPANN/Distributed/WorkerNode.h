// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_WORKERNODE_H_
#define _SPTAG_SPANN_WORKERNODE_H_

#include "inc/Core/SPANN/Distributed/NetworkNode.h"
#include "inc/Core/SPANN/Distributed/AsyncJobWatchdog.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Socket/SimpleSerialization.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <functional>
#include <future>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

namespace SPTAG::SPANN {

    /// Distributed compute worker node.
    ///
    /// Responsibilities:
    ///   - Route headIDs to owner nodes via consistent hash ring
    ///   - Queue and flush remote appends (batched RPC)
    ///   - HeadSync broadcast and remote locking
    ///   - Register with dispatcher and receive ring updates
    ///   - Handle incoming dispatch commands from the driver
    class WorkerNode : public NetworkNode {
    public:
        using AppendCallback = RemotePostingOps::AppendCallback;
        using DispatchCallback = DispatchCoordinator::DispatchCallback;
        using HeadSyncCallback = RemotePostingOps::HeadSyncCallback;
        using RemoteLockCallback = RemotePostingOps::RemoteLockCallback;
        using FenceValidator = RemotePostingOps::FenceValidator;

        /// Initialize with separate dispatcher/worker/store addresses.
        /// workerIndex is 0-based (0 = driver/local, 1+ = remote).
        /// Internal node index = workerIndex + 1 (0 is reserved for dispatcher).
        bool Initialize(
            std::shared_ptr<Helper::KeyValueIO> p_db,
            int workerIndex,
            const std::pair<std::string, std::string>& dispatcherAddr,
            const std::vector<std::pair<std::string, std::string>>& workerAddrs,
            const std::vector<std::string>& storeAddrs,
            int vnodeCount = 150)
        {
            if (storeAddrs.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "WorkerNode::Initialize: storeAddrs is empty\n");
                return false;
            }

            // Build combined addr list: [dispatcher, worker0, worker1, ...]
            std::vector<std::pair<std::string, std::string>> allAddrs;
            allAddrs.push_back(dispatcherAddr);
            allAddrs.insert(allAddrs.end(), workerAddrs.begin(), workerAddrs.end());

            int internalIdx = workerIndex + 1;  // 0 = dispatcher, 1..N = workers
            if (!InitializeNetwork(internalIdx, allAddrs, vnodeCount)) return false;

            // [Bug 30] Populate compute-role fields so callers can ask
            // "how many data shards?" / "which shard am I?" without
            // accidentally including the dispatcher slot.
            m_numDispatchNodes = 1;
            m_numWorkerNodes = static_cast<int>(workerAddrs.size());
            m_workerNodeIndex = workerIndex;

            m_db = p_db;
            m_nodeStores = storeAddrs;

            // Build store → node list mapping (worker internal indices 1..N)
            int numWorkers = static_cast<int>(workerAddrs.size());
            int numStores = static_cast<int>(storeAddrs.size());
            for (int wi = 0; wi < numWorkers; wi++) {
                int storeIdx = wi % numStores;
                m_storeToNodes[storeAddrs[storeIdx]].push_back(wi + 1);
            }
            for (auto& [store, nodes] : m_storeToNodes) {
                std::string nodeList;
                for (int n : nodes) { nodeList += std::to_string(n) + " "; }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "WorkerNode: store %s → nodes [%s]\n", store.c_str(), nodeList.c_str());
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "WorkerNode: initialized (workerIndex=%d, internalIdx=%d, %d stores, %d vnodes/node)\n",
                workerIndex, internalIdx, numStores, vnodeCount);

            m_dispatch.SetNetwork(this);
            m_remoteOps.SetNetwork(this);

            return true;
        }

    public:
        bool Start() { return StartNetwork(); }

        // ---- Callbacks ----
        //
        // ExtraDynamicSearcher passes its m_layer when binding callbacks so
        // that with multi-layer SPANN (Layers >= 2) each layer has its own
        // captured `this` and request dispatch on the receiver side routes by
        // request.m_layer.

        void SetAppendCallback(int layer, AppendCallback cb) { m_remoteOps.SetAppendCallback(layer, std::move(cb)); }
        void SetHeadSyncCallback(int layer, HeadSyncCallback cb) { m_remoteOps.SetHeadSyncCallback(layer, std::move(cb)); }
        void SetRemoteLockCallback(int layer, RemoteLockCallback cb) { m_remoteOps.SetRemoteLockCallback(layer, std::move(cb)); }
        void SetFenceValidator(int layer, FenceValidator cb) { m_remoteOps.SetFenceValidator(layer, std::move(cb)); }
        // Inject the searcher's shared compute pool so receiver-side
        // BatchAppend work runs there (high-priority Jobs) instead of in a
        // separate executor. Idempotent: safe to call multiple times.
        void SetJobSubmitter(int layer, RemotePostingOps::JobSubmitter s) {
            m_remoteOps.SetJobSubmitter(layer, std::move(s));
        }
        /// Atomically clear all RPC callbacks (every layer) and wait for any
        /// in-flight invocation to finish.
        void ClearCallbacks() {
            m_remoteOps.ClearCallbacks();
        }
        /// Per-layer ownership API used by ExtraDynamicSearcher to avoid having
        /// one layer's destructor wipe another layer's still-active callbacks.
        /// SetWorker calls ClaimCallbackOwnership(m_layer, this) before
        /// registering; the destructor calls ClearCallbacksIfOwner(m_layer, this).
        void ClaimCallbackOwnership(int layer, const void* owner) {
            m_remoteOps.ClaimCallbackOwnership(layer, owner);
        }
        bool ClearCallbacksIfOwner(int layer, const void* owner) {
            return m_remoteOps.ClearCallbacksIfOwner(layer, owner);
        }
        void SetDispatchCallback(DispatchCallback cb) { m_dispatch.SetDispatchCallback(std::move(cb)); }
        void ClearDispatchCallback() { m_dispatch.ClearDispatchCallback(); }

        // RPC tuning forwarders.  See RemotePostingOps for semantics.
        // MaxInflightPerNode caps how many auto-flush chunks may be on
        // the wire to a given peer at once; chunk size/retry/timeout
        // are forwarded directly into RemotePostingOps.
        void SetRpcChunkSize(int v) { m_remoteOps.SetRpcChunkSize(v); }
        void SetRpcRetry(int v) { m_remoteOps.SetRpcRetry(v); }
        void SetRpcTimeoutSec(int v) { m_remoteOps.SetRpcTimeoutSec(v); }
        void SetRpcMaxInflightPerNode(int v) {
            if (v > 0) m_maxInflightPerNode.store(v, std::memory_order_relaxed);
        }

        // ---- Routing ----

        RouteTarget GetOwner(SizeType headID) {
            RouteTarget target;
            target.isLocal = true;
            target.nodeIndex = m_localNodeIndex;

            if (!m_enabled) {
                m_routeStats.disabled++;
                return target;
            }
            {
                auto ring = std::atomic_load(&m_hashRing);
                if (!ring || ring->NodeCount() <= 1) {
                    m_routeStats.local++;
                    return target;
                }
                target.nodeIndex = ring->GetOwner(headID);
            }
            target.isLocal = (target.nodeIndex == m_localNodeIndex);
            if (target.isLocal) m_routeStats.local++;
            else m_routeStats.remote++;
            return target;
        }

        void LogRouteStats(const char* context = "") {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "WorkerNode stats%s: local=%d remote=%d disabled=%d keyMiss=%d noMapping=%d\n",
                context, (int)m_routeStats.local, (int)m_routeStats.remote,
                (int)m_routeStats.disabled, (int)m_routeStats.keyMiss,
                (int)m_routeStats.noMapping);
        }

        void ResetRouteStats() {
            m_routeStats.local.store(0);
            m_routeStats.remote.store(0);
            m_routeStats.disabled.store(0);
            m_routeStats.keyMiss.store(0);
            m_routeStats.noMapping.store(0);
        }

        // ---- Remote posting ops ----

        ErrorCode SendRemoteAppend(int targetNodeIndex, int layer, SizeType headID,
            const std::shared_ptr<std::string>& headVec, int appendNum,
            std::string& appendPosting)
        {
            return m_remoteOps.SendRemoteAppend(targetNodeIndex, layer, headID, headVec, appendNum, appendPosting);
        }

        ErrorCode SendBatchRemoteAppend(int targetNodeIndex, std::vector<RemoteAppendRequest>& items) {
            return m_remoteOps.SendBatchRemoteAppend(targetNodeIndex, items);
        }

        void BroadcastHeadSync(const std::vector<HeadSyncEntry>& entries) {
            if (!m_enabled) return;
            m_remoteOps.BroadcastHeadSync(entries);
        }

        // v33: expose HeadSync delivery diagnostics + retry queue.
        void DumpHeadSyncStats(const char* label) const {
            m_remoteOps.DumpHeadSyncStats(label);
        }
        // Cross-node merge-hint channel diagnostics.
        void DumpMergeRequestStats(const char* label) const {
            m_remoteOps.DumpMergeRequestStats(label);
        }
        size_t GetHeadSyncBacklogSize() const {
            return m_remoteOps.GetHeadSyncBacklogSize();
        }
        size_t DrainHeadSyncBacklog(size_t maxBatch = 1024) {
            return m_remoteOps.DrainHeadSyncBacklog(maxBatch);
        }
        void NoteHeadSyncApplyAdd() {
            m_remoteOps.NoteHeadSyncApplyAdd();
        }
        void NoteHeadSyncApplyDelete() {
            m_remoteOps.NoteHeadSyncApplyDelete();
        }

        // Returns issued fencing token on Lock success (0 = denied),
        // or 1 on Unlock accepted (0 = rejected / stale token).
        std::uint64_t SendRemoteLock(int nodeIndex, int layer, SizeType headID,
                                     bool lock, std::uint64_t token = 0) {
            if (!m_enabled) return 0;
            return m_remoteOps.SendRemoteLock(nodeIndex, layer, headID, lock, token);
        }

        // Synchronous, fenced remote append: includes the fencing token
        // so the owner can validate that the writer still holds the
        // bucket lease before applying.  Returns Success/Fail.
        ErrorCode SendFencedRemoteAppend(int nodeIndex, int layer, SizeType headID,
                                         const std::shared_ptr<std::string>& headVec,
                                         int appendNum, std::string& appendPosting,
                                         std::uint64_t fencingToken) {
            if (!m_enabled) return ErrorCode::Fail;
            return m_remoteOps.SendRemoteAppend(nodeIndex, layer, headID, headVec,
                                                appendNum, appendPosting, fencingToken);
        }

        void SetMergeCallback(int layer, RemotePostingOps::MergeCallback cb) {
            m_remoteOps.SetMergeCallback(layer, std::move(cb));
        }

        // ---- Append queue ----

        void QueueRemoteAppend(int nodeIndex, RemoteAppendRequest req) {
            std::vector<RemoteAppendRequest> toFlush;
            bool didReserveSlot = false;
            {
                std::lock_guard<std::mutex> lock(m_appendQueueMutex);
                auto& q = m_appendQueue[nodeIndex];
                q.push_back(std::move(req));
                m_remoteQueueSize.fetch_add(1, std::memory_order_relaxed);
                m_totalRemoteAppendsRouted.fetch_add(1, std::memory_order_relaxed);
                // [PERF] Auto-flush per node once we have a full chunk worth
                // (kAutoFlushThreshold items). Without this, every remote
                // append accumulates until end-of-batch FlushRemoteAppends —
                // which then sends hundreds of thousands of items serially
                // (10k chunks * ~3s/chunk) AFTER all insert compute is done.
                // Auto-flushing while inserts keep running overlaps the
                // network with CPU and drops end-of-batch tail latency.
                //
                // [v38] Allow up to kMaxInflightPerNode concurrent in-flight
                // chunks per node so a producer burst (split fan-out, reassign
                // wave) can saturate the receiver's bg-executor pool instead of
                // queueing up serially behind a single per-node mutex.
                if (q.size() >= kAutoFlushThreshold
                    && m_perNodeInflight[nodeIndex] < m_maxInflightPerNode.load(std::memory_order_relaxed)) {
                    toFlush.swap(q);
                    m_remoteQueueSize.fetch_sub(toFlush.size(), std::memory_order_relaxed);
                    ++m_perNodeInflight[nodeIndex];
                    didReserveSlot = true;
                }
            }
            if (!didReserveSlot) return;

            // Fire-and-forget async send. After the initial chunk completes,
            // the same thread loops to pick up any further accumulation so we
            // avoid thread-spawn churn while keeping per-node concurrency at
            // kMaxInflightPerNode. Order across batches is best-effort: the
            // receiver runs 8 worker threads on each chunk that already
            // interleave items within a chunk, so cross-chunk ordering adds
            // no extra correctness risk for the per-posting RMW path.
            auto items = std::make_shared<std::vector<RemoteAppendRequest>>(std::move(toFlush));
            m_inflightAppendFlushes.fetch_add(1, std::memory_order_relaxed);
            std::thread([this, nodeIndex, items]() {
                while (true) {
                    ErrorCode ret = SendBatchRemoteAppend(nodeIndex, *items);
                    if (ret != ErrorCode::Success) {
                        // Hand the failed batch to the watchdog. It owns
                        // backoff/retry until MaxAttempts; RemoteAppend is
                        // idempotent on the receive side so at-least-once
                        // delivery is safe.
                        auto retryItems =
                            std::make_shared<std::vector<RemoteAppendRequest>>(*items);
                        int n = nodeIndex;
                        auto self = this;
                        std::string tag = "QueueRemoteAppend node=" +
                            std::to_string(n) + " items=" +
                            std::to_string(retryItems->size());
                        uint64_t id = m_asyncWatchdog.Track(
                            [self, n, retryItems]() {
                                return self->SendBatchRemoteAppend(n, *retryItems)
                                    == ErrorCode::Success;
                            }, std::move(tag));
                        m_asyncWatchdog.MarkFailureAndScheduleResend(id);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "QueueRemoteAppend auto-flush: batch to node %d failed (%zu items), handed to watchdog\n",
                            nodeIndex, items->size());
                    }
                    items->clear();
                    {
                        std::lock_guard<std::mutex> lock(m_appendQueueMutex);
                        auto it = m_appendQueue.find(nodeIndex);
                        if (it == m_appendQueue.end()
                            || it->second.size() < kAutoFlushThreshold) {
                            --m_perNodeInflight[nodeIndex];
                            break;
                        }
                        items->swap(it->second);
                        m_remoteQueueSize.fetch_sub(items->size(),
                            std::memory_order_relaxed);
                    }
                }
                m_inflightAppendFlushes.fetch_sub(1, std::memory_order_relaxed);
            }).detach();
        }

        size_t GetRemoteQueueSize() const {
            return m_remoteQueueSize.load(std::memory_order_relaxed);
        }

        // Number of remote append items submitted via QueueRemoteAppend over
        // this WorkerNode's lifetime.  Used by ExtraDynamicSearcher progress
        // logging so users can tell whether "ALL DONE" on the local pool is
        // misleading because the remote send queue still has backlog.
        size_t GetTotalRemoteAppendsRouted() const {
            return m_totalRemoteAppendsRouted.load(std::memory_order_relaxed);
        }
        // In-flight chunk count across all peers (auto-flush async sends
        // currently running).
        int GetInflightAppendFlushes() const {
            return m_inflightAppendFlushes.load(std::memory_order_relaxed);
        }

        ErrorCode FlushRemoteAppends() {
            // Drain the queue under m_flushMutex so concurrent flush callers
            // serialize. Loop in case items get queued mid-send. This avoids
            // the thundering-herd of 100+ concurrent FlushRemoteAppends calls
            // (one per split worker) overwhelming the remote node's tiny
            // (8-thread, 256-connection-pool) network server.
            std::lock_guard<std::mutex> flushGuard(m_flushMutex);

            // Wait for any in-flight async auto-flushes triggered by
            // QueueRemoteAppend (>= kAutoFlushThreshold) to drain so the
            // residue we send below is the actual tail. Callers invoke
            // FlushRemoteAppends after all producers (AddIndex / split /
            // reassign) have quiesced, so no new auto-flushes will start
            // here.
            while (m_inflightAppendFlushes.load(std::memory_order_relaxed) > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }

            int errors = 0;
            int iterations = 0;
            while (true) {
                std::unordered_map<int, std::vector<RemoteAppendRequest>> toSend;
                {
                    std::lock_guard<std::mutex> lock(m_appendQueueMutex);
                    if (m_appendQueue.empty()) break;
                    toSend.swap(m_appendQueue);
                    m_remoteQueueSize.store(0, std::memory_order_relaxed);
                }
                if (toSend.empty()) break;
                ++iterations;

                std::atomic<int> iterErrors{0};
                std::vector<std::thread> threads;
                for (auto& [nodeIdx, items] : toSend) {
                    if (items.empty()) continue;
                    threads.emplace_back([this, &iterErrors, nodeIdx, &items]() {
                        // Per-node mutex serializes against any straggler
                        // auto-flush still in flight for this node.
                        std::mutex& nodeMtx = GetPerNodeAppendFlushMutex(nodeIdx);
                        std::lock_guard<std::mutex> nlock(nodeMtx);
                        ErrorCode ret = SendBatchRemoteAppend(nodeIdx, items);
                        if (ret != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "FlushRemoteAppends: batch to node %d failed (%d items)\n",
                                nodeIdx, (int)items.size());
                            iterErrors++;
                        }
                    });
                }
                for (auto& t : threads) t.join();
                errors += iterErrors.load();
            }
            return errors > 0 ? ErrorCode::Fail : ErrorCode::Success;
        }

        // ---- Cross-node merge hint queue ----
        //
        // Search-side fire-and-forget notifications: node X sees posting H
        // underfull, where H is owned by Y. We dedup (layer, headID) within
        // a flush window and batch-send to Y in one packet. The receiver's
        // m_mergeList dedups on top of this, so an occasional dropped or
        // duplicated notification only costs a few cycles.
        void QueueRemoteMerge(int nodeIndex, int layer, SizeType headID) {
            std::vector<RemoteMergeRequest> toFlush;
            {
                std::lock_guard<std::mutex> lock(m_mergeQueueMutex);
                std::int64_t key = (static_cast<std::int64_t>(layer) << 32)
                                 | static_cast<std::uint32_t>(headID);
                auto& bucket = m_mergeQueue[nodeIndex];
                if (!bucket.insert(key).second) return;  // already pending
                m_mergeQueueSize.fetch_add(1, std::memory_order_relaxed);

                if (bucket.size() >= kMergeAutoFlushThreshold) {
                    toFlush.reserve(bucket.size());
                    for (std::int64_t k : bucket) {
                        RemoteMergeRequest req;
                        req.m_layer = static_cast<std::int32_t>(k >> 32);
                        req.m_headID = static_cast<SizeType>(static_cast<std::int32_t>(k & 0xFFFFFFFF));
                        toFlush.push_back(std::move(req));
                    }
                    m_mergeQueueSize.fetch_sub(bucket.size(), std::memory_order_relaxed);
                    bucket.clear();
                }
            }
            if (!toFlush.empty()) {
                m_remoteOps.SendBatchRemoteMerge(nodeIndex, toFlush);
            }
        }

        ErrorCode FlushRemoteMerges() {
            std::unordered_map<int, std::vector<RemoteMergeRequest>> toSend;
            {
                std::lock_guard<std::mutex> lock(m_mergeQueueMutex);
                if (m_mergeQueue.empty()) return ErrorCode::Success;
                for (auto& [nodeIdx, bucket] : m_mergeQueue) {
                    auto& vec = toSend[nodeIdx];
                    vec.reserve(bucket.size());
                    for (std::int64_t k : bucket) {
                        RemoteMergeRequest req;
                        req.m_layer = static_cast<std::int32_t>(k >> 32);
                        req.m_headID = static_cast<SizeType>(static_cast<std::int32_t>(k & 0xFFFFFFFF));
                        vec.push_back(std::move(req));
                    }
                }
                m_mergeQueue.clear();
                m_mergeQueueSize.store(0, std::memory_order_relaxed);
            }
            for (auto& [nodeIdx, items] : toSend) {
                if (!items.empty()) m_remoteOps.SendBatchRemoteMerge(nodeIdx, items);
            }
            return ErrorCode::Success;
        }

        // ---- Ring protocol (worker side) ----

        bool WaitForRing(int timeoutSec = 120) {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
            while (std::chrono::steady_clock::now() < deadline) {
                auto ring = std::atomic_load(&m_hashRing);
                if (ring && ring->NodeCount() > 0) return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "WorkerNode: Timed out waiting for ring (%ds)\n", timeoutSec);
            return false;
        }

        // ---- Data members (public for ExtraDynamicSearcher access) ----

        std::shared_ptr<Helper::KeyValueIO> m_db;
        std::vector<std::string> m_nodeStores;
        std::unordered_map<std::string, std::vector<int>> m_storeToNodes;

        struct RouteStats {
            std::atomic<int> local{0};
            std::atomic<int> remote{0};
            std::atomic<int> disabled{0};
            std::atomic<int> keyMiss{0};
            std::atomic<int> noMapping{0};
        } m_routeStats;

    protected:
        void RegisterServerHandlers(Socket::PacketHandlerMapPtr& handlers) override {
            handlers->emplace(Socket::PacketType::AppendRequest,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleAppendRequest(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::BatchAppendRequest,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleBatchAppendRequest(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::HeadSyncRequest,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleHeadSyncRequest(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::RemoteLockRequest,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleRemoteLockRequest(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::MergeRequest,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleMergeRequest(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::DispatchCommand,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_dispatch.HandleDispatchCommand(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::DispatchResult,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_dispatch.HandleDispatchResult(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::RingUpdate,
                [this](Socket::ConnectionID c, Socket::Packet p) { HandleRingUpdate(c, std::move(p)); });
        }

        void RegisterClientHandlers(Socket::PacketHandlerMapPtr& handlers) override {
            handlers->emplace(Socket::PacketType::AppendResponse,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleAppendResponse(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::BatchAppendResponse,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleBatchAppendResponse(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::RemoteLockResponse,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_remoteOps.HandleRemoteLockResponse(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::DispatchResult,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_dispatch.HandleDispatchResult(c, std::move(p)); });
        }

        void BgProtocolStep() override {
            // Keep sending NodeRegister until ring is populated
            auto ring = std::atomic_load(&m_hashRing);
            if (!ring || ring->NodeCount() == 0) {
                Socket::ConnectionID connID = Socket::c_invalidConnectionID;
                {
                    std::lock_guard<std::mutex> lock(m_connMutex);
                    if (m_dispatcherNodeIndex < (int)m_peerConnections.size())
                        connID = m_peerConnections[m_dispatcherNodeIndex];
                }
                if (connID != Socket::c_invalidConnectionID) {
                    SendNodeRegister();
                }
            }
        }

        bool IsRingSettled() const override {
            auto ring = std::atomic_load(&m_hashRing);
            return ring && ring->NodeCount() > 0;
        }

    private:
        void SendNodeRegister() {
            NodeRegisterMsg msg;
            msg.m_nodeIndex = m_localNodeIndex;
            msg.m_host = m_nodeAddrs[m_localNodeIndex].first;
            msg.m_port = m_nodeAddrs[m_localNodeIndex].second;
            // Worker's 0-based index = m_localNodeIndex - 1 (since 0 is dispatcher)
            int workerIdx = m_localNodeIndex - 1;
            int numStores = static_cast<int>(m_nodeStores.size());
            msg.m_store = (numStores > 0) ? m_nodeStores[workerIdx % numStores] : "";

            std::size_t bodySize = msg.EstimateBufferSize();
            Socket::Packet pkt;
            pkt.Header().m_packetType = Socket::PacketType::NodeRegisterRequest;
            pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
            pkt.Header().m_resourceID = 0;
            pkt.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
            pkt.AllocateBuffer(static_cast<std::uint32_t>(bodySize));
            msg.Write(pkt.Body());
            pkt.Header().WriteBuffer(pkt.HeaderBuffer());

            auto connID = GetPeerConnection(m_dispatcherNodeIndex);
            if (connID != Socket::c_invalidConnectionID) {
                m_client->SendPacket(connID, std::move(pkt), nullptr);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "WorkerNode: Sent NodeRegister (node %d) to dispatcher\n", m_localNodeIndex);
            }
        }

        void HandleRingUpdate(Socket::ConnectionID connID, Socket::Packet packet) {
            RingUpdateMsg msg;
            if (!msg.Read(packet.Body())) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WorkerNode: Failed to parse RingUpdate\n");
                return;
            }

            auto newRing = std::make_shared<ConsistentHashRing>(msg.m_vnodeCount);
            for (auto idx : msg.m_nodeIndices) {
                newRing->AddNode(idx);
            }
            {
                std::lock_guard<std::mutex> guard(m_ringWriteMutex);
                std::atomic_store(&m_hashRing,
                    std::shared_ptr<const ConsistentHashRing>(std::move(newRing)));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "WorkerNode: Ring updated — %d nodes (v%u)\n",
                (int)msg.m_nodeIndices.size(), msg.m_ringVersion);

            SendRingUpdateACK(msg.m_ringVersion);
        }

        void SendRingUpdateACK(std::uint32_t ringVersion) {
            RingUpdateACKMsg msg;
            msg.m_nodeIndex = m_localNodeIndex;
            msg.m_ringVersion = ringVersion;

            std::size_t bodySize = msg.EstimateBufferSize();
            Socket::Packet pkt;
            pkt.Header().m_packetType = Socket::PacketType::RingUpdateACK;
            pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
            pkt.Header().m_resourceID = 0;
            pkt.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
            pkt.AllocateBuffer(static_cast<std::uint32_t>(bodySize));
            msg.Write(pkt.Body());
            pkt.Header().WriteBuffer(pkt.HeaderBuffer());

            auto connID = GetPeerConnection(m_dispatcherNodeIndex);
            if (connID != Socket::c_invalidConnectionID) {
                m_client->SendPacket(connID, std::move(pkt), nullptr);
            }
        }

        int m_dispatcherNodeIndex = 0;
        RemotePostingOps m_remoteOps;
        DispatchCoordinator m_dispatch;

        mutable std::mutex m_appendQueueMutex;
        std::unordered_map<int, std::vector<RemoteAppendRequest>> m_appendQueue;
        std::atomic<size_t> m_remoteQueueSize{0};
        // Cumulative count of items handed to QueueRemoteAppend over this
        // worker's lifetime (does not decrement on send completion).
        std::atomic<size_t> m_totalRemoteAppendsRouted{0};
        // Serializes concurrent FlushRemoteAppends() callers so we don't open
        // hundreds of simultaneous RPC streams to the remote worker (which has
        // only 8 server threads / 256 connection slots). With this mutex, only
        // one thread sends at a time; concurrent callers either wait for the
        // current flush to finish or contribute their items to the queue.
        std::mutex m_flushMutex;

        // Per-node mutex used by end-of-batch FlushRemoteAppends so concurrent
        // sends to the SAME node from the final-drain path remain ordered.
        // Auto-flushes (QueueRemoteAppend) instead use m_perNodeInflight to
        // cap concurrency at kMaxInflightPerNode per node.
        std::mutex m_perNodeAppendFlushMutexMapLock;
        std::unordered_map<int, std::unique_ptr<std::mutex>> m_perNodeAppendFlushMutex;
        std::atomic<int> m_inflightAppendFlushes{0};
        std::unordered_map<int, int> m_perNodeInflight; // guarded by m_appendQueueMutex
        static constexpr size_t kAutoFlushThreshold = 50000;
        std::atomic<int> m_maxInflightPerNode{4};

        // Resends failed async fire-and-forget batches with exponential
        // backoff (see AsyncJobWatchdog.h). Constructed last so it tears
        // down before the queues; declared here so destruction order
        // matches the design's fault-tolerance contract.
        Distributed::AsyncJobWatchdog m_asyncWatchdog{3, 200};

        std::mutex& GetPerNodeAppendFlushMutex(int nodeIndex) {
            std::lock_guard<std::mutex> lk(m_perNodeAppendFlushMutexMapLock);
            auto it = m_perNodeAppendFlushMutex.find(nodeIndex);
            if (it == m_perNodeAppendFlushMutex.end()) {
                auto ins = m_perNodeAppendFlushMutex.emplace(
                    nodeIndex, std::make_unique<std::mutex>());
                return *ins.first->second;
            }
            return *it->second;
        }

        // Cross-node merge hint queue. Per-target dedup set of packed
        // (layer << 32 | headID) values; QueueRemoteMerge inserts and
        // auto-flushes when the per-target bucket reaches threshold.
        mutable std::mutex m_mergeQueueMutex;
        std::unordered_map<int, std::unordered_set<std::int64_t>> m_mergeQueue;
        std::atomic<size_t> m_mergeQueueSize{0};
        // Merge hints are non-urgent (best-effort optimization). A larger
        // bucket trades a small amount of latency for much better dedup and
        // network batching. End-of-batch FlushRemoteMerges() guarantees no
        // hint is permanently dropped.
        static constexpr size_t kMergeAutoFlushThreshold = 8192;
    };

} // namespace SPTAG::SPANN

#endif // _SPTAG_SPANN_WORKERNODE_H_
