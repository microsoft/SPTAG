// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/SPANN/Distributed/NetworkNode.h"

namespace SPTAG::SPANN {

    /// Dispatcher node: manages the consistent hash ring and coordinates
    /// external dispatch commands (Insert/Search/Stop) to worker nodes.
    ///
    /// The dispatcher does NOT perform search or posting operations.
    /// It is a lightweight coordination point that:
    ///   - Accepts NodeRegister requests from workers
    ///   - Maintains the authoritative hash ring and broadcasts updates
    ///   - Tracks per-worker ACK status with retry
    ///   - Delegates BroadcastDispatchCommand / WaitForAllResults
    class DispatcherNode : public NetworkNode {
    public:
        using DispatchCallback = DispatchCoordinator::DispatchCallback;

        /// Initialize the dispatcher with separate addresses.
        /// Builds the full hash ring at startup (workers 1..N).
        bool Initialize(
            const std::pair<std::string, std::string>& dispatcherAddr,
            const std::vector<std::pair<std::string, std::string>>& workerAddrs,
            int vnodeCount = 150)
        {
            // Build combined addr list: [dispatcher, worker0, worker1, ...]
            std::vector<std::pair<std::string, std::string>> allAddrs;
            allAddrs.push_back(dispatcherAddr);
            allAddrs.insert(allAddrs.end(), workerAddrs.begin(), workerAddrs.end());

            if (!InitializeNetwork(0, allAddrs, vnodeCount)) return false;

            // [Bug 30] Dispatcher has no local data shard; mark with -1.
            m_numDispatchNodes = 1;
            m_numWorkerNodes = static_cast<int>(workerAddrs.size());
            m_workerNodeIndex = -1;

            // Pre-build complete ring with all workers (internal indices 1..N)
            int numWorkers = static_cast<int>(workerAddrs.size());
            auto ring = std::make_shared<ConsistentHashRing>(vnodeCount);
            for (int i = 1; i <= numWorkers; i++) {
                ring->AddNode(i);
            }
            std::atomic_store(&m_hashRing,
                std::shared_ptr<const ConsistentHashRing>(std::move(ring)));
            m_currentRingVersion.store(1);

            m_dispatch.SetNetwork(this);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatcherNode: initialized with %d workers, ring v1\n", numWorkers);
            return true;
        }

        bool Start() { return StartNetwork(); }

        // ---- Dispatch protocol ----

        /// Mark the driver's local worker node so broadcasts skip it.
        void SetLocalWorkerIndex(int idx) { m_dispatch.SetLocalWorkerIndex(idx); }

        std::uint64_t BroadcastDispatchCommand(DispatchCommand::Type type, std::uint32_t round) {
            return m_dispatch.BroadcastDispatchCommand(type, round);
        }

        std::vector<double> WaitForAllResults(std::uint64_t dispatchId, int timeoutSec = 300) {
            return m_dispatch.WaitForAllResults(dispatchId, timeoutSec);
        }

        void SetDispatchCallback(DispatchCallback cb) {
            m_dispatch.SetDispatchCallback(std::move(cb));
        }

        void ClearDispatchCallback() {
            m_dispatch.ClearDispatchCallback();
        }

        // ---- Heartbeat pump ----
        //
        // Periodically broadcasts a Heartbeat dispatch to every remote worker.
        // Workers use the heartbeat to detect driver failure / network
        // partition and exit cleanly rather than relying on a fixed
        // wall-clock receiver timeout.
        //
        // Idempotent: callable from any thread; second call without StopHeartbeat
        // is a no-op. StopHeartbeat joins the thread; destructor calls it.

        void StartHeartbeat(int intervalSec) {
            if (intervalSec <= 0) return;
            if (m_heartbeatThread.joinable()) return;
            m_heartbeatStop.store(false);
            m_heartbeatThread = std::thread([this, intervalSec]() {
                std::uint32_t round = 0;
                while (!m_heartbeatStop.load()) {
                    BroadcastDispatchCommand(DispatchCommand::Type::Heartbeat, round++);
                    for (int i = 0; i < intervalSec * 10 && !m_heartbeatStop.load(); i++) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
            });
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatcherNode: heartbeat pump started (interval=%ds)\n", intervalSec);
        }

        void StopHeartbeat() {
            if (!m_heartbeatThread.joinable()) return;
            m_heartbeatStop.store(true);
            m_heartbeatThread.join();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatcherNode: heartbeat pump stopped\n");
        }

        ~DispatcherNode() {
            StopHeartbeat();
        }

        // ---- Ring management ----

        bool AllWorkersAcked() const {
            std::uint32_t currentVer = m_currentRingVersion.load();
            if (currentVer == 0) return false;
            std::lock_guard<std::mutex> lock(m_ackMutex);
            int numNodes = static_cast<int>(m_nodeAddrs.size());
            for (int i = 0; i < numNodes; i++) {
                if (i == m_localNodeIndex) continue;
                auto it = m_workerAckedVersion.find(i);
                if (it == m_workerAckedVersion.end() || it->second < currentVer) return false;
            }
            return true;
        }

    protected:
        void RegisterServerHandlers(Socket::PacketHandlerMapPtr& handlers) override {
            handlers->emplace(Socket::PacketType::NodeRegisterRequest,
                [this](Socket::ConnectionID c, Socket::Packet p) { HandleNodeRegisterRequest(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::RingUpdateACK,
                [this](Socket::ConnectionID c, Socket::Packet p) { HandleRingUpdateACK(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::DispatchCommand,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_dispatch.HandleDispatchCommand(c, std::move(p)); });
            handlers->emplace(Socket::PacketType::DispatchResult,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_dispatch.HandleDispatchResult(c, std::move(p)); });
        }

        void RegisterClientHandlers(Socket::PacketHandlerMapPtr& handlers) override {
            handlers->emplace(Socket::PacketType::DispatchResult,
                [this](Socket::ConnectionID c, Socket::Packet p) { m_dispatch.HandleDispatchResult(c, std::move(p)); });
        }

        void BgProtocolStep() override {
            if (m_currentRingVersion.load() > 0) {
                RetryUnackedRingUpdates();
            }
        }

        bool IsRingSettled() const override {
            return AllWorkersAcked();
        }

    private:
        void HandleNodeRegisterRequest(Socket::ConnectionID connID, Socket::Packet packet) {
            NodeRegisterMsg msg;
            if (!msg.Read(packet.Body())) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DispatcherNode: Failed to parse NodeRegisterRequest\n");
                return;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatcherNode: NodeRegister from node %d (%s:%s, store=%s)\n",
                msg.m_nodeIndex, msg.m_host.c_str(), msg.m_port.c_str(), msg.m_store.c_str());

            // Ring is pre-built at startup, just broadcast current ring to the new connection
            BroadcastRingUpdate();
        }

        void HandleRingUpdateACK(Socket::ConnectionID connID, Socket::Packet packet) {
            RingUpdateACKMsg msg;
            if (!msg.Read(packet.Body())) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DispatcherNode: Failed to parse RingUpdateACK\n");
                return;
            }
            {
                std::lock_guard<std::mutex> lock(m_ackMutex);
                auto& ver = m_workerAckedVersion[msg.m_nodeIndex];
                if (msg.m_ringVersion > ver) ver = msg.m_ringVersion;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatcherNode: RingUpdateACK from node %d (v%u)\n",
                msg.m_nodeIndex, msg.m_ringVersion);
        }

        void BroadcastRingUpdate() {
            auto ring = std::atomic_load(&m_hashRing);
            if (!ring) return;

            std::uint32_t version = m_currentRingVersion.load();
            RingUpdateMsg msg;
            msg.m_ringVersion = version;
            msg.m_vnodeCount = ring->GetVNodeCount();
            for (int idx : ring->GetNodes()) {
                msg.m_nodeIndices.push_back(idx);
            }

            std::size_t bodySize = msg.EstimateBufferSize();
            int numNodes = static_cast<int>(m_nodeAddrs.size());

            for (int i = 0; i < numNodes; i++) {
                if (i == m_localNodeIndex) continue;
                auto peerConn = GetPeerConnection(i);
                if (peerConn == Socket::c_invalidConnectionID) continue;

                Socket::Packet pkt;
                pkt.Header().m_packetType = Socket::PacketType::RingUpdate;
                pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
                pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
                pkt.Header().m_resourceID = 0;
                pkt.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
                pkt.AllocateBuffer(static_cast<std::uint32_t>(bodySize));
                msg.Write(pkt.Body());
                pkt.Header().WriteBuffer(pkt.HeaderBuffer());

                m_client->SendPacket(peerConn, std::move(pkt), nullptr);
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatcherNode: Broadcast RingUpdate v%u (%d nodes)\n",
                version, (int)msg.m_nodeIndices.size());
        }

        void RetryUnackedRingUpdates() {
            auto ring = std::atomic_load(&m_hashRing);
            if (!ring) return;
            std::uint32_t currentVer = m_currentRingVersion.load();
            if (currentVer == 0) return;

            std::vector<int> unacked;
            {
                std::lock_guard<std::mutex> lock(m_ackMutex);
                int numNodes = static_cast<int>(m_nodeAddrs.size());
                for (int i = 0; i < numNodes; i++) {
                    if (i == m_localNodeIndex) continue;
                    auto it = m_workerAckedVersion.find(i);
                    if (it == m_workerAckedVersion.end() || it->second < currentVer)
                        unacked.push_back(i);
                }
            }
            if (unacked.empty()) return;

            RingUpdateMsg msg;
            msg.m_ringVersion = currentVer;
            msg.m_vnodeCount = ring->GetVNodeCount();
            for (int idx : ring->GetNodes()) msg.m_nodeIndices.push_back(idx);
            std::size_t bodySize = msg.EstimateBufferSize();

            for (int nodeIdx : unacked) {
                auto peerConn = GetPeerConnection(nodeIdx);
                if (peerConn == Socket::c_invalidConnectionID) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "DispatcherNode: RetryUnackedRingUpdates skip node %d (no peer conn)\n", nodeIdx);
                    continue;
                }

                Socket::Packet pkt;
                pkt.Header().m_packetType = Socket::PacketType::RingUpdate;
                pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
                pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
                pkt.Header().m_resourceID = 0;
                pkt.Header().m_bodyLength = static_cast<std::uint32_t>(bodySize);
                pkt.AllocateBuffer(static_cast<std::uint32_t>(bodySize));
                msg.Write(pkt.Body());
                pkt.Header().WriteBuffer(pkt.HeaderBuffer());

                m_client->SendPacket(peerConn, std::move(pkt), nullptr);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "DispatcherNode: Retried RingUpdate to node %d (connID=%u)\n", nodeIdx, peerConn);
            }
        }

        DispatchCoordinator m_dispatch;
        std::atomic<std::uint32_t> m_currentRingVersion{0};
        mutable std::mutex m_ackMutex;
        std::unordered_map<int, std::uint32_t> m_workerAckedVersion;

        std::thread m_heartbeatThread;
        std::atomic<bool> m_heartbeatStop{false};
    };

} // namespace SPTAG::SPANN
