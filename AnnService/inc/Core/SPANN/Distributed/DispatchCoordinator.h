// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/SPANN/Distributed/DistributedProtocol.h"
#include "inc/Socket/Client.h"
#include "inc/Socket/Packet.h"
#include "inc/Socket/SimpleSerialization.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace SPTAG::SPANN {

    /// Coordinates driver↔worker dispatch for distributed benchmarks.
    ///
    /// The driver broadcasts Insert/Search/Stop commands to all workers and
    /// collects their results.  Workers execute commands via a callback and
    /// report results back.
    ///
    /// This class is independent of posting routing — it only needs a way to
    /// send packets to peer nodes (provided via PeerNetwork interface).
    class DispatchCoordinator {
    public:
        /// Abstract interface for sending packets to peer nodes.
        /// NetworkNode implements this so DispatchCoordinator doesn't
        /// depend on the full node class.
        class PeerNetwork {
        public:
            virtual ~PeerNetwork() = default;
            /// Get connection to a peer node (reconnecting if needed).
            virtual Socket::ConnectionID GetPeerConnection(int nodeIndex) = 0;
            /// Total number of nodes in the cluster.
            virtual int GetNumNodes() const = 0;
            /// Index of this node.
            virtual int GetLocalNodeIndex() const = 0;
            /// Send a packet via the client socket.
            virtual void SendPacket(Socket::ConnectionID connID, Socket::Packet&& pkt,
                                    std::function<void(bool)> callback) = 0;
        };

        using DispatchCallback = std::function<DispatchResult(const DispatchCommand&)>;

        DispatchCoordinator() = default;

        ~DispatchCoordinator() {
            ClearDispatchCallback();
        }

        /// Attach to a peer network (must outlive this coordinator).
        void SetNetwork(PeerNetwork* network) {
            m_network = network;
        }

        /// Mark a worker node as "local" — its work is done inline by the
        /// driver so it should be skipped during broadcast/result collection.
        void SetLocalWorkerIndex(int idx) { m_localWorkerIndex = idx; }

        /// Set the callback for executing dispatch commands (worker side).
        void SetDispatchCallback(DispatchCallback cb) {
            m_dispatchCallback = std::move(cb);
        }

        /// Clear the dispatch callback and wait for in-flight dispatch
        /// threads to complete. Call before destroying callback state.
        void ClearDispatchCallback() {
            m_dispatchCallback = nullptr;
            std::unique_lock<std::mutex> lock(m_activeDispatchMutex);
            m_activeDispatchCV.wait(lock, [this]() {
                return m_activeDispatchCount == 0;
            });
        }

        // ---- Driver side ----

        /// Broadcast a dispatch command to all worker nodes.
        /// Returns the dispatchId assigned to this command.
        std::uint64_t BroadcastDispatchCommand(DispatchCommand::Type type, std::uint32_t round) {
            std::uint64_t dispatchId = m_nextDispatchId.fetch_add(1);

            DispatchCommand cmd;
            cmd.m_type = type;
            cmd.m_dispatchId = dispatchId;
            cmd.m_round = round;

            int numNodes = m_network->GetNumNodes();
            int localIdx = m_network->GetLocalNodeIndex();

            // Build list of nodes to skip (dispatcher + local worker if set)
            auto shouldSkip = [&](int i) {
                return i == localIdx || i == m_localWorkerIndex;
            };

            // Count remote workers (nodes we will actually dispatch to)
            int remoteWorkers = 0;
            for (int i = 0; i < numNodes; i++) {
                if (!shouldSkip(i)) remoteWorkers++;
            }

            // Set up pending state for collecting results (not for Stop / Heartbeat)
            if (type != DispatchCommand::Type::Stop &&
                type != DispatchCommand::Type::Heartbeat &&
                remoteWorkers > 0) {
                auto state = std::make_shared<PendingDispatch>();
                state->remaining.store(remoteWorkers);
                for (int i = 0; i < numNodes; i++) {
                    if (!shouldSkip(i)) state->pendingNodes.insert(i);
                }
                {
                    std::lock_guard<std::mutex> lock(m_dispatchMutex);
                    m_pendingDispatches[dispatchId] = state;
                }
            }

            auto bodySize = static_cast<std::uint32_t>(cmd.EstimateBufferSize());

            for (int i = 0; i < numNodes; i++) {
                if (shouldSkip(i)) continue;

                Socket::ConnectionID connID = m_network->GetPeerConnection(i);
                if (connID == Socket::c_invalidConnectionID) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "DispatchCoordinator: Cannot dispatch to node %d (no connection)\n", i);
                    if (type != DispatchCommand::Type::Stop &&
                        type != DispatchCommand::Type::Heartbeat) {
                        std::lock_guard<std::mutex> lock(m_dispatchMutex);
                        auto it = m_pendingDispatches.find(dispatchId);
                        if (it != m_pendingDispatches.end()) {
                            it->second->errors++;
                            if (it->second->remaining.fetch_sub(1) == 1) {
                                it->second->done.set_value();
                            }
                        }
                    }
                    continue;
                }

                Socket::Packet pkt;
                pkt.Header().m_packetType = Socket::PacketType::DispatchCommand;
                pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
                pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
                pkt.Header().m_resourceID = 0;
                pkt.Header().m_bodyLength = bodySize;
                pkt.AllocateBuffer(bodySize);
                cmd.Write(pkt.Body());
                pkt.Header().WriteBuffer(pkt.HeaderBuffer());

                m_network->SendPacket(connID, std::move(pkt), nullptr);
            }

            // Heartbeats fire every interval seconds — keep logs clean.
            if (type != DispatchCommand::Type::Heartbeat) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "DispatchCoordinator: Dispatched %s (id=%llu round=%u) to %d workers\n",
                    type == DispatchCommand::Type::Search ? "Search" :
                    type == DispatchCommand::Type::Insert ? "Insert" : "Stop",
                    (unsigned long long)dispatchId, round, remoteWorkers);
            }

            return dispatchId;
        }

        /// Wait for all workers to report results for a dispatch.
        /// Returns collected wall times from workers. Empty on timeout.
        std::vector<double> WaitForAllResults(std::uint64_t dispatchId, int timeoutSec = 300) {
            std::shared_ptr<PendingDispatch> state;
            {
                std::lock_guard<std::mutex> lock(m_dispatchMutex);
                auto it = m_pendingDispatches.find(dispatchId);
                if (it == m_pendingDispatches.end()) return {};
                state = it->second;
            }

            auto future = state->done.get_future();
            auto status = future.wait_for(std::chrono::seconds(timeoutSec));

            {
                std::lock_guard<std::mutex> lock(m_dispatchMutex);
                m_pendingDispatches.erase(dispatchId);
            }

            if (status == std::future_status::timeout) {
                std::string nodeList;
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    for (int n : state->pendingNodes) {
                        if (!nodeList.empty()) nodeList += ",";
                        nodeList += std::to_string(n);
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DispatchCoordinator: Timeout waiting for results (id=%llu, %d remaining, nodes=[%s])\n",
                    (unsigned long long)dispatchId, state->remaining.load(), nodeList.c_str());
                return {};
            }

            if (state->errors > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "DispatchCoordinator: Dispatch %llu completed with %d errors\n",
                    (unsigned long long)dispatchId, (int)state->errors);
            }

            std::lock_guard<std::mutex> lock(state->mutex);
            return state->wallTimes;
        }

        // ---- Worker side ----

        /// Send a dispatch result back to the driver (worker side).
        void SendDispatchResult(const DispatchResult& result) {
            int driverNode = 0;
            if (driverNode == m_network->GetLocalNodeIndex()) return;

            Socket::ConnectionID connID = m_network->GetPeerConnection(driverNode);
            if (connID == Socket::c_invalidConnectionID) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DispatchCoordinator: Cannot send result to driver\n");
                return;
            }

            Socket::Packet pkt;
            auto bodySize = static_cast<std::uint32_t>(result.EstimateBufferSize());
            pkt.Header().m_packetType = Socket::PacketType::DispatchResult;
            pkt.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
            pkt.Header().m_connectionID = Socket::c_invalidConnectionID;
            pkt.Header().m_resourceID = 0;
            pkt.Header().m_bodyLength = bodySize;
            pkt.AllocateBuffer(bodySize);
            result.Write(pkt.Body());
            pkt.Header().WriteBuffer(pkt.HeaderBuffer());

            m_network->SendPacket(connID, std::move(pkt), nullptr);
        }

        // ---- Packet handlers (called by NetworkNode's server/client) ----

        /// Handle an incoming dispatch command from the driver (worker side).
        void HandleDispatchCommand(Socket::ConnectionID connID, Socket::Packet packet) {
            if (packet.Header().m_bodyLength == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DispatchCoordinator: Empty DispatchCommand received\n");
                return;
            }

            DispatchCommand cmd;
            if (cmd.Read(packet.Body()) == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DispatchCoordinator: DispatchCommand parse failed\n");
                return;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatchCoordinator: Received command type=%d id=%llu round=%u\n",
                (int)cmd.m_type, (unsigned long long)cmd.m_dispatchId, cmd.m_round);

            auto callback = m_dispatchCallback;
            if (!callback) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "DispatchCoordinator: No callback set, ignoring command\n");
                return;
            }

            {
                std::lock_guard<std::mutex> lock(m_activeDispatchMutex);
                m_activeDispatchCount++;
            }

            auto self = this;
            int localIdx = m_network->GetLocalNodeIndex();
            std::thread([self, callback, cmd, localIdx]() {
                DispatchResult result = callback(cmd);
                result.m_nodeIndex = localIdx;
                result.m_dispatchId = cmd.m_dispatchId;
                result.m_round = cmd.m_round;

                if (cmd.m_type != DispatchCommand::Type::Stop &&
                    cmd.m_type != DispatchCommand::Type::Heartbeat) {
                    self->SendDispatchResult(result);
                }

                {
                    std::lock_guard<std::mutex> lock(self->m_activeDispatchMutex);
                    self->m_activeDispatchCount--;
                }
                self->m_activeDispatchCV.notify_all();
            }).detach();
        }

        /// Handle an incoming dispatch result from a worker (driver side).
        void HandleDispatchResult(Socket::ConnectionID connID, Socket::Packet packet) {
            if (packet.Header().m_bodyLength == 0) return;

            DispatchResult result;
            if (result.Read(packet.Body()) == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "DispatchCoordinator: DispatchResult parse failed\n");
                return;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DispatchCoordinator: Result id=%llu round=%u node=%d status=%d wallTime=%.3f\n",
                (unsigned long long)result.m_dispatchId, result.m_round,
                result.m_nodeIndex, (int)result.m_status, result.m_wallTime);

            std::shared_ptr<PendingDispatch> state;
            {
                std::lock_guard<std::mutex> lock(m_dispatchMutex);
                auto it = m_pendingDispatches.find(result.m_dispatchId);
                if (it == m_pendingDispatches.end()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "DispatchCoordinator: Result for unknown dispatch %llu (late/expired)\n",
                        (unsigned long long)result.m_dispatchId);
                    return;
                }
                state = it->second;
            }

            if (result.m_status != DispatchResult::Status::Success) {
                state->errors++;
            }

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->wallTimes.push_back(result.m_wallTime);
                if (result.m_nodeIndex >= 0)
                    state->pendingNodes.erase(result.m_nodeIndex);
            }

            if (state->remaining.fetch_sub(1) == 1) {
                state->done.set_value();
            }
        }

    private:
        struct PendingDispatch {
            std::atomic<int> remaining{0};
            std::atomic<int> errors{0};
            std::promise<void> done;
            std::mutex mutex;
            std::vector<double> wallTimes;
            std::set<int> pendingNodes;  // nodes that haven't responded yet
        };

        PeerNetwork* m_network = nullptr;
        int m_localWorkerIndex = -1;  // driver's worker node to skip in broadcasts
        DispatchCallback m_dispatchCallback;
        std::atomic<std::uint64_t> m_nextDispatchId{1};
        std::mutex m_dispatchMutex;
        std::unordered_map<std::uint64_t, std::shared_ptr<PendingDispatch>> m_pendingDispatches;

        std::mutex m_activeDispatchMutex;
        std::condition_variable m_activeDispatchCV;
        int m_activeDispatchCount{0};
    };

} // namespace SPTAG::SPANN
