// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/Common.h"
#include "inc/Socket/SimpleSerialization.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace SPTAG::SPANN {

    /// Serializable request for remote Append operations sent between compute nodes.
    /// MirrorVersion 1 added m_layer to disambiguate which ExtraDynamicSearcher on
    /// the receiver side handles the request. Version 0 packets default m_layer=0.
    /// MirrorVersion 2 added m_fencingToken: when nonzero the receiver must
    /// validate the token against its RemoteLeaseTable for the head's bucket
    /// before applying.  Token 0 means "no fencing required" (used by the
    /// normal owner-ring auto-route path that does not hold any remote lock).
    struct RemoteAppendRequest {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 2; }

        SizeType m_headID = 0;
        std::string m_headVec;        // raw head vector bytes
        std::int32_t m_appendNum = 0;
        std::string m_appendPosting;  // serialized posting data
        std::int32_t m_layer = 0;     // originating ExtraDynamicSearcher layer
        std::uint64_t m_fencingToken = 0;  // 0 = unfenced (legacy path)

        std::size_t EstimateBufferSize() const {
            std::size_t size = 0;
            size += sizeof(std::uint16_t) * 2;  // version fields
            size += sizeof(SizeType);            // headID
            size += sizeof(std::uint32_t) + m_headVec.size();       // headVec (len-prefixed)
            size += sizeof(std::int32_t);        // appendNum
            size += sizeof(std::uint32_t) + m_appendPosting.size(); // appendPosting (len-prefixed)
            size += sizeof(std::int32_t);        // layer (mirrorVer >= 1)
            size += sizeof(std::uint64_t);       // fencingToken (mirrorVer >= 2)
            return size;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_headID, p_buffer);
            p_buffer = SimpleWriteBuffer(m_headVec, p_buffer);
            p_buffer = SimpleWriteBuffer(m_appendNum, p_buffer);
            p_buffer = SimpleWriteBuffer(m_appendPosting, p_buffer);
            p_buffer = SimpleWriteBuffer(m_layer, p_buffer);
            p_buffer = SimpleWriteBuffer(m_fencingToken, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            return Read(p_buffer, nullptr);
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer, const std::uint8_t* p_bufEnd) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, majorVer);
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, mirrorVer);
            if (p_buffer == nullptr || majorVer != MajorVersion()) return nullptr;
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_headID);
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_headVec);
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_appendNum);
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_appendPosting);
            if (mirrorVer >= 1) {
                p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_layer);
            } else {
                m_layer = 0;
            }
            if (mirrorVer >= 2) {
                p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_fencingToken);
            } else {
                m_fencingToken = 0;
            }
            return p_buffer;
        }
    };

    /// Response for remote Append operations.
    struct RemoteAppendResponse {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        enum class Status : std::uint8_t { Success = 0, Failed = 1 };
        Status m_status = Status::Success;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t);
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_status, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            p_buffer = SimpleReadBuffer(p_buffer, m_status);
            return p_buffer;
        }
    };

    /// Identifies a compute node target for routing decisions.
    struct RouteTarget {
        int nodeIndex = -1;
        bool isLocal = true;
    };

    /// Batch of remote append requests sent to a single node in one round-trip.
    struct BatchRemoteAppendRequest {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        std::uint32_t m_count = 0;
        std::vector<RemoteAppendRequest> m_items;

        std::size_t EstimateBufferSize() const {
            std::size_t size = sizeof(std::uint16_t) * 2;  // version
            size += sizeof(std::uint32_t);  // count
            for (auto& item : m_items) size += item.EstimateBufferSize();
            return size;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_count, p_buffer);
            for (auto& item : m_items) p_buffer = item.Write(p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer, std::uint32_t bodyLength = 0) {
            using namespace Socket::SimpleSerialization;
            const std::uint8_t* bufEnd = (bodyLength > 0) ? (p_buffer + bodyLength) : nullptr;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SafeSimpleReadBuffer(p_buffer, bufEnd, majorVer);
            p_buffer = SafeSimpleReadBuffer(p_buffer, bufEnd, mirrorVer);
            if (p_buffer == nullptr || majorVer != MajorVersion()) {
                m_items.clear();
                return nullptr;
            }
            p_buffer = SafeSimpleReadBuffer(p_buffer, bufEnd, m_count);
            if (p_buffer == nullptr) {
                m_items.clear();
                return nullptr;
            }
            // Reject obviously corrupt counts before allocating
            if (bodyLength > 0 && m_count > bodyLength / 8) {
                m_items.clear();
                return nullptr;
            }
            m_items.resize(m_count);
            for (std::uint32_t i = 0; i < m_count; i++) {
                if (bufEnd && p_buffer >= bufEnd) {
                    m_items.clear();
                    return nullptr;
                }
                p_buffer = m_items[i].Read(p_buffer, bufEnd);
                if (!p_buffer) {
                    m_items.clear();
                    return nullptr;
                }
                if (bufEnd && p_buffer > bufEnd) {
                    m_items.clear();
                    return nullptr;
                }
            }
            return p_buffer;
        }
    };

    /// Response for batch remote append.
    struct BatchRemoteAppendResponse {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        std::uint32_t m_successCount = 0;
        std::uint32_t m_failCount = 0;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::uint32_t) * 2;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_successCount, p_buffer);
            p_buffer = SimpleWriteBuffer(m_failCount, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            p_buffer = SimpleReadBuffer(p_buffer, m_successCount);
            p_buffer = SimpleReadBuffer(p_buffer, m_failCount);
            return p_buffer;
        }
    };

    /// Cross-node merge hint. Search-side trigger on node X observed that
    /// posting `m_headID` (owned by the target node based on consistent-hash
    /// ownership) is below the merge threshold. The receiver enqueues a
    /// local MergeAsync; the local MergePostings logic decides whether the
    /// posting really needs merging at execution time. Fire-and-forget: no
    /// response packet, no retry queue. Multiple notifications for the same
    /// head are dedup'd by m_mergeList on the receiver.
    struct RemoteMergeRequest {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        SizeType m_headID = 0;
        std::int32_t m_layer = 0;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(SizeType) + sizeof(std::int32_t);
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_headID, p_buffer);
            p_buffer = SimpleWriteBuffer(m_layer, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer, const std::uint8_t* p_bufEnd) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, majorVer);
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, mirrorVer);
            if (p_buffer == nullptr || majorVer != MajorVersion()) return nullptr;
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_headID);
            p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, m_layer);
            return p_buffer;
        }
    };

    /// Batch of cross-node merge hints sent to a single owner node in one
    /// fire-and-forget packet. Sender-side dedups by (layer, headID) so
    /// each entry appears at most once per flush window.
    struct BatchRemoteMergeRequest {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        std::uint32_t m_count = 0;
        std::vector<RemoteMergeRequest> m_items;

        std::size_t EstimateBufferSize() const {
            std::size_t size = sizeof(std::uint16_t) * 2;
            size += sizeof(std::uint32_t);
            for (auto& item : m_items) size += item.EstimateBufferSize();
            return size;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_count, p_buffer);
            for (auto& item : m_items) p_buffer = item.Write(p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer, std::uint32_t bodyLength = 0) {
            using namespace Socket::SimpleSerialization;
            const std::uint8_t* bufEnd = (bodyLength > 0) ? (p_buffer + bodyLength) : nullptr;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SafeSimpleReadBuffer(p_buffer, bufEnd, majorVer);
            p_buffer = SafeSimpleReadBuffer(p_buffer, bufEnd, mirrorVer);
            if (p_buffer == nullptr || majorVer != MajorVersion()) {
                m_items.clear();
                return nullptr;
            }
            p_buffer = SafeSimpleReadBuffer(p_buffer, bufEnd, m_count);
            if (p_buffer == nullptr) { m_items.clear(); return nullptr; }
            if (bodyLength > 0 && m_count > bodyLength / 8) {
                m_items.clear();
                return nullptr;
            }
            m_items.resize(m_count);
            for (std::uint32_t i = 0; i < m_count; i++) {
                if (bufEnd && p_buffer >= bufEnd) { m_items.clear(); return nullptr; }
                p_buffer = m_items[i].Read(p_buffer, bufEnd);
                if (!p_buffer) { m_items.clear(); return nullptr; }
                if (bufEnd && p_buffer > bufEnd) { m_items.clear(); return nullptr; }
            }
            return p_buffer;
        }
    };

    /// Entry in a head sync broadcast: one add or delete of a head node.
    /// `m_layer` identifies the originating ExtraDynamicSearcher so the
    /// receiver applies the entry to the matching layer's head index
    /// (with multi-layer SPANN, layer 0 and layer 1 both broadcast head
    /// add/delete; without the layer field every entry would be misrouted
    /// to a single shared callback).
    struct HeadSyncEntry {
        enum class Op : std::uint8_t { Add = 0, Delete = 1 };
        Op op;
        SizeType headVID;
        std::string headVector;       // only for Add; empty for Delete
        std::int32_t m_layer = 0;     // originating ExtraDynamicSearcher layer

        size_t EstimateBufferSize() const {
            return sizeof(std::uint8_t)   // op
                 + sizeof(SizeType)       // headVID
                 + sizeof(std::uint32_t)  // headVector length
                 + headVector.size()
                 + sizeof(std::int32_t);  // layer
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(static_cast<std::uint8_t>(op), p_buffer);
            p_buffer = SimpleWriteBuffer(headVID, p_buffer);
            std::uint32_t vecLen = static_cast<std::uint32_t>(headVector.size());
            p_buffer = SimpleWriteBuffer(vecLen, p_buffer);
            if (vecLen > 0) {
                memcpy(p_buffer, headVector.data(), vecLen);
                p_buffer += vecLen;
            }
            p_buffer = SimpleWriteBuffer(m_layer, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint8_t rawOp = 0;
            p_buffer = SimpleReadBuffer(p_buffer, rawOp);
            op = static_cast<Op>(rawOp);
            p_buffer = SimpleReadBuffer(p_buffer, headVID);
            std::uint32_t vecLen = 0;
            p_buffer = SimpleReadBuffer(p_buffer, vecLen);
            if (vecLen > 0) {
                headVector.assign(reinterpret_cast<const char*>(p_buffer), vecLen);
                p_buffer += vecLen;
            } else {
                headVector.clear();
            }
            p_buffer = SimpleReadBuffer(p_buffer, m_layer);
            return p_buffer;
        }
    };

    /// Dispatch command from driver to workers (replaces file-based barriers).
    struct DispatchCommand {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        enum class Type : std::uint8_t { Search = 0, Insert = 1, Stop = 2, Heartbeat = 3 };
        Type m_type = Type::Search;
        std::uint64_t m_dispatchId = 0;   // unique ID from driver
        std::uint32_t m_round = 0;        // search round or insert batch index

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t)
                 + sizeof(std::uint64_t) + sizeof(std::uint32_t);
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(static_cast<std::uint8_t>(m_type), p_buffer);
            p_buffer = SimpleWriteBuffer(m_dispatchId, p_buffer);
            p_buffer = SimpleWriteBuffer(m_round, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            std::uint8_t rawType = 0;
            p_buffer = SimpleReadBuffer(p_buffer, rawType);
            m_type = static_cast<Type>(rawType);
            p_buffer = SimpleReadBuffer(p_buffer, m_dispatchId);
            p_buffer = SimpleReadBuffer(p_buffer, m_round);
            return p_buffer;
        }
    };

    /// Result from worker back to driver after executing a dispatch command.
    /// MirrorVersion 2 added m_errorCode so failures can carry SPTAG::ErrorCode
    /// detail back to the driver instead of collapsing into a boolean.
    struct DispatchResult {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 2; }

        enum class Status : std::uint8_t { Success = 0, Failed = 1 };
        Status m_status = Status::Success;
        std::uint64_t m_dispatchId = 0;
        std::uint32_t m_round = 0;
        double m_wallTime = 0.0;
        std::int32_t m_nodeIndex = -1;  // which worker sent this result
        // SPTAG::ErrorCode cast to int32 (Success == 0). Populated by the
        // worker's dispatch callback so the driver can distinguish e.g.
        // KeyNotFound from disk-full from network-fail. Older peers (mirror
        // 1) leave this at 0 even when m_status == Failed.
        std::int32_t m_errorCode = 0;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t)
                 + sizeof(std::uint64_t) + sizeof(std::uint32_t) + sizeof(double)
                 + sizeof(std::int32_t) * 2;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(static_cast<std::uint8_t>(m_status), p_buffer);
            p_buffer = SimpleWriteBuffer(m_dispatchId, p_buffer);
            p_buffer = SimpleWriteBuffer(m_round, p_buffer);
            p_buffer = SimpleWriteBuffer(m_wallTime, p_buffer);
            p_buffer = SimpleWriteBuffer(m_nodeIndex, p_buffer);
            p_buffer = SimpleWriteBuffer(m_errorCode, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            std::uint8_t rawStatus = 0;
            p_buffer = SimpleReadBuffer(p_buffer, rawStatus);
            m_status = static_cast<Status>(rawStatus);
            p_buffer = SimpleReadBuffer(p_buffer, m_dispatchId);
            p_buffer = SimpleReadBuffer(p_buffer, m_round);
            p_buffer = SimpleReadBuffer(p_buffer, m_wallTime);
            if (mirrorVer >= 1) {
                p_buffer = SimpleReadBuffer(p_buffer, m_nodeIndex);
            }
            if (mirrorVer >= 2) {
                p_buffer = SimpleReadBuffer(p_buffer, m_errorCode);
            }
            return p_buffer;
        }
    };

    /// Request to lock/unlock a headID on its owner node (for cross-node Merge).
    /// MirrorVersion 1 added m_layer so multi-layer setups dispatch to the
    /// correct lock pool (each ExtraDynamicSearcher owns its own bucket flags).
    /// MirrorVersion 2 added m_token for fencing: Lock requests send token=0;
    /// Unlock requests send the token issued by the prior Lock so a zombie
    /// holder whose lease expired cannot release a lock now held by someone else.
    struct RemoteLockRequest {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 2; }

        enum class Op : std::uint8_t { Lock = 0, Unlock = 1 };
        Op m_op = Op::Lock;
        SizeType m_headID = 0;
        std::int32_t m_layer = 0;
        std::uint64_t m_token = 0;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t)
                 + sizeof(SizeType) + sizeof(std::int32_t) + sizeof(std::uint64_t);
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(static_cast<std::uint8_t>(m_op), p_buffer);
            p_buffer = SimpleWriteBuffer(m_headID, p_buffer);
            p_buffer = SimpleWriteBuffer(m_layer, p_buffer);
            p_buffer = SimpleWriteBuffer(m_token, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            std::uint8_t rawOp = 0;
            p_buffer = SimpleReadBuffer(p_buffer, rawOp);
            m_op = static_cast<Op>(rawOp);
            p_buffer = SimpleReadBuffer(p_buffer, m_headID);
            if (mirrorVer >= 1) {
                p_buffer = SimpleReadBuffer(p_buffer, m_layer);
            } else {
                m_layer = 0;
            }
            if (mirrorVer >= 2) {
                p_buffer = SimpleReadBuffer(p_buffer, m_token);
            } else {
                m_token = 0;
            }
            return p_buffer;
        }
    };

    /// Response for remote lock operations.
    /// MirrorVersion 1 added m_token: the owner returns the issued fencing
    /// token on a successful Lock so the holder can attach it to subsequent
    /// lock-protected operations.  Unlock responses return token=0.
    struct RemoteLockResponse {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 1; }

        enum class Status : std::uint8_t { Granted = 0, Denied = 1 };
        Status m_status = Status::Granted;
        std::uint64_t m_token = 0;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::uint8_t) + sizeof(std::uint64_t);
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(static_cast<std::uint8_t>(m_status), p_buffer);
            p_buffer = SimpleWriteBuffer(m_token, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            std::uint8_t rawOp = 0;
            p_buffer = SimpleReadBuffer(p_buffer, rawOp);
            m_status = static_cast<Status>(rawOp);
            if (mirrorVer >= 1) {
                p_buffer = SimpleReadBuffer(p_buffer, m_token);
            } else {
                m_token = 0;
            }
            return p_buffer;
        }
    };

    /// Worker → dispatcher registration message.
    struct NodeRegisterMsg {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        std::int32_t m_nodeIndex = 0;
        std::string m_host;
        std::string m_port;
        std::string m_store;

        std::size_t EstimateBufferSize() const {
            std::size_t size = 0;
            size += sizeof(std::uint16_t) * 2;
            size += sizeof(std::int32_t);
            size += sizeof(std::uint32_t) + m_host.size();
            size += sizeof(std::uint32_t) + m_port.size();
            size += sizeof(std::uint32_t) + m_store.size();
            return size;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_nodeIndex, p_buffer);
            p_buffer = SimpleWriteBuffer(m_host, p_buffer);
            p_buffer = SimpleWriteBuffer(m_port, p_buffer);
            p_buffer = SimpleWriteBuffer(m_store, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            p_buffer = SimpleReadBuffer(p_buffer, m_nodeIndex);
            p_buffer = SimpleReadBuffer(p_buffer, m_host);
            p_buffer = SimpleReadBuffer(p_buffer, m_port);
            p_buffer = SimpleReadBuffer(p_buffer, m_store);
            return p_buffer;
        }
    };

    /// Dispatcher → worker ring update (full node list, versioned).
    struct RingUpdateMsg {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        std::uint32_t m_ringVersion = 0;
        std::int32_t m_vnodeCount = 150;
        std::vector<std::int32_t> m_nodeIndices;

        std::size_t EstimateBufferSize() const {
            std::size_t size = 0;
            size += sizeof(std::uint16_t) * 2;
            size += sizeof(std::uint32_t);      // ringVersion
            size += sizeof(std::int32_t);       // vnodeCount
            size += sizeof(std::uint32_t);      // numNodes
            size += sizeof(std::int32_t) * m_nodeIndices.size();
            return size;
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_ringVersion, p_buffer);
            p_buffer = SimpleWriteBuffer(m_vnodeCount, p_buffer);
            std::uint32_t count = static_cast<std::uint32_t>(m_nodeIndices.size());
            p_buffer = SimpleWriteBuffer(count, p_buffer);
            for (auto idx : m_nodeIndices) {
                p_buffer = SimpleWriteBuffer(idx, p_buffer);
            }
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            p_buffer = SimpleReadBuffer(p_buffer, m_ringVersion);
            p_buffer = SimpleReadBuffer(p_buffer, m_vnodeCount);
            std::uint32_t count = 0;
            p_buffer = SimpleReadBuffer(p_buffer, count);
            m_nodeIndices.resize(count);
            for (std::uint32_t i = 0; i < count; i++) {
                p_buffer = SimpleReadBuffer(p_buffer, m_nodeIndices[i]);
            }
            return p_buffer;
        }
    };

    /// Worker → dispatcher ACK for a ring update.
    struct RingUpdateACKMsg {
        static constexpr std::uint16_t MajorVersion() { return 1; }
        static constexpr std::uint16_t MirrorVersion() { return 0; }

        std::int32_t m_nodeIndex = -1;
        std::uint32_t m_ringVersion = 0;

        std::size_t EstimateBufferSize() const {
            return sizeof(std::uint16_t) * 2 + sizeof(std::int32_t) + sizeof(std::uint32_t);
        }

        std::uint8_t* Write(std::uint8_t* p_buffer) const {
            using namespace Socket::SimpleSerialization;
            p_buffer = SimpleWriteBuffer(MajorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(MirrorVersion(), p_buffer);
            p_buffer = SimpleWriteBuffer(m_nodeIndex, p_buffer);
            p_buffer = SimpleWriteBuffer(m_ringVersion, p_buffer);
            return p_buffer;
        }

        const std::uint8_t* Read(const std::uint8_t* p_buffer) {
            using namespace Socket::SimpleSerialization;
            std::uint16_t majorVer = 0, mirrorVer = 0;
            p_buffer = SimpleReadBuffer(p_buffer, majorVer);
            p_buffer = SimpleReadBuffer(p_buffer, mirrorVer);
            if (majorVer != MajorVersion()) return nullptr;
            p_buffer = SimpleReadBuffer(p_buffer, m_nodeIndex);
            p_buffer = SimpleReadBuffer(p_buffer, m_ringVersion);
            return p_buffer;
        }
    };

} // namespace SPTAG::SPANN
