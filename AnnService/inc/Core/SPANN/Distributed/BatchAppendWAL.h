// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_DISTRIBUTED_BATCHAPPENDWAL_H_
#define _SPTAG_SPANN_DISTRIBUTED_BATCHAPPENDWAL_H_

#include "inc/Core/Common.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/Logging.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace SPTAG {
namespace SPANN {
namespace Distributed {

// BatchAppendWAL: durable write-ahead log for accepted BatchRemoteAppend
// requests on the receiver side.
//
// Sender → Receiver flow with this WAL enabled:
//   1. Receiver decodes a BatchRemoteAppendRequest.
//   2. Receiver serializes the request blob and Put()s it under
//        wal/rappend/<receiverNode>/<batchID>.
//   3. Receiver ACKs the sender immediately ("Accepted").
//   4. Receiver schedules the per-item Append jobs as before.
//   5. After every item in the batch has been processed, the receiver
//      Delete()s the WAL key (best-effort).
//
// Recovery: at startup (after SetWorker has wired the searcher's
// append-callback and job submitter) the receiver scans
// `wal/rappend/<receiverNode>/` and re-submits each pending batch.
// Items are idempotent — the Append callback checks the versionMap and
// skips RMWs that are already at the recorded version, so duplicate
// replays after a crash do not corrupt postings.
//
// Key schema:
//   wal/rappend/<receiverNode>/<batchID>  →  raw BatchRemoteAppendRequest bytes
class BatchAppendWAL {
public:
    explicit BatchAppendWAL(std::shared_ptr<Helper::KeyValueIO> db, int receiverNode)
        : m_db(std::move(db)), m_receiverNode(receiverNode) {}

    bool Enabled() const { return static_cast<bool>(m_db); }

    bool Put(std::uint64_t batchID, const std::string& blob) {
        if (!m_db) return false;
        auto ec = m_db->Put(MakeKey(m_receiverNode, batchID), blob, kTimeout, nullptr);
        if (ec != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "BatchAppendWAL::Put node=%d batchID=%llu failed (%d)\n",
                m_receiverNode, (unsigned long long)batchID, (int)ec);
            return false;
        }
        return true;
    }

    bool Delete(std::uint64_t batchID) {
        if (!m_db) return false;
        std::vector<std::string> k{ MakeKey(m_receiverNode, batchID) };
        auto ec = m_db->MultiDelete(k, kTimeout);
        if (ec != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                "BatchAppendWAL::Delete node=%d batchID=%llu failed (%d) — recovery will replay\n",
                m_receiverNode, (unsigned long long)batchID, (int)ec);
            return false;
        }
        return true;
    }

    // Returns all (batchID, blob) pairs currently durable for this receiver.
    ErrorCode Scan(std::vector<std::pair<std::uint64_t, std::string>>& out) {
        out.clear();
        if (!m_db) return ErrorCode::Undefined;
        std::vector<std::pair<std::string, std::string>> kvs;
        std::string prefix = MakePrefix(m_receiverNode);
        auto ec = m_db->ScanPrefix(prefix, kvs, 0);
        if (ec == ErrorCode::Undefined) {
            // Backend without ScanPrefix support — no recovery, but logged
            // so operators see the gap.
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                "BatchAppendWAL::Scan: backend has no ScanPrefix; recovery skipped\n");
            return ec;
        }
        if (ec != ErrorCode::Success) return ec;
        for (auto& kv : kvs) {
            // kv.first looks like "wal/rappend/<node>/<batchID>"
            auto pos = kv.first.find_last_of('/');
            if (pos == std::string::npos) continue;
            std::uint64_t batchID = 0;
            try { batchID = std::stoull(kv.first.substr(pos + 1)); }
            catch (...) { continue; }
            out.emplace_back(batchID, std::move(kv.second));
        }
        return ErrorCode::Success;
    }

    static std::string MakePrefix(int receiverNode) {
        return "wal/rappend/" + std::to_string(receiverNode) + "/";
    }
    static std::string MakeKey(int receiverNode, std::uint64_t batchID) {
        return MakePrefix(receiverNode) + std::to_string(batchID);
    }

private:
    static constexpr auto kTimeout = std::chrono::microseconds(5'000'000);
    std::shared_ptr<Helper::KeyValueIO> m_db;
    int m_receiverNode = -1;
};

} // namespace Distributed
} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_DISTRIBUTED_BATCHAPPENDWAL_H_
