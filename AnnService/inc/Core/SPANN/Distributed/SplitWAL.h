// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_DISTRIBUTED_SPLITWAL_H_
#define _SPTAG_SPANN_DISTRIBUTED_SPLITWAL_H_

#include "inc/Core/Common.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/Logging.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SPTAG {
namespace SPANN {
namespace Distributed {

// SplitWAL: durable write-ahead log entry for a cross-owner split.
//
// Per the distributed design's Split Happy Path, when a split produces
// two child heads owned by different nodes, the split writes the local
// child via PutPostingToDB and the remote child via the remote queue.
// If either write fails after the other succeeded, a WAL-driven GC job
// must clean up the orphan posting under the partner head.
//
// Key schema:
//   wal/split/<headID>/<jobID>  →  encoded SplitWALRecord
// Garbage-collection (background): scan `wal/split/` prefix; if a
// record is older than `kStaleSec` and not marked committed, it
// represents either an in-flight split or a crashed one — issue
// best-effort deletes against both children using the recorded headIDs.
//
// Today this is scaffolding: Begin/Commit hooks should wrap the split's
// cross-owner write path in ExtraDynamicSearcher.  GC sweep can run on
// the existing RefineIndex cadence.
class SplitWAL {
public:
    enum class Stage : std::uint8_t {
        Begin       = 0, // both children allocated, neither written
        LocalDone   = 1, // local write succeeded; remote pending
        RemoteDone  = 2, // remote write succeeded; local pending
        BothDone    = 3, // both written; safe to remove WAL + delete src
    };

    struct Record {
        std::uint64_t jobID;
        SizeType      srcHeadID;
        SizeType      localChildHeadID;
        SizeType      remoteChildHeadID;
        int           remoteOwnerNodeIndex;
        std::int64_t  startTimestampSec;
        Stage         stage;

        std::string Encode() const {
            std::string s(sizeof(Record), '\0');
            memcpy(&s[0], this, sizeof(Record));
            return s;
        }
        bool Decode(const std::string& s) {
            if (s.size() < sizeof(Record)) return false;
            memcpy(this, s.data(), sizeof(Record));
            return true;
        }
    };

    explicit SplitWAL(std::shared_ptr<Helper::KeyValueIO> db, int layer = 0)
        : m_db(std::move(db)), m_layer(layer) {}

    // Write or update a WAL record. Stage transitions are monotonic.
    bool Write(const Record& r) {
        if (!m_db) return false;
        auto ec = m_db->Put(MakeKey(m_layer, r.srcHeadID, r.jobID), r.Encode(), kTimeout, nullptr);
        if (ec != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "SplitWAL::Write head=%lld job=%llu stage=%u failed (%d)\n",
                (long long)r.srcHeadID, (unsigned long long)r.jobID,
                (unsigned)r.stage, (int)ec);
            return false;
        }
        return true;
    }

    // Remove a completed WAL record after both writes succeeded.
    bool Clear(SizeType srcHeadID, std::uint64_t jobID) {
        if (!m_db) return false;
        std::vector<std::string> k{ MakeKey(m_layer, srcHeadID, jobID) };
        return m_db->MultiDelete(k, kTimeout) == ErrorCode::Success;
    }

    static std::string MakeKey(int layer, SizeType srcHeadID, std::uint64_t jobID) {
        return "wal/split/" + std::to_string(layer) + "/"
            + std::to_string(srcHeadID) + "/" + std::to_string(jobID);
    }

private:
    static constexpr auto kTimeout = std::chrono::microseconds(2'000'000);
    std::shared_ptr<Helper::KeyValueIO> m_db;
    int m_layer = 0;
};

} // namespace Distributed
} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_DISTRIBUTED_SPLITWAL_H_
