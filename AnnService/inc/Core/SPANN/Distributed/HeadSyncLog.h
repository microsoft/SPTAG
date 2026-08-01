// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_DISTRIBUTED_HEADSYNCLOG_H_
#define _SPTAG_SPANN_DISTRIBUTED_HEADSYNCLOG_H_

#include "inc/Core/Common.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/Logging.h"
#include "inc/Core/SPANN/Distributed/DistributedProtocol.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace SPTAG {
namespace SPANN {
namespace Distributed {

// HeadSyncLog: durable per-shard log of HeadSync entries in TiKV.
//
// Per the distributed design, the canonical source of truth for head
// topology changes is TiKV, not the in-memory broadcast.  Each shard
// (today: per owner node) holds:
//   * `hs/v/<shard>`           little-endian uint64 latest version
//   * `hs/e/<shard>/<verBE>`   serialized HeadSyncEntry payload
//   * `hs/c/<node>/<shard>`    little-endian uint64 applied version
//
// Versions are monotonically increasing per shard.  Producers serialize
// their version-bump under `m_appendMutex` and write entry-then-version;
// readers tolerate a transient lag where version points slightly past
// the last entry (treat the missing entry as not-yet-visible and retry).
// TiKV's raw KV API gives no multi-key atomicity; the design (per user
// direction) accepts this and relies on idempotent apply + cursor
// catch-up to converge.
//
// This header is intentionally self-contained; it does not depend on
// any SPANN searcher type.  ExtraDynamicSearcher wires it up by
// constructing one instance per layer-0 ExtraDynamicSearcher, calling
// Append() in BroadcastHeadSync, and supplying an ApplyFn callback for
// the reconciler.
class HeadSyncLog {
public:
    // Decoded entry returned by ReadSince. Carries the version so the
    // reconciler can advance its cursor strictly past it on success.
    struct VersionedEntry {
        std::uint64_t version;
        HeadSyncEntry entry;
    };

    using ApplyFn = std::function<bool(const VersionedEntry&)>;

    HeadSyncLog(std::shared_ptr<Helper::KeyValueIO> db,
                int nodeIndex,
                int reconcileIntervalMs = 2000)
        : m_db(std::move(db)),
          m_nodeIndex(nodeIndex),
          m_reconcileIntervalMs(reconcileIntervalMs),
          m_stop(false) {}

    ~HeadSyncLog() { Stop(); }

    // Append a batch of entries to the given shard's log.  Returns the
    // version of the last written entry (>= 1 on success, 0 on failure).
    std::uint64_t Append(int shard, std::vector<HeadSyncEntry>& entries) {
        if (!m_db || entries.empty()) return 0;
        std::lock_guard<std::mutex> lk(GetShardAppendMutex(shard));
        std::uint64_t base = LoadLatestVersion(shard);
        std::vector<std::string> keys;
        std::vector<std::string> values;
        keys.reserve(entries.size());
        values.reserve(entries.size());
        std::uint64_t v = base;
        for (auto& e : entries) {
            ++v;
            e.m_shard = shard;
            keys.push_back(MakeEntryKey(shard, v));
            values.push_back(EncodeEntry(e));
        }
        auto ec = m_db->MultiPut(keys, values, kTimeout, nullptr);
        if (ec != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "HeadSyncLog::Append shard=%d entries=%zu MultiPut failed (%d)\n",
                shard, entries.size(), (int)ec);
            return 0;
        }
        ec = m_db->Put(MakeVersionKey(shard),
                       EncodeUint64(v),
                       kTimeout, nullptr);
        if (ec != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "HeadSyncLog::Append shard=%d version Put failed (%d), entries durable but version lag\n",
                shard, (int)ec);
            // Entries are durable; the next Append (or reconciler in
            // another node) will discover them via probe.
            return v;
        }
        return v;
    }

    // Read latest version that the shard publisher has advanced to.
    // Returns 0 if no version is published yet or on read failure.
    std::uint64_t GetLatestVersion(int shard) const { return LoadLatestVersion(shard); }

    // Read entries (cursor, latest], one at a time. Stops at the first
    // missing version (which indicates writer lag).
    std::vector<VersionedEntry> ReadSince(int shard,
                                          std::uint64_t cursor,
                                          std::uint64_t latest,
                                          size_t maxBatch = 256) const {
        std::vector<VersionedEntry> out;
        if (!m_db || cursor >= latest) return out;
        size_t want = std::min<size_t>(maxBatch,
            static_cast<size_t>(latest - cursor));
        std::vector<std::string> keys;
        keys.reserve(want);
        for (size_t i = 0; i < want; ++i) {
            keys.push_back(MakeEntryKey(shard, cursor + 1 + i));
        }
        std::vector<std::string> values;
        auto ec = m_db->MultiGet(keys, &values, kTimeout, nullptr);
        if (ec != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                "HeadSyncLog::ReadSince shard=%d MultiGet failed (%d)\n",
                shard, (int)ec);
            return out;
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i].empty()) break; // writer lag; stop here
            VersionedEntry ve;
            ve.version = cursor + 1 + i;
            if (!DecodeEntry(values[i], ve.entry)) break;
            out.push_back(std::move(ve));
        }
        return out;
    }

    // Cursor I/O for a (node, shard) pair.
    std::uint64_t LoadCursor(int shard) const {
        if (!m_db) return 0;
        std::string out;
        auto ec = m_db->Get(MakeCursorKey(m_nodeIndex, shard), &out, kTimeout, nullptr);
        if (ec != ErrorCode::Success || out.size() < sizeof(std::uint64_t)) return 0;
        return DecodeUint64(out);
    }

    bool StoreCursor(int shard, std::uint64_t version) {
        if (!m_db) return false;
        auto ec = m_db->Put(MakeCursorKey(m_nodeIndex, shard),
                            EncodeUint64(version),
                            kTimeout, nullptr);
        return ec == ErrorCode::Success;
    }

    // Start a background reconciler that wakes every interval and, for
    // each known shard, fetches missing entries since the local cursor
    // and feeds them to `apply`. `apply` must be idempotent.
    void StartReconciler(std::vector<int> shards, ApplyFn apply) {
        if (m_reconciler.joinable()) return;
        m_shards = std::move(shards);
        m_apply = std::move(apply);
        m_stop = false;
        m_reconciler = std::thread([this]() { ReconcileLoop(); });
    }

    void ReconcileShard(int shard) {
        if (!m_apply || shard < 0) return;
        std::lock_guard<std::mutex> lk(GetShardReconcileMutex(shard));
        std::uint64_t cursor = LoadCursor(shard);
        while (true) {
            std::uint64_t latest = LoadLatestVersion(shard);
            if (latest <= cursor) return;
            auto entries = ReadSince(shard, cursor, latest);
            if (entries.empty()) return;
            std::uint64_t advanced = cursor;
            for (const auto& ve : entries) {
                if (!m_apply(ve)) break;
                advanced = ve.version;
            }
            if (advanced == cursor) return;
            if (!StoreCursor(shard, advanced)) return;
            cursor = advanced;
        }
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lk(m_cvMutex);
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_reconciler.joinable()) m_reconciler.join();
    }

private:
    static constexpr auto kTimeout = std::chrono::microseconds(2'000'000);

    static std::string EncodeUint64(std::uint64_t v) {
        std::string s(sizeof(v), '\0');
        memcpy(&s[0], &v, sizeof(v));
        return s;
    }
    static std::uint64_t DecodeUint64(const std::string& s) {
        std::uint64_t v = 0;
        if (s.size() >= sizeof(v)) memcpy(&v, s.data(), sizeof(v));
        return v;
    }
    static std::string MakeVersionKey(int shard) {
        return "hs/v/" + std::to_string(shard);
    }
    static std::string MakeEntryKey(int shard, std::uint64_t version) {
        // Big-endian version so byte-range scans (if added later) are
        // monotonically sorted.
        std::string s = "hs/e/" + std::to_string(shard) + "/";
        char be[8];
        for (int i = 0; i < 8; ++i) be[i] = static_cast<char>((version >> ((7 - i) * 8)) & 0xff);
        s.append(be, 8);
        return s;
    }
    static std::string MakeCursorKey(int node, int shard) {
        return "hs/c/" + std::to_string(node) + "/" + std::to_string(shard);
    }

    static std::string EncodeEntry(const HeadSyncEntry& e) {
        std::string s(e.EstimateBufferSize(), '\0');
        std::uint8_t* end = e.Write(reinterpret_cast<std::uint8_t*>(&s[0]));
        s.resize(static_cast<size_t>(end - reinterpret_cast<std::uint8_t*>(&s[0])));
        return s;
    }
    static bool DecodeEntry(const std::string& s, HeadSyncEntry& e) {
        if (s.empty()) return false;
        e.Read(reinterpret_cast<const std::uint8_t*>(s.data()));
        return true;
    }

    std::uint64_t LoadLatestVersion(int shard) const {
        std::string out;
        auto ec = m_db->Get(MakeVersionKey(shard), &out, kTimeout, nullptr);
        if (ec != ErrorCode::Success) return 0;
        return DecodeUint64(out);
    }

    std::mutex& GetShardAppendMutex(int shard) {
        std::lock_guard<std::mutex> lk(m_appendMutexMapLock);
        auto& slot = m_appendMutexes[shard];
        if (!slot) slot = std::make_unique<std::mutex>();
        return *slot;
    }

    std::mutex& GetShardReconcileMutex(int shard) {
        std::lock_guard<std::mutex> lk(m_reconcileMutexMapLock);
        auto& slot = m_reconcileMutexes[shard];
        if (!slot) slot = std::make_unique<std::mutex>();
        return *slot;
    }

    void ReconcileLoop() {
        std::unique_lock<std::mutex> lk(m_cvMutex);
        while (!m_stop) {
            lk.unlock();
            for (int shard : m_shards) {
                ReconcileShard(shard);
            }
            lk.lock();
            m_cv.wait_for(lk, std::chrono::milliseconds(m_reconcileIntervalMs),
                          [this]() { return m_stop; });
        }
    }

    std::shared_ptr<Helper::KeyValueIO> m_db;
    int m_nodeIndex;
    int m_reconcileIntervalMs;

    std::mutex m_appendMutexMapLock;
    std::unordered_map<int, std::unique_ptr<std::mutex>> m_appendMutexes;
    std::mutex m_reconcileMutexMapLock;
    std::unordered_map<int, std::unique_ptr<std::mutex>> m_reconcileMutexes;

    std::vector<int> m_shards;
    ApplyFn m_apply;

    mutable std::mutex m_cvMutex;
    std::condition_variable m_cv;
    bool m_stop;
    std::thread m_reconciler;
};

} // namespace Distributed
} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_DISTRIBUTED_HEADSYNCLOG_H_
