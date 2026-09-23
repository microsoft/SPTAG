// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_DISTRIBUTED_ASYNCJOBWATCHDOG_H_
#define _SPTAG_SPANN_DISTRIBUTED_ASYNCJOBWATCHDOG_H_

#include "inc/Helper/Logging.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace SPTAG {
namespace SPANN {
namespace Distributed {

// AsyncJobWatchdog tracks async (fire-and-forget) inter-node dispatches
// and resends them on timeout or transport failure.
//
// Today the only fire-and-forget path is QueueRemoteAppend auto-flush in
// WorkerNode: it ships a batch of RemoteAppendRequests to a peer with no
// synchronous error propagation. Without a watchdog, transient network
// or peer-crash failures silently lose those appends.
//
// The watchdog is intentionally small: callers register a batch with a
// resend callback; the watchdog reschedules the callback up to
// MaxAttempts with exponential backoff. RemoteAppend is idempotent on
// the receive side (HandleRemoteAppend de-dups via per-posting RMW), so
// at-least-once delivery is safe.
class AsyncJobWatchdog {
public:
    using ResendFn = std::function<bool()>; // returns true on success

    AsyncJobWatchdog(int maxAttempts = 3,
                     int initialBackoffMs = 200)
        : m_maxAttempts(maxAttempts),
          m_initialBackoffMs(initialBackoffMs),
          m_stop(false) {
        m_worker = std::thread([this]() { Loop(); });
    }

    ~AsyncJobWatchdog() {
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_worker.joinable()) m_worker.join();
    }

    // Submit a fire-and-forget dispatch. The watchdog calls `resend` if
    // and only if a prior attempt has failed; the caller is responsible
    // for the initial attempt. After success, call MarkSuccess(id).
    uint64_t Track(ResendFn resend, std::string tag = "") {
        std::lock_guard<std::mutex> lk(m_mutex);
        uint64_t id = ++m_nextId;
        Entry e;
        e.resend = std::move(resend);
        e.attempts = 0;
        e.tag = std::move(tag);
        e.nextDeadline = std::chrono::steady_clock::time_point::max();
        m_entries.emplace(id, std::move(e));
        return id;
    }

    void MarkSuccess(uint64_t id) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_entries.erase(id);
    }

    // Schedule a resend after backoff for entry `id`. Called by producer
    // when its synchronous attempt fails. Gives up after MaxAttempts.
    void MarkFailureAndScheduleResend(uint64_t id) {
        std::unique_lock<std::mutex> lk(m_mutex);
        auto it = m_entries.find(id);
        if (it == m_entries.end()) return;
        if (++it->second.attempts >= m_maxAttempts) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                "AsyncJobWatchdog: %s giving up after %d attempts\n",
                it->second.tag.c_str(), it->second.attempts);
            m_entries.erase(it);
            return;
        }
        int backoffMs = m_initialBackoffMs << (it->second.attempts - 1);
        it->second.nextDeadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(backoffMs);
        lk.unlock();
        m_cv.notify_all();
    }

    size_t OutstandingCount() const {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_entries.size();
    }

private:
    struct Entry {
        ResendFn resend;
        int attempts;
        std::string tag;
        std::chrono::steady_clock::time_point nextDeadline;
    };

    void Loop() {
        std::unique_lock<std::mutex> lk(m_mutex);
        while (!m_stop) {
            auto now = std::chrono::steady_clock::now();
            auto nextWake = now + std::chrono::seconds(1);
            std::vector<uint64_t> due;
            for (auto& kv : m_entries) {
                if (kv.second.nextDeadline <= now) {
                    due.push_back(kv.first);
                } else if (kv.second.nextDeadline < nextWake) {
                    nextWake = kv.second.nextDeadline;
                }
            }
            for (uint64_t id : due) {
                auto it = m_entries.find(id);
                if (it == m_entries.end()) continue;
                ResendFn fn = it->second.resend;
                std::string tag = it->second.tag;
                int attempt = it->second.attempts;
                it->second.nextDeadline =
                    std::chrono::steady_clock::time_point::max();
                lk.unlock();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "AsyncJobWatchdog: resending %s attempt=%d\n",
                    tag.c_str(), attempt + 1);
                bool ok = false;
                try { ok = fn(); } catch (...) { ok = false; }
                lk.lock();
                if (ok) {
                    m_entries.erase(id);
                } else {
                    auto it2 = m_entries.find(id);
                    if (it2 != m_entries.end()) {
                        if (++it2->second.attempts >= m_maxAttempts) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "AsyncJobWatchdog: %s giving up after %d attempts\n",
                                it2->second.tag.c_str(), it2->second.attempts);
                            m_entries.erase(it2);
                        } else {
                            int backoffMs =
                                m_initialBackoffMs << (it2->second.attempts - 1);
                            it2->second.nextDeadline =
                                std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(backoffMs);
                        }
                    }
                }
            }
            m_cv.wait_until(lk, nextWake, [this]() { return m_stop; });
        }
    }

    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::unordered_map<uint64_t, Entry> m_entries;
    uint64_t m_nextId = 0;
    int m_maxAttempts;
    int m_initialBackoffMs;
    bool m_stop;
    std::thread m_worker;
};

} // namespace Distributed
} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_DISTRIBUTED_ASYNCJOBWATCHDOG_H_
