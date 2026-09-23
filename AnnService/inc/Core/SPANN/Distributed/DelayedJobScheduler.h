// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_DISTRIBUTED_DELAYEDJOBSCHEDULER_H_
#define _SPTAG_SPANN_DISTRIBUTED_DELAYEDJOBSCHEDULER_H_

#include "inc/Helper/Concurrent.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Helper/Logging.h"
#include "inc/Core/Common.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace SPTAG {
namespace SPANN {
namespace Distributed {

// DelayedJobScheduler runs a single worker thread that re-enqueues
// previously-failed ThreadPool jobs after an exponential backoff. It
// exists so async Merge/Split job retries can wait before retrying
// (instead of busy-spinning the pool worker) without blocking any actual
// pool slot during the wait.
//
// Jobs are owned by the scheduler between Schedule() and the moment they
// are transferred to the destination pool. If the scheduler is destroyed
// while jobs are still pending (e.g. process shutdown), the destructor
// drains the heap and deletes every undispatched job so the Helper::
// ThreadPool::Job allocations do not leak.
//
// The destination pool is held via shared_ptr so the scheduler can survive
// teardown ordering — the pool stays alive as long as either the scheduler
// or the original owner still holds a reference.
class DelayedJobScheduler {
public:
    DelayedJobScheduler() : m_stop(false) {
        m_worker = std::thread([this] { Loop(); });
    }

    ~DelayedJobScheduler() {
        {
            std::lock_guard<std::mutex> g(m_mu);
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_worker.joinable()) m_worker.join();
        std::lock_guard<std::mutex> g(m_mu);
        for (auto& e : m_heap) {
            if (e.job) delete e.job;
        }
        m_heap.clear();
    }

    // Take ownership of `job` and add it to `pool` after `delayMs`.
    // `pool` must be non-null; `job` must be non-null and not already
    // queued anywhere.
    void Schedule(std::shared_ptr<Helper::ThreadPool> pool,
                  Helper::ThreadPool::Job* job, int delayMs) {
        if (!pool || !job) { if (job) delete job; return; }
        Entry e;
        e.deadline = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(delayMs);
        e.pool = std::move(pool);
        e.job = job;
        {
            std::lock_guard<std::mutex> g(m_mu);
            m_heap.push_back(std::move(e));
            std::push_heap(m_heap.begin(), m_heap.end(), Cmp{});
        }
        m_cv.notify_all();
    }

    std::size_t Pending() const {
        std::lock_guard<std::mutex> g(m_mu);
        return m_heap.size();
    }

private:
    struct Entry {
        std::chrono::steady_clock::time_point deadline;
        std::shared_ptr<Helper::ThreadPool> pool;
        Helper::ThreadPool::Job* job = nullptr;
    };
    struct Cmp {
        bool operator()(const Entry& a, const Entry& b) const {
            return a.deadline > b.deadline;
        }
    };

    void Loop() {
        std::unique_lock<std::mutex> lk(m_mu);
        while (!m_stop) {
            if (m_heap.empty()) {
                m_cv.wait(lk);
                continue;
            }
            auto now = std::chrono::steady_clock::now();
            if (m_heap.front().deadline <= now) {
                Entry e = std::move(m_heap.front());
                std::pop_heap(m_heap.begin(), m_heap.end(), Cmp{});
                m_heap.pop_back();
                lk.unlock();
                if (e.pool) {
                    e.pool->add(e.job);
                } else if (e.job) {
                    delete e.job;
                }
                lk.lock();
                continue;
            }
            m_cv.wait_until(lk, m_heap.front().deadline);
        }
    }

    mutable std::mutex m_mu;
    std::condition_variable m_cv;
    std::vector<Entry> m_heap;
    bool m_stop;
    std::thread m_worker;
};

// Classify an async-job failure into transient (retry with backoff)
// vs permanent (drop with warning). Transient codes capture TiKV / IO
// errors that should clear on a later attempt; permanent codes capture
// logical inconsistencies (e.g. a vector ID outside the version map,
// a posting whose serialized header is malformed) that no number of
// retries can repair.
//
// ErrorCode::Fail is intentionally classified transient: every TiKV
// failure path in ExtraTiKVController returns Fail, and the few logical
// callers that also return Fail (e.g. MergePostings when the head vector
// is missing from its own posting) are rare enough that a bounded number
// of wasted retries is acceptable. If a more specific ErrorCode value
// becomes available for the logical case, demote those returns there
// and remove Fail from the transient set.
inline bool IsTransientAsyncJobError(ErrorCode ret) {
    switch (ret) {
        case ErrorCode::Fail:
        case ErrorCode::DiskIOFail:
        case ErrorCode::EmptyDiskIO:
        case ErrorCode::Socket_FailedConnectToEndPoint:
        case ErrorCode::Socket_FailedResolveEndPoint:
            return true;
        default:
            return false;
    }
}

// Exponential backoff with a cap. `attempt` is 0-based (0 = first retry).
inline int AsyncJobRetryBackoffMs(int attempt,
                                  int initialMs = 200,
                                  int capMs = 30000) {
    if (attempt < 0) attempt = 0;
    if (attempt > 20) attempt = 20;
    long long delay = (long long)initialMs << attempt;
    if (delay > capMs) delay = capMs;
    return (int)delay;
}

} // namespace Distributed
} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_DISTRIBUTED_DELAYEDJOBSCHEDULER_H_
