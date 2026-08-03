// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// RemoteLeaseTable
// ----------------
// Owner-side bookkeeping for cross-node merge / structural-op locks.
// Each bucket has a TTL-bounded lease AND a monotonically increasing
// fencing token so a zombie holder that resumes after lease expiry has
// its late operations rejected (see Async Job Fault Tolerance in the
// design doc).
//
// API:
//   TryAcquire(bucket)              -> uint64_t token (0 = denied)
//   Validate(bucket, token)         -> bool, the held token still matches
//   Release(bucket, token)          -> bool, only releases if token matches
//   IsLocked(bucket)                -> bool, auto-clears expired entries
//
// The TTL knob is `RemoteLockTtlMs` in SPANN options (default 30s).
#ifndef _SPTAG_SPANN_DISTRIBUTED_REMOTELEASETABLE_H_
#define _SPTAG_SPANN_DISTRIBUTED_REMOTELEASETABLE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace SPTAG::SPANN {

    // RAII lease holder for a remote per-bucket lock issued by
    // WorkerNode::SendRemoteLock.  Stores the fencing token so the
    // release call can be validated by the owner.  Used by both Split
    // (via a token map for batched acquisition) and MergePostings
    // (per-candidate, one lease at a time).
    struct RemoteLeaseGuard {
        WorkerNode* router = nullptr;
        int nodeIndex = -1;
        int layer = 0;
        SizeType vid = -1;
        std::uint64_t token = 0;

        RemoteLeaseGuard() = default;
        RemoteLeaseGuard(const RemoteLeaseGuard&) = delete;
        RemoteLeaseGuard& operator=(const RemoteLeaseGuard&) = delete;
        RemoteLeaseGuard(RemoteLeaseGuard&& o) noexcept { *this = std::move(o); }
        RemoteLeaseGuard& operator=(RemoteLeaseGuard&& o) noexcept {
            release();
            router = o.router; nodeIndex = o.nodeIndex; layer = o.layer;
            vid = o.vid; token = o.token;
            o.router = nullptr; o.token = 0;
            return *this;
        }
        ~RemoteLeaseGuard() { release(); }

        // Returns true on success (token != 0).  Caller decides whether
        // a denial means "skip candidate" or "propagate failure".
        bool acquire(WorkerNode* r, int n, int l, SizeType v) {
            release();
            if (!r) return false;
            std::uint64_t t = r->SendRemoteLock(n, l, v, true, 0);
            if (t == 0) return false;
            router = r; nodeIndex = n; layer = l; vid = v; token = t;
            return true;
        }
        void release() {
            if (router && token) {
                router->SendRemoteLock(nodeIndex, layer, vid, false, token);
            }
            router = nullptr; token = 0;
        }
        bool active() const { return router != nullptr && token != 0; }
    };

    class RemoteLeaseTable {
    public:
        using Clock = std::chrono::steady_clock;

        explicit RemoteLeaseTable(std::size_t bucketCount, int ttlMs = 30000)
            : m_count(bucketCount + 1), m_ttlMs(ttlMs)
        {
            m_expiry = std::make_unique<std::atomic<std::int64_t>[]>(m_count);
            m_tokens = std::make_unique<std::atomic<std::uint64_t>[]>(m_count);
            for (std::size_t i = 0; i < m_count; ++i) {
                m_expiry[i].store(0, std::memory_order_relaxed);
                m_tokens[i].store(0, std::memory_order_relaxed);
            }
        }

        void SetTtlMs(int ttlMs) { if (ttlMs > 0) m_ttlMs.store(ttlMs, std::memory_order_relaxed); }
        int GetTtlMs() const { return m_ttlMs.load(std::memory_order_relaxed); }

        // Try to grant a lease for bucket. Succeeds iff bucket is unlocked
        // OR the previous holder's lease has expired (auto-reclamation).
        // Returns the fencing token (>= 1) on success, 0 on denial.
        std::uint64_t TryAcquire(unsigned bucket) {
            if (bucket >= m_count) return 0;
            const std::int64_t nowNs = NowNs();
            const std::int64_t ttlNs = (std::int64_t)m_ttlMs.load(std::memory_order_relaxed) * 1'000'000LL;
            std::int64_t current = m_expiry[bucket].load(std::memory_order_acquire);
            for (;;) {
                if (current != 0 && current > nowNs) return 0;     // still held by live lease
                const std::int64_t newExpiry = nowNs + ttlNs;
                if (m_expiry[bucket].compare_exchange_weak(current, newExpiry,
                        std::memory_order_acq_rel)) {
                    std::uint64_t tok = m_nextToken.fetch_add(1, std::memory_order_acq_rel) + 1;
                    m_tokens[bucket].store(tok, std::memory_order_release);
                    return tok;
                }
            }
        }

        // True iff bucket currently holds `token` AND lease not expired.
        bool Validate(unsigned bucket, std::uint64_t token) const {
            if (bucket >= m_count || token == 0) return false;
            std::int64_t exp = m_expiry[bucket].load(std::memory_order_acquire);
            if (exp == 0 || exp <= NowNs()) return false;
            return m_tokens[bucket].load(std::memory_order_acquire) == token;
        }

        // Release the lease only if the caller's token still matches.
        // Late unlocks from a zombie holder whose lease expired (and was
        // reacquired by another holder) silently no-op.
        bool Release(unsigned bucket, std::uint64_t token) {
            if (bucket >= m_count) return false;
            std::uint64_t held = m_tokens[bucket].load(std::memory_order_acquire);
            if (token == 0 || held != token) return false;
            // Clear token first so a concurrent Validate sees the release
            // before the expiry window closes.
            m_tokens[bucket].store(0, std::memory_order_release);
            m_expiry[bucket].store(0, std::memory_order_release);
            return true;
        }

        // True iff the lease is currently held AND not expired.
        bool IsLocked(unsigned bucket) {
            if (bucket >= m_count) return false;
            std::int64_t current = m_expiry[bucket].load(std::memory_order_acquire);
            if (current == 0) return false;
            if (current > NowNs()) return true;
            std::int64_t expected = current;
            if (m_expiry[bucket].compare_exchange_strong(expected, 0,
                    std::memory_order_acq_rel)) {
                m_tokens[bucket].store(0, std::memory_order_release);
            }
            return false;
        }

    private:
        static std::int64_t NowNs() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now().time_since_epoch()).count();
        }

        std::size_t m_count;
        std::atomic<int> m_ttlMs;
        std::unique_ptr<std::atomic<std::int64_t>[]> m_expiry;
        std::unique_ptr<std::atomic<std::uint64_t>[]> m_tokens;
        std::atomic<std::uint64_t> m_nextToken{0};
    };

} // namespace SPTAG::SPANN

#endif // _SPTAG_SPANN_DISTRIBUTED_REMOTELEASETABLE_H_
