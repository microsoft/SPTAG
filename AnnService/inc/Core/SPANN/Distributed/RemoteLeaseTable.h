// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// RemoteLeaseTable
// ----------------
// Owner-side bookkeeping for cross-node merge / structural-op locks.
// Backs the per-bucket advisory flag that local Split / Merge consult via
// WaitForRemoteBucketUnlocked before mutating a head whose ownership is
// shared with a remote candidate.
//
// Design contract (see Async Job Fault Tolerance):
//   * Each acquired lock carries a bounded TTL.  If the holder crashes or
//     stops responding, the lease auto-expires and the owner is free to
//     proceed (or grant the bucket to another holder).
//   * No keepalive: structural ops are expected to complete in under one
//     TTL.  If they exceed the TTL, the holder must retry the whole job;
//     the owner has already released the lease.
//
// The TTL is the single configurable knob (default 30s, matching the
// design's lease-TTL recommendation).  A future iteration can add a
// fencing token so a zombie holder that resumes after expiry has its
// late unlock rejected — that requires a protocol bump on
// RemoteLockRequest/Response, which we'll do once a real owner-restart
// test exists to validate the change.  For now the in-memory lease
// table provides the safety net the design requires: zombie holders
// never indefinitely block the owner.

#ifndef _SPTAG_SPANN_DISTRIBUTED_REMOTELEASETABLE_H_
#define _SPTAG_SPANN_DISTRIBUTED_REMOTELEASETABLE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace SPTAG::SPANN {

    class RemoteLeaseTable {
    public:
        using Clock = std::chrono::steady_clock;

        // bucketCount must match the searcher's lock-pool bucket count
        // (FineGrainedRWLock::BucketIndex range).  Allocates one slot per
        // bucket; slots start in the unlocked state (expiry == 0).
        explicit RemoteLeaseTable(std::size_t bucketCount, int ttlMs = 30000)
            : m_count(bucketCount + 1), m_ttlMs(ttlMs)
        {
            m_expiry = std::make_unique<std::atomic<std::int64_t>[]>(m_count);
            for (std::size_t i = 0; i < m_count; ++i) m_expiry[i].store(0, std::memory_order_relaxed);
        }

        void SetTtlMs(int ttlMs) { if (ttlMs > 0) m_ttlMs.store(ttlMs, std::memory_order_relaxed); }
        int GetTtlMs() const { return m_ttlMs.load(std::memory_order_relaxed); }

        // Try to grant a lease for bucket.  Succeeds iff bucket is unlocked
        // OR the previous holder's lease has expired (auto-reclamation).
        // Records the new expiry deadline.
        bool TryAcquire(unsigned bucket) {
            if (bucket >= m_count) return false;
            const std::int64_t nowNs = NowNs();
            const std::int64_t ttlNs = (std::int64_t)m_ttlMs.load(std::memory_order_relaxed) * 1'000'000LL;
            std::int64_t current = m_expiry[bucket].load(std::memory_order_acquire);
            for (;;) {
                if (current != 0 && current > nowNs) return false;     // still held by live lease
                const std::int64_t newExpiry = nowNs + ttlNs;
                if (m_expiry[bucket].compare_exchange_weak(current, newExpiry,
                        std::memory_order_acq_rel)) return true;
                // CAS lost: re-evaluate with the updated `current`.
            }
        }

        // Release the lease unconditionally.  In the current protocol the
        // caller is trusted (holder cooperates).  When a fencing token is
        // added, this becomes a token-validated release.
        void Release(unsigned bucket) {
            if (bucket >= m_count) return;
            m_expiry[bucket].store(0, std::memory_order_release);
        }

        // True iff the lease is currently held AND not expired.  Auto-clears
        // expired entries so a stuck holder doesn't permanently block the
        // owner's Split/Merge path.
        bool IsLocked(unsigned bucket) {
            if (bucket >= m_count) return false;
            std::int64_t current = m_expiry[bucket].load(std::memory_order_acquire);
            if (current == 0) return false;
            if (current > NowNs()) return true;
            // Expired: try to clear (best-effort; loss of race is OK because
            // a concurrent holder either renewed or is also expired).
            std::int64_t expected = current;
            m_expiry[bucket].compare_exchange_strong(expected, 0,
                std::memory_order_acq_rel);
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
    };

} // namespace SPTAG::SPANN

#endif // _SPTAG_SPANN_DISTRIBUTED_REMOTELEASETABLE_H_
