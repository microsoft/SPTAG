// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_FINEGRAINEDLOCK_H_
#define _SPTAG_COMMON_FINEGRAINEDLOCK_H_

#include "ResidentHashMap.h"
#include "ResidentMemory.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace SPTAG
{
    namespace COMMON
    {
        class FineGrainedLock {
        public:
            FineGrainedLock() : m_account(ResidentMemory::CurrentAccount()) {}
            ~FineGrainedLock() = default;

            std::mutex& operator[](SizeType idx) {
                return (*EnsureLocks())[hash_func((std::uint64_t)idx)];
            }

            const std::mutex& operator[](SizeType idx) const {
                return (*EnsureLocks())[hash_func((std::uint64_t)idx)];
            }

            uint64_t ResidentGrowthBytes() const
            {
                return m_locksPtr.load(std::memory_order_acquire)
                           ? 0
                           : ResidentAllocator<std::mutex>::AllocationBytes(PoolSize + 1);
            }

            static inline std::uint64_t hash_func(std::uint64_t idx)
            {
                return (idx * 99991 + _rotl64(idx, 2) + 101) & PoolSize;
            }

        private:
            ResidentArray<std::mutex>* EnsureLocks() const
            {
                auto* locks = m_locksPtr.load(std::memory_order_acquire);
                if (!locks)
                {
                    std::call_once(m_initFlag, [this, &locks] {
                        if (m_account && ResidentMemory::CurrentAccount() != m_account)
                        {
                            ResidentMemory::Scope scope(m_account);
                            m_locks = std::make_unique<ResidentArray<std::mutex>>(PoolSize + 1, m_account);
                        }
                        else
                        {
                            m_locks = std::make_unique<ResidentArray<std::mutex>>(PoolSize + 1, m_account);
                        }
                        locks = m_locks.get();
                        m_locksPtr.store(locks, std::memory_order_release);
                    });
                    locks = m_locksPtr.load(std::memory_order_acquire);
                }
                return locks;
            }

            static const std::uint64_t PoolSize = 4095;
            std::shared_ptr<ResidentMemory::Account> m_account;
            mutable std::once_flag m_initFlag;
            mutable std::unique_ptr<ResidentArray<std::mutex>> m_locks;
            mutable std::atomic<ResidentArray<std::mutex>*> m_locksPtr{nullptr};
        };

        class FineGrainedRWLock {
        public:
            FineGrainedRWLock() : m_account(ResidentMemory::CurrentAccount()) {}
            ~FineGrainedRWLock() = default;

            std::shared_timed_mutex& operator[](SizeType idx) {
                return GetLock(idx);
            }

            const std::shared_timed_mutex& operator[](SizeType idx) const {
                return GetLock(idx);
            }

            static inline SizeType hash_func(SizeType idx)
            {
                return idx;
            }

            uint64_t ResidentGrowthBytes(uint64_t admittedIDs) const
            {
                if (admittedIDs >= (uint64_t(1) << 62))
                    return (std::numeric_limits<uint64_t>::max)();
                const auto existing = m_created.load(std::memory_order_relaxed);
                const auto remaining = admittedIDs > existing ? admittedIDs - existing : 0;
                uint64_t growth = m_bucketsPtr.load(std::memory_order_acquire)
                                      ? 0
                                      : ResidentAllocator<Bucket>::AllocationBytes(BucketSize + 1);
                if (!remaining)
                    return growth;
                // Below 2^62 rotation has no wrap: the odd multiplier permutes
                // bucket IDs, so a contiguous VID range distributes uniformly.
                const auto capacity = LockMap::CapacityFor((admittedIDs + BucketSize) / (BucketSize + 1));
                const auto mapGrowth = ResidentAddBytes(
                    ResidentMultiplyBytes(remaining, LockMap::NodeBytes()),
                    ResidentMultiplyBytes(2 * (std::min<uint64_t>)(remaining, BucketSize + 1),
                                          LockMap::BucketBytes(capacity)));
                return ResidentAddBytes(growth, mapGrowth);
            }

        private:
            using LockMap = ResidentHashMap<SizeType, std::shared_timed_mutex>;
            struct alignas(64) Bucket
            {
                LockMap locks;
            };

            ResidentArray<Bucket>* EnsureBuckets() const
            {
                auto* buckets = m_bucketsPtr.load(std::memory_order_acquire);
                if (!buckets)
                {
                    std::call_once(m_initFlag, [this, &buckets] {
                        if (m_account && ResidentMemory::CurrentAccount() != m_account)
                        {
                            ResidentMemory::Scope scope(m_account);
                            m_buckets = std::make_unique<ResidentArray<Bucket>>(BucketSize + 1, m_account);
                        }
                        else
                        {
                            m_buckets = std::make_unique<ResidentArray<Bucket>>(BucketSize + 1, m_account);
                        }
                        buckets = m_buckets.get();
                        m_bucketsPtr.store(buckets, std::memory_order_release);
                    });
                    buckets = m_bucketsPtr.load(std::memory_order_acquire);
                }
                return buckets;
            }

            std::shared_timed_mutex& GetLock(SizeType idx) const {
                Bucket& bucket = (*EnsureBuckets())[BucketIndex(idx)];
                bool created = false;
                auto &result = bucket.locks.GetOrCreateReference(idx, &created);
                if (created)
                    m_created.fetch_add(1, std::memory_order_relaxed);
                return result;
            }

            static inline std::uint64_t BucketIndex(SizeType idx)
            {
                std::uint64_t key = (std::uint64_t)idx;
                return (key * 99991 + _rotl64(key, 2) + 101) & BucketSize;
            }

            static const std::uint64_t BucketSize = 4095;
            std::shared_ptr<ResidentMemory::Account> m_account;
            mutable std::once_flag m_initFlag;
            mutable std::unique_ptr<ResidentArray<Bucket>> m_buckets;
            mutable std::atomic<ResidentArray<Bucket>*> m_bucketsPtr{nullptr};
            mutable std::atomic<uint64_t> m_created{0};
        };
    }
}

#endif // _SPTAG_COMMON_FINEGRAINEDLOCK_H_
