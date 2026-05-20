// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_FINEGRAINEDLOCK_H_
#define _SPTAG_COMMON_FINEGRAINEDLOCK_H_

#include <shared_mutex>
#include <vector>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace SPTAG
{
    namespace COMMON
    {
        class FineGrainedLock {
        public:
            FineGrainedLock() {
                m_locks.reset(new std::mutex[PoolSize + 1]);
            }
            ~FineGrainedLock() {}

            std::mutex& operator[](SizeType idx) {
                unsigned index = hash_func((unsigned)idx);
                return m_locks[index];
            }

            const std::mutex& operator[](SizeType idx) const {
                unsigned index = hash_func((unsigned)idx);
                return m_locks[index];
            }

            static inline unsigned hash_func(unsigned idx)
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & PoolSize;
            }

        private:
            static const int PoolSize = 32767;
            std::unique_ptr<std::mutex[]> m_locks;
        };

        class FineGrainedRWLock {
        public:
            FineGrainedRWLock() {
                m_buckets.reset(new Bucket[BucketCount]);
            }
            ~FineGrainedRWLock() {}

            std::shared_timed_mutex& operator[](SizeType idx) {
                return GetLock(idx);
            }

            const std::shared_timed_mutex& operator[](SizeType idx) const {
                return GetLock(idx);
            }

            static inline unsigned hash_func(unsigned idx)
            {
                return idx;
            }

            // Bucket index for the internal mutex-sharded unordered_map of
            // per-posting locks. Exposed for callers that need an array sized
            // to BucketCount and indexed by the same granularity as the lock
            // pool (e.g. ExtraDynamicSearcher::m_remoteBucketLocked).
            static inline unsigned BucketIndex(SizeType idx)
            {
                unsigned key = static_cast<unsigned>(idx);
                return ((unsigned)(key * 99991) + _rotl(key, 2) + 101) & BucketMask;
            }

            static const int BucketMask = 32767;
            static const int BucketCount = BucketMask + 1;
        private:
            struct Bucket {
                std::mutex mutex;
                std::unordered_map<SizeType, std::unique_ptr<std::shared_timed_mutex>> locks;
            };

            std::shared_timed_mutex& GetLock(SizeType idx) const {
                Bucket& bucket = m_buckets[BucketIndex(idx)];
                std::lock_guard<std::mutex> guard(bucket.mutex);
                auto iter = bucket.locks.find(idx);
                if (iter == bucket.locks.end()) {
                    iter = bucket.locks.emplace(idx, std::unique_ptr<std::shared_timed_mutex>(new std::shared_timed_mutex())).first;
                }
                return *iter->second;
            }

            mutable std::unique_ptr<Bucket[]> m_buckets;
        };
    }
}

#endif // _SPTAG_COMMON_FINEGRAINEDLOCK_H_