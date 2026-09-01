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
                std::uint64_t index = hash_func((std::uint64_t)idx);
                return m_locks[index];
            }

            const std::mutex& operator[](SizeType idx) const {
		std::uint64_t index = hash_func((std::uint64_t)idx);
                return m_locks[index];
            }

            static inline std::uint64_t hash_func(std::uint64_t idx)
            {
                return (idx * 99991 + _rotl64(idx, 2) + 101) & PoolSize;
            }

        private:
            static const std::uint64_t PoolSize = 32767;
            std::unique_ptr<std::mutex[]> m_locks;
        };

        class FineGrainedRWLock {
        public:
            FineGrainedRWLock() {
                m_buckets.reset(new Bucket[BucketSize + 1]);
            }
            ~FineGrainedRWLock() {}

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

            static inline std::uint64_t BucketIndex(SizeType idx)
            {
                std::uint64_t key = (std::uint64_t)idx;
                return (key * 99991 + _rotl64(key, 2) + 101) & BucketSize;
            }

            static const std::uint64_t BucketSize = 32767;
            mutable std::unique_ptr<Bucket[]> m_buckets;
        };
    }
}

#endif // _SPTAG_COMMON_FINEGRAINEDLOCK_H_
