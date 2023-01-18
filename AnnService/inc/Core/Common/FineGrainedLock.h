// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_FINEGRAINEDLOCK_H_
#define _SPTAG_COMMON_FINEGRAINEDLOCK_H_

#include <shared_mutex>
#include <vector>
#include <mutex>
#include <memory>

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
        private:
            static const int PoolSize = 32767;
            std::unique_ptr<std::mutex[]> m_locks;

            inline unsigned hash_func(unsigned idx) const
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & PoolSize;
            }
        };

        class FineGrainedRWLock {
        public:
            FineGrainedRWLock() {
                m_locks.reset(new std::shared_timed_mutex[PoolSize + 1]);
            }
            ~FineGrainedRWLock() {}

            std::shared_timed_mutex& operator[](SizeType idx) {
                unsigned index = hash_func((unsigned)idx);
                return m_locks[index];
            }

            const std::shared_timed_mutex& operator[](SizeType idx) const {
                unsigned index = hash_func((unsigned)idx);
                return m_locks[index];
            }

            inline unsigned hash_func(unsigned idx) const
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & PoolSize;
            }
        private:
            static const int PoolSize = 32767;
            std::unique_ptr<std::shared_timed_mutex[]> m_locks;
        };
    }
}

#endif // _SPTAG_COMMON_FINEGRAINEDLOCK_H_