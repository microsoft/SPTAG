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
                rwlock.reset(new std::shared_timed_mutex);
            }
            ~FineGrainedLock() { 
                for (size_t i = 0; i < locks.size(); i++)
                    locks[i].reset();
                locks.clear();
            }
            
            void resize(SizeType n) {
                SizeType current = (SizeType)locks.size();
                if (current <= n) {
                    {
                        std::unique_lock<std::shared_timed_mutex> lock(*rwlock);
                        locks.resize(n);
                    }
                    for (SizeType i = current; i < n; i++)
                        locks[i].reset(new std::mutex);
                }
                else {
                    for (SizeType i = n; i < current; i++)
                        locks[i].reset();
                    locks.resize(n);
                }
            }

            std::mutex& operator[](SizeType idx) {
                std::shared_lock<std::shared_timed_mutex> lock(*rwlock);
                return *locks[idx];
            }

            const std::mutex& operator[](SizeType idx) const {
                std::shared_lock<std::shared_timed_mutex> lock(*rwlock);
                return *locks[idx];
            }
        private:
            std::unique_ptr<std::shared_timed_mutex> rwlock;
            std::vector<std::shared_ptr<std::mutex>> locks;
        };
    }
}

#endif // _SPTAG_COMMON_FINEGRAINEDLOCK_H_