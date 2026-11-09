// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_LAYERREADLOCK_H_
#define _SPTAG_SPANN_LAYERREADLOCK_H_

#include <algorithm>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace SPTAG
{
    namespace SPANN
    {
        // Hold admission while waiting for the exclusive lock so fresh readers cannot extend the drain.
        class LayerMutex
        {
            std::shared_timed_mutex m_mutex;
            std::timed_mutex m_admission;

        public:
            void lock()
            {
                std::lock_guard<std::timed_mutex> admission(m_admission);
                m_mutex.lock();
            }

            void unlock() { m_mutex.unlock(); }

            void lock_shared()
            {
                std::lock_guard<std::timed_mutex> admission(m_admission);
                m_mutex.lock_shared();
            }

            void unlock_shared() { m_mutex.unlock_shared(); }

            bool try_lock()
            {
                std::unique_lock<std::timed_mutex> admission(m_admission, std::try_to_lock);
                return admission.owns_lock() && m_mutex.try_lock();
            }

            template <typename Clock, typename Duration>
            bool try_lock_until(const std::chrono::time_point<Clock, Duration>& deadline)
            {
                std::unique_lock<std::timed_mutex> admission(m_admission, std::defer_lock);
                return admission.try_lock_until(deadline) && m_mutex.try_lock_until(deadline);
            }

            template <typename Rep, typename Period>
            bool try_lock_for(const std::chrono::duration<Rep, Period>& duration)
            {
                return try_lock_until(std::chrono::steady_clock::now() + duration);
            }
        };

        // Public SPANN operations call one another; acquire each topology lock only once per thread.
        class LayerReadLock
        {
            std::shared_lock<LayerMutex> m_lock;
            bool m_owner = false;

            struct HeldLock
            {
                const LayerMutex* mutex;
                bool writeContext;
            };

            static std::vector<HeldLock>& HeldLocks()
            {
                static thread_local std::vector<HeldLock> locks;
                return locks;
            }

        public:
            static bool IsHeld(const LayerMutex& mutex)
            {
                const auto& locks = HeldLocks();
                return std::find_if(locks.begin(), locks.end(),
                    [&](const HeldLock& lock) { return lock.mutex == &mutex; }) != locks.end();
            }

            static bool IsWriteHeld(const LayerMutex& mutex)
            {
                const auto& locks = HeldLocks();
                return std::find_if(locks.begin(), locks.end(),
                    [&](const HeldLock& lock) { return lock.mutex == &mutex && lock.writeContext; }) != locks.end();
            }

            LayerReadLock(LayerMutex& mutex, bool enabled, bool alreadyLocked = false, bool writeContext = false)
            {
                if (!enabled) return;
                auto& locks = HeldLocks();
                if (IsHeld(mutex)) return;
                if (!alreadyLocked) m_lock = std::shared_lock<LayerMutex>(mutex);
                locks.push_back({&mutex, writeContext});
                m_owner = true;
            }

            ~LayerReadLock()
            {
                if (m_owner) HeldLocks().pop_back();
            }

            LayerReadLock(const LayerReadLock&) = delete;
            LayerReadLock& operator=(const LayerReadLock&) = delete;
        };
    }
}

#endif
