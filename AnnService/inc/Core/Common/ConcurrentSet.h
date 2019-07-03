// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_CONCURRENTSET_H_
#define _SPTAG_COMMON_CONCURRENTSET_H_

#include <shared_mutex>
#include <unordered_set>

namespace SPTAG
{
    namespace COMMON
    {
        template <typename T>
        class ConcurrentSet
        {
        public:
            ConcurrentSet();

            ~ConcurrentSet();

            bool find(const T& key) const;

            void insert(const T& key);

            void lock();

            void unlock();

            void lock_shared();

            void unlock_shared();

            size_t size() const;

        private:
            std::unique_ptr<std::shared_mutex> m_lock;
            std::unordered_set<T> m_data;
        };

        template<typename T>
        ConcurrentSet<T>::ConcurrentSet()
        {
            m_lock.reset(new std::shared_mutex);
        }

        template<typename T>
        ConcurrentSet<T>::~ConcurrentSet()
        {
        }

        template<typename T>
        bool ConcurrentSet<T>::find(const T& key) const
        {
            m_lock->lock_shared();
            bool res = (m_data.find(key) != m_data.end());
            m_lock->unlock_shared();
            return res;
        }

        template<typename T>
        void ConcurrentSet<T>::insert(const T& key)
        {
            m_lock->lock();
            m_data.insert(key);
            m_lock->unlock();
        }

        template<typename T>
        void ConcurrentSet<T>::lock()
        {
            m_lock->lock();
        }

        template<typename T>
        void ConcurrentSet<T>::unlock()
        {
            m_lock->unlock();
        }

        template<typename T>
        void ConcurrentSet<T>::lock_shared()
        {
            m_lock->lock_shared();
        }

        template<typename T>
        void ConcurrentSet<T>::unlock_shared()
        {
            m_lock->unlock_shared();
        }

        template<typename T>
        size_t ConcurrentSet<T>::size() const
        {
            m_lock->lock_shared();
            size_t res = m_data.size();
            m_lock->unlock_shared();
            return res;
        }
    }
}
#endif