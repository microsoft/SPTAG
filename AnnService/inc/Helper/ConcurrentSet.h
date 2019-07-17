// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_CONCURRENTSET_H_
#define _SPTAG_HELPER_CONCURRENTSET_H_

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

            size_t size() const;
      
            bool contains(const T& key) const;

            void insert(const T& key);

            std::shared_timed_mutex& getLock();

        private:
            std::unique_ptr<std::shared_timed_mutex> m_lock;
            std::unordered_set<T> m_data;
        };

        template<typename T>
        ConcurrentSet<T>::ConcurrentSet()
        {
            m_lock.reset(new std::shared_timed_mutex);
        }

        template<typename T>
        ConcurrentSet<T>::~ConcurrentSet()
        {
        }

        template<typename T>
        size_t ConcurrentSet<T>::size() const
        {
            std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
            return m_data.size();
        }

        template<typename T>
        bool ConcurrentSet<T>::contains(const T& key) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
            return (m_data.find(key) != m_data.end());
        }

        template<typename T>
        void ConcurrentSet<T>::insert(const T& key)
        {
            std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
            m_data.insert(key);
        }

        template<typename T>
        std::shared_timed_mutex& ConcurrentSet<T>::getLock()
        {
            return *m_lock;
        }
    }
}
#endif // _SPTAG_HELPER_CONCURRENTSET_H_