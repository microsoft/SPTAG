// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_CONCURRENTSET_H_
#define _SPTAG_HELPER_CONCURRENTSET_H_

#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

namespace SPTAG
{
    namespace Helper
    {
        namespace Concurrent
        {
            template <typename T>
            class ConcurrentSet
            {
            public:
                ConcurrentSet() { m_lock.reset(new std::shared_timed_mutex); }

                ~ConcurrentSet() {}

                size_t size() const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.size();
                }

                bool contains(const T& key) const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return (m_data.find(key) != m_data.end());
                }

                void insert(const T& key)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    m_data.insert(key);
                }

            private:
                std::unique_ptr<std::shared_timed_mutex> m_lock;
                std::unordered_set<T> m_data;
            };

            template <typename K, typename V>
            class ConcurrentMap
            {
                typedef typename std::unordered_map<K, V>::iterator iterator;
            public:
                ConcurrentMap() { m_lock.reset(new std::shared_timed_mutex); }

                ~ConcurrentMap() {}

                iterator find(const K& k)
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.find(k);
                }

                iterator end() noexcept
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.end();
                }

                V& operator[] (const K& k)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data[k];
                }

            private:
                std::unique_ptr<std::shared_timed_mutex> m_lock;
                std::unordered_map<K, V> m_data;
            };
        }
    }
}
#endif // _SPTAG_HELPER_CONCURRENTSET_H_