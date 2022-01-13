// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_CONCURRENTSET_H_
#define _SPTAG_HELPER_CONCURRENTSET_H_

#ifndef _MSC_VER
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#else
#include <concurrent_unordered_map.h>
#include <concurrent_queue.h>
#include <concurrent_unordered_set.h>
#endif
namespace SPTAG
{
    namespace Helper
    {
        namespace Concurrent
        {
#ifndef _MSC_VER
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

                size_t count(const T& key) const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.count(key);
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
                ConcurrentMap(int capacity = 8) { m_lock.reset(new std::shared_timed_mutex); m_data.reserve(capacity); }

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

            template <typename T>
            class ConcurrentQueue
            {
            public:

                ConcurrentQueue() {}

                ~ConcurrentQueue() {}

                void push(const T& j)
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    m_queue.push(j);
                }

                bool try_pop(T& j)
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    if (m_queue.empty()) return false;

                    j = m_queue.front();
                    m_queue.pop();
                    return true;
                }

            protected:
                std::mutex m_lock;
                std::queue<T> m_queue;
            };
#else
            template <typename T>
            using ConcurrentSet = Concurrency::concurrent_unordered_set<T>;

            template <typename K, typename V>
            using ConcurrentMap = Concurrency::concurrent_unordered_map<K, V>;
            
            template <typename T>
            using ConcurrentQueue = Concurrency::concurrent_queue<T>;
#endif
        }
    }
}
#endif // _SPTAG_HELPER_CONCURRENTSET_H_