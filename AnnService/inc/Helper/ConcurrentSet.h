// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_CONCURRENTSET_H_
#define _SPTAG_HELPER_CONCURRENTSET_H_

#ifndef _MSC_VER
#ifdef TBB
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_priority_queue.h>
#else
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <deque>
#endif // TBB
#else
#include <concurrent_unordered_map.h>
#include <concurrent_queue.h>
#include <concurrent_unordered_set.h>
#include <concurrent_priority_queue.h>
#endif // _MSC_VER

namespace SPTAG
{
    namespace Helper
    {
        namespace Concurrent
        {
#ifndef _MSC_VER
#ifdef TBB
            template <typename T>
            using ConcurrentSet = tbb::concurrent_unordered_set<T>;

            template <typename K, typename V>
            using ConcurrentMap = tbb::concurrent_unordered_map<K, V>;

            template <typename T>
            using ConcurrentQueue = tbb::concurrent_queue<T>;

            template <typename T>
            using ConcurrentPriorityQueue = tbb::concurrent_priority_queue<T>;
#else
            template <typename T>
            class ConcurrentSet
            {
            public:
                typedef typename std::unordered_set<T>::iterator iterator;
                typedef typename std::unordered_set<T>::const_iterator const_iterator;

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

                std::pair<iterator, bool> insert(const T& key)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.insert(key);
                }

                // Unsafe iteration: caller must ensure no concurrent modification.
                // Mirrors the semantics of tbb::concurrent_unordered_set::unsafe_begin/unsafe_end,
                // which (unlike the locked accessors above) do not synchronize.
                iterator begin() { return m_data.begin(); }
                iterator end() { return m_data.end(); }
                const_iterator begin() const { return m_data.begin(); }
                const_iterator end() const { return m_data.end(); }

            private:
                std::unique_ptr<std::shared_timed_mutex> m_lock;
                std::unordered_set<T> m_data;
            };

            template <typename K, typename V>
            class ConcurrentMap
            {
            public:
                typedef typename std::unordered_map<K, V>::iterator iterator;
                typedef typename std::unordered_map<K, V>::value_type value_type;

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

                size_t unsafe_erase(const K& k)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.erase(k);
                }

                template<class P>
                std::pair<iterator, bool> insert(P&& v)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.insert(v);
                }

            private:
                std::unique_ptr<std::shared_timed_mutex> m_lock;
                std::unordered_map<K, V> m_data;
            };

            template <typename T>
            class ConcurrentQueue
            {
            public:
                typedef typename std::deque<T>::iterator iterator;
                typedef typename std::deque<T>::const_iterator const_iterator;

                ConcurrentQueue() {}

                ~ConcurrentQueue() {}

                void push(const T& j)
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    m_queue.push_back(j);
                }

                bool try_pop(T& j)
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    if (m_queue.empty()) {
                        return false;
                    }
                    j = m_queue.front();
                    m_queue.pop_front();
                    return true;
                }

                // The TBB concurrent_queue exposes empty() and unsafe_size() as
                // best-effort, lock-free queries. Here we take the lock so the
                // snapshot is consistent; callers should still treat the result
                // as advisory in concurrent contexts.
                bool empty() const
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    return m_queue.empty();
                }

                size_t unsafe_size() const
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    return m_queue.size();
                }

                // Unsafe iteration: caller must ensure no concurrent modification,
                // matching tbb::concurrent_queue::unsafe_begin/unsafe_end semantics.
                iterator unsafe_begin() { return m_queue.begin(); }
                iterator unsafe_end() { return m_queue.end(); }
                const_iterator unsafe_begin() const { return m_queue.begin(); }
                const_iterator unsafe_end() const { return m_queue.end(); }

            protected:
                mutable std::mutex m_lock;
                std::deque<T> m_queue;
            };

            template <typename T>
            class ConcurrentPriorityQueue 
            {
            public:
                ConcurrentPriorityQueue() {}
                ~ConcurrentPriorityQueue() {}

            size_t size() const {
                std::lock_guard<std::mutex> lock(m_lock);
                return m_queue.size();
            }

            void push(const T& value) {
                std::lock_guard<std::mutex> lock(m_lock);
                m_queue.push(value);
            }

            bool try_pop(T& value) {
                std::lock_guard<std::mutex> lock(m_lock);
                if (m_queue.empty()) {
                    return false;
                }
                value = m_queue.top();
                m_queue.pop();
                return true;
            }

            private:
                mutable std::mutex m_lock;
                std::priority_queue<T> m_queue;
            };
#endif // TBB
#else
            template <typename T>
            using ConcurrentSet = Concurrency::concurrent_unordered_set<T>;

            template <typename K, typename V>
            using ConcurrentMap = Concurrency::concurrent_unordered_map<K, V>;
            
            template <typename T>
            using ConcurrentQueue = Concurrency::concurrent_queue<T>;

            template <typename T>
            using ConcurrentPriorityQueue = Concurrency::concurrent_priority_queue<T>;
#endif
        }
    }
}
#endif // _SPTAG_HELPER_CONCURRENTSET_H_
