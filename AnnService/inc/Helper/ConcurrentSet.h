// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_CONCURRENTSET_H_
#define _SPTAG_HELPER_CONCURRENTSET_H_

#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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
#endif // TBB
#else
#include <concurrent_unordered_map.h>
#include <concurrent_queue.h>
#include <concurrent_unordered_set.h>
#include <concurrent_priority_queue.h>
#endif // _MSC_VER

#ifdef FOLLY
#include <folly/concurrency/ConcurrentHashMap.h>
#endif // FOLLY

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
                typedef typename std::unordered_set<T>::iterator iterator;

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

                std::pair<iterator, bool> insert(const T& key)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.insert(key);
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

                iterator find(const K& k) const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.find(k);
                }

                iterator begin() const noexcept
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.begin();
                }

                iterator end() const noexcept
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

                void clear()
                {
                    std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                    m_data.clear();
                }
                
                size_t size() const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    return m_data.size();
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
                    if (m_queue.empty()) {
                        return false;
                    }
                    j = m_queue.front();
                    m_queue.pop();
                    return true;
                }

            protected:
                std::mutex m_lock;
                std::queue<T> m_queue;
            };

            template <typename T>
            class ConcurrentPriorityQueue 
            {
            public:
                ConcurrentPriorityQueue() {}
                ~ConcurrentPriorityQueue() {}

            size_type size() const {
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
                std::mutex m_lock;
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

#ifdef FOLLY
            template <typename K, typename V>
            using ConcurrentHashMap = folly::ConcurrentHashMap<K, V>;
#else
            template <typename K, typename V>
            class ConcurrentHashMap {
                using Map = std::unordered_map<K, V>;
                using Entry = typename Map::value_type;
                using Snapshot = std::vector<Entry>;

            public:
                // Snapshot iterators remain valid after the operation releases its lock.
                class iterator {
                public:
                    using iterator_category = std::forward_iterator_tag;
                    using value_type = Entry;
                    using difference_type = std::ptrdiff_t;
                    using pointer = const Entry*;
                    using reference = const Entry&;

                    iterator() = default;
                    explicit iterator(std::shared_ptr<Snapshot> entries) : m_entries(std::move(entries)) {}
                    reference operator*() const { return (*m_entries)[m_position]; }
                    pointer operator->() const { return &**this; }
                    iterator& operator++() { ++m_position; return *this; }
                    iterator operator++(int) { auto previous = *this; ++*this; return previous; }
                    bool operator==(const iterator& other) const {
                        if (AtEnd() || other.AtEnd()) return AtEnd() && other.AtEnd();
                        return m_entries == other.m_entries && m_position == other.m_position;
                    }
                    bool operator!=(const iterator& other) const { return !(*this == other); }

                private:
                    bool AtEnd() const { return !m_entries || m_position == m_entries->size(); }
                    std::shared_ptr<Snapshot> m_entries;
                    size_t m_position = 0;
                };

                iterator find(const K& k) const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(m_lock);
                    auto found = m_data.find(k);
                    return found == m_data.end() ? end() : CopyEntry(found);
                }

                iterator begin() const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(m_lock);
                    return iterator(std::make_shared<Snapshot>(m_data.begin(), m_data.end()));
                }

                iterator end() const noexcept
                {
                    return iterator();
                }

                std::pair<iterator, bool> insert_or_assign(K&& k, V&& v) 
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_lock);
                    auto result = m_data.insert_or_assign(std::move(k), std::move(v));
                    return {CopyEntry(result.first), result.second};
                }

                bool assign_if_equal(K&& k, const V& expected, V&& desired)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_lock);
                    auto iter = m_data.find(k);
                    if (iter != m_data.end() && iter->second == expected) {
                        iter->second = desired;
                        return true;
                    }
                    return false;
                }

                template <typename Predicate>
                bool assign_if(K&& k, V&& desired, Predicate&& predicate)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_lock);
                    auto iter = m_data.find(k);
                    if (iter != m_data.end() && predicate(iter->second)) {
                        iter->second = desired;
                        return true;
                    }
                    return false;
                }

                size_t erase(const K& k)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_lock);
                    return m_data.erase(k);
                }

                template<class P>
                std::pair<iterator, bool> insert(P&& v)
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_lock);
                    auto result = m_data.insert(std::forward<P>(v));
                    return {CopyEntry(result.first), result.second};
                }

                void clear()
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_lock);
                    m_data.clear();
                }
                
                size_t size() const
                {
                    std::shared_lock<std::shared_timed_mutex> lock(m_lock);
                    return m_data.size();
                }
            private:
                static iterator CopyEntry(typename Map::const_iterator entry) {
                    auto snapshot = std::make_shared<Snapshot>();
                    snapshot->push_back(*entry);
                    return iterator(std::move(snapshot));
                }

                mutable std::shared_timed_mutex m_lock;
                Map m_data;
            };
#endif
        }
    }
}
#endif // _SPTAG_HELPER_CONCURRENTSET_H_
