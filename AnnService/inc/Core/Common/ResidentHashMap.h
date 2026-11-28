// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ResidentMemory.h"
#include <shared_mutex>
#include <tuple>

namespace SPTAG
{
namespace COMMON
{
// Individually allocated nodes, not a slab: growth costs do not depend on
// implementation-private STL node layouts or lazy read-side bucket creation.
template <typename K, typename V> class ResidentHashMap
{
    struct Node
    {
        std::pair<const K, V> value;
        Node *next;
        Node(const K &key, Node *link)
            : value(std::piecewise_construct, std::forward_as_tuple(key), std::tuple<>()), next(link)
        {
        }
    };
    using NodeAllocator = ResidentAllocator<Node>;
    using BucketAllocator = ResidentAllocator<Node *>;

  public:
    class iterator
    {
        friend class ResidentHashMap;
        const ResidentHashMap *owner = nullptr;
        Node *node = nullptr;
        size_t bucket = 0;
        iterator(const ResidentHashMap *map, Node *entry, size_t index) : owner(map), node(entry), bucket(index)
        {
        }

      public:
        iterator() = default;
        const std::pair<const K, V> &operator*() const
        {
            return node->value;
        }
        const std::pair<const K, V> *operator->() const
        {
            return &node->value;
        }
        iterator &operator++()
        {
            if (node)
                node = node->next;
            while (!node && owner && ++bucket < owner->m_capacity)
                node = owner->m_buckets[bucket];
            return *this;
        }
        iterator operator++(int)
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const iterator &other) const
        {
            return node == other.node;
        }
        bool operator!=(const iterator &other) const
        {
            return !(*this == other);
        }
    };
    class Reference
    {
        ResidentHashMap *owner;
        K key;

      public:
        Reference(ResidentHashMap *map, const K &value) : owner(map), key(value)
        {
        }
        Reference &operator=(const V &value)
        {
            owner->Set(key, value);
            return *this;
        }
        operator V() const
        {
            V result{};
            owner->Get(key, result);
            return result;
        }
    };
    using allocator_type = ResidentAllocator<std::pair<const K, V>>;
    explicit ResidentHashMap(std::shared_ptr<ResidentMemory::Account> account = ResidentMemory::CurrentAccount())
        : m_nodes(account), m_bucketAllocator(std::move(account))
    {
    }
    ResidentHashMap(size_t, std::hash<K>, std::equal_to<K>, const allocator_type &allocator)
        : ResidentHashMap(allocator.Account())
    {
    }
    ResidentHashMap(const ResidentHashMap &) = delete;
    ResidentHashMap &operator=(const ResidentHashMap &) = delete;
    ~ResidentHashMap()
    {
        clear();
        if (m_buckets)
            m_bucketAllocator.deallocate(m_buckets, m_capacity);
    }
    allocator_type get_allocator() const
    {
        return allocator_type(m_nodes.Account());
    }
    size_t size() const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        return m_size;
    }
    size_t bucket_count() const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        return m_capacity;
    }
    static size_t CapacityFor(size_t count)
    {
        if (!count)
            return 0;
        size_t result = 8;
        while (result < count)
        {
            if (result > (std::numeric_limits<size_t>::max)() / 2)
                throw std::bad_alloc();
            result *= 2;
        }
        return result;
    }
    static uint64_t NodeBytes()
    {
        return NodeAllocator::AllocationBytes(1);
    }
    static uint64_t BucketBytes(size_t capacity)
    {
        return capacity ? BucketAllocator::AllocationBytes(capacity) : 0;
    }
    uint64_t GrowthBytes(size_t target) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        const auto nodes = target > m_size ? target - m_size : 0;
        const auto capacity = CapacityFor(target);
        if (nodes > (std::numeric_limits<uint64_t>::max)() / NodeBytes())
            throw std::bad_alloc();
        const auto buckets = capacity > m_capacity ? BucketBytes(capacity) : 0;
        const auto nodeBytes = nodes * NodeBytes();
        if (buckets > ((std::numeric_limits<uint64_t>::max)() - nodeBytes) / 2)
            throw std::bad_alloc();
        return nodeBytes + 2 * buckets;
    }
    bool Get(const K &key, V &value) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto *node = Find(key);
        if (!node)
            return false;
        value = node->value.second;
        return true;
    }
    V at(const K &key) const
    {
        V value{};
        if (!Get(key, value))
            throw std::out_of_range("Resident hash key not found");
        return value;
    }
    void Set(const K &key, const V &value)
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        GetOrCreate(key)->value.second = value;
    }
    template <typename F> bool Update(const K &key, F &&update)
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        auto *node = Find(key);
        return node && update(node->value.second);
    }
    // Stable until erase/clear/destruction; lock maps never erase published locks.
    V &GetOrCreateReference(const K &key, bool *created = nullptr)
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        if (created)
            *created = Find(key) == nullptr;
        return GetOrCreate(key)->value.second;
    }
    Reference operator[](const K &key)
    {
        return Reference(this, key);
    }
    iterator find(const K &key) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        return iterator(this, Find(key), m_capacity ? std::hash<K>()(key) & (m_capacity - 1) : 0);
    }
    // Iteration requires the owner's lifecycle/tree lock, like the previous map.
    iterator begin() const
    {
        size_t bucket = 0;
        while (bucket < m_capacity && !m_buckets[bucket])
            ++bucket;
        return iterator(this, bucket < m_capacity ? m_buckets[bucket] : nullptr, bucket);
    }
    iterator end() const
    {
        return iterator(this, nullptr, m_capacity);
    }
    template <typename F> void ForEach(F &&visit) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        for (auto iter = begin(); iter != end(); ++iter)
            visit(iter->first, iter->second);
    }
    bool erase(const K &key)
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        if (!m_capacity)
            return false;
        auto **link = &m_buckets[std::hash<K>()(key) & (m_capacity - 1)];
        while (*link && (*link)->value.first != key)
            link = &(*link)->next;
        if (!*link)
            return false;
        auto *node = *link;
        *link = node->next;
        Destroy(node);
        --m_size;
        return true;
    }
    void clear()
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        for (size_t i = 0; i < m_capacity; ++i)
            while (m_buckets[i])
            {
                auto *node = m_buckets[i];
                m_buckets[i] = node->next;
                Destroy(node);
            }
        m_size = 0;
    }
    void swap(ResidentHashMap &other)
    {
        if (this == &other)
            return;
        std::scoped_lock lock(m_mutex, other.m_mutex);
        std::swap(m_nodes, other.m_nodes);
        std::swap(m_bucketAllocator, other.m_bucketAllocator);
        std::swap(m_buckets, other.m_buckets);
        std::swap(m_capacity, other.m_capacity);
        std::swap(m_size, other.m_size);
    }

  private:
    Node *Find(const K &key) const
    {
        if (!m_capacity)
            return nullptr;
        auto *node = m_buckets[std::hash<K>()(key) & (m_capacity - 1)];
        while (node && node->value.first != key)
            node = node->next;
        return node;
    }
    Node *GetOrCreate(const K &key)
    {
        if (auto *existing = Find(key))
            return existing;
        if (m_size == m_capacity)
            Rehash(CapacityFor(m_size + 1));
        const auto bucket = std::hash<K>()(key) & (m_capacity - 1);
        auto *node = m_nodes.allocate(1);
        try
        {
            new (node) Node(key, m_buckets[bucket]);
        }
        catch (...)
        {
            m_nodes.deallocate(node, 1);
            throw;
        }
        m_buckets[bucket] = node;
        ++m_size;
        return node;
    }
    void Destroy(Node *node)
    {
        node->~Node();
        m_nodes.deallocate(node, 1);
    }
    void Rehash(size_t capacity)
    {
        auto *buckets = m_bucketAllocator.allocate(capacity);
        std::fill_n(buckets, capacity, nullptr);
        for (size_t i = 0; i < m_capacity; ++i)
            while (m_buckets[i])
            {
                auto *node = m_buckets[i];
                m_buckets[i] = node->next;
                const auto bucket = std::hash<K>()(node->value.first) & (capacity - 1);
                node->next = buckets[bucket];
                buckets[bucket] = node;
            }
        if (m_buckets)
            m_bucketAllocator.deallocate(m_buckets, m_capacity);
        m_buckets = buckets;
        m_capacity = capacity;
    }
    mutable std::shared_mutex m_mutex;
    NodeAllocator m_nodes;
    BucketAllocator m_bucketAllocator;
    Node **m_buckets = nullptr;
    size_t m_capacity = 0, m_size = 0;
};
} // namespace COMMON
} // namespace SPTAG
