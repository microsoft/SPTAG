// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ChunkedHeadIndexCache: Production cache for tens of thousands of tenants
// - HeadIndex split into fixed-size chunks for slab-friendly allocation
// - S3-FIFO eviction at tenant granularity (all chunks evicted together)
// - Slab allocator: pre-allocated chunk pool, O(1) alloc/free, zero fragmentation
// - Single-flight loading per tenant
// - Versioned keys (tenant_id:epoch)
// - Lock-free hit path via atomic freq counter + shared_mutex
//
#ifndef _SPTAG_CHUNKED_HEAD_INDEX_CACHE_H_
#define _SPTAG_CHUNKED_HEAD_INDEX_CACHE_H_

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>

namespace SPTAG {
namespace Cache {

// ============================================================
// Slab Allocator: Fixed-size chunk pool
// Pre-allocates N chunks of CHUNK_SIZE bytes.
// alloc/free are O(1), no fragmentation, no system malloc on hot path.
// ============================================================
class SlabAllocator {
public:
    explicit SlabAllocator(size_t chunk_size, size_t max_chunks)
        : m_chunkSize(chunk_size), m_maxChunks(max_chunks)
    {
        m_pool.resize(max_chunks * chunk_size);
        m_freeList.reserve(max_chunks);
        for (size_t i = 0; i < max_chunks; i++) {
            m_freeList.push_back(i);
        }
    }

    // Allocate a chunk, returns pointer or nullptr if pool exhausted
    void* Alloc() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_freeList.empty()) return nullptr;
        size_t idx = m_freeList.back();
        m_freeList.pop_back();
        m_allocatedCount++;
        return &m_pool[idx * m_chunkSize];
    }

    // Free a chunk back to pool
    void Free(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t offset = static_cast<char*>(ptr) - m_pool.data();
        size_t idx = offset / m_chunkSize;
        m_freeList.push_back(idx);
        m_allocatedCount--;
    }

    size_t ChunkSize() const { return m_chunkSize; }
    size_t MaxChunks() const { return m_maxChunks; }
    size_t AllocatedChunks() const { return m_allocatedCount; }
    size_t FreeChunks() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_freeList.size();
    }
    size_t TotalBytes() const { return m_maxChunks * m_chunkSize; }
    size_t UsedBytes() const { return m_allocatedCount * m_chunkSize; }

private:
    size_t m_chunkSize;
    size_t m_maxChunks;
    std::vector<char> m_pool;              // Contiguous memory pool
    std::vector<size_t> m_freeList;        // Free chunk indices
    mutable std::mutex m_mutex;
    size_t m_allocatedCount = 0;
};

// ============================================================
// ChunkHandle: RAII wrapper for a slab-allocated chunk
// ============================================================
struct ChunkHandle {
    void* data = nullptr;
    size_t used_bytes = 0;      // Actual data size (<= chunk_size)
    SlabAllocator* allocator = nullptr;

    ChunkHandle() = default;
    ChunkHandle(void* d, size_t used, SlabAllocator* a) : data(d), used_bytes(used), allocator(a) {}
    ~ChunkHandle() { if (data && allocator) allocator->Free(data); }

    // Move only
    ChunkHandle(ChunkHandle&& o) noexcept : data(o.data), used_bytes(o.used_bytes), allocator(o.allocator) {
        o.data = nullptr; o.allocator = nullptr;
    }
    ChunkHandle& operator=(ChunkHandle&& o) noexcept {
        if (this != &o) {
            if (data && allocator) allocator->Free(data);
            data = o.data; used_bytes = o.used_bytes; allocator = o.allocator;
            o.data = nullptr; o.allocator = nullptr;
        }
        return *this;
    }
    ChunkHandle(const ChunkHandle&) = delete;
    ChunkHandle& operator=(const ChunkHandle&) = delete;
};

// ============================================================
// CachedTenant: All chunks for one tenant + metadata
// ============================================================
struct CachedTenant {
    std::string tenant_id;
    uint64_t epoch = 0;
    std::vector<std::shared_ptr<ChunkHandle>> chunks;  // shared_ptr for safe concurrent access
    size_t total_bytes = 0;       // Sum of chunk used_bytes

    // S3-FIFO frequency
    std::atomic<uint8_t> freq{0};

    // Timing
    std::chrono::steady_clock::time_point created_at;

    // Single-flight loading
    enum class State : uint8_t { EMPTY, LOADING, READY, NEGATIVE };
    std::atomic<State> state{State::EMPTY};
    std::mutex load_mutex;
    std::condition_variable load_cv;

    void WaitUntilReady(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(load_mutex);
        load_cv.wait_for(lk, timeout, [this] {
            auto s = state.load(std::memory_order_acquire);
            return s == State::READY || s == State::NEGATIVE;
        });
    }

    void NotifyReady() { load_cv.notify_all(); }

    void Touch() {
        uint8_t cur = freq.load(std::memory_order_relaxed);
        if (cur < 2) freq.store(cur + 1, std::memory_order_relaxed);
    }

    size_t ChunkCount() const { return chunks.size(); }
    size_t MemoryUsage() const { return chunks.size() * (chunks.empty() ? 0 : chunks[0]->allocator->ChunkSize()); }
};

// ============================================================
// Cache statistics
// ============================================================
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> expirations{0};
    std::atomic<uint64_t> loads{0};
    std::atomic<uint64_t> load_errors{0};
    std::atomic<uint64_t> single_flight_waits{0};
    std::atomic<uint64_t> slab_alloc_fails{0};

    double HitRate() const {
        uint64_t h = hits.load(std::memory_order_relaxed);
        uint64_t total = h + misses.load(std::memory_order_relaxed);
        return total > 0 ? (double)h / total : 0.0;
    }
};

// ============================================================
// S3-FIFO Queue (tenant-level eviction)
// ============================================================
class TenantS3FIFO {
public:
    explicit TenantS3FIFO(double small_ratio = 0.1) : m_smallRatio(small_ratio) {}

    void SetCapacity(size_t cap) {
        m_capacity = cap;
        m_smallCap = static_cast<size_t>(cap * m_smallRatio);
    }

    void Insert(std::shared_ptr<CachedTenant> t) {
        m_smallQ.push_back(t);
        m_smallUsed += t->MemoryUsage();
    }

    std::vector<std::shared_ptr<CachedTenant>> EvictUntilFree(size_t needed) {
        std::vector<std::shared_ptr<CachedTenant>> evicted;
        size_t freed = 0;

        while (freed < needed) {
            if (!m_smallQ.empty()) {
                auto t = m_smallQ.front(); m_smallQ.pop_front();
                m_smallUsed -= t->MemoryUsage();

                if (t->freq.load(std::memory_order_relaxed) >= 1) {
                    t->freq.store(0, std::memory_order_relaxed);
                    m_mainQ.push_back(t);
                    m_mainUsed += t->MemoryUsage();
                    EvictMainOverflow(evicted, freed);
                } else {
                    if (m_ghost.count(t->tenant_id)) {
                        m_ghost.erase(t->tenant_id);
                        m_mainQ.push_back(t);
                        m_mainUsed += t->MemoryUsage();
                    } else {
                        evicted.push_back(t);
                        freed += t->MemoryUsage();
                    }
                }
            } else if (!m_mainQ.empty()) {
                EvictOneMain(evicted, freed);
            } else {
                break;
            }
        }

        while (m_ghost.size() > 50000) m_ghost.erase(m_ghost.begin());
        return evicted;
    }

    void Remove(const std::string& tid) {
        auto remove_from = [&](std::deque<std::shared_ptr<CachedTenant>>& q, size_t& used) {
            for (auto it = q.begin(); it != q.end(); ++it) {
                if ((*it)->tenant_id == tid) {
                    used -= (*it)->MemoryUsage();
                    q.erase(it);
                    return true;
                }
            }
            return false;
        };
        if (!remove_from(m_smallQ, m_smallUsed))
            remove_from(m_mainQ, m_mainUsed);
    }

    size_t TotalUsed() const { return m_smallUsed + m_mainUsed; }

private:
    void EvictMainOverflow(std::vector<std::shared_ptr<CachedTenant>>& evicted, size_t& freed) {
        size_t mainCap = m_capacity - m_smallCap;
        while (m_mainUsed > mainCap && !m_mainQ.empty()) {
            EvictOneMain(evicted, freed);
        }
    }

    void EvictOneMain(std::vector<std::shared_ptr<CachedTenant>>& evicted, size_t& freed) {
        auto v = m_mainQ.front(); m_mainQ.pop_front();
        m_mainUsed -= v->MemoryUsage();
        if (v->freq.load(std::memory_order_relaxed) >= 1) {
            v->freq.store(0, std::memory_order_relaxed);
            m_mainQ.push_back(v);
            m_mainUsed += v->MemoryUsage();
        } else {
            m_ghost.insert(v->tenant_id);
            evicted.push_back(v);
            freed += v->MemoryUsage();
        }
    }

    double m_smallRatio;
    size_t m_capacity = 0, m_smallCap = 0;
    std::deque<std::shared_ptr<CachedTenant>> m_smallQ, m_mainQ;
    std::unordered_set<std::string> m_ghost;
    size_t m_smallUsed = 0, m_mainUsed = 0;
};

// ============================================================
// ShardedIndex: N independent shards, each with its own lock
// Miss on shard K only blocks shard K. Other shards unaffected.
// ============================================================
class ShardedIndex {
    static constexpr size_t NUM_SHARDS = 64;

    struct Shard {
        mutable std::shared_mutex mutex;
        std::unordered_map<std::string, std::shared_ptr<CachedTenant>> map;
    };

    std::array<Shard, NUM_SHARDS> m_shards;

    size_t ShardOf(const std::string& key) const {
        return std::hash<std::string>{}(key) % NUM_SHARDS;
    }

public:
    // Shared-lock find (concurrent reads within same shard)
    std::shared_ptr<CachedTenant> Find(const std::string& key) const {
        auto& shard = m_shards[ShardOf(key)];
        std::shared_lock<std::shared_mutex> rlock(shard.mutex);
        auto it = shard.map.find(key);
        return (it != shard.map.end()) ? it->second : nullptr;
    }

    // Exclusive-lock insert (only blocks this shard)
    // Returns {existing_or_new, was_inserted}
    std::pair<std::shared_ptr<CachedTenant>, bool> InsertIfAbsent(
        const std::string& key, std::shared_ptr<CachedTenant> entry)
    {
        auto& shard = m_shards[ShardOf(key)];
        std::unique_lock<std::shared_mutex> wlock(shard.mutex);
        auto [it, inserted] = shard.map.emplace(key, entry);
        return {it->second, inserted};
    }

    // Exclusive-lock upsert
    void Upsert(const std::string& key, std::shared_ptr<CachedTenant> entry) {
        auto& shard = m_shards[ShardOf(key)];
        std::unique_lock<std::shared_mutex> wlock(shard.mutex);
        shard.map[key] = std::move(entry);
    }

    // Exclusive-lock erase
    void Erase(const std::string& key) {
        auto& shard = m_shards[ShardOf(key)];
        std::unique_lock<std::shared_mutex> wlock(shard.mutex);
        shard.map.erase(key);
    }

    // Erase multiple keys (may span shards)
    void EraseBatch(const std::vector<std::string>& keys) {
        for (auto& k : keys) Erase(k);
    }

    void Clear() {
        for (auto& shard : m_shards) {
            std::unique_lock<std::shared_mutex> wlock(shard.mutex);
            shard.map.clear();
        }
    }

    size_t Size() const {
        size_t total = 0;
        for (auto& shard : m_shards) {
            std::shared_lock<std::shared_mutex> rlock(shard.mutex);
            total += shard.map.size();
        }
        return total;
    }
};

// ============================================================
// ChunkedHeadIndexCache: Main class
// Sharded index + separate eviction lock = fine-grained concurrency
// ============================================================
class ChunkedHeadIndexCache {
public:
    // Loader: given tenant_id and epoch, return serialized HeadIndex bytes
    using LoadFunc = std::function<std::vector<char>(const std::string& tenant_id, uint64_t epoch)>;

    struct Config {
        size_t capacity_bytes;           // Total cache memory
        size_t chunk_size;               // Slab chunk size
        std::chrono::seconds ttl;
        std::chrono::milliseconds load_timeout;
        double s3fifo_small_ratio;

        Config()
            : capacity_bytes(1ULL << 30)     // 1 GB
            , chunk_size(4ULL << 20)          // 4 MB
            , ttl(600)
            , load_timeout(30000)
            , s3fifo_small_ratio(0.1)
        {}
    };

    explicit ChunkedHeadIndexCache(Config cfg = Config())
        : m_config(cfg)
        , m_slab(cfg.chunk_size, cfg.capacity_bytes / cfg.chunk_size)
        , m_eviction(cfg.s3fifo_small_ratio)
    {
        m_eviction.SetCapacity(cfg.capacity_bytes);
    }

    void SetLoader(LoadFunc loader) { m_loader = std::move(loader); }

    // ----------------------------------------------------------
    // Get: Hit path uses shard-level shared_lock (no global lock)
    // ----------------------------------------------------------
    std::shared_ptr<CachedTenant> Get(const std::string& tenant_id, uint64_t epoch) {
        // Fast path: shard-level shared lock (only blocks this shard's writes)
        auto existing = m_index.Find(tenant_id);
        if (existing) {
            auto st = existing->state.load(std::memory_order_acquire);

            if (st == CachedTenant::State::LOADING) {
                m_stats.single_flight_waits.fetch_add(1, std::memory_order_relaxed);
                existing->WaitUntilReady(m_config.load_timeout);
                st = existing->state.load(std::memory_order_acquire);
            }

            if (st == CachedTenant::State::READY && existing->epoch == epoch) {
                auto age = std::chrono::steady_clock::now() - existing->created_at;
                if (age < m_config.ttl) {
                    existing->Touch();
                    m_stats.hits.fetch_add(1, std::memory_order_relaxed);
                    return existing;
                }
                m_stats.expirations.fetch_add(1, std::memory_order_relaxed);
            }

            if (st == CachedTenant::State::NEGATIVE) {
                return nullptr;
            }
        }

        m_stats.misses.fetch_add(1, std::memory_order_relaxed);
        return LoadAndInsert(tenant_id, epoch);
    }

    // ----------------------------------------------------------
    // Reassemble: Copy tenant's chunks into contiguous buffer
    // ----------------------------------------------------------
    static std::vector<char> Reassemble(const CachedTenant& tenant) {
        std::vector<char> result;
        result.reserve(tenant.total_bytes);
        for (auto& chunk : tenant.chunks) {
            result.insert(result.end(),
                static_cast<char*>(chunk->data),
                static_cast<char*>(chunk->data) + chunk->used_bytes);
        }
        return result;
    }

    // ----------------------------------------------------------
    // Invalidate
    // ----------------------------------------------------------
    void Invalidate(const std::string& tenant_id) {
        m_index.Erase(tenant_id);
        std::lock_guard<std::mutex> elock(m_evictionMutex);
        m_eviction.Remove(tenant_id);
    }

    void Clear() {
        m_index.Clear();
        std::lock_guard<std::mutex> elock(m_evictionMutex);
        m_eviction = TenantS3FIFO(m_config.s3fifo_small_ratio);
        m_eviction.SetCapacity(m_config.capacity_bytes);
    }

    // ----------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------
    const CacheStats& Stats() const { return m_stats; }
    size_t SlabUsedBytes() const { return m_slab.UsedBytes(); }
    size_t SlabFreeChunks() const { return m_slab.FreeChunks(); }
    size_t SlabTotalBytes() const { return m_slab.TotalBytes(); }
    size_t EntryCount() const { return m_index.Size(); }

private:
    // ----------------------------------------------------------
    // LoadAndInsert: Single-flight, shard-level lock only
    // ----------------------------------------------------------
    std::shared_ptr<CachedTenant> LoadAndInsert(const std::string& tenant_id, uint64_t epoch) {
        // Try to insert LOADING placeholder (shard-level exclusive lock)
        auto placeholder = std::make_shared<CachedTenant>();
        placeholder->tenant_id = tenant_id;
        placeholder->epoch = epoch;
        placeholder->state.store(CachedTenant::State::LOADING, std::memory_order_release);
        placeholder->created_at = std::chrono::steady_clock::now();

        auto [entry, inserted] = m_index.InsertIfAbsent(tenant_id, placeholder);

        if (!inserted) {
            // Another thread already has an entry
            auto st = entry->state.load(std::memory_order_acquire);
            if (st == CachedTenant::State::LOADING) {
                m_stats.single_flight_waits.fetch_add(1, std::memory_order_relaxed);
                entry->WaitUntilReady(m_config.load_timeout);
            }
            st = entry->state.load(std::memory_order_acquire);
            if (st == CachedTenant::State::READY && entry->epoch == epoch) {
                auto age = std::chrono::steady_clock::now() - entry->created_at;
                if (age < m_config.ttl) {
                    entry->Touch();
                    m_stats.hits.fetch_add(1, std::memory_order_relaxed);
                    return entry;
                }
            }
            // Stale entry — remove and retry (rare path)
            m_index.Erase(tenant_id);
            { std::lock_guard<std::mutex> el(m_evictionMutex); m_eviction.Remove(tenant_id); }
            return LoadAndInsert(tenant_id, epoch);  // Recursion depth = 1
        }

        // We inserted the LOADING placeholder — we are the loader.
        // Load OUTSIDE any lock (IO-bound).
        std::vector<char> raw_data;
        if (m_loader) {
            try { raw_data = m_loader(tenant_id, epoch); }
            catch (...) { raw_data.clear(); }
        }

        if (raw_data.empty()) {
            placeholder->state.store(CachedTenant::State::NEGATIVE, std::memory_order_release);
            placeholder->NotifyReady();
            m_stats.load_errors.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        // Allocate chunks from slab
        size_t num_chunks = (raw_data.size() + m_config.chunk_size - 1) / m_config.chunk_size;
        std::vector<std::shared_ptr<ChunkHandle>> chunks;
        chunks.reserve(num_chunks);

        size_t offset = 0;
        for (size_t i = 0; i < num_chunks; i++) {
            size_t this_chunk = std::min(m_config.chunk_size, raw_data.size() - offset);
            void* ptr = m_slab.Alloc();
            if (!ptr) {
                // Evict (eviction lock only, not index lock)
                std::vector<std::string> keys_to_erase;
                {
                    std::lock_guard<std::mutex> elock(m_evictionMutex);
                    auto evicted = m_eviction.EvictUntilFree(m_config.chunk_size);
                    for (auto& victim : evicted) {
                        keys_to_erase.push_back(victim->tenant_id);
                        m_stats.evictions.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                // Erase from index after releasing eviction lock (different locks, no deadlock)
                m_index.EraseBatch(keys_to_erase);

                ptr = m_slab.Alloc();
                if (!ptr) {
                    m_stats.slab_alloc_fails.fetch_add(1, std::memory_order_relaxed);
                    placeholder->state.store(CachedTenant::State::NEGATIVE, std::memory_order_release);
                    placeholder->NotifyReady();
                    return nullptr;
                }
            }
            memcpy(ptr, raw_data.data() + offset, this_chunk);
            chunks.push_back(std::make_shared<ChunkHandle>(ptr, this_chunk, &m_slab));
            offset += this_chunk;
        }

        placeholder->chunks = std::move(chunks);
        placeholder->total_bytes = raw_data.size();

        // Register in eviction (eviction lock only)
        { std::lock_guard<std::mutex> elock(m_evictionMutex); m_eviction.Insert(placeholder); }

        placeholder->state.store(CachedTenant::State::READY, std::memory_order_release);
        placeholder->NotifyReady();
        m_stats.loads.fetch_add(1, std::memory_order_relaxed);
        return placeholder;
    }

    Config m_config;
    SlabAllocator m_slab;
    ShardedIndex m_index;                    // 64 shards, each with own shared_mutex
    TenantS3FIFO m_eviction;
    mutable std::mutex m_evictionMutex;      // Separate lock for eviction queue
    CacheStats m_stats;
    LoadFunc m_loader;
};

}  // namespace Cache
}  // namespace SPTAG

#endif // _SPTAG_CHUNKED_HEAD_INDEX_CACHE_H_
