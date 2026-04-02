// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// HeadIndexCache: Production-grade per-tenant HeadIndex cache
// - S3-FIFO eviction (SOSP'23) for scan resistance + low promotion overhead
// - Single-flight loading to avoid thundering herd on cache miss
// - shared_ptr for zero-copy hit path + safe concurrent use
// - Versioned keys (tenant_id:epoch) for lazy invalidation
// - TTL with lazy expiration
//
#ifndef _SPTAG_HEAD_INDEX_CACHE_H_
#define _SPTAG_HEAD_INDEX_CACHE_H_

#include <atomic>
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

namespace SPTAG {

// Forward declaration
class VectorIndex;

namespace Cache {

// ============================================================
// Cache statistics for monitoring
// ============================================================
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> expirations{0};
    std::atomic<uint64_t> loads{0};
    std::atomic<uint64_t> load_errors{0};
    std::atomic<uint64_t> single_flight_waits{0};
    std::atomic<uint64_t> negative_hits{0};

    double HitRate() const {
        uint64_t h = hits.load(std::memory_order_relaxed);
        uint64_t m = misses.load(std::memory_order_relaxed);
        uint64_t total = h + m;
        return total > 0 ? (double)h / total : 0.0;
    }

    void Reset() {
        hits = misses = evictions = expirations = loads = 0;
        load_errors = single_flight_waits = negative_hits = 0;
    }
};

// ============================================================
// Cache entry: atomic unit of caching
// ============================================================
struct CacheEntry {
    std::string key;               // "tenant_id:epoch"
    std::shared_ptr<VectorIndex> value;  // nullptr = negative cache
    size_t size_bytes = 0;

    // Timing
    std::chrono::steady_clock::time_point created_at;

    // S3-FIFO: frequency counter (only needs 0/1/2+ distinction)
    std::atomic<uint8_t> freq{0};

    // Single-flight: loading state
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

    void NotifyReady() {
        load_cv.notify_all();
    }

    void Touch() {
        // S3-FIFO only needs to know "accessed at least once in current cycle"
        uint8_t cur = freq.load(std::memory_order_relaxed);
        if (cur < 2) {
            freq.store(cur + 1, std::memory_order_relaxed);
        }
    }
};

// ============================================================
// S3-FIFO Queue: Three-queue FIFO eviction
// ============================================================
class S3FIFOQueue {
public:
    // small_ratio: fraction of capacity for the small queue (default 10%)
    explicit S3FIFOQueue(double small_ratio = 0.1)
        : m_smallRatio(small_ratio) {}

    void SetCapacity(size_t capacity_bytes) {
        m_capacityBytes = capacity_bytes;
        m_smallCapBytes = static_cast<size_t>(capacity_bytes * m_smallRatio);
        m_mainCapBytes = capacity_bytes - m_smallCapBytes;
    }

    // Insert a new entry into the small queue
    void Insert(std::shared_ptr<CacheEntry> entry) {
        m_smallQueue.push_back(entry);
        m_smallUsed += entry->size_bytes;
    }

    // Evict entries until we free at least `needed` bytes
    // Returns evicted entries (caller handles cleanup)
    std::vector<std::shared_ptr<CacheEntry>> EvictUntilFree(size_t needed) {
        std::vector<std::shared_ptr<CacheEntry>> evicted;
        size_t freed = 0;

        while (freed < needed) {
            // Try evicting from small queue first
            if (!m_smallQueue.empty()) {
                auto entry = m_smallQueue.front();
                m_smallQueue.pop_front();
                m_smallUsed -= entry->size_bytes;

                if (entry->freq.load(std::memory_order_relaxed) >= 1) {
                    // Accessed at least once → promote to main queue
                    entry->freq.store(0, std::memory_order_relaxed);
                    m_mainQueue.push_back(entry);
                    m_mainUsed += entry->size_bytes;

                    // If main queue overflows, evict from main
                    while (m_mainUsed > m_mainCapBytes && !m_mainQueue.empty()) {
                        auto victim = m_mainQueue.front();
                        m_mainQueue.pop_front();
                        m_mainUsed -= victim->size_bytes;

                        if (victim->freq.load(std::memory_order_relaxed) >= 1) {
                            // Re-insert at tail with freq reset
                            victim->freq.store(0, std::memory_order_relaxed);
                            m_mainQueue.push_back(victim);
                            // Don't count as freed or evicted
                        } else {
                            // Evict from main
                            m_ghostKeys.insert(victim->key);
                            evicted.push_back(victim);
                            freed += victim->size_bytes;
                        }
                    }
                } else {
                    // Never accessed → evict directly
                    // Check ghost queue for re-admission hint
                    if (m_ghostKeys.count(entry->key)) {
                        // Previously evicted and re-inserted → give it another chance
                        m_ghostKeys.erase(entry->key);
                        m_mainQueue.push_back(entry);
                        m_mainUsed += entry->size_bytes;
                    } else {
                        evicted.push_back(entry);
                        freed += entry->size_bytes;
                    }
                }
            } else if (!m_mainQueue.empty()) {
                // Small queue empty, evict from main
                auto victim = m_mainQueue.front();
                m_mainQueue.pop_front();
                m_mainUsed -= victim->size_bytes;

                if (victim->freq.load(std::memory_order_relaxed) >= 1) {
                    victim->freq.store(0, std::memory_order_relaxed);
                    m_mainQueue.push_back(victim);
                } else {
                    m_ghostKeys.insert(victim->key);
                    evicted.push_back(victim);
                    freed += victim->size_bytes;
                }
            } else {
                break;  // Nothing to evict
            }
        }

        // Trim ghost queue to bounded size
        while (m_ghostKeys.size() > 10000) {
            m_ghostKeys.erase(m_ghostKeys.begin());
        }

        return evicted;
    }

    // Remove a specific entry from queues (for explicit invalidation)
    bool Remove(const std::string& key) {
        for (auto it = m_smallQueue.begin(); it != m_smallQueue.end(); ++it) {
            if ((*it)->key == key) {
                m_smallUsed -= (*it)->size_bytes;
                m_smallQueue.erase(it);
                return true;
            }
        }
        for (auto it = m_mainQueue.begin(); it != m_mainQueue.end(); ++it) {
            if ((*it)->key == key) {
                m_mainUsed -= (*it)->size_bytes;
                m_mainQueue.erase(it);
                return true;
            }
        }
        return false;
    }

    size_t TotalUsed() const { return m_smallUsed + m_mainUsed; }
    size_t SmallSize() const { return m_smallQueue.size(); }
    size_t MainSize() const { return m_mainQueue.size(); }
    size_t GhostSize() const { return m_ghostKeys.size(); }

private:
    double m_smallRatio;
    size_t m_capacityBytes = 0;
    size_t m_smallCapBytes = 0;
    size_t m_mainCapBytes = 0;

    std::deque<std::shared_ptr<CacheEntry>> m_smallQueue;
    std::deque<std::shared_ptr<CacheEntry>> m_mainQueue;
    std::unordered_set<std::string> m_ghostKeys;

    size_t m_smallUsed = 0;
    size_t m_mainUsed = 0;
};

// ============================================================
// HeadIndexCache: Main cache class
// ============================================================
class HeadIndexCache {
public:
    using LoadFunc = std::function<std::shared_ptr<VectorIndex>(const std::string& tenant_id, uint64_t epoch)>;
    using SizeEstimator = std::function<size_t(const VectorIndex*)>;

    struct Config {
        size_t capacity_bytes;
        std::chrono::seconds ttl;
        std::chrono::milliseconds load_timeout;
        std::chrono::seconds negative_ttl;
        double s3fifo_small_ratio;

        Config()
            : capacity_bytes(1ULL << 30)
            , ttl(600)
            , load_timeout(30000)
            , negative_ttl(10)
            , s3fifo_small_ratio(0.1)
        {}
    };

    explicit HeadIndexCache(Config config = {})
        : m_config(config), m_evictionQueue(config.s3fifo_small_ratio)
    {
        m_evictionQueue.SetCapacity(config.capacity_bytes);
    }

    // Set the loader function (called on cache miss)
    void SetLoader(LoadFunc loader) { m_loader = std::move(loader); }

    // Set the size estimator (for memory accounting)
    void SetSizeEstimator(SizeEstimator estimator) { m_sizeEstimator = std::move(estimator); }

    // ----------------------------------------------------------
    // Get: Hit path (hot path, must be fast)
    // ----------------------------------------------------------
    std::shared_ptr<VectorIndex> Get(const std::string& tenant_id, uint64_t epoch) {
        std::string key = MakeKey(tenant_id, epoch);

        // Fast path: shared lock read
        {
            std::shared_lock<std::shared_mutex> rlock(m_indexMutex);
            auto it = m_index.find(key);
            if (it != m_index.end()) {
                auto& entry = it->second;
                auto st = entry->state.load(std::memory_order_acquire);

                if (st == CacheEntry::State::LOADING) {
                    // Another thread is loading — wait
                    m_stats.single_flight_waits.fetch_add(1, std::memory_order_relaxed);
                    rlock.unlock();
                    entry->WaitUntilReady(m_config.load_timeout);
                    st = entry->state.load(std::memory_order_acquire);
                }

                if (st == CacheEntry::State::NEGATIVE) {
                    // Check negative TTL
                    auto age = std::chrono::steady_clock::now() - entry->created_at;
                    if (age < m_config.negative_ttl) {
                        m_stats.negative_hits.fetch_add(1, std::memory_order_relaxed);
                        return nullptr;
                    }
                    // Negative expired — fall through to miss
                }
                else if (st == CacheEntry::State::READY) {
                    // TTL check
                    auto age = std::chrono::steady_clock::now() - entry->created_at;
                    if (age < m_config.ttl) {
                        entry->Touch();
                        m_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        return entry->value;  // Zero-copy return
                    }
                    // TTL expired — treat as miss
                    m_stats.expirations.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        // Miss path
        m_stats.misses.fetch_add(1, std::memory_order_relaxed);
        return LoadAndInsert(key, tenant_id, epoch);
    }

    // ----------------------------------------------------------
    // Invalidate: Remove a specific tenant's cache
    // ----------------------------------------------------------
    void Invalidate(const std::string& tenant_id, uint64_t epoch) {
        std::string key = MakeKey(tenant_id, epoch);
        std::unique_lock<std::shared_mutex> wlock(m_indexMutex);
        auto it = m_index.find(key);
        if (it != m_index.end()) {
            m_usage.fetch_sub(it->second->size_bytes, std::memory_order_relaxed);
            m_evictionQueue.Remove(key);
            m_index.erase(it);
        }
    }

    // ----------------------------------------------------------
    // InvalidateAll for a tenant (all epochs)
    // ----------------------------------------------------------
    void InvalidateTenant(const std::string& tenant_id) {
        std::string prefix = tenant_id + ":";
        std::unique_lock<std::shared_mutex> wlock(m_indexMutex);
        for (auto it = m_index.begin(); it != m_index.end(); ) {
            if (it->first.compare(0, prefix.size(), prefix) == 0) {
                m_usage.fetch_sub(it->second->size_bytes, std::memory_order_relaxed);
                m_evictionQueue.Remove(it->first);
                it = m_index.erase(it);
            } else {
                ++it;
            }
        }
    }

    // ----------------------------------------------------------
    // Clear: Remove all entries
    // ----------------------------------------------------------
    void Clear() {
        std::unique_lock<std::shared_mutex> wlock(m_indexMutex);
        m_index.clear();
        m_evictionQueue = S3FIFOQueue(m_config.s3fifo_small_ratio);
        m_evictionQueue.SetCapacity(m_config.capacity_bytes);
        m_usage.store(0, std::memory_order_relaxed);
    }

    // ----------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------
    const CacheStats& Stats() const { return m_stats; }
    size_t Usage() const { return m_usage.load(std::memory_order_relaxed); }
    size_t Capacity() const { return m_config.capacity_bytes; }
    size_t EntryCount() const {
        std::shared_lock<std::shared_mutex> rlock(m_indexMutex);
        return m_index.size();
    }

    void SetCapacity(size_t bytes) {
        m_config.capacity_bytes = bytes;
        m_evictionQueue.SetCapacity(bytes);
    }

    void SetTTL(std::chrono::seconds ttl) { m_config.ttl = ttl; }

private:
    static std::string MakeKey(const std::string& tenant_id, uint64_t epoch) {
        return tenant_id + ":" + std::to_string(epoch);
    }

    // ----------------------------------------------------------
    // LoadAndInsert: Miss path with single-flight
    // ----------------------------------------------------------
    std::shared_ptr<VectorIndex> LoadAndInsert(
        const std::string& key, const std::string& tenant_id, uint64_t epoch)
    {
        std::shared_ptr<CacheEntry> entry;

        // Acquire exclusive lock to insert LOADING placeholder
        {
            std::unique_lock<std::shared_mutex> wlock(m_indexMutex);

            // Double-check: another thread may have loaded while we waited for the lock
            auto it = m_index.find(key);
            if (it != m_index.end()) {
                auto& existing = it->second;
                auto st = existing->state.load(std::memory_order_acquire);

                if (st == CacheEntry::State::LOADING) {
                    // Someone else is loading — release lock and wait
                    entry = existing;
                    wlock.unlock();
                    m_stats.single_flight_waits.fetch_add(1, std::memory_order_relaxed);
                    entry->WaitUntilReady(m_config.load_timeout);
                    if (entry->state.load() == CacheEntry::State::READY) {
                        m_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        return entry->value;
                    }
                    return nullptr;
                }

                if (st == CacheEntry::State::READY) {
                    auto age = std::chrono::steady_clock::now() - existing->created_at;
                    if (age < m_config.ttl) {
                        existing->Touch();
                        m_stats.hits.fetch_add(1, std::memory_order_relaxed);
                        return existing->value;
                    }
                    // Expired — remove old entry, will reload below
                    m_usage.fetch_sub(existing->size_bytes, std::memory_order_relaxed);
                    m_evictionQueue.Remove(key);
                    m_index.erase(it);
                }
            }

            // Insert LOADING placeholder
            entry = std::make_shared<CacheEntry>();
            entry->key = key;
            entry->state.store(CacheEntry::State::LOADING, std::memory_order_release);
            entry->created_at = std::chrono::steady_clock::now();
            m_index[key] = entry;
        }

        // Load outside the lock (IO-bound, may take seconds)
        std::shared_ptr<VectorIndex> loaded;
        if (m_loader) {
            try {
                loaded = m_loader(tenant_id, epoch);
            } catch (...) {
                loaded = nullptr;
            }
        }

        if (loaded) {
            // Success
            size_t sz = m_sizeEstimator ? m_sizeEstimator(loaded.get()) : (64 * 1024 * 1024);  // Default 64MB estimate
            entry->value = loaded;
            entry->size_bytes = sz;

            // Evict if needed (under exclusive lock)
            {
                std::unique_lock<std::shared_mutex> wlock(m_indexMutex);
                size_t current = m_usage.load(std::memory_order_relaxed);
                if (current + sz > m_config.capacity_bytes) {
                    auto evicted = m_evictionQueue.EvictUntilFree(current + sz - m_config.capacity_bytes);
                    for (auto& victim : evicted) {
                        m_index.erase(victim->key);
                        m_stats.evictions.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                m_evictionQueue.Insert(entry);
                m_usage.fetch_add(sz, std::memory_order_relaxed);
            }

            entry->state.store(CacheEntry::State::READY, std::memory_order_release);
            entry->NotifyReady();
            m_stats.loads.fetch_add(1, std::memory_order_relaxed);
            return loaded;
        } else {
            // Load failed → negative cache
            entry->size_bytes = 0;
            entry->state.store(CacheEntry::State::NEGATIVE, std::memory_order_release);
            entry->NotifyReady();
            m_stats.load_errors.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    // ----------------------------------------------------------
    // Members
    // ----------------------------------------------------------
    Config m_config;
    mutable std::shared_mutex m_indexMutex;
    std::unordered_map<std::string, std::shared_ptr<CacheEntry>> m_index;
    S3FIFOQueue m_evictionQueue;
    std::atomic<size_t> m_usage{0};
    CacheStats m_stats;

    LoadFunc m_loader;
    SizeEstimator m_sizeEstimator;
};

}  // namespace Cache
}  // namespace SPTAG

#endif // _SPTAG_HEAD_INDEX_CACHE_H_
