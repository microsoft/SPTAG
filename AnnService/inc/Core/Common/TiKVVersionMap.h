// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_TIKV_VERSIONMAP_H_
#define _SPTAG_COMMON_TIKV_VERSIONMAP_H_

#include "IVersionMap.h"
#include "inc/Helper/KeyValueIO.h"
#include <atomic>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <list>

namespace SPTAG
{
    namespace COMMON
    {
        /// TiKVVersionMap stores per-VID version bytes in TiKV as chunks.
        ///
        /// TiKV key schema:
        ///   "vc:{layer}"         → uint64 (vector count)
        ///   "v:{layer}:{chunkId}" → uint8_t[chunkSize]
        ///
        /// Each chunk holds chunkSize VIDs' version bytes.
        /// chunk_id = VID / chunkSize, offset = VID % chunkSize.
        class TiKVVersionMap : public IVersionMap
        {
        private:
            std::shared_ptr<Helper::KeyValueIO> m_db;
            int m_layer;
            int m_chunkSize;
            std::atomic<SizeType> m_count{0};
            std::atomic<SizeType> m_deleted{0};

            // Striped mutexes for per-chunk write serialization
            static constexpr int kWriteStripes = 64;
            mutable std::mutex m_chunkWriteMutex[kWriteStripes];
            std::mutex& ChunkMutex(SizeType chunkId) const { return m_chunkWriteMutex[chunkId % kWriteStripes]; }

            // Striped mutexes to avoid duplicate refreshes for the same expired/missing chunk.
            static constexpr int kRefreshStripes = 64;
            mutable std::mutex m_chunkRefreshMutex[kRefreshStripes];
            std::mutex& RefreshMutex(SizeType chunkId) const { return m_chunkRefreshMutex[chunkId % kRefreshStripes]; }

            // LRU chunk cache: list front = most recently used
            struct CachedChunk {
                SizeType chunkId;
                std::string data;
                std::chrono::steady_clock::time_point refreshTime;
            };
            using LruList = std::list<CachedChunk>;
            mutable std::shared_mutex m_cacheMutex;
            mutable LruList m_lruList;
            mutable std::unordered_map<SizeType, LruList::iterator> m_cacheMap;
            int m_cacheTTLMs{0}; // <= 0 means cached chunks do not expire
            int m_cacheMaxChunks{10000}; // max cached chunks; <= 0 disables caching

            bool CacheEnabled() const { return m_cacheMaxChunks > 0; }

            bool CacheFresh(const LruList::iterator& it, std::chrono::steady_clock::time_point now) const
            {
                if (m_cacheTTLMs <= 0) return true;
                auto ageMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->refreshTime).count();
                return ageMs < m_cacheTTLMs;
            }

            // Insert or update a chunk in the LRU cache. Caller must hold exclusive m_cacheMutex.
            void CachePut(SizeType chunkId, const std::string& data, std::chrono::steady_clock::time_point now) const
            {
                if (!CacheEnabled()) return;

                auto it = m_cacheMap.find(chunkId);
                if (it != m_cacheMap.end()) {
                    // Update existing: move to front
                    it->second->data = data;
                    it->second->refreshTime = now;
                    m_lruList.splice(m_lruList.begin(), m_lruList, it->second);
                } else {
                    // Evict LRU entries if at capacity
                    while ((int)m_cacheMap.size() >= m_cacheMaxChunks) {
                        auto& back = m_lruList.back();
                        m_cacheMap.erase(back.chunkId);
                        m_lruList.pop_back();
                    }
                    m_lruList.push_front({chunkId, data, now});
                    m_cacheMap[chunkId] = m_lruList.begin();
                }
            }

            static constexpr auto MaxTimeout = std::chrono::microseconds(60000000); // 60s

            std::string CountKey() const
            {
                return "vc:" + std::to_string(m_layer);
            }

            std::string ChunkKey(SizeType chunkId) const
            {
                return "v:" + std::to_string(m_layer) + ":" + std::to_string(chunkId);
            }

            SizeType ChunkId(SizeType vid) const { return vid / m_chunkSize; }
            int ChunkOffset(SizeType vid) const { return vid % m_chunkSize; }

            // Read a single chunk from TiKV. Returns empty string on miss.
            std::string ReadChunk(SizeType chunkId) const
            {
                std::string value;
                auto ret = m_db->Get(ChunkKey(chunkId), &value, MaxTimeout, nullptr);
                if (ret != ErrorCode::Success || value.empty()) {
                    return std::string();
                }
                return value;
            }

            // Write a chunk to TiKV and update LRU cache.
            ErrorCode WriteChunk(SizeType chunkId, const std::string& data)
            {
                auto ret = m_db->Put(ChunkKey(chunkId), data, MaxTimeout, nullptr);
                if (ret == ErrorCode::Success) {
                    std::unique_lock<std::shared_mutex> lock(m_cacheMutex);
                    CachePut(chunkId, data, std::chrono::steady_clock::now());
                }
                return ret;
            }

            // Read a chunk with LRU cache and optional TTL refresh.
            // Uses shared_lock for cache hits (no LRU reorder) to allow concurrent reads.
            // Only takes exclusive lock on cache miss for insertion.
            std::string ReadChunkCached(SizeType chunkId) const
            {
                if (!CacheEnabled()) return ReadChunk(chunkId);

                auto now = std::chrono::steady_clock::now();

                // Try cache with shared lock — concurrent reads OK
                {
                    std::shared_lock<std::shared_mutex> lock(m_cacheMutex);
                    auto it = m_cacheMap.find(chunkId);
                    if (it != m_cacheMap.end() && CacheFresh(it->second, now)) {
                        return it->second->data;
                    }
                }

                std::lock_guard<std::mutex> refreshLock(RefreshMutex(chunkId));
                now = std::chrono::steady_clock::now();

                // Another thread may have refreshed this chunk while we waited.
                {
                    std::shared_lock<std::shared_mutex> lock(m_cacheMutex);
                    auto it = m_cacheMap.find(chunkId);
                    if (it != m_cacheMap.end() && CacheFresh(it->second, now)) {
                        return it->second->data;
                    }
                }

                // Cache miss or expired — fetch from TiKV, then exclusive lock to insert
                std::string data = ReadChunk(chunkId);
                if (!data.empty()) {
                    std::unique_lock<std::shared_mutex> lock(m_cacheMutex);
                    CachePut(chunkId, data, now);
                }
                return data;
            }

            // Read a single byte for a VID. Returns 0xfe on error/miss.
            uint8_t ReadVersionByte(SizeType vid, VersionReadPolicy policy = VersionReadPolicy::UseCache) const
            {
                SizeType cid = ChunkId(vid);
                std::string chunk = (policy == VersionReadPolicy::BypassCacheNoFill) ?
                    ReadChunk(cid) : ReadChunkCached(cid);
                if (chunk.empty() || (int)chunk.size() <= ChunkOffset(vid)) {
                    return 0xfe;
                }
                return static_cast<uint8_t>(chunk[ChunkOffset(vid)]);
            }

            // Read-modify-write a single byte. Returns the old value via oldVal.
            // Returns true on success, false if TiKV write failed.
            // Thread-safe: locks the chunk stripe to prevent concurrent overwrites.
            bool WriteVersionByte(SizeType vid, uint8_t newVal, uint8_t& oldVal)
            {
                SizeType cid = ChunkId(vid);
                int offset = ChunkOffset(vid);
                std::lock_guard<std::mutex> lock(ChunkMutex(cid));
                std::string chunk = ReadChunk(cid);
                if (chunk.empty()) {
                    // Create new chunk, uninitialized (matching VersionLabel's 0xff)
                    chunk.assign(m_chunkSize, static_cast<char>(0xff));
                }
                oldVal = static_cast<uint8_t>(chunk[offset]);
                chunk[offset] = static_cast<char>(newVal);
                auto ret = WriteChunk(cid, chunk);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "TiKVVersionMap::WriteVersionByte: WriteChunk failed vid=%d chunk=%d layer=%d\n",
                        vid, cid, m_layer);
                    return false;
                }
                return true;
            }

            void SaveCount()
            {
                SizeType count = m_count.load();
                std::string val(reinterpret_cast<const char*>(&count), sizeof(SizeType));
                m_db->Put(CountKey(), val, MaxTimeout, nullptr);
            }

        public:
            TiKVVersionMap() : m_layer(0), m_chunkSize(4096) {}

            void SetDB(std::shared_ptr<Helper::KeyValueIO> db) { m_db = db; }
            void SetLayer(int layer) { m_layer = layer; }
            void SetChunkSize(int chunkSize) { m_chunkSize = chunkSize; }
            void SetCacheTTL(int ttlMs) { m_cacheTTLMs = ttlMs; }
            void SetCacheMaxChunks(int maxChunks) { m_cacheMaxChunks = maxChunks; }

            std::shared_ptr<Helper::KeyValueIO> GetDB() const { return m_db; }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity, COMMON::Dataset<SizeType>* globalIDs = nullptr) override
            {
                m_count = size;

                SizeType totalChunks = (size + m_chunkSize - 1) / m_chunkSize;

                if (m_layer > 0 && globalIDs != nullptr && globalIDs->R() > 0) {
                    // Non-leaf layer: only globalIDs are alive, rest deleted
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVVersionMap::Initialize layer=%d: initializing non-leaf with size=%d, totalChunks=%d, globalIDs=%d\n",
                        m_layer, size, totalChunks, globalIDs->R());

                    std::string defaultChunk(m_chunkSize, static_cast<char>(0xfe));
                    SizeType writtenChunks = 0;
                    for (SizeType c = 0; c < totalChunks; c++) {
                        auto ret = WriteChunk(c, defaultChunk);
                        if (ret == ErrorCode::Success) writtenChunks++;
                        else if (c < 5) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "TiKVVersionMap::Initialize: failed to write default chunk (layer=%d, chunk=%d, ret=%d)\n",
                                m_layer, c, static_cast<int>(ret));
                        }
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVVersionMap::Initialize layer=%d: wrote %d/%d default chunks (all-deleted)\n",
                        m_layer, writtenChunks, totalChunks);

                    m_deleted = size;

                    // Mark vectors in globalIDs as version 0 (not deleted)
                    std::unordered_map<SizeType, std::string> dirtyChunks;
                    SizeType markedAlive = 0;
                    for (SizeType i = 0; i < globalIDs->R(); i++) {
                        SizeType globalID = *(globalIDs->At(i));
                        SizeType cid = ChunkId(globalID);
                        if (dirtyChunks.find(cid) == dirtyChunks.end()) {
                            dirtyChunks[cid] = ReadChunk(cid);
                            if (dirtyChunks[cid].empty()) {
                                if (i < 50) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                        "TiKVVersionMap::Initialize layer=%d: ReadChunk(%d) returned empty; reinitializing locally\n",
                                        m_layer, cid);
                                }
                                dirtyChunks[cid].assign(m_chunkSize, static_cast<char>(0xfe));
                            }
                        }
                        uint8_t oldVal = static_cast<uint8_t>(dirtyChunks[cid][ChunkOffset(globalID)]);
                        dirtyChunks[cid][ChunkOffset(globalID)] = 0x00;
                        markedAlive++;
                        if (oldVal == 0xfe) m_deleted--;
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVVersionMap::Initialize layer=%d: marked %d globalIDs alive, %d dirty chunks to flush\n",
                        m_layer, markedAlive, static_cast<int>(dirtyChunks.size()));

                    SizeType flushedChunks = 0;
                    for (auto& [cid, data] : dirtyChunks) {
                        auto ret = WriteChunk(cid, data);
                        if (ret == ErrorCode::Success) flushedChunks++;
                        else {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "TiKVVersionMap::Initialize: failed to flush dirty chunk (layer=%d, chunk=%d, ret=%d)\n",
                                m_layer, cid, static_cast<int>(ret));
                        }
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVVersionMap::Initialize layer=%d: flushed %d/%d dirty chunks, m_deleted=%d\n",
                        m_layer, flushedChunks, static_cast<int>(dirtyChunks.size()), m_deleted.load());
                } else {
                    // Leaf layer (layer 0) or no globalIDs: all VIDs start alive (version 0)
                    std::string aliveChunk(m_chunkSize, static_cast<char>(0x00));
                    for (SizeType c = 0; c < totalChunks; c++) {
                        WriteChunk(c, aliveChunk);
                    }
                    m_deleted = 0;
                }

                SaveCount();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVVersionMap::Initialize: layer=%d, size=%d, totalChunks=%d, deleted=%d, globalIDs=%d\n",
                    m_layer, size, totalChunks, m_deleted.load(), (globalIDs && globalIDs->R() > 0) ? globalIDs->R() : 0);
            }

            void DeleteAll() override
            {
                SizeType totalChunks = (m_count.load() + m_chunkSize - 1) / m_chunkSize;
                std::string deletedChunk(m_chunkSize, static_cast<char>(0xfe));
                for (SizeType c = 0; c < totalChunks; c++) {
                    WriteChunk(c, deletedChunk);
                }
                m_deleted = m_count.load();
            }

            SizeType Count() const override { return m_count.load(); }
            SizeType GetDeleteCount() const override { return m_deleted.load(); }
            SizeType GetVectorNum() override { return m_count.load(); }
            std::uint64_t BufferSize() const override { return static_cast<std::uint64_t>(m_count.load()) + sizeof(SizeType); }

            bool Deleted(const SizeType& key) const override
            {
                return Deleted(key, VersionReadPolicy::UseCache);
            }

            bool Deleted(const SizeType& key, VersionReadPolicy policy) const override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::Deleted: invalid key %d (max %d)\n", key, m_count.load());
                    return true;
                }
                return ReadVersionByte(key, policy) == 0xfe;
            }

            bool Delete(const SizeType& key) override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::Delete: invalid key %d (max %d)\n", key, m_count.load());
                    return false;
                }
                uint8_t oldVal;
                if (!WriteVersionByte(key, 0xfe, oldVal)) {
                    return false; // TiKV write failed, already logged
                }
                if (oldVal == 0xfe) {
                    if (key < 10) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVVersionMap::Delete: key %d already deleted (layer=%d, chunk=%d, offset=%d)\n",
                                     key, m_layer, ChunkId(key), ChunkOffset(key));
                    }
                    return false;
                }
                m_deleted++;
                return true;
            }

            uint8_t GetVersion(const SizeType& key) override
            {
                return GetVersion(key, VersionReadPolicy::UseCache);
            }

            uint8_t GetVersion(const SizeType& key, VersionReadPolicy policy) override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::GetVersion: invalid key %d (max %d)\n", key, m_count.load());
                    return 0xfe;
                }
                return ReadVersionByte(key, policy);
            }

            void SetVersion(const SizeType& key, const uint8_t& version) override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::SetVersion: invalid key %d (max %d)\n", key, m_count.load());
                    return;
                }
                uint8_t oldVal;
                if (!WriteVersionByte(key, version, oldVal)) {
                    return; // TiKV write failed, already logged
                }
                if (oldVal == 0xfe && version != 0xfe) m_deleted--;
                else if (oldVal != 0xfe && version == 0xfe) m_deleted++;
            }

            // Group writes by chunk: 1 ReadChunk + N byte-modifications + 1 WriteChunk
            // per chunk, instead of N × (ReadChunk + WriteChunk). Bypasses the LRU
            // cache because runs that exercise this path always have
            // VersionCacheMaxChunks=0; reading TiKV directly removes a layer of
            // bookkeeping (cache invalidate-on-write) we no longer benefit from.
            void SetVersionBatch(const std::vector<SizeType>& vids, const std::vector<uint8_t>& versions) override
            {
                size_t n = std::min(vids.size(), versions.size());
                if (n == 0) return;
                const SizeType localCount = m_count.load();

                // Group (idx into vids/versions) by chunk id.
                std::unordered_map<SizeType, std::vector<size_t>> byChunk;
                byChunk.reserve(n);
                for (size_t i = 0; i < n; i++) {
                    SizeType vid = vids[i];
                    if (vid < 0 || vid >= localCount) continue;
                    byChunk[ChunkId(vid)].push_back(i);
                }
                if (byChunk.empty()) return;

                long deletedDelta = 0;
                for (auto& kv : byChunk) {
                    SizeType cid = kv.first;
                    auto& idxs = kv.second;
                    std::lock_guard<std::mutex> lock(ChunkMutex(cid));
                    std::string chunk = ReadChunk(cid);
                    if (chunk.empty()) {
                        chunk.assign(m_chunkSize, static_cast<char>(0xff));
                    }
                    bool dirty = false;
                    for (size_t i : idxs) {
                        SizeType vid = vids[i];
                        uint8_t newVal = versions[i];
                        int offset = ChunkOffset(vid);
                        if (offset < 0 || offset >= (int)chunk.size()) continue;
                        uint8_t oldVal = static_cast<uint8_t>(chunk[offset]);
                        if (oldVal == newVal) continue;
                        if (oldVal == 0xfe && newVal != 0xfe) deletedDelta--;
                        else if (oldVal != 0xfe && newVal == 0xfe) deletedDelta++;
                        chunk[offset] = static_cast<char>(newVal);
                        dirty = true;
                    }
                    if (dirty) {
                        auto ret = WriteChunk(cid, chunk);
                        if (ret != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "TiKVVersionMap::SetVersionBatch: WriteChunk failed chunk=%d layer=%d\n",
                                cid, m_layer);
                        }
                    }
                }
                if (deletedDelta != 0) m_deleted += deletedDelta;
            }

            bool IncVersion(const SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::IncVersion: invalid key %d (max %d)\n", key, m_count.load());
                    return false;
                }

                const int MAX_RETRIES = 3;
                for (int retry = 0; retry < MAX_RETRIES; retry++) {
                    SizeType cid = ChunkId(key);
                    int offset = ChunkOffset(key);
                    std::lock_guard<std::mutex> lock(ChunkMutex(cid));
                    std::string chunk = ReadChunk(cid);
                    if (chunk.empty()) return false;

                    uint8_t current = static_cast<uint8_t>(chunk[offset]);
                    if (current == 0xfe) return false; // deleted

                    uint8_t target;
                    if (expectedOld != 0xff) {
                        target = (expectedOld + 1) & 0x7f;
                        // If already at target, another node did the same increment → success
                        if (current == target) {
                            *newVersion = target;
                            return true;
                        }
                        // If not at expected old, unexpected state → conflict
                        if (current != expectedOld) {
                            return false;
                        }
                    } else {
                        target = (current + 1) & 0x7f;
                    }

                    chunk[offset] = static_cast<char>(target);
                    // TODO: Replace with RawCompareAndSwap when available in kvproto
                    // for true atomic CAS across nodes. For now, best-effort write.
                    ErrorCode ret = WriteChunk(cid, chunk);
                    if (ret == ErrorCode::Success) {
                        *newVersion = target;
                        return true;
                    }
                    // Write failed, retry
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "TiKVVersionMap::IncVersion: write failed for key %d, retry %d\n", key, retry);
                }
                return false;
            }

            ErrorCode AddBatch(SizeType num) override
            {
                return AddBatch(num, false);
            }

            ErrorCode AddBatch(SizeType num, bool deleted) override
            {
                if (num <= 0) return ErrorCode::Success;

                SizeType oldCount = m_count.load();
                SizeType newCount = oldCount + num;

                // Create any new chunks needed (init to 0xff = uninitialized, matching VersionLabel)
                SizeType oldLastChunk = (oldCount > 0) ? ChunkId(oldCount - 1) : -1;
                SizeType newLastChunk = ChunkId(newCount - 1);

                if (!deleted) {
                    for (SizeType c = oldLastChunk + 1; c <= newLastChunk; c++) {
                        std::string newChunk(m_chunkSize, static_cast<char>(0xff));
                        WriteChunk(c, newChunk);
                    }
                } else {
                    SizeType firstChunk = ChunkId(oldCount);
                    for (SizeType c = firstChunk; c <= newLastChunk; c++) {
                        std::lock_guard<std::mutex> lock(ChunkMutex(c));
                        std::string chunk = (c <= oldLastChunk) ? ReadChunk(c) : std::string();
                        if (chunk.empty()) {
                            chunk.assign(m_chunkSize, static_cast<char>(0xff));
                        }

                        int beginOffset = (c == firstChunk) ? ChunkOffset(oldCount) : 0;
                        int endOffset = (c == newLastChunk) ? ChunkOffset(newCount - 1) + 1 : m_chunkSize;
                        std::fill(chunk.begin() + beginOffset, chunk.begin() + endOffset, static_cast<char>(0xfe));

                        WriteChunk(c, chunk);
                    }
                    m_deleted.fetch_add(num, std::memory_order_relaxed);
                }

                m_count = newCount;
                SaveCount();
                return ErrorCode::Success;
            }

            void SetR(SizeType num) override
            {
                m_count = num;
                SaveCount();
            }

            // Save/Load: For TiKV mode, data is already persisted in TiKV.
            // These are no-ops for TiKV mode.
            ErrorCode Save(std::shared_ptr<Helper::DiskIO> output) override
            {
                SaveCount();
                return ErrorCode::Success;
            }

            ErrorCode Save(const std::string& filename) override
            {
                SaveCount();
                return ErrorCode::Success;
            }

            ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity) override
            {
                // Load count from TiKV
                return LoadCountFromTiKV();
            }

            ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity) override
            {
                return LoadCountFromTiKV();
            }

            ErrorCode Load(char* pmemoryFile, SizeType blockSize, SizeType capacity) override
            {
                return LoadCountFromTiKV();
            }

            /// Batch version lookup with local cache support.
            /// Checks cache first, only fetches misses from TiKV via BatchGet.
            void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions) override
            {
                BatchGetVersions(vids, versions, VersionReadPolicy::UseCache);
            }

            void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions, VersionReadPolicy policy) override
            {
                versions.resize(vids.size());
                if (vids.empty()) return;

                SizeType count = m_count.load();
                bool bypassCache = (policy == VersionReadPolicy::BypassCacheNoFill);
                bool cacheEnabled = (CacheEnabled() && !bypassCache);
                auto now = std::chrono::steady_clock::now();

                // Group VIDs by chunk
                std::unordered_map<SizeType, std::vector<size_t>> chunkToIndices;
                for (size_t i = 0; i < vids.size(); i++) {
                    if (vids[i] < 0 || vids[i] >= count) {
                        versions[i] = 0xfe;
                    } else {
                        chunkToIndices[ChunkId(vids[i])].push_back(i);
                    }
                }

                // Phase 1: Resolve from cache (shared lock, no LRU reorder), collect misses
                // Copy data out to avoid dangling pointers after lock release.
                std::unordered_map<SizeType, std::string> resolvedChunks;
                std::vector<SizeType> missChunkIds;

                if (cacheEnabled) {
                    std::shared_lock<std::shared_mutex> lock(m_cacheMutex);
                    for (auto& [cid, indices] : chunkToIndices) {
                        auto it = m_cacheMap.find(cid);
                        if (it != m_cacheMap.end() && CacheFresh(it->second, now)) {
                            resolvedChunks[cid] = it->second->data; // copy
                            continue;
                        }
                        missChunkIds.push_back(cid);
                    }
                } else {
                    for (auto& [cid, indices] : chunkToIndices) {
                        missChunkIds.push_back(cid);
                    }
                }

                // Phase 2: BatchGet cache misses from TiKV
                std::vector<std::string> fetchedValues;
                std::unordered_map<SizeType, std::string> fetchedChunks;
                if (!missChunkIds.empty()) {
                    std::vector<std::string> keys;
                    keys.reserve(missChunkIds.size());
                    for (SizeType cid : missChunkIds) {
                        keys.push_back(ChunkKey(cid));
                    }
                    auto batchRet = m_db->MultiGet(keys, &fetchedValues, MaxTimeout, nullptr);
                    if (batchRet != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                     "TiKVVersionMap::BatchGetVersions: MultiGet failed (layer=%d, ret=%d, missChunks=%d, vids=%d). Missing versions are treated as deleted (0xfe).\n",
                                     m_layer, static_cast<int>(batchRet), static_cast<int>(missChunkIds.size()), static_cast<int>(vids.size()));
                    }

                    for (size_t i = 0; i < missChunkIds.size(); i++) {
                        if (i < fetchedValues.size() && !fetchedValues[i].empty()) {
                            fetchedChunks[missChunkIds[i]] = std::move(fetchedValues[i]);
                        }
                    }

                    SizeType missingChunks = static_cast<SizeType>(missChunkIds.size() - fetchedChunks.size());
                    if (missingChunks > 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                     "TiKVVersionMap::BatchGetVersions: missing chunk data after MultiGet (layer=%d, missing=%d/%d, sampleChunk=%d). Missing versions are treated as deleted (0xfe).\n",
                                     m_layer, missingChunks, static_cast<int>(missChunkIds.size()), missChunkIds[0]);
                    }

                    // Update LRU cache with fetched chunks (exclusive lock). Bypass reads never fill cache.
                    if (cacheEnabled && !fetchedChunks.empty()) {
                        std::unique_lock<std::shared_mutex> lock(m_cacheMutex);
                        for (auto& [cid, data] : fetchedChunks) {
                            CachePut(cid, data, now);
                        }
                    }
                }

                // Phase 3: Resolve all VIDs
                SizeType fallbackDeletedCount = 0;
                for (auto& [cid, indices] : chunkToIndices) {
                    const std::string* chunkData = nullptr;

                    // Check resolved (copied from cache)
                    auto rit = resolvedChunks.find(cid);
                    if (rit != resolvedChunks.end()) {
                        chunkData = &rit->second;
                    } else {
                        // Check fetched
                        auto fit = fetchedChunks.find(cid);
                        if (fit != fetchedChunks.end()) {
                            chunkData = &fit->second;
                        }
                    }

                    for (size_t idx : indices) {
                        if (chunkData == nullptr) {
                            versions[idx] = 0xfe;
                            fallbackDeletedCount++;
                        } else {
                            int offset = ChunkOffset(vids[idx]);
                            if (offset < (int)chunkData->size()) {
                                versions[idx] = static_cast<uint8_t>((*chunkData)[offset]);
                            } else {
                                versions[idx] = 0xfe;
                                fallbackDeletedCount++;
                            }
                        }
                    }
                }
                if (fallbackDeletedCount > 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "TiKVVersionMap::BatchGetVersions: fallback-to-deleted count=%d/%d (layer=%d).\n",
                                 fallbackDeletedCount, static_cast<int>(vids.size()), m_layer);
                }
            }

        private:
            ErrorCode LoadCountFromTiKV()
            {
                std::string val;
                auto ret = m_db->Get(CountKey(), &val, MaxTimeout, nullptr);
                if (ret == ErrorCode::Success && val.size() >= sizeof(SizeType)) {
                    m_count = *reinterpret_cast<const SizeType*>(val.data());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVVersionMap: Loaded count=%d from TiKV\n", m_count.load());
                } else {
                    m_count = 0;
                    if (ret != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                     "TiKVVersionMap: failed to read count key '%s' (layer=%d, ret=%d). Set m_count=0; subsequent GetVersion may return 0xfe for all VIDs.\n",
                                     CountKey().c_str(), m_layer, static_cast<int>(ret));
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                     "TiKVVersionMap: invalid count value for key '%s' (layer=%d, valueSize=%d, expected>=%d). Set m_count=0; subsequent GetVersion may return 0xfe for all VIDs.\n",
                                     CountKey().c_str(), m_layer, static_cast<int>(val.size()), static_cast<int>(sizeof(SizeType)));
                    }
                }
                // Scan all chunks to compute accurate delete count
                SizeType count = m_count.load();
                SizeType deleted = 0;
                if (count > 0) {
                    SizeType totalChunks = (count + m_chunkSize - 1) / m_chunkSize;
                    for (SizeType c = 0; c < totalChunks; c++) {
                        std::string chunk = ReadChunk(c);
                        if (chunk.empty()) {
                            // Missing chunk — treat all entries as deleted
                            SizeType chunkEntries = (c == totalChunks - 1) ? (count - c * m_chunkSize) : m_chunkSize;
                            deleted += chunkEntries;
                            continue;
                        }
                        SizeType chunkEntries = (c == totalChunks - 1) ? (count - c * m_chunkSize) : m_chunkSize;
                        for (SizeType i = 0; i < chunkEntries; i++) {
                            if (static_cast<uint8_t>(chunk[i]) == 0xfe) deleted++;
                        }
                    }
                }
                m_deleted = deleted;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TiKVVersionMap: Scanned %d chunks, deleted=%d/%d\n",
                             count > 0 ? (count + m_chunkSize - 1) / m_chunkSize : 0, deleted, count);
                return ErrorCode::Success;
            }
        };
    }
}

#endif // _SPTAG_COMMON_TIKV_VERSIONMAP_H_
