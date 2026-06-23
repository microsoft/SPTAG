// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_PW_COREINTERFACE_H_
#define _SPTAG_PW_COREINTERFACE_H_

#include "TransferDataType.h"
#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/ExtraFileController.h"
#include "inc/Core/SPANN/Options.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Core/Cache/HeadIndexCache.h"
#include "inc/Core/Cache/PostingSignature.h"
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <list>
#include <atomic>
#include <tuple>
#include <unordered_map>

typedef int SizeType;
typedef int DimensionType;

class AnnIndex
{
public:
    AnnIndex(DimensionType p_dimension);

    AnnIndex(const char* p_algoType, const char* p_valueType, DimensionType p_dimension);

    ~AnnIndex();

    void SetBuildParam(const char* p_name, const char* p_value, const char* p_section);

    void SetSearchParam(const char* p_name, const char* p_value, const char* p_section);

    bool LoadQuantizer(const char* p_quantizerFile);

    void SetQuantizerADC(bool p_adc);

    ByteArray QuantizeVector(ByteArray p_data, int p_num);

    ByteArray ReconstructVector(ByteArray p_data, int p_num);

    bool BuildSPANN(bool p_normalized);

    bool BuildSPANNWithMetaData(ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized);

    bool BuildSPANNWithDataAndMeta(ByteArray p_data, ByteArray p_meta, SizeType p_num,
                                    bool p_withMetaIndex, bool p_normalized);

    bool Build(ByteArray p_data, SizeType p_num, bool p_normalized);

    bool BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized);

    std::shared_ptr<ResultIterator> GetIterator(ByteArray p_target);

    std::shared_ptr<QueryResult> Search(ByteArray p_data, int p_resultNum);

    std::shared_ptr<QueryResult> SearchWithMetaData(ByteArray p_data, int p_resultNum);

    std::shared_ptr<QueryResult> BatchSearch(ByteArray p_data, int p_vectorNum, int p_resultNum, bool p_withMetaData);

    std::shared_ptr<QueryResult> SearchWithTenantFilter(ByteArray p_data, int p_resultNum, const char* p_tenantId);

    std::shared_ptr<QueryResult> BatchSearchWithTenantFilter(ByteArray p_data, int p_vectorNum, int p_resultNum, const char* p_tenantId);

    bool ReadyToServe() const;

    // Inject a shared KeyValueIO (e.g., a Helper::TenantPrefixedKeyValueIO around
    // a shared RocksDB) into the underlying SPANN index. Must be called BEFORE
    // Build/Load. Returns true if the underlying index is SPANN<Float> and the
    // injection succeeded.
    bool SetSharedDB(std::shared_ptr<SPTAG::Helper::KeyValueIO> p_db);

    void UpdateIndex();

    bool Save(const char* p_saveFile) const;

    bool Add(ByteArray p_data, SizeType p_num, bool p_normalized);

    bool AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized);

    bool Delete(ByteArray p_data, SizeType p_num);

    bool DeleteByMetaData(ByteArray p_meta);

    uint64_t CalculateBufferSize();

    ByteArray Dump(ByteArray p_blobs);

    static AnnIndex LoadFromDump(ByteArray p_config, ByteArray p_blobs);

    static AnnIndex Load(const char* p_loaderFile);

    static AnnIndex Merge(const char* p_indexFilePath1, const char* p_indexFilePath2);

    std::shared_ptr<SPTAG::VectorIndex> GetInternalIndex() const { return m_index; }

    // Set per-vector tags for embedding in SPANN posting metadata
    void SetVectorTags(const uint32_t* tags, int numVecs, int numTagsPerVec);

    // Set build-time node->vector assignments for SPANN posting construction
    void SetNodeVectorAssignments(const std::vector<std::vector<int>>& nodeVectorAssignments);

    // Set build-time primary node ownership for SPANN head construction
    void SetPrimaryNodeVectorAssignments(const std::vector<std::vector<int>>& primaryNodeVectorAssignments);

    // When set, the next Build() borrows the caller's vector buffer instead of
    // copying it into the core (passes p_shareOwnership=true to BuildIndex). The
    // caller must keep the buffer alive for the duration of the build. Used by
    // TenantIndexManager for the single contiguous tenant (1B-scale) build to
    // avoid a full duplicate of the vector data.
    void SetShareBuildOwnership(bool p_share) { m_shareBuildOwnership = p_share; }

    // Wrap an already-built/loaded VectorIndex (used internally and by
    // TenantIndexManager when injecting a shared RocksDB during load).
    AnnIndex(const std::shared_ptr<SPTAG::VectorIndex>& p_index);

private:
    std::shared_ptr<SPTAG::VectorIndex> m_index;

    size_t m_inputVectorSize;
    
    DimensionType m_dimension;

    SPTAG::IndexAlgoType m_algoType;

    SPTAG::VectorValueType m_inputValueType;

    bool m_shareBuildOwnership = false;
};

// Per-tenant index strategy based on data size
enum class TenantIndexType : uint8_t {
    SPANN = 0,    // Large tenants: full SPANN with SSD posting lists
    BKT = 1,      // Medium tenants: in-memory BKT graph index
    BRUTEFORCE = 2 // Small tenants: linear scan
};

// Cached tenant entry for unified LRU cache management
// Holds either a SPANN HeadIndex, a BKT in-memory index, or raw vectors for brute force
struct CachedTenantEntry {
    int tenantId;
    TenantIndexType indexType;
    std::shared_ptr<AnnIndex> index;       // For BKT / brute force: in-memory index
    std::shared_ptr<SPTAG::VectorIndex> headIndex; // For SPANN: head index only
    size_t sizeBytes;                      // Memory footprint for LRU eviction
};

// Multi-tenant index manager: manages separate indices for each tenant
// Storage strategy:
//   - Hybrid per-tenant: SPANN (large), BKT (medium), BruteForce (small)
//   - Unified KV storage for all tenant data (postings + vectors)
//   - Unified LRU cache for in-memory tenant index management
class TenantIndexManager
{
public:
    TenantIndexManager(DimensionType p_dimension, const char* p_algoType = "BKT", const char* p_valueType = "Float");

    ~TenantIndexManager();
    struct TagRoutingStats {
        int vectorCount = 0;
        int postingCount = 0;
    };


    // Build indices from global vectors and metadata (metadata are tenant IDs as integers)
    // Returns true on success, false otherwise
    bool BuildFromData(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum, 
                      bool p_withMetaIndex, bool p_normalized);

    // --- String tenant ID support ---
    // Register a string tenant ID, returns its internal integer ID
    int RegisterTenantId(const char* p_tenantStr);
    // Get internal ID for a string tenant ID (-1 if not found)
    int GetInternalTenantId(const char* p_tenantStr) const;
    // Get string tenant ID from internal ID (nullptr if not found)
    const char* GetTenantIdStr(int p_internalId) const;

    // Search by string tenant ID
    std::shared_ptr<QueryResult> SearchByTenant(ByteArray p_queryVector, const char* p_tenantStr, int p_resultNum);

    // Search within a specific tenant (internal integer ID)
    // Returns QueryResult or nullptr on failure
    std::shared_ptr<QueryResult> Search(ByteArray p_queryVector, int p_tenantId, int p_resultNum);

    // Batch search within a specific tenant  
    // Returns QueryResult with results for all queries or nullptr on failure
    std::shared_ptr<QueryResult> BatchSearch(ByteArray p_queryVectors, int p_vectorNum, 
                                            int p_tenantId, int p_resultNum);

    // Multi-tenant batch search: each query can target a different tenant.
    // p_queryVectors: N query vectors concatenated.
    // p_tenantIds: ByteArray of N int32 tenant IDs (one per query).
    // Groups queries by tenant, dispatches BatchSearch per tenant in parallel,
    // returns results in original query order.
    std::shared_ptr<QueryResult> MultiBatchSearch(ByteArray p_queryVectors, int p_vectorNum,
                                                   ByteArray p_tenantIds, int p_resultNum);

    // Get list of tenant IDs (sorted)
    void GetTenantIds(int* p_tenants, int* p_count) const;

    // Get number of tenants
    int GetTenantCount() const;

    // Save all tenant indices to a base directory (unified storage)
    bool SaveAll(const char* p_baseDir);

    // Load all tenant indices from a base directory (unified storage)
    bool LoadAll(const char* p_baseDir);

    // Get vector count for a specific tenant
    int GetTenantVectorCount(int p_tenantId) const;

    // Get on-disk HeadIndex size for a specific tenant in bytes.
    // Returns 0 when the tenant has no HeadIndex workdir.
    uint64_t GetTenantHeadIndexSize(int p_tenantId) const;

    // Get exact tag routing stats for a tenant as a packed byte buffer.
    // Each entry uses the layout: uint32_t tag, int32_t vectorCount, int32_t postingCount.
    ByteArray GetTagRoutingStatsBlob(int p_tenantId) const;

    // Build-time pivot planner cost estimator.
    // Returns a UTF-8 JSON payload with all candidates and the selected best plan.
    // p_tags layout: [p_numVectors * p_numTagsPerVec] uint32_t.
    // p_levelWeightsCsv: comma-separated weights for each tag level, empty = uniform.
    ByteArray EstimatePivotBuildPlan(ByteArray p_tags,
                                     int p_numVectors,
                                     int p_numTagsPerVec,
                                     int p_maxNodes,
                                     float p_recallTarget,
                                     float p_lambdaRecall,
                                     float p_estimatedRecall,
                                     ByteArray p_levelWeightsCsv) const;

    // Set build/search parameters for all tenant indices
    void SetBuildParam(const char* p_name, const char* p_value, const char* p_section);
    void SetSearchParam(const char* p_name, const char* p_value, const char* p_section);

    // Set HeadIndex LRU cache size limit (in bytes). 0 = unlimited.
    void SetHeadIndexCacheLimit(uint64_t p_bytesLimit);

    // Set/get the on-disk -> loaded-memory safety factor used for cache accounting.
    // Values are clamped to [1.0, 8.0].
    void SetHeadIndexCacheSafetyFactor(double p_factor);
    double GetHeadIndexCacheSafetyFactor() const;

    // Get current estimated HeadIndex cache usage (bytes)
    uint64_t GetHeadIndexCacheUsage() const;

    // Observe current process RSS in bytes. Returns 0 if unavailable on the platform.
    uint64_t GetCurrentRSSBytes() const;

    // Set/get process RSS high-water mark in bytes. 0 disables reject-on-high-water.
    void SetRSSHighWaterMark(uint64_t p_bytesLimit);
    uint64_t GetRSSHighWaterMark() const;

    // Last ACL search posting-level stats on the current thread.
    uint64_t GetLastPostingReadCount() const;
    uint64_t GetLastPostingMatchCount() const;
    uint64_t GetLastPostingFP() const;
    uint64_t GetLastPostingPrePS() const;
    uint64_t GetLastScannedVectors() const;
    uint64_t GetLastMatchedVectors() const;

    // Enable/disable dropping OS page cache on HeadIndex eviction (for benchmarking)
    void SetDropPageCacheOnEvict(bool enable) { m_dropPageCacheOnEvict = enable; }

    // Unload a single tenant (release HeadIndex memory + close fd)
    bool UnloadTenant(int p_tenantId);

    // --- ACL/Tag Filtered Search ---
    // Build posting signatures (PS + NS) for a tenant from per-vector tags.
    // p_tags: ByteArray of uint32_t, layout [p_numVectors × p_numTagsPerVec].
    // Each vector can have multiple tags (e.g. org, dept, team, project).
    bool BuildSignatures(int p_tenantId, ByteArray p_tags, int p_numVectors, int p_numTagsPerVec);

    // Build index with per-vector tags integrated.
    // Same as BuildFromData but also generates PS/NS signatures per tenant.
    // p_tags: ByteArray of uint32_t, layout [p_vectorNum × p_numTagsPerVec].
    bool BuildFromDataWithTags(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum,
                               ByteArray p_tags, int p_numTagsPerVec,
                               bool p_withMetaIndex, bool p_normalized);

    // Search with ACL tag filter: PS hard reject + NS soft navigation.
    // p_queryTags: ByteArray of uint32_t allowed tag IDs.
    std::shared_ptr<QueryResult> SearchWithACL(ByteArray p_queryVector, int p_tenantId,
                                                int p_resultNum,
                                                ByteArray p_queryTags, int p_numTags);

    // Set posting storage backend: "FILEIO" (default) or "ROCKSDBIO"
    void SetStorageBackend(const char* backend) { m_storageBackend = std::string(backend); }

    // Toggle the shared-RocksDB code path. When true (and storage backend is
    // ROCKSDBIO), all SPANN tenants are routed through a single shared
    // RocksDB instance via Helper::TenantPrefixedKeyValueIO. Default false.
    void SetUseSharedDB(bool p_use) { m_useSharedDB = p_use; }
    void SetUseDirectIO(bool p_use) { m_useDirectIO = p_use; }
    void SetEnableWAL(bool p_enable) { m_enableWAL = p_enable; }

    // Stage an extra [BuildSSDIndex] parameter (name/value as in indexloader.ini)
    // to be applied before the SPANN build so it persists to the index config.
    // Used to configure the in-posting quantization knobs (PostingQuantizer,
    // PostingQuantM, QuantizeHead, ...) without environment variables.
    void SetSSDBuildParam(const char* p_name, const char* p_value) {
        m_extraSSDBuildParams.emplace_back(std::string(p_name), std::string(p_value));
    }

private:
    DimensionType m_dimension;
    SPTAG::IndexAlgoType m_algoType;
    SPTAG::VectorValueType m_valueType;
    size_t m_inputVectorSize;

    // tenant_id -> AnnIndex mapping (in-memory indices)
    // Protected by m_tenantIndicesMutex for concurrent multi-tenant search
    std::map<int, std::shared_ptr<AnnIndex>> m_tenantIndices;
    mutable std::shared_mutex m_tenantIndicesMutex;

    // LRU tracking: most-recently-used tenant IDs (front=LRU, back=MRU)
    std::list<int> m_lruList;
    std::unordered_map<int, std::list<int>::iterator> m_lruMap;
    uint64_t m_loadedHeadIndexBytes = 0;  // current estimated total loaded HeadIndex size
    std::map<int, uint64_t> m_tenantHeadIndexAccountedBytes;

    // Per-tenant sparse tag index: tag → [posting_ids] for low-selectivity tags
    std::map<int, std::shared_ptr<SPTAG::Cache::SparseTagIndex>> m_tenantSparseIdx;

    // Per-tenant tag-pure postings (chunked, stored inside the same KV/FileIO
    // backend that holds the regular SPANN postings — reuses its cache).
    // Each metadata entry carries the chunk-key list + per-chunk entry counts;
    // payload bytes live in the KV store under keys [numHeads, numHeads+N).
    std::map<int, std::unordered_map<uint32_t,
        std::shared_ptr<SPTAG::Cache::TagPurePosting>>> m_tenantTagPurePostings;

    // Cached KV-store handle per tenant for the tag-pure path. Populated at
    // BuildSignatures time. Used by the SearchWithACL fast path to issue
    // MultiGet without re-resolving the SPANN index pointer chain.
    std::map<int, std::shared_ptr<SPTAG::Helper::KeyValueIO>> m_tenantTagPureKV;

    // Per-tenant key cursor for tag-pure chunks; monotonically advances each
    // BuildSignatures call (no reuse — drops the old keys from the LRU).
    std::map<int, SPTAG::SizeType> m_tenantTagPureNextKey;

    // Per-tenant FileIO page budget snapshot (postingPageLimit + bufferLength + 1
    // pages and the per-chunk page count). Used to size the per-query
    // ExtraWorkSpace used by MultiGet on the fast path.
    std::map<int, int> m_tenantTagPureBlockLimit;
    std::map<int, int> m_tenantTagPurePagesPerChunk;

    // Temporary storage during BuildFromDataWithTags
    ByteArray m_buildTags;
    int m_buildNumTagsPerVec = 0;
    std::map<int, std::vector<int>> m_tenantGlobalIndices;  // tenantId → [global vector indices]
    
    // tenant_id -> vector count mapping  
    std::map<int, int> m_tenantVectorCounts;

    // Exact per-tag routing stats computed by BuildSignatures.
    std::map<int, std::unordered_map<uint32_t, TagRoutingStats>> m_tenantTagRoutingStats;

    // Per-level tag value offsets (minimum tag value at each hierarchical
    // level/column), computed by BuildSignatures and persisted to
    // tag_level_offsets.bin. Used at query time to map a raw tag value back to
    // its hierarchical level (0=org,1=dept,2=team,...). Without this, the query
    // path cannot know which level a single ACL tag belongs to.
    std::map<int, std::vector<uint32_t>> m_tenantTagLevelOffsets;

    // Numeric attribute metadata per tenant: the first numeric tag-column index
    // (numBaseCols) and the per-column quantization domain [lo,hi]. Used by the
    // query path to build the quantized numeric pre-filter. Loaded from
    // numeric_meta.bin; absent => no numeric columns.
    struct NumericMeta {
        int numBaseCols = 0;
        std::vector<SPTAG::Cache::NumQuantParam> params;
    };
    std::map<int, NumericMeta> m_tenantNumericMeta;

    // Build-time pivot plan selected by the estimator.
    std::map<int, int> m_tenantPivotLevels;
    std::map<int, int> m_tenantPivotNodeCounts;
    std::map<int, std::vector<std::vector<uint32_t>>> m_tenantNodePivotTags;
    std::map<int, std::vector<std::vector<int>>> m_tenantPlannedNodeVectors;
    std::map<int, std::vector<std::vector<int>>> m_tenantPlannedPrimaryNodeVectors;

    // Tag -> node routing index derived from the pivot plan.
    // For levels above the pivot, one tag can map to multiple nodes.
    // For pivot and deeper levels, one tag maps to a unique node.
    std::map<int, std::unordered_map<uint32_t, std::vector<int>>> m_tenantTagToNodes;

    // Head sample -> node assignment for the selected pivot partitioning.
    std::map<int, std::vector<int>> m_tenantHeadNodeToNode;

    // Unified storage path (base directory for all tenants)
    std::string m_baseStoragePath;

    // tenant_id -> on-disk index path (for legacy loading/saving)
    std::map<int, std::string> m_tenantIndexPaths;

    // Search params to apply to both currently loaded and future lazy-loaded tenants.
    std::vector<std::tuple<std::string, std::string, std::string>> m_pendingSearchParams;

    // tenant_id -> SPANN build work directory (IndexDirectory)
    std::map<int, std::string> m_tenantSpannWorkDirs;

    // tenant_id -> whether HeadIndex/head_cross_edges.bin exists.
    // Cached on first query to enable cross-subindex routing trim.
    mutable std::unordered_map<int, bool> m_tenantHasCrossEdges;
    mutable std::mutex m_tenantHasCrossEdgesMutex;

    // --- String tenant ID ↔ internal integer ID mapping ---
    std::unordered_map<std::string, int> m_tenantStrToInt;
    std::unordered_map<int, std::string> m_tenantIntToStr;
    int m_nextInternalId = 0;
    mutable std::mutex m_tenantIdMutex;  // Protects tenant ID mapping

    // Posting storage backend
    std::string m_storageBackend = "FILEIO";

    // Extra [BuildSSDIndex] params staged via SetSSDBuildParam, applied before
    // the SPANN build (e.g. in-posting quantization config).
    std::vector<std::pair<std::string, std::string>> m_extraSSDBuildParams;

    // --- Shared RocksDB (multi-tenant) ---
    // When m_useSharedDB is true and m_storageBackend == "ROCKSDBIO", every
    // SPANN tenant is wired to a single shared RocksDB via a per-tenant
    // Helper::TenantPrefixedKeyValueIO wrapper. Lifetime of the underlying
    // RocksDB is owned by the manager.
    bool m_useSharedDB = false;
    bool m_useDirectIO = false;
    bool m_enableWAL = false;
    std::shared_ptr<SPTAG::Helper::KeyValueIO> m_sharedDB;
    mutable std::mutex m_sharedDBMutex;

    bool EnsureSharedDB();
    bool InjectSharedDB(const std::shared_ptr<AnnIndex>& p_idx, int p_internalId);

    // Load a tenant SPANN index from disk while injecting the shared
    // RocksDB BEFORE LoadIndexData (so SPANN's ExtraDynamicSearcher uses the
    // shared store instead of opening its own per-tenant DB). Returns the
    // loaded index wrapped in AnnIndex on success, or an empty AnnIndex on
    // failure.
    std::shared_ptr<AnnIndex> LoadSpannWithSharedDB(const std::string& p_folder, int p_internalId);

    // --- Per-tenant index strategy ---
    std::map<int, TenantIndexType> m_tenantIndexTypes;

    // --- Unified SPANN posting storage ---
    // tenant_id -> posting offset in the shared SSD file
    std::map<int, int> m_tenantPostingOffsets;
    // tenant_id -> number of head vectors (= number of postings for SPANN tenants)
    std::map<int, int> m_tenantHeadCounts;
    // Total posting count across all tenants (SPANN postings + small tenant vector blocks)
    int m_totalPostingCount = 0;
    // Path to the shared SPANN SSD working directory
    std::string m_sharedSpannWorkDir;

    // --- Shared SPANN posting storage ---
    // Shared FileIO (BlockController) for all SPANN tenants
    std::shared_ptr<SPTAG::SPANN::FileIO> m_sharedFileIO;
    // Per-SPANN-tenant HeadIndex (VectorIndex) for routing queries
    std::map<int, std::shared_ptr<SPTAG::VectorIndex>> m_tenantHeadIndices;
    // Per-SPANN-tenant vector translate map (head vector ID → full vector ID)
    std::map<int, std::shared_ptr<SPTAG::COMMON::Dataset<std::uint64_t>>> m_tenantTranslateMaps;

    // Thresholds for hybrid strategy (vector count)
    // All tenants use SPANN — HeadIndex stays in memory, postings read from SSD
    static constexpr int BRUTEFORCE_THRESHOLD = 0;
    static constexpr int BKT_THRESHOLD = 0;  // 0 = all tenants use SPANN

    // --- Concurrent LRU Cache for HeadIndex ---
    // Production S3-FIFO cache with single-flight loading
    std::unique_ptr<SPTAG::Cache::HeadIndexCache> m_headCache;

    // HeadIndex cache limit
    uint64_t m_headIndexCacheLimitBytes;

    // Estimated loaded bytes = on-disk HeadIndex bytes * safety factor.
    double m_headIndexCacheSafetyFactor;

    // Process RSS high-water mark. 0 disables reject-on-high-water.
    uint64_t m_rssHighWaterMarkBytes = 0;

    // Drop OS page cache for HeadIndex files on eviction (for benchmarking real IO)
    bool m_dropPageCacheOnEvict = false;

    // Unified storage management functions
    bool SaveUnifiedStorage(const char* p_baseDir);
    bool LoadUnifiedStorage(const char* p_baseDir);

    // Load per-tenant sparse_tags.bin sidecars into m_tenantSparseIdx.
    // Called from both LoadUnifiedStorage and the legacy load path; safe to
    // call repeatedly (skips tenants that already have an entry).
    void LoadTenantSparseIndices();

    // Load per-tenant tagpure_meta.bin sidecars into m_tenantTagPurePostings.
    // Also resolves and caches the KV store handle + page budget needed by
    // the SearchWithACL fast path. Mirrors LoadTenantSparseIndices.
    void LoadTenantTagPureIndices();

    // Concurrent cache management
    void InitCache();
    bool EnsureTenantCached(int p_tenantId);
    void TouchLRU(int p_tenantId);
    void EvictIfNeeded();
    uint64_t EstimateTenantHeadIndexBytes(int p_tenantId) const;
    bool EnsureTenantPivotIndexLoaded(int p_tenantId);

    bool EnsureTenantLoaded(int p_tenantId);

    // Unload a tenant while holding exclusive lock (called by eviction)
    bool UnloadTenantLocked(int p_tenantId);

    // Shared SPANN search: uses HeadIndex + shared FileIO + PostingOffset
    std::shared_ptr<QueryResult> SearchSharedSPANN(ByteArray p_queryVector, int p_tenantId, int p_resultNum);
    bool InitSharedFileIO();

    // Hybrid build helpers
    TenantIndexType ChooseIndexType(int vectorCount) const;
};

#endif // _SPTAG_PW_COREINTERFACE_H_
