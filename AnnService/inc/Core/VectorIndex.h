// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_VECTORINDEX_H_
#define _SPTAG_VECTORINDEX_H_

#include <unordered_set>
#include "Common.h"
#include "Common/WorkSpace.h"
#include "inc/Helper/DiskIO.h"
#include "SearchQuery.h"
#include "VectorSet.h"
#include "MetadataSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Cache/PostingSignature.h"

class ResultIterator;

namespace SPTAG
{

extern std::shared_ptr<Helper::DiskIO>(*f_createIO)();
class IAbortOperation
{
public:
    virtual bool ShouldAbort() = 0;
};

class VectorIndex
{
public:
    VectorIndex();

    virtual ~VectorIndex();

    virtual ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized = false, bool p_shareOwnership = false) = 0;
    
    virtual ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false) = 0;
    virtual ErrorCode AddIndexId(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, int& beginHead, int& endHead) { return ErrorCode::Undefined; }
    virtual ErrorCode AddIndexIdx(SizeType begin, SizeType end) { return ErrorCode::Undefined; }
    // Like AddIndexIdx, but builds only the OUT-edges of the newly appended
    // nodes (updateNeighbors=false): existing nodes never get a back-edge to
    // the new ones. Used to inject dual-pool U_extra heads into a bundle
    // subgraph so they can route outward to H1 without polluting H1's RNG
    // neighbor lists (H1 -> U_extra is reachable only via cross-edges).
    virtual ErrorCode AddIndexIdxNoBackEdge(SizeType begin, SizeType end) { return ErrorCode::Undefined; }
    virtual void SetAddCountForRebuild(int val) {}


    virtual ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum) = 0;

    virtual ErrorCode SearchIndex(QueryResult& p_results, bool p_searchDeleted = false) const = 0;
    
    virtual std::shared_ptr<ResultIterator> GetIterator(const void* p_target, bool p_searchDeleted = false, std::function<bool(const ByteArray&)> p_filterFunc = nullptr, int p_maxCheck = 0) const = 0;

    virtual ErrorCode SearchIndexIterativeNext(QueryResult& p_query, COMMON::WorkSpace* workSpace, int p_batch, int& resultCount, bool p_isFirst, bool p_searchDeleted) const = 0;

    virtual ErrorCode SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> workSpace) const = 0;

    virtual bool SearchIndexIterativeFromNeareast(QueryResult& p_query, COMMON::WorkSpace* p_space, bool p_isFirst, bool p_searchDeleted = false) const = 0;

    virtual std::unique_ptr<COMMON::WorkSpace> RentWorkSpace(int batch, std::function<bool(const ByteArray&)> p_filterFunc = nullptr, int p_maxCheck = 0) const = 0;

    virtual ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const = 0;

    virtual ErrorCode SearchIndexWithFilter(QueryResult& p_query, std::function<bool(const ByteArray&)> filterFunc, int maxCheck = 0, bool p_searchDeleted = false) const = 0;

    virtual ErrorCode SearchTree(QueryResult &p_query) const = 0;

    virtual ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) = 0;

    virtual float AccurateDistance(const void* pX, const void* pY) const = 0;
    virtual float ComputeDistance(const void* pX, const void* pY) const = 0;
    virtual float GetDistance(const void* target, const SizeType idx) const = 0;
    virtual const void* GetSample(const SizeType idx) const = 0;
    virtual bool ContainSample(const SizeType idx) const = 0;
    virtual bool NeedRefine() const = 0;
   
    virtual DimensionType GetFeatureDim() const = 0;
    virtual SizeType GetNumSamples() const = 0;
    virtual SizeType GetNumDeleted() const = 0;

    virtual DistCalcMethod GetDistCalcMethod() const = 0;
    virtual IndexAlgoType GetIndexAlgoType() const = 0;
    virtual VectorValueType GetVectorValueType() const = 0;

    virtual std::string GetParameter(const char* p_param, const char* p_section = nullptr) const = 0;
    virtual ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr) = 0;
    virtual ErrorCode UpdateIndex() = 0;

    virtual bool IsReady() const { return m_bReady; }
    virtual void SetReady(bool p_ready) { m_bReady = p_ready; }

    virtual std::shared_ptr<std::vector<std::uint64_t>> CalculateBufferSize() const;

    virtual ErrorCode SaveIndex(std::string& p_config, const std::vector<ByteArray>& p_indexBlobs);

    virtual ErrorCode SaveIndex(const std::string& p_folderPath);

    virtual ErrorCode SaveIndexToFile(const std::string& p_file, IAbortOperation* p_abort = nullptr);

    virtual ErrorCode BuildIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false, bool p_shareOwnership = false);
    
    virtual ErrorCode BuildIndex(bool p_normalized = false) { return ErrorCode::Undefined; }

    virtual ErrorCode AddIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false);

    virtual ErrorCode DeleteIndex(ByteArray p_meta);

    virtual ErrorCode MergeIndex(VectorIndex* p_addindex, int p_threadnum, IAbortOperation* p_abort);
    
    virtual const void* GetSample(ByteArray p_meta, bool& deleteFlag);

    virtual ErrorCode SearchIndex(const void* p_vector, int p_vectorCount, int p_neighborCount, bool p_withMeta, BasicResult* p_results) const;

    virtual void ApproximateRNG(std::shared_ptr<VectorSet>& fullVectors, std::unordered_set<SizeType>& exceptIDS, int candidateNum, Edge* selections, int replicaCount, int numThreads, int numTrees, int leafSize, float RNGFactor, int numGPUs);

    static void SortSelections(std::vector<Edge>* selections);

    virtual std::string GetParameter(const std::string& p_param, const std::string& p_section = "Index") const;
    virtual ErrorCode SetParameter(const std::string& p_param, const std::string& p_value, const std::string& p_section = "Index");

    virtual ByteArray GetMetadata(SizeType p_vectorID) const;
    virtual MetadataSet* GetMetadata() const;
    virtual void SetMetadata(MetadataSet* p_new);

    virtual std::string GetIndexName() const 
    { 
        if (m_sIndexName == "") return Helper::Convert::ConvertToString(GetIndexAlgoType());
        return m_sIndexName; 
    }
    virtual void SetIndexName(std::string p_name) { m_sIndexName = p_name; }

    virtual void SetQuantizerFileName(std::string p_QuantizerFileName) { m_sQuantizerFile = p_QuantizerFileName; }

    virtual void SetQuantizerADC(bool enableADC) {
        if (m_pQuantizer) m_pQuantizer->SetEnableADC(enableADC);
    }

    virtual void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer) = 0;

    virtual ErrorCode LoadQuantizer(std::string p_quantizerFile);

    virtual std::shared_ptr<SPTAG::COMMON::IQuantizer> GetQuantizer() {
        return m_pQuantizer;
    }

    virtual ErrorCode QuantizeVector(const void* p_data, SizeType p_num, ByteArray p_out) {
        if (m_pQuantizer != nullptr && p_out.Length() >= m_pQuantizer->GetNumSubvectors() * (size_t)p_num) {
            for (int i = 0; i < p_num; i++) 
                m_pQuantizer->QuantizeVector(((std::uint8_t*)p_data) + i * (size_t)(m_pQuantizer->ReconstructSize()), p_out.Data() + i * (size_t)(m_pQuantizer->GetNumSubvectors()), false);
            return ErrorCode::Success;
        }
        return ErrorCode::Fail;
    }

    virtual ErrorCode ReconstructVector(const void* p_data, SizeType p_num, ByteArray p_out) {
        if (m_pQuantizer != nullptr && p_out.Length() >= m_pQuantizer->ReconstructSize() * (size_t)p_num) {
            for (int i = 0; i < p_num; i++)
                m_pQuantizer->ReconstructVector(((std::uint8_t*)p_data) + i * (size_t)(m_pQuantizer->GetNumSubvectors()), p_out.Data() + i * (size_t)(m_pQuantizer->ReconstructSize()));
            return ErrorCode::Success;
        }
        return ErrorCode::Fail;
    }

    static std::shared_ptr<VectorIndex> CreateInstance(IndexAlgoType p_algo, VectorValueType p_valuetype);

    static ErrorCode LoadIndex(const std::string& p_loaderFilePath, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static ErrorCode LoadIndexFromFile(const std::string& p_file, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static ErrorCode LoadIndex(const std::string& p_config, const std::vector<ByteArray>& p_indexBlobs, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static std::uint64_t EstimatedVectorCount(std::uint64_t p_memory, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_vectorsInBlock, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize);

    static std::uint64_t EstimatedMemoryUsage(std::uint64_t p_vectorCount, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_vectorsInBlock, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize);

    virtual std::shared_ptr<VectorIndex> Clone(std::string p_clone);

    virtual std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const = 0;

    virtual std::shared_ptr<std::vector<std::string>> GetIndexFiles() const = 0;

    virtual ErrorCode SaveConfig(std::shared_ptr<Helper::DiskIO> p_configout) = 0;

    virtual ErrorCode SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams) = 0;

    virtual ErrorCode LoadConfig(Helper::IniReader& p_reader) = 0;

    virtual ErrorCode LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams) = 0;

    virtual ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs) = 0;

    virtual ErrorCode DeleteIndex(const SizeType& p_id) = 0;

    virtual ErrorCode RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams, IAbortOperation* p_abort, std::vector<SizeType>* p_mapping) = 0;

    virtual ErrorCode SetWorkSpaceFactory(std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>> up_workSpaceFactory) = 0;

    inline bool HasMetaMapping() const { return nullptr != m_pMetaToVec; }

    inline SizeType GetMetaMapping(std::string& meta) const;

    void UpdateMetaMapping(const std::string& meta, SizeType i);

    void BuildMetaMapping(bool p_checkDeleted = true);

    virtual ErrorCode Check()
    {
        return ErrorCode::Undefined;
    }

    virtual std::string GetPriorityID(int queryID) const { return ""; }

        void ClearHeadNodeMeta();

        void InitializeHeadNodeMeta(
            SizeType p_numSamples,
            int p_numQuantCols = 0);

        void InitializeHeadNodeMeta(
            SizeType p_numSamples,
            int p_numQuantCols,
            const Cache::HierWidthTable& p_hierWidths);

        static bool TryComputeHeadNodeMetaStride(int p_numQuantCols, size_t& p_stride);

        static bool TryComputeHeadNodeMetaStride(
            int p_numQuantCols,
            const Cache::HierWidthTable& p_hierWidths,
            size_t& p_stride);

        bool HasHeadNodeMeta() const { return m_headNodeMetaStride > 0 && !m_headNodeMeta.empty(); }

        SizeType GetHeadNodeMetaSampleCount() const;

        size_t GetHeadNodeMetaStride() const { return m_headNodeMetaStride; }

        Cache::HierWidthTable GetHeadNodeHierWidths() const;

        const std::vector<std::uint8_t>& GetHeadNodeMetaBlob() const { return m_headNodeMeta; }

        std::vector<std::uint8_t>& GetHeadNodeMetaBlob() { return m_headNodeMeta; }

        void SetHeadNodeGlobalVID(SizeType p_sampleId, SizeType p_globalVID);

        SizeType GetHeadNodeGlobalVID(SizeType p_sampleId) const;

        void SetHeadNodePS(SizeType p_sampleId, const Cache::PostingBitmask& p_ps);

        const Cache::PostingBitmask* GetHeadNodePS(SizeType p_sampleId) const;

        bool HeadNodePSMayIntersect(SizeType p_sampleId, const Cache::PostingBitmask& p_queryMask) const;

        void SetHeadNodeHeadOnly(SizeType p_sampleId, bool p_isHeadOnly);

        bool IsHeadNodeHeadOnly(SizeType p_sampleId) const;

        void SetHeadNodeHierMask(SizeType p_sampleId, const Cache::HierarchicalOwnTags& p_mask);

        const Cache::HierarchicalOwnTags* GetHeadNodeHierMask(SizeType p_sampleId) const;

        // Posting-content mask: union of all member-vector tags in the head's
        // posting. Distinct from the head's own-tag HierMask (used by
        // HeadNodeMatchesQuery) which gates whether the head's centroid vector
        // is admissible as a top-K result. The posting-content mask is the
        // correct pre-filter for "this posting MAY contain a vector that
        // matches the query".
        void SetHeadNodePostingHierMask(SizeType p_sampleId, const Cache::HierarchicalPostingMask& p_mask);

        const Cache::HierarchicalPostingMask* GetHeadNodePostingHierMask(SizeType p_sampleId) const;

        // Quantized numeric posting signature (range pruning). Per head, a flat
        // M*NUM_QUANT_WORDS uint64 block = union of member-vector numeric buckets
        // (one 256-bit lane per numeric column). M=0 => no block (V3 layout, byte
        // identical). Used by the posting pre-filter for numeric range predicates.
        int GetHeadNodeNumQuantCols() const { return m_headNodeNumQuantCols; }
        std::uint64_t* GetHeadNodeNumQuantMutable(SizeType p_sampleId);
        const std::uint64_t* GetHeadNodeNumQuant(SizeType p_sampleId) const;

        void SetHeadNodeBundleNodeId(SizeType p_sampleId, int16_t p_bundleNodeId);

        int16_t GetHeadNodeBundleNodeId(SizeType p_sampleId) const;

        // p_routedNodeMask: per-bundle-node allow-list (size = nodeCount, 1=allowed).
        // Pass an empty vector to disable bundle-node routing (scan all nodes).
        // Replaces the previous uint32_t bitmask (which silently dropped nodes >= 32).
        bool HeadNodeMatchesQuery(SizeType p_sampleId, const Cache::HierarchicalPostingMask& p_queryMask, const std::vector<uint8_t>& p_routedNodeMask) const;

        bool HeadNodeMatchesQuery(
            SizeType p_sampleId,
            const Cache::HierarchicalPostingMask& p_queryMask,
            const std::vector<uint8_t>& p_routedNodeMask,
            const Cache::HierWidthTable& p_hierWidths) const;

        // Lightweight tag-content gate for heads using the head's OWN-tag mask
        // (post V3 dual-mask: HierMask reflects the head VID's own tags only).
        // Use this only when you want "head centroid's own tag must match query"
        // semantics (e.g., gating top-K return of ghost head-only vectors).
        // For posting-aware pre-filtering (the historic "joint" behaviour where
        // a head is kept whenever its posting MAY contain a query-matching
        // member), use HeadPostingHierMaskMayIntersect instead.
        bool HeadHierMaskMayIntersect(SizeType p_sampleId, const Cache::HierarchicalPostingMask& p_queryMask) const;

        bool HeadHierMaskMayIntersect(
            SizeType p_sampleId,
            const Cache::HierarchicalPostingMask& p_queryMask,
            const Cache::HierWidthTable& p_hierWidths) const;

        // Posting-member-union mask intersect. Safe (no false negatives) head
        // pre-filter for routed graph search: kept iff the head's posting
        // MAY contain a query-matching vector. If posting mask is absent
        // (e.g. legacy/V2 indexes that only stored own-tag), fails open and
        // returns true so the caller doesn't drop the head spuriously.
        bool HeadPostingHierMaskMayIntersect(SizeType p_sampleId, const Cache::HierarchicalPostingMask& p_queryMask) const;

        bool HeadPostingHierMaskMayIntersect(
            SizeType p_sampleId,
            const Cache::HierarchicalPostingMask& p_queryMask,
            const Cache::HierWidthTable& p_hierWidths) const;

        struct PostingScanStats {
            uint64_t m_readPostings = 0;
            uint64_t m_matchedPostings = 0;
            uint64_t m_prePSPostings = 0;
            uint64_t m_scannedVectors = 0;
            uint64_t m_matchedVectors = 0;
            uint64_t m_uniqueMatchedPostings = 0;
            uint64_t m_uniqueMatchedVectors = 0;
            uint64_t m_primaryHeadCandidates = 0;
            uint64_t m_postingPageReads = 0;
            uint64_t m_postingLogicalBytes = 0;
            uint64_t m_postingPhysicalBytes = 0;
            uint64_t m_adcScannedVectors = 0;
            uint64_t m_adcSurvivors = 0;
            uint64_t m_rerankCandidates = 0;
            uint64_t m_rerankReadRequests = 0;
            uint64_t m_rerankPhysicalBytes = 0;

            uint64_t FalsePositivePostings() const
            {
                return (m_readPostings >= m_matchedPostings) ? (m_readPostings - m_matchedPostings) : 0;
            }
        };

        struct ThreadLocalSearchContext {
            bool m_active = false;
            std::function<bool(int)> m_postingFilter;
            std::vector<uint32_t> m_queryTags;
            float m_filterSelectivity = 1.0f;
            // Unmodified predicate selectivity for cost-based route selection.
            // m_filterSelectivity may include adaptive-nprobe safety scaling.
            float m_routeSelectivity = 1.0f;
            std::vector<SizeType> m_directPostingIDs;
            // Optional local head IDs whose centroid vectors must be merged with
            // a direct posting scan. Kept separate so generic sparse-tag callers
            // that supply arbitrary posting IDs retain their existing behavior.
            std::vector<SizeType> m_directHeadLocalIDs;
            std::vector<int> m_searchHeadBundleNodes;
            // Per-level minimum tag value (ascending, disjoint ranges) persisted at
            // build time as tag_level_offsets.bin. Used to map a raw tag value to its
            // hierarchical level (org/dept/team/project) for HierarchicalPostingMask
            // construction. Empty -> fall back to legacy fixed thresholds.
            std::vector<uint32_t> m_tagLevelOffsets;

            // Optional DNF predicate (OR of AND-clauses). When non-empty it is the
            // authoritative filter and supersedes the flat OR/IN m_queryTags list.
            SPTAG::Cache::DNFPredicate m_dnf;

            void Reset()
            {
                m_active = false;
                m_postingFilter = nullptr;
                m_queryTags.clear();
                m_filterSelectivity = 1.0f;
                m_routeSelectivity = 1.0f;
                m_directPostingIDs.clear();
                m_directHeadLocalIDs.clear();
                m_searchHeadBundleNodes.clear();
                m_tagLevelOffsets.clear();
                m_dnf.Clear();
            }

            const SPTAG::Cache::DNFPredicate* DNF() const
            {
                return m_dnf.Empty() ? nullptr : &m_dnf;
            }

            const uint32_t* QueryTags() const
            {
                return m_queryTags.empty() ? nullptr : m_queryTags.data();
            }

            int NumQueryTags() const
            {
                return static_cast<int>(m_queryTags.size());
            }
        };

        class ThreadLocalSearchContextGuard {
        public:
            explicit ThreadLocalSearchContextGuard(ThreadLocalSearchContext p_context);
            ~ThreadLocalSearchContextGuard();

            ThreadLocalSearchContextGuard(const ThreadLocalSearchContextGuard&) = delete;
            ThreadLocalSearchContextGuard& operator=(const ThreadLocalSearchContextGuard&) = delete;

        private:
            bool m_hadPrevious = false;
            ThreadLocalSearchContext m_previous;
        };

        static void ResetThreadLocalPostingScanStats();

        static void SetThreadLocalPostingScanStats(uint64_t p_readPostings, uint64_t p_matchedPostings,
                                                   uint64_t p_prePSPostings = 0,
                                                   uint64_t p_scannedVectors = 0,
                                                   uint64_t p_matchedVectors = 0,
                                                   uint64_t p_primaryHeadCandidates = 0,
                                                   uint64_t p_postingPageReads = 0,
                                                   uint64_t p_postingLogicalBytes = 0,
                                                   uint64_t p_postingPhysicalBytes = 0,
                                                   uint64_t p_adcScannedVectors = 0,
                                                   uint64_t p_adcSurvivors = 0,
                                                   uint64_t p_rerankCandidates = 0,
                                                   uint64_t p_rerankReadRequests = 0,
                                                   uint64_t p_rerankPhysicalBytes = 0,
                                                   uint64_t p_uniqueMatchedPostings = 0,
                                                   uint64_t p_uniqueMatchedVectors = 0);

        static PostingScanStats GetThreadLocalPostingScanStats();

        static void SetThreadLocalSearchContext(ThreadLocalSearchContext p_context);

        static void ResetThreadLocalSearchContext();

        static const ThreadLocalSearchContext* GetThreadLocalSearchContext();
    
    // Public to allow TenantIndexManager to perform a custom load flow that
    // injects a shared KeyValueIO between LoadIndexConfig and LoadIndexData.
    ErrorCode LoadIndexConfig(Helper::IniReader& p_reader);

  private:
    ErrorCode SaveIndexConfig(std::shared_ptr<Helper::DiskIO> p_configOut);

protected:
    bool m_bReady = false;
    std::string m_sIndexName = "";
    std::string m_sMetadataFile = "metadata.bin";
    std::string m_sMetadataIndexFile = "metadataIndex.bin";
    std::string m_sQuantizerFile = "quantizer.bin";
    std::shared_ptr<MetadataSet> m_pMetadata;
    std::shared_ptr<void> m_pMetaToVec;

public:
    // Per-head-node metadata blob, indexed by local head sample id (hid).
    // Layout V3 (each record stores):
    //   [PostingBitmask][HierarchicalOwnTags own-tags][HierarchicalPostingMask posting-content][globalVID][bundleNodeId][headOnly]
    // Aligned to alignof(PostingBitmask)=8 for stride.
    size_t m_headNodeMetaStride = 0;
    size_t m_headNodePSOffset = 0;
    size_t m_headNodeHierMaskOffset = 0;
    size_t m_headNodePostingHierMaskOffset = 0;
    size_t m_headNodeGlobalVIDOffset = 0;
    size_t m_headNodeBundleNodeIdOffset = 0;
    size_t m_headNodeHeadOnlyOffset = 0;
    size_t m_headNodeNumQuantOffset = 0;   // offset of quantized numeric block (V4)
    int m_headNodeNumQuantCols = 0;        // numeric columns in quant block (0 = none)
    std::vector<std::uint8_t> m_headNodeMeta;

public:
    int m_iDataBlockSize = 1024 * 1024;
    int m_iDataCapacity = MaxSize;
    int m_iMetaRecordSize = 10;
    std::shared_ptr<SPTAG::COMMON::IQuantizer> m_pQuantizer = nullptr;
};


} // namespace SPTAG

#endif // _SPTAG_VECTORINDEX_H_
