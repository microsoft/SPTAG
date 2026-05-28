// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_INDEX_H_
#define _SPTAG_SPANN_INDEX_H_

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/SIMDUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/BKTree.h"
#include "inc/Core/Common/WorkSpacePool.h"
#include "inc/Core/Common/FineGrainedLock.h"
#include "inc/Core/Common/VersionLabel.h"
#include "inc/Core/Common/PostingSizeRecord.h"

#include "inc/Core/Common/LabelSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/IQuantizer.h"

#include "IExtraSearcher.h"
#include "Options.h"

#include <functional>
#include <shared_mutex>
#include <atomic>

namespace SPTAG
{

    namespace Helper
    {
        class IniReader;
    }


    namespace SPANN
    {
        struct HeadBundleNodeInfo
        {
            int nodeId = 0;
            std::string headIndexRelativePath;
            SizeType headOffset = 0;
            SizeType headCount = 0;
            SizeType postingOffset = 0;
            SizeType postingCount = 0;
            SizeType assignmentCount = 0;
        };

        template<typename T>
	    class SPANNResultIterator;

        template<typename T>
        class Index : public VectorIndex
        {
        private:
            std::shared_ptr<VectorIndex> m_index;
	        std::vector<HeadBundleNodeInfo> m_headBundleNodes;
            mutable std::vector<std::shared_ptr<VectorIndex>> m_loadedHeadBundleIndexes;
            mutable std::vector<std::vector<SizeType>> m_headBundleLocalToGlobalHIDs;
            mutable std::unordered_map<SizeType, SizeType> m_globalHeadVIDToLocalHID;
            mutable std::mutex m_headBundleLoadLock;
            mutable std::unordered_map<SizeType, std::vector<SizeType>> m_headCrossEdges;
            mutable std::atomic<bool> m_headCrossEdgesLoaded{false};
            mutable std::mutex m_headCrossEdgesMutex;
            // globalVID -> (bundleNodeId, localHidWithinBundle) reverse map, populated
            // on each EnsureHeadBundleNodeLoaded for the loaded node only.
            mutable std::unordered_map<SizeType, std::pair<int, SizeType>> m_globalVIDToBundleLoc;
            mutable std::mutex m_globalVIDToBundleLocMutex;
            std::string m_headBundleBaseDir;
	        COMMON::Dataset<std::uint64_t> m_vectorTranslateMap;
            std::unordered_map<std::string, std::string> m_headParameters;

            COMMON::VersionLabel m_versionMap;
            std::shared_ptr<IExtraSearcher> m_extraSearcher;
            std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>> m_workSpaceFactory;

            Options m_options;

            std::function<float(const T*, const T*, DimensionType)> m_fComputeDistance;
            int m_iBaseSquare;

            std::mutex m_dataAddLock;
            std::shared_timed_mutex m_dataDeleteLock;
            std::shared_timed_mutex m_checkPointLock;

            // Pre-set vector tags for embedding in postings during build
            std::vector<uint32_t> m_pendingVectorTags;
            int m_pendingNumTagsPerVec = 0;
            std::vector<std::vector<SizeType>> m_pendingNodeVectorAssignments;
            std::vector<std::vector<SizeType>> m_pendingPrimaryNodeVectorAssignments;
            std::vector<std::vector<SizeType>> m_pendingNodeHeadSelections;
            std::unordered_map<SizeType, int> m_pendingHeadVectorOwners;

            // Dual-pool v3: per-bundle U_extra VID lists and head role vector
            std::vector<std::vector<SizeType>> m_pendingNodeUExtraSelections;
            std::vector<uint8_t> m_pendingHeadRoles;

 

        public:
            Index()
            {
                m_workSpaceFactory = std::make_unique<SPTAG::COMMON::ThreadLocalWorkSpaceFactory<ExtraWorkSpace>>();
                //m_workSpaceFactory = std::make_unique<SPTAG::COMMON::SharedPoolWorkSpaceFactory<ExtraWorkSpace>>();
                m_fComputeDistance = std::function<float(const T*, const T*, DimensionType)>(COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod));
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
            }

            ~Index() {}

            inline std::shared_ptr<VectorIndex> GetMemoryIndex() { return m_index; }
            inline std::shared_ptr<IExtraSearcher> GetDiskIndex() { return m_extraSearcher; }
            inline Options* GetOptions() { return &m_options; }
            inline const std::vector<HeadBundleNodeInfo>& GetHeadBundleNodes() const { return m_headBundleNodes; }
            inline bool HasHeadBundleNodes() const { return !m_headBundleNodes.empty(); }

            // v5: Σ bundle.headCount — canonical "total head count" after cross-edges unified
            // the per-bundle subgraphs into one logical graph. Replaces m_index->GetNumSamples()
            // at sites where the value means "how many heads exist", not "navigate via m_index".
            inline SizeType TotalHeadSampleCount() const {
                SizeType n = 0;
                for (const auto& bn : m_headBundleNodes) n += bn.headCount;
                return n;
            }

            // Dual-pool v3: role-based head classification using loaded head_role.bin sidecar.
            inline bool HasHeadRoles() const {
                return m_extraSearcher && m_extraSearcher->HasHeadRoles();
            }
            inline bool IsHeadRoleUnfilterOnly(SizeType globalHeadVID) const {
                if (!m_extraSearcher) return false;
                auto bIt = m_globalHeadVIDToLocalHID.find(globalHeadVID);
                if (bIt == m_globalHeadVIDToLocalHID.end()) return false;
                return m_extraSearcher->IsUnfilterOnlyHead(static_cast<int>(bIt->second));
            }

            // Set per-vector tags to be embedded in posting metadata during build
            void SetVectorTags(const uint32_t* tags, int numVecs, int numTagsPerVec) {
                m_pendingNumTagsPerVec = numTagsPerVec;
                m_options.m_numTagsPerVec = numTagsPerVec;
                m_pendingVectorTags.assign(tags, tags + (size_t)numVecs * numTagsPerVec);
            }

            void SetNodeVectorAssignments(const std::vector<std::vector<SizeType>>& nodeVectorAssignments)
            {
                m_pendingNodeVectorAssignments = nodeVectorAssignments;
            }

            void SetPrimaryNodeVectorAssignments(const std::vector<std::vector<SizeType>>& primaryNodeVectorAssignments)
            {
                m_pendingPrimaryNodeVectorAssignments = primaryNodeVectorAssignments;
            }

            // Shared-DB hook: when set BEFORE BuildIndex / LoadIndex, the
            // ExtraDynamicSearcher will reuse this KeyValueIO instead of opening
            // its own RocksDB. Wrap with Helper::TenantPrefixedKeyValueIO when
            // multiplexing several tenants over a single physical store.
            void SetSharedDB(std::shared_ptr<Helper::KeyValueIO> p_db) { m_options.m_externalDB = std::move(p_db); }
            std::shared_ptr<Helper::KeyValueIO> GetSharedDB() const { return m_options.m_externalDB; }

            inline SizeType GetNumSamples() const { return m_versionMap.Count(); }
            inline DimensionType GetFeatureDim() const { return m_index->GetFeatureDim(); }
            inline SizeType GetValueSize() const { return m_options.m_dim * sizeof(T); }

            inline int GetCurrMaxCheck() const { return m_options.m_maxCheck; }
            inline int GetNumThreads() const { return m_options.m_iSSDNumberOfThreads; }
            inline DistCalcMethod GetDistCalcMethod() const { return m_options.m_distCalcMethod; }
            inline IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::SPANN; }
            inline VectorValueType GetVectorValueType() const { return GetEnumValueType<T>(); }

            void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer);

            inline float AccurateDistance(const void* pX, const void* pY) const { 
                if (m_options.m_distCalcMethod == DistCalcMethod::L2) return m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim);

                float xy = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim);
                float xx = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pX, m_options.m_dim);
                float yy = m_iBaseSquare - m_fComputeDistance((const T*)pY, (const T*)pY, m_options.m_dim);
                return 1.0f - xy / (sqrt(xx) * sqrt(yy));
            }
            inline float ComputeDistance(const void* pX, const void* pY) const { return m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim); }
            inline float GetDistance(const void* target, const SizeType idx) const {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "GetDistance NOT SUPPORT FOR SPANN");
                return -1;
            }
            inline bool ContainSample(const SizeType idx) const { return idx >= 0 && idx < m_versionMap.Count() && !m_versionMap.Deleted(idx); }

            std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const
            {
                std::shared_ptr<std::vector<std::uint64_t>> buffersize(new std::vector<std::uint64_t>);
                auto headIndexBufferSize = m_index->BufferSize();
                buffersize->insert(buffersize->end(), headIndexBufferSize->begin(), headIndexBufferSize->end());
                buffersize->push_back(sizeof(long long) * m_index->GetNumSamples());
                return std::move(buffersize);
            }

            std::shared_ptr<std::vector<std::string>> GetIndexFiles() const
            {
                std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
                auto headfiles = m_index->GetIndexFiles();
                for (auto file : *headfiles) {
                    files->push_back(m_options.m_headIndexFolder + FolderSep + file);
                }
                files->push_back(m_options.m_headIDFile);
                return std::move(files);
            }

            ErrorCode SaveConfig(std::shared_ptr<Helper::DiskIO> p_configout);
            ErrorCode SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams);

            ErrorCode LoadConfig(Helper::IniReader& p_reader);
            ErrorCode LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams);
            ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs);

            ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized = false, bool p_shareOwnership = false);
            ErrorCode BuildIndex(bool p_normalized = false);
            ErrorCode SearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const;

            std::shared_ptr<ResultIterator> GetIterator(const void* p_target, bool p_searchDeleted = false, std::function<bool(const ByteArray&)> p_filterFunc = nullptr, int p_maxCheck = 0) const;
            ErrorCode SearchIndexIterativeNext(QueryResult& p_results, COMMON::WorkSpace* workSpace, int batch, int& resultCount, bool p_isFirst, bool p_searchDeleted = false) const;
            ErrorCode SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> workSpace) const;
            ErrorCode SearchIndexIterativeEnd(std::unique_ptr<SPANN::ExtraWorkSpace> extraWorkspace) const;
            bool SearchIndexIterativeFromNeareast(QueryResult& p_query, COMMON::WorkSpace* p_space, bool p_isFirst, bool p_searchDeleted = false) const;
            std::unique_ptr<COMMON::WorkSpace> RentWorkSpace(int batch, std::function<bool(const ByteArray&)> p_filterFunc = nullptr, int p_maxCheck = 0) const;
            ErrorCode SearchIndexIterative(QueryResult& p_headQuery, QueryResult& p_query, COMMON::WorkSpace* p_indexWorkspace, ExtraWorkSpace* p_extraWorkspace, int p_batch, int& resultCount, bool first) const;

            ErrorCode SearchIndexWithFilter(QueryResult& p_query, std::function<bool(const ByteArray&)> filterFunc, int maxCheck = 0, bool p_searchDeleted = false) const;

            ErrorCode SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats = nullptr) const;
	        ErrorCode SearchDiskIndexIterative(QueryResult& p_headQuery, QueryResult& p_query, ExtraWorkSpace* extraWorkspace) const;
            ErrorCode DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
                SearchStats* p_stats = nullptr, std::set<int>* truth = nullptr, std::map<int, std::set<int>>* found = nullptr) const;
            ErrorCode UpdateIndex();

            void InitializeDefaultHeadBundle();
            ErrorCode SaveHeadBundleManifest(const std::string& p_baseDir) const;
            ErrorCode LoadHeadBundleManifest(const std::string& p_baseDir);
            ErrorCode InitializeHeadBundleRuntime(const std::string& p_baseDir);
            ErrorCode EnsureHeadBundleNodeLoaded(int p_nodeId) const;
            ErrorCode LoadHeadCrossEdges() const;

            // Multi-BKT cross-subgraph unified best-first traversal. Used when
            // a query tag scope spans multiple routing nodes and cross-edge
            // data is available. Uses the entry node's BKT to seed, then
            // unwinds a single priority queue across all bundle nodes via
            // RNG edges (intra-node) + cross-edges (inter-node).
            ErrorCode CrossSubgraphGraphSearch(
                QueryResult& p_query,
                COMMON::QueryResultSet<T>* p_queryResults,
                const std::vector<int>& p_candidateNodes,
                const std::uint32_t* p_queryTags,
                int p_numQueryTags,
                int p_graphResultNum,
                int& p_scannedOut) const;

            ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr);
            std::string GetParameter(const char* p_param, const char* p_section = nullptr) const;

            inline const void* GetSample(const SizeType idx) const { return nullptr; }
            inline SizeType GetNumDeleted() const { return m_versionMap.GetDeleteCount(); }
            inline bool NeedRefine() const
            {
                return m_versionMap.GetDeleteCount() > (size_t)(GetNumSamples() * m_options.m_fDeletePercentageForRefine);
            }
            ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const { return ErrorCode::Undefined; }
            ErrorCode SearchTree(QueryResult& p_query) const { return ErrorCode::Undefined; }
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false);
            ErrorCode DeleteIndex(const SizeType& p_id);

            ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum);
            ErrorCode RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams,
                                  IAbortOperation *p_abort, std::vector<SizeType> *p_mapping);
            ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) { return ErrorCode::Undefined; }

            ErrorCode Check() override;

            ErrorCode SetWorkSpaceFactory(std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>> up_workSpaceFactory)
            {
                SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>* raw_generic_ptr = up_workSpaceFactory.release();
                if (!raw_generic_ptr) return ErrorCode::Fail;


                SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>* raw_specialized_ptr = dynamic_cast<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>*>(raw_generic_ptr);
                if (!raw_specialized_ptr)
                {
                    // If it is of type SPTAG::COMMON::WorkSpace, we should pass on to child index
                    if (!m_index) 
                    {
                        delete raw_generic_ptr;
                        return ErrorCode::Fail;
                    }
                    else
                    {
                        return m_index->SetWorkSpaceFactory(std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>>(raw_generic_ptr));
                    }
                    
                }
                else
                {
                    m_workSpaceFactory = std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>>(raw_specialized_ptr);
                    return ErrorCode::Success;
                }
            }

            SizeType GetGlobalVID(SizeType vid)
            {
                return static_cast<SizeType>(*(m_vectorTranslateMap[vid]));
            }

            ErrorCode GetPostingDebug(SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs);

        private:
            bool CheckHeadIndexType();
            void SelectHeadAdjustOptions(int p_vectorCount);
            int SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID, const Options& p_opts, std::vector<int>& p_selected);
            void SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount, std::vector<int>& p_selected);

            template <typename InternalDataType>
            bool SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader);

            ErrorCode BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader);

        public:
            bool AllFinished() { if (m_options.m_storage != Storage::STATIC) return m_extraSearcher->AllFinished(); return true; }

            void GetDBStat() { 
                if (m_options.m_storage != Storage::STATIC) m_extraSearcher->GetDBStats();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current Vector Num: %d, Deleted: %d .\n", GetNumSamples(), GetNumDeleted());
            }

            void GetIndexStat(int finishedInsert, bool cost, bool reset) { if (m_options.m_storage != Storage::STATIC) m_extraSearcher->GetIndexStats(finishedInsert, cost, reset); }
            
            void ForceCompaction() { if (m_options.m_storage == Storage::ROCKSDBIO) m_extraSearcher->ForceCompaction(); }

            void StopMerge() { m_options.m_inPlace = true; }

            void OpenMerge() { m_options.m_inPlace = false; }

            void ForceGC() { 
                auto workSpace = m_workSpaceFactory->GetWorkSpace();
                if (!workSpace) {
                    workSpace.reset(new ExtraWorkSpace());
                    m_extraSearcher->InitWorkSpace(workSpace.get(), false);
                }
                else {
                    m_extraSearcher->InitWorkSpace(workSpace.get(), true);
                }
                workSpace->m_deduper.clear();
                workSpace->m_postingIDs.clear();
                m_extraSearcher->ForceGC(workSpace.get(), m_index.get()); 
            }
            
            ErrorCode Checkpoint() {
                /** Lock & wait until all jobs done **/
                while (!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }

                /** Lock **/
                if (m_options.m_persistentBufferPath == "") return ErrorCode::FailedCreateFile;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Locking Index\n");
                std::unique_lock<std::shared_timed_mutex> lock(m_checkPointLock);

                // Flush block pool states & block mapping states
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving storage states\n");
                ErrorCode ret;
                if ((ret = m_extraSearcher->Checkpoint(m_options.m_persistentBufferPath)) != ErrorCode::Success)
                    return ret;

                /** Flush the checkpoint file: SPTAG states, block pool states, block mapping states **/
                std::string filename = m_options.m_persistentBufferPath + FolderSep + m_options.m_headIndexFolder;
                // Flush SPTAG
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving in-memory index to %s\n", filename.c_str());
                if ((ret = m_index->SaveIndex(filename)) != ErrorCode::Success)
                    return ret;
                return ErrorCode::Success;
            }

            ErrorCode AddIndexSPFresh(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, SizeType* VID) {
                if (m_options.m_storage == Storage::STATIC || m_extraSearcher == nullptr) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
                    return ErrorCode::Fail;
                }

                if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;
                if (p_dimension != GetFeatureDim()) return ErrorCode::DimensionSizeMismatch;

                std::shared_lock<std::shared_timed_mutex> lock(m_checkPointLock);

                SizeType begin;
                {
                    std::lock_guard<std::mutex> lock(m_dataAddLock);

                    begin = m_versionMap.GetVectorNum();

                    if (begin == 0) { return ErrorCode::EmptyIndex; }

                    if (m_versionMap.AddBatch(p_vectorNum) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MemoryOverFlow: VID: %d, Map Size:%d\n", begin, m_versionMap.BufferSize());
                        return ErrorCode::MemoryOverFlow;
                    }
                }
                for (int i = 0; i < p_vectorNum; i++) VID[i] = begin + i;

                std::shared_ptr<VectorSet> vectorSet;
                if (m_options.m_distCalcMethod == DistCalcMethod::Cosine) {
                    ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                    memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
                    vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
                    int base = COMMON::Utils::GetBase<T>();
                    for (SizeType i = 0; i < p_vectorNum; i++) {
                        COMMON::Utils::Normalize((T*)(vectorSet->GetVector(i)), p_dimension, base);
                    }
                }
                else {
                    vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                        GetEnumValueType<T>(), p_dimension, p_vectorNum));
                }

                auto workSpace = m_workSpaceFactory->GetWorkSpace();
                if (!workSpace) {
                    workSpace.reset(new ExtraWorkSpace());
                    m_extraSearcher->InitWorkSpace(workSpace.get(), false);
                }
                else {
                    m_extraSearcher->InitWorkSpace(workSpace.get(), true);
                }
                workSpace->m_deduper.clear();
                workSpace->m_postingIDs.clear();
                return m_extraSearcher->AddIndex(workSpace.get(), vectorSet, m_index, begin);
            }
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_INDEX_H_
