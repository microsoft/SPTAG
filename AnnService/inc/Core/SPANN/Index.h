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
#include "inc/Helper/KeyValueIO.h"
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
#include <memory>

namespace SPTAG
{

    namespace Helper
    {
        class IniReader;
    }


    namespace SPANN
    {
        template<typename T>
	    class SPANNResultIterator;

        template<typename T>
        class Index;
        template<typename T>
        class Index : public VectorIndex
        {
        private:
            std::shared_ptr<VectorIndex> m_topIndex;
	        COMMON::Dataset<SizeType> m_topLocalToGlobalID;
            Helper::Concurrent::ConcurrentMap<SizeType, SizeType> m_topGlobalToLocalID;
            std::shared_timed_mutex m_topLocalToGlobalIDLock;
            std::unordered_map<std::string, std::string> m_topParameters;

            std::shared_ptr<Helper::KeyValueIO> m_db;
            std::vector<std::shared_ptr<IExtraSearcher>> m_extraSearchers;
            std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>> m_workSpaceFactory;

            Options m_options;

            std::function<float(const T*, const T*, DimensionType)> m_fComputeDistance;
            int m_iBaseSquare;

            std::mutex m_dataAddLock;
            std::shared_timed_mutex m_dataDeleteLock;
            std::shared_timed_mutex m_checkPointLock;

            std::shared_ptr<Helper::Concurrent::ConcurrentQueue<int>> m_freeWorkSpaceIds;
            std::atomic<int> m_workspaceCount = 0;

        public:
            Index()
            {
                m_workSpaceFactory = std::make_unique<SPTAG::COMMON::ThreadLocalWorkSpaceFactory<ExtraWorkSpace>>();
                //m_workSpaceFactory = std::make_unique<SPTAG::COMMON::SharedPoolWorkSpaceFactory<ExtraWorkSpace>>();
                m_fComputeDistance = std::function<float(const T*, const T*, DimensionType)>(COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod));
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
            }

            ~Index() {}

            void InitWorkSpace(ExtraWorkSpace* p_exWorkSpace, bool clear = false) const
            {
                if (clear) {
                    p_exWorkSpace->Clear(m_options.m_searchInternalResultNum, (max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit) + m_options.m_bufferLength) << PageSizeEx, true, m_options.m_enableDataCompression);
                }
                else {
                    p_exWorkSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, max(m_options.m_searchInternalResultNum, m_options.m_reassignK), (max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit) + m_options.m_bufferLength) << PageSizeEx, true, m_options.m_enableDataCompression);
                    int wid = 0;
                    if (m_freeWorkSpaceIds == nullptr || !m_freeWorkSpaceIds->try_pop(wid))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FreeWorkSpaceIds is not initalized or the workspace number is not enough! Please increase iothread number.\n");
                        p_exWorkSpace->m_diskRequests[0].m_status = -1;
                        return;
                    }
                    for (auto & req : p_exWorkSpace->m_diskRequests)
                    {
                        req.m_status = wid;
                    }
                    p_exWorkSpace->m_callback = [m_freeWorkSpaceIds = m_freeWorkSpaceIds, wid] () {
                        if (m_freeWorkSpaceIds) m_freeWorkSpaceIds->push(wid);
                    };
                }
            }

            inline std::shared_ptr<VectorIndex> GetMemoryIndex() { return m_topIndex; }
            inline std::shared_ptr<IExtraSearcher> GetDiskIndex(int layer = 0) { if (layer < m_extraSearchers.size()) return m_extraSearchers[layer]; else return nullptr; }
            inline Options* GetOptions() { return &m_options; }

            inline SizeType GetNumSamples() const { return GetNumSamples(0); }
            inline SizeType GetNumSamples(int layer) const { if (layer < m_extraSearchers.size()) return m_extraSearchers[layer]->GetNumSamples(); else return m_topIndex->GetNumSamples(); }
            inline DimensionType GetFeatureDim() const { return m_topIndex->GetFeatureDim(); }
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
            inline bool ContainSample(const SizeType idx) const { return ContainSample(idx, 0); }
            inline bool ContainSample(const SizeType idx, int layer) const { 
                if (layer < m_extraSearchers.size()) return m_extraSearchers[layer]->ContainSample(idx); 
                else {
                    auto it = m_topGlobalToLocalID.find(idx);
                    if (it == m_topGlobalToLocalID.end()) return false;
                    return m_topIndex->ContainSample(it->second);
                }
            }

            std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const
            {
                std::shared_ptr<std::vector<std::uint64_t>> buffersize(new std::vector<std::uint64_t>);
                auto headIndexBufferSize = m_topIndex->BufferSize();
                buffersize->insert(buffersize->end(), headIndexBufferSize->begin(), headIndexBufferSize->end());
                buffersize->push_back(sizeof(long long) * m_topIndex->GetNumSamples());
                return std::move(buffersize);
            }

            std::shared_ptr<std::vector<std::string>> GetIndexFiles() const
            {
                std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
                auto headfiles = m_topIndex->GetIndexFiles();
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

            ErrorCode SearchHeadIndex(QueryResult& p_query, int p_tolayer, ExtraWorkSpace *p_exWorkSpace = nullptr) const;
            ErrorCode SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats = nullptr, int p_tolayer = 0, ExtraWorkSpace *p_exWorkSpace = nullptr) const;
	        ErrorCode SearchDiskIndexIterative(QueryResult& p_headQuery, QueryResult& p_query, ExtraWorkSpace* extraWorkspace) const;
            ErrorCode DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
                SearchStats* p_stats = nullptr, std::set<SizeType>* truth = nullptr, std::map<SizeType, std::set<SizeType>>* found = nullptr) const;
            ErrorCode UpdateIndex();

            ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr);
            std::string GetParameter(const char* p_param, const char* p_section = nullptr) const;

            inline const void* GetSample(const SizeType idx) const { return nullptr; }
            inline SizeType GetNumDeleted() const { return GetNumDeleted(0); }
            inline SizeType GetNumDeleted(int layer) const { if (layer < m_extraSearchers.size()) return m_extraSearchers[layer]->GetNumDeleted(); else return m_topIndex->GetNumDeleted(); }
            inline bool NeedRefine() const
            {
                return GetNumDeleted(0) > (size_t)(GetNumSamples(0) * m_options.m_fDeletePercentageForRefine);
            }
            ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const { return ErrorCode::Undefined; }
            ErrorCode SearchTree(QueryResult& p_query) const { return ErrorCode::Undefined; }
            ErrorCode AddHeadIndex(const void* p_data, SizeType VID, uint8_t version, DimensionType p_dimension, int tolayer, ExtraWorkSpace* extraWorkspace)
            {
                if (tolayer < m_extraSearchers.size()) {
                    std::shared_ptr<VectorSet> vectorSet(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, m_options.m_dim * sizeof(T), false),
                        GetEnumValueType<T>(), m_options.m_dim, 1));
                    return m_extraSearchers[tolayer]->AddIndex(extraWorkspace, vectorSet, VID);
                }

                SizeType begin, end;
                ErrorCode ret;
                if ((ret = m_topIndex->AddIndexId(p_data, 1, p_dimension, begin, end)) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to add head index for VID %d to top index.\n", VID);
                    return ret;
                }

                {
                    std::unique_lock<std::shared_timed_mutex> tmplock(m_topLocalToGlobalIDLock);
                    m_topLocalToGlobalID.AddBatch(1);
                    *(m_topLocalToGlobalID.At(begin)) = VID;
                    m_topGlobalToLocalID[VID] = begin;
                }
                return m_topIndex->AddIndexIdx(begin, end);
            }
            
            ErrorCode GetHeadIndexMapping(int tolayer, std::vector<SizeType>& globalIDs) {
                globalIDs.clear();
                if (tolayer >= m_extraSearchers.size()) {
                    for (SizeType i = 0; i < m_topLocalToGlobalID.R(); i++) {
                        if (m_topIndex->ContainSample(i))
                            globalIDs.push_back(*(m_topLocalToGlobalID.At(i)));
                    }
                    return ErrorCode::Success;
                }
                return m_extraSearchers[tolayer]->GetContainedIDs(globalIDs);
            }

            std::string GetPriorityID(SizeType headID, int tolayer)
            {
                return std::to_string(headID);
            }

            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false) 
            { 
                return AddIndex(p_data, p_vectorNum, p_dimension, p_metadataSet, p_withMetaIndex, p_normalized, nullptr); 
            }
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex, bool p_normalized, SizeType* VID);
            ErrorCode DeleteIndex(const SizeType& p_id) { return DeleteIndex(p_id, 0); }
            ErrorCode DeleteIndex(const SizeType& p_id, int layer);

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
                    if (!m_topIndex) 
                    {
                        delete raw_generic_ptr;
                        return ErrorCode::Fail;
                    }
                    else
                    {
                        return m_topIndex->SetWorkSpaceFactory(std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>>(raw_generic_ptr));
                    }
                    
                }
                else
                {
                    m_workSpaceFactory = std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>>(raw_specialized_ptr);
                    return ErrorCode::Success;
                }
            }

        private:
            bool CheckHeadIndexType();
            void SelectHeadAdjustOptions(int p_vectorCount);
            int SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID, const Options& p_opts, std::vector<int>& p_selected);
            void SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount, std::vector<int>& p_selected);

            template <typename InternalDataType>
            bool SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader);

            ErrorCode BuildIndexInternalLayer(std::shared_ptr<Helper::VectorSetReader>& p_reader);

            ErrorCode BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<Helper::ReaderOptions> &vectorOptions);

        public:
            void PrepareDB(std::shared_ptr<Helper::KeyValueIO>& db, int layer = 0);
            
            bool AllFinished() { 
                if (m_options.m_storage != Storage::STATIC) {
                    bool allDone = true;
                    for (auto& searcher : m_extraSearchers) {
                        if (searcher && !searcher->AllFinished()) allDone = false; 
                    }
                    return allDone;
                }
                return true;
            }

            void GetDBStat() { 
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current Vector Num: %d, Deleted: %d .\n", GetNumSamples(), GetNumDeleted());
                if (m_options.m_storage != Storage::STATIC) {
                    for (auto& searcher : m_extraSearchers) {
                        if (searcher) searcher->GetDBStats();
                    }
                }
            }
            
            void ForceCompaction() { 
                if (m_options.m_storage == Storage::ROCKSDBIO) {
                    for (auto& searcher : m_extraSearchers) {
                        if (searcher) searcher->ForceCompaction(); 
                    }
                }
            }

            void StopMerge() { m_options.m_inPlace = true; }

            void OpenMerge() { m_options.m_inPlace = false; }
            
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
                for (auto& searcher : m_extraSearchers) {
                    if (searcher) {
                        if ((ret = searcher->Checkpoint(m_options.m_persistentBufferPath)) != ErrorCode::Success)
                            return ret;
                    }
                }

                /** Flush the checkpoint file: SPTAG states, block pool states, block mapping states **/
                std::string filename = m_options.m_persistentBufferPath + FolderSep + m_options.m_headIndexFolder;
                // Flush SPTAG
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving in-memory index to %s\n", filename.c_str());
                if ((ret = m_topIndex->SaveIndex(filename)) != ErrorCode::Success)
                    return ret;
                return ErrorCode::Success;
            }
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_INDEX_H_
