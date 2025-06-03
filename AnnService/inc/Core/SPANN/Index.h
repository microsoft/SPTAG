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

#include "inc/Core/Common/Labelset.h"
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

        struct IndexStat {
            std::atomic_uint32_t m_headMiss{ 0 };
            uint32_t m_appendTaskNum{ 0 };
            uint32_t m_splitNum{ 0 };
            uint32_t m_theSameHeadNum{ 0 };
            uint32_t m_reAssignNum{ 0 };
            uint32_t m_garbageNum{ 0 };
            uint64_t m_reAssignScanNum{ 0 };

            //Split
            double m_splitCost{ 0 };
            double m_clusteringCost{ 0 };
            double m_updateHeadCost{ 0 };
            double m_reassignScanCost{ 0 };
            double m_reassignScanIOCost{ 0 };

            // Append
            double m_appendCost{ 0 };
            double m_appendIOCost{ 0 };

            // reAssign
            double m_reAssignCost{ 0 };
            double m_selectCost{ 0 };
            double m_reAssignAppendCost{ 0 };

            // GC
            double m_garbageCost{ 0 };

            void PrintStat(int finishedInsert) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After %d insertion, head vectors split %d times, head missing %d times, same head %d times, reassign %d times, reassign scan %ld times, garbage collection %d times\n", 
                    finishedInsert, m_splitNum, m_headMiss.load(), m_theSameHeadNum, m_reAssignNum, m_reAssignScanNum, m_garbageNum);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendCost, m_appendCost / m_appendTaskNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, AppendIO TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendIOCost, m_appendIOCost / m_appendTaskNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_splitCost, m_splitCost / m_splitNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Clustering TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_clusteringCost, m_clusteringCost / m_splitNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, UpdateHead TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_updateHeadCost, m_updateHeadCost / m_splitNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScan TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_reassignScanCost, m_reassignScanCost / m_splitNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScanIO TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_reassignScanIOCost, m_reassignScanIOCost / m_splitNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "GCNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_garbageNum, m_garbageCost, m_garbageCost / m_garbageNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignCost, m_reAssignCost / m_reAssignNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, Select TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_selectCost, m_selectCost / m_reAssignNum);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, ReassignAppend TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignAppendCost, m_reAssignAppendCost / m_reAssignNum);
            }

            void ResetStat()
            {
                m_splitNum = 0;
                m_headMiss = 0;
                m_theSameHeadNum = 0;
                m_reAssignNum = 0;
                m_reAssignScanNum = 0;
                m_garbageNum = 0;
                m_appendTaskNum = 0;
                m_splitCost = 0;
                m_clusteringCost = 0;
                m_garbageCost = 0;
                m_updateHeadCost = 0;
                m_reassignScanCost = 0;
                m_reassignScanIOCost = 0;
                m_appendCost = 0;
                m_appendIOCost = 0;
                m_reAssignCost = 0;
                m_selectCost = 0;
                m_reAssignAppendCost = 0;
            }
        };

        template<typename T>
        class Index : public VectorIndex
        {
        private:
            std::shared_ptr<VectorIndex> m_index;
            std::shared_ptr<std::uint64_t> m_vectorTranslateMap;
            std::unordered_map<std::string, std::string> m_headParameters;

            std::shared_ptr<IExtraSearcher> m_extraSearcher;
            std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>> m_workSpaceFactory;

            Options m_options;

            std::function<float(const T*, const T*, DimensionType)> m_fComputeDistance;
            int m_iBaseSquare;

            std::mutex m_dataAddLock;
            
            std::shared_timed_mutex m_checkPointLock;

            COMMON::VersionLabel m_versionMap;

        public:
            static thread_local std::shared_ptr<ExtraWorkSpace> m_workspace;

        public:
            Index()
            {
                m_workSpaceFactory = std::make_unique<SPTAG::COMMON::ThreadLocalWorkSpaceFactory<ExtraWorkSpace>>();
                m_fComputeDistance = std::function<float(const T*, const T*, DimensionType)>(COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod));
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
            }

            ~Index() {}

            inline std::shared_ptr<VectorIndex> GetMemoryIndex() { return m_index; }
            inline std::shared_ptr<IExtraSearcher> GetDiskIndex() { return m_extraSearcher; }
            inline Options* GetOptions() { return &m_options; }

            inline SizeType GetNumSamples() const { return m_versionMap.Count(); }
            inline DimensionType GetFeatureDim() const { return m_pQuantizer ? m_pQuantizer->ReconstructDim() : m_index->GetFeatureDim(); }
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
            inline bool ContainSample(const SizeType idx) const { return idx < m_options.m_vectorSize; }

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

            std::shared_ptr<ResultIterator> GetIterator(const void* p_target, bool p_searchDeleted = false) const;
            ErrorCode SearchIndexIterativeNext(QueryResult& p_results, COMMON::WorkSpace* workSpace, int batch, int& resultCount, bool p_isFirst, bool p_searchDeleted = false) const;
            ErrorCode SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> workSpace) const;
            ErrorCode SearchIndexIterativeEnd(std::unique_ptr<SPANN::ExtraWorkSpace> extraWorkspace) const;
            bool SearchIndexIterativeFromNeareast(QueryResult& p_query, COMMON::WorkSpace* p_space, bool p_isFirst, bool p_searchDeleted = false) const;
            std::unique_ptr<COMMON::WorkSpace> RentWorkSpace(int batch) const;
            ErrorCode SearchIndexIterative(QueryResult& p_headQuery, QueryResult& p_query, COMMON::WorkSpace* p_indexWorkspace, ExtraWorkSpace* p_extraWorkspace, int p_batch, int& resultCount, bool first) const;

            ErrorCode SearchIndexWithFilter(QueryResult& p_query, std::function<bool(const ByteArray&)> filterFunc, int maxCheck = 0, bool p_searchDeleted = false) const;

            ErrorCode SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats = nullptr) const;
	        bool SearchDiskIndexIterative(QueryResult& p_headQuery, QueryResult& p_query, ExtraWorkSpace* extraWorkspace) const;
            ErrorCode DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
                SearchStats* p_stats = nullptr, std::set<int>* truth = nullptr, std::map<int, std::set<int>>* found = nullptr) const;
            ErrorCode UpdateIndex();

            ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr);
            std::string GetParameter(const char* p_param, const char* p_section = nullptr) const;

            inline const void* GetSample(const SizeType idx) const { return nullptr; }
            inline SizeType GetNumDeleted() const { return m_versionMap.GetDeleteCount(); }
            inline bool NeedRefine() const { return false; }
            ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const { return ErrorCode::Undefined; }
            ErrorCode SearchTree(QueryResult& p_query) const { return ErrorCode::Undefined; }
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false);
            ErrorCode DeleteIndex(const SizeType& p_id);

            ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum);
            ErrorCode RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams, IAbortOperation* p_abort) { return ErrorCode::Undefined; }
            ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) { return ErrorCode::Undefined; }

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
                return static_cast<SizeType>((m_vectorTranslateMap.get())[vid]);
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
            bool AllFinished() { if (m_options.m_useKV || m_options.m_useSPDK || m_options.m_useFileIO) return m_extraSearcher->AllFinished(); return true; }

            void GetDBStat() { 
                if (m_options.m_useKV || m_options.m_useSPDK || m_options.m_useFileIO) m_extraSearcher->GetDBStats(); 
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current Vector Num: %d, Deleted: %d .\n", GetNumSamples(), GetNumDeleted());
            }

            void GetIndexStat(int finishedInsert, bool cost, bool reset) { if (m_options.m_useKV || m_options.m_useSPDK || m_options.m_useFileIO) m_extraSearcher->GetIndexStats(finishedInsert, cost, reset); }
            
            void ForceCompaction() { if (m_options.m_useKV) m_extraSearcher->ForceCompaction(); }

            void StopMerge() { m_options.m_inPlace = true; }

            void OpenMerge() { m_options.m_inPlace = false; }

            void ForceGC() { m_extraSearcher->ForceGC(m_index.get()); }

            bool Initialize() { return m_extraSearcher->Initialize(); }

            bool ExitBlockController() { return m_extraSearcher->ExitBlockController(); }
            void Checkpoint() {
                /** Lock & wait until all jobs done **/

                /** Lock **/

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Locking Index\n");
                std::unique_lock<std::shared_timed_mutex> lock(m_checkPointLock);

                /** Wait **/
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Waiting for index update complete\n");
                while(!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }

                /** Flush the checkpoint file: SPTAG states, block pool states, block mapping states **/
                std::string filename = m_options.m_persistentBufferPath + "_headIndex";
                // Flush SPTAG
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving in-memory index\n");
                m_index->SaveIndex(filename);
                // Flush block pool states & block mapping states
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving storage states\n");
                m_extraSearcher->Checkpoint(m_options.m_persistentBufferPath);
            }

            ErrorCode AddIndexSPFresh(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, SizeType* VID) {
                if ((!m_options.m_useKV &&!m_options.m_useSPDK && !m_options.m_useFileIO) || m_extraSearcher == nullptr) {
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
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: VID: %d, Map Size:%d\n", begin, m_versionMap.BufferSize());
                        exit(1);
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

                return m_extraSearcher->AddIndex(vectorSet, m_index, begin);
            }
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_INDEX_H_
