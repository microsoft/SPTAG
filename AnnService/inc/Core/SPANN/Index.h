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
#include "PersistentBuffer.h"

#include <functional>
#include <shared_mutex>
#include <utility>
#include <random>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>

namespace SPTAG
{

    namespace Helper
    {
        class IniReader;
    }

    namespace SPANN
    {
        template<typename T>
    class Index : public VectorIndex
    {
            // class AppendAsyncJob : public Helper::ThreadPool::Job
            // {
            // private:
            //     VectorIndex* m_index;
            //     SizeType headID;
            //     int appendNum;
            //     std::shared_ptr<std::string> appendPosting;
            //     std::function<void()> m_callback;
            // public:
            //     AppendAsyncJob(VectorIndex* m_index, SizeType headID, int appendNum, std::shared_ptr<std::string> appendPosting, std::function<void()> p_callback)
            //             : m_index(m_index), headID(headID), appendNum(appendNum), appendPosting(std::move(appendPosting)), m_callback(std::move(p_callback)) {}

            //     ~AppendAsyncJob() {}

            //     inline void exec(IAbortOperation* p_abort) override {
            //         m_index->Append(headID, appendNum, *appendPosting);
            //         if (m_callback != nullptr) {
            //             m_callback();
            //         }
            //     }
            // };

            class SplitAsyncJob : public Helper::ThreadPool::Job
            {
            private:
                VectorIndex* m_index;
                SizeType headID;
                std::function<void()> m_callback;
            public:
                SplitAsyncJob(VectorIndex* m_index, SizeType headID, std::function<void()> p_callback)
                        : m_index(m_index), headID(headID), m_callback(std::move(p_callback)) {}

                ~SplitAsyncJob() {}

                inline void exec(IAbortOperation* p_abort) override {
                    m_index->Split(headID);
                    if (m_callback != nullptr) {
                        m_callback();
                    }
                }
            };

            class ReassignAsyncJob : public SPTAG::Helper::ThreadPool::Job
            {
            private:
                VectorIndex* m_index;
                std::shared_ptr<std::string> vectorContain;
                SizeType VID;
                SizeType HeadPrev;
                uint8_t version;
                std::function<void()> m_callback;
            public:
                ReassignAsyncJob(VectorIndex* m_index,
                                 std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version, std::function<void()> p_callback)
                        : m_index(m_index),
                          vectorContain(std::move(vectorContain)), VID(VID), HeadPrev(HeadPrev), version(version), m_callback(std::move(p_callback)) {}

                ~ReassignAsyncJob() {}

                void exec(IAbortOperation* p_abort) override {
                    m_index->ProcessAsyncReassign(vectorContain, VID, HeadPrev, version, std::move(m_callback));
                }
            };

            class ThreadPool : public Helper::ThreadPool 
            {
            private:
                std::atomic_uint32_t currentJobs{0};
            public:
                ThreadPool() : Helper::ThreadPool() {}
                
                ~ThreadPool() {}
                
                void init(int numberOfThreads = 1)
                {
                    m_abort.SetAbort(false);
                    for (int i = 0; i < numberOfThreads; i++)
                    {
                        m_threads.emplace_back([this] {
                            Job *j;
                            while (get(j))
                            {
                                try 
                                {
                                    currentJobs++;
                                    j->exec(&m_abort);
                                    currentJobs--;
                                }
                                catch (std::exception& e) {
                                    LOG(Helper::LogLevel::LL_Error, "ThreadPool: exception in %s %s\n", typeid(*j).name(), e.what());
                                }
                                
                                delete j;
                            }
                        });
                    }
                }

                inline uint32_t runningJobs() { return currentJobs; }

                inline bool     allClear()    { return currentJobs == 0 && jobsize() == 0; }
            };

            class Dispatcher
            {
            private:
                std::thread t;

                std::size_t batch;
                std::atomic_bool running{false};
                std::atomic_uint32_t sentAssignment{0};

                Index* m_index;
                std::shared_ptr<PersistentBuffer> m_persistentBuffer;
                std::shared_ptr<ThreadPool> appendThreadPool;
                std::shared_ptr<ThreadPool> reassignThreadPool;

            public:
                Dispatcher(std::shared_ptr<PersistentBuffer> pb, std::size_t batch, std::shared_ptr<ThreadPool> append, std::shared_ptr<ThreadPool> reassign, Index* m_index)
                        : m_persistentBuffer(pb), batch(batch), appendThreadPool(append), reassignThreadPool(reassign), m_index(m_index) {
                            LOG(Helper::LogLevel::LL_Info, "Dispatcher: batch size: %d\n", batch);
                        }

                ~Dispatcher() { running = false; t.join(); }

                void dispatch();

                inline void run() { running = true; t = std::thread(&Dispatcher::dispatch, this); }

                inline void stop() { running = false; }

                inline bool allFinished()
                {
                    return appendThreadPool->allClear()
                           && reassignThreadPool->allClear();
                }

                inline bool allFinishedExceptReassign()
                {
                    return appendThreadPool->allClear();
                }

                inline bool reassignFinished()
                {
                    return reassignThreadPool->allClear();
                }

                void GetStatus(SizeType* appendJobsNum, SizeType* reassignJobsNum) {
                    *appendJobsNum = appendThreadPool->jobsize();
                    *reassignJobsNum = reassignThreadPool->jobsize();
                }
            };

            struct EdgeInsert
            {
                EdgeInsert() : headID(INT64_MAX), fullID(INT64_MAX), distance(INT64_MAX), order(0) {}
                uint64_t headID;
                uint64_t fullID;
                float distance;
                char order;
            };

            struct EdgeCompareInsert
                {
                    bool operator()(const EdgeInsert& a, int b) const
                    {
                        return a.headID < b;
                    };

                    bool operator()(int a, const EdgeInsert& b) const
                    {
                        return a < b.headID;
                    };

                    bool operator()(const EdgeInsert& a, const EdgeInsert& b) const
                    {
                        if (a.headID == b.headID)
                        {
                            if (a.distance == b.distance)
                            {
                                return a.fullID < b.fullID;
                            }

                            return a.distance < b.distance;
                        }

                        return a.headID < b.headID;
                    };
                } g_edgeComparerInsert;

        private:
            std::shared_ptr<VectorIndex> m_index;
            std::shared_ptr<std::uint64_t> m_vectorTranslateMap;
            std::unordered_map<std::string, std::string> m_headParameters;
            //std::unique_ptr<std::shared_timed_mutex[]> m_rwLocks;
            std::vector<std::string> m_postingVecs;
            COMMON::FineGrainedRWLock m_rwLocks;

            // std::unique_ptr<std::atomic_uint32_t[]> m_postingSizes;
            COMMON::PostingSizeRecord m_postingSizes;

            std::shared_ptr<IExtraSearcher> m_extraSearcher;
            std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>> m_workSpaceFactory;

            Options m_options;

            std::function<float(const T*, const T*, DimensionType)> m_fComputeDistance;
            int m_iBaseSquare;
            
            int m_metaDataSize;

            std::shared_ptr<Dispatcher> m_dispatcher;
            std::shared_ptr<PersistentBuffer> m_persistentBuffer;
            std::shared_ptr<Helper::ThreadPool> m_threadPool;
            std::shared_ptr<ThreadPool> m_splitThreadPool;
            std::shared_ptr<ThreadPool> m_reassignThreadPool;

            COMMON::VersionLabel m_versionMap;
            //COMMON::Labelset m_reassignedID;

            tbb::concurrent_hash_map<SizeType, SizeType> m_reassignMap;
            tbb::concurrent_queue<int> m_assignmentQueue;

            std::atomic_uint32_t m_headMiss{0};
            uint32_t m_appendTaskNum{0};
            uint32_t m_splitNum{0};
            uint32_t m_theSameHeadNum{0};
            uint32_t m_reAssignNum{0};
            uint32_t m_garbageNum{0};
            uint64_t m_reAssignScanNum{0};

            //Split
            double m_splitCost{0};
            double m_clusteringCost{0};
            double m_updateHeadCost{0};
            double m_reassignScanCost{0};
            double m_reassignScanIOCost{0};

            // Append
            double m_appendCost{0};
            double m_appendIOCost{0};

            // reAssign
            double m_reAssignCost{0};
            double m_selectCost{0};
            double m_reAssignAppendCost{0};

            // GC
            double m_garbageCost{0};
            std::mutex m_dataAddLock;
        public:
                static thread_local std::shared_ptr<ExtraWorkSpace> m_workspace;
        public:
            Index()
            {
                m_workSpaceFactory = std::make_unique<SPTAG::COMMON::ThreadLocalWorkSpaceFactory<ExtraWorkSpace>>();
                m_fComputeDistance = std::function<float(const T*, const T*, DimensionType)>(COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod));
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
                m_metaDataSize = sizeof(int) + sizeof(uint8_t);
            }

            ~Index() {}

            inline std::shared_ptr<VectorIndex> GetMemoryIndex() { return m_index; }
            inline std::shared_ptr<IExtraSearcher> GetDiskIndex() { return m_extraSearcher; }
            inline Options* GetOptions() { return &m_options; }

            inline SizeType GetNumSamples() const { return m_vectorNum.load(); }
            inline DimensionType GetFeatureDim() const { return m_pQuantizer ? m_pQuantizer->ReconstructDim() : m_index->GetFeatureDim(); }
            inline SizeType GetValueSize() const { return m_options.m_dim * sizeof(T); }

            inline int GetCurrMaxCheck() const { return m_options.m_maxCheck; }
            inline int GetNumThreads() const { return m_options.m_iSSDNumberOfThreads; }
            inline DistCalcMethod GetDistCalcMethod() const { return m_options.m_distCalcMethod; }
            inline IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::SPANN; }
            inline VectorValueType GetVectorValueType() const { return GetEnumValueType<T>(); }

            void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer);
            
            void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer);

            inline float AccurateDistance(const void* pX, const void* pY) const { 
                if (m_options.m_distCalcMethod == DistCalcMethod::L2) return m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim);

                float xy = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim);
                float xx = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pX, m_options.m_dim);
                float yy = m_iBaseSquare - m_fComputeDistance((const T*)pY, (const T*)pY, m_options.m_dim);
                return 1.0f - xy / (sqrt(xx) * sqrt(yy));
            }
            inline float ComputeDistance(const void* pX, const void* pY) const { return m_fComputeDistance((const T*)pX, (const T*)pY, m_options.m_dim); }
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
            ErrorCode SearchIndexWithFilter(QueryResult& p_query, std::function<bool(const ByteArray&)> filterFunc, int maxCheck = 0, bool p_searchDeleted = false) const;
            ErrorCode SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats = nullptr) const;
            ErrorCode DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
                SearchStats* p_stats = nullptr, std::set<int>* truth = nullptr, std::map<int, std::set<int>>* found = nullptr) const;
            ErrorCode UpdateIndex();

            ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr);
            std::string GetParameter(const char* p_param, const char* p_section = nullptr) const;

            inline const void* GetSample(const SizeType idx) const { return nullptr; }
            inline SizeType GetNumDeleted() const { return 0; }
            inline bool NeedRefine() const { return false; }
            inline bool CheckIdDeleted(const SizeType& p_id) { return m_versionMap.Contains(p_id); }
            inline bool CheckVersionValid(const SizeType& p_id, const uint8_t version) {return m_versionMap.GetVersion(p_id) == version;}

            ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const { return ErrorCode::Undefined; }
            ErrorCode SearchTree(QueryResult& p_query) const { return ErrorCode::Undefined; }
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false);
            ErrorCode AddIndexId(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, int& beginHead, int& endHead)  { return ErrorCode::Undefined; }
            ErrorCode AddIndexIdx(SizeType begin, SizeType end) { return ErrorCode::Undefined; }
            ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum) { return ErrorCode::Undefined; }
            ErrorCode DeleteIndex(const SizeType& p_id);
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
            bool SelectHead(std::shared_ptr<Helper::VectorSetReader>& p_reader);

            template <typename InternalDataType>
            bool SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader>&p_reader);

            ErrorCode BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader);

            ErrorCode Append(SizeType headID, int appendNum, std::string& appendPosting);
            ErrorCode Split(const SizeType headID);
            ErrorCode ReAssign(SizeType headID, std::vector<std::string>& postingLists, std::vector<SizeType>& newHeadsID);
            void ReAssignVectors(std::map<SizeType, T*>& reAssignVectors, std::map<SizeType, SizeType>& HeadPrevs, std::map<SizeType, uint8_t>& versions);
            bool ReAssignUpdate(const std::shared_ptr<std::string>&, SizeType VID, SizeType HeadPrev, uint8_t version);

        public:
            // inline void AppendAsync(SizeType headID, int appendNum, std::shared_ptr<std::string> appendPosting, std::function<void()> p_callback=nullptr)
            // {
            //     auto* curJob = new AppendAsyncJob(this, headID, appendNum, std::move(appendPosting), p_callback);
            //     m_splitThreadPool->add(curJob);
            // }
            inline void SplitAsync(SizeType headID, std::function<void()> p_callback=nullptr)
            {
                auto* curJob = new SplitAsyncJob(this, headID, p_callback);
                m_splitThreadPool->add(curJob);
            }

            inline void ReassignAsync(std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version, std::function<void()> p_callback=nullptr)
            {   
                auto* curJob = new ReassignAsyncJob(this, std::move(vectorContain), VID, HeadPrev, version, p_callback);
                m_splitThreadPool->add(curJob);
            }

            void ProcessAsyncReassign(std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version, std::function<void()> p_callback);

            // bool AllFinished() {return m_dispatcher->allFinished() && m_assignmentQueue.empty();}
            bool AllFinished() {return m_splitThreadPool->allClear() && m_reassignThreadPool->allClear();}

            // bool AllFinishedExceptReassign() {return m_dispatcher->allFinishedExceptReassign() && m_assignmentQueue.empty();}

            bool ReassignFinished() {return m_reassignThreadPool->allClear();}

            void ForceCompaction() {if (m_options.m_useKV) m_extraSearcher->ForceCompaction();}

            int getSplitTimes() {return m_splitNum;}

            int getHeadMiss() {return m_headMiss.load();}

            int getSameHead() {return m_theSameHeadNum;}

            int getReassignNum() {return m_reAssignNum;}
            
            int getGarbageNum() {return m_garbageNum;}

            unsigned long getReAssignScanNum() {return m_reAssignScanNum;}

            void GetSplitReassignPoolStatus(int* splitJobs, int* reassignJobs) 
            {
                *splitJobs = m_splitThreadPool->jobsize();
                *reassignJobs = m_reassignThreadPool->jobsize();
            }

            void UpdateStop()
            {
                m_persistentBuffer->StopPB();
                m_dispatcher->stop();
            }

            void PrintUpdateCostStatus()
            {
                LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendCost, m_appendCost/m_appendTaskNum);
                LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, AppendIO TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendIOCost, m_appendIOCost/m_appendTaskNum);
                LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_splitCost, m_splitCost/m_splitNum);
                LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Clustering TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_clusteringCost, m_clusteringCost/m_splitNum);
                LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, UpdateHead TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_updateHeadCost, m_updateHeadCost/m_splitNum);
                LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScan TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_reassignScanCost, m_reassignScanCost/m_splitNum);
                LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScanIO TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_reassignScanIOCost, m_reassignScanIOCost/m_splitNum);
                LOG(Helper::LogLevel::LL_Info, "GCNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_garbageNum, m_garbageCost, m_garbageCost/m_garbageNum);
                LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignCost, m_reAssignCost/m_reAssignNum);
                LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, Select TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_selectCost, m_selectCost/m_reAssignNum);
                LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, ReassignAppend TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignAppendCost, m_reAssignAppendCost/m_reAssignNum);
            }

            void PrintUpdateStatus(int finishedInsert)
            {
                LOG(Helper::LogLevel::LL_Info, "After %d insertion, head vectors split %d times, head missing %d times, same head %d times, reassign %d times, reassign scan %ld times, garbage collection %d times\n", finishedInsert, getSplitTimes(), getHeadMiss(), getSameHead(), getReassignNum(), getReAssignScanNum(), getGarbageNum());
            }

            void ResetUpdateStatus()
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

            void Rebuild(std::shared_ptr<Helper::VectorSetReader>& p_reader, SizeType upperBound = -1)
            {
                auto fullVectors = p_reader->GetVectorSet();
                int curCount;
                if (upperBound == -1) {
                    curCount = fullVectors->Count();
                } else {
                    curCount = upperBound;
                }
                LOG(Helper::LogLevel::LL_Info, "Rebuild SSD Index.\n");
                auto rebuildTimeBegin = std::chrono::high_resolution_clock::now();
                std::vector<EdgeInsert> selections(static_cast<size_t>(curCount)* m_options.m_replicaCount);

                std::vector<int> replicaCount(curCount, 0);
                std::vector<std::atomic_int> postingListSize(m_index->GetNumSamples());
                for (auto& pls : postingListSize) pls = 0;
                LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");

                std::vector<std::thread> threads;
                threads.reserve(m_options.m_iSSDNumberOfThreads);

                std::atomic_int nextFullID(0);
                std::atomic_size_t rngFailedCountTotal(0);

                for (int tid = 0; tid <  m_options.m_iSSDNumberOfThreads; ++tid)
                {
                    threads.emplace_back([&, tid]()
                        {
                            COMMON::QueryResultSet<T> resultSet(NULL, m_options.m_internalResultNum);

                            size_t rngFailedCount = 0;

                            while (true)
                            {
                                int fullID = nextFullID.fetch_add(1);
                                if (fullID >= curCount)
                                {
                                    break;
                                }

                                T* buffer = reinterpret_cast<T*>(fullVectors->GetVector(fullID));
                                resultSet.SetTarget(buffer);
                                resultSet.Reset();

                                m_index->SearchIndex(resultSet);

                                size_t selectionOffset = static_cast<size_t>(fullID)* m_options.m_replicaCount;

                                BasicResult* queryResults = resultSet.GetResults();
                                for (int i = 0; i < m_options.m_internalResultNum && replicaCount[fullID] < m_options.m_replicaCount; ++i)
                                {
                                    if (queryResults[i].VID == -1)
                                    {
                                        break;
                                    }

                                    // RNG Check.
                                    bool rngAccpeted = true;
                                    for (int j = 0; j < replicaCount[fullID]; ++j)
                                    {
                                        // VQANNSearch::QueryResultSet<ValueType> resultSet(NULL, candidateNum);

                                        float nnDist = m_index->ComputeDistance(
                                            m_index->GetSample(queryResults[i].VID),
                                            m_index->GetSample(selections[selectionOffset + j].headID));

                                        // LOG(Helper::LogLevel::LL_Info,  "NNDist: %f Original: %f\n", nnDist, queryResults[i].Score);
                                        if (m_options.m_rngFactor * nnDist <= queryResults[i].Dist)
                                        {
                                            rngAccpeted = false;
                                            break;
                                        }
                                    }

                                    if (!rngAccpeted)
                                    {
                                        ++rngFailedCount;
                                        continue;
                                    }

                                    ++postingListSize[queryResults[i].VID];
                                    selections[selectionOffset + replicaCount[fullID]].headID = queryResults[i].VID;
                                    selections[selectionOffset + replicaCount[fullID]].fullID = fullID;
                                    selections[selectionOffset + replicaCount[fullID]].distance = queryResults[i].Dist;
                                    selections[selectionOffset + replicaCount[fullID]].order = (char)replicaCount[fullID];
                                    ++replicaCount[fullID];
                                }
                                // if (replicaCount[fullID] == 1) {
                                //     for (int i = 0; i < m_options.m_internalResultNum; i++) {
                                //         for (int j = 0; j < replicaCount[fullID]; ++j)
                                //         {
                                //             float nnDist = m_index->ComputeDistance(
                                //                 m_index->GetSample(queryResults[i].VID),
                                //                 m_index->GetSample(selections[selectionOffset + j].headID));

                                //             LOG(Helper::LogLevel::LL_Info, "VID: %d, The %d th head dist: %f, rng check %d: %f\n", fullID, i, queryResults[i].Dist, j, nnDist);
                                //         }
                                //     }
                                // }
                            }

                            rngFailedCountTotal += rngFailedCount;
                        });
                }

                for (int tid = 0; tid < m_options.m_iSSDNumberOfThreads; ++tid)
                {
                    threads[tid].join();
                }

                LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. RNG failed count: %llu\n", static_cast<uint64_t>(rngFailedCountTotal.load()));

                std::sort(selections.begin(), selections.end(), g_edgeComparerInsert);

                {
                    std::vector<int> replicaCountDist(m_options.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        ++replicaCountDist[replicaCount[i]];
                    }

                    LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < postingListSize.size(); ++i)
                {
                    if (postingListSize[i] <= m_extraSearcher->GetPostingSizeLimit()) continue;

                    std::size_t selectIdx = std::lower_bound(selections.begin(), selections.end(), i, g_edgeComparerInsert) - selections.begin();

                    for (size_t dropID = m_extraSearcher->GetPostingSizeLimit(); dropID < postingListSize[i]; ++dropID)
                    {
                        int tonode = selections[selectIdx + dropID].fullID;
                        --replicaCount[tonode];
                    }
                    postingListSize[i] = m_extraSearcher->GetPostingSizeLimit();
                }

                {
                    std::vector<int> replicaCountDist(m_options.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        ++replicaCountDist[replicaCount[i]];
                    }

                    LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }
                m_postingVecs.clear();
                m_postingVecs.resize(m_index->GetNumSamples());
                #pragma omp parallel for num_threads(10)
                for (int id = 0; id < postingListSize.size(); id++) 
                {
                    std::string postinglist;
                    std::size_t selectIdx = std::lower_bound(selections.begin(), selections.end(), id, g_edgeComparerInsert)
                                            - selections.begin();
                    for (int j = 0; j < postingListSize[id]; ++j) {
                        if (selections[selectIdx].headID != id) {
                            LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                            exit(1);
                        }
                        float distance = selections[selectIdx].distance;
                        int fullID = selections[selectIdx++].fullID;
                        uint8_t version = 0;
                        m_versionMap.UpdateVersion(fullID, 0);
                        size_t dim = fullVectors->Dimension();
                        // First Vector ID, then Vector
                        postinglist += Helper::Convert::Serialize<int>(&fullID, 1);
                        postinglist += Helper::Convert::Serialize<uint8_t>(&version, 1);
                        // postinglist += Helper::Convert::Serialize<float>(&distance, 1);
                        postinglist += Helper::Convert::Serialize<T>(fullVectors->GetVector(fullID), dim);
                    }
                    m_extraSearcher->OverrideIndex(id, postinglist);
                    // m_postingVecs[id] = postinglist;
                    m_postingSizes.UpdateSize(id, postingListSize[id]);
                }
                auto rebuildTimeEnd = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(rebuildTimeEnd - rebuildTimeBegin).count();
                LOG(Helper::LogLevel::LL_Info, "rebuild cost: %.2lf s\n", elapsedSeconds);
            }

            void QuantifyAssumptionBrokenTotally()
            {

                std::atomic_int32_t isIn64(0); 
                int vectorNum = m_vectorNum.load();
                std::vector<std::set<SizeType>> vectorHeadMap(vectorNum);
                std::vector<bool> vectorFoundMap(vectorNum);
                std::vector<std::string> vectorIdValueMap(vectorNum);
                for (int i = 0; i < m_index->GetNumSamples(); i++) {
                    std::string postingList;
                    if (!m_index->ContainSample(i)) continue;
                    m_extraSearcher->SearchIndex(i, postingList);
                    int postVectorNum = postingList.size() / (m_options.m_dim * sizeof(T) + m_metaDataSize);
                    uint8_t* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                    for (int j = 0; j < postVectorNum; j++) {
                        uint8_t* vectorId = postingP + j * (m_options.m_dim * sizeof(T) + m_metaDataSize);
                        SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                        if (m_versionMap.Contains(vid)) continue;
                        vectorHeadMap[vid].insert(i);
                        if (vectorFoundMap[vid]) continue;
                        vectorFoundMap[vid] = true;
                        vectorIdValueMap[vid] = Helper::Convert::Serialize<uint8_t>(vectorId + m_metaDataSize, m_options.m_dim * sizeof(T));
                    }
                }
                int totalNum = 0;
                for (int i = 0; i < vectorNum; i++) {
                    if (vectorFoundMap[i]) totalNum++;
                    else LOG(Helper::LogLevel::LL_Info, "missing VID: %d\n", i);
                }
                LOG(Helper::LogLevel::LL_Info, "Acutall Vector Number: %d\n", totalNum);
                #pragma omp parallel for num_threads(32)
                for (int vid = 0; vid < vectorNum; vid++) {
                    COMMON::QueryResultSet<T> headCandidates(NULL, 64);
                    headCandidates.SetTarget(reinterpret_cast<T*>(&vectorIdValueMap[vid].front()));
                    headCandidates.Reset();
                    m_index->SearchIndex(headCandidates);
                    int replicaCount = 0;
                    BasicResult* queryResults = headCandidates.GetResults();
                    std::vector<EdgeInsert> selections(static_cast<size_t>(m_options.m_replicaCount));
                    std::set<SizeType>HeadMap;
                    for (int i = 0; i < headCandidates.GetResultNum() && replicaCount < m_options.m_replicaCount; ++i) {
                        if (queryResults[i].VID == -1) {
                            break;
                        }
                        if (vectorHeadMap[vid].count(queryResults[i].VID)) {
                            isIn64++;
                            break;
                        }
                    }
                }
                LOG(Helper::LogLevel::LL_Info, "%d/%d Vectors has at least 1 replica in top64\n", isIn64.load(), vectorNum);
            }

            //headCandidates: search data structrue for "vid" vector
            //headID: the head vector that stands for vid
            bool IsAssumptionBroken(SizeType headID, COMMON::QueryResultSet<T>& headCandidates, SizeType vid)
            {
                m_index->SearchIndex(headCandidates);
                int replicaCount = 0;
                BasicResult* queryResults = headCandidates.GetResults();
                std::vector<EdgeInsert> selections(static_cast<size_t>(m_options.m_replicaCount));
                for (int i = 0; i < headCandidates.GetResultNum() && replicaCount < m_options.m_replicaCount; ++i) {
                    if (queryResults[i].VID == -1) {
                        break;
                    }
                    // RNG Check.
                    bool rngAccpeted = true;
                    for (int j = 0; j < replicaCount; ++j) {
                        float nnDist = m_index->ComputeDistance(
                                                    m_index->GetSample(queryResults[i].VID),
                                                    m_index->GetSample(selections[j].headID));
                        if (nnDist <= queryResults[i].Dist) {
                            rngAccpeted = false;
                            break;
                        }
                    }
                    if (!rngAccpeted)
                        continue;

                    selections[replicaCount].headID = queryResults[i].VID;
                    // LOG(Helper::LogLevel::LL_Info, "head:%d\n", queryResults[i].VID);
                    if (selections[replicaCount].headID == headID) return false;
                    ++replicaCount;
                }
                return true;
            }

            //Measure that in "headID" posting list, how many vectors break their assumption
            int QuantifyAssumptionBroken(SizeType headID, std::string& postingList, SizeType SplitHead, std::vector<SizeType>& newHeads, std::set<int>& brokenID, int topK = 0, float ratio = 1.0)
            {
                int assumptionBrokenNum = 0;
                int m_vectorInfoSize = sizeof(T) * m_options.m_dim + m_metaDataSize;
                int postVectorNum = postingList.size() / m_vectorInfoSize;
                uint8_t* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                float minDist;
                float maxDist;
                float avgDist = 0;
                std::vector<float> distanceSet;
                //#pragma omp parallel for num_threads(32)
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<int*>(vectorId));
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    // float_t dist = *(reinterpret_cast<float*>(vectorId + sizeof(int) + sizeof(uint8_t)));
                    float_t dist = m_index->ComputeDistance(reinterpret_cast<T*>(vectorId + m_metaDataSize), m_index->GetSample(headID));
                    // if (dist < Epsilon) LOG(Helper::LogLevel::LL_Info, "head found: vid: %d, head: %d\n", vid, headID);
                    avgDist += dist;
                    distanceSet.push_back(dist);
                    if (CheckIdDeleted(vid) || !CheckVersionValid(vid, version)) continue;
                    COMMON::QueryResultSet<T> headCandidates(NULL, 64);
                    headCandidates.SetTarget(reinterpret_cast<T*>(vectorId + m_metaDataSize));
                    headCandidates.Reset();
                    if (brokenID.find(vid) == brokenID.end() && IsAssumptionBroken(headID, headCandidates, vid)) {
                        /*
                        float_t headDist = m_index->ComputeDistance(headCandidates.GetTarget(), m_index->GetSample(SplitHead));
                        float_t newHeadDist_1 = m_index->ComputeDistance(headCandidates.GetTarget(), m_index->GetSample(newHeads[0]));
                        float_t newHeadDist_2 = m_index->ComputeDistance(headCandidates.GetTarget(), m_index->GetSample(newHeads[1]));

                        float_t splitDist = m_index->ComputeDistance(m_index->GetSample(SplitHead), m_index->GetSample(headID));

                        float_t headToNewHeadDist_1 = m_index->ComputeDistance(m_index->GetSample(headID), m_index->GetSample(newHeads[0]));
                        float_t headToNewHeadDist_2 = m_index->ComputeDistance(m_index->GetSample(headID), m_index->GetSample(newHeads[1]));

                        LOG(Helper::LogLevel::LL_Info, "broken vid to head distance: %f, to split head distance: %f\n", dist, headDist);
                        LOG(Helper::LogLevel::LL_Info, "broken vid to new head 1 distance: %f, to new head 2 distance: %f\n", newHeadDist_1, newHeadDist_2);
                        LOG(Helper::LogLevel::LL_Info, "head to spilit head distance: %f\n", splitDist);
                        LOG(Helper::LogLevel::LL_Info, "head to new head 1 distance: %f, to new head 2 distance: %f\n", headToNewHeadDist_1, headToNewHeadDist_2);
                        */
                        assumptionBrokenNum++;
                        brokenID.insert(vid);
                    }
                }
                
                if (assumptionBrokenNum != 0) {
                    std::sort(distanceSet.begin(), distanceSet.end());
                    minDist = distanceSet[1];
                    maxDist = distanceSet[distanceSet.size() - 1];
                    // LOG(Helper::LogLevel::LL_Info, "distance: min: %f, max: %f, avg: %f, 50th: %f\n", minDist, maxDist, avgDist/postVectorNum, distanceSet[distanceSet.size() * 0.5]);
                    // LOG(Helper::LogLevel::LL_Info, "assumption broken num: %d\n", assumptionBrokenNum);
                    float_t splitDist = m_index->ComputeDistance(m_index->GetSample(SplitHead), m_index->GetSample(headID));

                    float_t headToNewHeadDist_1 = m_index->ComputeDistance(m_index->GetSample(headID), m_index->GetSample(newHeads[0]));
                    float_t headToNewHeadDist_2 = m_index->ComputeDistance(m_index->GetSample(headID), m_index->GetSample(newHeads[1]));

                    // LOG(Helper::LogLevel::LL_Info, "head to spilt head distance: %f/%d/%.2f\n", splitDist, topK, ratio);
                    // LOG(Helper::LogLevel::LL_Info, "head to new head 1 distance: %f, to new head 2 distance: %f\n", headToNewHeadDist_1, headToNewHeadDist_2);
                }
                
                return assumptionBrokenNum;
            }

            int QuantifySplitCaseA(std::vector<SizeType>& newHeads, std::vector<std::string>& postingLists, SizeType SplitHead, int split_order, std::set<int>& brokenID)
            {
                int assumptionBrokenNum = 0;
                assumptionBrokenNum += QuantifyAssumptionBroken(newHeads[0], postingLists[0], SplitHead, newHeads, brokenID);
                assumptionBrokenNum += QuantifyAssumptionBroken(newHeads[1], postingLists[1], SplitHead, newHeads, brokenID);
                int vectorNum = (postingLists[0].size() + postingLists[1].size()) / (sizeof(T) * m_options.m_dim + m_metaDataSize);
                LOG(Helper::LogLevel::LL_Info, "After Split%d, Top0 nearby posting lists, caseA : %d/%d\n", split_order, assumptionBrokenNum, vectorNum);
                return assumptionBrokenNum;
            }

            //Measure that around "headID", how many vectors break their assumption
            //"headID" is the head vector before split
            void QuantifySplitCaseB(SizeType headID, std::vector<SizeType>& newHeads, SizeType SplitHead, int split_order, int assumptionBrokenNum_top0, std::set<int>& brokenID)
            {
                auto headVector = reinterpret_cast<const T*>(m_index->GetSample(headID));
                COMMON::QueryResultSet<T> nearbyHeads(NULL, 64);
                nearbyHeads.SetTarget(headVector);
                nearbyHeads.Reset();
                std::vector<std::string> postingLists;
                m_index->SearchIndex(nearbyHeads);
                std::string postingList;
                BasicResult* queryResults = nearbyHeads.GetResults();
                int topk = 8;
                int assumptionBrokenNum = assumptionBrokenNum_top0;
                int assumptionBrokenNum_topK = assumptionBrokenNum_top0;
                int i;
                int containedHead = 0;
                if (assumptionBrokenNum_top0 != 0) containedHead++;
                int vectorNum = 0;
                float furthestDist = 0;
                for (i = 0; i < nearbyHeads.GetResultNum(); i++) {
                    if (queryResults[i].VID == -1) {
                        break;
                    }
                    furthestDist = queryResults[i].Dist;
                    if (i == topk) {
                        LOG(Helper::LogLevel::LL_Info, "After Split%d, Top%d nearby posting lists, caseB : %d in %d/%d\n", split_order, i, assumptionBrokenNum, containedHead, vectorNum);
                        topk *= 2;
                    }
                    if (queryResults[i].VID == newHeads[0] || queryResults[i].VID == newHeads[1]) continue;
                    m_extraSearcher->SearchIndex(queryResults[i].VID, postingList);
                    vectorNum += postingList.size() / (sizeof(T) * m_options.m_dim + m_metaDataSize);
                    int tempNum = QuantifyAssumptionBroken(queryResults[i].VID, postingList, SplitHead, newHeads, brokenID, i, queryResults[i].Dist/queryResults[1].Dist);
                    assumptionBrokenNum += tempNum;
                    if (tempNum != 0) containedHead++;
                }
                LOG(Helper::LogLevel::LL_Info, "After Split%d, Top%d nearby posting lists, caseB : %d in %d/%d\n", split_order, i, assumptionBrokenNum, containedHead, vectorNum);
            }

            void QuantifySplit(SizeType headID, std::vector<std::string>& postingLists, std::vector<SizeType>& newHeads, SizeType SplitHead, int split_order)
            {
                std::set<int> brokenID;
                brokenID.clear();
                // LOG(Helper::LogLevel::LL_Info, "Split Quantify: %d, head1:%d, head2:%d\n", split_order, newHeads[0], newHeads[1]);
                int assumptionBrokenNum = QuantifySplitCaseA(newHeads, postingLists, SplitHead, split_order, brokenID);
                QuantifySplitCaseB(headID, newHeads, SplitHead, split_order, assumptionBrokenNum, brokenID);
            }

            bool CheckIsNeedReassign(std::vector<SizeType>& newHeads, T* data, SizeType splitHead, float_t headToSplitHeadDist, float_t currentHeadDist, bool isInSplitHead, SizeType currentHead)
            {
                
                float_t splitHeadDist = m_index->ComputeDistance(data, m_index->GetSample(splitHead));

                if (isInSplitHead) {
                    if (splitHeadDist >= currentHeadDist) return false;
                } else {
                    float_t newHeadDist_1 = m_index->ComputeDistance(data, m_index->GetSample(newHeads[0]));
                    float_t newHeadDist_2 = m_index->ComputeDistance(data, m_index->GetSample(newHeads[1]));
                    if (splitHeadDist <= newHeadDist_1 && splitHeadDist <= newHeadDist_2) return false;
                    if (currentHeadDist <= newHeadDist_1 && currentHeadDist <= newHeadDist_2) return false;
                }
                return true;
            }

            int GetNextAssignment() 
            {
                int assignId;
                if (m_assignmentQueue.try_pop(assignId)) {
                    return assignId;
                }
                return -1;
            }

            void CalculatePostingDistribution()
            {
                if (m_options.m_inPlace) return;
                int top = m_extraSearcher->GetPostingSizeLimit() / 10 + 1;
                int page = m_options.m_postingPageLimit + 1;
                std::vector<int> lengthDistribution(top, 0);
                std::vector<int> sizeDistribution(page + 2, 0);
                size_t vectorInfoSize = m_options.m_dim * sizeof(T) + m_metaDataSize;
                int deletedHead = 0;
                for (int i = 0; i < m_index->GetNumSamples(); i++) {
                    if (!m_index->ContainSample(i)) deletedHead++;
                    lengthDistribution[m_postingSizes.GetSize(i)/10]++;
                    int size = m_postingSizes.GetSize(i) * vectorInfoSize;
                    if (size < PageSize) {
                        if (size < 512) sizeDistribution[0]++;
                        else if (size < 1024) sizeDistribution[1]++;
                        else sizeDistribution[2]++;
                    } else {
                        sizeDistribution[size/PageSize + 2]++;
                    }
                }
                LOG(Helper::LogLevel::LL_Info, "Posting Length (Vector Num):\n");
                for (int i = 0; i < top; ++i)
                {
                    LOG(Helper::LogLevel::LL_Info, "%d ~ %d: %d, \n", i * 10, (i + 1) * 10 - 1, lengthDistribution[i]);
                }
                LOG(Helper::LogLevel::LL_Info, "Posting Length (Data Size):\n");
                for (int i = 0; i < page + 2; ++i)
                {
                    if (i <= 2) {
                        if (i == 0) LOG(Helper::LogLevel::LL_Info, "0 ~ 512 B: %d, \n", sizeDistribution[0] - deletedHead);
                        else if (i == 1) LOG(Helper::LogLevel::LL_Info, "512 B ~ 1 KB: %d, \n", sizeDistribution[1]);
                        else LOG(Helper::LogLevel::LL_Info, "1 KB ~ 4 KB: %d, \n", sizeDistribution[2]);
                    }
                    else
                        LOG(Helper::LogLevel::LL_Info, "%d ~ %d KB: %d, \n", (i - 2) * 4, (i - 1) * 4, sizeDistribution[i]);
                }
            }

            void PreReassign(std::shared_ptr<Helper::VectorSetReader>& p_reader) 
            {
                LOG(Helper::LogLevel::LL_Info, "Begin PreReassign\n");
                std::atomic_bool doneReassign = false;
                // m_index->UpdateIndex();
                // m_postingVecs.clear();
                // m_postingVecs.resize(m_index->GetNumSamples());
                // LOG(Helper::LogLevel::LL_Info, "Setting\n");
                // #pragma omp parallel for num_threads(32)
                // for (int i = 0; i < m_index->GetNumSamples(); i++) {
                //     m_extraSearcher->SearchIndex(i, m_postingVecs[i]);
                // }
                LOG(Helper::LogLevel::LL_Info, "Into PreReassign Loop\n");
                while (!doneReassign) {
                    auto preReassignTimeBegin = std::chrono::high_resolution_clock::now();
                    doneReassign = true;
                    std::vector<std::thread> threads;
                    std::atomic_int nextPostingID(0);
                    int currentPostingNum = m_index->GetNumSamples();
                    int limit = m_extraSearcher->GetPostingSizeLimit() * m_options.m_preReassignRatio;
                    LOG(Helper::LogLevel::LL_Info,"Batch PreReassign, Current PostingNum: %d, Current Limit: %d\n", currentPostingNum, limit);
                    auto func = [&]()
                    {
                        int index = 0;
                        while (true)
                        {
                            index = nextPostingID.fetch_add(1);
                            if (index < currentPostingNum) 
                            {
                                if ((index & ((1 << 14) - 1)) == 0)
                                {
                                    LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / currentPostingNum);
                                }
                                if (m_postingSizes.GetSize(index) >= limit) 
                                {
                                    doneReassign = false;
                                    // std::string postingList = m_postingVecs[index];
                                    std::string postingList;
                                    m_extraSearcher->SearchIndex(index, postingList);
                                    auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                                    size_t vectorInfoSize = m_options.m_dim * sizeof(T) + m_metaDataSize;
                                    size_t postVectorNum = postingList.size() / vectorInfoSize;
                                    COMMON::Dataset<T> smallSample;  // smallSample[i] -> VID
                                    std::shared_ptr<uint8_t> vectorBuffer(new uint8_t[m_options.m_dim * sizeof(T) * postVectorNum], std::default_delete<uint8_t[]>());
                                    std::vector<int> localIndices(postVectorNum);
                                    auto vectorBuf = vectorBuffer.get();
                                    for (int j = 0; j < postVectorNum; j++)
                                    {
                                        uint8_t* vectorId = postingP + j * vectorInfoSize;
                                        localIndices[j] = j;
                                        memcpy(vectorBuf, vectorId + m_metaDataSize, m_options.m_dim * sizeof(T));
                                        vectorBuf += m_options.m_dim * sizeof(T);
                                    }
                                    smallSample.Initialize(postVectorNum, m_options.m_dim, m_index->m_iDataBlockSize, m_index->m_iDataCapacity, reinterpret_cast<T*>(vectorBuffer.get()), false);
                                    SPTAG::COMMON::KmeansArgs<T> args(2, smallSample.C(), (SizeType)localIndices.size(), 1, m_index->GetDistCalcMethod());
                                    std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));
                                    int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr, m_options.m_virtualHead);
                                    bool theSameHead = false;
                                    for (int k = 0; k < 2; k++) {
                                        if (args.counts[k] == 0)	continue;
                                        if (!theSameHead && m_index->ComputeDistance(args.centers + k * args._D, m_index->GetSample(index)) < Epsilon) {
                                            theSameHead = true;
                                        }
                                        else {
                                            int begin, end = 0;
                                            m_index->AddIndexId(args.centers + k * args._D, 1, m_options.m_dim, begin, end);
                                            m_index->AddIndexIdx(begin, end);
                                            {
                                                std::lock_guard<std::mutex> lock(m_dataAddLock);
                                                auto ret = m_postingSizes.AddBatch(1);
                                                if (ret == ErrorCode::MemoryOverFlow) {
                                                    LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: newHeadVID: %d, Map Size:%d\n", begin, m_postingSizes.BufferSize());
                                                    exit(1);
                                                }
                                            }
                                        }
                                    }
                                    if (!theSameHead) {
                                        m_index->DeleteIndex(index);
                                    }
                                }
                            }
                            else 
                            {
                                return;
                            }
                        }
                    };
                    for (int j = 0; j < m_options.m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
                    for (auto& thread : threads) { thread.join(); }
                    auto preReassignTimeEnd = std::chrono::high_resolution_clock::now();
                    double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(preReassignTimeEnd - preReassignTimeBegin).count();
                    LOG(Helper::LogLevel::LL_Info, "rebuild cost: %.2lf s\n", elapsedSeconds);
                    m_index->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder);
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
                    m_postingSizes.Save(m_options.m_ssdInfoFile);
                    for (int i = 0; i < m_index->GetNumSamples(); i++) {
                        m_extraSearcher->DeleteIndex(i);
                    }
                    ForceCompaction();
                    Rebuild(p_reader);
                    ForceCompaction();
                    CalculatePostingDistribution();
                }
                return;
            }

            void GetDBStat() 
            {
                m_extraSearcher->GetDBStats();
            }

            int ClusteringSPFresh(const COMMON::Dataset<T>& data, 
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            COMMON::KmeansArgs<T>& args, int tryIters, bool debug, bool virtualCenter)
            {
                int bestCount = -1;
                for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
                    for (int k = 0; k < args._DK; k++) {
                        SizeType randid = COMMON::Utils::rand(last, first);
                        std::memcpy(args.centers + k*args._D, data[indices[randid]], sizeof(T)*args._D);
                    }
                    args.ClearCounts();
                    args.ClearDists(-MaxDist);
                    COMMON::KmeansAssign(data, indices, first, last, args, true, 0);
                    int tempCount = __INT_MAX__;
                    for (int i = 0; i < args._K; i++) if (args.newCounts[i] < tempCount) tempCount = args.newCounts[i];
                    if (tempCount > bestCount) {
                        bestCount = tempCount;
                        memcpy(args.newTCenters, args.centers, sizeof(T)*args._K*args._D);
                        memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);
                    }
                }
                float currDiff, currDist, minClusterDist = MaxDist;
                int noImprovement = 0;
                for (int iter = 0; iter < 100; iter++) {
                    std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
                    std::random_shuffle(indices.begin() + first, indices.begin() + last);
                    args.ClearCenters();
                    args.ClearCounts();
                    args.ClearDists(-MaxDist);
                    currDist = COMMON::KmeansAssign(data, indices, first, last, args, true, 0);
                    std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                    if (currDist < minClusterDist) {
                        noImprovement = 0;
                        minClusterDist = currDist;
                    }
                    else {
                        noImprovement++;
                    }
                    
                    if (debug) {
                        std::string log = "";
                        for (int k = 0; k < args._DK; k++) {
                            log += std::to_string(args.counts[k]) + " ";
                        }
                        LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f counts:%s\n", iter, currDist, log.c_str());
                    }

                    currDiff = COMMON::RefineCenters(data, args);
                    if (debug) LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iter, currDist, currDiff);

                    if (currDiff < 1e-3 || noImprovement >= 5) break;
                }

                if (!virtualCenter) {
                    args.ClearCounts();
                    args.ClearDists(MaxDist);
                    currDist = KmeansAssign(data, indices, first, last, args, false, 0);
                    for (int k = 0; k < args._DK; k++) {
                        if (args.clusterIdx[k] != -1) std::memcpy(args.centers + k * args._D, data[args.clusterIdx[k]], sizeof(T) * args._D);
                    }
                    std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);
                    if (debug) {
                        std::string log = "";
                        for (int k = 0; k < args._DK; k++) {
                            log += std::to_string(args.counts[k]) + " ";
                        }
                        LOG(Helper::LogLevel::LL_Info, "not virtualCenter: dist:%f counts:%s\n", currDist, log.c_str());
                    }
                } 

                args.ClearCounts();
                args.ClearDists(MaxDist);
                currDist = COMMON::KmeansAssign(data, indices, first, last, args, false, 0);
                memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                SizeType maxCount = 0, minCount = (std::numeric_limits<SizeType>::max)(), availableClusters = 0;
                float CountStd = 0.0, CountAvg = (last - first) * 1.0f / args._DK;
                for (int i = 0; i < args._DK; i++) {
                    if (args.counts[i] > maxCount) maxCount = args.counts[i];
                    if (args.counts[i] < minCount) minCount = args.counts[i];
                    CountStd += (args.counts[i] - CountAvg) * (args.counts[i] - CountAvg);
                    if (args.counts[i] > 0) availableClusters++;
                }
                CountStd = sqrt(CountStd / args._DK) / CountAvg;
                if (debug) LOG(Helper::LogLevel::LL_Info, "Max:%d Min:%d Avg:%f Std/Avg:%f Dist:%f NonZero/Total:%d/%d\n", maxCount, minCount, CountAvg, CountStd, currDist, availableClusters, args._DK);
                int numClusters = 0;
                for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;
                args.Shuffle(indices, first, last);
                return numClusters;
            }
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_INDEX_H_
