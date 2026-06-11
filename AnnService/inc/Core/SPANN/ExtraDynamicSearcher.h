// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_
#define _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "ExtraStaticSearcher.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Core/Common/FineGrainedLock.h"
#include "inc/Core/Common/Checksum.h"
#include "PersistentBuffer.h"
#include "inc/Core/Common/PostingSizeRecord.h"
#include "inc/Core/Common/IVersionMap.h"
#include "inc/Core/Common/LocalVersionMap.h"
#include "inc/Core/Common/TiKVVersionMap.h"
#include "ExtraFileController.h"
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <map>
#include <cmath>
#include <cstring>
#include <climits>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <deque>
#include <condition_variable>

#ifdef SPDK
#include "ExtraSPDKController.h"
#endif

#ifdef ROCKSDB
#include "ExtraRocksDBController.h"
// enable rocksdb io_uring
extern "C" bool RocksDbIOUringEnable() { return true; }
#endif

#ifdef TIKV
#include "ExtraTiKVController.h"
#endif

namespace SPTAG::SPANN {

    template <typename ValueType>
    class ExtraDynamicSearcher : public IExtraSearcher
    {
        class MergeAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType m_headID;
            std::function<void()> m_callback;
        public:
            MergeAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_headID(headID), m_callback(std::move(p_callback)) {}

            ~MergeAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->MergePostings((ExtraWorkSpace*)p_workSpace, m_headID);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                m_extraIndex->m_mergeJobsInFlight--;
                m_extraIndex->m_totalMergeCompleted++;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class SplitAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType m_headID;
            std::function<void()> m_callback;
        public:
            SplitAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_headID(headID), m_callback(std::move(p_callback)) {}

            ~SplitAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                auto splitStart = std::chrono::high_resolution_clock::now();
                ErrorCode ret = m_extraIndex->Split((ExtraWorkSpace*)p_workSpace, m_headID);
                auto splitEnd = std::chrono::high_resolution_clock::now();
                uint64_t elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(splitEnd - splitStart).count();
                m_extraIndex->m_totalSplitTimeUs += elapsedUs;
                uint64_t prevMax = m_extraIndex->m_maxSplitTimeUs.load();
                while (elapsedUs > prevMax && !m_extraIndex->m_maxSplitTimeUs.compare_exchange_weak(prevMax, elapsedUs));
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                m_extraIndex->m_splitJobsInFlight--;
                m_extraIndex->m_totalSplitCompleted++;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class AppendAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType m_headID;
            std::shared_ptr<std::string> m_vectorInfo;
            std::function<void()> m_callback;
        public:
            AppendAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, std::shared_ptr<std::string> vectorInfo,  std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_headID(headID), m_vectorInfo(std::move(vectorInfo)), m_callback(std::move(p_callback)) {}

            ~AppendAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->Append((ExtraWorkSpace*)p_workSpace, m_headID, (int)(m_vectorInfo->size() / m_extraIndex->m_vectorInfoSize), *m_vectorInfo);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                m_extraIndex->m_appendJobsInFlight--;
                m_extraIndex->m_totalAppendCompleted++;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class ReassignAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            std::shared_ptr<std::string> m_vectorInfo;
            SizeType m_headPrev;
            std::function<void()> m_callback;
        public:
            ReassignAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex,
                std::shared_ptr<std::string> vectorInfo, SizeType headPrev, std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_vectorInfo(std::move(vectorInfo)), m_headPrev(headPrev), m_callback(std::move(p_callback)) {}

            ~ReassignAsyncJob() {}
            
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }

            void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                auto _rsBegin = std::chrono::high_resolution_clock::now();
                ErrorCode ret = m_extraIndex->Reassign((ExtraWorkSpace*)p_workSpace, m_vectorInfo, m_headPrev);
                auto _rsEnd = std::chrono::high_resolution_clock::now();
                uint64_t _rsUs = std::chrono::duration_cast<std::chrono::microseconds>(_rsEnd - _rsBegin).count();
                IndexStats::HistAdd(m_extraIndex->m_stat.m_reassignJobLatencyUs, _rsUs);
                m_extraIndex->m_stat.m_reassignJobLatencyTotalUs.fetch_add(_rsUs, std::memory_order_relaxed);
                m_extraIndex->m_stat.m_reassignJobSampleCount.fetch_add(1, std::memory_order_relaxed);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                m_extraIndex->m_reassignJobsInFlight--;
                m_extraIndex->m_totalReassignCompleted++;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class SPDKThreadPool : public Helper::ThreadPool
        {
        public:
            void initSPDK(int numberOfThreads, ExtraDynamicSearcher<ValueType>* extraIndex) 
            {
                m_abort.SetAbort(false);
                for (int i = 0; i < numberOfThreads; i++)
                {
                    m_threads.emplace_back([this, extraIndex] {
                        Job *j;
                        ExtraWorkSpace workSpace;
                        extraIndex->GetHeadIndex()->InitWorkSpace(&workSpace);
                        while (get(j))
                        {
                            try 
                            {
                                j->exec(&workSpace, &m_abort);
                            }
                            catch (std::exception& e) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ThreadPool: exception in %s %s\n", typeid(*j).name(), e.what());
                            }
                            delete j;
                            currentJobs--;
                        }
                    });
                }
            }
        };

    private:
        std::shared_ptr<Helper::KeyValueIO> db;

        SPANN::Index<ValueType>* m_headIndex;
        std::unique_ptr<COMMON::IVersionMap> m_versionMap;
        Options* m_opt;
        int m_layer;

        COMMON::FineGrainedRWLock m_rwLocks;

        IndexStats m_stat;

        std::shared_ptr<PersistentBuffer> m_wal;

        std::shared_timed_mutex m_splitListLock;
        Helper::Concurrent::ConcurrentMap<SizeType, int> m_splitList;
        std::atomic_size_t m_splitJobsInFlight{ 0 };
        std::atomic_size_t m_totalSplitSubmitted{ 0 };
        std::atomic_size_t m_totalSplitCompleted{ 0 };
        std::atomic<uint64_t> m_totalSplitTimeUs{ 0 };
        std::atomic<uint64_t> m_maxSplitTimeUs{ 0 };

        std::shared_timed_mutex m_mergeListLock;
        Helper::Concurrent::ConcurrentSet<SizeType> m_mergeList;
        std::atomic_size_t m_mergeJobsInFlight{ 0 };
        std::atomic_size_t m_totalMergeSubmitted{ 0 };
        std::atomic_size_t m_totalMergeCompleted{ 0 };

        std::atomic_size_t m_appendJobsInFlight{ 0 };
        std::atomic_size_t m_totalAppendSubmitted{ 0 };
        std::atomic_size_t m_totalAppendCompleted{ 0 };
        std::atomic_size_t m_totalAppendCount{ 0 };

        std::atomic_size_t m_reassignJobsInFlight{ 0 };
        std::atomic_size_t m_totalReassignSubmitted{ 0 };
        std::atomic_size_t m_totalReassignCompleted{ 0 };

        bool m_allDonePrinted = false;

        std::mutex m_progressLogMutex;
        std::chrono::steady_clock::time_point m_lastProgressLogTime = (std::chrono::steady_clock::time_point::min)();
        size_t m_lastProgressLogQueueSize = (std::numeric_limits<size_t>::max)();
        size_t m_lastProgressLogSplit = (std::numeric_limits<size_t>::max)();
        size_t m_lastProgressLogMerge = (std::numeric_limits<size_t>::max)();
        size_t m_lastProgressLogAppend = (std::numeric_limits<size_t>::max)();
        size_t m_lastProgressLogReassign = (std::numeric_limits<size_t>::max)();

        bool ShouldLogProgress(size_t totalJobs, bool force = false) {
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(m_progressLogMutex);

            size_t splitJobs = m_splitJobsInFlight.load();
            size_t mergeJobs = m_mergeJobsInFlight.load();
            size_t appendJobs = m_appendJobsInFlight.load();
            size_t reassignJobs = m_reassignJobsInFlight.load();
            bool queueChanged = (totalJobs != m_lastProgressLogQueueSize) ||
                               (splitJobs != m_lastProgressLogSplit) ||
                               (mergeJobs != m_lastProgressLogMerge) ||
                               (appendJobs != m_lastProgressLogAppend) ||
                               (reassignJobs != m_lastProgressLogReassign);

            if (force) {
                m_lastProgressLogTime = now;
                m_lastProgressLogQueueSize = totalJobs;
                m_lastProgressLogSplit = splitJobs;
                m_lastProgressLogMerge = mergeJobs;
                m_lastProgressLogAppend = appendJobs;
                m_lastProgressLogReassign = reassignJobs;
                return true;
            }

            bool enoughTimeElapsed =
                (m_lastProgressLogTime == (std::chrono::steady_clock::time_point::min)()) ||
                (std::chrono::duration_cast<std::chrono::seconds>(now - m_lastProgressLogTime).count() >= 5);

            bool shouldLog = queueChanged && enoughTimeElapsed;
            if (shouldLog) {
                m_lastProgressLogTime = now;
                m_lastProgressLogQueueSize = totalJobs;
                m_lastProgressLogSplit = splitJobs;
                m_lastProgressLogMerge = mergeJobs;
                m_lastProgressLogAppend = appendJobs;
                m_lastProgressLogReassign = reassignJobs;
            }
            return shouldLog;
        }

    public:
        ExtraDynamicSearcher(SPANN::Options& p_opt, int layer, SPANN::Index<ValueType>* headIndex, std::shared_ptr<Helper::KeyValueIO> p_db) {
            m_opt = &p_opt;
            m_layer = layer;
            m_headIndex = headIndex;
            m_metaDataSize = sizeof(SizeType) + sizeof(uint8_t);
            m_vectorDataSize = sizeof(ValueType) * m_opt->m_dim;
            m_vectorInfoSize = m_vectorDataSize + m_metaDataSize;
            p_opt.m_postingPageLimit = max(p_opt.m_postingPageLimit, static_cast<int>((p_opt.m_postingVectorLimit * m_vectorInfoSize + PageSize - 1) / PageSize));
            p_opt.m_searchPostingPageLimit = p_opt.m_postingPageLimit;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting index with posting page limit:%d\n", p_opt.m_postingPageLimit);
            m_postingSizeLimit = p_opt.m_postingPageLimit * PageSize / m_vectorInfoSize;
            m_bufferSizeLimit = p_opt.m_bufferLength * PageSize / m_vectorInfoSize;

            if (p_db != nullptr) {
                db = p_db;
            } else {
                m_headIndex->PrepareDB(db, layer);
            }

            // Initialize version map: TiKV-backed or local
#ifdef TIKV
            if (p_opt.m_storage == Storage::TIKVIO && p_opt.m_distributedVersionMap) {
                auto tikvMap = std::make_unique<COMMON::TiKVVersionMap>();
                tikvMap->SetDB(db);
                tikvMap->SetLayer(layer);
                tikvMap->SetChunkSize(p_opt.m_versionChunkSize);
                m_versionMap = std::move(tikvMap);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using distributed TiKV VersionMap (layer=%d, per-VID keys, no local cache)\n",
                    layer);
            } else 
#endif
            {
                m_versionMap = std::make_unique<COMMON::LocalVersionMap>();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using local in-memory VersionMap (layer=%d)\n", layer);
            }

            
            m_hardLatencyLimit = std::chrono::microseconds((int)(p_opt.m_latencyLimit) * 1000);
            m_mergeThreshold = p_opt.m_mergeThreshold;          

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d, search limit: %f, merge threshold: %d\n", m_postingSizeLimit, p_opt.m_latencyLimit, m_mergeThreshold);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[CONFIG] layer=%d DistributedVersionMap=%s SearchCheckVersionMapOnlyLayer0=%s UseMultiChunkPosting=%s PostingPageLimit=%d\n",
                layer, p_opt.m_distributedVersionMap ? "true" : "false", p_opt.m_searchCheckVersionMapOnlyLayer0 ? "true" : "false", p_opt.m_useMultiChunkPosting ? "true" : "false", p_opt.m_postingPageLimit);
        }

        ~ExtraDynamicSearcher() {}

        virtual bool Available() override
        {
            return db->Available();
        }

        virtual SizeType GetNumSamples() const override
        {
            return m_versionMap->Count();
        }

        virtual bool ContainSample(const SizeType idx) const override
        {
            return ContainSample(idx, COMMON::VersionReadPolicy::UseCache);
        }

        virtual bool ContainSample(const SizeType idx, COMMON::VersionReadPolicy policy) const override
        {
            // Out-of-range or sentinel (-1) IDs are common in head-search
            // result lists (unfilled slots). Treat them as "not contained"
            // and avoid calling Deleted(), which would otherwise emit a
            // spurious LL_Error log inside TiKVVersionMap.
            if (idx < 0) return false;
            return !m_versionMap->Deleted(idx, policy);
        }

        virtual void ContainSamples(const std::vector<SizeType>& ids, std::vector<uint8_t>& contains, COMMON::VersionReadPolicy policy) const override
        {
            contains.assign(ids.size(), 0);
            if (ids.empty()) return;

            std::vector<SizeType> validIDs;
            std::vector<size_t> validIndices;
            validIDs.reserve(ids.size());
            validIndices.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] < 0) continue;
                validIDs.push_back(ids[i]);
                validIndices.push_back(i);
            }
            if (validIDs.empty()) return;

            std::vector<uint8_t> versions;
            m_versionMap->BatchGetVersions(validIDs, versions, policy);
            for (size_t i = 0; i < validIndices.size() && i < versions.size(); ++i) {
                contains[validIndices[i]] = (versions[i] != 0xfe) ? 1 : 0;
            }
        }

        virtual SizeType GetNumDeleted() const override
        {
            return m_versionMap->GetDeleteCount();
        }

        virtual ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override
        {
            return m_versionMap->GetContainedIDs(globalIDs);
        }

        bool ShouldCheckVersionMapInSearch(bool p_checkVersionMap) const
        {
            if (!(m_opt->m_storage == Storage::TIKVIO &&
                  m_opt->m_distributedVersionMap &&
                  m_opt->m_searchCheckVersionMapOnlyLayer0)) {
                return true;
            }
            return p_checkVersionMap;
        }

        SPANN::Index<ValueType>* GetHeadIndex() const { return m_headIndex; }

        bool CheckIsNeedReassign(std::vector<std::shared_ptr<std::string>>& newHeadsVec, const ValueType* data, const ValueType* splitHeadVec, float_t headToSplitHeadDist, float_t currentHeadDist, bool isInSplitHead)
        {
            float_t splitHeadDist = m_headIndex->ComputeDistance(data, splitHeadVec);

            if (isInSplitHead) {
                if (splitHeadDist >= currentHeadDist) return false;
            }
            else {
                float_t newHeadDist_1 = m_headIndex->ComputeDistance(data, newHeadsVec[0]->data());
                float_t newHeadDist_2 = m_headIndex->ComputeDistance(data, newHeadsVec[1]->data());
                if (splitHeadDist <= newHeadDist_1 && splitHeadDist <= newHeadDist_2) return false;
                if (currentHeadDist <= newHeadDist_1 && currentHeadDist <= newHeadDist_2) return false;
            }
            return true;
        }

        inline void Serialize(char* ptr, SizeType VID, std::uint8_t version, const void* vector) {
            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            memcpy(ptr + m_metaDataSize, vector, m_vectorDataSize);
        }

        void PrintErrorInPosting(std::string &posting, SizeType headID)
        {
            SizeType postVectorNum = posting.size() / m_vectorInfoSize;
            uint8_t *vectorId = reinterpret_cast<uint8_t *>(posting.data());
            for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
            {
                SizeType VID = *((SizeType *)(vectorId));
                if (VID < 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "PrintErrorInPosting found wrong VID:%lld in headID:%lld\n", (std::int64_t)VID,
                                 (std::int64_t)headID);
                }
            }
        }

        // TODO
        ErrorCode RefineIndex() override
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin RefineIndex\n");

            int iterations = 0;
            Helper::Concurrent::ConcurrentSet<SizeType> mergelist;
            while (iterations < 2) {
                while(!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                auto preReassignTimeBegin = std::chrono::high_resolution_clock::now();
                std::atomic<ErrorCode> finalcode = ErrorCode::Success;
                std::vector<std::thread> threads;
                std::atomic<SizeType> nextPostingID(0);
                std::vector<SizeType> globalIDs;
                m_headIndex->GetHeadIndexMapping(m_layer + 1, globalIDs);
                SizeType currentPostingNum = globalIDs.size();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch Refine for layer %d with %lld posting lists\n", m_layer, (std::int64_t)currentPostingNum);
                auto func = [&]()
                {
                    ErrorCode ret;
                    SizeType index = 0;
                    ExtraWorkSpace workSpace;
                    m_headIndex->InitWorkSpace(&workSpace);
                    while (true)
                    {
                        index = nextPostingID.fetch_add(1);
                        if (index < currentPostingNum)
                        {
                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / currentPostingNum);
                            }

                            SizeType globalID = globalIDs[index];

                            // ForceCompaction
                            std::string postingList;
                            if ((ret = db->Get(DBKey(globalID), &postingList, MaxTimeout, &(workSpace.m_diskRequests))) !=
                                    ErrorCode::Success)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                            "RefineIndex failed to get posting %lld, read size:%d\n",
                                            (std::int64_t)globalID, (int)(postingList.size()));
                                PrintErrorInPosting(postingList, globalID);
                                finalcode = ErrorCode::Fail;
                                continue;
                                //return;
                            }
                            SizeType postVectorNum = (SizeType)(postingList.size() / m_vectorInfoSize);
                            auto *postingP = reinterpret_cast<uint8_t *>(postingList.data());
                            uint8_t *vectorId = postingP;
                            int vectorCount = 0;
                            std::shared_ptr<std::string> vecStr;
                            bool hasHead = false;
                            for (int j = 0; j < postVectorNum;
                                    j++, vectorId += m_vectorInfoSize)
                            {
                                uint8_t version = *(vectorId + sizeof(SizeType));
                                SizeType VID = *((SizeType *)(vectorId));

                                if (VID == globalID) vecStr = std::make_shared<std::string>((char*)vectorId + m_metaDataSize, m_vectorDataSize);
                                
                                if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version)
                                    continue;

                                if (VID == globalID) hasHead = true;

                                *(vectorId + sizeof(SizeType)) = -1;
                                if (j != vectorCount)
                                {
                                    memcpy(postingP + vectorCount * m_vectorInfoSize, vectorId, m_vectorInfoSize);
                                }
                                vectorCount++;
                            }
                            if (vecStr == nullptr) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RefineIndex failed to find head vector in posting %lld\n", (std::int64_t)globalID);
                            }
                            if (!hasHead && vecStr != nullptr)
                            {
                                Serialize((char*)postingP + vectorCount * m_vectorInfoSize, globalID, m_versionMap->GetVersion(globalID), vecStr->data());
                                vectorCount++;
                            }
                            if (vectorCount <= m_mergeThreshold) mergelist.insert(globalID);

                            postingList.resize(vectorCount * m_vectorInfoSize);
                            if ((ret = db->Put(DBKey(globalID), postingList, MaxTimeout,
                                                    &(workSpace.m_diskRequests))) !=
                                ErrorCode::Success)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                "RefineIndex Failed to write back compacted posting %lld\n",
                                                (std::int64_t)(globalID));
                                finalcode = ret;
                                return;
                            }
                            CheckCentroid(globalID, postingList, "RefineIndex");
                        }
                        else
                        {
                            return;
                        }
                    }
                };
                for (int j = 0; j < m_opt->m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }

                globalIDs.clear();
                m_versionMap->GetContainedIDs(globalIDs);
                for (auto id : globalIDs) {
                    if (!m_versionMap->Deleted(id)) m_versionMap->SetVersion(id, -1);
                }

                auto preReassignTimeEnd = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(preReassignTimeEnd - preReassignTimeBegin).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "refine cost: %.2lf s mergelist size:%d\n", elapsedSeconds, (int)(mergelist.size()));

                if (finalcode != ErrorCode::Success)
                    return finalcode;

                if (mergelist.size() > 0 && iterations == 0)
                {
                    for (auto it = mergelist.begin(); it != mergelist.end(); ++it)
                    {
                        MergeAsync(*it);
                    }
                    mergelist.clear();
                }
                iterations++;
            }
            Checkpoint(m_opt->m_indexDirectory);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
            return ErrorCode::Success;
        }
        
        void CheckCentroid(SizeType pid, std::string& posting, std::string where)
        {
            SizeType postVectorNum = posting.size() / m_vectorInfoSize;
            uint8_t* vectorId = reinterpret_cast<uint8_t*>(posting.data());
            bool hasHead = false;
            for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
            {
                SizeType VID = *((SizeType*)(vectorId));
                if (VID == pid) {
                    hasHead = true;
                    break;
                }
            }
            if (!hasHead) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "CheckCentroid cannot find head in posting! pid:%lld, where:%s\n", (std::int64_t)pid, where.c_str());
                exit(-1);
            }
        }

        ErrorCode Split(ExtraWorkSpace* p_exWorkSpace, const SizeType headID, bool requirelock = true)
        {
            auto splitBegin = std::chrono::high_resolution_clock::now();
            std::vector<SizeType> newHeadsID(2, -1);
            std::vector<std::shared_ptr<std::string>> newHeadsVec(2, nullptr);
            std::vector<std::string> newPostingLists(2, "");
            std::shared_ptr<std::string> headVec;
            ErrorCode ret;
            bool theSameHead = false;
            double elapsedMSeconds;
            uint64_t splitPostingVectors = 0;
            uint64_t splitNewHeadCount = 0;
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID], std::defer_lock);
                if (requirelock) {
                    // [DIAG] measure split lock wait (suspect A: lock contention)
                    auto _lockBegin = std::chrono::high_resolution_clock::now();
                    lock.lock();
                    auto _lockAcq = std::chrono::high_resolution_clock::now();
                    uint64_t _lockWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(_lockAcq - _lockBegin).count();
                    IndexStats::HistAdd(m_stat.m_splitLockWaitUs, _lockWaitUs);
                    m_stat.m_splitLockWaitTotalUs.fetch_add(_lockWaitUs, std::memory_order_relaxed);
                    m_stat.m_splitLockSampleCount.fetch_add(1, std::memory_order_relaxed);
                }

                int retry = 0;
             Retry:
                if (!m_headIndex->ContainSample(headID, m_layer + 1)) return ErrorCode::Success;

                std::string postingList;
                auto splitGetBegin = std::chrono::high_resolution_clock::now();
                {
                    if ((ret=db->Get(DBKey(headID), &postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                        ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Split fail to get oversized postings: key=%lld read size=%d\n",
                                     (std::int64_t)headID, (int)(postingList.size()), (int)(ret == ErrorCode::Success));
                        return ret;
                    }
                }
                auto splitGetEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitGetEnd - splitGetBegin).count();
                m_stat.m_getCost += elapsedMSeconds;
                // reinterpret postingList to vectors and IDs
                uint8_t* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                SizeType postVectorNum = (SizeType)(postingList.size() / m_vectorInfoSize);
                splitPostingVectors = static_cast<uint64_t>(postVectorNum);
               
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: db get Posting %lld successfully with length %d real length:%d vectorNum:%d\n", (std::int64_t)headID, (int)(postingList.size()), m_postingSizes.GetSize(headID), postVectorNum);
                COMMON::Dataset<ValueType> smallSample(postVectorNum, m_opt->m_dim, m_headIndex->m_iDataBlockSize, m_headIndex->m_iDataCapacity, (ValueType*)postingP, true, nullptr, m_metaDataSize, m_vectorInfoSize);

                std::vector<SizeType> localIndices;
                localIndices.reserve(postVectorNum);
                uint8_t* vectorId = postingP;
                bool hasHead = false;
                for (SizeType j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                {
                    uint8_t version = *(vectorId + sizeof(SizeType));
                    SizeType VID = *((SizeType*)(vectorId));
                    if (VID < 0)
                    {
                        if (retry < 3)
                        {
                            retry++;
                            goto Retry;
                        }
                        else
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Split fail: Get posting %lld fail after 3 times retries.\n", (std::int64_t)(headID));
                            return ErrorCode::DiskIOFail;
                        }
                    }
                    
                    if (VID == headID) headVec = std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize);

		            //if (VID >= m_versionMap.Count()) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "DEBUG: vector ID:%lld total size:%lld\n", (std::int64_t)VID, (std::int64_t)m_versionMap.Count());
                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;

                    if (VID == headID) hasHead = true;
                    localIndices.push_back(j);
                }
                if (headVec == nullptr) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split fail: cannot find head in posting! headID:%lld\n", (std::int64_t)headID);
                    return ErrorCode::Fail;
                } else {
                    *((uint8_t*)(headVec->data() + sizeof(SizeType))) = m_versionMap->GetVersion(headID);
                }
                // double gcEndTime = sw.getElapsedMs();
                // m_splitGcCost += gcEndTime;
		
                if (localIndices.size() < m_postingSizeLimit)
                {
                    char* ptr = (char*)(postingList.data());
                    for (int j = 0; j < localIndices.size(); j++, ptr += m_vectorInfoSize)
                    {
                        if (j == localIndices[j]) continue;
                        memcpy(ptr, postingList.data() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                    }
                    if (!hasHead) {
                        memcpy(ptr, headVec->data(), m_vectorInfoSize);
                        localIndices.push_back(0); // just to make sure head is included in posting, the index won't be used
                    }
                    postingList.resize(localIndices.size() * m_vectorInfoSize);
                    if ((ret=db->Put(DBKey(headID), postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split Fail to write back posting %lld\n", (std::int64_t)(headID));
                        return ret;
                    }
                    CheckCentroid(headID, postingList, "Split-GC");
                    m_stat.m_garbageNum++;
                    auto GCEnd = std::chrono::high_resolution_clock::now();
                    elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(GCEnd - splitBegin).count();
                    m_stat.m_garbageCost += elapsedMSeconds;
                    {
                        std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                        m_splitList.unsafe_erase(headID);
                    }
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "GC triggered: %lld, new length: %d\n", (std::int64_t)headID, index);
                    return ErrorCode::Success;
                }

                auto clusterBegin = std::chrono::high_resolution_clock::now();
                // k = 2, maybe we can change the split number, now it is fixed
                SPTAG::COMMON::KmeansArgs<ValueType> args(2, smallSample.C(), (SizeType)localIndices.size(), 1, m_headIndex->GetDistCalcMethod(), m_headIndex->m_pQuantizer);
                std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));

                int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr);

                auto clusterEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(clusterEnd - clusterBegin).count();
                m_stat.m_clusteringCost += elapsedMSeconds;
                // int numClusters = ClusteringSPFresh(smallSample, localIndices, 0, localIndices.size(), args, 10, false, m_opt->m_virtualHead);
                if (numClusters <= 1)
                {
                    int cut = (m_opt->m_oneClusterCutMax)? m_postingSizeLimit: 1;
                    std::string newpostingList(cut * m_vectorInfoSize, '\0');
                    char* ptr = (char*)(newpostingList.data());
                    hasHead = false;
                    for (int j = 0; j < cut; j++, ptr += m_vectorInfoSize)
                    {
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                        if (*((SizeType*)(ptr)) == headID) hasHead = true;
                    }
                    if (!hasHead) memcpy(newpostingList.data(), headVec->data(), m_vectorInfoSize);
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cluserting Failed (The same vector), Only Keep %d vectors.\n", cut);
                   
                    if ((ret=db->Put(DBKey(headID), newpostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split fail to override posting cut to limit for posting %lld\n", (std::int64_t)(headID));
                        return ret;
                    }
                    CheckCentroid(headID, newpostingList, "Split-one-cluster");
                    {
                        std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                        m_splitList.unsafe_erase(headID);
                    }
                    return ErrorCode::Success;
                }

                std::vector<int> ks(2, 0);
                if (m_headIndex->ComputeDistance(args.centers, headVec->c_str() + m_metaDataSize) < m_headIndex->ComputeDistance(args.centers + args._D, headVec->c_str() + m_metaDataSize)) {
                    ks[0] = 1;
                } else {
                    ks[1] = 1;
                }
                SizeType newHeadVID = -1;
                int first = 0;
                for (int k : ks) {
                    if (args.counts[k] == 0)	continue;
                    first = (k == 0) ? 0 : args.counts[0];
                    newPostingLists[k].resize(args.counts[k] * m_vectorInfoSize);
                    char* ptr = (char*)(newPostingLists[k].c_str());
                    for (int j = 0; j < args.counts[k]; j++, ptr += m_vectorInfoSize)
                    {
                        memcpy(ptr, postingList.c_str() + localIndices[first + j] * m_vectorInfoSize, m_vectorInfoSize);
                    }
                    if (!theSameHead && m_headIndex->ComputeDistance(args.centers + k * args._D, headVec->c_str() + m_metaDataSize) < Epsilon) {
                        newHeadsID[k] = headID;
                        newHeadsVec[k] = std::make_shared<std::string>(headVec->c_str() + m_metaDataSize, m_vectorDataSize);
                        newHeadVID = headID;
                        theSameHead = true;
                        if (!hasHead) newPostingLists[k] += *headVec;
                        
                        auto splitPutBegin = std::chrono::high_resolution_clock::now();
                        if ((ret=db->Put(DBKey(newHeadVID), newPostingLists[k], MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to override posting %lld\n", (std::int64_t)(newHeadVID));
                            return ret;
                        }
                        CheckCentroid(newHeadVID, newPostingLists[k], "Split-SameHead");
                        auto splitPutEnd = std::chrono::high_resolution_clock::now();
                        elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin).count();
                        m_stat.m_putCost += elapsedMSeconds;
                        m_stat.m_theSameHeadNum++;
                        m_stat.m_splitSameHeadCount.fetch_add(1, std::memory_order_relaxed);
                    }
                    else {
                        newHeadVID = *((SizeType*)(postingP + args.clusterIdx[k] * m_vectorInfoSize));
                        uint8_t version = *((uint8_t*)(postingP + args.clusterIdx[k] * m_vectorInfoSize + sizeof(SizeType)));

                        newHeadsID[k] = newHeadVID;
                        newHeadsVec[k] = std::make_shared<std::string>((char *)(args.centers + k * args._D), m_vectorDataSize);

                        std::unique_lock<std::shared_timed_mutex> anotherLock(m_rwLocks[newHeadVID], std::defer_lock);
                        if (m_rwLocks.hash_func(newHeadVID) != m_rwLocks.hash_func(headID))
                        {
                            int retry = 0;
                            while (!anotherLock.try_lock() && retry < 20)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                             "Split: new head VID %lld is being locked. Wait for lock and do "
                                             "merging after getting lock... (attempt %d)\n",
                                             (std::int64_t)(newHeadVID), retry + 1);
                                retry++;
                                std::this_thread::sleep_for(std::chrono::milliseconds(3 * retry));
                            }
                            if (!anotherLock.owns_lock())
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                             "Split: new head VID %lld is being locked after %d retries. Skip merging and return split failed...\n",
                                             (std::int64_t)(newHeadVID), retry);
                                {
                                    std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                                    m_splitList.unsafe_erase(headID);
                                }
                                SplitAsync(headID, postingList.size() / m_vectorInfoSize);
                                return ErrorCode::Success;
                            }
                        }

                        if (m_headIndex->ContainSample(newHeadVID, m_layer + 1)) {
                            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Split: new head VID %lld already exists in head index. Do merging...\n", (std::int64_t)(newHeadVID));
                            m_stat.m_splitExistingHeadMergeCount.fetch_add(1, std::memory_order_relaxed);


                            std::string mergedPostingList;
                            std::set<SizeType> vectorIdSet;
                            std::string currentPostingList;
                            {
                                if ((ret = db->Get(DBKey(newHeadVID), &currentPostingList, MaxTimeout,
                                                   &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to get posting %lld\n",
                                                 (std::int64_t)(newHeadVID));
                                    return ret;
                                }
                            }

                            auto *postingO = reinterpret_cast<uint8_t *>(newPostingLists[k].data());
                            size_t postVectorNumO = newPostingLists[k].size() / m_vectorInfoSize;
                            int currentLength = 0;
                            bool hasHeadO = false;
                            for (int j = 0; j < postVectorNumO; j++, postingO += m_vectorInfoSize)
                            {
                                SizeType VID = *((SizeType *)(postingO));
                                if (vectorIdSet.insert(VID).second) {
                                    mergedPostingList += newPostingLists[k].substr(j * m_vectorInfoSize, m_vectorInfoSize);
                                    currentLength++;
                                    if (VID == newHeadVID) hasHeadO = true;
                                }
                            }

                            if (!hasHeadO) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Split: after merging head VID %lld, the head vector is missing in posting list. Add head vector back to posting list.\n", (std::int64_t)(newHeadVID));
                                vectorIdSet.insert(newHeadVID);
                                mergedPostingList = postingList.substr(args.clusterIdx[k] * m_vectorInfoSize, m_vectorInfoSize) + mergedPostingList;
                                currentLength++;
                            }

                            auto *postingK = reinterpret_cast<uint8_t *>(currentPostingList.data());
                            size_t newPostVectorNum = currentPostingList.size() / m_vectorInfoSize;
                            for (int j = 0; j < newPostVectorNum; j++, postingK += m_vectorInfoSize)
                            {
                                SizeType VID = *((SizeType *)(postingK));
                                uint8_t version = *(postingK + sizeof(SizeType));

                                if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version)
                                    continue;

                                if (vectorIdSet.find(VID) != vectorIdSet.end())
                                    continue;

                                vectorIdSet.insert(VID);
                                mergedPostingList += currentPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                                currentLength++;
                            }

                            if (currentLength > (m_postingSizeLimit + m_bufferSizeLimit) && m_opt->m_storage == Storage::FILEIO)
                            {
                                /*
                                SPTAGLIB_LOG(
                                    Helper::LogLevel::LL_Warning,
                                    "Split: merged posting list length %d exceeds hard limit %d after merging head "
                                    "VID %lld. Cut to limit and put back to db.\n",
                                    currentLength, m_postingSizeLimit + m_bufferSizeLimit, (std::int64_t)(newHeadVID));
                                */
                                mergedPostingList.resize((m_postingSizeLimit + m_bufferSizeLimit) * m_vectorInfoSize);
                                currentLength = m_postingSizeLimit + m_bufferSizeLimit;
                            }

                            auto splitPutBegin = std::chrono::high_resolution_clock::now();
                            if ((ret = db->Put(DBKey(newHeadVID), mergedPostingList, MaxTimeout,
                                               &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to put posting %lld\n",
                                             (std::int64_t)(newHeadVID));
                                return ret;
                            }
                            CheckCentroid(newHeadVID, mergedPostingList, "Split-MergePosting");
                            auto splitPutEnd = std::chrono::high_resolution_clock::now();
                            elapsedMSeconds =
                                std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin)
                                    .count();
                            m_stat.m_putCost += elapsedMSeconds;

                            if (currentLength > m_postingSizeLimit)
                            {
                                m_stat.m_splitExistingHeadMergeResplitCount.fetch_add(1, std::memory_order_relaxed);
                                SplitAsync(newHeadVID, currentLength);
                            }
                        } else {
                            auto splitPutBegin = std::chrono::high_resolution_clock::now();
                            if ((ret=db->Put(DBKey(newHeadVID), newPostingLists[k], MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to add new posting %lld\n", (std::int64_t)(newHeadVID));
                                return ret;
                            }
                            CheckCentroid(newHeadVID, newPostingLists[k], "Split-NewPosting");
                            auto splitPutEnd = std::chrono::high_resolution_clock::now();
                            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin).count();
                            m_stat.m_putCost += elapsedMSeconds;

                            auto updateHeadBegin = std::chrono::high_resolution_clock::now();
                            if ((ret = m_headIndex->AddHeadIndex(args.centers + k * args._D, newHeadVID, version, m_opt->m_dim, m_layer + 1, p_exWorkSpace)) != ErrorCode::Success) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to update head index %lld\n", (std::int64_t)(newHeadVID));
                                if (db->Delete(DBKey(newHeadVID)) != ErrorCode::Success) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete gc posting %lld\n", (std::int64_t)(newHeadVID));
                                }
                                return ret;
                            }
                            splitNewHeadCount++;
                            m_stat.m_splitCreatedNewHeadCount.fetch_add(1, std::memory_order_relaxed);
                            auto updateHeadEnd = std::chrono::high_resolution_clock::now();
                            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(updateHeadEnd - updateHeadBegin).count();
                            m_stat.m_updateHeadCost += elapsedMSeconds;
                        }
                        if (m_rwLocks.hash_func(newHeadVID) != m_rwLocks.hash_func(headID)) anotherLock.unlock();
                    }
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head id: %lld split into : %lld, length: %d\n", (std::int64_t)headID, (std::int64_t)newHeadVID, args.counts[k]);
                }
                if (!theSameHead) {
                    m_headIndex->DeleteIndex(headID, m_layer + 1);
                    if ((ret=db->Delete(DBKey(headID))) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting in Split: %lld\n", (std::int64_t)headID);
                        return ret;
                    }
                }

                {
                    std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"erase: %lld\n", (std::int64_t)headID);
                    m_splitList.unsafe_erase(headID);
                }
                
                for (int k = 0; k < 2; k++) {
                    if (args.counts[k] > m_postingSizeLimit) {
                        {
                            SplitAsync(newHeadsID[k], args.counts[k]);
                        }
                    }
                }
            }
            
            m_stat.m_splitNum++;
            IndexStats::HistAdd(m_stat.m_splitPostingVectors, splitPostingVectors);
            m_stat.m_splitPostingVectorsTotal.fetch_add(splitPostingVectors, std::memory_order_relaxed);
            m_stat.m_splitPostingVectorSampleCount.fetch_add(1, std::memory_order_relaxed);
            IndexStats::HistAdd(m_stat.m_splitNewHeadCount, splitNewHeadCount);
            m_stat.m_splitNewHeadCountTotal.fetch_add(splitNewHeadCount, std::memory_order_relaxed);
            m_stat.m_splitNewHeadSampleCount.fetch_add(1, std::memory_order_relaxed);

            if (!m_opt->m_disableReassign) {
                auto reassignScanBegin = std::chrono::high_resolution_clock::now();

                CollectReAssign(p_exWorkSpace, headID, headVec, newPostingLists, newHeadsID, newHeadsVec, theSameHead);

                auto reassignScanEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanEnd - reassignScanBegin).count();

                m_stat.m_reassignScanCost += elapsedMSeconds;
            }
            auto splitEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - splitBegin).count();
            m_stat.m_splitCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        ErrorCode MergePostings(ExtraWorkSpace *p_exWorkSpace, SizeType headID)
        {
            std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);

            if (!m_headIndex->ContainSample(headID, m_layer + 1)) {
                std::unique_lock<std::shared_timed_mutex> tmplock(m_mergeListLock);
                m_mergeList.unsafe_erase(headID);
                return ErrorCode::Success;
            }

            {
                std::shared_lock<std::shared_timed_mutex> tmplock(m_mergeListLock);
                if (m_mergeList.find(headID) == m_mergeList.end()) {
                    return ErrorCode::Success;
                }
            }

            std::string mergedPostingList;
            std::set<SizeType> vectorIdSet;

            std::string currentPostingList;
            ErrorCode ret;
            {
                if ((ret = db->Get(DBKey(headID), &currentPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                        ErrorCode::Success)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Fail to get original merge postings: %lld, get size:%d\n",
                        (std::int64_t)headID, (int)(currentPostingList.size()));
                    PrintErrorInPosting(currentPostingList, headID);
                    return ret;
                }
            }

            auto* postingP = reinterpret_cast<uint8_t*>(currentPostingList.data());
            size_t postVectorNum = currentPostingList.size() / m_vectorInfoSize;
            int currentLength = 0;
            uint8_t* vectorId = postingP;
            std::shared_ptr<std::string> headVec;
            for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
            {
                SizeType VID = *((SizeType*)(vectorId));
                uint8_t version = *(vectorId + sizeof(SizeType));
                if (VID == headID) {
                    headVec = std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize);
                }
                if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                vectorIdSet.insert(VID);
                mergedPostingList += currentPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                currentLength++;
            }

            if (headVec == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail: cannot find head vector in posting! headID:%lld\n", (std::int64_t)headID);
                return ErrorCode::Fail;
            } else {
                *((uint8_t*)(headVec->data() + sizeof(SizeType))) = m_versionMap->GetVersion(headID);
            }

            if (currentLength > m_mergeThreshold)
            {
                if (vectorIdSet.find(headID) == vectorIdSet.end() && headVec != nullptr) {
                    mergedPostingList += *headVec;
                }
                if ((ret=db->Put(DBKey(headID), mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge Fail to write back posting %lld\n", (std::int64_t)headID);
                    return ret;
                }
                CheckCentroid(headID, mergedPostingList, "MergePostings-ignore");
                {
                    std::unique_lock<std::shared_timed_mutex> tmplock(m_mergeListLock);
                    m_mergeList.unsafe_erase(headID);
                }
                return ErrorCode::Success;
            }

            COMMON::QueryResultSet<ValueType> queryResults((ValueType*)(headVec->data() + m_metaDataSize), m_opt->m_internalResultNum, false, false);
            std::shared_ptr<std::uint8_t> rec_query;
            if (m_headIndex->m_pQuantizer) {
                rec_query.reset((uint8_t*)ALIGN_ALLOC(m_headIndex->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                m_headIndex->m_pQuantizer->ReconstructVector((const uint8_t*)queryResults.GetTarget(), rec_query.get());
                queryResults.SetTarget((ValueType*)(rec_query.get()), m_headIndex->m_pQuantizer);
            }
            m_headIndex->SearchHeadIndex(queryResults, m_layer + 1, p_exWorkSpace);

            std::string nextPostingList;
            for (int i = 1; i < queryResults.GetResultNum(); ++i)
            {
                BasicResult* queryResult = queryResults.GetResult(i);
  
                int dedupLength = 0;
                SizeType nextHeadID = -1;
                std::shared_ptr<std::string> nextHeadVec;
                std::shared_ptr<std::string> deletedHeadVec;
                std::string * deletedPostingList = nullptr;
                std::shared_ptr<std::string> resultVec;
                std::set<SizeType> nextVectorIdSet;
                int deletedLength = 0;
                {
                    std::unique_lock<std::shared_timed_mutex> anotherLock(m_rwLocks[queryResult->VID], std::defer_lock);
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Locked: %lld, to be lock: %lld\n", (std::int64_t)headID, (std::int64_t)queryResult->VID);
                    if (m_rwLocks.hash_func(queryResult->VID) != m_rwLocks.hash_func(headID)) {
                        if (!anotherLock.try_lock()) {
                            auto* curJob = new MergeAsyncJob(this, headID, nullptr);
                            // Re-queue counts as a new submission; matched by the
                            // m_mergeJobsInFlight-- / m_totalMergeCompleted++ in
                            // MergeAsyncJob::exec(). Without these increments
                            // m_mergeJobsInFlight underflows to a huge uint64
                            // and m_totalMergeCompleted exceeds m_totalMergeSubmitted.
                            m_mergeJobsInFlight++;
                            m_totalMergeSubmitted++;
                            m_splitThreadPool->add(curJob);
                            return ErrorCode::Success;
                        }
                    }
                    if (!m_headIndex->ContainSample(queryResult->VID, m_layer + 1)) continue;
                    if ((ret=db->Get(DBKey(queryResult->VID), &nextPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                        "Fail to get to be merged posting: %lld, get size:%d\n",
                                        (std::int64_t)(queryResult->VID), (int)(nextPostingList.size()));
                        PrintErrorInPosting(nextPostingList, queryResult->VID);
                        return ret;
                    }
                    postingP = reinterpret_cast<uint8_t*>(nextPostingList.data());
                    postVectorNum = nextPostingList.size() / m_vectorInfoSize;
                    vectorId = postingP;
                    int nextLength = 0;
                    for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                    {
                        SizeType VID = *((SizeType*)(vectorId));
                        uint8_t version = *(vectorId + sizeof(SizeType));
                        if (VID == queryResult->VID) resultVec = std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize);
                        if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                        if (vectorIdSet.find(VID) == vectorIdSet.end()) {
                            nextVectorIdSet.insert(VID);
                            mergedPostingList += nextPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                            dedupLength++;
                        }
                        nextLength++;
                    }
                    if (resultVec == nullptr) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail: cannot find another head vector in posting! headID:%lld\n", (std::int64_t)(queryResult->VID));
                        return ErrorCode::Fail;
                    }
                    if (currentLength + dedupLength >= m_postingSizeLimit) {
                        mergedPostingList.resize(currentLength * m_vectorInfoSize);
                        continue;
                    }
                    if (currentLength >= nextLength) 
                    {               
                        if (vectorIdSet.find(headID) == vectorIdSet.end() && nextVectorIdSet.find(headID) == nextVectorIdSet.end() && headVec != nullptr) {
                            mergedPostingList += *headVec;
                        }            
                        if ((ret=db->Put(DBKey(headID), mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to override old posting %lld after merge\n", (std::int64_t)headID);
                            return ret;
                        }
                        CheckCentroid(headID, mergedPostingList, "MergePostings-currentLength >= nextLength");
                        m_headIndex->DeleteIndex(queryResult->VID, m_layer + 1);
                        if ((ret=db->Delete(DBKey(queryResult->VID))) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting %lld in Merge\n", (std::int64_t)(queryResult->VID));
                            return ret;
                        }
                        nextHeadID = headID;
                        nextHeadVec = headVec;
                        deletedHeadVec = resultVec;
                        deletedPostingList = &nextPostingList;
                        deletedLength = nextLength;
                    } else
                    { 
                        if (vectorIdSet.find(queryResult->VID) == vectorIdSet.end() && nextVectorIdSet.find(queryResult->VID) == nextVectorIdSet.end() && resultVec != nullptr) {
                            mergedPostingList += *resultVec;
                        }
                        if ((ret=db->Put(DBKey(queryResult->VID), mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to override posting %lld after merge\n", (std::int64_t)(queryResult->VID));
                            return ret;
                        }
                        CheckCentroid(queryResult->VID, mergedPostingList, "MergePostings-currentLength < nextLength");
                        m_headIndex->DeleteIndex(headID, m_layer + 1);
                        if ((ret = db->Delete(DBKey(headID))) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting %lld in Merge\n", (std::int64_t)(headID));
                            return ret;
                        }
                        nextHeadID = queryResult->VID;
                        nextHeadVec = resultVec;
                        deletedHeadVec = headVec;
                        deletedPostingList = &currentPostingList;
                        deletedLength = currentLength;
                    }
                    if (m_rwLocks.hash_func(queryResult->VID) != m_rwLocks.hash_func(headID)) anotherLock.unlock();
                }

                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Release: %lld, Release: %lld\n", (std::int64_t)headID, (std::int64_t)queryResult->VID);
                lock.unlock();

                if (!m_opt->m_disableReassign) 
                {
                    postingP = reinterpret_cast<uint8_t*>(deletedPostingList->data());
                    for (int j = 0; j < deletedLength; j++) {
                        uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                        SizeType VID = *(reinterpret_cast<SizeType*>(vectorId));
                        uint8_t version = *(vectorId + sizeof(SizeType));
                        ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                        if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                        float origin_dist = m_headIndex->ComputeDistance(deletedHeadVec->data() + m_metaDataSize, vector);
                        float current_dist = m_headIndex->ComputeDistance(nextHeadVec->data() + m_metaDataSize, vector);
                        if (current_dist > origin_dist) {
                            m_stat.m_reassignSubmittedFromMerge.fetch_add(1, std::memory_order_relaxed);
                            ReassignAsync(std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), nextHeadID);
                        }
                    }
                }

                {
                    {
                        std::unique_lock<std::shared_timed_mutex> tmplock(m_mergeListLock);
                        m_mergeList.unsafe_erase(headID);
                        m_mergeList.unsafe_erase(queryResult->VID);
                    }
                    if (currentLength + dedupLength <= m_mergeThreshold) {
                        MergeAsync(nextHeadID);
                    }
                }
                m_stat.m_mergeNum++;
                return ErrorCode::Success;
            }

            mergedPostingList.resize(currentLength * m_vectorInfoSize);
            if (vectorIdSet.find(headID) == vectorIdSet.end() && headVec != nullptr) {
                mergedPostingList += *headVec;
            }            
            if ((ret=db->Put(DBKey(headID), mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge Fail to write back posting %lld\n", (std::int64_t)headID);
                return ret;
            }
            CheckCentroid(headID, mergedPostingList, "MergePostings-GC");
            {
                std::unique_lock<std::shared_timed_mutex> tmplock(m_mergeListLock);
                m_mergeList.unsafe_erase(headID);
            }
            return ErrorCode::Success;
        }

        inline void SplitAsync(SizeType headID, int postingSize, std::function<void()> p_callback = nullptr)
        {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Into SplitAsync, current headID: %lld, size: %d\n", (std::int64_t)headID, m_postingSizes.GetSize(headID));
            // tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor headIDAccessor;
            // if (m_splitList.find(headIDAccessor, headID)) {
            //     return;
            // }
            // tbb::concurrent_hash_map<SizeType, SizeType>::value_type workPair(headID, headID);
            // m_splitList.insert(workPair);
            {
                Helper::Concurrent::ConcurrentMap<SizeType, int>::value_type workPair(headID, postingSize);
                std::shared_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                auto res = m_splitList.insert(workPair);
                if (!res.second)
                {
                    m_splitList[headID] = max(res.first->second, postingSize);
                    return;
                }
            }

            auto* curJob = new SplitAsyncJob(this, headID, p_callback);
            m_splitJobsInFlight++;
            m_totalSplitSubmitted++;
            m_splitThreadPool->add(curJob);
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Add to thread pool\n");
        }

        inline void MergeAsync(SizeType headID, std::function<void()> p_callback = nullptr)
        {
            {
                std::shared_lock<std::shared_timed_mutex> tmplock(m_mergeListLock);
                auto res = m_mergeList.insert(headID);
                if (!res.second)
                {
                    // Already in queue
                    return;
                }
            }

            auto* curJob = new MergeAsyncJob(this, headID, p_callback);
            m_mergeJobsInFlight++;
            m_totalMergeSubmitted++;
            m_splitThreadPool->add(curJob);
        }

        inline void AppendAsync(SizeType headID, std::shared_ptr<std::string> postingList, bool urgent = false,std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new AppendAsyncJob(this, headID, std::move(postingList), p_callback);
            m_appendJobsInFlight++;
            m_totalAppendSubmitted++;
            if (urgent) {
                m_splitThreadPool->addfront(curJob);
            } else {
                m_splitThreadPool->add(curJob);
            }
        }

        inline void ReassignAsync(std::shared_ptr<std::string> vectorInfo, SizeType headPrev, bool urgent = false, std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new ReassignAsyncJob(this, std::move(vectorInfo), headPrev, p_callback);
            m_reassignJobsInFlight++;
            m_totalReassignSubmitted++;
            if (urgent) {
                m_splitThreadPool->addfront(curJob);
            } else {
                m_splitThreadPool->add(curJob);
            }
        }

        ErrorCode CollectReAssign(ExtraWorkSpace *p_exWorkSpace, SizeType headID, std::shared_ptr<std::string> headVec,
                                  std::vector<std::string> &postingLists, std::vector<SizeType> &newHeadsID, std::vector<std::shared_ptr<std::string>> &newHeadsVec,
                                  bool theSameHead)
        {
            auto headVector = reinterpret_cast<const ValueType*>(headVec->data() + m_metaDataSize);

            // Collect vectors that need reassign, then do RNGSelection inline
            // and batch Append by target head to reduce TiKV RPCs.
            // batchReassign: targetHead -> merged posting data
            std::unordered_map<SizeType, std::string> batchReassign;
            std::unordered_set<SizeType> batchReassignVids;
            size_t batchReassignCount = 0;

            // Helper lambda: run RNGSelection for a vector and add to batch
            auto tryBatchReassign = [&](uint8_t* vectorId, SizeType headPrev) {
                SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType)));
                ValueType* vectorData = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);

                if (vid < 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "CollectReAssign: skip invalid VID %lld\n",
                                 (std::int64_t)vid);
                    return;
                }

                if (m_versionMap->Deleted(vid) || m_versionMap->GetVersion(vid) != version) return;

                m_stat.m_reAssignNum++;
                std::vector<BasicResult> selections(static_cast<size_t>(m_opt->m_replicaCount));
                int replicaCount;
                bool isNeedReassign = RNGSelection(p_exWorkSpace, selections, vectorData, replicaCount, headPrev);

                if (isNeedReassign && m_versionMap->GetVersion(vid) == version) {
                    m_versionMap->IncVersion(vid, &version, version);
                    *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType))) = version;
                    batchReassignVids.insert(vid);
                    for (int r = 0; r < replicaCount && m_versionMap->GetVersion(vid) == version; r++) {
                        batchReassign[selections[r].VID].append((char*)vectorId, m_vectorInfoSize);
                        batchReassignCount++;
                    }
                }
            };

            std::vector<float> newHeadsDist(2, 0.0f);
            std::set<SizeType> reAssignVectorsTopK;
            if (newHeadsVec[0]) newHeadsDist[0] = m_headIndex->ComputeDistance(headVector, newHeadsVec[0]->data());
            if (newHeadsVec[1]) newHeadsDist[1] = m_headIndex->ComputeDistance(headVector, newHeadsVec[1]->data());
            for (int i = 0; i < postingLists.size(); i++) {
                if (!newHeadsVec[i]) continue;
                auto& postingList = postingLists[i];
                size_t postVectorNum = postingList.size() / m_vectorInfoSize;
                auto* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType)));
                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                    if (vid < 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                     "CollectReAssign: skip invalid VID %lld in posting headID=%lld\n",
                                     (std::int64_t)vid, (std::int64_t)newHeadsID[i]);
                        continue;
                    }
                    if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                        m_stat.m_reAssignScanNum++;
                        float dist = m_headIndex->ComputeDistance(newHeadsVec[i]->data(), vector);
                        if (CheckIsNeedReassign(newHeadsVec, vector, headVector, newHeadsDist[i], dist, true)) {
                            tryBatchReassign(vectorId, newHeadsID[i]);
                            reAssignVectorsTopK.insert(vid);
                        }
                    }
                }
            }
            if (m_opt->m_reassignK > 0) {
                std::vector<SizeType> HeadPrevTopK;
                std::vector<std::shared_ptr<std::string>> HeadPrevTopKVec;
                newHeadsDist.clear();
                newHeadsDist.resize(0);
                COMMON::QueryResultSet<ValueType> nearbyHeads((ValueType*)headVector, m_opt->m_reassignK, false, true);
                std::shared_ptr<std::uint8_t> rec_query;
                if (m_headIndex->m_pQuantizer) {
                    rec_query.reset((uint8_t*)ALIGN_ALLOC(m_headIndex->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                    m_headIndex->m_pQuantizer->ReconstructVector((const uint8_t*)nearbyHeads.GetTarget(), rec_query.get());
                    nearbyHeads.SetTarget((ValueType*)(rec_query.get()), m_headIndex->m_pQuantizer);
                }
                m_headIndex->SearchHeadIndex(nearbyHeads, m_layer + 1, p_exWorkSpace);
                BasicResult* queryResults = nearbyHeads.GetResults();
                for (int i = 0; i < nearbyHeads.GetResultNum(); i++) {
                    auto vid = queryResults[i].VID;
                    if (vid == -1) break;

                    if (find(newHeadsID.begin(), newHeadsID.end(), vid) == newHeadsID.end()) {
                        HeadPrevTopK.push_back(vid);
                        HeadPrevTopKVec.push_back(std::make_shared<std::string>((char*)(queryResults[i].Vec.Data()), m_vectorDataSize));
                        newHeadsID.push_back(vid);
                        newHeadsDist.push_back(queryResults[i].Dist);
                    }
                }
                auto reassignScanIOBegin = std::chrono::high_resolution_clock::now();
                ErrorCode ret;
                {
                    auto keys = DBKeys(HeadPrevTopK);
                    if ((ret = db->MultiGet(*keys, p_exWorkSpace->m_pageBuffers, m_hardLatencyLimit,
                                            &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ReAssign can't get all the near postings\n");
                        return ret;
                    }
                }

                auto reassignScanIOEnd = std::chrono::high_resolution_clock::now();
                auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignScanIOEnd - reassignScanIOBegin).count();
                m_stat.m_reassignScanIOCost += elapsedMSeconds;

                std::vector<std::string> nearbyPostings(HeadPrevTopK.size());
                for (int i = 0; i < HeadPrevTopK.size(); i++)
                {
                    auto &buffer = (p_exWorkSpace->m_pageBuffers[i]);
                    size_t bufBytes = buffer.GetAvailableSize();
                    if (bufBytes > 0) {
                        nearbyPostings[i].assign(reinterpret_cast<const char*>(buffer.GetBuffer()), bufBytes);
                    }
                }
                for (int i = 0; i < HeadPrevTopK.size(); i++)
                {
                    auto& postingList = nearbyPostings[i];
                    size_t postVectorNum = postingList.size() / m_vectorInfoSize;
                    auto* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                    for (int j = 0; j < postVectorNum; j++) {
                        uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                        SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                        uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType)));
                        ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                        if (vid < 0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "CollectReAssign(nearby): skip invalid VID %lld in posting headID=%lld\n",
                                (std::int64_t)vid, (std::int64_t)HeadPrevTopK[i]);
                            continue;
                        }
                        if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                            m_stat.m_reAssignScanNum++;
                            float dist = m_headIndex->ComputeDistance(HeadPrevTopKVec[i]->data(), vector);
                            if (CheckIsNeedReassign(newHeadsVec, vector, headVector, newHeadsDist[i], dist, false)) {
                                tryBatchReassign(vectorId, HeadPrevTopK[i]);
                                reAssignVectorsTopK.insert(vid);
                            }
                        }
                    }
                }
            }


            // Batch Append: one Append call per target head instead of one ReassignAsync per vector
            // Use reassignThreshold=0 so that if the posting overflows, it goes through
            // SplitAsync (async) rather than synchronous Split, avoiding recursive deadlock:
            // Split -> CollectReAssign -> Append -> Split -> CollectReAssign -> ...
            ErrorCode ret = ErrorCode::Success;
            if (m_opt->m_storage == Storage::TIKVIO) ret = BatchAppend(p_exWorkSpace, batchReassign, "CollectReAssign");
            else {
                for (auto& kv : batchReassign) {
                    AppendAsync(kv.first, std::make_shared<std::string>(kv.second), true);
                }
            }
            if (batchReassignCount > 0) {
                m_totalReassignSubmitted += batchReassignCount;
                m_totalReassignCompleted += batchReassignCount;
                m_stat.m_reassignSubmittedFromSplitBatch.fetch_add(batchReassignCount, std::memory_order_relaxed);
            }
            uint64_t reassignVectors = static_cast<uint64_t>(batchReassignVids.size());
            uint64_t reassignRecords = static_cast<uint64_t>(batchReassignCount);
            uint64_t reassignTargetHeads = static_cast<uint64_t>(batchReassign.size());
            IndexStats::HistAdd(m_stat.m_splitReassignVectors, reassignVectors);
            IndexStats::HistAdd(m_stat.m_splitReassignRecords, reassignRecords);
            IndexStats::HistAdd(m_stat.m_splitReassignTargetHeads, reassignTargetHeads);
            m_stat.m_splitReassignVectorsTotal.fetch_add(reassignVectors, std::memory_order_relaxed);
            m_stat.m_splitReassignRecordsTotal.fetch_add(reassignRecords, std::memory_order_relaxed);
            m_stat.m_splitReassignTargetHeadsTotal.fetch_add(reassignTargetHeads, std::memory_order_relaxed);
            m_stat.m_splitReassignSampleCount.fetch_add(1, std::memory_order_relaxed);
            m_stat.m_splitReassignRecordSampleCount.fetch_add(1, std::memory_order_relaxed);
            m_stat.m_splitReassignTargetHeadSampleCount.fetch_add(1, std::memory_order_relaxed);
            return ret;
        }

        bool RNGSelection(ExtraWorkSpace* p_exWorkSpace, std::vector<BasicResult>& selections, ValueType* queryVector, int& replicaCount, SizeType checkHeadID = -1)
        {
            COMMON::QueryResultSet<ValueType> queryResults(queryVector, m_opt->m_internalResultNum, false, true);
            std::shared_ptr<std::uint8_t> rec_query;
            if (m_headIndex->m_pQuantizer) {
                rec_query.reset((uint8_t*)ALIGN_ALLOC(m_headIndex->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                m_headIndex->m_pQuantizer->ReconstructVector((const uint8_t*)queryResults.GetTarget(), rec_query.get());
                queryResults.SetTarget((ValueType*)(rec_query.get()), m_headIndex->m_pQuantizer);
            }
            m_headIndex->SearchHeadIndex(queryResults, m_layer + 1, p_exWorkSpace);

            replicaCount = 0;
            for (int i = 0; i < queryResults.GetResultNum() && replicaCount < m_opt->m_replicaCount; ++i)
            {
                BasicResult* queryResult = queryResults.GetResult(i);
                if (queryResult->VID == -1) {
                    break;
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j)
                {
                    float nnDist = m_headIndex->ComputeDistance((queryResult->Vec).Data(), selections[j].Vec.Data());
                    if (m_opt->m_rngFactor * nnDist <= queryResult->Dist)
                    {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted) continue;
                selections[replicaCount] = *queryResult;

                if (queryResult->VID == checkHeadID) {
                    return false;
                }
                ++replicaCount;
            }
            return true;
        }


        ErrorCode Append(ExtraWorkSpace* p_exWorkSpace, SizeType headID, int appendNum, std::string& appendPosting, int reassignThreshold = 0)
        {
            auto appendBegin = std::chrono::high_resolution_clock::now();
            if (appendPosting.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }

            if (appendNum == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Error!, headID :%lld, appendNum:%d\n", (std::int64_t)headID, appendNum);
            }

        checkDeleted:
            if (!m_headIndex->ContainSample(headID, m_layer + 1)) {
                for (int i = 0; i < appendNum; i++)
                {
                    uint32_t idx = i * m_vectorInfoSize;
                    SizeType VID = *(SizeType*)(&appendPosting[idx]);
                    uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(SizeType)]);
                    auto vectorInfo = std::make_shared<std::string>(appendPosting.c_str() + idx, m_vectorInfoSize);
                    if (m_versionMap->GetVersion(VID) == version) {
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head Miss To ReAssign: VID: %lld, current version: %d\n", (std::int64_t)VID, (int)version);
                        m_stat.m_headMiss++;
                        ReassignAsync(vectorInfo, headID, true);
                    }
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head Miss Do Not To ReAssign: VID: %lld, version: %d, current version: %d\n", (std::int64_t)VID, (int)version, (int)m_versionMap->GetVersion(VID));
                }
                return ErrorCode::Success;
            }
            double appendIOSeconds = 0;
            int postingSize = 0;
            {
                //std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //ROCKSDB
                // [DIAG] measure lock wait time (suspect A: lock contention)
                auto _lockBegin = std::chrono::high_resolution_clock::now();
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //SPDK
                auto _lockAcq = std::chrono::high_resolution_clock::now();
                uint64_t _lockWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(_lockAcq - _lockBegin).count();
                IndexStats::HistAdd(m_stat.m_appendLockWaitUs, _lockWaitUs);
                m_stat.m_appendLockWaitTotalUs.fetch_add(_lockWaitUs, std::memory_order_relaxed);

                ErrorCode ret;
                if (!m_headIndex->ContainSample(headID, m_layer + 1)) {
                    lock.unlock();
                    goto checkDeleted;
                }
                {
                    std::shared_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                    auto it = m_splitList.find(headID);
                    if (it != m_splitList.end()) {
                        postingSize = it->second;
                    }
                }

                if (postingSize + appendNum > (m_postingSizeLimit + m_bufferSizeLimit) && m_opt->m_storage == Storage::FILEIO) {
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "After appending, the number of vectors in %lld exceeds the postingsize + buffersize (%d + %d)! Do split now...\n", (std::int64_t)headID, m_postingSizeLimit, m_bufferSizeLimit);
                    ret = Split(p_exWorkSpace, headID, false);
                    if (ret != ErrorCode::Success)
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split %lld failed!\n", (std::int64_t)headID);
                    lock.unlock();
                    goto checkDeleted;
                }

                auto appendIOBegin = std::chrono::high_resolution_clock::now();
                if ((ret = db->Merge(
                         DBKey(headID), appendPosting, MaxTimeout, &(p_exWorkSpace->m_diskRequests), postingSize)) != ErrorCode::Success)
                {
                    if (ret == ErrorCode::Posting_OverFlow) {
                        int tofill = m_postingSizeLimit + m_bufferSizeLimit - (postingSize / m_vectorInfoSize);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Merge failed:Posting overflow when appending to %lld! existing size:%d tofill:%d Do split and then retry...\n", (std::int64_t)headID, (int)postingSize, tofill);
                        if (tofill > 0) {
                            std::string subPosting = appendPosting.substr(0, tofill * m_vectorInfoSize);
                            ret = db->Merge(DBKey(headID), subPosting, MaxTimeout, &(p_exWorkSpace->m_diskRequests), postingSize);
                            if (ret != ErrorCode::Success) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge %lld failed!\n", (std::int64_t)headID);
                                return ret;
                            }
                            appendPosting = appendPosting.substr(tofill * m_vectorInfoSize);
                            appendNum -= tofill;
                        }
                        ret = Split(p_exWorkSpace, headID, false);
                        if (ret != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split %lld failed!\n", (std::int64_t)headID);
                            return ret;
                        }
                        lock.unlock();
                        goto checkDeleted;
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge failed for %lld! Posting Size:%d, limit: %d\n", (std::int64_t)headID, postingSize, m_postingSizeLimit);
                    GetDBStats();
                    return ret;
                }
                auto appendIOEnd = std::chrono::high_resolution_clock::now();
                appendIOSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendIOEnd - appendIOBegin).count();

                if (postingSize % m_vectorInfoSize != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "Append: stale-aligned posting in TiKV headID=%lld size=%zu mod=%zu (m_vectorInfoSize=%d)\n",
                                (std::int64_t)headID, postingSize,
                                postingSize % m_vectorInfoSize,
                                m_vectorInfoSize);
                }
                IndexStats::HistAdd(m_stat.m_appendPostingBytes, (uint64_t)postingSize);
                m_stat.m_appendPostingBytesTotal.fetch_add((uint64_t)postingSize, std::memory_order_relaxed);
                m_stat.m_appendRmwSampleCount.fetch_add(1, std::memory_order_relaxed);
                postingSize /= m_vectorInfoSize;
            }
            if (postingSize > (m_postingSizeLimit + reassignThreshold)) {
                // SizeType VID = *(int*)(&appendPosting[0]);
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split Triggered by inserting VID: %lld, reAssign: %d\n", (std::int64_t)VID, reassignThreshold);
                // GetDBStats();
                // if (m_postingSizes.GetSize(headID) > 120) {
                //     GetDBStats();
                // }
                m_stat.m_appendTriggeredSplit.fetch_add(1, std::memory_order_relaxed);
                if (!reassignThreshold) SplitAsync(headID, postingSize);
                else Split(p_exWorkSpace, headID);
            }
            auto appendEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
            if (!reassignThreshold) {
                m_totalAppendCount++;
                m_stat.m_appendTaskNum++;
                m_stat.m_appendIOCost += appendIOSeconds;
                m_stat.m_appendCost += elapsedMSeconds;
            }
            // } else {
            //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReAssign Append To: %lld\n", (std::int64_t)headID);
            // }
            return ErrorCode::Success;
        }
        
        ErrorCode BatchAppend(ExtraWorkSpace* p_exWorkSpace, std::unordered_map<SizeType, std::string>& headAppends, const char* caller, bool disableSplit = false)
        {
            if (headAppends.empty()) return ErrorCode::Success;

            auto appendBegin = std::chrono::high_resolution_clock::now();
            
            std::vector<SizeType> keys;
            std::vector<std::string> values;

            std::vector<SizeType> sortedHeadIDs;
            sortedHeadIDs.reserve(headAppends.size());
            for (const auto& kv : headAppends) sortedHeadIDs.push_back(kv.first);
            std::sort(sortedHeadIDs.begin(), sortedHeadIDs.end());

            std::vector<std::unique_lock<std::shared_timed_mutex>> heldLocks;
            heldLocks.reserve(sortedHeadIDs.size());

            for (SizeType headID : sortedHeadIDs)
            {
                auto appendIt = headAppends.find(headID);
                if (appendIt == headAppends.end()) continue;

                std::unique_lock<std::shared_timed_mutex> headLock(m_rwLocks[headID]);

                if (!m_headIndex->ContainSample(headID, m_layer + 1)) {
                    headLock.unlock();
                    for (std::uint8_t* ptr = (std::uint8_t*)(appendIt->second.data());
                        ptr < (std::uint8_t*)(appendIt->second.data() + appendIt->second.size());
                        ptr += m_vectorInfoSize) {
                        SizeType VID = *(SizeType*)(ptr);
                        uint8_t version = *(uint8_t*)(ptr + sizeof(SizeType));
                        if (m_versionMap->GetVersion(VID) == version) {
                            m_stat.m_headMiss++;
                            ReassignAsync(std::make_shared<std::string>((char*)ptr, m_vectorInfoSize), headID, true);
                        }
                    }
                    continue;
                }

                keys.push_back(headID);
                values.push_back(appendIt->second);
                heldLocks.emplace_back(std::move(headLock));
            }

            if (keys.empty()) return ErrorCode::Success;

            std::vector<int> postingSizes(keys.size(), 0);
            auto appendIOBegin = std::chrono::high_resolution_clock::now();
            ErrorCode ret;
            auto dbkeys = DBKeys(keys);
            if ((ret = db->MultiMerge(
                         *dbkeys, values, MaxTimeout, &(p_exWorkSpace->m_diskRequests), postingSizes)) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MultiMerge failed!\n");
                GetDBStats();
                return ret;
            }
            auto appendIOEnd = std::chrono::high_resolution_clock::now();
            auto appendIOSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendIOEnd - appendIOBegin).count();

            for (size_t i = 0; i < keys.size(); i++) {
                heldLocks[i].unlock();
                int postingSize = postingSizes[i];
                if (postingSize % m_vectorInfoSize != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "Append: stale-aligned posting in TiKV headID=%lld size=%zu mod=%zu (m_vectorInfoSize=%d)\n",
                                (std::int64_t)keys[i], postingSize,
                                postingSize % m_vectorInfoSize,
                                m_vectorInfoSize);
                }
                IndexStats::HistAdd(m_stat.m_appendPostingBytes, (uint64_t)postingSize);
                m_stat.m_appendPostingBytesTotal.fetch_add((uint64_t)postingSize, std::memory_order_relaxed);
                m_stat.m_appendRmwSampleCount.fetch_add(1, std::memory_order_relaxed);
                postingSize /= m_vectorInfoSize;
                if (postingSize > m_postingSizeLimit && !disableSplit) {
                    m_stat.m_appendTriggeredSplit.fetch_add(1, std::memory_order_relaxed);
                    SplitAsync(keys[i], postingSize);
                }
                auto appendEnd = std::chrono::high_resolution_clock::now();
                double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
                m_totalAppendCount++;
                m_stat.m_appendTaskNum++;
                m_stat.m_appendIOCost += appendIOSeconds;
                m_stat.m_appendCost += elapsedMSeconds;
            }
            return ErrorCode::Success;
        }

        ErrorCode Reassign(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<std::string> vectorInfo, SizeType headPrev)
        {
            SizeType VID = *((SizeType*)vectorInfo->c_str());
            uint8_t version = *((uint8_t*)(vectorInfo->c_str() + sizeof(VID)));
            if (VID < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                             "Reassign: skip invalid VID %lld\n",
                             (std::int64_t)VID);
                return ErrorCode::Success;
            }
            // return;
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignID: %lld, version: %d, current version: %d, headPrev: %lld\n", (std::int64_t)VID, (int)version, (int)m_versionMap->GetVersion(VID), (std::int64_t)headPrev);
            if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) {
                return ErrorCode::Success;
            }
            auto reassignBegin = std::chrono::high_resolution_clock::now();

            m_stat.m_reAssignNum++;

            auto selectBegin = std::chrono::high_resolution_clock::now();
            std::vector<BasicResult> selections(static_cast<size_t>(m_opt->m_replicaCount));
            int replicaCount;
            bool isNeedReassign = RNGSelection(p_exWorkSpace, selections, (ValueType*)(vectorInfo->c_str() + m_metaDataSize), replicaCount, headPrev);
            auto selectEnd = std::chrono::high_resolution_clock::now();
            auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(selectEnd - selectBegin).count();
            m_stat.m_selectCost += elapsedMSeconds;

            auto reassignAppendBegin = std::chrono::high_resolution_clock::now();
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Need ReAssign\n");
            if (isNeedReassign && m_versionMap->GetVersion(VID) == version) {
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Update Version: VID: %lld, version: %d, current version: %d\n", (std::int64_t)VID, (int)version, (int)m_versionMap->GetVersion(VID));
                m_versionMap->IncVersion(VID, &version, version);
                (*vectorInfo)[sizeof(VID)] = version;

                //LOG(Helper::LogLevel::LL_Info, "Reassign: oldVID:%lld, replicaCount:%d, candidateNum:%d, dist0:%f\n", (std::int64_t)oldVID, replicaCount, i, selections[0].distance);
                for (int i = 0; i < replicaCount && m_versionMap->GetVersion(VID) == version; i++) {
                    //LOG(Helper::LogLevel::LL_Info, "Reassign: headID :%lld, oldVID:%lld, newVID:%lld, posting length: %d, dist: %f, string size: %d\n", (std::int64_t)headID, (std::int64_t)oldVID, (std::int64_t)VID, m_postingSizes[headID].load(), selections[i].distance, newPart.size());
                    // [FIX H3] use reassignThreshold=0 so that an oversized
                    // target posting triggers SplitAsync (not a synchronous
                    // Split on this worker thread). This matches the
                    // CollectReAssign batch path and avoids a single merge-
                    // path reassign blocking a worker for the full duration
                    // of a Split (observed up to tens of seconds).
                    ErrorCode tmp = Append(p_exWorkSpace, selections[i].VID, 1, *vectorInfo, 0);
                    if (ErrorCode::Success != tmp) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Head Miss: VID: %lld, current version: %d, another re-assign\n", (std::int64_t)VID, (int)version);
                        return tmp;
                    }
                }
            }
            auto reassignAppendEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignAppendEnd - reassignAppendBegin).count();
            m_stat.m_reAssignAppendCost += elapsedMSeconds;

            auto reassignEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignEnd - reassignBegin).count();
            m_stat.m_reAssignCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        bool LoadIndex(Options& p_opt) override {
            m_opt = &p_opt;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DataBlockSize: %lld, Capacity: %lld\n", (std::int64_t)(m_opt->m_datasetRowsInBlock), (std::int64_t)(m_opt->m_datasetCapacity));
            std::string versionmapPath = m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer);
            if (m_opt->m_recovery) {
                versionmapPath = m_opt->m_persistentBufferPath + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading version map\n");
                m_versionMap->Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Current vector num: %lld.\n", (std::int64_t)(m_versionMap->Count()));
            }
            else if (m_opt->m_storage == Storage::ROCKSDBIO || m_opt->m_storage == Storage::TIKVIO) {
                m_versionMap->Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current vector num: %lld.\n", (std::int64_t)(m_versionMap->Count()));
            } else if (m_opt->m_storage == Storage::SPDKIO || m_opt->m_storage == Storage::FILEIO) {
		        if (fileexists((m_opt->m_indexDirectory + FolderSep + m_opt->m_ssdIndex + "_" + std::to_string(m_layer)).c_str())) {
                    m_versionMap->DeleteAll();
			        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Copying data from static to SPDK\n");
			        std::shared_ptr<IExtraSearcher> storeExtraSearcher;
			        storeExtraSearcher.reset(new ExtraStaticSearcher<ValueType>(m_layer, m_headIndex));
			        if (!storeExtraSearcher->LoadIndex(*m_opt)) {
			            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Load Static Index Initialize Error\n");
			            return false;
			        }
                    std::vector<SizeType> allPostingIDs;
                    m_headIndex->GetHeadIndexMapping(m_layer + 1, allPostingIDs);
			        int totalPostingNum = allPostingIDs.size();

			        std::vector<std::thread> threads;
			        std::atomic_size_t vectorsSent(0);
                    ErrorCode ret = ErrorCode::Success;
			        auto func = [&]() {
                        ExtraWorkSpace workSpace;
                        m_headIndex->InitWorkSpace(&workSpace);
                        size_t index = 0;
                        while (true)
                        {
                            index = vectorsSent.fetch_add(1);
                            if (index < totalPostingNum)
                            {

                                if ((index & ((1 << 14) - 1)) == 0)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Copy to SPDK: Sent %.2lf%%...\n",
                                                 index * 100.0 / totalPostingNum);
                                }
                                std::string tempPosting;
                                if (storeExtraSearcher->GetWritePosting(&workSpace, allPostingIDs[index], tempPosting) !=
                                    ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Static Index Read Posting fail\n");
                                    ret = ErrorCode::Fail;
                                    return;
                                }
                                int vectorNum = (int)(tempPosting.size() / (m_vectorInfoSize - sizeof(uint8_t)));

                                if (vectorNum > m_postingSizeLimit) vectorNum = m_postingSizeLimit;
                                auto *postingP = reinterpret_cast<char *>(tempPosting.data());
                                std::string newPosting(m_vectorInfoSize * vectorNum, '\0');
                                char *ptr = (char *)(newPosting.c_str());
                                for (int j = 0; j < vectorNum; ++j, ptr += m_vectorInfoSize)
                                {
                                    char *vectorInfo = postingP + j * (m_vectorInfoSize - sizeof(uint8_t));
                                    SizeType VID = *(reinterpret_cast<SizeType *>(vectorInfo));
                                    m_versionMap->SetVersion(VID, -1);
                                    Serialize(ptr, VID, -1, vectorInfo + sizeof(SizeType));
                                }
                                if (GetWritePosting(&workSpace, allPostingIDs[index], newPosting, true) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Index Write Posting fail\n");                                  
                                    ret = ErrorCode::Fail;
                                    return;
                                }
                            }
                            else
                            {
                                return;
                            }
                        }
                    };
			    for (int j = 0; j < m_opt->m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
			    for (auto& thread : threads) { thread.join(); }
                if (ret != ErrorCode::Success)
                    return false;
		    } else {
                m_versionMap->Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
            } 
	    }
            if (m_opt->m_update) {
                if (m_splitThreadPool == nullptr) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_opt->m_appendThreadNum, m_opt->m_reassignThreadNum);

                    m_splitThreadPool = std::make_shared<SPDKThreadPool>();
                    m_splitThreadPool->initSPDK(m_opt->m_appendThreadNum, this);
                    //m_reassignThreadPool = std::make_shared<SPDKThreadPool>();
                    //m_reassignThreadPool->initSPDK(m_opt->m_reassignThreadNum, this);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
                }
                
                if (m_opt->m_enableWAL && !m_opt->m_persistentBufferPath.empty()) {
                    std::string p_persistenWAL = m_opt->m_persistentBufferPath + FolderSep + "WAL";
                    std::shared_ptr<Helper::KeyValueIO> pdb;
#ifdef ROCKSDB
                    pdb.reset(new RocksDBIO(p_persistenWAL.c_str(), false, false));
                    m_wal.reset(new PersistentBuffer(pdb));
#else
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPFresh: Wal only support RocksDB! Please use -DROCKSDB when doing cmake.\n");
                    return false;
#endif
                } 
            }

            /** recover the previous WAL **/
            if (m_opt->m_recovery && m_opt->m_enableWAL && m_wal) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: WAL\n");
                std::string assignment;
                int countAssignment = 0;
                if (!m_wal->StartToScan(assignment)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: No log\n");
                    return true;
                }
                ExtraWorkSpace workSpace;
                m_headIndex->InitWorkSpace(&workSpace);
                do {
                    countAssignment++;
                    if (countAssignment % 10000 == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Process %d logs\n", countAssignment);
                    char* ptr = (char*)(assignment.c_str());
                    SizeType VID = *(reinterpret_cast<SizeType*>(ptr));
                    if (assignment.size() == m_vectorInfoSize) {
                        std::shared_ptr<VectorSet> vectorSet;
                        vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)ptr + m_metaDataSize, m_vectorDataSize, false),
                            GetEnumValueType<ValueType>(), m_opt->m_dim, 1));
                        AddIndex(&workSpace, vectorSet, VID);
                    } else {
                        m_versionMap->Delete(VID);
                    }
                } while (m_wal->NextToScan(assignment));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: No more to repeat, wait for rebalance\n");
                while(!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            }
            return true;
        }

        virtual ErrorCode SearchIndex(ExtraWorkSpace* p_exWorkSpace,
            QueryResult& p_queryResults,
            SearchStats* p_stats, std::set<SizeType>* truth, std::map<SizeType, std::set<SizeType>>* found,
            bool p_checkVersionMap) override
        {
            // Use coprocessor search if enabled and storage is TiKV
#ifdef TIKV
            if (m_opt->m_useCoprocessorSearch && m_opt->m_storage == Storage::TIKVIO) {
                return SearchIndexWithCoprocessor(p_exWorkSpace, p_queryResults, p_stats, truth, found, p_checkVersionMap);
            }
#endif
            if (p_stats) p_stats->m_exSetUpLatency = 0;

            auto layerTotalStart = std::chrono::high_resolution_clock::now();

            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);

            int diskRead = 0;
            int diskIO = 0;
            int listElements = 0;

            double compLatency = 0;
            double readLatency = 0;
            double versionMapLatency = 0;
            int versionCheckCount = 0;
            std::chrono::microseconds remainLimit;
            if (p_stats) remainLimit = m_hardLatencyLimit - std::chrono::microseconds((int)p_stats->m_totalLatency);
            else remainLimit = m_hardLatencyLimit;

            auto readStart = std::chrono::high_resolution_clock::now();
            {
                auto keys = DBKeys(p_exWorkSpace->m_postingIDs);
                if (db->MultiGet(*keys, p_exWorkSpace->m_pageBuffers, remainLimit, &(p_exWorkSpace->m_diskRequests)) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[SearchIndex] read postings fail!\n");
                    return ErrorCode::DiskIOFail;
                }
            }
            auto readEnd = std::chrono::high_resolution_clock::now();
            readLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count());

            const auto postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
            bool isTiKV = (m_opt->m_storage == Storage::TIKVIO);
            bool checkVersionMapInSearch = ShouldCheckVersionMapInSearch(p_checkVersionMap);
            for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                auto& buffer = (p_exWorkSpace->m_pageBuffers[pi]);
                char* p_postingListFullData = (char*)(buffer.GetBuffer());
                int vectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);

                diskIO += int((buffer.GetAvailableSize() + PageSize - 1) >> PageSizeEx);
                diskRead += (int)(buffer.GetAvailableSize());
                
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: postingList %lld size:%d m_vectorInfoSize:%d vectorNum:%d\n", (std::int64_t)pi, (int)(postingList.size()), m_vectorInfoSize, vectorNum);
                int realNum = vectorNum;
                listElements += vectorNum;
                auto compStart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < vectorNum; i++) {
                    char* vectorInfo = p_postingListFullData + i * m_vectorInfoSize;
                    SizeType vectorID = *(reinterpret_cast<SizeType*>(vectorInfo));

		            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: vectorID:%lld\n", (std::int64_t)vectorID);
                    if (!isTiKV && m_versionMap->Deleted(vectorID)) {
                        realNum--;
                        listElements--;
                        continue;
                    }
                    if(p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) {
                        listElements--;
                        continue;
                    }
                    auto distance2leaf = m_headIndex->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    queryResults.AddPoint(vectorID, distance2leaf, queryResults.WithVec()? ByteArray::Alloc((std::uint8_t*)(vectorInfo + m_metaDataSize), m_vectorDataSize) : ByteArray::c_empty);
                }
                auto compEnd = std::chrono::high_resolution_clock::now();
                if (m_opt->m_asyncMergeInSearch && realNum <= m_mergeThreshold) MergeAsync(curPostingID); // TODO: Control merge

                compLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(compEnd - compStart).count());

                if (truth) {
                    for (int i = 0; i < vectorNum; ++i) {
                        char* vectorInfo = p_postingListFullData + i * m_vectorInfoSize;
                        SizeType vectorID = *(reinterpret_cast<SizeType*>(vectorInfo));
                        if (truth->count(vectorID) != 0)
                            (*found)[curPostingID].insert(vectorID);
                    }
                }
            }

            // For TiKV mode: post-heap version check via BatchGetVersions
            if (isTiKV && checkVersionMapInSearch) {
                int fetchCount = queryResults.GetResultNum();

                // Collect candidate VIDs from the top results
                std::vector<SizeType> candidateVIDs;
                candidateVIDs.reserve(fetchCount);
                for (int i = 0; i < fetchCount && i < queryResults.GetResultNum(); i++) {
                    auto* result = queryResults.GetResult(i);
                    if (result->VID >= 0) {
                        candidateVIDs.push_back(result->VID);
                    }
                }

                // Batch check versions
                std::vector<uint8_t> versions;
                auto versionMapStart = std::chrono::high_resolution_clock::now();
                m_versionMap->BatchGetVersions(candidateVIDs, versions, p_exWorkSpace->m_versionReadPolicy);
                auto versionMapEnd = std::chrono::high_resolution_clock::now();
                versionMapLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(versionMapEnd - versionMapStart).count());
                versionCheckCount += static_cast<int>(candidateVIDs.size());

                // Filter: rebuild results without deleted entries
                // We mark deleted entries with MaxDist so they sort to the end
                int vidIdx = 0;
                for (int i = 0; i < fetchCount && i < queryResults.GetResultNum(); i++) {
                    auto* result = queryResults.GetResult(i);
                    if (result->VID >= 0) {
                        if (versions[vidIdx] == 0xfe) {
                            result->VID = -1;
                            result->Dist = (std::numeric_limits<float>::max)();
                        }
                        vidIdx++;
                    }
                }
                queryResults.SortResult();
            }

            auto layerTotalEnd = std::chrono::high_resolution_clock::now();
            double layerTotalLatency = ((double)std::chrono::duration_cast<std::chrono::microseconds>(layerTotalEnd - layerTotalStart).count()) / 1000;

            if (p_stats)
            {
                double compLatencyMs = compLatency / 1000;
                double readLatencyMs = readLatency / 1000;
                double versionMapLatencyMs = versionMapLatency / 1000;
                int diskAccessKB = diskRead / 1024;
                p_stats->m_compLatency += compLatencyMs;
                p_stats->m_diskReadLatency += readLatencyMs;
                p_stats->m_versionMapLatency += versionMapLatencyMs;
                p_stats->m_totalListElementsCount += listElements;
                p_stats->m_diskIOCount += diskIO;
                p_stats->m_diskAccessCount += diskAccessKB;
                if (SearchStats::IsValidBreakdownLayer(m_layer)) {
                    p_stats->m_layerPostingReadLatency[m_layer] += readLatencyMs;
                    p_stats->m_layerCompLatency[m_layer] += compLatencyMs;
                    p_stats->m_layerVersionMapLatency[m_layer] += versionMapLatencyMs;
                    p_stats->m_layerTotalLatency[m_layer] += layerTotalLatency;
                    p_stats->m_layerPostingCount[m_layer] += static_cast<int>(postingListCount);
                    p_stats->m_layerListElementsCount[m_layer] += listElements;
                    p_stats->m_layerVersionCheckCount[m_layer] += versionCheckCount;
                    p_stats->m_layerDiskAccessCount[m_layer] += diskAccessKB;
                }
            }
            queryResults.SetScanned(listElements);
            return ErrorCode::Success;
        }

#ifdef TIKV
        // Coprocessor-based search: push distance computation into TiKV.
        // Instead of fetching raw posting data, sends the query vector and
        // posting keys to TiKV, which reads postings locally, computes L2
        // distances, and returns only top-N candidates.
        ErrorCode SearchIndexWithCoprocessor(ExtraWorkSpace* p_exWorkSpace,
            QueryResult& p_queryResults,
            SearchStats* p_stats, std::set<SizeType>* truth, std::map<SizeType, std::set<SizeType>>* found,
            bool p_checkVersionMap)
        {
            auto layerTotalStart = std::chrono::high_resolution_clock::now();

            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);

            int listElements = 0;
            double readLatency = 0;
            double versionMapLatency = 0;
            int versionCheckCount = 0;

            int valueType = 0; // UInt8
            if (std::is_same<ValueType, float>::value) valueType = 3;
            else if (std::is_same<ValueType, int8_t>::value) valueType = 1;

            int topN = m_opt->m_coprocessorTopN;
            if (topN <= 0) topN = m_opt->m_resultNum * 10;

            // Get the TiKVIO instance from the database
            auto* tikvDB = dynamic_cast<TiKVIO*>(db.get());
            if (!tikvDB) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[SearchIndexWithCoprocessor] db is not TiKVIO!\n");
                return ErrorCode::Fail;
            }

            auto readStart = std::chrono::high_resolution_clock::now();

            std::vector<TiKVIO::CoprocessorResult> coprResults;
            auto ret = tikvDB->CoprocessorSearch(
                p_exWorkSpace->m_postingIDs,
                reinterpret_cast<const uint8_t*>(queryResults.GetQuantizedTarget()),
                m_opt->m_dim,
                valueType,
                m_metaDataSize,
                topN,
                m_hardLatencyLimit,
                coprResults);

            auto readEnd = std::chrono::high_resolution_clock::now();
            readLatency = (double)std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count();

            if (ret != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[SearchIndexWithCoprocessor] CoprocessorSearch failed!\n");
                return ErrorCode::DiskIOFail;
            }

            // Process results: dedup, skip per-entry version check for TiKV mode
            for (auto& cr : coprResults) {
                if (p_exWorkSpace->m_deduper.CheckAndSet(cr.vectorID)) continue;
                queryResults.AddPoint(cr.vectorID, cr.distance);
                listElements++;
            }

            // Post-heap batch version check for TiKV mode
            if (ShouldCheckVersionMapInSearch(p_checkVersionMap)) {
                int fetchCount = queryResults.GetResultNum();

                std::vector<SizeType> candidateVIDs;
                candidateVIDs.reserve(fetchCount);
                for (int i = 0; i < fetchCount && i < queryResults.GetResultNum(); i++) {
                    auto* result = queryResults.GetResult(i);
                    if (result->VID >= 0) candidateVIDs.push_back(result->VID);
                }

                std::vector<uint8_t> versions;
                auto versionMapStart = std::chrono::high_resolution_clock::now();
                m_versionMap->BatchGetVersions(candidateVIDs, versions, p_exWorkSpace->m_versionReadPolicy);
                auto versionMapEnd = std::chrono::high_resolution_clock::now();
                versionMapLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(versionMapEnd - versionMapStart).count());
                versionCheckCount += static_cast<int>(candidateVIDs.size());

                int vidIdx = 0;
                for (int i = 0; i < fetchCount && i < queryResults.GetResultNum(); i++) {
                    auto* result = queryResults.GetResult(i);
                    if (result->VID >= 0) {
                        if (versions[vidIdx] == 0xfe) {
                            result->VID = -1;
                            result->Dist = (std::numeric_limits<float>::max)();
                        }
                        vidIdx++;
                    }
                }
                queryResults.SortResult();
            }

            auto layerTotalEnd = std::chrono::high_resolution_clock::now();
            double layerTotalLatency = ((double)std::chrono::duration_cast<std::chrono::microseconds>(layerTotalEnd - layerTotalStart).count()) / 1000;

            if (p_stats)
            {
                double readLatencyMs = readLatency / 1000;
                double versionMapLatencyMs = versionMapLatency / 1000;
                p_stats->m_compLatency += 0; // computation done on TiKV side
                p_stats->m_diskReadLatency += readLatencyMs;
                p_stats->m_versionMapLatency += versionMapLatencyMs;
                p_stats->m_totalListElementsCount += listElements;
                p_stats->m_diskIOCount += 0;
                p_stats->m_diskAccessCount += 0;
                if (SearchStats::IsValidBreakdownLayer(m_layer)) {
                    p_stats->m_layerPostingReadLatency[m_layer] += readLatencyMs;
                    p_stats->m_layerVersionMapLatency[m_layer] += versionMapLatencyMs;
                    p_stats->m_layerTotalLatency[m_layer] += layerTotalLatency;
                    p_stats->m_layerPostingCount[m_layer] += static_cast<int>(p_exWorkSpace->m_postingIDs.size());
                    p_stats->m_layerListElementsCount[m_layer] += listElements;
                    p_stats->m_layerVersionCheckCount[m_layer] += versionCheckCount;
                }
            }
            queryResults.SetScanned(listElements);
            return ErrorCode::Success;
        }
#endif
        virtual ErrorCode SearchIndexWithoutParsing(ExtraWorkSpace* p_exWorkSpace)
        {
            int retry = 0;
            ErrorCode ret = ErrorCode::Undefined;
            while (retry < 2 && ret != ErrorCode::Success)
            {
                auto keys = DBKeys(p_exWorkSpace->m_postingIDs);
                ret = db->MultiGet(*keys, p_exWorkSpace->m_pageBuffers, m_hardLatencyLimit,
                                   &(p_exWorkSpace->m_diskRequests));
                retry++;
            }
            return ret;
        }

        virtual ErrorCode SearchNextInPosting(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
            QueryResult& p_queryResults)
        {
            /*
            COMMON::QueryResultSet<ValueType>& headResults = *((COMMON::QueryResultSet<ValueType>*) & p_headResults);
            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
            bool foundResult = false;
            BasicResult* head = headResults.GetResult(p_exWorkSpace->m_ri);
            while (!foundResult && p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) {
                if (head && head->VID != -1 && p_exWorkSpace->m_ri <= p_exWorkSpace->m_pi) {
                    if (!m_versionMap->Deleted(head->VID) && !p_exWorkSpace->m_deduper.CheckAndSet(head->VID) &&
                    (p_exWorkSpace->m_filterFunc == nullptr || p_exWorkSpace->m_filterFunc(p_spann->GetMetadata(head->VID)))) {
                        queryResults.AddPoint(head->VID, head->Dist);
                        foundResult = true;
                    }
                    head = headResults.GetResult(++p_exWorkSpace->m_ri);
                    continue;
                }
                auto& buffer = (p_exWorkSpace->m_pageBuffers[p_exWorkSpace->m_pi]);
                char* p_postingListFullData = (char*)(buffer.GetBuffer());
                int vectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);
                while (p_exWorkSpace->m_offset < vectorNum) {
                    char* vectorInfo = p_postingListFullData + p_exWorkSpace->m_offset * m_vectorInfoSize;
                    p_exWorkSpace->m_offset++;

                    SizeType vectorID = *(reinterpret_cast<SizeType*>(vectorInfo));
                    if (vectorID < 0) return ErrorCode::Key_OverFlow;
                    if (m_versionMap->Deleted(vectorID)) continue;
                    if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue;
                    if (p_exWorkSpace->m_filterFunc != nullptr && !p_exWorkSpace->m_filterFunc(p_spann->GetMetadata(vectorID))) continue;

                    auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    queryResults.AddPoint(vectorID, distance2leaf);
                    foundResult = true;
                    break;
                }
                if (p_exWorkSpace->m_offset == vectorNum) {
                    p_exWorkSpace->m_pi++;
                    p_exWorkSpace->m_offset = 0;
                }
            }
            while (!foundResult && head && head->VID != -1) {
                if (!m_versionMap->Deleted(head->VID) && !p_exWorkSpace->m_deduper.CheckAndSet(head->VID) &&
                (p_exWorkSpace->m_filterFunc == nullptr || p_exWorkSpace->m_filterFunc(p_spann->GetMetadata(head->VID)))) {
                    queryResults.AddPoint(head->VID, head->Dist);
                    foundResult = true;
                }
                head = headResults.GetResult(++p_exWorkSpace->m_ri);
            }
            if (foundResult) p_queryResults.SetScanned(p_queryResults.GetScanned() + 1);
            return (foundResult) ? ErrorCode::Success : ErrorCode::VectorNotFound;
            */
            return ErrorCode::Undefined;
        }

        virtual ErrorCode SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
            QueryResult& p_query)
        {
            /*
            if (p_exWorkSpace->m_loadPosting) {
                ErrorCode ret = SearchIndexWithoutParsing(p_exWorkSpace);
                if (ret != ErrorCode::Success) return ret;
                p_exWorkSpace->m_ri = 0;
                p_exWorkSpace->m_pi = 0;
                p_exWorkSpace->m_offset = 0;
                p_exWorkSpace->m_loadPosting = false;
            }

            return SearchNextInPosting(p_exWorkSpace, p_headResults, p_query, p_index, p_spann);
            */
           return ErrorCode::Undefined;
        }

        bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt, COMMON::Dataset<SizeType>& p_headToLocal, Helper::Concurrent::ConcurrentMap<SizeType, SizeType>& p_headGlobaltoLocal, COMMON::Dataset<SizeType>& p_localToGlobal, SizeType upperBound = -1) override {
            m_opt = &p_opt;

            int numThreads = m_opt->m_iSSDNumberOfThreads;
            int candidateNum = m_opt->m_internalResultNum;
            if (m_opt->m_headIDFile.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                return false;
            }

            if (m_layer > 0 && p_localToGlobal.R() == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty localToGlobal for non-leaf layer!\n");
                return false;
            }

            SizeType fullCount = 0;
            {
                auto fullVectors = p_reader->GetVectorSet();
                fullCount = fullVectors->Count();
                m_metaDataSize = sizeof(SizeType) + sizeof(uint8_t);
                m_vectorDataSize = fullVectors->PerVectorDataSize();
                m_vectorInfoSize = m_vectorDataSize + m_metaDataSize;
            }
            if (upperBound > 0) fullCount = upperBound;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build SSD Index.\n");

            Selection selections(static_cast<size_t>(fullCount) * m_opt->m_replicaCount, m_opt->m_tmpdir);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Full vector count:%lld Edge bytes:%llu selection size:%zu, capacity size:%zu\n", (std::int64_t)fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
            std::vector<std::atomic_int> replicaCount(fullCount);
            std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
            for (auto& pls : postingListSize) pls = 0;
            std::unordered_set<SizeType> emptySet;
            SizeType batchSize = (fullCount + m_opt->m_batches - 1) / m_opt->m_batches;

            auto t1 = std::chrono::high_resolution_clock::now();
            if (p_opt.m_batches > 1)
            {
                if (selections.SaveBatch() != ErrorCode::Success)
                {
                    return false;
                }
            }
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                SizeType sampleSize = m_opt->m_samples;
                std::vector<SizeType> samples(sampleSize, 0);
                for (int i = 0; i < m_opt->m_batches; i++) {
                    SizeType start = i * batchSize;
                    SizeType end = min(start + batchSize, fullCount);
                    auto fullVectors = p_reader->GetVectorSet(start, end);
                    if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);

                    if (p_opt.m_batches > 1) {
                        if (selections.LoadBatch(static_cast<size_t>(start) * p_opt.m_replicaCount, static_cast<size_t>(end) * p_opt.m_replicaCount) != ErrorCode::Success)
                        {
                            return false;
                        }
                    }

                    int sampleNum = 0;
                    for (SizeType j = start; j < end && sampleNum < sampleSize; j++)
                    {
                        samples[sampleNum++] = j - start;
                    }

                    float acc = 0;
                    for (int j = 0; j < sampleNum; j++)
                    {
                        COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                    }
                    acc = acc / sampleNum;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%lld,%lld) loaded with %lld vectors (%zu) HeadIndex acc @%d:%f.\n", i, (std::int64_t)start, (std::int64_t)end, (std::int64_t)(fullVectors->Count()), selections.m_selections.size(), candidateNum, acc);

                    p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), m_opt->m_replicaCount, numThreads, m_opt->m_gpuSSDNumTrees, m_opt->m_gpuSSDLeafSize, m_opt->m_rngFactor, m_opt->m_numGPUs);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                    for (SizeType j = start; j < end; j++) {
                        replicaCount[j] = 0;
                        size_t vecOffset = j * (size_t)m_opt->m_replicaCount;
                        for (int resNum = 0; resNum < m_opt->m_replicaCount && selections[vecOffset + resNum].node != MaxSize; resNum++) {
                            ++postingListSize[selections[vecOffset + resNum].node];
                            selections[vecOffset + resNum].tonode = j;
                            ++replicaCount[j];
                        }
                    }

                    if (p_opt.m_batches > 1)
                    {
                        if (selections.SaveBatch() != ErrorCode::Success)
                        {
                            return false;
                        }
                    }
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

            if (p_opt.m_batches > 1)
            {
                if (selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount) != ErrorCode::Success)
                {
                    return false;
                }
            }

            // Sort results either in CPU or GPU
            VectorIndex::SortSelections(&selections.m_selections);

            auto t3 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", m_postingSizeLimit);
            {
                std::vector<SizeType> replicaCountDist(m_opt->m_replicaCount + 1, 0);
                for (SizeType i = 0; i < replicaCount.size(); ++i)
                {
                    ++replicaCountDist[replicaCount[i]];
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %lld\n", i, (std::int64_t)(replicaCountDist[i]));
                }
            }

            Helper::Concurrent::ConcurrentSet<SizeType> zeroReplicaSet;
            std::atomic_int64_t originalSize(0), relaxSize(0);
            std::atomic_int64_t overflowPostingCount(0), droppedReplicaCount(0);
            {
                std::vector<std::thread> mythreads;
                mythreads.reserve(m_opt->m_iSSDNumberOfThreads);
                std::atomic_size_t sent(0);
                int relaxLimit = m_postingSizeLimit + m_bufferSizeLimit;
                for (int tid = 0; tid < m_opt->m_iSSDNumberOfThreads; tid++)
                {
                    mythreads.emplace_back([&, tid]() {
                        size_t i = 0;
                        while (true)
                        {
                            i = sent.fetch_add(1);
                            if (i < postingListSize.size())
                            {
                                if (postingListSize[i] <= m_postingSizeLimit)
                                    originalSize += postingListSize[i];
                                else
                                    originalSize += m_postingSizeLimit;

                                if (postingListSize[i] <= relaxLimit)
                                {
                                    relaxSize += postingListSize[i];
                                    continue;
                                }
                                ++overflowPostingCount;
                                relaxSize += relaxLimit;

                                std::size_t selectIdx =
                                    std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i,
                                                     Selection::g_edgeComparer) -
                                    selections.m_selections.begin();

                                for (size_t dropID = relaxLimit;
                                     dropID < postingListSize[i]; ++dropID)
                                {
                                    int tonode = selections.m_selections[selectIdx + dropID].tonode;
                                    ++droppedReplicaCount;
                                    --replicaCount[tonode];
                                    if (replicaCount[tonode] == 0)
                                    {
                                        zeroReplicaSet.insert(tonode);
                                    }
                                }
                                postingListSize[i] = relaxLimit;
                            }
                            else
                            {
                                return;
                            }
                        }
                    });
                }
                for (auto &t : mythreads)
                {
                    t.join();
                }
                mythreads.clear();
            }
            {
                std::vector<SizeType> replicaCountDist(m_opt->m_replicaCount + 1, 0);
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    ++replicaCountDist[replicaCount[i]];
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %lld\n", i, (std::int64_t)(replicaCountDist[i]));
                }
            }
            size_t zeroReplicaCount = zeroReplicaSet.size();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting cut original:%lld relax:%lld\n", originalSize.load(),
                         relaxSize.load());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "Posting cut overflowHeadCount:%lld droppedReplicaCount:%lld zeroReplicaCount:%zu\n",
                         overflowPostingCount.load(), droppedReplicaCount.load(), zeroReplicaCount);

            auto t4 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

            auto fullVectors = p_reader->GetVectorSet();
            if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);

            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize versionMap\n");
            //m_versionMap->Initialize(m_opt->m_vectorSize, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity, &p_localToGlobal);

            if (p_localToGlobal.R() > 0) {
                for (SizeType i = 0; i < p_localToGlobal.R(); i++) {
                    SizeType globalID = *(p_localToGlobal[i]);
                    if (m_versionMap->Deleted(globalID)) m_versionMap->SetVersion(globalID, -1);
                }
            } else {
                for (SizeType i = 0; i < m_opt->m_vectorSize; i++) {
                    if (m_versionMap->Deleted(i)) m_versionMap->SetVersion(i, -1);
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing values to DB\n");

            if (p_localToGlobal.R() > 0) {
                p_headGlobaltoLocal.clear();
                for (int i = 0; i < p_headToLocal.R(); i++) {
                    *(p_headToLocal[i]) = *(p_localToGlobal[*(p_headToLocal[i])]);
                    if (p_headGlobaltoLocal.find(*(p_headToLocal[i])) == p_headGlobaltoLocal.end()) {
                        p_headGlobaltoLocal[*(p_headToLocal[i])] = i;
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Duplicate entry found for global ID: %lld previous i:%lld current i:%lld\n", static_cast<std::int64_t>(*(p_headToLocal[i])), 
                            static_cast<std::int64_t>(p_headGlobaltoLocal[*(p_headToLocal[i])]), static_cast<std::int64_t>(i));
                    }
                } 
            }
            if (ErrorCode::Success != WriteDownAllPostingToDB(p_headIndex, selections, fullVectors, postingListSize, p_headToLocal, p_localToGlobal)) return false;

            if (m_opt->m_update && !m_opt->m_allowZeroReplica && zeroReplicaCount > 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_opt->m_appendThreadNum, m_opt->m_reassignThreadNum);
                m_splitThreadPool = std::make_shared<SPDKThreadPool>();
                m_splitThreadPool->initSPDK(m_opt->m_appendThreadNum, this);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization, zeroReplicaCount:%zu\n", zeroReplicaCount);

                uint32_t splitNumBeforeZeroReplica = m_stat.m_splitNum;
                uint32_t reassignNumBeforeZeroReplica = m_stat.m_reAssignNum;
                uint32_t headMissBeforeZeroReplica = m_stat.m_headMiss.load();

                int zeroReplicaWorkerNum = (std::max)(1, (std::min)(static_cast<int>(zeroReplicaCount), m_opt->m_appendThreadNum));
                size_t zeroReplicaBatchSize = 4096;
                size_t zeroReplicaQueueLimit = (std::max)(static_cast<size_t>(4), static_cast<size_t>(zeroReplicaWorkerNum) * 2);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "SPFresh: zero-replica refill workers:%d batchSize:%zu queueLimit:%zu\n",
                             zeroReplicaWorkerNum, zeroReplicaBatchSize, zeroReplicaQueueLimit);

                std::mutex zeroReplicaQueueLock;
                std::condition_variable zeroReplicaQueueCv;
                std::deque<std::vector<SizeType>> zeroReplicaQueue;
                bool zeroReplicaQueueDone = false;
                std::atomic<bool> zeroReplicaFailed(false);
                std::atomic<SizeType> zeroReplicaProcessed(0);
                ErrorCode zeroReplicaRet = ErrorCode::Success;

                auto zeroReplicaFail = [&](ErrorCode code, SizeType vid) {
                    bool expected = false;
                    if (zeroReplicaFailed.compare_exchange_strong(expected, true)) {
                        zeroReplicaRet = code;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Fail to add index for zero replica ID: %lld, err=%d\n",
                                     static_cast<std::int64_t>(vid), static_cast<int>(code));
                    }
                    zeroReplicaQueueCv.notify_all();
                };

                auto enqueueZeroReplicaBatch = [&](std::vector<SizeType>& batch) {
                    std::unique_lock<std::mutex> lock(zeroReplicaQueueLock);
                    zeroReplicaQueueCv.wait(lock, [&]() {
                        return zeroReplicaFailed.load() || zeroReplicaQueue.size() < zeroReplicaQueueLimit;
                    });
                    if (zeroReplicaFailed.load()) return;
                    zeroReplicaQueue.emplace_back(std::move(batch));
                    lock.unlock();
                    zeroReplicaQueueCv.notify_one();
                };

                std::vector<std::thread> zeroReplicaWorkers;
                zeroReplicaWorkers.reserve(zeroReplicaWorkerNum);
                for (int workerId = 0; workerId < zeroReplicaWorkerNum; ++workerId)
                {
                    zeroReplicaWorkers.emplace_back([&, workerId]() {
                        ExtraWorkSpace workSpace;
                        m_headIndex->InitWorkSpace(&workSpace);
                        while (true)
                        {
                            std::vector<SizeType> batch;
                            {
                                std::unique_lock<std::mutex> lock(zeroReplicaQueueLock);
                                zeroReplicaQueueCv.wait(lock, [&]() {
                                    return zeroReplicaFailed.load() || !zeroReplicaQueue.empty() || zeroReplicaQueueDone;
                                });

                                if (zeroReplicaFailed.load()) return;
                                if (zeroReplicaQueue.empty()) {
                                    if (zeroReplicaQueueDone) return;
                                    continue;
                                }

                                batch = std::move(zeroReplicaQueue.front());
                                zeroReplicaQueue.pop_front();
                            }
                            zeroReplicaQueueCv.notify_one();

                            for (SizeType it : batch)
                            {
                                std::shared_ptr<VectorSet> vectorSet(new BasicVectorSet(ByteArray((std::uint8_t*)fullVectors->GetVector(it), m_vectorDataSize, false),
                                    GetEnumValueType<ValueType>(), m_opt->m_dim, 1));
                                ErrorCode addRet = AddIndex(&workSpace, vectorSet, it, m_opt->m_storage == Storage::TIKVIO && !m_opt->m_globalIDPath.empty());
                                if (addRet != ErrorCode::Success) {
                                    zeroReplicaFail(addRet, it);
                                    return;
                                }

                                SizeType processed = zeroReplicaProcessed.fetch_add(1) + 1;
                                if (processed % 1000000 == 0) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                                 "SPFresh: zero-replica refill progress %lld/%zu\n",
                                                 static_cast<std::int64_t>(processed), zeroReplicaCount);
                                }
                            }
                        }
                    });
                }

                std::vector<SizeType> zeroReplicaBatch;
                zeroReplicaBatch.reserve(zeroReplicaBatchSize);
                for (SizeType it : zeroReplicaSet)
                {
                    if (zeroReplicaFailed.load()) break;
                    zeroReplicaBatch.push_back(it);
                    if (zeroReplicaBatch.size() >= zeroReplicaBatchSize) {
                        enqueueZeroReplicaBatch(zeroReplicaBatch);
                        zeroReplicaBatch.clear();
                    }
                }
                if (!zeroReplicaFailed.load() && !zeroReplicaBatch.empty()) {
                    enqueueZeroReplicaBatch(zeroReplicaBatch);
                }

                {
                    std::lock_guard<std::mutex> lock(zeroReplicaQueueLock);
                    zeroReplicaQueueDone = true;
                }
                zeroReplicaQueueCv.notify_all();

                for (auto& worker : zeroReplicaWorkers) {
                    worker.join();
                }
                if (zeroReplicaFailed.load()) {
                    return false;
                }

                while (!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }

                if (m_opt->m_storage == Storage::FILEIO && !m_opt->m_globalIDPath.empty()) RefineIndex();

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "SPFresh: zero-replica refill done, processed:%zu, newSplits:%u, newHeadMiss:%u, newReassign:%u\n",
                             zeroReplicaCount, m_stat.m_splitNum - splitNumBeforeZeroReplica,
                             m_stat.m_headMiss.load() - headMissBeforeZeroReplica,
                             m_stat.m_reAssignNum - reassignNumBeforeZeroReplica);

                std::vector<SizeType> headOldtoNew;
                if (p_headIndex->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder, &headOldtoNew) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to save head index!\n");
                    return false;
                }
                if (!headOldtoNew.empty()) {
                    COMMON::Dataset<SizeType> new_topLocalToGlobalID(p_headIndex->GetNumSamples() - p_headIndex->GetNumDeleted(), 1,
                                                                     p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);
                    for (int i = 0; i < p_headToLocal.R(); i++)
                    {
                        if (p_headIndex->ContainSample(i))
                        {
                            *(new_topLocalToGlobalID[headOldtoNew[i]]) = *(p_headToLocal[i]);
                        }
                    }
                    new_topLocalToGlobalID.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile);
                } else {
                    p_headToLocal.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile);
                } 
                std::error_code ec;
                std::string prevHeadVectorFile = m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder + FolderSep + p_headIndex->GetParameter("VectorFilePath");
                std::string curHeadVectorFile = m_opt->m_indexDirectory + FolderSep + m_opt->m_headVectorFile;
                std::filesystem::copy_file(prevHeadVectorFile, curHeadVectorFile, std::filesystem::copy_options::overwrite_existing, ec);
                if (ec) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Could not copy previous layer headVectorFile %s to %s: %s\n",
                                prevHeadVectorFile.c_str(), curHeadVectorFile.c_str(), ec.message().c_str());
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Update headVectorFile from %s to %s for layer %d\n",
                                prevHeadVectorFile.c_str(), curHeadVectorFile.c_str(), m_layer);
                }
            } else {
                p_headToLocal.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile);
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: save versionMap\n");
            m_versionMap->Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer));

            auto t5 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        ErrorCode WriteDownAllPostingToDB(std::shared_ptr<VectorIndex>& p_headIndex, Selection& p_postingSelections, std::shared_ptr<VectorSet> p_fullVectors, std::vector<std::atomic_int>& postingSizes, COMMON::Dataset<SizeType>& p_headToGlobal, COMMON::Dataset<SizeType>& p_localToGlobal) {

            std::vector<std::thread> threads;
            std::atomic<SizeType> vectorsSent(0);
            ErrorCode ret = ErrorCode::Success;
            auto func = [&]()
            {
                ExtraWorkSpace workSpace;
                m_headIndex->InitWorkSpace(&workSpace);
                SizeType index = 0;
                while (true)
                {
                    index = vectorsSent.fetch_add(1);
                    if (index < postingSizes.size()) {
                        std::string postinglist(m_vectorInfoSize * postingSizes[index].load(), '\0');
                        char* ptr = (char*)postinglist.c_str();
			            std::size_t selectIdx = p_postingSelections.lower_bound(index);
                        SizeType postingID = *(p_headToGlobal[index]);
                        bool hasHead = false;
                        for (int j = 0; j < postingSizes[index].load(); ++j)
                        {
                            if (p_postingSelections[selectIdx].node != index) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                                ret = ErrorCode::Fail;
                                return;
                            }
                            SizeType localID = p_postingSelections[selectIdx++].tonode;
                            if (p_localToGlobal.R() > 0 && (localID < 0 || localID >= p_localToGlobal.R())) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                             "WriteDownAllPostingToDB: localID %lld out of range for localToGlobal size %lld\n",
                                             (std::int64_t)localID, (std::int64_t)p_localToGlobal.R());
                                ret = ErrorCode::Key_OverFlow;
                                return;
                            }
                            SizeType fullID = (p_localToGlobal.R() > 0) ? *(p_localToGlobal[localID]) : localID;
                            if (fullID == postingID) hasHead = true;
                            // if (id == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ID: %lld\n", (std::int64_t)fullID);
                            uint8_t version = m_versionMap->GetVersion(fullID);
                            // First Vector ID, then version, then Vector
                            Serialize(ptr, fullID, version, p_fullVectors->GetVector(localID));
                            ptr += m_vectorInfoSize;
                        }
                        if (!hasHead) {
                            if (postingSizes[index].load() < m_postingSizeLimit + m_bufferSizeLimit) {
                                postinglist.append(m_vectorInfoSize, '\0');
                                postingSizes[index]++;
                            }
                            Serialize(postinglist.data() + m_vectorInfoSize * (postingSizes[index].load() - 1), postingID, m_versionMap->GetVersion(postingID), p_headIndex->GetSample(index));
                        }

                        ErrorCode tmp;
                        int vsize;
                        if ((tmp = db->MergeWithCAS(DBKey(postingID), postinglist, MaxTimeout, &(workSpace.m_diskRequests), vsize)) !=
                            ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[WriteDB] Put %lld fail!\n", (std::int64_t)index);
                            ret = tmp;
                            return;
                        }
                        CheckCentroid(postingID, postinglist, "WriteDownAllPostingToDB");
                    }
                    else
                    {
                        return;
                    }
                }
            };

            for (int j = 0; j < m_opt->m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
            for (auto& thread : threads) { thread.join(); }
	        return ret;
        }

        ErrorCode AddIndex(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorSet>& p_vectorSet,
            SizeType begin, bool disableSplit = false) override {

            // Phase 1: RNGSelection + serialize + WAL for each vector, group by headID
            std::unordered_map<SizeType, std::string> headAppends;
            for (int v = 0; v < p_vectorSet->Count(); v++) {
                SizeType VID = begin + v;
                uint8_t version;
                if (!m_versionMap->TryGetDefaultVersionForNewVector(version)) {
                    if (m_versionMap->Deleted(VID)) m_versionMap->SetVersion(VID, -1);
                    version = m_versionMap->GetVersion(VID);
                }
                std::vector<BasicResult> selections(static_cast<size_t>(m_opt->m_replicaCount));
                int replicaCount = 1;
                RNGSelection(p_exWorkSpace, selections, (ValueType*)(p_vectorSet->GetVector(v)), replicaCount);

                std::string appendPosting(m_vectorInfoSize, '\0');
                Serialize((char*)(appendPosting.c_str()), VID, version, p_vectorSet->GetVector(v));
                if (m_opt->m_enableWAL && m_wal) {
                    m_wal->PutAssignment(appendPosting);
                }
                for (int i = 0; i < replicaCount; i++)
                {
                    headAppends[selections[i].VID] += appendPosting;
                }
            }

            // Phase 2: Batch append to each headID (one Merge per head instead of per vector)
            if (m_opt->m_storage == Storage::TIKVIO) return BatchAppend(p_exWorkSpace, headAppends, "AddIndex", disableSplit);

            for (auto& [headID, posting] : headAppends) {
                int appendNum = static_cast<int>(posting.size() / m_vectorInfoSize);
                ErrorCode ret;
                if ((ret = Append(p_exWorkSpace, headID, appendNum, posting)) != ErrorCode::Success)
                    return ret;
            }
            return ErrorCode::Success;
        }

        ErrorCode DeleteIndex(SizeType p_id) override {
            if (m_opt->m_enableWAL && m_wal) {
                std::string assignment(sizeof(SizeType), '\0');
                memcpy((char*)assignment.c_str(), &p_id, sizeof(SizeType));
                m_wal->PutAssignment(assignment);
            }
            if (m_versionMap->Delete(p_id)) return ErrorCode::Success;
            return ErrorCode::VectorNotFound;
        }

        ErrorCode ResetIndex(SizeType p_id) override {
            m_versionMap->SetVersion(p_id, -1);
            return ErrorCode::Success;
        }

        bool AllFinished() {
            if (!m_splitThreadPool) return true;

            size_t totalJobs = m_splitThreadPool->jobsize();
            unsigned int runningJobs = static_cast<unsigned int>(m_splitThreadPool->runningJobs());

            // [DIAG] sample worker pool queue depth (throttled to 1 per 50ms via CAS).
            // AllFinished is hot — checked in tight wait loops — so we MUST throttle to
            // avoid (a) histogram atomic contention and (b) flooding samples that would
            // drown out true depth distribution.
            {
                using clk = std::chrono::steady_clock;
                uint64_t nowUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    clk::now().time_since_epoch()).count();
                uint64_t prev = m_stat.m_queueDepthLastSampleUs.load(std::memory_order_relaxed);
                if (nowUs - prev >= 50000ULL) { // 50 ms
                    if (m_stat.m_queueDepthLastSampleUs.compare_exchange_strong(
                            prev, nowUs, std::memory_order_relaxed)) {
                        IndexStats::HistAdd(m_stat.m_queueDepthHist,  (uint64_t)totalJobs);
                        IndexStats::HistAdd(m_stat.m_runningJobsHist, (uint64_t)runningJobs);
                        m_stat.m_queueDepthTotal.fetch_add((uint64_t)totalJobs,    std::memory_order_relaxed);
                        m_stat.m_runningJobsTotal.fetch_add((uint64_t)runningJobs, std::memory_order_relaxed);
                        m_stat.m_queueDepthSampleCount.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            if (totalJobs > 0 && (totalJobs % 500 == 0 || totalJobs <= 10) && ShouldLogProgress(totalJobs)) {
                size_t completed = m_totalSplitCompleted.load();
                double avgSplitMs = completed > 0 ? (m_totalSplitTimeUs.load() / 1000.0 / completed) : 0;
                double maxSplitMs = m_maxSplitTimeUs.load() / 1000.0;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "layer %d pending queue:%zu split:%zu merge:%zu append:%zu reassign:%zu running:%u | "
                             "total_submitted split:%zu merge:%zu reassign:%zu append:%zu | "
                             "total_completed split:%zu merge:%zu reassign:%zu | "
                             "split_latency avg:%.1fms max:%.1fms\n",
                             m_layer, totalJobs, m_splitJobsInFlight.load(),
                             m_mergeJobsInFlight.load(), m_appendJobsInFlight.load(), m_reassignJobsInFlight.load(), runningJobs,
                             m_totalSplitSubmitted.load(), m_totalMergeSubmitted.load(), m_totalReassignSubmitted.load(), m_totalAppendCount.load(),
                             m_totalSplitCompleted.load(), m_totalMergeCompleted.load(), m_totalReassignCompleted.load(),
                             avgSplitMs, maxSplitMs);
            }
            if (runningJobs == 0 && totalJobs == 0) {
                if (!m_allDonePrinted) {
                    size_t totalSplit = m_totalSplitSubmitted.load();
                    size_t totalMerge = m_totalMergeSubmitted.load();
                    size_t totalAppend = m_totalAppendCount.load();
                    if (totalSplit > 0 || totalMerge > 0 || totalAppend > 0) {
                        size_t completedSplit = m_totalSplitCompleted.load();
                        double avgSplitMs = completedSplit > 0 ? (m_totalSplitTimeUs.load() / 1000.0 / completedSplit) : 0;
                        double maxSplitMs = m_maxSplitTimeUs.load() / 1000.0;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                     "layer %d ALL DONE | total_submitted split:%zu merge:%zu reassign:%zu append:%zu | "
                                     "total_completed split:%zu merge:%zu reassign:%zu | "
                                     "split_latency avg:%.1fms max:%.1fms\n",
                                     m_layer, totalSplit, totalMerge, m_totalReassignSubmitted.load(), totalAppend,
                                     m_totalSplitCompleted.load(), m_totalMergeCompleted.load(), m_totalReassignCompleted.load(),
                                     avgSplitMs, maxSplitMs);
                        // [DIAG] dump diagnostic histograms (lock/RMW/grpc/byte) at every ALL DONE boundary
                        {
                            uint64_t rmwN = m_stat.m_appendRmwSampleCount.load();
                            uint64_t splN = m_stat.m_splitLockSampleCount.load();
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("AppendLockWait", m_stat.m_appendLockWaitUs, m_stat.m_appendLockWaitTotalUs.load(), rmwN, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("AppendGetUs",    m_stat.m_appendGetUs,      m_stat.m_appendGetTotalUs.load(),     rmwN, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("AppendPutUs",    m_stat.m_appendPutUs,      m_stat.m_appendPutTotalUs.load(),     rmwN, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("AppendPostBytes",m_stat.m_appendPostingBytes,m_stat.m_appendPostingBytesTotal.load(), rmwN, "B").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("SplitLockWait",  m_stat.m_splitLockWaitUs,  m_stat.m_splitLockWaitTotalUs.load(),  splN, "us").c_str());
                            uint64_t reassignN = m_stat.m_reassignJobSampleCount.load();
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("ReassignJobUs",  m_stat.m_reassignJobLatencyUs, m_stat.m_reassignJobLatencyTotalUs.load(), reassignN, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[DIAG] layer %d ReassignSrc fromMerge=%lu fromSplitBatch=%lu (cumulative)\n",
                                m_layer,
                                (unsigned long)m_stat.m_reassignSubmittedFromMerge.load(),
                                (unsigned long)m_stat.m_reassignSubmittedFromSplitBatch.load());
                            uint64_t qN = m_stat.m_queueDepthSampleCount.load();
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("PoolQueueDepth", m_stat.m_queueDepthHist, m_stat.m_queueDepthTotal.load(), qN, "jobs").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("PoolRunning",    m_stat.m_runningJobsHist, m_stat.m_runningJobsTotal.load(), qN, "wkrs").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("AppendPreBytes", m_stat.m_appendPreBytes, m_stat.m_appendPreBytesTotal.load(), rmwN, "B").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[DIAG] layer %d AppendOutcome triggeredSplit=%lu nearThreshold(>=80%%)=%lu / RMWs=%lu (cumulative)\n",
                                m_layer,
                                (unsigned long)m_stat.m_appendTriggeredSplit.load(),
                                (unsigned long)m_stat.m_appendNearThreshold.load(),
                                (unsigned long)rmwN);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHistAndReset("BatchSplitPostingVectors", m_stat.m_splitPostingVectors, m_stat.m_splitPostingVectorsTotal, m_stat.m_splitPostingVectorSampleCount, "vecs").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHistAndReset("BatchSplitNewHeads", m_stat.m_splitNewHeadCount, m_stat.m_splitNewHeadCountTotal, m_stat.m_splitNewHeadSampleCount, "heads").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHistAndReset("BatchSplitReassignVectors", m_stat.m_splitReassignVectors, m_stat.m_splitReassignVectorsTotal, m_stat.m_splitReassignSampleCount, "vecs").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHistAndReset("BatchSplitReassignRecords", m_stat.m_splitReassignRecords, m_stat.m_splitReassignRecordsTotal, m_stat.m_splitReassignRecordSampleCount, "records").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] layer %d %s\n", m_layer,
                                IndexStats::FormatHistAndReset("BatchSplitReassignTargetHeads", m_stat.m_splitReassignTargetHeads, m_stat.m_splitReassignTargetHeadsTotal, m_stat.m_splitReassignTargetHeadSampleCount, "heads").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[DIAG] layer %d SplitHeadPath sameHead=%lu existingHeadMerge=%lu existingHeadResplit=%lu newHead=%lu (batch)\n",
                                m_layer,
                                (unsigned long)m_stat.m_splitSameHeadCount.exchange(0, std::memory_order_relaxed),
                                (unsigned long)m_stat.m_splitExistingHeadMergeCount.exchange(0, std::memory_order_relaxed),
                                (unsigned long)m_stat.m_splitExistingHeadMergeResplitCount.exchange(0, std::memory_order_relaxed),
                                (unsigned long)m_stat.m_splitCreatedNewHeadCount.exchange(0, std::memory_order_relaxed));
                            uint64_t mcN = m_stat.m_mcAppendSampleCount.load();
                            uint64_t mcS = m_stat.m_mcSplitWriteSampleCount.load();
                            uint64_t mcGM = m_stat.m_mcGetCountCacheMiss.load();
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("MCAppendUs", m_stat.m_mcAppendUs, m_stat.m_mcAppendTotalUs.load(), mcN, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("MCGetCountMissUs", m_stat.m_mcGetCountMissUs, m_stat.m_mcGetCountMissTotalUs.load(), mcGM, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("MCSplitDelUs", m_stat.m_mcSplitDelUs, m_stat.m_mcSplitDelTotalUs.load(), mcS, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("MCSplitPutBaseUs", m_stat.m_mcSplitPutBaseUs, m_stat.m_mcSplitPutBaseTotalUs.load(), mcS, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] layer %d %s\n", m_layer,
                                IndexStats::FormatHist("MCSplitSetCountUs", m_stat.m_mcSplitSetCountUs, m_stat.m_mcSplitSetCountTotalUs.load(), mcS, "us").c_str());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[DIAG-MC] layer %d CountCache hit=%lu miss=%lu (miss_ratio=%.4f)\n",
                                m_layer,
                                (unsigned long)m_stat.m_mcGetCountCacheHit.load(),
                                (unsigned long)mcGM,
                                (m_stat.m_mcGetCountCacheHit.load() + mcGM) ?
                                    (double)mcGM / (m_stat.m_mcGetCountCacheHit.load() + mcGM) : 0.0);
                            db->LogAsyncWaitStatsAndReset(m_layer);
                        }
                    }
                    m_allDonePrinted = true;
                }
                return true;
            }
            m_allDonePrinted = false;
            return false;
        } // && m_reassignThreadPool->allClear(); }
        void ForceCompaction() override { db->ForceCompaction(); }
        void GetDBStats() override { 
            db->GetStat();
            size_t completedSplit = m_totalSplitCompleted.load();
            double avgSplitMs = completedSplit > 0 ? (m_totalSplitTimeUs.load() / 1000.0 / completedSplit) : 0;
            double maxSplitMs = m_maxSplitTimeUs.load() / 1000.0;
            size_t totalJobs = m_splitThreadPool ? m_splitThreadPool->jobsize() : 0;
            // if (!ShouldLogProgress(totalJobs)) return;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "layer %d pending queue:%zu split:%zu merge:%zu append:%zu reassign:%zu running:%u | "
                         "total_submitted split:%zu merge:%zu reassign:%zu append:%zu | "
                         "total_completed split:%zu merge:%zu reassign:%zu | "
                         "split_latency avg:%.1fms max:%.1fms\n",
                         m_layer, totalJobs,
                         m_splitJobsInFlight.load(), m_mergeJobsInFlight.load(), m_appendJobsInFlight.load(), m_reassignJobsInFlight.load(),
                         m_splitThreadPool ? static_cast<unsigned int>(m_splitThreadPool->runningJobs()) : 0,
                         m_totalSplitSubmitted.load(), m_totalMergeSubmitted.load(), m_totalReassignSubmitted.load(), m_totalAppendCount.load(),
                         m_totalSplitCompleted.load(), m_totalMergeCompleted.load(), m_totalReassignCompleted.load(),
                         avgSplitMs, maxSplitMs);
        }

        int64_t GetNumBlocks() override
        {
            return db->GetNumBlocks();   
        }

        void GetIndexStats(int finishedInsert, bool cost, bool reset) override { m_stat.PrintStat(finishedInsert, cost, reset); }

        virtual ErrorCode CheckPosting(SizeType postingID, std::vector<std::uint8_t> *visited = nullptr,
                                       ExtraWorkSpace *p_exWorkSpace = nullptr) override
        {
            if (postingID < 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: Error postingID %lld (should be 0 ~ %d)\n",
                             (std::int64_t)postingID, MaxSize);
                return ErrorCode::Key_OverFlow;
            }
            ErrorCode ret = db->Check(DBKey(postingID), visited);
            if (ret != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: postingID %lld has wrong meta data\n",
                             (std::int64_t)postingID);
                return ret;
            }
            return ErrorCode::Success;
        }

        ErrorCode GetWritePosting(ExtraWorkSpace* p_exWorkSpace, SizeType pid, std::string& posting, bool write = false) override {
            ErrorCode ret;
            if (write) {
                if ((ret = db->Put(DBKey(pid), posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[GetWritePosting] Put fail!\n");
                    return ret;
                }                  
                CheckCentroid(pid, posting, "GetWritePosting"); 
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "PostingSize: %d\n", m_postingSizes.GetSize(pid));
            } else {
                if ((ret = db->Get(DBKey(pid), &posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[GetWritePosting] Get fail!\n");
                    return ret;
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode Checkpoint(std::string prefix) override {
            /**flush SPTAG, versionMap, block mapping, block pool**/
            /** Wait **/
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Checkpoint: waiting for index update complete (layer %d)\n", m_layer);
            auto waitStart = std::chrono::steady_clock::now();
            int pollCount = 0;
            while(!AllFinished())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (++pollCount % 250 == 0) { // every ~5 seconds
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - waitStart).count();
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Checkpoint: layer %d still waiting (%lld s elapsed)\n", m_layer, (long long)elapsed);
                }
            }
            {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - waitStart).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Checkpoint: layer %d background jobs done (waited %lld s)\n", m_layer, (long long)elapsed);
            }
            if (m_asyncStatus != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Checkpoint: resetting transient async error (code=%d) for layer %d\n",
                             (int)m_asyncStatus, m_layer);
                m_asyncStatus = ErrorCode::Success;
            }
            std::string p_persistenMap = prefix + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving version map\n");
            
            ErrorCode ret;
            if ((ret = m_versionMap->Save(p_persistenMap)) != ErrorCode::Success)
                return ret;

            if ((ret = db->Checkpoint(prefix)) != ErrorCode::Success)
                return ret;
            if (m_opt->m_enableWAL && m_wal) {
                /** delete all the previous record **/
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Checkpoint done, delete previous record\n");
                m_wal->ClearPreviousRecord();
            }
            return ErrorCode::Success;
        }

        inline SizeType DBKey(SizeType postingID) {
            return m_opt->m_maxID * m_layer + postingID;
        }

        inline std::shared_ptr<std::vector<SizeType>> DBKeys(std::vector<SizeType>& postingIDs) {
            std::shared_ptr<std::vector<SizeType>> keys = std::make_shared<std::vector<SizeType>>(postingIDs.size());
            for (int i = 0; i < postingIDs.size(); i++) {
                (*keys)[i] = DBKey(postingIDs[i]);
            }
            return keys;
        }

        private:

        int m_metaDataSize = 0;

        int m_vectorDataSize = 0;

        int m_vectorInfoSize = 0;

        int m_postingSizeLimit = INT_MAX;

        int m_bufferSizeLimit = INT_MAX;

        std::chrono::microseconds m_hardLatencyLimit = std::chrono::microseconds(2000);

        int m_mergeThreshold = 10;
        ErrorCode m_asyncStatus = ErrorCode::Success;

        std::shared_ptr<SPDKThreadPool> m_splitThreadPool;
        std::shared_ptr<SPDKThreadPool> m_reassignThreadPool;
    };
} // namespace SPTAG
#endif // _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_
