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
#include "ExtraFileController.h"
#include <chrono>
#include <cstdint>
#include <map>
#include <cmath>
#include <cstring>
#include <climits>
#include <future>
#include <numeric>
#include <utility>
#include <random>

#ifdef SPDK
#include "ExtraSPDKController.h"
#endif

#ifdef ROCKSDB
#include "ExtraRocksDBController.h"
// enable rocksdb io_uring
extern "C" bool RocksDbIOUringEnable() { return true; }
#endif

namespace SPTAG::SPANN {
    template <typename ValueType>
    class ExtraDynamicSearcher : public IExtraSearcher
    {
        struct AppendPair
        {
            std::string BKTID;
            int headID;
            std::shared_ptr<std::string> headVec;
            std::string posting;

            AppendPair(std::string p_BKTID = "", int p_headID = -1, std::shared_ptr<std::string> p_headVec = nullptr, std::string p_posting = "") : BKTID(p_BKTID), headID(p_headID), headVec(p_headVec), posting(p_posting) {}
            inline bool operator < (const AppendPair& rhs) const
            {
                return std::strcmp(BKTID.c_str(), rhs.BKTID.c_str()) < 0;
            }

            inline bool operator > (const AppendPair& rhs) const
            {
                return std::strcmp(BKTID.c_str(), rhs.BKTID.c_str()) > 0;
            }
        };

        class MergeAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType m_headID;
            std::shared_ptr<std::string> m_headVec;
            bool m_disableReassign;
            std::function<void()> m_callback;
        public:
            MergeAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, std::shared_ptr<std::string> headVec, bool disableReassign, std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_headID(headID), m_headVec(headVec), m_disableReassign(disableReassign), m_callback(std::move(p_callback)) {}

            ~MergeAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->MergePostings((ExtraWorkSpace*)p_workSpace, m_headID, m_headVec, !m_disableReassign);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
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
            std::shared_ptr<std::string> m_headVec;
            bool m_disableReassign;
            std::function<void()> m_callback;
        public:
            SplitAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, std::shared_ptr<std::string> headVec, bool disableReassign, std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_headID(headID), m_headVec(headVec), m_disableReassign(disableReassign), m_callback(std::move(p_callback)) {}

            ~SplitAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->Split((ExtraWorkSpace*)p_workSpace, m_headID, m_headVec, !m_disableReassign);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
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
            std::shared_ptr<std::string> m_headVec;
            std::function<void()> m_callback;
        public:
            ReassignAsyncJob(ExtraDynamicSearcher<ValueType>* extraIndex,
                std::shared_ptr<std::string> vectorInfo, SizeType headPrev, std::shared_ptr<std::string> headVec, std::function<void()> p_callback)
                : m_extraIndex(extraIndex), m_vectorInfo(std::move(vectorInfo)), m_headPrev(headPrev), m_headVec(std::move(headVec)), m_callback(std::move(p_callback)) {}

            ~ReassignAsyncJob() {}
            
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }

            void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->Reassign((ExtraWorkSpace*)p_workSpace, m_vectorInfo, m_headPrev, m_headVec);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
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
                        extraIndex->InitWorkSpace(&workSpace);
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
        std::shared_ptr<Helper::Concurrent::ConcurrentQueue<int>> m_freeWorkSpaceIds;
        std::atomic<int> m_workspaceCount = 0;

        std::mutex m_asyncAppendLock;
        Helper::Concurrent::ConcurrentPriorityQueue<AppendPair> m_asyncAppendQueue;
             
        std::shared_ptr<Helper::KeyValueIO> db;

        SPANN::Index<ValueType>* m_headIndex;
        COMMON::VersionLabel m_versionMap;
        Options* m_opt;
        int m_layer;

        COMMON::FineGrainedRWLock m_rwLocks;

        IndexStats m_stat;

        std::shared_ptr<PersistentBuffer> m_wal;

        std::shared_timed_mutex m_splitListLock;
        Helper::Concurrent::ConcurrentMap<SizeType, int> m_splitList;

        std::shared_timed_mutex m_mergeListLock;
        Helper::Concurrent::ConcurrentSet<SizeType> m_mergeList;        

    public:
        ExtraDynamicSearcher(SPANN::Options& p_opt, int layer, SPANN::Index<ValueType>* headIndex) {
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

            if(p_opt.m_storage == Storage::FILEIO) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseFileIO\n");
                db.reset(new FileIO(p_opt, layer));
            }
            else if (p_opt.m_storage == Storage::SPDKIO) {
#ifdef SPDK
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseSPDK\n");
                db.reset(new SPDKIO(p_opt));
#else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ExtraDynamicSearcher:SPDK unsupport! Use -DSPDK to enable SPDK when doing cmake.\n");
                return;
#endif
            } 
            else if (p_opt.m_storage == Storage::ROCKSDBIO) {
#ifdef ROCKSDB
                std::string indexDir = (p_opt.m_recovery)? p_opt.m_persistentBufferPath + FolderSep: p_opt.m_indexDirectory + FolderSep;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseKV\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:dbPath:%s\n", (indexDir + p_opt.m_KVFile + "_" + std::to_string(layer)).c_str());
                db.reset(new RocksDBIO((indexDir + p_opt.m_KVFile + "_" + std::to_string(layer)).c_str(), p_opt.m_useDirectIO, p_opt.m_enableWAL, p_opt.m_recovery));
#else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ExtraDynamicSearcher:RocksDB unsupport! Use -DROCKSDB to enable RocksDB when doing cmake.\n");
                return;
#endif
            }

            
            m_hardLatencyLimit = std::chrono::microseconds((int)(p_opt.m_latencyLimit) * 1000);
            m_mergeThreshold = p_opt.m_mergeThreshold;          

            int maxIOThreads =  max(p_opt.m_ioThreads, (2 * max(p_opt.m_searchThreadNum, p_opt.m_iSSDNumberOfThreads) +
                                    p_opt.m_insertThreadNum + p_opt.m_reassignThreadNum + p_opt.m_appendThreadNum));
            m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
            for (int i = 0; i < maxIOThreads; i++) {
                m_freeWorkSpaceIds->push(i);
            }
            m_workspaceCount = maxIOThreads;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d, search limit: %f, merge threshold: %d\n", m_postingSizeLimit, p_opt.m_latencyLimit, m_mergeThreshold);
        }

        ~ExtraDynamicSearcher() {}

        virtual bool Available() override
        {
            return db->Available();
        }

        virtual SizeType GetNumSamples() const override
        {
            return m_versionMap.Count();
        }

        virtual bool ContainSample(const SizeType idx) const override
        {
            return !m_versionMap.Deleted(idx);
        }

        virtual SizeType GetNumDeleted() const override
        {
            return m_versionMap.GetDeleteCount();
        }

        virtual ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override
        {
            for (SizeType i = 0; i < m_versionMap.Count(); i++) 
            {
                if (!m_versionMap.Deleted(i))
                    globalIDs.push_back(i);
            }
            return ErrorCode::Success;
        }
        
        virtual ErrorCode AddIDCapacity(SizeType capa, bool deleted) override
        {
            SizeType begin = m_versionMap.Count();
            auto ret = m_versionMap.AddBatch(capa);
            if (ret == ErrorCode::Success && deleted) {
                for (SizeType i = begin; i < begin + capa; i++) {
                    m_versionMap.Delete(i);
                }
            }
            return ret;
        }
        
        bool CheckIsNeedReassign(std::vector<std::shared_ptr<std::string>>& newHeadsVec, ValueType* data, std::shared_ptr<std::string> splitHeadVec, float_t headToSplitHeadDist, float_t currentHeadDist, bool isInSplitHead)
        {
            float_t splitHeadDist = m_headIndex->ComputeDistance(data, splitHeadVec->data());

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
                if (VID < 0 || VID >= m_versionMap.Count())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "PrintErrorInPosting found wrong VID:%d in headID:%d (should be less than %d)\n", VID,
                                 headID, m_versionMap.Count());
                }
            }
        }

        // TODO
        ErrorCode RefineIndex() override
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin RefineIndex\n");

            std::atomic_bool doneReassign = false;
            Helper::Concurrent::ConcurrentMap<SizeType, std::shared_ptr<std::string>> mergelist;
            while (!doneReassign) {
                auto preReassignTimeBegin = std::chrono::high_resolution_clock::now();
                std::atomic<ErrorCode> finalcode = ErrorCode::Success;
                doneReassign = true;
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
                    InitWorkSpace(&workSpace);
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
                            if ((ret = db->Get(globalID, &postingList, MaxTimeout, &(workSpace.m_diskRequests))) !=
                                    ErrorCode::Success)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                "RefineIndex failed to get posting %lld, read size:%d\n",
                                                (std::int64_t)globalID, (int)(postingList.size()));
                                PrintErrorInPosting(postingList, globalID);
                                finalcode = ErrorCode::Fail;
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
                                
                                if (m_versionMap.Deleted(VID) || m_versionMap.GetVersion(VID) != version)
                                    continue;

                                if (VID == globalID) hasHead = true;

                                if (j != vectorCount)
                                {
                                    memcpy(postingP + vectorCount * m_vectorInfoSize, vectorId, m_vectorInfoSize);
                                }
                                vectorCount++;
                            }
                            if (!hasHead && vecStr != nullptr)
                            {
                                Serialize((char*)postingP + vectorCount * m_vectorInfoSize, globalID, m_versionMap.GetVersion(globalID), vecStr->data());
                                vectorCount++;
                            }
                            if (vectorCount <= m_mergeThreshold) mergelist.insert(Helper::Concurrent::ConcurrentMap<SizeType, std::shared_ptr<std::string>>::value_type(globalID, vecStr));

                            postingList.resize(vectorCount * m_vectorInfoSize);
                            if ((ret = db->Put(globalID, postingList, MaxTimeout,
                                                    &(workSpace.m_diskRequests))) !=
                                ErrorCode::Success)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                "RefineIndex Failed to write back compacted posting %lld\n",
                                                (std::int64_t)(globalID));
                                finalcode = ret;
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
                auto preReassignTimeEnd = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(preReassignTimeEnd - preReassignTimeBegin).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rebuild cost: %.2lf s\n", elapsedSeconds);

                if (finalcode != ErrorCode::Success)
                    return finalcode;

                if (mergelist.size() > 0)
                {
                    for (auto it = mergelist.begin(); it != mergelist.end(); ++it)
                    {
                        MergeAsync(it->first, it->second);
                    }
                }
                Checkpoint(m_opt->m_indexDirectory);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
            }
            return ErrorCode::Success;
        }
        
        ErrorCode Split(ExtraWorkSpace* p_exWorkSpace, const SizeType headID, std::shared_ptr<std::string> headVec, bool reassign = false, bool requirelock = true)
        {
            auto splitBegin = std::chrono::high_resolution_clock::now();
            std::vector<SizeType> newHeadsID;
            std::vector<std::shared_ptr<std::string>> newHeadsVec;
            std::vector<std::string> newPostingLists;
            ErrorCode ret;
            bool theSameHead = false;
            double elapsedMSeconds;
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID], std::defer_lock);
                if (requirelock) lock.lock();

                int retry = 0;
             Retry:
                if (!m_headIndex->ContainSample(headID, m_layer + 1)) return ErrorCode::Success;

                std::string postingList;
                auto splitGetBegin = std::chrono::high_resolution_clock::now();
                if ((ret=db->Get(headID, &postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                    ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Split fail to get oversized postings: key=%lld read size=%d\n",
                                 (std::int64_t)headID, (int)(postingList.size()), (int)(ret == ErrorCode::Success));
                    return ret;
                }
                auto splitGetEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitGetEnd - splitGetBegin).count();
                m_stat.m_getCost += elapsedMSeconds;
                // reinterpret postingList to vectors and IDs
                auto* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                SizeType postVectorNum = (SizeType)(postingList.size() / m_vectorInfoSize);
               
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: db get Posting %d successfully with length %d real length:%d vectorNum:%d\n", headID, (int)(postingList.size()), m_postingSizes.GetSize(headID), postVectorNum);
                COMMON::Dataset<ValueType> smallSample(postVectorNum, m_opt->m_dim, m_headIndex->m_iDataBlockSize, m_headIndex->m_iDataCapacity, (ValueType*)postingP, true, nullptr, m_metaDataSize, m_vectorInfoSize);
                //COMMON::Dataset<ValueType> smallSample(0, m_opt->m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity);  // smallSample[i] -> VID
                //std::vector<int> localIndicesInsert(postVectorNum);  // smallSample[i] = j <-> localindices[j] = i
                //std::vector<uint8_t> localIndicesInsertVersion(postVectorNum);
                std::vector<SizeType> localIndices;
                localIndices.reserve(postVectorNum);
                uint8_t* vectorId = postingP;
                SizeType headj = -1;
                bool hasHead = false;
                for (SizeType j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                {
                    //LOG(Helper::LogLevel::LL_Info, "vector index/total:id: %d/%d:%d\n", j, m_postingSizes[headID].load(), *(reinterpret_cast<int*>(vectorId)));
                    uint8_t version = *(vectorId + sizeof(SizeType));
                    SizeType VID = *((SizeType*)(vectorId));
                    if (VID < 0 || VID >= m_versionMap.Count())
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
                    
                    if (VID == headID) headj = j;
		            //if (VID >= m_versionMap.Count()) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "DEBUG: vector ID:%d total size:%d\n", VID, m_versionMap.Count());
                    if (m_versionMap.Deleted(VID) || m_versionMap.GetVersion(VID) != version) continue;

                    if (VID == headID) hasHead = true;
                    //localIndicesInsert[index] = VID;
                    //localIndicesInsertVersion[index] = version;
                    //smallSample.AddBatch(1, (ValueType*)(vectorId + m_metaDataSize));
                    localIndices.push_back(j);
                }
                if (!hasHead && headj >= 0) {
                    localIndices.push_back(headj);
                }

                // double gcEndTime = sw.getElapsedMs();
                // m_splitGcCost += gcEndTime;
		
                if (localIndices.size() < m_postingSizeLimit)
                {
                    char* ptr = (char*)(postingList.c_str());
                    for (int j = 0; j < localIndices.size(); j++, ptr += m_vectorInfoSize)
                    {
                        if (j == localIndices[j]) continue;
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                    }
                    postingList.resize(localIndices.size() * m_vectorInfoSize);
                    if ((ret=db->Put(headID, postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split Fail to write back posting %lld\n", (std::int64_t)(headID));
                        return ret;
                    }
                    m_stat.m_garbageNum++;
                    auto GCEnd = std::chrono::high_resolution_clock::now();
                    elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(GCEnd - splitBegin).count();
                    m_stat.m_garbageCost += elapsedMSeconds;
                    {
                        std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                        m_splitList.unsafe_erase(headID);
                    }
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "GC triggered: %d, new length: %d\n", headID, index);
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
                    int cut = 1;
                    if (m_opt->m_oneClusterCutMax) cut = m_postingSizeLimit;
                    std::string newpostingList(cut * m_vectorInfoSize, '\0');
                    char* ptr = (char*)(newpostingList.c_str());
                    float totaldist = 0.0f;
                    bool hasHead = false;
                    for (int j = 0; j < cut; j++, ptr += m_vectorInfoSize)
                    {
                        totaldist += m_headIndex->ComputeDistance(ptr + m_metaDataSize, args.centers);
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                        if (*((SizeType*)(ptr)) == headID) hasHead = true;
                        //Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                    }
                    if (!hasHead) memcpy(newpostingList.data(), postingList.c_str() + headj * m_vectorInfoSize, m_vectorInfoSize);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cluserting Failed (The same vector), Cluster total dist:%f Only Keep %d vectors.\n", totaldist, cut);
                   
                    if ((ret=db->Put(headID, newpostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split fail to override posting cut to limit for posting %lld\n", (std::int64_t)(headID));
                        return ret;
                    }
                    {
                        std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                        m_splitList.unsafe_erase(headID);
                    }
                    return ErrorCode::Success;
                }

                SizeType newHeadVID = -1;
                int first = 0;                
                newPostingLists.resize(2);
                for (int k = 0; k < 2; k++) {
                    if (args.counts[k] == 0)	continue;

                    newPostingLists[k].resize(args.counts[k] * m_vectorInfoSize);
                    char* ptr = (char*)(newPostingLists[k].c_str());
                    for (int j = 0; j < args.counts[k]; j++, ptr += m_vectorInfoSize)
                    {
                        memcpy(ptr, postingList.c_str() + localIndices[first + j] * m_vectorInfoSize, m_vectorInfoSize);
                        //Serialize(ptr, localIndicesInsert[localIndices[first + j]], localIndicesInsertVersion[localIndices[first + j]], smallSample[localIndices[first + j]]);
                    }
                    if (!theSameHead && m_headIndex->ComputeDistance(args.centers + k * args._D, headVec->c_str()) < Epsilon) {
                        newHeadsID.push_back(headID);
                        newHeadsVec.push_back(headVec);
                        newHeadVID = headID;
                        theSameHead = true;
                        auto splitPutBegin = std::chrono::high_resolution_clock::now();
                        if ((ret=db->Put(newHeadVID, newPostingLists[k], MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to override posting %lld\n", (std::int64_t)(newHeadVID));
                            return ret;
                        }
                        auto splitPutEnd = std::chrono::high_resolution_clock::now();
                        elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin).count();
                        m_stat.m_putCost += elapsedMSeconds;
                        m_stat.m_theSameHeadNum++;
                    }
                    else {
                        newHeadVID = *((SizeType*)(postingP + args.clusterIdx[k] * m_vectorInfoSize));
                        uint8_t version = *((uint8_t*)(postingP + args.clusterIdx[k] * m_vectorInfoSize + sizeof(SizeType)));

                        if (m_headIndex->ContainSample(newHeadVID, m_layer + 1)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Split: new head VID %lld already exists in head index. Do merging...\n", (std::int64_t)(newHeadVID));
                            // TODO：merge postings
                        } else {
                            newHeadsID.push_back(newHeadVID);
                            newHeadsVec.push_back(std::make_shared<std::string>((char*)(args.centers + k * args._D), m_vectorDataSize));
                            auto splitPutBegin = std::chrono::high_resolution_clock::now();
                            if ((ret=db->Put(newHeadVID, newPostingLists[k], MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to add new posting %lld\n", (std::int64_t)(newHeadVID));
                                return ret;
                            }                        
                            auto splitPutEnd = std::chrono::high_resolution_clock::now();
                            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin).count();
                            m_stat.m_putCost += elapsedMSeconds;

                            auto updateHeadBegin = std::chrono::high_resolution_clock::now();
                            m_headIndex->AddHeadIndex(args.centers + k * args._D, newHeadVID, version, m_opt->m_dim, m_layer + 1, p_exWorkSpace);
                            auto updateHeadEnd = std::chrono::high_resolution_clock::now();
                            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(updateHeadEnd - updateHeadBegin).count();
                            m_stat.m_updateHeadCost += elapsedMSeconds;
                            
                            if (m_opt->m_excludehead) m_versionMap.IncVersion(newHeadVID, &version);
                        }
                    }
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head id: %d split into : %d, length: %d\n", headID, newHeadVID, args.counts[k]);
                    first += args.counts[k];
                }
                if (!theSameHead) {
                    m_headIndex->DeleteIndex(headID, m_layer + 1);
                    if ((ret=db->Delete(headID)) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting in Split\n");
                        return ret;
                    }
                }

                {
                    std::unique_lock<std::shared_timed_mutex> tmplock(m_splitListLock);
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"erase: %d\n", headID);
                    m_splitList.unsafe_erase(headID);
                }
                
                for (int k = 0; k < 2; k++) {
                    if (args.counts[k] > m_postingSizeLimit) {
                        {
                            SplitAsync(newHeadsID[k], newHeadsVec[k], args.counts[k]);
                        }
                    }
                }
            }
            
            m_stat.m_splitNum++;
            if (reassign) {
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

        ErrorCode MergePostings(ExtraWorkSpace *p_exWorkSpace, SizeType headID, std::shared_ptr<std::string> headVec, bool reassign = false)
        {
            std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);

            if (!m_headIndex->ContainSample(headID, m_layer + 1)) {
                std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                m_mergeList.unsafe_erase(headID);
                return ErrorCode::Success;
            }

            {
                std::shared_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                if (m_mergeList.find(headID) == m_mergeList.end()) {
                    return ErrorCode::Success;
                }
            }

            std::string mergedPostingList;
            std::set<SizeType> vectorIdSet;

            std::string currentPostingList;
            ErrorCode ret;
            if ((ret = db->Get(headID, &currentPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                    ErrorCode::Success)
            {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Fail to get original merge postings: %lld, get size:%d\n",
                    (std::int64_t)headID, (int)(currentPostingList.size()));
                PrintErrorInPosting(currentPostingList, headID);
                return ret;
            }

            auto* postingP = reinterpret_cast<uint8_t*>(currentPostingList.data());
            size_t postVectorNum = currentPostingList.size() / m_vectorInfoSize;
            int currentLength = 0;
            uint8_t* vectorId = postingP;
            SizeType headj = -1;
            for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
            {
                SizeType VID = *((SizeType*)(vectorId));
                uint8_t version = *(vectorId + sizeof(SizeType));
                if (VID == headID) headj = j;
                if (m_versionMap.Deleted(VID) || m_versionMap.GetVersion(VID) != version) continue;
                vectorIdSet.insert(VID);
                mergedPostingList += currentPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                currentLength++;
            }
            if (vectorIdSet.find(headID) == vectorIdSet.end() && headj >= 0) {
                vectorIdSet.insert(headID);
                mergedPostingList += currentPostingList.substr(headj * m_vectorInfoSize, m_vectorInfoSize);
                currentLength++;
            }

            if (currentLength > m_mergeThreshold)
            {
                if ((ret=db->Put(headID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge Fail to write back posting %lld\n", (std::int64_t)headID);
                    return ret;
                }
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                    m_mergeList.unsafe_erase(headID);
                }
                return ErrorCode::Success;
            }

            COMMON::QueryResultSet<ValueType> queryResults((ValueType*)(headVec->data()), m_opt->m_internalResultNum, false, true);
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
                SizeType deletedHeadID = -1;
                std::shared_ptr<std::string> deletedHeadVec;
                std::string * deletedPostingList = nullptr;
                int deletedLength = 0;
                {
                    std::unique_lock<std::shared_timed_mutex> anotherLock(m_rwLocks[queryResult->VID], std::defer_lock);
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Locked: %d, to be lock: %d\n", headID, queryResult->VID);
                    if (m_rwLocks.hash_func(queryResult->VID) != m_rwLocks.hash_func(headID)) {
                        if (!anotherLock.try_lock()) {
                            auto* curJob = new MergeAsyncJob(this, headID, headVec, reassign, nullptr);
                            m_splitThreadPool->add(curJob);
                            return ErrorCode::Success;
                        }
                    }
                    if (!m_headIndex->ContainSample(queryResult->VID, m_layer + 1)) continue;
                    if ((ret=db->Get(queryResult->VID, &nextPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
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
                        if (m_versionMap.Deleted(VID) || m_versionMap.GetVersion(VID) != version) continue;
                        if (vectorIdSet.find(VID) == vectorIdSet.end()) {
                            mergedPostingList += nextPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                            dedupLength++;
                        }
                        nextLength++;
                    }
                    if (currentLength + dedupLength >= m_postingSizeLimit) continue;

                    if (currentLength >= nextLength) 
                    {                           
                        if ((ret=db->Put(headID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to override old posting %lld after merge\n", (std::int64_t)headID);
                            return ret;
                        }
                        m_headIndex->DeleteIndex(queryResult->VID, m_layer + 1);
                        if ((ret=db->Delete(queryResult->VID)) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting %lld in Merge\n", (std::int64_t)(queryResult->VID));
                            return ret;
                        }
                        nextHeadID = headID;
                        nextHeadVec = headVec;
                        deletedHeadID = queryResult->VID;
                        deletedHeadVec = std::make_shared<std::string>((char*)(queryResult->Vec.Data()), m_vectorDataSize);
                        deletedPostingList = &nextPostingList;
                        deletedLength = nextLength;
                    } else
                    { 
                        if ((ret=db->Put(queryResult->VID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to override posting %lld after merge\n", (std::int64_t)(queryResult->VID));
                            return ret;
                        }
                        m_headIndex->DeleteIndex(headID, m_layer + 1);
                        if ((ret = db->Delete(headID)) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting %lld in Merge\n", (std::int64_t)(headID));
                            return ret;
                        }
                        nextHeadID = queryResult->VID;
                        nextHeadVec = std::make_shared<std::string>((char*)(queryResult->Vec.Data()), m_vectorDataSize);
                        deletedHeadID = headID;
                        deletedHeadVec = headVec;
                        deletedPostingList = &currentPostingList;
                        deletedLength = currentLength;
                    }
                    if (m_rwLocks.hash_func(queryResult->VID) != m_rwLocks.hash_func(headID)) anotherLock.unlock();
                }

                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Release: %d, Release: %d\n", headID, queryResult->VID);
                lock.unlock();

                if (reassign) 
                {
                    postingP = reinterpret_cast<uint8_t*>(deletedPostingList->data());
                    for (int j = 0; j < deletedLength; j++) {
                        uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                        // SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                        ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                        float origin_dist = m_headIndex->ComputeDistance(deletedHeadVec->data(), vector);
                        float current_dist = m_headIndex->ComputeDistance(nextHeadVec->data(), vector);
                        if (current_dist > origin_dist)
                            ReassignAsync(std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), nextHeadID, nextHeadVec);
                    }

                    if (m_opt->m_excludehead)
                    {
                        if (!m_versionMap.Deleted(deletedHeadID))
                        {
                            std::shared_ptr<std::string> vectorinfo =
                                std::make_shared<std::string>(m_vectorInfoSize, ' ');
                            Serialize(vectorinfo->data(), deletedHeadID, m_versionMap.GetVersion(deletedHeadID),
                                        deletedHeadVec->data());
                            ReassignAsync(vectorinfo, -1, nextHeadVec);
                        }
                    }
                }

                {
                    {
                        std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                        m_mergeList.unsafe_erase(headID);
                        m_mergeList.unsafe_erase(queryResult->VID);
                    }
                    if (currentLength + dedupLength <= m_mergeThreshold) {
                        MergeAsync(nextHeadID, nextHeadVec);
                    }
                }
                m_stat.m_mergeNum++;
                return ErrorCode::Success;
            }
            mergedPostingList.resize(currentLength * m_vectorInfoSize);
            if ((ret=db->Put(headID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge Fail to write back posting %lld\n", (std::int64_t)headID);
                return ret;
            }
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                m_mergeList.unsafe_erase(headID);
            }
            return ErrorCode::Success;
        }

        inline void SplitAsync(SizeType headID, std::shared_ptr<std::string> headVec, int postingSize, std::function<void()> p_callback = nullptr)
        {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Into SplitAsync, current headID: %d, size: %d\n", headID, m_postingSizes.GetSize(headID));
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

            auto* curJob = new SplitAsyncJob(this, headID, headVec, m_opt->m_disableReassign, p_callback);
            m_splitThreadPool->add(curJob);
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Add to thread pool\n");
        }

        inline void MergeAsync(SizeType headID, std::shared_ptr<std::string> headVec, std::function<void()> p_callback = nullptr)
        {
            if (headVec == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergeAsync headVec is null, headID: %lld\n", (std::int64_t)headID);
                return;
            }
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                auto res = m_mergeList.insert(headID);
                if (!res.second)
                {
                    // Already in queue
                    return;
                }
            }

            auto* curJob = new MergeAsyncJob(this, headID, headVec, m_opt->m_disableReassign, p_callback);
            m_splitThreadPool->add(curJob);
        }

        inline void ReassignAsync(std::shared_ptr<std::string> vectorInfo, SizeType headPrev, std::shared_ptr<std::string> headVec, std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new ReassignAsyncJob(this, std::move(vectorInfo), headPrev, std::move(headVec), p_callback);
            m_splitThreadPool->add(curJob);
        }

        ErrorCode CollectReAssign(ExtraWorkSpace *p_exWorkSpace, SizeType headID, std::shared_ptr<std::string> headVec,
                                  std::vector<std::string> &postingLists, std::vector<SizeType> &newHeadsID, std::vector<std::shared_ptr<std::string>> &newHeadsVec,
                                  bool theSameHead)
        {
            auto headVector = reinterpret_cast<const ValueType*>(headVec->data());
            if (m_opt->m_excludehead && !theSameHead)
            {
                if (!m_versionMap.Deleted(headID))
                {
                    std::shared_ptr<std::string> vectorinfo = std::make_shared<std::string>(m_vectorInfoSize, ' ');
                    Serialize(vectorinfo->data(), headID, m_versionMap.GetVersion(headID), headVector);
                    ReassignAsync(vectorinfo, -1, headVec);
                }
            }
            std::vector<float> newHeadsDist;
            std::set<SizeType> reAssignVectorsTopK;
            newHeadsDist.push_back(m_headIndex->ComputeDistance(headVec->data(), newHeadsVec[0]->data()));
            newHeadsDist.push_back(m_headIndex->ComputeDistance(headVec->data(), newHeadsVec[1]->data()));
            for (int i = 0; i < postingLists.size(); i++) {
                auto& postingList = postingLists[i];
                size_t postVectorNum = postingList.size() / m_vectorInfoSize;
                auto* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "VID: %d, Head: %d\n", vid, newHeadsID[i]);
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType)));
                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                    if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap.Deleted(vid) && m_versionMap.GetVersion(vid) == version) {
                        m_stat.m_reAssignScanNum++;
                        float dist = m_headIndex->ComputeDistance(newHeadsVec[i]->data(), vector);
                        if (CheckIsNeedReassign(newHeadsVec, vector, headVec, newHeadsDist[i], dist, true)) {
                            ReassignAsync(std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), newHeadsID[i], newHeadsVec[i]);
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
                        auto vec = std::make_shared<std::string>((char*)(queryResults[i].Vec.Data()), m_vectorDataSize);
                        HeadPrevTopK.push_back(vid);
                        HeadPrevTopKVec.push_back(vec);
                        newHeadsID.push_back(vid);
                        newHeadsDist.push_back(queryResults[i].Dist);
                    }
                }
                auto reassignScanIOBegin = std::chrono::high_resolution_clock::now();
                ErrorCode ret;
                if ((ret = db->MultiGet(HeadPrevTopK, p_exWorkSpace->m_pageBuffers, m_hardLatencyLimit,
                                        &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ReAssign can't get all the near postings\n");
                    return ret;
                }

                auto reassignScanIOEnd = std::chrono::high_resolution_clock::now();
                auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignScanIOEnd - reassignScanIOBegin).count();
                m_stat.m_reassignScanIOCost += elapsedMSeconds;

                for (int i = 0; i < HeadPrevTopK.size(); i++)
                {
                    auto &buffer = (p_exWorkSpace->m_pageBuffers[i]);
                    size_t postVectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);
                    auto *postingP = buffer.GetBuffer();
                    for (int j = 0; j < postVectorNum; j++) {
                        uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                        SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d: VID: %d, Head: %d, size:%d/%d\n", i, vid, HeadPrevTopK[i], postingLists.size(), HeadPrevTopK.size());
                        uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType)));
                        ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                        if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap.Deleted(vid) && m_versionMap.GetVersion(vid) == version) {
                            m_stat.m_reAssignScanNum++;
                            float dist = m_headIndex->ComputeDistance(HeadPrevTopKVec[i]->data(), vector);
                            if (CheckIsNeedReassign(newHeadsVec, vector, headVec, newHeadsDist[i], dist, false)) {
                                ReassignAsync(std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), HeadPrevTopK[i], HeadPrevTopKVec[i]);
                                reAssignVectorsTopK.insert(vid);
                            }
                        }
                    }
                }
            }
            return ErrorCode::Success;
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

        void InitWorkSpace(ExtraWorkSpace* p_exWorkSpace, bool clear = false) override
        {
            if (clear) {
                p_exWorkSpace->Clear(m_opt->m_searchInternalResultNum, (max(m_opt->m_postingPageLimit, m_opt->m_searchPostingPageLimit) + m_opt->m_bufferLength) << PageSizeEx, true, m_opt->m_enableDataCompression);
            }
            else {
                p_exWorkSpace->Initialize(m_opt->m_maxCheck, m_opt->m_hashExp, max(m_opt->m_searchInternalResultNum, m_opt->m_reassignK), (max(m_opt->m_postingPageLimit, m_opt->m_searchPostingPageLimit) + m_opt->m_bufferLength) << PageSizeEx, true, m_opt->m_enableDataCompression);
                int wid = 0;
                if (m_freeWorkSpaceIds == nullptr || !m_freeWorkSpaceIds->try_pop(wid))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FreeWorkSpaceIds is not initalized or the workspace number is not enough! Please increase iothread number.\n");
                    p_exWorkSpace->m_diskRequests[0].m_status = -1;
                    return;
                }
                p_exWorkSpace->m_diskRequests[0].m_status = wid;
                p_exWorkSpace->m_callback = [m_freeWorkSpaceIds = m_freeWorkSpaceIds, wid] () {
                    if (m_freeWorkSpaceIds) m_freeWorkSpaceIds->push(wid);
                };
            }
        }

        ErrorCode AsyncAppend(ExtraWorkSpace* p_exWorkSpace, SizeType headID, std::shared_ptr<std::string> headVec, int appendNum, std::string& appendPosting, int reassignThreshold = 0)
        {
            if (m_asyncAppendQueue.size() >= m_opt->m_asyncAppendQueueSize) {
                std::lock_guard<std::mutex> lock(m_asyncAppendLock);
                if (m_asyncAppendQueue.size() < m_opt->m_asyncAppendQueueSize) {
                    m_asyncAppendQueue.push(AppendPair(m_headIndex->GetPriorityID(headID, headVec, m_layer + 1), headID, headVec, appendPosting));
                    return ErrorCode::Success;
                }

                AppendPair workPair;
                ErrorCode ret;
                while (m_asyncAppendQueue.try_pop(workPair)) {
                    if ((ret = Append(p_exWorkSpace, workPair.headID, workPair.headVec, 1, workPair.posting, reassignThreshold)) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncAppend: Append failed in async queue processing, headID: %d\n", workPair.headID);
                        return ret;
                    }
                }
            } else {
                m_asyncAppendQueue.push(AppendPair(m_headIndex->GetPriorityID(headID, headVec, m_layer + 1), headID, headVec, appendPosting));
            }
            return ErrorCode::Success;
        }

        ErrorCode Append(ExtraWorkSpace* p_exWorkSpace, SizeType headID, std::shared_ptr<std::string> headVec, int appendNum, std::string& appendPosting, int reassignThreshold = 0)
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
                    if (m_versionMap.GetVersion(VID) == version) {
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head Miss To ReAssign: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);
                        m_stat.m_headMiss++;
                        ReassignAsync(vectorInfo, headID, headVec);
                    }
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head Miss Do Not To ReAssign: VID: %d, version: %d, current version: %d\n", *(int*)(&appendPosting[idx]), m_versionMap.GetVersion(*(int*)(&appendPosting[idx])), version);
                }
                return ErrorCode::Success;
            }
            double appendIOSeconds = 0;
            int postingSize = 0;
            {
                //std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //ROCKSDB
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //SPDK
                ErrorCode ret;
                if (!m_headIndex->ContainSample(headID, m_layer + 1)) {
                    lock.unlock();
                    goto checkDeleted;
                }
                {
                    std::shared_lock<std::shared_timed_mutex> lock(m_splitListLock);
                    auto it = m_splitList.find(headID);
                    if (it != m_splitList.end()) {
                        postingSize = it->second;
                    }
                }
                if (postingSize + appendNum > (m_postingSizeLimit + m_bufferSizeLimit)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "After appending, the number of vectors in %lld exceeds the postingsize + buffersize (%d + %d)! Do split now...\n", (std::int64_t)headID, m_postingSizeLimit, m_bufferSizeLimit);
                    ret = Split(p_exWorkSpace, headID, headVec, !m_opt->m_disableReassign, false);
                    if (ret != ErrorCode::Success)
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split %lld failed!\n", (std::int64_t)headID);
                    lock.unlock();
                    goto checkDeleted;
                }

                auto appendIOBegin = std::chrono::high_resolution_clock::now();
                if ((ret = db->Merge(
                         headID, appendPosting, MaxTimeout, &(p_exWorkSpace->m_diskRequests), postingSize)) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge failed for %lld! Posting Size:%d, limit: %d\n", (std::int64_t)headID, postingSize, m_postingSizeLimit);
                    GetDBStats();
                    return ret;
                }
                auto appendIOEnd = std::chrono::high_resolution_clock::now();
                appendIOSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendIOEnd - appendIOBegin).count();

                postingSize /= m_vectorInfoSize;
            }
            if (postingSize > (m_postingSizeLimit + reassignThreshold)) {
                // SizeType VID = *(int*)(&appendPosting[0]);
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split Triggered by inserting VID: %d, reAssign: %d\n", VID, reassignThreshold);
                // GetDBStats();
                // if (m_postingSizes.GetSize(headID) > 120) {
                //     GetDBStats();
                // }
                if (!reassignThreshold) SplitAsync(headID, headVec, postingSize);
                else Split(p_exWorkSpace, headID, headVec, !m_opt->m_disableReassign);
            }
            auto appendEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
            if (!reassignThreshold) {
                m_stat.m_appendTaskNum++;
                m_stat.m_appendIOCost += appendIOSeconds;
                m_stat.m_appendCost += elapsedMSeconds;
            }
            // } else {
            //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReAssign Append To: %d\n", headID);
            // }
            return ErrorCode::Success;
        }
        
        ErrorCode Reassign(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<std::string> vectorInfo, SizeType headPrev, std::shared_ptr<std::string> headVector)
        {
            SizeType VID = *((SizeType*)vectorInfo->c_str());
            uint8_t version = *((uint8_t*)(vectorInfo->c_str() + sizeof(VID)));
            // return;
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignID: %d, version: %d, current version: %d, headPrev: %d\n", VID, version, m_versionMap.GetVersion(VID), headPrev);
            if (m_versionMap.Deleted(VID) || m_versionMap.GetVersion(VID) != version) {
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
            if (isNeedReassign && m_versionMap.GetVersion(VID) == version) {
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Update Version: VID: %d, version: %d, current version: %d\n", VID, version, m_versionMap.GetVersion(VID));
                m_versionMap.IncVersion(VID, &version);
                (*vectorInfo)[sizeof(VID)] = version;

                //LOG(Helper::LogLevel::LL_Info, "Reassign: oldVID:%d, replicaCount:%d, candidateNum:%d, dist0:%f\n", oldVID, replicaCount, i, selections[0].distance);
                for (int i = 0; i < replicaCount && m_versionMap.GetVersion(VID) == version; i++) {
                    //LOG(Helper::LogLevel::LL_Info, "Reassign: headID :%d, oldVID:%d, newVID:%d, posting length: %d, dist: %f, string size: %d\n", headID, oldVID, VID, m_postingSizes[headID].load(), selections[i].distance, newPart.size());
                    ErrorCode tmp = Append(p_exWorkSpace, selections[i].VID, std::make_shared<std::string>((char*)(selections[i].Vec.Data()), m_vectorDataSize), 1, *vectorInfo, 3);
                    if (ErrorCode::Success != tmp) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Head Miss: VID: %d, current version: %d, another re-assign\n", VID, version);
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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DataBlockSize: %d, Capacity: %d\n", m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
            std::string versionmapPath = m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer);
            if (m_opt->m_recovery) {
                versionmapPath = m_opt->m_persistentBufferPath + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading version map\n");
                m_versionMap.Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Current vector num: %d.\n", m_versionMap.Count());
            }
            else if (m_opt->m_storage == Storage::ROCKSDBIO) {
                m_versionMap.Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", m_versionMap.Count());
            } else if (m_opt->m_storage == Storage::SPDKIO || m_opt->m_storage == Storage::FILEIO) {
		        if (fileexists((m_opt->m_indexDirectory + FolderSep + m_opt->m_ssdIndex + "_" + std::to_string(m_layer)).c_str())) {
                	m_versionMap.Initialize(m_opt->m_vectorSize, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                    m_versionMap.DeleteAll();
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
                        InitWorkSpace(&workSpace);
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
                                    m_versionMap.SetVersion(VID, -1);
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
                m_versionMap.Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
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
                InitWorkSpace(&workSpace);
                do {
                    countAssignment++;
                    if (countAssignment % 10000 == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Process %d logs\n", countAssignment);
                    char* ptr = (char*)(assignment.c_str());
                    SizeType VID = *(reinterpret_cast<SizeType*>(ptr));
                    if (assignment.size() == m_vectorInfoSize) {
                        if (VID >= m_versionMap.Count()) {
                            if (m_versionMap.AddBatch(VID - m_versionMap.GetVectorNum() + 1) != ErrorCode::Success) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MemoryOverFlow: VID: %lld, Map Size:%d\n", (std::int64_t)VID, m_versionMap.BufferSize());
                                return false;
                            }
                        }
                        std::shared_ptr<VectorSet> vectorSet;
                        vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)ptr + m_metaDataSize, m_vectorDataSize, false),
                            GetEnumValueType<ValueType>(), m_opt->m_dim, 1));
                        AddIndex(&workSpace, vectorSet, VID);
                    } else {
                        m_versionMap.Delete(VID);
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
            SearchStats* p_stats, std::set<SizeType>* truth, std::map<SizeType, std::set<SizeType>>* found) override
        {
            if (p_stats) p_stats->m_exSetUpLatency = 0;

            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);

            int diskRead = 0;
            int diskIO = 0;
            int listElements = 0;

            double compLatency = 0;
            double readLatency = 0;
            std::chrono::microseconds remainLimit;
            if (p_stats) remainLimit = m_hardLatencyLimit - std::chrono::microseconds((int)p_stats->m_totalLatency);
            else remainLimit = m_hardLatencyLimit;

            auto readStart = std::chrono::high_resolution_clock::now();
            if (db->MultiGet(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers, remainLimit, &(p_exWorkSpace->m_diskRequests)) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[SearchIndex] read postings fail!\n");
                return ErrorCode::DiskIOFail;
            }
            auto readEnd = std::chrono::high_resolution_clock::now();
            readLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count());

            const auto postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
            for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                auto& buffer = (p_exWorkSpace->m_pageBuffers[pi]);
                char* p_postingListFullData = (char*)(buffer.GetBuffer());
                int vectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);

                diskIO += int((buffer.GetAvailableSize() + PageSize - 1) >> PageSizeEx);
                diskRead += (int)(buffer.GetAvailableSize());
                
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: postingList %d size:%d m_vectorInfoSize:%d vectorNum:%d\n", pi, (int)(postingList.size()), m_vectorInfoSize, vectorNum);
                int realNum = vectorNum;
                listElements += vectorNum;
                std::shared_ptr<std::string> headVec;
                auto compStart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < vectorNum; i++) {
                    char* vectorInfo = p_postingListFullData + i * m_vectorInfoSize;
                    SizeType vectorID = *(reinterpret_cast<SizeType*>(vectorInfo));

                    if (vectorID == curPostingID) headVec = std::make_shared<std::string>(vectorInfo + m_metaDataSize, m_vectorDataSize);
		            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: vectorID:%d\n", vectorID);
                    if (m_versionMap.Deleted(vectorID)) {
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
                if (m_opt->m_asyncMergeInSearch && realNum <= m_mergeThreshold) MergeAsync(curPostingID, headVec); // TODO: Control merge

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

            if (p_stats)
            {
                p_stats->m_compLatency = compLatency / 1000;
                p_stats->m_diskReadLatency = readLatency / 1000;
                p_stats->m_totalListElementsCount = listElements;
                p_stats->m_diskIOCount = diskIO;
                p_stats->m_diskAccessCount = diskRead / 1024;
            }
            queryResults.SetScanned(listElements);
            return ErrorCode::Success;
        }

        virtual ErrorCode SearchIndexWithoutParsing(ExtraWorkSpace* p_exWorkSpace)
        {
            int retry = 0;
            ErrorCode ret = ErrorCode::Undefined;
            while (retry < 2 && ret != ErrorCode::Success)
            {
                ret = db->MultiGet(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers, m_hardLatencyLimit,
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
                    if (!m_versionMap.Deleted(head->VID) && !p_exWorkSpace->m_deduper.CheckAndSet(head->VID) &&
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
                    if (vectorID >= m_versionMap.Count()) return ErrorCode::Key_OverFlow;
                    if (m_versionMap.Deleted(vectorID)) continue;
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
                if (!m_versionMap.Deleted(head->VID) && !p_exWorkSpace->m_deduper.CheckAndSet(head->VID) &&
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
            std::unordered_map<SizeType, SizeType> headVectorIDS;
            if (m_opt->m_headIDFile.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                return false;
            }

            if (m_layer > 0 && p_localToGlobal.R() == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty localToGlobal for non-leaf layer!\n");
                return false;
            }

            for (SizeType i = 0; i < p_headToLocal.R(); i++)
            {
                headVectorIDS[*(p_headToLocal[i])] = i;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded %llu Vector IDs\n", static_cast<uint64_t>(headVectorIDS.size()));

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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
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
                        emptySet.clear();
                        for (auto& pair : headVectorIDS) {
                            if (pair.first >= start && pair.first < end) emptySet.insert(pair.first - start);
                        }
                    }
                    else {
                        for (auto& pair : headVectorIDS) {
                            emptySet.insert(pair.first);
                        }
                    }

                    int sampleNum = 0;
                    for (SizeType j = start; j < end && sampleNum < sampleSize; j++)
                    {
                        if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
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
                        if (headVectorIDS.count(j) == 0) {
                            for (int resNum = 0; resNum < m_opt->m_replicaCount && selections[vecOffset + resNum].node != MaxSize; resNum++) {
                                ++postingListSize[selections[vecOffset + resNum].node];
                                selections[vecOffset + resNum].tonode = j;
                                ++replicaCount[j];
                            }
                        } else if (!p_opt.m_excludehead) {
                                selections[vecOffset].node = headVectorIDS[j];
                                selections[vecOffset].tonode = j;
                                ++postingListSize[selections[vecOffset].node];
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
                                relaxSize += relaxLimit;

                                std::size_t selectIdx =
                                    std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i,
                                                     Selection::g_edgeComparer) -
                                    selections.m_selections.begin();

                                for (size_t dropID = relaxLimit;
                                     dropID < postingListSize[i]; ++dropID)
                                {
                                    int tonode = selections.m_selections[selectIdx + dropID].tonode;
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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting cut original:%lld relax:%lld\n", originalSize.load(),
                         relaxSize.load());

    //         if (m_opt->m_outputEmptyReplicaID)
    //         {
    //             std::vector<int> replicaCountDist(m_opt->m_replicaCount + 1, 0);
    //             auto ptr = SPTAG::f_createIO();
    //             if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
    //                 SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
    //                 return false;
    //             }
    //             for (int i = 0; i < replicaCount.size(); ++i)
    //             {
    //                 if (headVectorIDS.count(i) > 0) continue;

    //                 ++replicaCountDist[replicaCount[i]];

    //                 if (replicaCount[i] < 2)
    //                 {
    //                     long long vid = i;
    //                     if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
    //                         SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
    //                         return false;
    //                     }
    //                 }
    //             }

    //             SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
    //             for (int i = 0; i < replicaCountDist.size(); ++i)
    //             {
    //                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
    //             }
    //         }


            auto t4 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

            auto fullVectors = p_reader->GetVectorSet();
            if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize versionMap\n");
            m_versionMap.Initialize(m_opt->m_vectorSize, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity, &p_localToGlobal);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing values to DB\n");

            if (p_localToGlobal.R() > 0) {
                p_headGlobaltoLocal.clear();
                for (int i = 0; i < p_headToLocal.R(); i++) {
                    *(p_headToLocal[i]) = *(p_localToGlobal[*(p_headToLocal[i])]);
                    p_headGlobaltoLocal[*(p_headToLocal[i])] = i;
                } 
            }
            if (ErrorCode::Success != WriteDownAllPostingToDB(selections, fullVectors, postingListSize, p_headToLocal, p_localToGlobal)) return false;

            if (m_opt->m_update && !m_opt->m_allowZeroReplica && zeroReplicaSet.size() > 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_opt->m_appendThreadNum, m_opt->m_reassignThreadNum);
                m_splitThreadPool = std::make_shared<SPDKThreadPool>();
                m_splitThreadPool->initSPDK(m_opt->m_appendThreadNum, this);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization, zeroReplicaCount:%d\n", (int)(zeroReplicaSet.size()));

                ExtraWorkSpace workSpace;
                InitWorkSpace(&workSpace);
                for (SizeType it : zeroReplicaSet)
                {
                    std::shared_ptr<VectorSet> vectorSet(new BasicVectorSet(ByteArray((std::uint8_t*)fullVectors->GetVector(it), m_vectorDataSize, false),
                        GetEnumValueType<ValueType>(), m_opt->m_dim, 1));
                    if (AddIndex(&workSpace, vectorSet, it) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to add index for zero replica ID: %d\n", it);
                        return false;
                    }
                }
                while (!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }

                if (p_headIndex->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to save head index!\n");
                    return false;
                }
                
                p_headToLocal.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile);
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: save versionMap\n");
            m_versionMap.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer));

            auto t5 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        ErrorCode WriteDownAllPostingToDB(Selection& p_postingSelections, std::shared_ptr<VectorSet> p_fullVectors, std::vector<std::atomic_int>& postingSizes, COMMON::Dataset<SizeType>& p_headToGlobal, COMMON::Dataset<SizeType>& p_localToGlobal) {

            std::vector<std::thread> threads;
            std::atomic<SizeType> vectorsSent(0);
            ErrorCode ret = ErrorCode::Success;
            auto func = [&]()
            {
                ExtraWorkSpace workSpace;
                InitWorkSpace(&workSpace);
                SizeType index = 0;
                while (true)
                {
                    index = vectorsSent.fetch_add(1);
                    if (index < postingSizes.size()) {
                        std::string postinglist(m_vectorInfoSize * postingSizes[index].load(), '\0');
                        char* ptr = (char*)postinglist.c_str();
			            std::size_t selectIdx = p_postingSelections.lower_bound(index);
                        for (int j = 0; j < postingSizes[index].load(); ++j)
                        {
                            if (p_postingSelections[selectIdx].node != index) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                                ret = ErrorCode::Fail;
                                return;
                            }
                            SizeType localID = p_postingSelections[selectIdx++].tonode;
                            SizeType fullID = (p_localToGlobal.R() > 0) ? *(p_localToGlobal[localID]) : localID;
                            // if (id == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ID: %d\n", fullID);
                            uint8_t version = m_versionMap.GetVersion(fullID);
                            // First Vector ID, then version, then Vector
                            Serialize(ptr, fullID, version, p_fullVectors->GetVector(localID));
                            ptr += m_vectorInfoSize;
                        }
                        ErrorCode tmp;
                        SizeType postingID = *(p_headToGlobal[index]);
                        if ((tmp = db->Put(postingID, postinglist, MaxTimeout, &(workSpace.m_diskRequests))) !=
                            ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[WriteDB] Put %lld fail!\n", (std::int64_t)index);
                            ret = tmp;
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
	        return ret;
        }

        ErrorCode AddIndex(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorSet>& p_vectorSet,
            SizeType begin) override {

            for (int v = 0; v < p_vectorSet->Count(); v++) {
                SizeType VID = begin + v;
                if (m_versionMap.Deleted(VID)) m_versionMap.SetVersion(VID, -1);
                std::vector<BasicResult> selections(static_cast<size_t>(m_opt->m_replicaCount));
                int replicaCount = 1;
                RNGSelection(p_exWorkSpace, selections, (ValueType*)(p_vectorSet->GetVector(v)), replicaCount);

                uint8_t version = m_versionMap.GetVersion(VID);
                std::string appendPosting(m_vectorInfoSize, '\0');
                Serialize((char*)(appendPosting.c_str()), VID, version, p_vectorSet->GetVector(v));
                if (m_opt->m_enableWAL && m_wal) {
                    m_wal->PutAssignment(appendPosting);
                }
                for (int i = 0; i < replicaCount; i++)
                {
                    // AppendAsync(selections[i].node, 1, appendPosting_ptr);
                    ErrorCode ret;
                    std::shared_ptr<std::string> headVec = std::make_shared<std::string>((char*)(selections[i].Vec.Data()), m_vectorDataSize);
                    if (m_opt->m_asyncAppendQueueSize > 0) {
                        if ((ret = AsyncAppend(p_exWorkSpace, selections[i].VID, headVec, 1, appendPosting)) != ErrorCode::Success)
                            return ret;
                    } else {
                        if ((ret = Append(p_exWorkSpace, selections[i].VID, headVec, 1, appendPosting)) !=
                            ErrorCode::Success)
                            return ret;
                    }
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode DeleteIndex(SizeType p_id) override {
            if (m_opt->m_enableWAL && m_wal) {
                std::string assignment(sizeof(SizeType), '\0');
                memcpy((char*)assignment.c_str(), &p_id, sizeof(SizeType));
                m_wal->PutAssignment(assignment);
            }
            if (m_versionMap.Delete(p_id)) return ErrorCode::Success;
            return ErrorCode::VectorNotFound;
        }

        bool AllFinished() { return m_splitThreadPool->allClear(); } // && m_reassignThreadPool->allClear(); }
        void ForceCompaction() override { db->ForceCompaction(); }
        void GetDBStats() override { 
            db->GetStat();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "remain splitJobs: %d, reassignJobs: %d, running split: %d, running reassign: %d\n", m_splitThreadPool->jobsize(), 0, m_splitThreadPool->runningJobs(), 0);
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
            ErrorCode ret = db->Check(postingID, visited);
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
                if ((ret = db->Put(pid, posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[GetWritePosting] Put fail!\n");
                    return ret;
                }                   
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "PostingSize: %d\n", m_postingSizes.GetSize(pid));
            } else {
                if ((ret = db->Get(pid, &posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) 
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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Waiting for index update complete\n");
            while(!AllFinished())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (m_asyncStatus != ErrorCode::Success)
                return m_asyncStatus;

            std::string p_persistenMap = prefix + FolderSep + m_opt->m_deleteIDFile + "_" + std::to_string(m_layer);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving version map\n");
            
            ErrorCode ret;
            if ((ret = m_versionMap.Save(p_persistenMap)) != ErrorCode::Success)
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
