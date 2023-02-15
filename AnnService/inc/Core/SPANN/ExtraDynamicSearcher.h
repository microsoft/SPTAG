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
#include "inc/Core/Common/FineGrainedLock.h"
#include "PersistentBuffer.h"
#include "inc/Core/Common/PostingSizeRecord.h"
#include <map>
#include <cmath>
#include <climits>
#include <future>
#include <numeric>
#include <utility>
#include <random>
#include <tbb/concurrent_hash_map.h>

#ifdef ROCKSDB
#include "ExtraRocksDBController.h"
#endif

namespace SPTAG::SPANN {
    template <typename ValueType>
    class ExtraDynamicSearcher : public IExtraSearcher
    {
        class MergeAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType headID;
            bool disableReassign;
            std::function<void()> m_callback;
        public:
            MergeAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, bool disableReassign, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), headID(headID), disableReassign(disableReassign), m_callback(std::move(p_callback)) {}

            ~MergeAsyncJob() {}

            inline void exec(IAbortOperation* p_abort) override {
                m_extraIndex->MergePostings(m_index, headID, !disableReassign);
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class SplitAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType headID;
            bool disableReassign;
            std::function<void()> m_callback;
        public:
            SplitAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, bool disableReassign, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), headID(headID), disableReassign(disableReassign), m_callback(std::move(p_callback)) {}

            ~SplitAsyncJob() {}

            inline void exec(IAbortOperation* p_abort) override {
                m_extraIndex->Split(m_index, headID, !disableReassign);
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class ReassignAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            std::shared_ptr<std::string> vectorInfo;
            SizeType HeadPrev;
            std::function<void()> m_callback;
        public:
            ReassignAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex,
                std::shared_ptr<std::string> vectorInfo, SizeType HeadPrev, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), vectorInfo(std::move(vectorInfo)), HeadPrev(HeadPrev), m_callback(std::move(p_callback)) {}

            ~ReassignAsyncJob() {}

            void exec(IAbortOperation* p_abort) override {
                m_extraIndex->Reassign(m_index, vectorInfo, HeadPrev);
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

    private:
        std::shared_ptr<Helper::KeyValueIO> db;

        COMMON::VersionLabel* m_versionMap;
        Options* m_opt;

        std::mutex m_dataAddLock;

        COMMON::FineGrainedRWLock m_rwLocks;

        COMMON::PostingSizeRecord m_postingSizes;

        std::shared_ptr<Helper::ThreadPool> m_splitThreadPool;
        std::shared_ptr<Helper::ThreadPool> m_reassignThreadPool;

        IndexStats m_stat;

        tbb::concurrent_hash_map<SizeType, SizeType> m_splitList;
        tbb::concurrent_hash_map<SizeType, SizeType> m_mergeList;

    public:
        ExtraDynamicSearcher(const char* dbPath, int dim, int vectorlimit, bool useDirectIO, float searchLatencyHardLimit) {
#ifdef ROCKSDB
            db.reset(new RocksDBIO(dbPath, useDirectIO));
#endif
            m_metaDataSize = sizeof(int) + sizeof(uint8_t);
            m_vectorInfoSize = dim * sizeof(ValueType) + m_metaDataSize;
            m_postingSizeLimit = vectorlimit;
            m_hardLatencyLimit = searchLatencyHardLimit;
            LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", m_postingSizeLimit);
        }

        ~ExtraDynamicSearcher() {}

        //headCandidates: search data structrue for "vid" vector
        //headID: the head vector that stands for vid
        bool IsAssumptionBroken(VectorIndex* p_index, SizeType headID, QueryResult& headCandidates, SizeType vid)
        {
            p_index->SearchIndex(headCandidates);
            int replicaCount = 0;
            BasicResult* queryResults = headCandidates.GetResults();
            std::vector<Edge> selections(static_cast<size_t>(m_opt->m_replicaCount));
            for (int i = 0; i < headCandidates.GetResultNum() && replicaCount < m_opt->m_replicaCount; ++i) {
                if (queryResults[i].VID == -1) {
                    break;
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j) {
                    float nnDist = p_index->ComputeDistance(
                        p_index->GetSample(queryResults[i].VID),
                        p_index->GetSample(selections[j].node));
                    if (nnDist < queryResults[i].Dist) {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted)
                    continue;

                selections[replicaCount].node = queryResults[i].VID;
                // LOG(Helper::LogLevel::LL_Info, "head:%d\n", queryResults[i].VID);
                if (selections[replicaCount].node == headID) return false;
                ++replicaCount;
            }
            return true;
        }

        //Measure that in "headID" posting list, how many vectors break their assumption
        int QuantifyAssumptionBroken(VectorIndex* p_index, SizeType headID, std::string& postingList, SizeType SplitHead, std::vector<SizeType>& newHeads, std::set<int>& brokenID, int topK = 0, float ratio = 1.0)
        {
            int assumptionBrokenNum = 0;
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
                float_t dist = p_index->ComputeDistance(reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), p_index->GetSample(headID));
                // if (dist < Epsilon) LOG(Helper::LogLevel::LL_Info, "head found: vid: %d, head: %d\n", vid, headID);
                avgDist += dist;
                distanceSet.push_back(dist);
                if (m_versionMap->Deleted(vid) || m_versionMap->GetVersion(vid) != version) continue;
                COMMON::QueryResultSet<ValueType> headCandidates(reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), 64);
                if (brokenID.find(vid) == brokenID.end() && IsAssumptionBroken(headID, headCandidates, vid)) {
                    /*
                    float_t headDist = p_index->ComputeDistance(headCandidates.GetTarget(), p_index->GetSample(SplitHead));
                    float_t newHeadDist_1 = p_index->ComputeDistance(headCandidates.GetTarget(), p_index->GetSample(newHeads[0]));
                    float_t newHeadDist_2 = p_index->ComputeDistance(headCandidates.GetTarget(), p_index->GetSample(newHeads[1]));

                    float_t splitDist = p_index->ComputeDistance(p_index->GetSample(SplitHead), p_index->GetSample(headID));

                    float_t headToNewHeadDist_1 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[0]));
                    float_t headToNewHeadDist_2 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[1]));

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
                maxDist = distanceSet.back();
                // LOG(Helper::LogLevel::LL_Info, "distance: min: %f, max: %f, avg: %f, 50th: %f\n", minDist, maxDist, avgDist/postVectorNum, distanceSet[distanceSet.size() * 0.5]);
                // LOG(Helper::LogLevel::LL_Info, "assumption broken num: %d\n", assumptionBrokenNum);
                float_t splitDist = p_index->ComputeDistance(p_index->GetSample(SplitHead), p_index->GetSample(headID));

                float_t headToNewHeadDist_1 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[0]));
                float_t headToNewHeadDist_2 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[1]));

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
            int vectorNum = (postingLists[0].size() + postingLists[1].size()) / m_vectorInfoSize;
            LOG(Helper::LogLevel::LL_Info, "After Split%d, Top0 nearby posting lists, caseA : %d/%d\n", split_order, assumptionBrokenNum, vectorNum);
            return assumptionBrokenNum;
        }

        //Measure that around "headID", how many vectors break their assumption
        //"headID" is the head vector before split
        void QuantifySplitCaseB(VectorIndex* p_index, SizeType headID, std::vector<SizeType>& newHeads, SizeType SplitHead, int split_order, int assumptionBrokenNum_top0, std::set<int>& brokenID)
        {
            COMMON::QueryResultSet<ValueType> nearbyHeads(reinterpret_cast<const ValueType*>(p_index->GetSample(headID)), 64);
            std::vector<std::string> postingLists;
            p_index->SearchIndex(nearbyHeads);
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
                db->Get(queryResults[i].VID, &postingList);
                vectorNum += postingList.size() / m_vectorInfoSize;
                int tempNum = QuantifyAssumptionBroken(queryResults[i].VID, postingList, SplitHead, newHeads, brokenID, i, queryResults[i].Dist / queryResults[1].Dist);
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

        bool CheckIsNeedReassign(VectorIndex* p_index, std::vector<SizeType>& newHeads, ValueType* data, SizeType splitHead, float_t headToSplitHeadDist, float_t currentHeadDist, bool isInSplitHead, SizeType currentHead)
        {

            float_t splitHeadDist = p_index->ComputeDistance(data, p_index->GetSample(splitHead));

            if (isInSplitHead) {
                if (splitHeadDist >= currentHeadDist) return false;
            }
            else {
                float_t newHeadDist_1 = p_index->ComputeDistance(data, p_index->GetSample(newHeads[0]));
                float_t newHeadDist_2 = p_index->ComputeDistance(data, p_index->GetSample(newHeads[1]));
                if (splitHeadDist <= newHeadDist_1 && splitHeadDist <= newHeadDist_2) return false;
                if (currentHeadDist <= newHeadDist_1 && currentHeadDist <= newHeadDist_2) return false;
            }
            return true;
        }

        inline void Serialize(char* ptr, SizeType VID, std::uint8_t version, const void* vector) {
            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            memcpy(ptr + m_metaDataSize, vector, m_vectorInfoSize - m_metaDataSize);
        }

        void CalculatePostingDistribution(VectorIndex* p_index)
        {
            if (m_opt->m_inPlace) return;
            int top = m_postingSizeLimit / 10 + 1;
            int page = m_opt->m_postingPageLimit + 1;
            std::vector<int> lengthDistribution(top, 0);
            std::vector<int> sizeDistribution(page + 2, 0);
            int deletedHead = 0;
            for (int i = 0; i < p_index->GetNumSamples(); i++) {
                if (!p_index->ContainSample(i)) deletedHead++;
                lengthDistribution[m_postingSizes.GetSize(i) / 10]++;
                int size = m_postingSizes.GetSize(i) * m_vectorInfoSize;
                if (size < PageSize) {
                    if (size < 512) sizeDistribution[0]++;
                    else if (size < 1024) sizeDistribution[1]++;
                    else sizeDistribution[2]++;
                }
                else {
                    sizeDistribution[size / PageSize + 2]++;
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

        // TODO
        void RefineIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader,
            std::shared_ptr<VectorIndex> p_index)
        {
            LOG(Helper::LogLevel::LL_Info, "Begin PreReassign\n");
            std::atomic_bool doneReassign = false;
            // p_index->UpdateIndex();
            LOG(Helper::LogLevel::LL_Info, "Into PreReassign Loop\n");
            while (!doneReassign) {
                auto preReassignTimeBegin = std::chrono::high_resolution_clock::now();
                doneReassign = true;
                std::vector<std::thread> threads;
                std::atomic_int nextPostingID(0);
                int currentPostingNum = p_index->GetNumSamples();
                int limit = m_postingSizeLimit * m_opt->m_preReassignRatio;
                LOG(Helper::LogLevel::LL_Info, "Batch PreReassign, Current PostingNum: %d, Current Limit: %d\n", currentPostingNum, limit);
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
                                Split(p_index.get(), index, false);
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
                LOG(Helper::LogLevel::LL_Info, "rebuild cost: %.2lf s\n", elapsedSeconds);

                //p_index->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder);
                //LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
                //m_postingSizes.Save(m_opt->m_ssdInfoFile);

                for (int i = 0; i < p_index->GetNumSamples(); i++) {
                    db->Delete(i);
                }
                ForceCompaction();
                BuildIndex(p_reader, p_index, *m_opt, *m_versionMap);
                ForceCompaction();
                CalculatePostingDistribution(p_index.get());

                p_index->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder);
                LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
                m_postingSizes.Save(m_opt->m_ssdInfoFile);
            }
        }

        ErrorCode Split(VectorIndex* p_index, const SizeType headID, bool reassign = false)
        {
            auto splitBegin = std::chrono::high_resolution_clock::now();
            std::vector<SizeType> newHeadsID;
            std::vector<std::string> newPostingLists;
            double elapsedMSeconds;
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);

                std::string postingList;
                if (db->Get(headID, &postingList) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "Split fail to get oversized postings\n");
                    exit(0);
                }
                // postingList += appendPosting;
                // reinterpret postingList to vectors and IDs
                auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                SizeType postVectorNum = (SizeType)(postingList.size() / m_vectorInfoSize);
               
                COMMON::Dataset<ValueType> smallSample(postVectorNum, m_opt->m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity, (const void*)postingP, true, nullptr, m_metaDataSize, m_vectorInfoSize);
                //COMMON::Dataset<ValueType> smallSample(0, m_opt->m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity);  // smallSample[i] -> VID
                //std::vector<int> localIndicesInsert(postVectorNum);  // smallSample[i] = j <-> localindices[j] = i
                //std::vector<uint8_t> localIndicesInsertVersion(postVectorNum);
                std::vector<int> localIndices(postVectorNum);
                int index = 0;
                uint8_t* vectorId = postingP;
                for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                {
                    int VID = *((int*)(vectorId));
                    //LOG(Helper::LogLevel::LL_Info, "vector index/total:id: %d/%d:%d\n", j, m_postingSizes[headID].load(), *(reinterpret_cast<int*>(vectorId)));
                    uint8_t version = *(vectorId + sizeof(int));
                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;

                    //localIndicesInsert[index] = VID;
                    //localIndicesInsertVersion[index] = version;
                    //smallSample.AddBatch(1, (ValueType*)(vectorId + m_metaDataSize));
                    localIndices[index] = j;
                    index++;
                }
                // double gcEndTime = sw.getElapsedMs();
                // m_splitGcCost += gcEndTime;
                if (index < m_postingSizeLimit)
                {
                    char* ptr = (char*)(postingList.c_str());
                    for (int j = 0; j < index; j++, ptr += m_vectorInfoSize)
                    {
                        if (j == localIndices[j]) continue;
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                        //Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                    }
                    postingList.resize(index * m_vectorInfoSize);
                    m_postingSizes.UpdateSize(headID, index);
                    if (db->Put(headID, postingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Split Fail to write back postings\n");
                        exit(0);
                    }
                    m_stat.m_garbageNum++;
                    auto GCEnd = std::chrono::high_resolution_clock::now();
                    elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(GCEnd - splitBegin).count();
                    m_stat.m_garbageCost += elapsedMSeconds;
                    m_splitList.erase(headID);
                    return ErrorCode::Success;
                }
                //LOG(Helper::LogLevel::LL_Info, "Resize\n");
                localIndices.resize(index);

                auto clusterBegin = std::chrono::high_resolution_clock::now();
                // k = 2, maybe we can change the split number, now it is fixed
                SPTAG::COMMON::KmeansArgs<ValueType> args(2, smallSample.C(), (SizeType)localIndices.size(), 1, p_index->GetDistCalcMethod());
                std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));

                int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr, m_opt->m_virtualHead);

                auto clusterEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(clusterEnd - clusterBegin).count();
                m_stat.m_clusteringCost += elapsedMSeconds;
                // int numClusters = ClusteringSPFresh(smallSample, localIndices, 0, localIndices.size(), args, 10, false, m_opt->m_virtualHead);
                // exit(0);
                if (numClusters <= 1)
                {
                    LOG(Helper::LogLevel::LL_Info, "Cluserting Failed (The same vector), Cut to limit\n");
                    std::string newpostingList(m_postingSizeLimit * m_vectorInfoSize, '\0');
                    char* ptr = (char*)(newpostingList.c_str());
                    for (int j = 0; j < m_postingSizeLimit; j++, ptr += m_vectorInfoSize)
                    {
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                        //Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                    }
                    m_postingSizes.UpdateSize(headID, m_postingSizeLimit);
                    if (db->Put(headID, newpostingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Split fail to override postings cut to limit\n");
                        exit(0);
                    }
                    m_splitList.erase(headID);
                    return ErrorCode::Success;
                }

                long long newHeadVID = -1;
                int first = 0;
                bool theSameHead = false;
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
                    if (!theSameHead && p_index->ComputeDistance(args.centers + k * args._D, p_index->GetSample(headID)) < Epsilon) {
                        newHeadsID.push_back(headID);
                        newHeadVID = headID;
                        theSameHead = true;
                        if (db->Put(newHeadVID, newPostingLists[k]) != ErrorCode::Success) {
                            LOG(Helper::LogLevel::LL_Info, "Fail to override postings\n");
                            exit(0);
                        }
                        m_stat.m_theSameHeadNum++;
                    }
                    else {
                        int begin, end = 0;
                        p_index->AddIndexId(args.centers + k * args._D, 1, m_opt->m_dim, begin, end);
                        newHeadVID = begin;
                        newHeadsID.push_back(begin);
                        if (db->Put(newHeadVID, newPostingLists[k]) != ErrorCode::Success) {
                            LOG(Helper::LogLevel::LL_Info, "Fail to add new postings\n");
                            exit(0);
                        }
                        auto updateHeadBegin = std::chrono::high_resolution_clock::now();
                        p_index->AddIndexIdx(begin, end);
                        auto updateHeadEnd = std::chrono::high_resolution_clock::now();
                        elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(updateHeadEnd - updateHeadBegin).count();
                        m_stat.m_updateHeadCost += elapsedMSeconds;

                        std::lock_guard<std::mutex> tmplock(m_dataAddLock);
                        if (m_postingSizes.AddBatch(1) == ErrorCode::MemoryOverFlow) {
                            LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: NnewHeadVID: %d, Map Size:%d\n", newHeadVID, m_postingSizes.BufferSize());
                            exit(1);
                        }
                    }
                    // LOG(Helper::LogLevel::LL_Info, "Head id: %d split into : %d, length: %d\n", headID, newHeadVID, args.counts[k]);
                    first += args.counts[k];
                    m_postingSizes.UpdateSize(newHeadVID, args.counts[k]);
                }
                if (!theSameHead) {
                    p_index->DeleteIndex(headID);
                    m_postingSizes.UpdateSize(headID, 0);
                }
            }
            m_splitList.erase(headID);
            m_stat.m_splitNum++;
            // if (theSameHead) LOG(Helper::LogLevel::LL_Info, "The Same Head\n");
            // LOG(Helper::LogLevel::LL_Info, "head1:%d, head2:%d\n", newHeadsID[0], newHeadsID[1]);

            // QuantifySplit(headID, newPostingLists, newHeadsID, headID, split_order);
            // QuantifyAssumptionBrokenTotally();
            if (reassign) {
                auto reassignScanBegin = std::chrono::high_resolution_clock::now();

                CollectReAssign(p_index, headID, newPostingLists, newHeadsID);

                auto reassignScanEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanEnd - reassignScanBegin).count();

                m_stat.m_reassignScanCost += elapsedMSeconds;
            }
            // LOG(Helper::LogLevel::LL_Info, "After ReAssign\n");

            // QuantifySplit(headID, newPostingLists, newHeadsID, headID, split_order);
            auto splitEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - splitBegin).count();
            m_stat.m_splitCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        ErrorCode MergePostings(VectorIndex* p_index, SizeType headID, bool reassign = false)
        {
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);

                if (!p_index->ContainSample(headID)) return ErrorCode::Success;

                std::string mergedPostingList;
                std::set<SizeType> vectorIdSet;

                std::string currentPostingList;
                if (db->Get(headID, &currentPostingList) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "Split fail to get to be merged postings: %d\n", headID);
                    exit(0);
                }

                auto* postingP = reinterpret_cast<uint8_t*>(&currentPostingList.front());
                size_t postVectorNum = currentPostingList.size() / m_vectorInfoSize;
                int currentLength = 0;
                uint8_t* vectorId = postingP;
                for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                {
                    int VID = *((int*)(vectorId));
                    uint8_t version = *(vectorId + sizeof(int));
                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                    vectorIdSet.insert(VID);
                    mergedPostingList += currentPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                    currentLength++;
                }
                int totalLength = currentLength;

                if (currentLength > m_mergeThreshold)
                {
                    m_postingSizes.UpdateSize(headID, currentLength);
                    if (db->Put(headID, mergedPostingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Split Fail to write back postings\n");
                        exit(0);
                    }
                    m_mergeList.erase(headID);
                    return ErrorCode::Success;
                }

                QueryResult queryResults(p_index->GetSample(headID), m_opt->m_internalResultNum, false);
                p_index->SearchIndex(queryResults);

                std::string nextPostingList;

                for (int i = 1; i < queryResults.GetResultNum(); ++i)
                {
                    BasicResult* queryResult = queryResults.GetResult(i);
                    int nextLength = m_postingSizes.GetSize(queryResult->VID);
                    tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor headIDAccessor;
                    if (currentLength + nextLength < m_postingSizeLimit && !m_mergeList.find(headIDAccessor, queryResult->VID))
                    {
                        std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[queryResult->VID]);
                        if (!p_index->ContainSample(queryResult->VID)) continue;
                        if (db->Get(queryResult->VID, &nextPostingList) != ErrorCode::Success) {
                            LOG(Helper::LogLevel::LL_Info, "Split fail to get to be merged postings: %d\n", queryResult->VID);
                            exit(0);
                        }

                        postingP = reinterpret_cast<uint8_t*>(&nextPostingList.front());
                        postVectorNum = nextPostingList.size() / m_vectorInfoSize;
                        nextLength = 0;
                        vectorId = postingP;
                        for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                        {
                            int VID = *((int*)(vectorId));
                            uint8_t version = *(vectorId + sizeof(int));
                            if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                            if (vectorIdSet.find(VID) == vectorIdSet.end()) {
                                mergedPostingList += nextPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                                totalLength++;
                            }
                            nextLength++;
                        }
                        if (currentLength > nextLength) 
                        {
                            p_index->DeleteIndex(queryResult->VID);
                            if (db->Put(headID, mergedPostingList) != ErrorCode::Success) {
                                LOG(Helper::LogLevel::LL_Info, "Split fail to override postings after merge\n");
                                exit(0);
                            }
                            m_postingSizes.UpdateSize(queryResult->VID, 0);
                            m_postingSizes.UpdateSize(headID, totalLength);
                        } else
                        {
                            p_index->DeleteIndex(headID);
                            if (db->Put(queryResult->VID, mergedPostingList) != ErrorCode::Success) {
                                LOG(Helper::LogLevel::LL_Info, "Split fail to override postings after merge\n");
                                exit(0);
                            }
                            m_postingSizes.UpdateSize(queryResult->VID, totalLength);
                            m_postingSizes.UpdateSize(headID, 0);
                        }

                        if (reassign) 
                        {
                            /* ReAssign */
                            if (currentLength > nextLength) 
                            {
                                /* ReAssign queryResult->VID*/
                                postingP = reinterpret_cast<uint8_t*>(&nextPostingList.front());
                                for (int j = 0; j < nextLength; j++) {
                                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                    float origin_dist = p_index->ComputeDistance(p_index->GetSample(queryResult->VID), vector);
                                    float current_dist = p_index->ComputeDistance(p_index->GetSample(headID), vector);
                                    if (current_dist > origin_dist)
                                        ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), headID);
                                }
                            } else
                            {
                                /* ReAssign headID*/
                                postingP = reinterpret_cast<uint8_t*>(&currentPostingList.front());
                                for (int j = 0; j < currentLength; j++) {
                                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                    float origin_dist = p_index->ComputeDistance(p_index->GetSample(headID), vector);
                                    float current_dist = p_index->ComputeDistance(p_index->GetSample(queryResult->VID), vector);
                                    if (current_dist > origin_dist)
                                        ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), queryResult->VID);
                                }
                            }
                        }

                        m_mergeList.erase(headID);
                        m_stat.m_mergeNum++;

                        return ErrorCode::Success;
                    }
                }
            }
            return ErrorCode::Success;
        }

        inline void SplitAsync(VectorIndex* p_index, SizeType headID, std::function<void()> p_callback = nullptr)
        {
            tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor headIDAccessor;
            if (m_splitList.find(headIDAccessor, headID)) {
                return;
            }
            tbb::concurrent_hash_map<SizeType, SizeType>::value_type workPair(headID, headID);
            m_splitList.insert(workPair);

            auto* curJob = new SplitAsyncJob(p_index, this, headID, m_opt->m_disableReassign, p_callback);
            m_splitThreadPool->add(curJob);

        }

        inline void MergeAsync(VectorIndex* p_index, SizeType headID, std::function<void()> p_callback = nullptr)
        {
            tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor headIDAccessor;
            if (m_mergeList.find(headIDAccessor, headID)) {
                return;
            }
            tbb::concurrent_hash_map<SizeType, SizeType>::value_type workPair(headID, headID);
            m_mergeList.insert(workPair);

            auto* curJob = new MergeAsyncJob(p_index, this, headID, m_opt->m_disableReassign, p_callback);
            m_splitThreadPool->add(curJob);
        }

        inline void ReassignAsync(VectorIndex* p_index, std::shared_ptr<std::string> vectorInfo, SizeType HeadPrev, std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new ReassignAsyncJob(p_index, this, std::move(vectorInfo), HeadPrev, p_callback);
            m_splitThreadPool->add(curJob);
        }

        ErrorCode CollectReAssign(VectorIndex* p_index, SizeType headID, std::vector<std::string>& postingLists, std::vector<SizeType>& newHeadsID) {
            auto headVector = reinterpret_cast<const ValueType*>(p_index->GetSample(headID));
            size_t newHeadsNum = newHeadsID.size();
            std::vector<SizeType> HeadPrevTopK;
            std::vector<float> newHeadsDist;
            newHeadsDist.push_back(p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeadsID[0])));
            newHeadsDist.push_back(p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeadsID[1])));
            if (m_opt->m_reassignK > 0) {
                COMMON::QueryResultSet<ValueType> nearbyHeads(headVector, m_opt->m_reassignK);
                p_index->SearchIndex(nearbyHeads);
                BasicResult* queryResults = nearbyHeads.GetResults();
                for (int i = 0; i < nearbyHeads.GetResultNum(); i++) {
                    auto vid = queryResults[i].VID;
                    if (vid == -1) break;

                    if (find(newHeadsID.begin(), newHeadsID.end(), vid) == newHeadsID.end()) {
                        HeadPrevTopK.push_back(vid);
                        newHeadsID.push_back(vid);
                        newHeadsDist.push_back(queryResults[i].Dist);
                    }
                }
                auto reassignScanIOBegin = std::chrono::high_resolution_clock::now();
                if (db->MultiGet(HeadPrevTopK, &postingLists) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "ReAssign can't get all the near postings\n");
                    exit(0);
                }
                auto reassignScanIOEnd = std::chrono::high_resolution_clock::now();
                auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanIOEnd - reassignScanIOBegin).count();
                m_stat.m_reassignScanIOCost += elapsedMSeconds;
            }

            std::set<SizeType> reAssignVectorsTopK;

            for (int i = 0; i < postingLists.size(); i++) {
                auto& postingList = postingLists[i];
                size_t postVectorNum = postingList.size() / m_vectorInfoSize;
                auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                    if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                        m_stat.m_reAssignScanNum++;
                        float dist = p_index->ComputeDistance(p_index->GetSample(newHeadsID[i]), vector);
                        if (CheckIsNeedReassign(p_index, newHeadsID, vector, headID, newHeadsDist[i], dist, i < newHeadsNum, newHeadsID[i])) {
                            ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), newHeadsID[i]);
                            reAssignVectorsTopK.insert(vid);
                        }
                    }
                }
            }
            return ErrorCode::Success;
        }

        bool RNGSelection(std::vector<Edge>& selections, ValueType* queryVector, VectorIndex* p_index, SizeType p_fullID, int& replicaCount, int checkHeadID = -1)
        {
            QueryResult queryResults(queryVector, m_opt->m_internalResultNum, false);
            p_index->SearchIndex(queryResults);

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
                    float nnDist = p_index->ComputeDistance(p_index->GetSample(queryResult->VID),
                        p_index->GetSample(selections[j].node));
                    if (m_opt->m_rngFactor * nnDist <= queryResult->Dist)
                    {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted) continue;
                selections[replicaCount].node = queryResult->VID;
                selections[replicaCount].tonode = p_fullID;
                selections[replicaCount].distance = queryResult->Dist;
                if (selections[replicaCount].node == checkHeadID) {
                    return false;
                }
                ++replicaCount;
            }
            return true;
        }

        ErrorCode Append(VectorIndex* p_index, SizeType headID, int appendNum, std::string& appendPosting, int reassignThreshold = 0)
        {
            auto appendBegin = std::chrono::high_resolution_clock::now();
            if (appendPosting.empty()) {
                LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }

            if (appendNum == 0) {
                LOG(Helper::LogLevel::LL_Info, "Error!, headID :%d, appendNum:%d\n", headID, appendNum);
            }

        checkDeleted:
            if (!p_index->ContainSample(headID)) {
                for (int i = 0; i < appendNum; i++)
                {
                    uint32_t idx = i * m_vectorInfoSize;
                    SizeType VID = *(int*)(&appendPosting[idx]);
                    uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(int)]);
                    auto vectorInfo = std::make_shared<std::string>(appendPosting.c_str() + idx, m_vectorInfoSize);
                    if (m_versionMap->GetVersion(VID) == version) {
                        // LOG(Helper::LogLevel::LL_Info, "Head Miss To ReAssign: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);
                        m_stat.m_headMiss++;
                        ReassignAsync(p_index, vectorInfo, headID);
                    }
                    // LOG(Helper::LogLevel::LL_Info, "Head Miss Do Not To ReAssign: VID: %d, version: %d, current version: %d\n", *(int*)(&appendPosting[idx]), m_versionMap->GetVersion(*(int*)(&appendPosting[idx])), version);
                }
                return ErrorCode::Undefined;
            }
            // if (m_postingSizes.GetSize(headID) + appendNum > (m_postingSizeLimit + reassignThreshold)) {
            //     if (Split(headID, appendNum, appendPosting) == ErrorCode::FailSplit) {
            //         goto checkDeleted;
            //     }
            //     auto splitEnd = std::chrono::high_resolution_clock::now();
            //     double elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - appendBegin).count();
            //     m_splitCost += elapsedMSeconds;
            //     return ErrorCode::Success;
            // } else {
            double appendIOSeconds = 0;
            {
                //std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //ROCKSDB
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //SPDK
                if (!p_index->ContainSample(headID)) {
                    goto checkDeleted;
                }
                // for (int i = 0; i < appendNum; i++)
                // {
                //     uint32_t idx = i * m_vectorInfoSize;
                //     uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(int)]);
                //     LOG(Helper::LogLevel::LL_Info, "Append: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);

                // }
                // LOG(Helper::LogLevel::LL_Info, "Merge: headID: %d, appendNum:%d\n", headID, appendNum);
                auto appendIOBegin = std::chrono::high_resolution_clock::now();
                if (db->Merge(headID, appendPosting) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Error, "Merge failed!\n");
                    exit(1);
                }
                auto appendIOEnd = std::chrono::high_resolution_clock::now();
                appendIOSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendIOEnd - appendIOBegin).count();
                m_postingSizes.IncSize(headID, appendNum);
            }
            if (m_postingSizes.GetSize(headID) > (m_postingSizeLimit + reassignThreshold)) {
                SplitAsync(p_index, headID);
            }
            auto appendEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
            if (!reassignThreshold) {
                m_stat.m_appendTaskNum++;
                m_stat.m_appendIOCost += appendIOSeconds;
                m_stat.m_appendCost += elapsedMSeconds;
            }
            return ErrorCode::Success;
        }
        
        void Reassign(VectorIndex* p_index, std::shared_ptr<std::string> vectorInfo, SizeType HeadPrev)
        {
            SizeType VID = *((SizeType*)vectorInfo->c_str());
            uint8_t version = *((uint8_t*)(vectorInfo->c_str() + sizeof(VID)));
            // return;
            if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) {
                // LOG(Helper::LogLevel::LL_Info, "ReassignID: %d, version: %d, current version: %d\n", VID, version, m_versionMap->GetVersion(VID));
                return;
            }


            // tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor VIDAccessor;
            // if (m_reassignMap.find(VIDAccessor, VID) && VIDAccessor->second < version) {
            //     return;
            // }
            // tbb::concurrent_hash_map<SizeType, SizeType>::value_type workPair(VID, version);
            // m_reassignMap.insert(workPair);
            auto reassignBegin = std::chrono::high_resolution_clock::now();

            m_stat.m_reAssignNum++;

            auto selectBegin = std::chrono::high_resolution_clock::now();
            std::vector<Edge> selections(static_cast<size_t>(m_opt->m_replicaCount));
            int replicaCount;
            bool isNeedReassign = RNGSelection(selections, (ValueType*)(vectorInfo->c_str() + m_metaDataSize), p_index, VID, replicaCount, HeadPrev);
            auto selectEnd = std::chrono::high_resolution_clock::now();
            auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(selectEnd - selectBegin).count();
            m_stat.m_selectCost += elapsedMSeconds;

            auto reassignAppendBegin = std::chrono::high_resolution_clock::now();
            if (isNeedReassign && m_versionMap->GetVersion(VID) == version) {
                // LOG(Helper::LogLevel::LL_Info, "Update Version: VID: %d, version: %d, current version: %d\n", VID, version, m_versionMap.GetVersion(VID));
                m_versionMap->IncVersion(VID, &version);
                (*vectorInfo)[sizeof(VID)] = version;

                //LOG(Helper::LogLevel::LL_Info, "Reassign: oldVID:%d, replicaCount:%d, candidateNum:%d, dist0:%f\n", oldVID, replicaCount, i, selections[0].distance);
                for (int i = 0; i < replicaCount && m_versionMap->GetVersion(VID) == version; i++) {
                    //LOG(Helper::LogLevel::LL_Info, "Reassign: headID :%d, oldVID:%d, newVID:%d, posting length: %d, dist: %f, string size: %d\n", headID, oldVID, VID, m_postingSizes[headID].load(), selections[i].distance, newPart.size());
                    if (ErrorCode::Undefined == Append(p_index, selections[i].node, 1, *vectorInfo, 3)) {
                        // LOG(Helper::LogLevel::LL_Info, "Head Miss: VID: %d, current version: %d, another re-assign\n", VID, version);
                        break;
                    }
                }
            }
            auto reassignAppendEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignAppendEnd - reassignAppendBegin).count();
            m_stat.m_reAssignAppendCost += elapsedMSeconds;

            auto reassignEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignEnd - reassignBegin).count();
            m_stat.m_reAssignCost += elapsedMSeconds;
        }

        bool LoadIndex(Options& p_opt, COMMON::VersionLabel& p_versionMap) override {
            m_versionMap = &p_versionMap;
            m_opt = &p_opt;
            LOG(Helper::LogLevel::LL_Info, "DataBlockSize: %d, Capacity: %d\n", m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);

            m_versionMap->Load(m_opt->m_deleteIDFile, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
            m_postingSizes.Load(m_opt->m_ssdInfoFile, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);

            LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", m_versionMap->GetVectorNum());
            LOG(Helper::LogLevel::LL_Info, "Current posting num: %d.\n", m_postingSizes.GetPostingNum());

            if (m_opt->m_update) {
                //LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize persistent buffer\n");
                //m_persistentBuffer = std::make_shared<PersistentBuffer>(m_opt->m_persistentBufferPath, db);
                LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
                LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_opt->m_appendThreadNum, m_opt->m_reassignThreadNum);
                m_splitThreadPool = std::make_shared<Helper::ThreadPool>();
                m_splitThreadPool->init(m_opt->m_appendThreadNum);
                m_reassignThreadPool = std::make_shared<Helper::ThreadPool>();
                m_reassignThreadPool->init(m_opt->m_reassignThreadNum);

                // LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize dispatcher\n");
                // m_dispatcher = std::make_shared<Dispatcher>(m_persistentBuffer, m_opt->m_batch, m_splitThreadPool, m_reassignThreadPool, this);
                // m_dispatcher->run();
                LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
            }
            return true;
        }

        virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
            QueryResult& p_queryResults,
            std::shared_ptr<VectorIndex> p_index,
            SearchStats* p_stats, std::set<int>* truth, std::map<int, std::set<int>>* found) override
        {
            auto exStart = std::chrono::high_resolution_clock::now();

            const auto postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

            p_exWorkSpace->m_deduper.clear();

            auto exSetUpEnd = std::chrono::high_resolution_clock::now();

            p_stats->m_exSetUpLatency = ((double)std::chrono::duration_cast<std::chrono::microseconds>(exSetUpEnd - exStart).count()) / 1000;

            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);

            int diskRead = 0;
            int diskIO = 0;
            int listElements = 0;

            double compLatency = 0;
            double readLatency = 0;

            std::vector<std::string> postingLists;

            auto readStart = std::chrono::high_resolution_clock::now();
            db->MultiGet(p_exWorkSpace->m_postingIDs, &postingLists);
            auto readEnd = std::chrono::high_resolution_clock::now();

            diskIO += postingListCount;

            readLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count());

            for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                std::string& postingList = postingLists[pi];

                int vectorNum = (int)(postingList.size() / m_vectorInfoSize);

                int realNum = vectorNum;

                diskRead += (int)(postingList.size());
                listElements += vectorNum;

                auto compStart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < vectorNum; i++) {
                    char* vectorInfo = postingList.data() + i * m_vectorInfoSize;
                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                    if (m_versionMap->Deleted(vectorID)) {
                        realNum--;
                        listElements--;
                        continue;
                    }
                    if(p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) {
                        listElements--;
                        continue;
                    }
                    auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    queryResults.AddPoint(vectorID, distance2leaf);
                }
                auto compEnd = std::chrono::high_resolution_clock::now();
                if (realNum <= m_mergeThreshold) MergeAsync(p_index.get(), curPostingID);

                compLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(compEnd - compStart).count());

                auto exEnd = std::chrono::high_resolution_clock::now();

                if ((((double)std::chrono::duration_cast<std::chrono::microseconds>(exEnd - exStart).count()) / 1000 + p_stats->m_totalLatency) >= m_hardLatencyLimit) {
                    break;
                }

                if (truth) {
                    for (int i = 0; i < vectorNum; ++i) {
                        char* vectorInfo = postingList.data() + i * m_vectorInfoSize;
                        int vectorID = *(reinterpret_cast<int*>(vectorInfo));
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
        }

        bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt, COMMON::VersionLabel& p_versionMap, SizeType upperBound = -1) override {
            m_versionMap = &p_versionMap;
            m_opt = &p_opt;

            int numThreads = m_opt->m_iSSDNumberOfThreads;
            int candidateNum = m_opt->m_internalResultNum;
            std::unordered_set<SizeType> headVectorIDS;
            if (m_opt->m_headIDFile.empty()) {
                LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                return false;
            }

            if (fileexists((m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile).c_str()))
            {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize((m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Error, "failed open VectorIDTranslate: %s\n", m_opt->m_headIDFile.c_str());
                    return false;
                }

                std::uint64_t vid;
                while (ptr->ReadBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) == sizeof(vid))
                {
                    headVectorIDS.insert(static_cast<SizeType>(vid));
                }
                LOG(Helper::LogLevel::LL_Info, "Loaded %u Vector IDs\n", static_cast<uint32_t>(headVectorIDS.size()));
            }

            SizeType fullCount = 0;
            {
                auto fullVectors = p_reader->GetVectorSet();
                fullCount = fullVectors->Count();
                m_vectorInfoSize = fullVectors->PerVectorDataSize() + m_metaDataSize;
            }
            if (upperBound > 0) fullCount = upperBound;

            // m_metaDataSize = sizeof(int) + sizeof(uint8_t) + sizeof(float);
            m_metaDataSize = sizeof(int) + sizeof(uint8_t);

            LOG(Helper::LogLevel::LL_Info, "Build SSD Index.\n");

            Selection selections(static_cast<size_t>(fullCount) * m_opt->m_replicaCount, m_opt->m_tmpdir);
            LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
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
                LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
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
                        for (auto vid : headVectorIDS) {
                            if (vid >= start && vid < end) emptySet.insert(vid - start);
                        }
                    }
                    else {
                        emptySet = headVectorIDS;
                    }

                    int sampleNum = 0;
                    for (int j = start; j < end && sampleNum < sampleSize; j++)
                    {
                        if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                    }

                    float acc = 0;
#pragma omp parallel for schedule(dynamic)
                    for (int j = 0; j < sampleNum; j++)
                    {
                        COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                    }
                    acc = acc / sampleNum;
                    LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%d,%d) loaded with %d vectors (%zu) HeadIndex acc @%d:%f.\n", i, start, end, fullVectors->Count(), selections.m_selections.size(), candidateNum, acc);

                    p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), m_opt->m_replicaCount, numThreads, m_opt->m_gpuSSDNumTrees, m_opt->m_gpuSSDLeafSize, m_opt->m_rngFactor, m_opt->m_numGPUs);
                    LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                    for (SizeType j = start; j < end; j++) {
                        replicaCount[j] = 0;
                        size_t vecOffset = j * (size_t)m_opt->m_replicaCount;
                        if (headVectorIDS.count(j) == 0) {
                            for (int resNum = 0; resNum < m_opt->m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                ++postingListSize[selections[vecOffset + resNum].node];
                                selections[vecOffset + resNum].tonode = j;
                                ++replicaCount[j];
                            }
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
            LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

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
            LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

            auto postingSizeLimit = m_postingSizeLimit;
            if (m_opt->m_postingPageLimit > 0)
            {
                postingSizeLimit = static_cast<int>(m_opt->m_postingPageLimit * PageSize / m_vectorInfoSize);
            }

            LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);


            {
                std::vector<int> replicaCountDist(m_opt->m_replicaCount + 1, 0);
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    if (headVectorIDS.count(i) > 0) continue;
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
                if (postingListSize[i] <= postingSizeLimit) continue;

                std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, Selection::g_edgeComparer) - selections.m_selections.begin();

                for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                {
                    int tonode = selections.m_selections[selectIdx + dropID].tonode;
                    --replicaCount[tonode];
                }
                postingListSize[i] = postingSizeLimit;
            }

            if (m_opt->m_outputEmptyReplicaID)
            {
                std::vector<int> replicaCountDist(m_opt->m_replicaCount + 1, 0);
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                    return false;
                }
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    if (headVectorIDS.count(i) > 0) continue;

                    ++replicaCountDist[replicaCount[i]];

                    if (replicaCount[i] < 2)
                    {
                        long long vid = i;
                        if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                            LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
                            return false;
                        }
                    }
                }

                LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                }
            }


            auto t4 = std::chrono::high_resolution_clock::now();
            LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

            auto fullVectors = p_reader->GetVectorSet();
            if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);

            LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize versionMap\n");
            m_versionMap->Initialize(fullCount, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);

            LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing values to DB\n");

            std::vector<int> postingListSize_int(postingListSize.begin(), postingListSize.end());

            WriteDownAllPostingToDB(postingListSize_int, selections, fullVectors);

            m_postingSizes.Initialize((SizeType)(postingListSize.size()), p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);
            for (int i = 0; i < postingListSize.size(); i++) {
                m_postingSizes.UpdateSize(i, postingListSize[i]);
            }
            LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing SSD Info\n");
            m_postingSizes.Save(m_opt->m_ssdInfoFile);
            LOG(Helper::LogLevel::LL_Info, "SPFresh: save versionMap\n");
            m_versionMap->Save(m_opt->m_deleteIDFile);

            auto t5 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
            LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        void WriteDownAllPostingToDB(const std::vector<int>& p_postingListSizes, Selection& p_postingSelections, std::shared_ptr<VectorSet> p_fullVectors) {
    #pragma omp parallel for num_threads(10)
            for (int id = 0; id < p_postingListSizes.size(); id++)
            {
                std::string postinglist(m_vectorInfoSize * p_postingListSizes[id], '\0');
                char* ptr = (char*)postinglist.c_str();
                std::size_t selectIdx = p_postingSelections.lower_bound(id);
                for (int j = 0; j < p_postingListSizes[id]; ++j) {
                    if (p_postingSelections[selectIdx].node != id) {
                        LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                        exit(1);
                    }
                    SizeType fullID = p_postingSelections[selectIdx++].tonode;
                    uint8_t version = m_versionMap->GetVersion(fullID);
                    // First Vector ID, then version, then Vector
                    Serialize(ptr, fullID, version, p_fullVectors->GetVector(fullID));
                    ptr += m_vectorInfoSize;
                }
                db->Put(id, postinglist);
            }
        }

        ErrorCode AddIndex(std::shared_ptr<VectorSet>& p_vectorSet,
            std::shared_ptr<VectorIndex> p_index, SizeType begin) override {

            for (int v = 0; v < p_vectorSet->Count(); v++) {
                SizeType VID = begin + v;
                std::vector<Edge> selections(static_cast<size_t>(m_opt->m_replicaCount));
                int replicaCount;
                RNGSelection(selections, (ValueType*)(p_vectorSet->GetVector(v)), p_index.get(), VID, replicaCount);

                uint8_t version = m_versionMap->GetVersion(VID);
                std::string appendPosting(m_vectorInfoSize, '\0');
                Serialize((char*)(appendPosting.c_str()), VID, version, p_vectorSet->GetVector(v));
                for (int i = 0; i < replicaCount; i++)
                {
                    // AppendAsync(selections[i].node, 1, appendPosting_ptr);
                    Append(p_index.get(), selections[i].node, 1, appendPosting);
                }
            }
            return ErrorCode::Success;
        }

        SizeType SearchVector(std::shared_ptr<VectorSet>& p_vectorSet,
            std::shared_ptr<VectorIndex> p_index) override {
            
            QueryResult queryResults(p_vectorSet->GetVector(0), m_opt->m_internalResultNum, false);
            p_index->SearchIndex(queryResults);
            COMMON::OptHashPosVector m_deduper;
            m_deduper.clear();
            std::string postingList;
            for (int i = 0; i < queryResults.GetResultNum(); ++i)
            {
                db->Get(queryResults.GetResult(i)->VID, &postingList);
                int vectorNum = (int)(postingList.size() / m_vectorInfoSize);

                for (int i = 0; i < vectorNum; i++) {
                    char* vectorInfo = postingList.data() + i * m_vectorInfoSize;
                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                    if(m_deduper.CheckAndSet(vectorID)) {
                        continue;
                    }
                    auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    if (distance2leaf < 1e-6) return vectorID;
                }
            }
            return -1;
        }

        bool AllFinished() { return m_splitThreadPool->allClear() && m_reassignThreadPool->allClear(); }
        void ForceCompaction() override { db->ForceCompaction(); }
        void GetDBStats() override { 
            db->GetStat();
            LOG(Helper::LogLevel::LL_Info, "remain splitJobs: %d, reassignJobs: %d\n", m_splitThreadPool->jobsize(), m_reassignThreadPool->jobsize());
        }

        void GetIndexStats(int finishedInsert, bool cost, bool reset) override { m_stat.PrintStat(finishedInsert, cost, reset); }

        bool CheckValidPosting(SizeType postingID) override {
            return m_postingSizes.GetSize(postingID) > 0;
        }

    private:

        int m_metaDataSize = 0;
        
        int m_vectorInfoSize = 0;

        int m_postingSizeLimit = INT_MAX;

        float m_hardLatencyLimit = 2;

        int m_mergeThreshold = 10;
    };
} // namespace SPTAG
#endif // _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_
