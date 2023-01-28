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

#ifdef ROCKDB
#include "ExtraRocksDBController.h"
#endif

namespace SPTAG::SPANN {
    template <typename ValueType>
    class ExtraDynamicSearcher : public IExtraSearcher
    {
        class SplitAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType headID;
            std::function<void()> m_callback;
        public:
            SplitAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), headID(headID), m_callback(std::move(p_callback)) {}

            ~SplitAsyncJob() {}

            inline void exec(IAbortOperation* p_abort) override {
                m_extraIndex->Split(m_index, headID);
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
            std::shared_ptr<std::string> vectorContain;
            SizeType VID;
            SizeType HeadPrev;
            uint8_t version;
            std::function<void()> m_callback;
        public:
            ReassignAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex,
                std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), vectorContain(std::move(vectorContain)), VID(VID), HeadPrev(HeadPrev), version(version), m_callback(std::move(p_callback)) {}

            ~ReassignAsyncJob() {}

            void exec(IAbortOperation* p_abort) override {
                m_extraIndex->Reassign(m_index, vectorContain, VID, HeadPrev, version);
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

        IndexStats m_stat;

    public:
        ExtraDynamicSearcher(const char* dbPath, int dim, int vectorlimit, bool useDirectIO, float searchLatencyHardLimit) {
#ifdef ROCKSDB
            db.reset(new RocksDBIO());
#endif
            db->Initialize(dbPath, useDirectIO);
            m_metaDataSize = sizeof(int) + sizeof(uint8_t);
            m_vectorInfoSize = dim * sizeof(ValueType) + m_metaDataSize;
            m_postingSizeLimit = vectorlimit;
            m_hardLatencyLimit = searchLatencyHardLimit;
            LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", m_postingSizeLimit);
        }

        ~ExtraDynamicSearcher() override = default;

        //headCandidates: search data structrue for "vid" vector
        //headID: the head vector that stands for vid
        bool IsAssumptionBroken(VectorIndex* p_index, SizeType headID, COMMON::QueryResultSet<T>& headCandidates, SizeType vid)
        {
            p_index->SearchIndex(headCandidates);
            int replicaCount = 0;
            BasicResult* queryResults = headCandidates.GetResults();
            std::vector<EdgeInsert> selections(static_cast<size_t>(p_opt.m_replicaCount));
            for (int i = 0; i < headCandidates.GetResultNum() && replicaCount < p_opt.m_replicaCount; ++i) {
                if (queryResults[i].VID == -1) {
                    break;
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j) {
                    float nnDist = p_index->ComputeDistance(
                        p_index->GetSample(queryResults[i].VID),
                        p_index->GetSample(selections[j].headID));
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
        int QuantifyAssumptionBroken(VectorIndex* p_index, SizeType headID, std::string& postingList, SizeType SplitHead, std::vector<SizeType>& newHeads, std::set<int>& brokenID, int topK = 0, float ratio = 1.0)
        {
            int assumptionBrokenNum = 0;
            int m_vectorInfoSize = sizeof(T) * p_opt.m_dim + m_metaDataSize;
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
                float_t dist = p_index->ComputeDistance(reinterpret_cast<T*>(vectorId + m_metaDataSize), p_index->GetSample(headID));
                // if (dist < Epsilon) LOG(Helper::LogLevel::LL_Info, "head found: vid: %d, head: %d\n", vid, headID);
                avgDist += dist;
                distanceSet.push_back(dist);
                if (m_versionMap->Deleted(vid) || m_versionMap->GetVersion(vid) != version) continue;
                COMMON::QueryResultSet<T> headCandidates(NULL, 64);
                headCandidates.SetTarget(reinterpret_cast<T*>(vectorId + m_metaDataSize));
                headCandidates.Reset();
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
                maxDist = distanceSet[distanceSet.size() - 1];
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
            int vectorNum = (postingLists[0].size() + postingLists[1].size()) / (sizeof(T) * p_opt.m_dim + m_metaDataSize);
            LOG(Helper::LogLevel::LL_Info, "After Split%d, Top0 nearby posting lists, caseA : %d/%d\n", split_order, assumptionBrokenNum, vectorNum);
            return assumptionBrokenNum;
        }

        //Measure that around "headID", how many vectors break their assumption
        //"headID" is the head vector before split
        void QuantifySplitCaseB(VectorIndex* p_index, SizeType headID, std::vector<SizeType>& newHeads, SizeType SplitHead, int split_order, int assumptionBrokenNum_top0, std::set<int>& brokenID)
        {
            auto headVector = reinterpret_cast<const T*>(p_index->GetSample(headID));
            COMMON::QueryResultSet<T> nearbyHeads(NULL, 64);
            nearbyHeads.SetTarget(headVector);
            nearbyHeads.Reset();
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
                db->Get(queryResults[i].VID, postingList);
                vectorNum += postingList.size() / (sizeof(T) * p_opt.m_dim + m_metaDataSize);
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

        bool CheckIsNeedReassign(VectorIndex* p_index, std::vector<SizeType>& newHeads, T* data, SizeType splitHead, float_t headToSplitHeadDist, float_t currentHeadDist, bool isInSplitHead, SizeType currentHead)
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
            if (p_opt.m_inPlace) return;
            int top = m_postingSizeLimit / 10 + 1;
            int page = p_opt.m_postingPageLimit + 1;
            std::vector<int> lengthDistribution(top, 0);
            std::vector<int> sizeDistribution(page + 2, 0);
            size_t vectorInfoSize = p_opt.m_dim * sizeof(T) + m_metaDataSize;
            int deletedHead = 0;
            for (int i = 0; i < p_index->GetNumSamples(); i++) {
                if (!p_index->ContainSample(i)) deletedHead++;
                lengthDistribution[m_postingSizes.GetSize(i) / 10]++;
                int size = m_postingSizes.GetSize(i) * vectorInfoSize;
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
            std::shared_ptr<VectorIndex> p_index,
            Options& p_opt)
        {
            LOG(Helper::LogLevel::LL_Info, "Begin PreReassign\n");
            std::atomic_bool doneReassign = false;
            // p_index->UpdateIndex();
            // m_postingVecs.clear();
            // m_postingVecs.resize(p_index->GetNumSamples());
            // LOG(Helper::LogLevel::LL_Info, "Setting\n");
            // #pragma omp parallel for num_threads(32)
            // for (int i = 0; i < p_index->GetNumSamples(); i++) {
            //     db->Get(i, m_postingVecs[i]);
            // }
            LOG(Helper::LogLevel::LL_Info, "Into PreReassign Loop\n");
            while (!doneReassign) {
                auto preReassignTimeBegin = std::chrono::high_resolution_clock::now();
                doneReassign = true;
                std::vector<std::thread> threads;
                std::atomic_int nextPostingID(0);
                int currentPostingNum = p_index->GetNumSamples();
                int limit = m_postingSizeLimit * p_opt.m_preReassignRatio;
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
                                // std::string postingList = m_postingVecs[index];
                                std::string postingList;
                                db->Get(index, postingList);
                                auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                                size_t vectorInfoSize = p_opt.m_dim * sizeof(T) + m_metaDataSize;
                                size_t postVectorNum = postingList.size() / vectorInfoSize;
                                COMMON::Dataset<T> smallSample;  // smallSample[i] -> VID
                                std::shared_ptr<uint8_t> vectorBuffer(new uint8_t[p_opt.m_dim * sizeof(T) * postVectorNum], std::default_delete<uint8_t[]>());
                                std::vector<int> localIndices(postVectorNum);
                                auto vectorBuf = vectorBuffer.get();
                                for (int j = 0; j < postVectorNum; j++)
                                {
                                    uint8_t* vectorId = postingP + j * vectorInfoSize;
                                    localIndices[j] = j;
                                    memcpy(vectorBuf, vectorId + m_metaDataSize, p_opt.m_dim * sizeof(T));
                                    vectorBuf += p_opt.m_dim * sizeof(T);
                                }
                                smallSample.Initialize(postVectorNum, p_opt.m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity, reinterpret_cast<T*>(vectorBuffer.get()), false);
                                SPTAG::COMMON::KmeansArgs<T> args(2, smallSample.C(), (SizeType)localIndices.size(), 1, p_index->GetDistCalcMethod());
                                std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));
                                int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr, p_opt.m_virtualHead);
                                bool theSameHead = false;
                                for (int k = 0; k < 2; k++) {
                                    if (args.counts[k] == 0)	continue;
                                    if (!theSameHead && p_index->ComputeDistance(args.centers + k * args._D, p_index->GetSample(index)) < Epsilon) {
                                        theSameHead = true;
                                    }
                                    else {
                                        int begin, end = 0;
                                        p_index->AddIndexId(args.centers + k * args._D, 1, p_opt.m_dim, begin, end);
                                        p_index->AddIndexIdx(begin, end);
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
                                    p_index->DeleteIndex(index);
                                }
                            }
                        }
                        else
                        {
                            return;
                        }
                    }
                };
                for (int j = 0; j < p_opt.m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }
                auto preReassignTimeEnd = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(preReassignTimeEnd - preReassignTimeBegin).count();
                LOG(Helper::LogLevel::LL_Info, "rebuild cost: %.2lf s\n", elapsedSeconds);
                p_index->SaveIndex(p_opt.m_indexDirectory + FolderSep + p_opt.m_headIndexFolder);
                LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
                m_postingSizes.Save(p_opt.m_ssdInfoFile);
                for (int i = 0; i < p_index->GetNumSamples(); i++) {
                    db->Delete(i);
                }
                ForceCompaction();
                Rebuild(p_reader);
                ForceCompaction();
                CalculatePostingDistribution();

                p_index->SaveIndex(p_opt.m_indexDirectory + FolderSep + p_opt.m_headIndexFolder);
                LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
                m_postingSizes.Save(p_opt.m_ssdInfoFile);
            }
            return;
        }

        int ClusteringSPFresh(const COMMON::Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last,
            COMMON::KmeansArgs<T>& args, int tryIters, bool debug, bool virtualCenter)
        {
            int bestCount = -1;
            for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
                for (int k = 0; k < args._DK; k++) {
                    SizeType randid = COMMON::Utils::rand(last, first);
                    std::memcpy(args.centers + k * args._D, data[indices[randid]], sizeof(T) * args._D);
                }
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                COMMON::KmeansAssign(data, indices, first, last, args, true, 0);
                int tempCount = __INT_MAX__;
                for (int i = 0; i < args._K; i++) if (args.newCounts[i] < tempCount) tempCount = args.newCounts[i];
                if (tempCount > bestCount) {
                    bestCount = tempCount;
                    memcpy(args.newTCenters, args.centers, sizeof(T) * args._K * args._D);
                    memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);
                }
            }
            float currDiff, currDist, minClusterDist = MaxDist;
            int noImprovement = 0;
            for (int iter = 0; iter < 100; iter++) {
                std::memcpy(args.centers, args.newTCenters, sizeof(T) * args._K * args._D);
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

        ErrorCode Split(VectorIndex* p_index, const SizeType headID)
        {
            auto splitBegin = std::chrono::high_resolution_clock::now();
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);

                if (m_postingSizes.GetSize(headID) < m_postingSizeLimit) {
                    return ErrorCode::FailSplit;
                }
                std::string postingList;
                if (db->Get(headID, &postingList) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "Split fail to get oversized postings\n");
                    exit(0);
                }
                // postingList += appendPosting;
                // reinterpret postingList to vectors and IDs
                auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                size_t postVectorNum = postingList.size() / m_vectorInfoSize;
                COMMON::Dataset<ValueType> smallSample(0, m_opt->m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity);  // smallSample[i] -> VID
                std::vector<int> localIndicesInsert(postVectorNum);  // smallSample[i] = j <-> localindices[j] = i
                std::vector<uint8_t> localIndicesInsertVersion(postVectorNum);
                std::vector<int> localIndices(postVectorNum);
                int index = 0;
                for (int j = 0, uint8_t* vectorId = postingP; j < postVectorNum; j++, vectorId += vectorInfoSize)
                {
                    int VID = *(reinterpret_cast<int*>(vectorId));
                    //LOG(Helper::LogLevel::LL_Info, "vector index/total:id: %d/%d:%d\n", j, m_postingSizes[headID].load(), *(reinterpret_cast<int*>(vectorId)));
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;

                    localIndicesInsert[index] = VID;
                    localIndicesInsertVersion[index] = version;
                    localIndices[index] = index;
                    smallSample.AddBatch((T*)(vectorId + m_metaDataSize), 1);
                    index++;
                }
                // double gcEndTime = sw.getElapsedMs();
                // m_splitGcCost += gcEndTime;
                if (index < m_postingSizeLimit)
                {
                    postingList.clear();
                    char* ptr = (char*)(postingList.c_str());
                    for (int j = 0; j < index; j++)
                    {
                        Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                        ptr += m_vectorInfoSize;
                    }
                    m_postingSizes.UpdateSize(headID, index);
                    if (db->Put(headID, postingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Split Fail to write back postings\n");
                        exit(0);
                    }
                    m_garbageNum++;
                    auto GCEnd = std::chrono::high_resolution_clock::now();
                    double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(GCEnd - splitBegin).count();
                    m_garbageCost += elapsedMSeconds;
                    return ErrorCode::Success;
                }
                //LOG(Helper::LogLevel::LL_Info, "Resize\n");
                localIndices.resize(realVectorNum);

                auto clusterBegin = std::chrono::high_resolution_clock::now();
                // k = 2, maybe we can change the split number, now it is fixed
                SPTAG::COMMON::KmeansArgs<ValueType> args(2, smallSample.C(), (SizeType)localIndices.size(), 1, p_index->GetDistCalcMethod());
                std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));

                int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr, m_opt->m_virtualHead);

                auto clusterEnd = std::chrono::high_resolution_clock::now();
                double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(clusterEnd - clusterBegin).count();
                m_clusteringCost += elapsedMSeconds;
                // int numClusters = ClusteringSPFresh(smallSample, localIndices, 0, localIndices.size(), args, 10, false, p_opt.m_virtualHead);
                // exit(0);
                if (numClusters <= 1)
                {
                    LOG(Helper::LogLevel::LL_Info, "Cluserting Failed (The same vector), Cut to limit\n");
                    postingList.clear();
                    char* ptr = (char*)(postingList.c_str());
                    for (int j = 0; j < m_postingSizeLimit; j++)
                    {
                        Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                        ptr += m_vectorInfoSize;
                    }
                    m_postingSizes.UpdateSize(headID, m_postingSizeLimit);
                    if (db->Put(headID, postingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Split fail to override postings cut to limit\n");
                        exit(0);
                    }
                    return ErrorCode::Success;
                }

                long long newHeadVID = -1;
                int first = 0;
                std::vector<SizeType> newHeadsID;
                std::vector<std::string> newPostingLists;
                bool theSameHead = false;
                for (int k = 0; k < 2; k++) {
                    if (args.counts[k] == 0)	continue;
                    postingList.clear();
                    char* ptr = (char*)(postingList.c_str());
                    for (int j = 0; j < args.counts[k]; j++)
                    {
                        Serialize(ptr, localIndicesInsert[localIndices[first + j]], localIndicesInsertVersion[localIndices[first + j]], smallSample[localIndices[first + j]]);
                    }
                    if (!theSameHead && p_index->ComputeDistance(args.centers + k * args._D, p_index->GetSample(headID)) < Epsilon) {
                        newHeadsID.push_back(headID);
                        newHeadVID = headID;
                        theSameHead = true;
                        if (db->Put(newHeadVID, postingList) != ErrorCode::Success) {
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
                        if (db->Put(newHeadVID, postingList) != ErrorCode::Success) {
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
                    newPostingLists.push_back(postingList);
                    // LOG(Helper::LogLevel::LL_Info, "Head id: %d split into : %d, length: %d\n", headID, newHeadVID, args.counts[k]);
                    first += args.counts[k];
                    m_postingSizes.UpdateSize(newHeadVID, args.counts[k]);
                }
                if (!theSameHead) {
                    p_index->DeleteIndex(headID);
                    m_postingSizes.UpdateSize(headID, 0);
                }
            }
            int split_order = ++m_stat.m_splitNum;
            // if (theSameHead) LOG(Helper::LogLevel::LL_Info, "The Same Head\n");
            // LOG(Helper::LogLevel::LL_Info, "head1:%d, head2:%d\n", newHeadsID[0], newHeadsID[1]);

            // QuantifySplit(headID, newPostingLists, newHeadsID, headID, split_order);
            // QuantifyAssumptionBrokenTotally();
            auto reassignScanBegin = std::chrono::high_resolution_clock::now();

            if (!p_opt.m_disableReassign) CollectReAssign(p_index, headID, newPostingLists, newHeadsID);

            auto reassignScanEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanEnd - reassignScanBegin).count();

            m_stat.m_reassignScanCost += elapsedMSeconds;

            // LOG(Helper::LogLevel::LL_Info, "After ReAssign\n");

            // QuantifySplit(headID, newPostingLists, newHeadsID, headID, split_order);
            auto splitEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - splitBegin).count();
            m_stat.m_splitCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        inline void SplitAsync(VectorIndex* p_index, SizeType headID, std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new SplitAsyncJob(p_index, this, headID, p_callback);
            m_splitThreadPool->add(curJob);
        }

        inline void ReassignAsync(VectorIndex* p_index, std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version, std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new ReassignAsyncJob(p_index, this, std::move(vectorContain), VID, HeadPrev, version, p_callback);
            m_splitThreadPool->add(curJob);
        }

        //TODO
        ErrorCode CollectReAssign(VectorIndex* p_index, SizeType headID, std::vector<std::string>& postingLists, std::vector<SizeType>& newHeadsID) {
            //            TimeUtils::StopW sw;
            auto headVector = reinterpret_cast<const ValueType*>(p_index->GetSample(headID));
            std::vector<SizeType> HeadPrevTopK;
            std::vector<float> HeadPrevToSplitHeadDist;
            if (p_opt.m_reassignK > 0) {
                COMMON::QueryResultSet<ValueType> nearbyHeads(NULL, p_opt.m_reassignK);
                nearbyHeads.SetTarget(headVector);
                nearbyHeads.Reset();
                p_index->SearchIndex(nearbyHeads);
                BasicResult* queryResults = nearbyHeads.GetResults();
                for (int i = 0; i < nearbyHeads.GetResultNum(); i++) {
                    std::string tempPostingList;
                    auto vid = queryResults[i].VID;
                    if (vid == -1) {
                        break;
                    }
                    if (find(newHeadsID.begin(), newHeadsID.end(), vid) == newHeadsID.end()) {
                        // db->Get(vid, tempPostingList);
                        // postingLists.push_back(tempPostingList);
                        HeadPrevTopK.push_back(vid);
                        HeadPrevToSplitHeadDist.push_back(queryResults[i].Dist);
                    }
                }
                auto reassignScanIOBegin = std::chrono::high_resolution_clock::now();
                std::vector<std::string> tempPostingLists;
                if (db->MultiGet(HeadPrevTopK, &tempPostingLists) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "ReAssign can't get all the near postings\n");
                    exit(0);
                }
                auto reassignScanIOEnd = std::chrono::high_resolution_clock::now();
                auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanIOEnd - reassignScanIOBegin).count();

                m_reassignScanIOCost += elapsedMSeconds;
                for (int i = 0; i < HeadPrevTopK.size(); i++) {
                    postingLists.push_back(tempPostingLists[i]);
                }
            }

            int vectorInfoSize = p_opt.m_dim * sizeof(ValueType) + m_metaDataSize;
            std::map<SizeType, ValueType*> reAssignVectorsTop0;
            std::map<SizeType, SizeType> reAssignVectorsHeadPrevTop0;
            std::map<SizeType, uint8_t> versionsTop0;
            std::map<SizeType, ValueType*> reAssignVectorsTopK;
            std::map<SizeType, SizeType> reAssignVectorsHeadPrevTopK;
            std::map<SizeType, uint8_t> versionsTopK;

            std::vector<float_t> newHeadDist;

            newHeadDist.push_back(p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeadsID[0])));
            newHeadDist.push_back(p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeadsID[1])));

            for (int i = 0; i < postingLists.size(); i++) {
                auto& postingList = postingLists[i];
                size_t postVectorNum = postingList.size() / vectorInfoSize;
                auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    // float dist = *(reinterpret_cast<float*>(vectorId + sizeof(int) + sizeof(uint8_t)));
                    float dist;
                    if (i <= 1) {
                        if (!m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                            m_reAssignScanNum++;
                            dist = p_index->ComputeDistance(p_index->GetSample(newHeadsID[i]), reinterpret_cast<ValueType*>(vectorId + m_metaDataSize));
                            if (CheckIsNeedReassign(newHeadsID, reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), headID, newHeadDist[i], dist, true, newHeadsID[i])) {
                                reAssignVectorsTop0[vid] = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                reAssignVectorsHeadPrevTop0[vid] = newHeadsID[i];
                                versionsTop0[vid] = version;
                            }
                        }
                    }
                    else {
                        if ((reAssignVectorsTop0.find(vid) == reAssignVectorsTop0.end()))
                        {
                            if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                                m_reAssignScanNum++;
                                dist = p_index->ComputeDistance(p_index->GetSample(HeadPrevTopK[i - 2]), reinterpret_cast<ValueType*>(vectorId + m_metaDataSize));
                                if (CheckIsNeedReassign(newHeadsID, reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), headID, HeadPrevToSplitHeadDist[i - 2], dist, false, HeadPrevTopK[i - 2])) {
                                    reAssignVectorsTopK[vid] = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                    reAssignVectorsHeadPrevTopK[vid] = HeadPrevTopK[i - 2];
                                    versionsTopK[vid] = version;
                                }
                            }
                        }
                    }
                }
            }
            // LOG(Helper::LogLevel::LL_Info, "Scan: %d\n", m_reAssignScanNum.load());
            // exit(0);


            ReAssignVectors(reAssignVectorsTop0, reAssignVectorsHeadPrevTop0, versionsTop0);
            ReAssignVectors(reAssignVectorsTopK, reAssignVectorsHeadPrevTopK, versionsTopK);
            return ErrorCode::Success;
        }

        void ReAssignVectors(std::map<SizeType, ValueType*>& reAssignVectors,
            std::map<SizeType, SizeType>& HeadPrevs, std::map<SizeType, uint8_t>& versions)
        {
            for (auto it = reAssignVectors.begin(); it != reAssignVectors.end(); ++it) {

                auto vectorContain = std::make_shared<std::string>(Helper::Convert::Serialize<uint8_t>(it->second, p_opt.m_dim));

                ReassignAsync(vectorContain, it->first, HeadPrevs[it->first], versions[it->first]);
            }
        }

        bool RNGSelection(std::vector<Edge>& selections, ValueType* queryVector, std::shared_ptr<VectorIndex>& p_index, SizeType p_fullID, int& replicaCount, int checkHeadID = -1)
        {
            QueryResult queryResults(queryVector, m_opt->m_internalResultNum, false);
            p_index->SearchIndex(queryResults);

            replicaCount = 0;
            for (int i = 0; i < queryResults.GetResultNum() && replicaCount < m_opt->m_replicaCount; ++i)
            {
                if (queryResults[i].VID == -1) {
                    break;
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j)
                {
                    float nnDist = p_index->ComputeDistance(p_index->GetSample(queryResults[i].VID),
                        p_index->GetSample(selections[j].node));
                    if (m_opt->m_rngFactor * nnDist < queryResults[i].Dist)
                    {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted) continue;
                selections[replicaCount].node = queryResults[i].VID;
                selections[replicaCount].tonode = p_fullID;
                selections[replicaCount].distance = queryResults[i].Dist;
                if (selections[replicaCount].node == checkHeadID) {
                    return false;
                }
                ++replicaCount;
            }
            return true;
        }

        template <typename ValueType>
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
                    uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(int)]);
                    auto vectorContain = std::make_shared<std::string>(appendPosting.substr(idx + m_metaDataSize, m_vectorInfoSize - m_metaDataSize));
                    if (m_versionMap->GetVersion(*(int*)(&appendPosting[idx])) == version) {
                        // LOG(Helper::LogLevel::LL_Info, "Head Miss To ReAssign: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);
                        m_headMiss++;
                        ReassignAsync(vectorContain, *(int*)(&appendPosting[idx]), headID, version);
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
                std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);
                if (!p_index->ContainSample(headID)) {
                    goto checkDeleted;
                }
                // for (int i = 0; i < appendNum; i++)
                // {
                //     uint32_t idx = i * vectorInfoSize;
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
                SplitAsync(headID);
            }
            auto appendEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
            if (!reassignThreshold) {
                m_appendTaskNum++;
                m_appendIOCost += appendIOSeconds;
                m_appendCost += elapsedMSeconds;
            }
            return ErrorCode::Success;
        }
        
        //TODO: discussion
        template <typename T>
        void Reassign(VectorIndex* p_index, std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version)
        {
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
            bool isNeedReassign = RNGSelection(selections, reinterpret_cast<ValueType*>(&vectorContain->front()), p_index, VID, replicaCount, HeadPrev);
            auto selectEnd = std::chrono::high_resolution_clock::now();
            auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(selectEnd - selectBegin).count();
            m_selectCost += elapsedMSeconds;

            if (isNeedReassign && m_versionMap->GetVersion(VID) == version) {
                // LOG(Helper::LogLevel::LL_Info, "Update Version: VID: %d, version: %d, current version: %d\n", VID, version, m_versionMap.GetVersion(VID));
                m_versionMap->IncVersion(VID, &version);
            }
            else {
                isNeedReassign = false;
            }

            //LOG(Helper::LogLevel::LL_Info, "Reassign: oldVID:%d, replicaCount:%d, candidateNum:%d, dist0:%f\n", oldVID, replicaCount, i, selections[0].distance);
            auto reassignAppendBegin = std::chrono::high_resolution_clock::now();
            for (int i = 0; isNeedReassign && i < replicaCount && m_versionMap->GetVersion(VID) == version; i++) {
                std::string newPart(m_vectorInfoSize, '\0');
                Serialize((char*)(newPart.c_str()), VID, version, vectorContain->c_str());
                //LOG(Helper::LogLevel::LL_Info, "Reassign: headID :%d, oldVID:%d, newVID:%d, posting length: %d, dist: %f, string size: %d\n", headID, oldVID, VID, m_postingSizes[headID].load(), selections[i].distance, newPart.size());
                if (ErrorCode::Undefined == Append(selections[i].node, 1, newPart, 3)) {
                    // LOG(Helper::LogLevel::LL_Info, "Head Miss: VID: %d, current version: %d, another re-assign\n", VID, version);
                    isNeedReassign = false;
                }
            }
            auto reassignAppendEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignAppendEnd - reassignAppendBegin).count();
            m_reAssignAppendCost += elapsedMSeconds;

            auto reassignEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignEnd - reassignBegin).count();
            m_reAssignCost += elapsedMSeconds;
            //     m_reassignMap.erase(VID);
        }

        bool LoadIndex(Options& p_opt, COMMON::VersionLabel& p_versionMap) override {
            LOG(Helper::LogLevel::LL_Info, "DataBlockSize: %d, Capacity: %d\n", p_opt.m_datasetRowsInBlock, p_opt.m_datasetCapacity);
            m_versionMap = &p_versionMap;
            m_opt = &p_opt;
            m_versionMap->Load(p_opt.m_deleteIDFile, p_opt.m_datasetRowsInBlock, p_opt.m_datasetCapacity);
            m_postingSizes.Load(p_opt.m_ssdInfoFile, p_opt.m_datasetRowsInBlock, p_opt.m_datasetCapacity);

            LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", m_versionMap->GetVectorNum());
            LOG(Helper::LogLevel::LL_Info, "Current posting num: %d.\n", m_postingSizes.GetPostingNum());

            if (p_opt.m_update) {
                //LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize persistent buffer\n");
                //m_persistentBuffer = std::make_shared<PersistentBuffer>(p_opt.m_persistentBufferPath, db);
                LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
                LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", p_opt.m_appendThreadNum, p_opt.m_reassignThreadNum);
                m_splitThreadPool = std::make_shared<Helper::ThreadPool>();
                m_splitThreadPool->init(p_opt.m_appendThreadNum);
                m_reassignThreadPool = std::make_shared<Helper::ThreadPool>();
                m_reassignThreadPool->init(p_opt.m_reassignThreadNum);

                // LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize dispatcher\n");
                // m_dispatcher = std::make_shared<Dispatcher>(m_persistentBuffer, p_opt.m_batch, m_splitThreadPool, m_reassignThreadPool, this);
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

                int vectorNum = postingList.size() / m_vectorInfoSize;

                diskRead += postingList.size();
                listElements += vectorNum;

                auto compStart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < vectorNum; i++) {
                    char* vectorInfo = postingList.data() + i * m_vectorInfoSize;
                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                    if (m_versionMap->Deleted(vectorID) || p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) {
                        listElements--;
                        continue;
                    }
                    auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    queryResults.AddPoint(vectorID, distance2leaf);
                }
                auto compEnd = std::chrono::high_resolution_clock::now();

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

            int numThreads = p_opt.m_iSSDNumberOfThreads;
            int candidateNum = p_opt.m_internalResultNum;

            SizeType fullCount = 0;
            size_t vectorInfoSize = 0;
            {
                auto fullVectors = p_reader->GetVectorSet();
                fullCount = fullVectors->Count();
                // vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);
                vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int) + sizeof(uint8_t);
            }
            if (upperBound > 0) fullCount = upperBound;

            // m_metaDataSize = sizeof(int) + sizeof(uint8_t) + sizeof(float);
            m_metaDataSize = sizeof(int) + sizeof(uint8_t);

            m_versionMap = &p_versionMap;
            m_opt = &p_opt;
            LOG(Helper::LogLevel::LL_Info, "Build SSD Index.\n");

            Selection selections(static_cast<size_t>(fullCount) * p_opt.m_replicaCount, p_opt.m_tmpdir);
            LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
            std::vector<std::atomic_int> replicaCount(fullCount);
            std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
            for (auto& pls : postingListSize) pls = 0;
            std::unordered_set<SizeType> emptySet;
            SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

            auto t1 = std::chrono::high_resolution_clock::now();
            if (p_opt.m_batches > 1) selections.SaveBatch();
            {
                LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                SizeType sampleSize = p_opt.m_samples;
                std::vector<SizeType> samples(sampleSize, 0);
                for (int i = 0; i < p_opt.m_batches; i++) {
                    SizeType start = i * batchSize;
                    SizeType end = min(start + batchSize, fullCount);
                    auto fullVectors = p_reader->GetVectorSet(start, end);
                    if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                    emptySet.clear();

                    p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);

                    for (SizeType j = start; j < end; j++) {
                        replicaCount[j] = 0;
                        size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                        for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                            ++postingListSize[selections[vecOffset + resNum].node];
                            selections[vecOffset + resNum].tonode = j;
                            ++replicaCount[j];
                        }
                    }

                    if (p_opt.m_batches > 1) selections.SaveBatch();
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

            if (p_opt.m_batches > 1) selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount);

            // Sort results either in CPU or GPU
            VectorIndex::SortSelections(&selections.m_selections);

            auto t3 = std::chrono::high_resolution_clock::now();
            LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

            auto postingSizeLimit = m_postingSizeLimit;

            if (p_opt.m_postingPageLimit > 0)
            {
                postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
            }

            LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);


            {
                std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
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
                if (postingListSize[i] <= postingSizeLimit) continue;

                std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, Selection::g_edgeComparer) - selections.m_selections.begin();

                for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                {
                    int tonode = selections.m_selections[selectIdx + dropID].tonode;
                    --replicaCount[tonode];
                }
                postingListSize[i] = postingSizeLimit;
            }

            {
                std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
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

            if (p_opt.m_outputEmptyReplicaID)
            {
                std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                    return false;
                }
                for (int i = 0; i < replicaCount.size(); ++i)
                {

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

            if (p_opt.m_ssdIndexFileNum > 1) selections.SaveBatch();

            auto fullVectors = p_reader->GetVectorSet();
            if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

            LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize versionMap\n");
            m_versionMap->Initialize(fullCount, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);

            LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing values to DB\n");

            std::vector<int> postingListSize_int(postingListSize.begin(), postingListSize.end());

            WriteDownAllPostingToDB(postingListSize_int, selections, fullVectors);

            m_postingSizes.Initialize(postingListSize.size(), p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);
            for (int i = 0; i < postingListSize.size(); i++) {
                m_postingSizes.UpdateSize(i, postingListSize[i]);
            }
            LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing SSD Info\n");
            m_postingSizes.Save(p_opt.m_ssdInfoFile);
            LOG(Helper::LogLevel::LL_Info, "SPFresh: save versionMap\n");
            m_versionMap->Save(p_opt.m_deleteIDFile);

            auto t5 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
            LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        void WriteDownAllPostingToDB(const std::vector<int>& p_postingListSizes, Selection& p_postingSelections, std::shared_ptr<VectorSet> p_fullVectors) {
            size_t dim = p_fullVectors->Dimension();
    #pragma omp parallel for num_threads(10)
            for (int id = 0; id < p_postingListSizes.size(); id++)
            {
                std::string postinglist(m_vectorInfoSize * p_postingListSize[id], '\0');
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
                std::vector<Edge> selections(static_cast<size_t>(p_opt.m_replicaCount));
                int replicaCount;
                RNGSelection(selections, p_vectorSet->GetVector(v), p_index, VID, replicaCount);

                uint8_t version = m_versionMap->GetVersion(VID);
                std::string appendPosting(m_vectorInfoSize, '\0');
                Serialize((char*)appendPosting.c_str(), VID, version, queryResults.GetTarget());
                for (int i = 0; i < replicaCount; i++)
                {
                    // AppendAsync(selections[i].node, 1, appendPosting_ptr);
                    Append(selections[i].node, 1, appendPosting);
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode DeleteIndex(SizeType p_id) {
            if (m_versionMap->Delete(p_id)) return ErrorCode::Success;
            /*
            LOG(Helper::LogLevel::LL_Info, "Delete not support\n");
            exit(0);
            char deleteCode = 1;
            int VID = p_id;
            std::string assignment;
            assignment += Helper::Convert::Serialize<char>(&deleteCode, 1);
            assignment += Helper::Convert::Serialize<int>(&VID, 1);
            m_assignmentQueue.push(m_persistentBuffer->PutAssignment(assignment));
            */
            return ErrorCode::VectorNotFound;
        }

        bool AllFinished() { return m_splitThreadPool->allClear() && m_reassignThreadPool->allClear(); }
        void ForceCompaction() override { db->ForceCompaction(); }
        void GetDBStats() override { 
            db->GetDBStat();
            LOG(Helper::LogLevel::LL_Info, "remain splitJobs: %d, reassignJobs: %d\n", m_splitThreadPool->jobsize(), m_reassignThreadPool->jobsize());
        }

        void GetIndexStats(int finishedInsert, bool cost, bool reset) override { m_stat.PrintStat(finishedInsert, cost, reset); }

    private:

        int m_metaDataSize = 0;
        
        int m_vectorInfoSize = 0;

        int m_postingSizeLimit = INT_MAX;

        float m_hardLatencyLimit = 2;
    };
} // namespace SPTAG
#endif // _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_