// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_BKTREE_H_
#define _SPTAG_COMMON_BKTREE_H_

#include <stack>
#include <string>
#include <vector>
#include <shared_mutex>

#include "../VectorIndex.h"

#include "CommonUtils.h"
#include "QueryResultSet.h"
#include "WorkSpace.h"
#include "Dataset.h"
#include "DistanceUtils.h"

namespace SPTAG
{
    namespace COMMON
    {
        // node type for storing BKT
        struct BKTNode
        {
            SizeType centerid;
            SizeType childStart;
            SizeType childEnd;

            BKTNode(SizeType cid = -1) : centerid(cid), childStart(-1), childEnd(-1) {}
        };

        template <typename T>
        struct KmeansArgs {
            int _K;
            int _DK;
            DimensionType _D;
            int _T;
            DistCalcMethod _M;
            T* centers;
            T* newTCenters;
            SizeType* counts;
            float* newCenters;
            SizeType* newCounts;
            int* label;
            SizeType* clusterIdx;
            float* clusterDist;
            float* weightedCounts;
            float* newWeightedCounts;
            float(*fComputeDistance)(const T* pX, const T* pY, DimensionType length);

            KmeansArgs(int k, DimensionType dim, SizeType datasize, int threadnum, DistCalcMethod distMethod) : _K(k), _DK(k), _D(dim), _T(threadnum), _M(distMethod) {
                centers = (T*)_mm_malloc(sizeof(T) * k * dim, ALIGN_SPTAG);
                newTCenters = (T*)_mm_malloc(sizeof(T) * k * dim, ALIGN_SPTAG);
                counts = new SizeType[k];
                newCenters = new float[threadnum * k * dim];
                newCounts = new SizeType[threadnum * k];
                label = new int[datasize];
                clusterIdx = new SizeType[threadnum * k];
                clusterDist = new float[threadnum * k];
                weightedCounts = new float[k];
                newWeightedCounts = new float[threadnum * k];
                fComputeDistance = COMMON::DistanceCalcSelector<T>(distMethod);
            }

            ~KmeansArgs() {
                _mm_free(centers);
                _mm_free(newTCenters);
                delete[] counts;
                delete[] newCenters;
                delete[] newCounts;
                delete[] label;
                delete[] clusterIdx;
                delete[] clusterDist;
                delete[] weightedCounts;
                delete[] newWeightedCounts;
            }

            inline void ClearCounts() {
                memset(newCounts, 0, sizeof(SizeType) * _T * _K);
                memset(newWeightedCounts, 0, sizeof(float) * _T * _K);
            }

            inline void ClearCenters() {
                memset(newCenters, 0, sizeof(float) * _T * _K * _D);
            }

            inline void ClearDists(float dist) {
                for (int i = 0; i < _T * _K; i++) {
                    clusterIdx[i] = -1;
                    clusterDist[i] = dist;
                }
            }

            void Shuffle(std::vector<SizeType>& indices, SizeType first, SizeType last) {
                SizeType* pos = new SizeType[_K];
                pos[0] = first;
                for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

                for (int k = 0; k < _K; k++) {
                    if (newCounts[k] == 0) continue;
                    SizeType i = pos[k];
                    while (newCounts[k] > 0) {
                        SizeType swapid = pos[label[i]] + newCounts[label[i]] - 1;
                        newCounts[label[i]]--;
                        std::swap(indices[i], indices[swapid]);
                        std::swap(label[i], label[swapid]);
                    }
                    while (indices[i] != clusterIdx[k]) i++;
                    std::swap(indices[i], indices[pos[k] + counts[k] - 1]);
                }
                delete[] pos;
            }
        };

        template <typename T>
        float RefineCenters(const Dataset<T>& data, KmeansArgs<T>& args)
        {
            int maxcluster = -1;
            SizeType maxCount = 0;
            for (int k = 0; k < args._DK; k++) {
                if (args.counts[k] > maxCount && args.newCounts[k] > 0 && DistanceUtils::ComputeDistance((T*)data[args.clusterIdx[k]], args.centers + k * args._D, args._D, DistCalcMethod::L2) > 1e-6)
                {
                    maxcluster = k;
                    maxCount = args.counts[k];
                }
            }

            if (maxcluster != -1 && (args.clusterIdx[maxcluster] < 0 || args.clusterIdx[maxcluster] >= data.R()))
                LOG(Helper::LogLevel::LL_Debug, "maxcluster:%d(%d) Error dist:%f\n", maxcluster, args.newCounts[maxcluster], args.clusterDist[maxcluster]);

            float diff = 0;
            for (int k = 0; k < args._DK; k++) {
                T* TCenter = args.newTCenters + k * args._D;
                if (args.counts[k] == 0) {
                    if (maxcluster != -1) {
                        //int nextid = Utils::rand_int(last, first);
                        //while (args.label[nextid] != maxcluster) nextid = Utils::rand_int(last, first);
                        SizeType nextid = args.clusterIdx[maxcluster];
                        std::memcpy(TCenter, data[nextid], sizeof(T)*args._D);
                    }
                    else {
                        std::memcpy(TCenter, args.centers + k * args._D, sizeof(T)*args._D);
                    }
                }
                else {
                    float* currCenters = args.newCenters + k * args._D;
                    for (DimensionType j = 0; j < args._D; j++) currCenters[j] /= args.counts[k];

                    if (args._M == DistCalcMethod::Cosine) {
                        COMMON::Utils::Normalize(currCenters, args._D, COMMON::Utils::GetBase<T>());
                    }
                    for (DimensionType j = 0; j < args._D; j++) TCenter[j] = (T)(currCenters[j]);
                }
                diff += args.fComputeDistance(args.centers + k*args._D, TCenter, args._D);
            }
            return diff;
        }

        template <typename T>
        inline float KmeansAssign(const Dataset<T>& data,
            std::vector<SizeType>& indices,
            const SizeType first, const SizeType last, KmeansArgs<T>& args, 
            const bool updateCenters, float lambda) {
            float currDist = 0;
            SizeType subsize = (last - first - 1) / args._T + 1;

#pragma omp parallel for num_threads(args._T) shared(data, indices) reduction(+:currDist)
            for (int tid = 0; tid < args._T; tid++)
            {
                SizeType istart = first + tid * subsize;
                SizeType iend = min(first + (tid + 1) * subsize, last);
                SizeType *inewCounts = args.newCounts + tid * args._K;
                float *inewCenters = args.newCenters + tid * args._K * args._D;
                SizeType * iclusterIdx = args.clusterIdx + tid * args._K;
                float * iclusterDist = args.clusterDist + tid * args._K;
                float * iweightedCounts = args.newWeightedCounts + tid * args._K;
                float idist = 0;
                for (SizeType i = istart; i < iend; i++) {
                    int clusterid = 0;
                    float smallestDist = MaxDist;
                    for (int k = 0; k < args._DK; k++) {
                        float dist = args.fComputeDistance(data[indices[i]], args.centers + k*args._D, args._D) + lambda*args.counts[k];
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k; smallestDist = dist;
                        }
                    }
                    args.label[i] = clusterid;
                    inewCounts[clusterid]++;
                    iweightedCounts[clusterid] += smallestDist;
                    idist += smallestDist;
                    if (updateCenters) {
                        const T* v = (const T*)data[indices[i]];
                        float* center = inewCenters + clusterid*args._D;
                        for (DimensionType j = 0; j < args._D; j++) center[j] += v[j];
                        if (smallestDist > iclusterDist[clusterid]) {
                            iclusterDist[clusterid] = smallestDist;
                            iclusterIdx[clusterid] = indices[i];
                        }
                    }
                    else {
                        if (smallestDist <= iclusterDist[clusterid]) {
                            iclusterDist[clusterid] = smallestDist;
                            iclusterIdx[clusterid] = indices[i];
                        }
                    }
                }
                currDist += idist;
            }

            for (int i = 1; i < args._T; i++) {
                for (int k = 0; k < args._DK; k++) {
                    args.newCounts[k] += args.newCounts[i * args._K + k];
                    args.newWeightedCounts[k] += args.newWeightedCounts[i * args._K + k];
                }
            }

            if (updateCenters) {
                for (int i = 1; i < args._T; i++) {
                    float* currCenter = args.newCenters + i*args._K*args._D;
                    for (size_t j = 0; j < ((size_t)args._DK) * args._D; j++) args.newCenters[j] += currCenter[j];

                    for (int k = 0; k < args._DK; k++) {
                        if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] > args.clusterDist[k]) {
                            args.clusterDist[k] = args.clusterDist[i*args._K + k];
                            args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                        }
                    }
                }
            }
            else {
                for (int i = 1; i < args._T; i++) {
                    for (int k = 0; k < args._DK; k++) {
                        if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] <= args.clusterDist[k]) {
                            args.clusterDist[k] = args.clusterDist[i*args._K + k];
                            args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                        }
                    }
                }
            }
            return currDist;
        }

        template <typename T>
        inline float InitCenters(const Dataset<T>& data, 
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            KmeansArgs<T>& args, int samples, int tryIters) {
            SizeType batchEnd = min(first + samples, last);
            float lambda, currDist, minClusterDist = MaxDist;
            for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
                for (int k = 0; k < args._DK; k++) {
                    SizeType randid = COMMON::Utils::rand(last, first);
                    std::memcpy(args.centers + k*args._D, data[indices[randid]], sizeof(T)*args._D);
                }
                args.ClearCounts();
                args.ClearDists(MaxDist);
                currDist = KmeansAssign(data, indices, first, batchEnd, args, false, 0);
                if (currDist < minClusterDist) {
                    minClusterDist = currDist;
                    memcpy(args.newTCenters, args.centers, sizeof(T)*args._K*args._D);
                    memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                    SizeType maxCluster = 0;
                    for (int k = 1; k < args._DK; k++) if (args.counts[k] > args.counts[maxCluster]) maxCluster = k;

                    float avgDist = args.newWeightedCounts[maxCluster] / args.counts[maxCluster];
                    lambda = (avgDist - args.clusterDist[maxCluster]) / args.counts[maxCluster];
                    if (lambda < 0) lambda = 0;
                }
            }
            return lambda;
        }

        template <typename T>
        float TryClustering(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last,
            KmeansArgs<T>& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false, IAbortOperation* abort = nullptr) {

            float adjustedLambda = InitCenters(data, indices, first, last, args, samples, 3);
            if (abort && abort->ShouldAbort()) return 0;

            SizeType batchEnd = min(first + samples, last);
            float currDiff, currDist, minClusterDist = MaxDist;
            int noImprovement = 0;
            float originalLambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / lambdaFactor / (batchEnd - first);
            for (int iter = 0; iter < 100; iter++) {
                std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
                std::random_shuffle(indices.begin() + first, indices.begin() + last);

                args.ClearCenters();
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                currDist = KmeansAssign(data, indices, first, batchEnd, args, true, min(adjustedLambda, originalLambda));
                std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                if (currDist < minClusterDist) {
                    noImprovement = 0;
                    minClusterDist = currDist;
                }
                else {
                    noImprovement++;
                }
                currDiff = RefineCenters(data, args);
                //if (debug) LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iter, currDist, currDiff);

                if (abort && abort->ShouldAbort()) return 0;
                if (currDiff < 1e-3 || noImprovement >= 5) break;
            }

            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign(data, indices, first, last, args, false, 0);
            for (int k = 0; k < args._DK; k++) {
                if (args.clusterIdx[k] != -1) std::memcpy(args.centers + k * args._D, data[args.clusterIdx[k]], sizeof(T) * args._D);
            }
            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign(data, indices, first, last, args, false, 0);
            std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

            SizeType maxCount = 0, minCount = (std::numeric_limits<SizeType>::max)(), availableClusters = 0;
            float CountStd = 0.0, CountAvg = (last - first) * 1.0f / args._DK;
            for (int i = 0; i < args._DK; i++) {
                if (args.counts[i] > maxCount) maxCount = args.counts[i];
                if (args.counts[i] < minCount) minCount = args.counts[i];
                CountStd += (args.counts[i] - CountAvg) * (args.counts[i] - CountAvg);
                if (args.counts[i] > 0) availableClusters++;
            }
            CountStd = sqrt(CountStd / args._DK) / CountAvg;
            if (debug) LOG(Helper::LogLevel::LL_Info, "Lambda:min(%g,%g) Max:%d Min:%d Avg:%f Std/Avg:%f Dist:%f NonZero/Total:%d/%d\n", originalLambda, adjustedLambda, maxCount, minCount, CountAvg, CountStd, currDist, availableClusters, args._DK);

            return CountStd;
        }

        template <typename T>
        float DynamicFactorSelect(const Dataset<T> & data,
            std::vector<SizeType> & indices, const SizeType first, const SizeType last,
            KmeansArgs<T> & args, int samples = 1000) {

            float bestLambdaFactor = 100.0f, bestCountStd = (std::numeric_limits<float>::max)();
            for (float lambdaFactor = 0.001f; lambdaFactor <= 1000.0f + 1e-3; lambdaFactor *= 10) {
                float CountStd = TryClustering(data, indices, first, last, args, samples, lambdaFactor, true);
                if (CountStd < bestCountStd) {
                    bestLambdaFactor = lambdaFactor;
                    bestCountStd = CountStd;
                }
            }
            /*
            std::vector<float> tries(16, 0);
            for (int i = 0; i < 8; i++) {
                tries[i] = bestLambdaFactor * (i + 2) / 10;
                tries[8 + i] = bestLambdaFactor * (i + 2);
            }
            for (float lambdaFactor : tries) {
                float CountStd = TryClustering(data, indices, first, last, args, samples, lambdaFactor, true);
                if (CountStd < bestCountStd) {
                    bestLambdaFactor = lambdaFactor;
                    bestCountStd = CountStd;
                }
            }
            */
            LOG(Helper::LogLevel::LL_Info, "Best Lambda Factor:%f\n", bestLambdaFactor);
            return bestLambdaFactor;
        }

        template <typename T>
        int KmeansClustering(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            KmeansArgs<T>& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false, IAbortOperation* abort = nullptr) {
            
            TryClustering(data, indices, first, last, args, samples, lambdaFactor, debug, abort);
            if (abort && abort->ShouldAbort()) return 1;

            int numClusters = 0;
            for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;

            if (numClusters <= 1) return numClusters;

            args.Shuffle(indices, first, last);
            return numClusters;
        }

        class BKTree
        {
        public:
            BKTree(): m_iTreeNumber(1), m_iBKTKmeansK(32), m_iBKTLeafSize(8), m_iSamples(1000), m_fBalanceFactor(-1.0f), m_bfs(0), m_lock(new std::shared_timed_mutex) {}
            
            BKTree(const BKTree& other): m_iTreeNumber(other.m_iTreeNumber), 
                                   m_iBKTKmeansK(other.m_iBKTKmeansK), 
                                   m_iBKTLeafSize(other.m_iBKTLeafSize),
                                   m_iSamples(other.m_iSamples),
                                   m_fBalanceFactor(other.m_fBalanceFactor),
                                   m_lock(new std::shared_timed_mutex) {}
            ~BKTree() {}

            inline const BKTNode& operator[](SizeType index) const { return m_pTreeRoots[index]; }
            inline BKTNode& operator[](SizeType index) { return m_pTreeRoots[index]; }

            inline SizeType size() const { return (SizeType)m_pTreeRoots.size(); }
            
            inline SizeType sizePerTree() const {
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                return (SizeType)m_pTreeRoots.size() - m_pTreeStart.back(); 
            }

            inline const std::unordered_map<SizeType, SizeType>& GetSampleMap() const { return m_pSampleCenterMap; }

            template <typename T>
            void Rebuild(const Dataset<T>& data, DistCalcMethod distMethod, IAbortOperation* abort)
            {
                BKTree newTrees(*this);
                newTrees.BuildTrees<T>(data, distMethod, 1, nullptr, nullptr, false, abort);

                std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                m_pTreeRoots.swap(newTrees.m_pTreeRoots);
                m_pTreeStart.swap(newTrees.m_pTreeStart);
                m_pSampleCenterMap.swap(newTrees.m_pSampleCenterMap);
            }

            template <typename T>
            void BuildTrees(const Dataset<T>& data, DistCalcMethod distMethod, int numOfThreads, 
                std::vector<SizeType>* indices = nullptr, std::vector<SizeType>* reverseIndices = nullptr, 
                bool dynamicK = false, IAbortOperation* abort = nullptr)
            {
                struct  BKTStackItem {
                    SizeType index, first, last;
                    bool debug;
                    BKTStackItem(SizeType index_, SizeType first_, SizeType last_, bool debug_ = false) : index(index_), first(first_), last(last_), debug(debug_) {}
                };
                std::stack<BKTStackItem> ss;

                std::vector<SizeType> localindices;
                if (indices == nullptr) {
                    localindices.resize(data.R());
                    for (SizeType i = 0; i < localindices.size(); i++) localindices[i] = i;
                }
                else {
                    localindices.assign(indices->begin(), indices->end());
                }
                KmeansArgs<T> args(m_iBKTKmeansK, data.C(), (SizeType)localindices.size(), numOfThreads, distMethod);

                if (m_fBalanceFactor < 0) m_fBalanceFactor = DynamicFactorSelect(data, localindices, 0, (SizeType)localindices.size(), args, m_iSamples);

                m_pSampleCenterMap.clear();
                for (char i = 0; i < m_iTreeNumber; i++)
                {
                    std::random_shuffle(localindices.begin(), localindices.end());

                    m_pTreeStart.push_back((SizeType)m_pTreeRoots.size());
                    m_pTreeRoots.emplace_back((SizeType)localindices.size());
                    LOG(Helper::LogLevel::LL_Info, "Start to build BKTree %d\n", i + 1);

                    ss.push(BKTStackItem(m_pTreeStart[i], 0, (SizeType)localindices.size(), true));
                    while (!ss.empty()) {
                        if (abort && abort->ShouldAbort()) return;

                        BKTStackItem item = ss.top(); ss.pop();
                        SizeType newBKTid = (SizeType)m_pTreeRoots.size();
                        m_pTreeRoots[item.index].childStart = newBKTid;
                        if (item.last - item.first <= m_iBKTLeafSize) {
                            for (SizeType j = item.first; j < item.last; j++) {
                                SizeType cid = (reverseIndices == nullptr)? localindices[j]: reverseIndices->at(localindices[j]);
                                m_pTreeRoots.emplace_back(cid);
                            }
                        }
                        else { // clustering the data into BKTKmeansK clusters
                            if (dynamicK) {
                                args._DK = std::min<int>((item.last - item.first) / m_iBKTLeafSize + 1, m_iBKTKmeansK);
                                args._DK = std::max<int>(args._DK, 2);
                            }

                            int numClusters = KmeansClustering(data, localindices, item.first, item.last, args, m_iSamples, m_fBalanceFactor, item.debug, abort);
                            if (numClusters <= 1) {
                                SizeType end = min(item.last + 1, (SizeType)localindices.size());
                                std::sort(localindices.begin() + item.first, localindices.begin() + end);
                                m_pTreeRoots[item.index].centerid = (reverseIndices == nullptr) ? localindices[item.first] : reverseIndices->at(localindices[item.first]);
                                m_pTreeRoots[item.index].childStart = -m_pTreeRoots[item.index].childStart;
                                for (SizeType j = item.first + 1; j < end; j++) {
                                    SizeType cid = (reverseIndices == nullptr) ? localindices[j] : reverseIndices->at(localindices[j]);
                                    m_pTreeRoots.emplace_back(cid);
                                    m_pSampleCenterMap[cid] = m_pTreeRoots[item.index].centerid;
                                }
                                m_pSampleCenterMap[-1 - m_pTreeRoots[item.index].centerid] = item.index;
                            }
                            else {
                                SizeType maxCount = 0;
                                for (int k = 0; k < m_iBKTKmeansK; k++) if (args.counts[k] > maxCount) maxCount = args.counts[k];
                                for (int k = 0; k < m_iBKTKmeansK; k++) {
                                    if (args.counts[k] == 0) continue;
                                    SizeType cid = (reverseIndices == nullptr) ? localindices[item.first + args.counts[k] - 1] : reverseIndices->at(localindices[item.first + args.counts[k] - 1]);
                                    m_pTreeRoots.emplace_back(cid);
                                    if (args.counts[k] > 1) ss.push(BKTStackItem(newBKTid++, item.first, item.first + args.counts[k] - 1, item.debug && (args.counts[k] == maxCount)));
                                    item.first += args.counts[k];
                                }
                            }
                        }
                        m_pTreeRoots[item.index].childEnd = (SizeType)m_pTreeRoots.size();
                    }
                    m_pTreeRoots.emplace_back(-1);
                    LOG(Helper::LogLevel::LL_Info, "%d BKTree built, %zu %zu\n", i + 1, m_pTreeRoots.size() - m_pTreeStart[i], localindices.size());
                }
            }

            inline std::uint64_t BufferSize() const
            {
                return sizeof(int) + sizeof(SizeType) * m_iTreeNumber +
                    sizeof(SizeType) + sizeof(BKTNode) * m_pTreeRoots.size();
            }

            ErrorCode SaveTrees(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
            {
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                IOBINARY(p_out, WriteBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
                IOBINARY(p_out, WriteBinary, sizeof(SizeType) * m_iTreeNumber, (char*)m_pTreeStart.data());
                SizeType treeNodeSize = (SizeType)m_pTreeRoots.size();
                IOBINARY(p_out, WriteBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
                IOBINARY(p_out, WriteBinary, sizeof(BKTNode) * treeNodeSize, (char*)m_pTreeRoots.data());
                LOG(Helper::LogLevel::LL_Info, "Save BKT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode SaveTrees(std::string sTreeFileName) const
            {
                LOG(Helper::LogLevel::LL_Info, "Save BKT to %s\n", sTreeFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return SaveTrees(ptr);
            }

            ErrorCode LoadTrees(char* pBKTMemFile)
            {
                m_iTreeNumber = *((int*)pBKTMemFile);
                pBKTMemFile += sizeof(int);
                m_pTreeStart.resize(m_iTreeNumber);
                memcpy(m_pTreeStart.data(), pBKTMemFile, sizeof(SizeType) * m_iTreeNumber);
                pBKTMemFile += sizeof(SizeType)*m_iTreeNumber;

                SizeType treeNodeSize = *((SizeType*)pBKTMemFile);
                pBKTMemFile += sizeof(SizeType);
                m_pTreeRoots.resize(treeNodeSize);
                memcpy(m_pTreeRoots.data(), pBKTMemFile, sizeof(BKTNode) * treeNodeSize);
                if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
                LOG(Helper::LogLevel::LL_Info, "Load BKT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode LoadTrees(std::shared_ptr<Helper::DiskPriorityIO> p_input)
            {
                IOBINARY(p_input, ReadBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
                m_pTreeStart.resize(m_iTreeNumber);
                IOBINARY(p_input, ReadBinary, sizeof(SizeType) * m_iTreeNumber, (char*)m_pTreeStart.data());

                SizeType treeNodeSize;
                IOBINARY(p_input, ReadBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
                m_pTreeRoots.resize(treeNodeSize);
                IOBINARY(p_input, ReadBinary, sizeof(BKTNode) * treeNodeSize, (char*)m_pTreeRoots.data());

                if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
                LOG(Helper::LogLevel::LL_Info, "Load BKT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode LoadTrees(std::string sTreeFileName)
            {
                LOG(Helper::LogLevel::LL_Info, "Load BKT From %s\n", sTreeFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return LoadTrees(ptr);
            }

            template <typename T>
            void InitSearchTrees(const Dataset<T>& data, float(*fComputeDistance)(const T* pX, const T* pY, DimensionType length), const COMMON::QueryResultSet<T>& p_query, COMMON::WorkSpace& p_space) const
            {
                // int node_bfschecked = 0;
                bool EnableBFS = false;
                for (char i = 0; i < m_iTreeNumber; i++) {
                    const BKTNode& node = m_pTreeRoots[m_pTreeStart[i]];
                    if (node.childStart < 0) {
                        p_space.m_SPTQueue.insert(NodeDistPair(m_pTreeStart[i], fComputeDistance(p_query.GetTarget(), data[node.centerid], data.C())));
                    }
                    else if (m_bfs) {

                        //LOG(Helper::LogLevel::LL_Info, "BFS start!");

                        int BFS_level = 1;
                        std::vector<float> FactorQ;
                        FactorQ.push_back(1.1);
                        FactorQ.push_back(1.05);
                        p_space.m_BSPTQueue[0].Resize(40);
                        p_space.m_BSPTQueue[0].clear();
                        p_space.m_BSPTQueue[1].Resize(100);
                        p_space.m_BSPTQueue[1].clear();
                        //p_space.m_BSPTQueue[2].Resize(100);
                        //p_space.m_BSPTQueue[2].clear();
                        //p_space.m_BSPTQueue[3].Resize(100000);
                        //p_space.m_BSPTQueue[3].clear();
                        float now_min = 1e9;
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            //node_bfschecked++;
                            //std::cout << begin << std::endl;
                            SizeType index = m_pTreeRoots[begin].centerid;
                            float now_dis = fComputeDistance(p_query.GetTarget(), data[index], data.C());
                            if (now_dis < now_min * FactorQ[0]) {
                                if (now_dis < now_min) now_min = now_dis;
                                p_space.m_BSPTQueue[0].insert(NodeDistPair(begin, now_dis));
                                p_space.m_SPTQueue.insert(NodeDistPair(begin, now_dis));
                            }
                        }
                        //LOG(Helper::LogLevel::LL_Info, "Queue 0 size is %d\n", (int)p_space.m_BSPTQueue[0].size());

                        int BFS_size = 100;
                        for (int i = 1; i <= BFS_level; i++) {
                            int now_size = 0;
                            while (!p_space.m_BSPTQueue[i - 1].empty() && now_size < BFS_size) {
                                NodeDistPair tmp = p_space.m_BSPTQueue[i - 1].pop();
                                if (i > 1 && tmp.distance > now_min * FactorQ[i]) continue;
                                now_size++;
                                if (now_size > BFS_size) break;
                                const BKTNode& tnode = m_pTreeRoots[tmp.node];
                                for (SizeType now = tnode.childStart; now < tnode.childEnd; now++) {
                                    //node_bfschecked++;
                                    SizeType index = m_pTreeRoots[now].centerid;
                                    float now_dis = fComputeDistance(p_query.GetTarget(), data[index], data.C());
                                    if (now_dis < now_min * FactorQ[i]) {
                                        if (now_dis < now_min) now_min = now_dis;
                                        p_space.m_BSPTQueue[i].insert(NodeDistPair(now, now_dis));
                                        p_space.m_SPTQueue.insert(NodeDistPair(now, now_dis));
                                    }
                                }
                            }
                        }

                        //LOG(Helper::LogLevel::LL_Info, "Queue %d size is %d\n", BFS_level, (int)p_space.m_BSPTQueue[BFS_level].size());
                        //LOG(Helper::LogLevel::LL_Info, "Queue size is %d\n", (int)p_space.m_BSPTQueue[BFS_level].size());
                        int pivot_size = 20, now_pivot = 0;
                        while (!p_space.m_BSPTQueue[BFS_level].empty() && now_pivot < pivot_size) {
                            now_pivot++;
                            NodeDistPair tmp = p_space.m_BSPTQueue[BFS_level].pop();
                            SizeType index = m_pTreeRoots[tmp.node].centerid;
                            p_space.m_SPTQueue.insert(NodeDistPair(tmp.node, fComputeDistance(p_query.GetTarget(), data[index], data.C())));
                        }
                        /*

                        */
                    }
                    else {
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(NodeDistPair(begin, fComputeDistance(p_query.GetTarget(), data[index], data.C())));
                        }
                    }
                }
                //LOG(Helper::LogLevel::LL_Info, "%d nodes checked by BFS!", node_bfschecked);
            }

            template <typename T>
            void SearchTrees(const Dataset<T>& data, float(*fComputeDistance)(const T* pX, const T* pY, DimensionType length), const COMMON::QueryResultSet<T> &p_query,
                COMMON::WorkSpace &p_space, const int p_limits) const
            {
                while (!p_space.m_SPTQueue.empty())
                {
                    NodeDistPair bcell = p_space.m_SPTQueue.pop();
                    const BKTNode& tnode = m_pTreeRoots[bcell.node];
                    if (tnode.childStart < 0) {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_iNumberOfCheckedLeaves++;
                            p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, bcell.distance));
                        }
                        if (p_space.m_iNumberOfCheckedLeaves >= p_limits) break;
                    }
                    else {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, bcell.distance));
                        }
                        for (SizeType begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(NodeDistPair(begin, fComputeDistance(p_query.GetTarget(), data[index], data.C())));
                        } 
                    }
                }
            }

        private:
            std::vector<SizeType> m_pTreeStart;
            std::vector<BKTNode> m_pTreeRoots;
            std::unordered_map<SizeType, SizeType> m_pSampleCenterMap;

        public:
            std::unique_ptr<std::shared_timed_mutex> m_lock;
            int m_iTreeNumber, m_iBKTKmeansK, m_iBKTLeafSize, m_iSamples, m_bfs;
            float m_fBalanceFactor;
        };
    }
}
#endif
