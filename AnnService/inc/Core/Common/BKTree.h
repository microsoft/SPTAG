// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_BKTREE_H_
#define _SPTAG_COMMON_BKTREE_H_

#include <iostream>
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

#pragma warning(disable:4996)  // 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

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
                centers = (T*)aligned_malloc(sizeof(T) * k * dim, ALIGN);
                newTCenters = (T*)aligned_malloc(sizeof(T) * k * dim, ALIGN);
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
                aligned_free(centers);
                aligned_free(newTCenters);
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
                if (args.counts[k] > maxCount && args.newCounts[k] > 0 && DistanceUtils::ComputeL2Distance((T*)data[args.clusterIdx[k]], args.centers + k * args._D, args._D) > 1e-6)
                {
                    maxcluster = k;
                    maxCount = args.counts[k];
                }
            }

            if (maxcluster != -1 && (args.clusterIdx[maxcluster] < 0 || args.clusterIdx[maxcluster] >= data.R()))
                std::cout << "maxcluster:" << maxcluster << "(" << args.newCounts[maxcluster] << ") Error dist:" << args.clusterDist[maxcluster] << std::endl;

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
                for (int k = 0; k < args._DK; k++)
                    args.newCounts[k] += args.newCounts[i*args._K + k];
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
        inline void InitCenters(const Dataset<T>& data, 
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            KmeansArgs<T>& args, int samples, int tryIters) {
            SizeType batchEnd = min(first + samples, last);
            float currDist, minClusterDist = MaxDist;
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
                }
            }
        }

        template <typename T>
        int KmeansClustering(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            KmeansArgs<T>& args, int samples = 1000) {
            
            InitCenters(data, indices, first, last, args, samples, 3);
            
            SizeType batchEnd = min(first + samples, last);
            float currDiff, currDist, minClusterDist = MaxDist;
            int noImprovement = 0;
            for (int iter = 0; iter < 100; iter++) {
                std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
                std::random_shuffle(indices.begin() + first, indices.begin() + last);

                args.ClearCenters();
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                currDist = KmeansAssign(data, indices, first, batchEnd, args, true, 
                    COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / (100.0f * (batchEnd - first)));
                std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                if (currDist < minClusterDist) {
                    noImprovement = 0;
                    minClusterDist = currDist;
                }
                else {
                    noImprovement++;
                }
                currDiff = RefineCenters(data, args);
                if (currDiff < 1e-3 || noImprovement >= 5) break;
            }

            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign(data, indices, first, last, args, false, 0);
            std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

            int numClusters = 0;
            for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;

            if (numClusters <= 1) {
                //if (last - first > 1) std::cout << "large cluster:" << last - first << " dist:" << currDist << std::endl;
                return numClusters;
            }
            args.Shuffle(indices, first, last);
            return numClusters;
        }

        class BKTree
        {
        public:
            BKTree(): m_iTreeNumber(1), m_iBKTKmeansK(32), m_iBKTLeafSize(8), m_iSamples(1000), m_lock(new std::shared_timed_mutex) {}
            
            BKTree(const BKTree& other): m_iTreeNumber(other.m_iTreeNumber), 
                                   m_iBKTKmeansK(other.m_iBKTKmeansK), 
                                   m_iBKTLeafSize(other.m_iBKTLeafSize),
                                   m_iSamples(other.m_iSamples),
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
            void Rebuild(const Dataset<T>& data, DistCalcMethod distMethod)
            {
                BKTree newTrees(*this);
                newTrees.BuildTrees<T>(data, distMethod, 1);

                std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                m_pTreeRoots.swap(newTrees.m_pTreeRoots);
                m_pTreeStart.swap(newTrees.m_pTreeStart);
                m_pSampleCenterMap.swap(newTrees.m_pSampleCenterMap);
            }

            template <typename T>
            void BuildTrees(const Dataset<T>& data, DistCalcMethod distMethod, int numOfThreads, std::vector<SizeType>* indices = nullptr, std::vector<SizeType>* reverseIndices = nullptr, bool dynamicK = false)
            {
                struct  BKTStackItem {
                    SizeType index, first, last;
                    BKTStackItem(SizeType index_, SizeType first_, SizeType last_) : index(index_), first(first_), last(last_) {}
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

                m_pSampleCenterMap.clear();
                for (char i = 0; i < m_iTreeNumber; i++)
                {
                    std::random_shuffle(localindices.begin(), localindices.end());

                    m_pTreeStart.push_back((SizeType)m_pTreeRoots.size());
                    m_pTreeRoots.emplace_back((SizeType)localindices.size());
                    std::cout << "Start to build BKTree " << i + 1 << std::endl;

                    ss.push(BKTStackItem(m_pTreeStart[i], 0, (SizeType)localindices.size()));
                    while (!ss.empty()) {
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

                            int numClusters = KmeansClustering(data, localindices, item.first, item.last, args, m_iSamples);
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
                                for (int k = 0; k < m_iBKTKmeansK; k++) {
                                    if (args.counts[k] == 0) continue;
                                    SizeType cid = (reverseIndices == nullptr) ? localindices[item.first + args.counts[k] - 1] : reverseIndices->at(localindices[item.first + args.counts[k] - 1]);
                                    m_pTreeRoots.emplace_back(cid);
                                    if (args.counts[k] > 1) ss.push(BKTStackItem(newBKTid++, item.first, item.first + args.counts[k] - 1));
                                    item.first += args.counts[k];
                                }
                            }
                        }
                        m_pTreeRoots[item.index].childEnd = (SizeType)m_pTreeRoots.size();
                    }
                    m_pTreeRoots.emplace_back(-1);
                    std::cout << i + 1 << " BKTree built, " << m_pTreeRoots.size() - m_pTreeStart[i] << " " << localindices.size() << std::endl;
                }
            }

            inline std::uint64_t BufferSize() const
            {
                return sizeof(int) + sizeof(SizeType) * m_iTreeNumber +
                    sizeof(SizeType) + sizeof(BKTNode) * m_pTreeRoots.size();
            }

            bool SaveTrees(std::ostream& p_outstream) const
            {
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                p_outstream.write((char*)&m_iTreeNumber, sizeof(int));
                p_outstream.write((char*)m_pTreeStart.data(), sizeof(SizeType) * m_iTreeNumber);
                SizeType treeNodeSize = (SizeType)m_pTreeRoots.size();
                p_outstream.write((char*)&treeNodeSize, sizeof(SizeType));
                p_outstream.write((char*)m_pTreeRoots.data(), sizeof(BKTNode) * treeNodeSize);
                std::cout << "Save BKT (" << m_iTreeNumber << "," << treeNodeSize << ") Finish!" << std::endl;
                return true;
            }

            bool SaveTrees(std::string sTreeFileName) const
            {
                std::cout << "Save BKT to " << sTreeFileName << std::endl;
                std::ofstream output(sTreeFileName, std::ios::binary);
                if (!output.is_open()) return false;
                SaveTrees(output);
                output.close();
                return true;
            }

            bool LoadTrees(char* pBKTMemFile)
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
                std::cout << "Load BKT (" << m_iTreeNumber << "," << treeNodeSize << ") Finish!" << std::endl;
                return true;
            }

            bool LoadTrees(std::istream& input)
            {
                input.read((char*)&m_iTreeNumber, sizeof(int));
                m_pTreeStart.resize(m_iTreeNumber);
                input.read((char*)m_pTreeStart.data(), sizeof(SizeType) * m_iTreeNumber);

                SizeType treeNodeSize;
                input.read((char*)&treeNodeSize, sizeof(SizeType));
                m_pTreeRoots.resize(treeNodeSize);
                input.read((char*)m_pTreeRoots.data(), sizeof(BKTNode) * treeNodeSize);

                if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
                std::cout << "Load BKT (" << m_iTreeNumber << "," << treeNodeSize << ") Finish!" << std::endl;
                return true;
            }

            bool LoadTrees(std::string sTreeFileName)
            {
                std::cout << "Load BKT From " << sTreeFileName << std::endl;
                std::ifstream input(sTreeFileName, std::ios::binary);
                if (!input.is_open()) return false;
                LoadTrees(input);
                input.close();
                return true;
            }

            template <typename T>
            void InitSearchTrees(const Dataset<T>& data, float(*fComputeDistance)(const T* pX, const T* pY, DimensionType length), const COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space) const
            {
                for (char i = 0; i < m_iTreeNumber; i++) {
                    const BKTNode& node = m_pTreeRoots[m_pTreeStart[i]];
                    if (node.childStart < 0) {
                        p_space.m_SPTQueue.insert(COMMON::HeapCell(m_pTreeStart[i], fComputeDistance(p_query.GetTarget(), data[node.centerid], data.C())));
                    } 
                    else {
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(COMMON::HeapCell(begin, fComputeDistance(p_query.GetTarget(), data[index], data.C())));
                        }
                    } 
                }
            }

            template <typename T>
            void SearchTrees(const Dataset<T>& data, float(*fComputeDistance)(const T* pX, const T* pY, DimensionType length), const COMMON::QueryResultSet<T> &p_query,
                COMMON::WorkSpace &p_space, const int p_limits) const
            {
                while (!p_space.m_SPTQueue.empty())
                {
                    COMMON::HeapCell bcell = p_space.m_SPTQueue.pop();
                    const BKTNode& tnode = m_pTreeRoots[bcell.node];
                    if (tnode.childStart < 0) {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_iNumberOfCheckedLeaves++;
                            p_space.m_NGQueue.insert(COMMON::HeapCell(tnode.centerid, bcell.distance));
                        }
                        if (p_space.m_iNumberOfCheckedLeaves >= p_limits) break;
                    }
                    else {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_NGQueue.insert(COMMON::HeapCell(tnode.centerid, bcell.distance));
                        }
                        for (SizeType begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(COMMON::HeapCell(begin, fComputeDistance(p_query.GetTarget(), data[index], data.C())));
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
            int m_iTreeNumber, m_iBKTKmeansK, m_iBKTLeafSize, m_iSamples;
        };
    }
}
#endif
