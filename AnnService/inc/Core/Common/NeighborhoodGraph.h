// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_NG_H_
#define _SPTAG_COMMON_NG_H_

#include "inc/Core/VectorIndex.h"

#include "CommonUtils.h"
#include "Dataset.h"
#include "FineGrainedLock.h"
#include "QueryResultSet.h"

#include <chrono>
#include <queue>
#include <atomic>

#if defined(GPU)
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <typeinfo>
#include <cuda_fp16.h>

#include "inc/Core/Common/cuda/KNN.hxx"
#include "inc/Core/Common/cuda/params.h"
#endif


namespace SPTAG
{
    namespace COMMON
    {
        class NeighborhoodGraph
        {
        public:
            NeighborhoodGraph() : m_iTPTNumber(32),
                m_iTPTLeafSize(2000),
                m_iSamples(1000),
                m_numTopDimensionTPTSplit(5),
                m_iNeighborhoodSize(32),
                m_fNeighborhoodScale(2.0),
                m_fCEFScale(2.0),
                m_fRNGFactor(1.0),
                m_iRefineIter(2),
                m_iCEF(1000),
                m_iAddCEF(500),
                m_iMaxCheckForRefineGraph(10000),
                m_iGPUGraphType(2),
                m_iGPURefineSteps(0),
                m_iGPURefineDepth(2),
                m_iGPULeafSize(500),
                m_iheadNumGPUs(1),
                m_iTPTBalanceFactor(2),
                m_rebuild(0)
            {}

            ~NeighborhoodGraph() {}

            virtual void InsertNeighbors(VectorIndex* index, const SizeType node, SizeType insertNode, float insertDist) = 0;

            virtual void RebuildNeighbors(VectorIndex* index, const SizeType node, SizeType* nodes, const BasicResult* queryResults, const int numResults) = 0;

            virtual float GraphAccuracyEstimation(VectorIndex* index, const SizeType samples, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                DimensionType* correct = new DimensionType[samples];

#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < samples; i++)
                {
                    SizeType x = COMMON::Utils::rand(m_iGraphSize);
                    //int x = i;
                    COMMON::QueryResultSet<void> query(nullptr, m_iCEF);
                    for (SizeType y = 0; y < m_iGraphSize; y++)
                    {
                        if ((idmap != nullptr && idmap->find(y) != idmap->end())) continue;
                        float dist = index->ComputeDistance(index->GetSample(x), index->GetSample(y));
                        query.AddPoint(y, dist);
                    }
                    query.SortResult();
                    SizeType* exact_rng = new SizeType[m_iNeighborhoodSize];
                    RebuildNeighbors(index, x, exact_rng, query.GetResults(), m_iCEF);

                    correct[i] = 0;
                    for (DimensionType j = 0; j < m_iNeighborhoodSize; j++) {
                        if (exact_rng[j] == -1) {
                            correct[i] += m_iNeighborhoodSize - j;
                            break;
                        }
                        for (DimensionType k = 0; k < m_iNeighborhoodSize; k++)
                            if ((m_pNeighborhoodGraph)[x][k] == exact_rng[j]) {
                                correct[i]++;
                                break;
                            }
                    }
                    delete[] exact_rng;
                }
                float acc = 0;
                for (SizeType i = 0; i < samples; i++) acc += float(correct[i]);
                acc = acc / samples / m_iNeighborhoodSize;
                delete[] correct;
                return acc;
            }

#if defined(GPU)
            template <typename T>
            void BuildInitKNNGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap)
            {
                SizeType initSize;
                SPTAG::Helper::Convert::ConvertStringTo(index->GetParameter("NumberOfInitialDynamicPivots").c_str(), initSize);

                // Build the entire RNG graph, both builds the KNN and refines it to RNG
                buildGraph<T>(index, m_iGraphSize, m_iNeighborhoodSize, m_iTPTNumber, (int*)m_pNeighborhoodGraph[0], m_iGPURefineSteps, m_iGPURefineDepth, m_iGPUGraphType, m_iGPULeafSize, initSize, m_iheadNumGPUs, m_iTPTBalanceFactor);

                if (idmap != nullptr) {
                    std::unordered_map<SizeType, SizeType>::const_iterator iter;
                    for (SizeType i = 0; i < m_iGraphSize; i++) {
                        for (DimensionType j = 0; j < m_iNeighborhoodSize; j++) {
                            if ((iter = idmap->find(m_pNeighborhoodGraph[i][j])) != idmap->end())
                                m_pNeighborhoodGraph[i][j] = iter->second;
                        }
                    }
                }
            }
#else
            template <typename T>
            void PartitionByTptree(VectorIndex* index, std::vector<SizeType>& indices, const SizeType first, const SizeType last,
                std::vector<std::pair<SizeType, SizeType>>& leaves)
            {
                if (index->m_pQuantizer)
                {
                    switch (index->m_pQuantizer->GetReconstructType())
                    {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
PartitionByTptreeCore<T, Type>(index, indices, first, last, leaves); \
break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                    default: break;
                    }
                }
                else
                {
                    PartitionByTptreeCore<T, T>(index, indices, first, last, leaves);
                }
            }

            template <typename T, typename R>
            void PartitionByTptreeCore(VectorIndex* index, std::vector<SizeType>& indices, const SizeType first, const SizeType last,
                std::vector<std::pair<SizeType, SizeType>>& leaves)
            {
                if (last - first <= m_iTPTLeafSize)
                {
                    leaves.emplace_back(first, last);
                }
                else
                {
                    SizeType splitidx = first;
                    {
                        int iIteration = 100;
                        SizeType end = min(first + m_iSamples, last);
                        SizeType count = end - first + 1;
                        bool quantizer_exists = (bool)index->m_pQuantizer;

                        SizeType cols = index->GetFeatureDim();
                        std::shared_ptr<VectorSet> indices_vectors;
                        if (quantizer_exists) {
                            cols = index->m_pQuantizer->ReconstructDim();
                            indices_vectors.reset(new BasicVectorSet(ByteArray::Alloc(sizeof(R) * cols * count), GetEnumValueType<R>(), cols, count));
                            for (int i = 0; i < count; i++) index->m_pQuantizer->ReconstructVector((uint8_t*)index->GetSample(indices[first + i]), indices_vectors->GetVector(i));
                        }

                        std::vector<float> Mean(cols, 0);
                        // calculate the mean of each dimension
                        for (SizeType j = first; j <= end; j++)
                        {
                            R* v = (quantizer_exists) ? (R*)indices_vectors->GetVector(j - first) : (R*)index->GetSample(indices[j]);
                            for (DimensionType k = 0; k < cols; k++)
                            {
                                Mean[k] += v[k];
                            }
                        }

                        std::vector<BasicResult> Variance;
                        Variance.reserve(cols);
                        for (DimensionType j = 0; j < cols; j++)
                        {
                            Mean[j] /= count;
                            Variance.emplace_back(j, 0.0f);
                        }

                        // calculate the variance of each dimension
                        for (SizeType j = first; j <= end; j++)
                        {
                            R* v = (quantizer_exists) ? (R*)indices_vectors->GetVector(j - first) : (R*)index->GetSample(indices[j]);
                            for (DimensionType k = 0; k < cols; k++)
                            {
                                float dist = v[k] - Mean[k];
                                Variance[k].Dist += dist * dist;
                            }
                        }

                        std::sort(Variance.begin(), Variance.end(), COMMON::Compare);
                        std::vector<SizeType> indexs(m_numTopDimensionTPTSplit);
                        std::vector<float> weight(m_numTopDimensionTPTSplit), bestweight(m_numTopDimensionTPTSplit);
                        float bestvariance = Variance[cols - 1].Dist;
                        for (int i = 0; i < m_numTopDimensionTPTSplit; i++)
                        {
                            indexs[i] = Variance[cols - 1 - i].VID;
                            bestweight[i] = 0;
                        }
                        bestweight[0] = 1;
                        float bestmean = Mean[indexs[0]];

                        std::vector<float> Val(count);
                        for (int i = 0; i < iIteration; i++)
                        {
                            float sumweight = 0;
                            for (int j = 0; j < m_numTopDimensionTPTSplit; j++)
                            {
                                weight[j] = float(rand() % 10000) / 5000.0f - 1.0f;
                                sumweight += weight[j] * weight[j];
                            }
                            sumweight = sqrt(sumweight);
                            for (int j = 0; j < m_numTopDimensionTPTSplit; j++)
                            {
                                weight[j] /= sumweight;
                            }
                            float mean = 0;
                            for (SizeType j = 0; j < count; j++)
                            {
                                Val[j] = 0;
                                R* v = (quantizer_exists) ? (R*)indices_vectors->GetVector(j) : (R*)index->GetSample(indices[first + j]);
                                for (int k = 0; k < m_numTopDimensionTPTSplit; k++)
                                {
                                    Val[j] += weight[k] * v[indexs[k]];
                                }
                                mean += Val[j];
                            }
                            mean /= count;
                            float var = 0;
                            for (SizeType j = 0; j < count; j++)
                            {
                                float dist = Val[j] - mean;
                                var += dist * dist;
                            }
                            if (var > bestvariance)
                            {
                                bestvariance = var;
                                bestmean = mean;
                                for (int j = 0; j < m_numTopDimensionTPTSplit; j++)
                                {
                                    bestweight[j] = weight[j];
                                }
                            }
                        }
                        SizeType i = first;
                        SizeType j = last;
                        // decide which child one point belongs
                        while (i <= j)
                        {
                            float val = 0;
                            R* v = (quantizer_exists) ? (R*)indices_vectors->GetVector(i - first) : (R*)index->GetSample(indices[i]);
                            for (int k = 0; k < m_numTopDimensionTPTSplit; k++)
                            {
                                val += bestweight[k] * v[indexs[k]];
                            }
                            if (val < bestmean)
                            {
                                i++;
                            }
                            else
                            {
                                std::swap(indices[i], indices[j]);
                                j--;
                            }
                        }
                        // if all the points in the node are equal,equally split the node into 2
                        if ((i == first) || (i == last + 1))
                        {
                            i = (first + last + 1) / 2;
                        }
                        splitidx = i;

                        Mean.clear();
                        Variance.clear();
                        Val.clear();
                        indexs.clear();
                        weight.clear();
                        bestweight.clear();
                    }

                    PartitionByTptreeCore<T, R>(index, indices, first, splitidx - 1, leaves);
                    PartitionByTptreeCore<T, R>(index, indices, splitidx, last, leaves);
                }
            }

            template <typename T>
            void BuildInitKNNGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap)
            {

                COMMON::Dataset<float> NeighborhoodDists(m_iGraphSize, m_iNeighborhoodSize, index->m_iDataBlockSize, index->m_iDataCapacity);
                std::vector<std::vector<SizeType>> TptreeDataIndices(m_iTPTNumber, std::vector<SizeType>(m_iGraphSize));
                std::vector<std::vector<std::pair<SizeType, SizeType>>> TptreeLeafNodes(m_iTPTNumber, std::vector<std::pair<SizeType, SizeType>>());

                for (SizeType i = 0; i < m_iGraphSize; i++)
                    for (DimensionType j = 0; j < m_iNeighborhoodSize; j++)
                        (NeighborhoodDists)[i][j] = MaxDist;

                auto t1 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Parallel TpTree Partition begin\n");
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < m_iTPTNumber; i++)
                {
                    Sleep(i * 100); std::srand(clock());
                    for (SizeType j = 0; j < m_iGraphSize; j++) TptreeDataIndices[i][j] = j;
                    std::shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end(), rg);
                    PartitionByTptree<T>(index, TptreeDataIndices[i], 0, m_iGraphSize - 1, TptreeLeafNodes[i]);
                    LOG(Helper::LogLevel::LL_Info, "Finish Getting Leaves for Tree %d\n", i);
                }
                LOG(Helper::LogLevel::LL_Info, "Parallel TpTree Partition done\n");
                auto t2 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Build TPTree time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count());

                for (int i = 0; i < m_iTPTNumber; i++)
                {
#pragma omp parallel for schedule(dynamic)
                    for (SizeType j = 0; j < (SizeType)TptreeLeafNodes[i].size(); j++)
                    {
                        SizeType start_index = TptreeLeafNodes[i][j].first;
                        SizeType end_index = TptreeLeafNodes[i][j].second;
                        if ((j * 5) % TptreeLeafNodes[i].size() == 0) LOG(Helper::LogLevel::LL_Info, "Processing Tree %d %d%%\n", i, static_cast<int>(j * 1.0 / TptreeLeafNodes[i].size() * 100));
                        for (SizeType x = start_index; x < end_index; x++)
                        {
                            for (SizeType y = x + 1; y <= end_index; y++)
                            {
                                SizeType p1 = TptreeDataIndices[i][x];
                                SizeType p2 = TptreeDataIndices[i][y];
                                float dist = index->ComputeDistance(index->GetSample(p1), index->GetSample(p2));
                                if (idmap != nullptr) {
                                    p1 = (idmap->find(p1) == idmap->end()) ? p1 : idmap->at(p1);
                                    p2 = (idmap->find(p2) == idmap->end()) ? p2 : idmap->at(p2);
                                }
                                COMMON::Utils::AddNeighbor(p2, dist, (m_pNeighborhoodGraph)[p1], (NeighborhoodDists)[p1], m_iNeighborhoodSize);
                                COMMON::Utils::AddNeighbor(p1, dist, (m_pNeighborhoodGraph)[p2], (NeighborhoodDists)[p2], m_iNeighborhoodSize);
                            }
                        }
                    }
                    TptreeDataIndices[i].clear();
                    TptreeLeafNodes[i].clear();
                }
                TptreeDataIndices.clear();
                TptreeLeafNodes.clear();

                auto t3 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Process TPTree time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count());
            }
#endif

            template <typename T>
            void BuildGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                LOG(Helper::LogLevel::LL_Info, "build RNG graph!\n");

                m_iGraphSize = index->GetNumSamples();
                m_iNeighborhoodSize = (DimensionType)(ceil(m_iNeighborhoodSize * m_fNeighborhoodScale) * (m_rebuild + 1));
                m_pNeighborhoodGraph.Initialize(m_iGraphSize, m_iNeighborhoodSize, index->m_iDataBlockSize, index->m_iDataCapacity);

                if (m_iGraphSize < 1000) {
                    RefineGraph<T>(index, idmap);
                    LOG(Helper::LogLevel::LL_Info, "Build RNG Graph end!\n");
                    return;
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                BuildInitKNNGraph<T>(index, idmap);
                auto t2 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "BuildInitKNNGraph time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count());

                RefineGraph<T>(index, idmap);

                auto t3 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "BuildGraph time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count());

                if (m_rebuild) {
                    m_iNeighborhoodSize = m_iNeighborhoodSize / 2;
                    RebuildGraph<T>(index, idmap);
                    auto t4 = std::chrono::high_resolution_clock::now();
                    LOG(Helper::LogLevel::LL_Info, "ReBuildGraph time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count());
                }

                if (idmap != nullptr) {
                    for (auto iter = idmap->begin(); iter != idmap->end(); iter++)
                        if (iter->first < 0)
                        {
                            m_pNeighborhoodGraph[-1 - iter->first][m_iNeighborhoodSize - 1] = -2 - iter->second;
                        }
                }
            }

            template <typename T>
            void RebuildGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                std::vector<int> indegree(m_iGraphSize);

#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < m_iGraphSize; i++) indegree[i] = 0;

                auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < m_iGraphSize; i++)
                {
                    SizeType* outnodes = m_pNeighborhoodGraph[i];
                    for (SizeType j = 0; j < m_iNeighborhoodSize; j++)
                    {
                        int node = outnodes[j];
                        if (node >= 0) {
                            indegree[node]++;
                        }
                    }
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Calculate Indegree time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count());
                int rebuild_threshold = m_iNeighborhoodSize / 2;
                int rebuildstart = m_iNeighborhoodSize / 2;
#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < m_iGraphSize; i++)
                {
                    SizeType* outnodes = m_pNeighborhoodGraph[i];
                    std::vector<bool> reserve(2 * m_iNeighborhoodSize, false);
                    int total = 0;
                    for (SizeType j = rebuildstart; j < m_iNeighborhoodSize * 2; j++)
                        if ( outnodes[j] >= 0 && indegree[outnodes[j]] < rebuild_threshold) {
                            reserve[j] = true;
                            total++;
                        }

                    for (SizeType j = rebuildstart; j < m_iNeighborhoodSize * 2 && total < m_iNeighborhoodSize - rebuildstart; j++) {
                        if (!reserve[j]) {
                            reserve[j] = true;
                            total++;
                        }
                    }
                    for (SizeType j = rebuildstart, z = rebuildstart; j < m_iNeighborhoodSize; j++) {
                        while (!reserve[z]) z++;
                        if(outnodes[j] >= 0) indegree[outnodes[j]] = indegree[outnodes[j]] - 1;
                        if(outnodes[z] >= 0) indegree[outnodes[z]] = indegree[outnodes[z]] + 1;
                        outnodes[j] = outnodes[z];
                        z++;
                    }
                    if ((i * 5) % m_iGraphSize == 0) LOG(Helper::LogLevel::LL_Info, "Rebuild %d%%\n", static_cast<int>(i * 1.0 / m_iGraphSize * 100));
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Rebuild RNG time (s): %lld Graph Acc: %f\n", std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count(), GraphAccuracyEstimation(index, 100, idmap));
            }

            template <typename T>
            void RefineGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                for (int iter = 0; iter < m_iRefineIter - 1; iter++)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
                    for (SizeType i = 0; i < m_iGraphSize; i++)
                    {
                        RefineNode<T>(index, i, false, false, (int)(m_iCEF * m_fCEFScale));
                        if ((i * 5) % m_iGraphSize == 0) LOG(Helper::LogLevel::LL_Info, "Refine %d %d%%\n", iter, static_cast<int>(i * 1.0 / m_iGraphSize * 100));
                    }
                    auto t2 = std::chrono::high_resolution_clock::now();
                    LOG(Helper::LogLevel::LL_Info, "Refine RNG time (s): %lld Graph Acc: %f\n", std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count(), GraphAccuracyEstimation(index, 100, idmap));
                }

                m_iNeighborhoodSize = (DimensionType)(m_iNeighborhoodSize / m_fNeighborhoodScale);

                if (m_iRefineIter > 0) {
                    auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
                    for (SizeType i = 0; i < m_iGraphSize; i++)
                    {
                        RefineNode<T>(index, i, false, false, m_iCEF);
                        if ((i * 5) % m_iGraphSize == 0) LOG(Helper::LogLevel::LL_Info, "Refine %d %d%%\n", m_iRefineIter - 1, static_cast<int>(i * 1.0 / m_iGraphSize * 100));
                    }
                    auto t2 = std::chrono::high_resolution_clock::now();
                    LOG(Helper::LogLevel::LL_Info, "Refine RNG time (s): %lld Graph Acc: %f\n", std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count(), GraphAccuracyEstimation(index, 100, idmap));
                }
                else {
                    LOG(Helper::LogLevel::LL_Info, "Graph Acc: %f\n", GraphAccuracyEstimation(index, 100, idmap));
                }
            }

            template <typename T>
            ErrorCode RefineGraph(VectorIndex* index, std::vector<SizeType>& indices, std::vector<SizeType>& reverseIndices,
                std::shared_ptr<Helper::DiskIO> output, NeighborhoodGraph* newGraph, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                std::shared_ptr<NeighborhoodGraph> tmp;
                if (newGraph == nullptr) {
                    tmp = NeighborhoodGraph::CreateInstance(Type());
                    newGraph = tmp.get();
                }

                SizeType R = (SizeType)indices.size();
                newGraph->m_pNeighborhoodGraph.Initialize(R, m_iNeighborhoodSize, index->m_iDataBlockSize, index->m_iDataCapacity);
                newGraph->m_iGraphSize = R;
                newGraph->m_iNeighborhoodSize = m_iNeighborhoodSize;

#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < R; i++)
                {
                    if ((i * 5) % R == 0) LOG(Helper::LogLevel::LL_Info, "Refine %d%%\n", static_cast<int>(i * 1.0 / R * 100));

                    SizeType* outnodes = newGraph->m_pNeighborhoodGraph[i];

                    COMMON::QueryResultSet<T> query((const T*)index->GetSample(indices[i]), m_iCEF + 1);
                    index->RefineSearchIndex(query, false);
                    RebuildNeighbors(index, indices[i], outnodes, query.GetResults(), m_iCEF + 1);

                    std::unordered_map<SizeType, SizeType>::const_iterator iter;
                    for (DimensionType j = 0; j < m_iNeighborhoodSize; j++)
                    {
                        if (outnodes[j] >= 0 && outnodes[j] < reverseIndices.size()) outnodes[j] = reverseIndices[outnodes[j]];
                        if (idmap != nullptr && (iter = idmap->find(outnodes[j])) != idmap->end()) outnodes[j] = iter->second;
                    }
                    if (idmap != nullptr && (iter = idmap->find(-1 - i)) != idmap->end())
                        outnodes[m_iNeighborhoodSize - 1] = -2 - iter->second;
                }

                if (output != nullptr) newGraph->SaveGraph(output);
                return ErrorCode::Success;
            }

            template <typename T>
            void RefineNode(VectorIndex* index, const SizeType node, bool updateNeighbors, bool searchDeleted, int CEF)
            {
                COMMON::QueryResultSet<T> query((const T*)index->GetSample(node), CEF + 1);
                void* rec_query = nullptr;
                if (index->m_pQuantizer) {
                    rec_query = ALIGN_ALLOC(index->m_pQuantizer->ReconstructSize());
                    index->m_pQuantizer->ReconstructVector((const uint8_t*)query.GetTarget(), rec_query);
                    query.SetTarget((T*)rec_query, index->m_pQuantizer);
                }
                index->RefineSearchIndex(query, searchDeleted);
                RebuildNeighbors(index, node, m_pNeighborhoodGraph[node], query.GetResults(), CEF + 1);
                if (rec_query)
                {
                    ALIGN_FREE(rec_query);
                }
                if (updateNeighbors) {
                    // update neighbors
                    for (int j = 0; j <= CEF; j++)
                    {
                        BasicResult* item = query.GetResult(j);
                        if (item->VID < 0) break;
                        if (item->VID == node) continue;

                        InsertNeighbors(index, item->VID, node, item->Dist);
                    }
                }
            }

            inline std::uint64_t BufferSize() const
            {
                return m_pNeighborhoodGraph.BufferSize();
            }

            ErrorCode LoadGraph(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity)
            {
                ErrorCode ret = ErrorCode::Success;
                if ((ret = m_pNeighborhoodGraph.Load(input, blockSize, capacity)) != ErrorCode::Success) return ret;

                m_iGraphSize = m_pNeighborhoodGraph.R();
                m_iNeighborhoodSize = m_pNeighborhoodGraph.C();
                return ret;
            }

            ErrorCode LoadGraph(std::string sGraphFilename, SizeType blockSize, SizeType capacity)
            {
                ErrorCode ret = ErrorCode::Success;
                if ((ret = m_pNeighborhoodGraph.Load(sGraphFilename, blockSize, capacity)) != ErrorCode::Success) return ret;

                m_iGraphSize = m_pNeighborhoodGraph.R();
                m_iNeighborhoodSize = m_pNeighborhoodGraph.C();
                return ret;
            }

            ErrorCode LoadGraph(char* pGraphMemFile, SizeType blockSize, SizeType capacity)
            {
                ErrorCode ret = ErrorCode::Success;
                if ((ret = m_pNeighborhoodGraph.Load(pGraphMemFile, blockSize, capacity)) != ErrorCode::Success) return ret;

                m_iGraphSize = m_pNeighborhoodGraph.R();
                m_iNeighborhoodSize = m_pNeighborhoodGraph.C();
                return ErrorCode::Success;
            }

            ErrorCode SaveGraph(std::string sGraphFilename) const
            {
                LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", m_pNeighborhoodGraph.Name().c_str(), sGraphFilename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sGraphFilename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return SaveGraph(ptr);
            }

            ErrorCode SaveGraph(std::shared_ptr<Helper::DiskIO> output) const
            {
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&m_iGraphSize);
                IOBINARY(output, WriteBinary, sizeof(DimensionType), (char*)&m_iNeighborhoodSize);

                for (int i = 0; i < m_iGraphSize; i++)
                    IOBINARY(output, WriteBinary, sizeof(SizeType) * m_iNeighborhoodSize, (char*)m_pNeighborhoodGraph[i]);
                LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", m_pNeighborhoodGraph.Name().c_str(), m_iGraphSize, m_iNeighborhoodSize);
                return ErrorCode::Success;
            }

            inline ErrorCode AddBatch(SizeType num)
            {
                ErrorCode ret = m_pNeighborhoodGraph.AddBatch(num);
                if (ret != ErrorCode::Success) return ret;

                m_iGraphSize += num;
                return ErrorCode::Success;
            }

            inline SizeType* operator[](SizeType index) { return m_pNeighborhoodGraph[index]; }

            inline const SizeType* operator[](SizeType index) const { return m_pNeighborhoodGraph[index]; }

            void Update(SizeType row, DimensionType col, SizeType val) {
                std::lock_guard<std::mutex> lock(m_dataUpdateLock[row]);
                m_pNeighborhoodGraph[row][col] = val;
            }

            inline void SetR(SizeType rows) {
                m_pNeighborhoodGraph.SetR(rows);
                m_iGraphSize = rows;
            }

            inline SizeType R() const { return m_iGraphSize; }

            inline std::string Type() const { return m_pNeighborhoodGraph.Name(); }

            static std::shared_ptr<NeighborhoodGraph> CreateInstance(std::string type);

        protected:
            // Graph structure
            SizeType m_iGraphSize;
            COMMON::Dataset<SizeType> m_pNeighborhoodGraph;
            FineGrainedLock m_dataUpdateLock;
        public:
            int m_iTPTNumber, m_iTPTLeafSize, m_iSamples, m_numTopDimensionTPTSplit;
            DimensionType m_iNeighborhoodSize;
            float m_fNeighborhoodScale, m_fCEFScale, m_fRNGFactor;
            int m_iRefineIter, m_iCEF, m_iAddCEF, m_iMaxCheckForRefineGraph, m_iGPUGraphType, m_iGPURefineSteps, m_iGPURefineDepth, m_iGPULeafSize, m_iheadNumGPUs, m_iTPTBalanceFactor, m_rebuild;
        };
    }
}
#endif
