// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_NG_H_
#define _SPTAG_COMMON_NG_H_

#include "../VectorIndex.h"

#include "CommonUtils.h"
#include "Dataset.h"
#include "FineGrainedLock.h"
#include "QueryResultSet.h"

namespace SPTAG
{
    namespace COMMON
    {
        class NeighborhoodGraph 
        {
        public:
            NeighborhoodGraph(): m_iTPTNumber(32), 
                                 m_iTPTLeafSize(2000), 
                                 m_iSamples(1000), 
                                 m_numTopDimensionTPTSplit(5),
                                 m_iNeighborhoodSize(32),
                                 m_iNeighborhoodScale(2),
                                 m_iCEFScale(2),
                                 m_iRefineIter(2),
                                 m_iCEF(1000),
                                 m_iMaxCheckForRefineGraph(10000) 
            {
                m_pNeighborhoodGraph.SetName("Graph");
            }

            ~NeighborhoodGraph() {}

            virtual void InsertNeighbors(VectorIndex* index, const SizeType node, SizeType insertNode, float insertDist) = 0;

            virtual void RebuildNeighbors(VectorIndex* index, const SizeType node, SizeType* nodes, const BasicResult* queryResults, const int numResults) = 0;

            virtual float GraphAccuracyEstimation(VectorIndex* index, const SizeType samples, const std::unordered_map<SizeType, SizeType>* idmap = nullptr) = 0;

            template <typename T>
            void BuildGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                std::cout << "build RNG graph!" << std::endl;

                m_iGraphSize = index->GetNumSamples();
                m_iNeighborhoodSize = m_iNeighborhoodSize * m_iNeighborhoodScale;
                m_pNeighborhoodGraph.Initialize(m_iGraphSize, m_iNeighborhoodSize);
                
                if (m_iGraphSize < 1000) {
                    RefineGraph<T>(index, idmap);
                    std::cout << "Build RNG Graph end!" << std::endl;
                    return;
                }

                {
                    COMMON::Dataset<float> NeighborhoodDists(m_iGraphSize, m_iNeighborhoodSize);
                    std::vector<std::vector<SizeType>> TptreeDataIndices(m_iTPTNumber, std::vector<SizeType>(m_iGraphSize));
                    std::vector<std::vector<std::pair<SizeType, SizeType>>> TptreeLeafNodes(m_iTPTNumber, std::vector<std::pair<SizeType, SizeType>>());

                    for (SizeType i = 0; i < m_iGraphSize; i++)
                        for (DimensionType j = 0; j < m_iNeighborhoodSize; j++)
                            (NeighborhoodDists)[i][j] = MaxDist;

                    std::cout << "Parallel TpTree Partition begin " << std::endl;
#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < m_iTPTNumber; i++)
                    {
                        Sleep(i * 100); std::srand(clock());
                        for (SizeType j = 0; j < m_iGraphSize; j++) TptreeDataIndices[i][j] = j;
                        std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
                        PartitionByTptree<T>(index, TptreeDataIndices[i], 0, m_iGraphSize - 1, TptreeLeafNodes[i]);
                        std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
                    }
                    std::cout << "Parallel TpTree Partition done" << std::endl;

                    for (int i = 0; i < m_iTPTNumber; i++)
                    {
#pragma omp parallel for schedule(dynamic)
                        for (SizeType j = 0; j < (SizeType)TptreeLeafNodes[i].size(); j++)
                        {
                            SizeType start_index = TptreeLeafNodes[i][j].first;
                            SizeType end_index = TptreeLeafNodes[i][j].second;
                            if (omp_get_thread_num() == 0) std::cout << "\rProcessing Tree " << i << ' ' << j * 100 / TptreeLeafNodes[i].size() << '%';
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
                        std::cout << std::endl;
                    }
                    TptreeDataIndices.clear();
                    TptreeLeafNodes.clear();
                }

                if (m_iMaxCheckForRefineGraph > 0) {
                    RefineGraph<T>(index, idmap);
                }
            }

            template <typename T>
            void RefineGraph(VectorIndex* index, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                m_iCEF *= m_iCEFScale;

                for (int iter = 0; iter < m_iRefineIter - 1; iter++)
                {
#pragma omp parallel for schedule(dynamic)
                    for (SizeType i = 0; i < m_iGraphSize; i++)
                    {
                        RefineNode<T>(index, i, false, false);
                        if (i % 1000 == 0) std::cout << "\rRefine " << iter << " " << static_cast<int>(i * 1.0 / m_iGraphSize * 100) << "%";
                    }
                    std::cout << "Refine RNG, graph acc:" << GraphAccuracyEstimation(index, 100, idmap) << std::endl;
                }

                m_iCEF /= m_iCEFScale;
                m_iNeighborhoodSize /= m_iNeighborhoodScale;

#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < m_iGraphSize; i++)
                {
                    RefineNode<T>(index, i, false, false);
                    if (i % 1000 == 0) std::cout << "\rRefine " << (m_iRefineIter - 1) << " " << static_cast<int>(i * 1.0 / m_iGraphSize * 100) << "%";
                }
                std::cout << "Refine RNG, graph acc:" << GraphAccuracyEstimation(index, 100, idmap) << std::endl;

                if (idmap != nullptr) {
                    for (auto iter = idmap->begin(); iter != idmap->end(); iter++)
                        if (iter->first < 0)
                        {
                            m_pNeighborhoodGraph[-1 - iter->first][m_iNeighborhoodSize - 1] = -2 - iter->second;
                        }
                }
            }

            template <typename T>
            ErrorCode RefineGraph(VectorIndex* index, std::vector<SizeType>& indices, std::vector<SizeType>& reverseIndices,
                std::ostream* output, NeighborhoodGraph* newGraph, const std::unordered_map<SizeType, SizeType>* idmap = nullptr)
            {
                SizeType R = (SizeType)indices.size();

                if (newGraph != nullptr)
                {
                    newGraph->m_pNeighborhoodGraph.Initialize(R, m_iNeighborhoodSize);
                    newGraph->m_iGraphSize = R;
                    newGraph->m_iNeighborhoodSize = m_iNeighborhoodSize;
                }

#pragma omp parallel for schedule(dynamic)
                for (SizeType i = 0; i < R; i++)
                {
                    RefineNode<T>(index, indices[i], false, false);
                    SizeType *nodes, *outnodes; 
                    nodes = outnodes = m_pNeighborhoodGraph[indices[i]];
                    if (newGraph != nullptr) outnodes = newGraph->m_pNeighborhoodGraph[i];
                    std::unordered_map<SizeType, SizeType>::const_iterator iter;
                    for (DimensionType j = 0; j < m_iNeighborhoodSize; j++)
                    {
                        if (nodes[j] >= 0 && nodes[j] < reverseIndices.size()) outnodes[j] = reverseIndices[nodes[j]];
                        if (idmap != nullptr && (iter = idmap->find(outnodes[j])) != idmap->end()) outnodes[j] = iter->second;
                    }
                    if (idmap != nullptr && (iter = idmap->find(-1 - i)) != idmap->end())
                        outnodes[m_iNeighborhoodSize - 1] = -2 - iter->second;
                }

                if (output != nullptr) {
                    output->write((char*)&R, sizeof(SizeType));
                    output->write((char*)&m_iNeighborhoodSize, sizeof(DimensionType));
                    for (SizeType i = 0; i < R; i++) {
                        output->write((char*)m_pNeighborhoodGraph[indices[i]], sizeof(SizeType) * m_iNeighborhoodSize);
                    }
                    std::cout << "Save Refine " << m_pNeighborhoodGraph.Name() << " (" << R << ", " << m_iNeighborhoodSize << ") Finish!" << std::endl;
                }
                return ErrorCode::Success;
            }


            template <typename T>
            void RefineNode(VectorIndex* index, const SizeType node, bool updateNeighbors, bool searchDeleted)
            {
                COMMON::QueryResultSet<T> query((const T*)index->GetSample(node), m_iCEF + 1);
                index->SearchIndex(query, searchDeleted);
                RebuildNeighbors(index, node, m_pNeighborhoodGraph[node], query.GetResults(), m_iCEF + 1);

                if (updateNeighbors) {
                    // update neighbors
                    for (int j = 0; j <= m_iCEF; j++)
                    {
                        BasicResult* item = query.GetResult(j);
                        if (item->VID < 0) break;
                        if (item->VID == node) continue;

                        InsertNeighbors(index, item->VID, node, item->Dist);
                    }
                }
            }

            template <typename T>
            void PartitionByTptree(VectorIndex* index, std::vector<SizeType>& indices, const SizeType first, const SizeType last,
                std::vector<std::pair<SizeType, SizeType>> & leaves)
            {
                if (last - first <= m_iTPTLeafSize)
                {
                    leaves.push_back(std::make_pair(first, last));
                }
                else
                {
                    std::vector<float> Mean(index->GetFeatureDim(), 0);

                    int iIteration = 100;
                    SizeType end = min(first + m_iSamples, last);
                    SizeType count = end - first + 1;
                    // calculate the mean of each dimension
                    for (SizeType j = first; j <= end; j++)
                    {
                        const T* v = (const T*)index->GetSample(indices[j]);
                        for (DimensionType k = 0; k < index->GetFeatureDim(); k++)
                        {
                            Mean[k] += v[k];
                        }
                    }
                    for (DimensionType k = 0; k < index->GetFeatureDim(); k++)
                    {
                        Mean[k] /= count;
                    }
                    std::vector<BasicResult> Variance;
                    Variance.reserve(index->GetFeatureDim());
                    for (DimensionType j = 0; j < index->GetFeatureDim(); j++)
                    {
                        Variance.push_back(BasicResult(j, 0));
                    }
                    // calculate the variance of each dimension
                    for (SizeType j = first; j <= end; j++)
                    {
                        const T* v = (const T*)index->GetSample(indices[j]);
                        for (DimensionType k = 0; k < index->GetFeatureDim(); k++)
                        {
                            float dist = v[k] - Mean[k];
                            Variance[k].Dist += dist*dist;
                        }
                    }
                    std::sort(Variance.begin(), Variance.end(), COMMON::Compare);
                    std::vector<SizeType> indexs(m_numTopDimensionTPTSplit);
                    std::vector<float> weight(m_numTopDimensionTPTSplit), bestweight(m_numTopDimensionTPTSplit);
                    float bestvariance = Variance[index->GetFeatureDim() - 1].Dist;
                    for (int i = 0; i < m_numTopDimensionTPTSplit; i++)
                    {
                        indexs[i] = Variance[index->GetFeatureDim() - 1 - i].VID;
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
                            const T* v = (const T*)index->GetSample(indices[first + j]);
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
                        const T* v = (const T*)index->GetSample(indices[i]);
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

                    Mean.clear();
                    Variance.clear();
                    Val.clear();
                    indexs.clear();
                    weight.clear();
                    bestweight.clear();

                    PartitionByTptree<T>(index, indices, first, i - 1, leaves);
                    PartitionByTptree<T>(index, indices, i, last, leaves);
                }
            }

            inline std::uint64_t BufferSize() const
            {
                return m_pNeighborhoodGraph.BufferSize();
            }

            bool LoadGraph(std::string sGraphFilename)
            {
                if (!m_pNeighborhoodGraph.Load(sGraphFilename)) return false;

                m_iGraphSize = m_pNeighborhoodGraph.R();
                m_iNeighborhoodSize = m_pNeighborhoodGraph.C();
                return true;
            }
            
            bool LoadGraph(char* pGraphMemFile)
            {
                m_pNeighborhoodGraph.Load(pGraphMemFile);

                m_iGraphSize = m_pNeighborhoodGraph.R();
                m_iNeighborhoodSize = m_pNeighborhoodGraph.C();
                return true;
            }
            
            bool SaveGraph(std::string sGraphFilename) const
            {
                std::cout << "Save " << m_pNeighborhoodGraph.Name() << " To " << sGraphFilename << std::endl;
                std::ofstream output(sGraphFilename, std::ios::binary);
                if (!output.is_open()) return false;
                SaveGraph(output);
                output.close();
                return true;
            }

            bool SaveGraph(std::ostream& output) const
            {
                output.write((char*)&m_iGraphSize, sizeof(SizeType));
                output.write((char*)&m_iNeighborhoodSize, sizeof(DimensionType));

                for (SizeType i = 0; i < m_iGraphSize; i++)
                    output.write((char*)m_pNeighborhoodGraph[i], sizeof(SizeType) * m_iNeighborhoodSize);

                std::cout << "Save " << m_pNeighborhoodGraph.Name() << " (" << m_iGraphSize << ", " << m_iNeighborhoodSize << ") Finish!" << std::endl;
                return true;
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
                std::lock_guard<std::mutex> lock(m_dataUpdateLock);
                m_pNeighborhoodGraph[row][col] = val;
            }

            inline void SetR(SizeType rows) { 
                m_pNeighborhoodGraph.SetR(rows); 
                m_iGraphSize = rows;
            }

            inline SizeType R() const { return m_iGraphSize; }

            static std::shared_ptr<NeighborhoodGraph> CreateInstance(std::string type);

        protected:
            // Graph structure
            SizeType m_iGraphSize;
            COMMON::Dataset<SizeType> m_pNeighborhoodGraph;
            std::mutex m_dataUpdateLock;
        public:
            int m_iTPTNumber, m_iTPTLeafSize, m_iSamples, m_numTopDimensionTPTSplit;
            DimensionType m_iNeighborhoodSize;
            int m_iNeighborhoodScale, m_iCEFScale, m_iRefineIter, m_iCEF, m_iMaxCheckForRefineGraph;
        };
    }
}
#endif
