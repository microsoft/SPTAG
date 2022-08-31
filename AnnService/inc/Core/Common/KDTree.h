// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_KDTREE_H_
#define _SPTAG_COMMON_KDTREE_H_

#include <vector>
#include <string>
#include <shared_mutex>

#include "inc/Core/VectorIndex.h"

#include "CommonUtils.h"
#include "QueryResultSet.h"
#include "WorkSpace.h"

namespace SPTAG
{
    namespace COMMON
    {
        // node type for storing KDT
        struct KDTNode
        {
            SizeType left;
            SizeType right;
            DimensionType split_dim;
            float split_value;
        };

        class KDTree
        {
        public:
            KDTree() : m_iTreeNumber(2), m_numTopDimensionKDTSplit(5), m_iSamples(1000), m_lock(new std::shared_timed_mutex), m_pQuantizer(nullptr) {}

            KDTree(const KDTree& other) : m_iTreeNumber(other.m_iTreeNumber),
                m_numTopDimensionKDTSplit(other.m_numTopDimensionKDTSplit),
                m_iSamples(other.m_iSamples), m_lock(new std::shared_timed_mutex), m_pQuantizer(other.m_pQuantizer) {}
            ~KDTree() {}

            inline const KDTNode& operator[](SizeType index) const { return m_pTreeRoots[index]; }
            inline KDTNode& operator[](SizeType index) { return m_pTreeRoots[index]; }

            inline SizeType size() const { return (SizeType)m_pTreeRoots.size(); }

            inline SizeType sizePerTree() const { 
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                return (SizeType)m_pTreeRoots.size() - m_pTreeStart.back(); 
            }

            template <typename T>
            void Rebuild(const Dataset<T>& data, IAbortOperation* abort)
            {
                COMMON::KDTree newTrees(*this);
                newTrees.BuildTrees<T>(data, 1, nullptr, abort);

                std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                m_pTreeRoots.swap(newTrees.m_pTreeRoots);
                m_pTreeStart.swap(newTrees.m_pTreeStart);
            }

            template <typename T>
            void BuildTrees(const Dataset<T>& data, int numOfThreads, std::vector<SizeType>* indices = nullptr, IAbortOperation* abort = nullptr)
            {
                if (m_pQuantizer)
                {
                    switch (m_pQuantizer->GetReconstructType())
                    {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
BuildTreesCore<T, Type>(data, numOfThreads, indices, abort); \
break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                    default: break;
                    }
                }
                else
                {
                    BuildTreesCore<T, T>(data, numOfThreads, indices, abort);
                }
            }

            template <typename T, typename R>
            void BuildTreesCore(const Dataset<T>& data, int numOfThreads, std::vector<SizeType>* indices = nullptr, IAbortOperation* abort = nullptr)
            {
                std::vector<SizeType> localindices;
                if (indices == nullptr) {
                    localindices.resize(data.R());
                    for (SizeType i = 0; i < localindices.size(); i++) localindices[i] = i;
                }
                else {
                    localindices.assign(indices->begin(), indices->end());
                }

                m_pTreeRoots.resize(m_iTreeNumber * localindices.size());
                m_pTreeStart.resize(m_iTreeNumber, 0);
#pragma omp parallel for num_threads(numOfThreads)
                for (int i = 0; i < m_iTreeNumber; i++)
                {
                    if (abort && abort->ShouldAbort()) continue;

                    Sleep(i * 100); std::srand(clock());

                    std::vector<SizeType> pindices(localindices.begin(), localindices.end());
                    std::shuffle(pindices.begin(), pindices.end(), rg);

                    m_pTreeStart[i] = i * (SizeType)pindices.size();
                    LOG(Helper::LogLevel::LL_Info, "Start to build KDTree %d\n", i + 1);
                    SizeType iTreeSize = m_pTreeStart[i];
                    DivideTree<T, R>(data, pindices, 0, (SizeType)pindices.size() - 1, m_pTreeStart[i], iTreeSize, abort);
                    LOG(Helper::LogLevel::LL_Info, "%d KDTree built, %d %zu\n", i + 1, iTreeSize - m_pTreeStart[i], pindices.size());
                }
            }

            inline std::uint64_t BufferSize() const 
            { 
                return sizeof(int) + sizeof(SizeType) * m_iTreeNumber + 
                    sizeof(SizeType) + sizeof(KDTNode) * m_pTreeRoots.size();
            }

            ErrorCode SaveTrees(std::shared_ptr<Helper::DiskIO> p_out) const
            {
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                IOBINARY(p_out, WriteBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
                IOBINARY(p_out, WriteBinary, sizeof(SizeType) * m_iTreeNumber, (char*)m_pTreeStart.data());
                SizeType treeNodeSize = (SizeType)m_pTreeRoots.size();
                IOBINARY(p_out, WriteBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
                IOBINARY(p_out, WriteBinary, sizeof(KDTNode) * treeNodeSize, (char*)m_pTreeRoots.data());
                LOG(Helper::LogLevel::LL_Info, "Save KDT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode SaveTrees(std::string sTreeFileName) const
            {
                LOG(Helper::LogLevel::LL_Info, "Save KDT to %s\n", sTreeFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return SaveTrees(ptr);
            }

            ErrorCode LoadTrees(char*  pKDTMemFile)
            {
                m_iTreeNumber = *((int*)pKDTMemFile);
                pKDTMemFile += sizeof(int);
                m_pTreeStart.resize(m_iTreeNumber);
                memcpy(m_pTreeStart.data(), pKDTMemFile, sizeof(SizeType) * m_iTreeNumber);
                pKDTMemFile += sizeof(SizeType)*m_iTreeNumber;

                SizeType treeNodeSize = *((SizeType*)pKDTMemFile);
                pKDTMemFile += sizeof(SizeType);
                m_pTreeRoots.resize(treeNodeSize);
                memcpy(m_pTreeRoots.data(), pKDTMemFile, sizeof(KDTNode) * treeNodeSize);
                LOG(Helper::LogLevel::LL_Info, "Load KDT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode LoadTrees(std::shared_ptr<Helper::DiskIO> p_input)
            {
                if (m_bOldVersion) {
                    struct KdtreeNode
                    {
                        int left;
                        int right;
                        short split_dim;
                        float split_value;
                    } tmpNode;

                    IOBINARY(p_input, ReadBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);

                    int treeNodeSize = 0;
                    for (int i = 0; i < m_iTreeNumber; i++)
                    {
                        m_pTreeStart.push_back(treeNodeSize);

                        int iNodeSize;
                        IOBINARY(p_input, ReadBinary, sizeof(iNodeSize), (char*)&iNodeSize);
                        m_pTreeRoots.resize(treeNodeSize + iNodeSize);
                        for (int j = treeNodeSize; j < treeNodeSize + iNodeSize; j++) {
                            IOBINARY(p_input, ReadBinary, sizeof(KdtreeNode), (char*)(&tmpNode));
                            m_pTreeRoots[j].left = tmpNode.left + treeNodeSize;
                            m_pTreeRoots[j].right = tmpNode.right + treeNodeSize;
                            m_pTreeRoots[j].split_dim = tmpNode.split_dim;
                            m_pTreeRoots[j].split_value = tmpNode.split_value;
                        }
                        treeNodeSize += iNodeSize;
                    }
                    LOG(Helper::LogLevel::LL_Info, "Load KDT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                    return ErrorCode::Success;
                }
                IOBINARY(p_input, ReadBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
                m_pTreeStart.resize(m_iTreeNumber);
                IOBINARY(p_input, ReadBinary, sizeof(SizeType) * m_iTreeNumber, (char*)m_pTreeStart.data());

                SizeType treeNodeSize;
                IOBINARY(p_input, ReadBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
                m_pTreeRoots.resize(treeNodeSize);
                IOBINARY(p_input, ReadBinary, sizeof(KDTNode) * treeNodeSize, (char*)m_pTreeRoots.data());

                LOG(Helper::LogLevel::LL_Info, "Load KDT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode LoadTrees(std::string sTreeFileName)
            {
                LOG(Helper::LogLevel::LL_Info, "Load KDT From %s\n", sTreeFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return LoadTrees(ptr);
            }

            template <typename T, typename Q>
            void InitSearchTrees(const Dataset<T>& p_data, std::function<float(const T*, const T*, DimensionType)> fComputeDistance, COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space) const
            {
                for (int i = 0; i < m_iTreeNumber; i++) {
                    KDTSearch<T, Q>(p_data, fComputeDistance, p_query, p_space, m_pTreeStart[i], 0);
                }
            }

            template <typename T, typename Q>
            void SearchTrees(const Dataset<T>& p_data, std::function<float(const T*, const T*, DimensionType)> fComputeDistance, COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, const int p_limits) const
            {
                while (!p_space.m_SPTQueue.empty() && p_space.m_iNumberOfCheckedLeaves < p_limits)
                {
                    auto& tcell = p_space.m_SPTQueue.pop();
                    KDTSearch<T, Q>(p_data, fComputeDistance, p_query, p_space, tcell.node, tcell.distance);
                }
            }

        private:

            template <typename T, typename Q>
            void KDTSearch(const Dataset<T>& p_data, std::function<float(const T*, const T*, DimensionType)> fComputeDistance, COMMON::QueryResultSet<T> &p_query,
                           COMMON::WorkSpace& p_space, const SizeType node, const float distBound) const {
                if (node < 0)
                {
                    SizeType index = -node - 1;
                    if (index >= p_data.R()) return;
#ifdef PREFETCH
                    const T* data = p_data[index];
                    _mm_prefetch((const char*)data, _MM_HINT_T0);
                    _mm_prefetch((const char*)(data + 64), _MM_HINT_T0);
#endif
                    if (p_space.CheckAndSet(index)) return;

                    ++p_space.m_iNumberOfTreeCheckedLeaves;
                    ++p_space.m_iNumberOfCheckedLeaves;
                    p_space.m_NGQueue.insert(NodeDistPair(index, fComputeDistance(p_query.GetQuantizedTarget(), data, p_data.C())));
                    return;
                }

                auto& tnode = m_pTreeRoots[node];

                float diff = ((Q*) p_query.GetTarget())[tnode.split_dim] - tnode.split_value;
                float distanceBound = distBound + diff * diff;
                SizeType otherChild, bestChild;
                if (diff < 0)
                {
                    bestChild = tnode.left;
                    otherChild = tnode.right;
                }
                else
                {
                    otherChild = tnode.left;
                    bestChild = tnode.right;
                }

                p_space.m_SPTQueue.insert(NodeDistPair(otherChild, distanceBound));
                KDTSearch<T,Q>(p_data, fComputeDistance, p_query, p_space, bestChild, distBound);
            }


            template <typename T, typename R>
            void DivideTree(const Dataset<T>& data, std::vector<SizeType>& indices, SizeType first, SizeType last,
                SizeType index, SizeType &iTreeSize, IAbortOperation* abort = nullptr) {
                if (abort && abort->ShouldAbort()) return;

                ChooseDivision<T, R>(data, m_pTreeRoots[index], indices, first, last);
                SizeType i = Subdivide<T, R>(data, m_pTreeRoots[index], indices, first, last);
                if (i - 1 <= first)
                {
                    m_pTreeRoots[index].left = -indices[first] - 1;
                }
                else
                {
                    iTreeSize++;
                    m_pTreeRoots[index].left = iTreeSize;
                    DivideTree<T, R>(data, indices, first, i - 1, iTreeSize, iTreeSize);
                }
                if (last == i)
                {
                    m_pTreeRoots[index].right = -indices[last] - 1;
                }
                else
                {
                    iTreeSize++;
                    m_pTreeRoots[index].right = iTreeSize;
                    DivideTree<T, R>(data, indices, i, last, iTreeSize, iTreeSize);
                }
            }

            template <typename T, typename R>
            void ChooseDivision(const Dataset<T>& data, KDTNode& node, const std::vector<SizeType>& indices, const SizeType first, const SizeType last)
            {
                SizeType cols = data.C();
                bool quantizer_exists = (nullptr != m_pQuantizer);
                R* v_holder = nullptr;
                if (quantizer_exists)
                {
                    cols = m_pQuantizer->ReconstructDim();
                    v_holder = (R*)ALIGN_ALLOC(m_pQuantizer->ReconstructSize());
                }
                std::vector<float> meanValues(cols, 0);
                std::vector<float> varianceValues(cols, 0);
                SizeType end = min(first + m_iSamples, last);
                SizeType count = end - first + 1;
                // calculate the mean of each dimension
                for (SizeType j = first; j <= end; j++)
                {
                    R* v;
                    if (quantizer_exists)
                    {
                        m_pQuantizer->ReconstructVector((uint8_t*)data[indices[j]], v_holder);
                        v = v_holder;
                    } 
                    else
                    {
                        v = (R*)data[indices[j]];
                    }
                    for (DimensionType k = 0; k < cols; k++)
                    {
                        meanValues[k] += v[k];
                    }
                }
                for (DimensionType k = 0; k < cols; k++)
                {
                    meanValues[k] /= count;
                }
                // calculate the variance of each dimension
                for (SizeType j = first; j <= end; j++)
                {
                    R* v;
                    if (quantizer_exists)
                    {
                        m_pQuantizer->ReconstructVector((uint8_t*)data[indices[j]], v_holder);
                        v = v_holder;
                    }
                    else
                    {
                        v = (R*)data[indices[j]];
                    }
                    for (DimensionType k = 0; k < cols; k++)
                    {
                        float dist = v[k] - meanValues[k];
                        varianceValues[k] += dist*dist;
                    }
                }
                if (quantizer_exists)
                {
                    ALIGN_FREE(v_holder);
                }
                // choose the split dimension as one of the dimension inside TOP_DIM maximum variance
                node.split_dim = SelectDivisionDimension(varianceValues);
                // determine the threshold
                node.split_value = meanValues[node.split_dim];
            }

            DimensionType SelectDivisionDimension(const std::vector<float>& varianceValues) const
            {
                // Record the top maximum variances
                std::vector<DimensionType> topind(m_numTopDimensionKDTSplit);
                int num = 0;
                // order the variances
                for (DimensionType i = 0; i < (DimensionType)varianceValues.size(); i++)
                {
                    if (num < m_numTopDimensionKDTSplit || varianceValues[i] > varianceValues[topind[num - 1]])
                    {
                        if (num < m_numTopDimensionKDTSplit)
                        {
                            topind[num++] = i;
                        }
                        else
                        {
                            topind[num - 1] = i;
                        }
                        int j = num - 1;
                        // order the TOP_DIM variances
                        while (j > 0 && varianceValues[topind[j]] > varianceValues[topind[j - 1]])
                        {
                            std::swap(topind[j], topind[j - 1]);
                            j--;
                        }
                    }
                }
                // randomly choose a dimension from TOP_DIM
                return topind[COMMON::Utils::rand(num)];
            }

            template <typename T, typename R>
            SizeType Subdivide(const Dataset<T>& data, const KDTNode& node, std::vector<SizeType>& indices, const SizeType first, const SizeType last) const
            {
                SizeType i = first;
                SizeType j = last;
                bool quantizer_exists = (m_pQuantizer != nullptr);
                R* v_holder = nullptr;
                if (quantizer_exists)
                {
                    v_holder = (R*)ALIGN_ALLOC(m_pQuantizer->ReconstructSize());
                }
                // decide which child one point belongs
                while (i <= j)
                {
                    R* v;
                    SizeType ind = indices[i];
                    if (quantizer_exists)
                    {
                        m_pQuantizer->ReconstructVector((uint8_t*)data[ind], v_holder);
                        v = v_holder;
                    } 
                    else
                    {
                        v = (R*)data[ind];
                    }
                    float val = v[node.split_dim];
                    if (val < node.split_value)
                    {
                        i++;
                    }
                    else
                    {
                        std::swap(indices[i], indices[j]);
                        j--;
                    }
                }
                if (quantizer_exists)
                {
                    ALIGN_FREE(v_holder);
                }
                // if all the points in the node are equal,equally split the node into 2
                if ((i == first) || (i == last + 1))
                {
                    i = (first + last + 1) / 2;
                }
                return i;
            }

        private:
            std::vector<SizeType> m_pTreeStart;
            std::vector<KDTNode> m_pTreeRoots;

        public:
            std::unique_ptr<std::shared_timed_mutex> m_lock;
            int m_iTreeNumber, m_numTopDimensionKDTSplit, m_iSamples;
            bool m_bOldVersion;
            std::shared_ptr<SPTAG::COMMON::IQuantizer> m_pQuantizer;
        };
    }
}
#endif
