#include "inc/Core/KDT/Index.h"
#include "inc/Core/Common/WorkSpacePool.h"
#include "inc/Core/MetadataSet.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/SimpleIniReader.h"

#pragma warning(disable:4996)  // 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
#pragma warning(disable:4242)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4244)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4127)  // conditional expression is constant

namespace SPTAG
{
    namespace KDT
    {
#pragma region Load data points, kd-tree, neighborhood graph
        template <typename T>
        bool Index<T>::LoadIndex()
        {
            bool loadedDataPoints = m_pDataPointsMemoryFile != NULL ? LoadDataPoints(m_pDataPointsMemoryFile) : LoadDataPoints(m_sDataPointsFilename);
            if (!loadedDataPoints) return false;

            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();

            bool loadedKDT = m_pKDTMemoryFile != NULL ? LoadKDT(m_pKDTMemoryFile) : LoadKDT(m_sKDTFilename);
            if (!loadedKDT) return false;

            bool loadedGraph = m_pGraphMemoryFile != NULL ? LoadGraph(m_pGraphMemoryFile) : LoadGraph(m_sGraphFilename);
            if (!loadedGraph) return false;

            if (m_iRefineIter > 0) {
                for (int i = 0; i < m_iRefineIter; i++) RefineRNG();
                SaveRNG(m_sGraphFilename);
            }
            return true;
        }

        template <typename T>
        bool Index<T>::LoadDataPoints(std::string sDataPointsFileName)
        {
            std::cout << "Load Data Points From " << sDataPointsFileName << std::endl;
            if (m_iCacheSize >= 0)
                m_pSamples.Initialize(0, 0, nullptr, sDataPointsFileName.c_str(), m_iCacheSize);
            else
            {
                FILE * fp = fopen(sDataPointsFileName.c_str(), "rb");
                if (fp == NULL) return false;

                int R, C;
                fread(&R, sizeof(int), 1, fp);
                fread(&C, sizeof(int), 1, fp);

                if (m_iDebugLoad > 0 && R > m_iDebugLoad) R = m_iDebugLoad;

                m_pSamples.Initialize(R, C);
                int i = 0, batch = 10000;
                while (i + batch < R) {
                    fread((m_pSamples)[i], sizeof(T), C * batch, fp);
                    i += batch;
                }
                fread((m_pSamples)[i], sizeof(T), C * (R - i), fp);
                fclose(fp);
            }
            std::cout << "Load Data Points (" << m_pSamples.R() << ", " << m_pSamples.C() << ") Finish!" << std::endl;
            return true;
        }

        // Functions for loading models from memory mapped files
        template <typename T>
        bool Index<T>::LoadDataPoints(char* pDataPointsMemFile)
        {
            int R, C;
            R = *((int*)pDataPointsMemFile);
            pDataPointsMemFile += sizeof(int);

            C = *((int*)pDataPointsMemFile);
            pDataPointsMemFile += sizeof(int);

            m_pSamples.Initialize(R, C, (T*)pDataPointsMemFile);

            return true;
        }

        template <typename T>
        bool Index<T>::LoadKDT(std::string sKDTFilename)
        {
            std::cout << "Load KDT From " << sKDTFilename << std::endl;
            FILE *fp = fopen(sKDTFilename.c_str(), "rb");
            if (fp == NULL) return false;
            int realKDTNumber;
            fread(&realKDTNumber, sizeof(int), 1, fp);
            if (realKDTNumber < m_iKDTNumber) m_iKDTNumber = realKDTNumber;
            for (int i = 0; i < m_iKDTNumber; i++) {
                m_pKDTStart.push_back((int)(m_pKDTRoots.size()));
                int treeNodeSize;
                fread(&treeNodeSize, sizeof(int), 1, fp);
                m_pKDTRoots.resize(m_pKDTRoots.size() + treeNodeSize);
                fread(&(m_pKDTRoots[m_pKDTStart[i]]), sizeof(KDTNode), treeNodeSize, fp);
            }
            m_pKDTStart.push_back((int)(m_pKDTRoots.size()));
            fclose(fp);
            std::cout << "Load KDT (" << m_iKDTNumber << ", " << m_pKDTRoots.size() << ") Finish!" << std::endl;
            return true;
        }

        template <typename T>
        bool Index<T>::LoadKDT(char* pKDTMemFile)
        {
            int realKDTNumber = *((int*)pKDTMemFile);
            pKDTMemFile += sizeof(int);
            if (realKDTNumber < m_iKDTNumber) m_iKDTNumber = realKDTNumber;
            for (int i = 0; i < m_iKDTNumber; i++) {
                m_pKDTStart.push_back((int)(m_pKDTRoots.size()));

                int treeNodeSize = *((int*)pKDTMemFile);
                pKDTMemFile += sizeof(int);
                m_pKDTRoots.resize(m_pKDTRoots.size() + treeNodeSize);
                std::memcpy(&(m_pKDTRoots[m_pKDTStart[i]]), pKDTMemFile, sizeof(KDTNode)*treeNodeSize);
                pKDTMemFile += sizeof(KDTNode)*treeNodeSize;
            }
            m_pKDTStart.push_back((int)(m_pKDTRoots.size()));
            return true;
        }

        template <typename T>
        bool Index<T>::LoadGraph(std::string sGraphFilename)
        {
            std::cout << "Load Graph From " << sGraphFilename << std::endl;
            FILE * fp = fopen(sGraphFilename.c_str(), "rb");
            if (fp == NULL) return false;
            fread(&m_iGraphSize, sizeof(int), 1, fp);
            int KNNinGraph;
            fread(&KNNinGraph, sizeof(int), 1, fp);
            if (KNNinGraph < m_iNeighborhoodSize) m_iNeighborhoodSize = KNNinGraph;

            m_pNeighborhoodGraph.Initialize(m_iGraphSize, m_iNeighborhoodSize);

            std::vector<int> unusedData(KNNinGraph);
            for (int i = 0; i < m_iGraphSize; i++)
            {
                fread((m_pNeighborhoodGraph)[i], sizeof(int), m_iNeighborhoodSize, fp);
                if (m_iNeighborhoodSize < KNNinGraph)
                {
                    fread(&unusedData[0], sizeof(int), KNNinGraph - m_iNeighborhoodSize, fp);
                }
            }
            fclose(fp);
            std::cout << "Load Graph (" << m_iGraphSize << "," << m_iNeighborhoodSize << ") Finish!" << std::endl;
            return true;
        }

        template <typename T>
        bool Index<T>::LoadGraph(char* pGraphMemFile) {
            m_iGraphSize = *((int*)pGraphMemFile);
            pGraphMemFile += sizeof(int);

            int KNNinGraph = *((int*)pGraphMemFile);
            pGraphMemFile += sizeof(int);

            // In the memory mapped file mode, we'll not accept NeighborhoodSize in graph file that's larger than expected size (m_iNeighborhoodSize)
            // as we don't want to make another copy to fit.
            if (KNNinGraph > m_iNeighborhoodSize) return false;

            if (KNNinGraph < m_iNeighborhoodSize) m_iNeighborhoodSize = KNNinGraph;

            m_pNeighborhoodGraph.Initialize(m_iGraphSize, m_iNeighborhoodSize, (int*)pGraphMemFile);

            return true;
        }
#pragma endregion

#pragma region K-NN search
        template <typename T>
        void Index<T>::KDTSearch(const int node, const bool isInit, const float distBound,
            COMMON::WorkSpace& space, COMMON::QueryResultSet<T> &query) const {
            if (node < 0)
            {
                int index = -node - 1;

#ifdef PREFETCH
                const char* data = (const char *)(m_pSamples[index]);
                _mm_prefetch(data, _MM_HINT_T0);
                _mm_prefetch(data + 64, _MM_HINT_T0);
#endif
                if (space.CheckAndSet(index)) return;

                float distance = m_fComputeDistance(query.GetTarget(), (T*)data, m_iDataDimension);
                query.AddPoint(index, distance);
                ++space.m_iNumberOfTreeCheckedLeaves;
                ++space.m_iNumberOfCheckedLeaves;
                space.m_NGQueue.insert(COMMON::HeapCell(index, distance));
                return;
            }

            auto& tnode = m_pKDTRoots[node];

            float diff = (query.GetTarget())[tnode.split_dim] - tnode.split_value;
            float distanceBound = distBound + diff * diff;
            int otherChild, bestChild;
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

            if (!isInit || distanceBound < query.worstDist())
            {
                space.m_SPTQueue.insert(COMMON::HeapCell(otherChild, distanceBound));
            }
            KDTSearch(bestChild, isInit, distBound, space, query);
        }

        template <typename T>
        void Index<T>::SearchIndex(COMMON::QueryResultSet<T> &query, COMMON::WorkSpace &space) const
        {
            for (char i = 0; i < m_iKDTNumber; i++) {
                KDTSearch(m_pKDTStart[i], true, 0, space, query);
            }

            while (!space.m_SPTQueue.empty() && space.m_iNumberOfCheckedLeaves < g_iNumberOfInitialDynamicPivots)
            {
                auto& tcell = space.m_SPTQueue.pop();
                if (query.worstDist() < tcell.distance) break;
                KDTSearch(tcell.node, true, tcell.distance, space, query);
            }

            while (!space.m_NGQueue.empty() && space.m_iNumberOfCheckedLeaves < space.m_iMaxCheck) {
                bool bLocalOpt = true;
                COMMON::HeapCell gnode = space.m_NGQueue.pop();
                const int *node = (m_pNeighborhoodGraph)[gnode.node];

#ifdef PREFETCH
                _mm_prefetch((const char *)node, _MM_HINT_T0);
                for (int i = 0; i < m_iNeighborhoodSize; i++)
                {
                    _mm_prefetch((const char *)(m_pSamples)[node[i]], _MM_HINT_T0);
                }
#endif

                for (int i = 0; i < m_iNeighborhoodSize; i++)
                {
                    int nn_index = node[i];

                    // do not check it if it has been checked
                    if (nn_index < 0) break;
                    if (space.CheckAndSet(nn_index)) continue;

                    // count the number of the computed nodes
                    float distance2leaf = m_fComputeDistance(query.GetTarget(), (m_pSamples)[nn_index], m_iDataDimension);
                    
                    bool inserted = query.AddPoint(nn_index, distance2leaf);
                    space.m_iNumberOfCheckedLeaves++;
                    space.m_NGQueue.insert(COMMON::HeapCell(nn_index, distance2leaf));
                    if (inserted || distance2leaf < gnode.distance) bLocalOpt = false;
                }

                if (bLocalOpt) space.m_iNumOfContinuousNoBetterPropagation++;
                else space.m_iNumOfContinuousNoBetterPropagation = 0;

                if (space.m_iNumOfContinuousNoBetterPropagation > g_iThresholdOfNumberOfContinuousNoBetterPropagation)
                {
                    if (space.m_iNumberOfTreeCheckedLeaves < space.m_iNumberOfCheckedLeaves / 10)
                    {
                        int nextNumberOfCheckedLeaves = g_iNumberOfOtherDynamicPivots + space.m_iNumberOfCheckedLeaves;
                        while (!space.m_SPTQueue.empty() && space.m_iNumberOfCheckedLeaves < nextNumberOfCheckedLeaves)
                        {
                            auto& tcell = space.m_SPTQueue.pop();
                            KDTSearch(tcell.node, false, tcell.distance, space, query);
                        }
                    }
                    else if (gnode.distance > query.worstDist()) {
                        break;
                    }
                }
            }
        }

        template<typename T>
        ErrorCode
            Index<T>::SearchIndex(QueryResult &query) const
        {
            auto workSpace = m_workSpacePool->Rent();
            workSpace->Reset(m_iMaxCheck);

            SearchIndex(*((COMMON::QueryResultSet<T>*)&query), *workSpace);
            m_workSpacePool->Return(workSpace);

            if (query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < query.GetResultNum(); ++i)
                {
                    query.SetMetadata(i, m_pMetadata->GetMetadata(query.GetResult(i)->Key));
                }
            }

            return ErrorCode::Success;
        }

#pragma endregion

#pragma region Build/Save kd-tree & neighborhood graphs
        template <typename T>
        bool Index<T>::BuildIndex()
        {
            if (!LoadDataPoints(m_sDataPointsFilename.c_str())) return false;
            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();

            BuildKDT();
            if (!SaveKDT(m_sKDTFilename)) return false;

            BuildRNG();
            if (!SaveRNG(m_sGraphFilename)) return false;
            return true;
        }

        template <typename T>
        bool Index<T>::BuildIndex(void* p_data, int p_vectorNum, int p_dimension)
        {
            m_pSamples.Initialize(p_vectorNum, p_dimension, reinterpret_cast<T*>(p_data));
            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();

            BuildKDT();
            BuildRNG();
            return true;
        }

        template<typename T>
        ErrorCode Index<T>::BuildIndex(std::shared_ptr<VectorSet> p_vectorSet,
            std::shared_ptr<MetadataSet> p_metadataSet)
        {
            if (nullptr == p_vectorSet || p_vectorSet->Count() == 0 || p_vectorSet->Dimension() == 0 || p_vectorSet->ValueType() != GetEnumValueType<T>())
            {
                return ErrorCode::Fail;
            }

            if (DistCalcMethod::Cosine == m_iDistCalcMethod)
            {
                m_pSamples.Initialize(p_vectorSet->Count(), p_vectorSet->Dimension());
                std::memcpy(m_pSamples.GetData(), p_vectorSet->GetData(), p_vectorSet->Count() * p_vectorSet->Dimension() * sizeof(T));

                int base = COMMON::Utils::GetBase<T>();
                for (SPTAG::SizeType i = 0; i < p_vectorSet->Count(); i++) {
                    COMMON::Utils::Normalize(m_pSamples[i], m_pSamples.C(), base);
                }
                p_vectorSet.reset();
            }
            else {
                m_pSamples.Initialize(p_vectorSet->Count(), p_vectorSet->Dimension(), reinterpret_cast<T*>(p_vectorSet->GetData()));
            }

            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();

            BuildKDT();
            BuildRNG();

            m_pMetadata = std::move(p_metadataSet);

            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_iMaxCheck, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);
            return ErrorCode::Success;
        }

#pragma region Build/Save kd-tree
        template <typename T>
        bool Index<T>::SaveKDT(std::string sKDTFilename) const
        {
            std::cout << "Save KDT to " << sKDTFilename << std::endl;
            FILE *fp = fopen(sKDTFilename.c_str(), "wb");
            if (fp == NULL) return false;
            fwrite(&m_iKDTNumber, sizeof(int), 1, fp);
            for (int i = 0; i < m_iKDTNumber; i++)
            {
                int treeNodeSize = m_pKDTStart[i + 1] - m_pKDTStart[i];
                fwrite(&treeNodeSize, sizeof(int), 1, fp);
                fwrite(&(m_pKDTRoots[m_pKDTStart[i]]), sizeof(KDTNode), treeNodeSize, fp);
            }
            fclose(fp);
            std::cout << "Save KDT Finish!" << std::endl;
            return true;
        }

        template <typename T>
        void Index<T>::BuildKDT()
        {
            omp_set_num_threads(m_iNumberOfThreads);
            m_pKDTRoots.resize(m_iKDTNumber * m_iDataSize);
            m_pKDTStart.resize(m_iKDTNumber + 1, (int)(m_pKDTRoots.size()));
            
#pragma omp parallel for
            for (int i = 0; i < m_iKDTNumber; i++)
            {
                Sleep(i * 100); std::srand(clock());
                m_pKDTStart[i] = i * m_iDataSize;
                std::vector<int> KDTDataIndices(m_iDataSize);
                for (int j = 0; j < m_iDataSize; j++) KDTDataIndices[j] = j;
                std::random_shuffle(KDTDataIndices.begin(), KDTDataIndices.end());

                std::cout << "Start to build tree " << i + 1 << std::endl;
                int iTreeSize = m_pKDTStart[i];
                DivideTree(m_pKDTRoots.data(), KDTDataIndices, 0, m_iDataSize - 1, m_pKDTStart[i], iTreeSize);
                std::cout << i + 1 << " trees built, " << iTreeSize - m_pKDTStart[i] << " " << m_iDataSize << std::endl;
            }
        }

        template <typename T>
        void Index<T>::DivideTree(KDTNode* pTree, std::vector<int>& indices, int first, int last,
            int index, int &iTreeSize) {
            ChooseDivision(pTree[index], indices, first, last);
            int i = Subdivide(pTree[index], indices, first, last);

            if (i - 1 == first)
            {
                pTree[index].left = -indices[first] - 1;
            }
            else
            {
                iTreeSize++;
                pTree[index].left = iTreeSize;
                DivideTree(pTree, indices, first, i - 1, iTreeSize, iTreeSize);
            }
            if (last == i)
            {
                pTree[index].right = -indices[last] - 1;
            }
            else
            {
                iTreeSize++;
                pTree[index].right = iTreeSize;
                DivideTree(pTree, indices, i, last, iTreeSize, iTreeSize);
            }
        }

        template <typename T>
        void Index<T>::ChooseDivision(KDTNode& node, const std::vector<int>& indices, int first, int last)
        {
            std::vector<float> meanValues(m_iDataDimension, 0);
            std::vector<float> varianceValues(m_iDataDimension, 0);
            int end = min(first + m_numSamplesKDTSplitConsideration, last);
            int count = end - first + 1;
            // calculate the mean of each dimension
            for (int j = first; j <= end; j++)
            {
                T* v = (m_pSamples)[indices[j]];
                for (int k = 0; k < m_iDataDimension; k++)
                {
                    meanValues[k] += v[k];
                }
            }
            for (int k = 0; k < m_iDataDimension; k++)
            {
                meanValues[k] /= count;
            }
            // calculate the variance of each dimension
            for (int j = first; j <= end; j++)
            {
                T* v = (m_pSamples)[indices[j]];
                for (int k = 0; k < m_iDataDimension; k++)
                {
                    float dist = v[k] - meanValues[k];
                    varianceValues[k] += dist*dist;
                }
            }
            // choose the split dimension as one of the dimension inside TOP_DIM maximum variance
            node.split_dim = SelectDivisionDimension(varianceValues);
            // determine the threshold
            node.split_value = meanValues[node.split_dim];
        }

        template <typename T>
        int Index<T>::SelectDivisionDimension(const std::vector<float>& varianceValues) const
        {
            // Record the top maximum variances
            std::vector<int> topind(m_numTopDimensionKDTSplit);
            int num = 0;
            // order the variances
            for (int i = 0; i < m_iDataDimension; i++)
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
            return topind[COMMON::Utils::rand_int(num)];
        }

        template <typename T>
        int Index<T>::Subdivide(const KDTNode& node, std::vector<int>& indices, const int first, const int last)
        {
            int i = first;
            int j = last;
            // decide which child one point belongs
            while (i <= j)
            {
                int ind = indices[i];
                float val = (m_pSamples)[ind][node.split_dim];
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
            // if all the points in the node are equal,equally split the node into 2
            if ((i == first) || (i == last + 1))
            {
                i = (first + last + 1) / 2;
            }
            return i;
        }
#pragma endregion

#pragma region Build/Save neighborhood graph
        template <typename T>
        bool Index<T>::SaveRNG(std::string sGraphFilename) const
        {
            std::cout << "Save Graph To " << sGraphFilename << std::endl;
            FILE *fp = fopen(sGraphFilename.c_str(), "wb");
            if (fp == NULL) return false;
            fwrite(&m_iGraphSize, sizeof(int), 1, fp);
            fwrite(&m_iNeighborhoodSize, sizeof(int), 1, fp);

            for (int i = 0; i < m_iGraphSize; i++)
            {
                fwrite((m_pNeighborhoodGraph)[i], sizeof(int), m_iNeighborhoodSize, fp);
            }
            fclose(fp);
            std::cout << "Save Graph Finish!" << std::endl;
            return true;
        }

        template <typename T>
        void Index<T>::PartitionByTptree(std::vector<int>& indices,
            const int first,
            const int last,
            std::vector<std::pair<int, int>> & leaves)
        {
            if (last - first <= m_iTPTLeafSize)
            {
                leaves.push_back(std::make_pair(first, last));
            }
            else
            {
                std::vector<float> Mean(m_iDataDimension, 0);

                int iIteration = 100;
                int end = min(first + m_numSamplesTPTSplitConsideration, last);
                int count = end - first + 1;
                // calculate the mean of each dimension
                for (int j = first; j <= end; j++)
                {
                    T* v = (m_pSamples)[indices[j]];
                    for (int k = 0; k < m_iDataDimension; k++)
                    {
                        Mean[k] += v[k];
                    }
                }
                for (int k = 0; k < m_iDataDimension; k++)
                {
                    Mean[k] /= count;
                }
                std::vector<BasicResult> Variance;
                Variance.reserve(m_iDataDimension);
                for (int j = 0; j < m_iDataDimension; j++)
                {
                    Variance.push_back(BasicResult(j, 0));
                }
                // calculate the variance of each dimension
                for (int j = first; j <= end; j++)
                {
                    T* v = (m_pSamples)[indices[j]];
                    for (int k = 0; k < m_iDataDimension; k++)
                    {
                        float dist = v[k] - Mean[k];
                        Variance[k].Dist += dist*dist;
                    }
                }
                std::sort(Variance.begin(), Variance.end(),COMMON::Compare);
                std::vector<int> index(m_numTopDimensionTPTSplit);
                std::vector<float> weight(m_numTopDimensionTPTSplit), bestweight(m_numTopDimensionTPTSplit);
                float bestvariance = Variance[m_iDataDimension - 1].Dist;
                for (int i = 0; i < m_numTopDimensionTPTSplit; i++)
                {
                    index[i] = Variance[m_iDataDimension - 1 - i].Key;
                    bestweight[i] = 0;
                }
                bestweight[0] = 1;
                float bestmean = Mean[index[0]];

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
                    for (int j = 0; j < count; j++)
                    {
                        Val[j] = 0;
                        for (int k = 0; k < m_numTopDimensionTPTSplit; k++)
                        {
                            Val[j] += weight[k] * (m_pSamples)[indices[first + j]][index[k]];
                        }
                        mean += Val[j];
                    }
                    mean /= count;
                    float var = 0;
                    for (int j = 0; j < count; j++)
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
                int i = first;
                int j = last;
                // decide which child one point belongs
                while (i <= j)
                {
                    float val = 0;
                    for (int k = 0; k < m_numTopDimensionTPTSplit; k++)
                    {
                        val += bestweight[k] * (m_pSamples)[indices[i]][index[k]];
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
                index.clear();
                weight.clear();
                bestweight.clear();

                PartitionByTptree(indices, first, i - 1, leaves);
                PartitionByTptree(indices, i, last, leaves);
            }
        }

        template <typename T>
        void Index<T>::RefineRNG() {
            std::vector<COMMON::WorkSpace> spaces(m_iNumberOfThreads);
            for (int i = 0; i < m_iNumberOfThreads; i++) spaces[i].Initialize(m_iMaxCheckForRefineGraph, m_iGraphSize);

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < m_iGraphSize; i++)
            {
                RefineRNGNode(i, spaces[omp_get_thread_num()], false);
            }
        }

        template <typename T>
        void Index<T>::BuildRNG()
        {
            std::cout << "build RNG graph!" << std::endl;

            omp_set_num_threads(m_iNumberOfThreads);

            int graphScale = 16;
            int cefScale = 4;
            m_iNeighborhoodSize *= graphScale;
            m_iGraphSize = m_iDataSize;

            m_pNeighborhoodGraph.Initialize(m_iGraphSize, m_iNeighborhoodSize);
            {
                COMMON::Dataset<float> NeighborhoodDists(m_iGraphSize, m_iNeighborhoodSize);
                std::vector<std::vector<int>> TptreeDataIndices(m_iTPTNumber, std::vector<int>(m_iGraphSize));
                std::vector<std::vector<std::pair<int, int>>> TptreeLeafNodes(m_iTPTNumber, std::vector<std::pair<int, int>>());
                for (int i = 0; i < m_iGraphSize; i++)
                {
                    for (int j = 0; j < m_iNeighborhoodSize; j++)
                    {
                        (m_pNeighborhoodGraph)[i][j] = -1;
                        (NeighborhoodDists)[i][j] = MaxDist;
                    }
                    TptreeDataIndices[0][i] = i;
                }
                for (int i = 1; i < m_iTPTNumber; i++) {
                    std::memcpy(TptreeDataIndices[i].data(), TptreeDataIndices[0].data(), sizeof(int) * m_iGraphSize);
                }

                std::cout << "Parallel TpTree Partition begin " << std::endl;
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < m_iTPTNumber; i++)
                {
                    Sleep(i * 100); std::srand(clock());
                    std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
                    PartitionByTptree(TptreeDataIndices[i], 0, m_iGraphSize - 1, TptreeLeafNodes[i]);
                    std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
                }
                std::cout << "Parallel TpTree Partition done" << std::endl;

                for (int i = 0; i < m_iTPTNumber; i++)
                {
#pragma omp parallel for schedule(dynamic)
                    for (int j = 0; j < TptreeLeafNodes[i].size(); j++)
                    {
                        int start_index = TptreeLeafNodes[i][j].first;
                        int end_index = TptreeLeafNodes[i][j].second;
                        if (omp_get_thread_num() == 0) std::cout << "\rProcessing Tree " << i << ' ' << j * 100 / TptreeLeafNodes[i].size() << '%';
                        for (int x = start_index; x < end_index; x++)
                        {
                            for (int y = x + 1; y <= end_index; y++)
                            {
                                int p1 = TptreeDataIndices[i][x];
                                int p2 = TptreeDataIndices[i][y];
                                float dist = m_fComputeDistance((m_pSamples)[p1], (m_pSamples)[p2], m_iDataDimension);
                                COMMON::Utils::AddNeighbor(p2, dist, (m_pNeighborhoodGraph)[p1], (NeighborhoodDists)[p1], m_iNeighborhoodSize);
                                COMMON::Utils::AddNeighbor(p1, dist, (m_pNeighborhoodGraph)[p2], (NeighborhoodDists)[p2], m_iNeighborhoodSize);                            }
                        }
                    }
                    TptreeDataIndices[i].clear();
                    TptreeLeafNodes[i].clear();
                    std::cout << std::endl;
                }
                TptreeDataIndices.clear();
                TptreeLeafNodes.clear();
            }
            std::cout << "NNG acc:" << GraphAccuracyEstimation(100, false) << std::endl;
            if (m_iMaxCheckForRefineGraph > 0) {
                m_iCEF *= cefScale;
                m_iMaxCheckForRefineGraph *= cefScale;
                RefineRNG();
                std::cout << "Refine RNG, graph acc:" << GraphAccuracyEstimation(100, true) << std::endl;

                m_iCEF /= cefScale;
                m_iMaxCheckForRefineGraph /= cefScale;
                m_iNeighborhoodSize /= graphScale;

                //RefineRNG();
                //std::cout << "Refine RNG, graph acc:" << GraphAccuracyEstimation(100, true) << std::endl;
                RefineRNG();
                std::cout << "Refine RNG, graph acc:" << GraphAccuracyEstimation(100, true) << std::endl;
            }
            std::cout << "Build RNG Graph end!" << std::endl;
        }

        template <typename T>
        float Index<T>::GraphAccuracyEstimation(int NSample, bool rng) {
            int* correct = new int[NSample];

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < NSample; i++)
            {
                //int x = Utils::rand_int(m_iGraphSize);
                int x = i;
                COMMON::QueryResultSet<T> query((m_pSamples)[x], m_iCEF);
                for (int y = 0; y < m_iGraphSize; y++)
                {
                    if (y == x) continue;
                    float dist = m_fComputeDistance(query.GetTarget(), (m_pSamples)[y], m_iDataDimension);
                    query.AddPoint(y, dist);
                }
                query.SortResult();
                int * exact_rng = new int[m_iNeighborhoodSize];
                if (rng) {
                    RebuildRNGNodeNeighbors(exact_rng, query.GetResults(), m_iCEF);
                }
                else {
                    for (int j = 0; j < m_iNeighborhoodSize && j < m_iCEF; j++) {
                        exact_rng[j] = query.GetResult(j)->Key;
                    }
                    for (int j = m_iCEF; j < m_iNeighborhoodSize; j++) exact_rng[j] = -1;
                }
                correct[i] = 0;
                for (int j = 0; j < m_iNeighborhoodSize; j++) {
                    if (exact_rng[j] == -1) {
                        correct[i] += m_iNeighborhoodSize - j;
                        break;
                    }
                    for (int k = 0; k < m_iNeighborhoodSize; k++)
                        if ((m_pNeighborhoodGraph)[x][k] == exact_rng[j]) {
                            correct[i]++;
                            break;
                        }
                }
                delete[] exact_rng;
            }
            float acc = 0;
            for (int i = 0; i < NSample; i++) acc += float(correct[i]);
            acc = acc / NSample / m_iNeighborhoodSize;
            delete[] correct;
            return acc;
        }

        template <typename T>
        void Index<T>::AddNodes(const T* pData, int num, COMMON::WorkSpace &space) {
            m_pSamples.AddBatch(pData, num);
            m_pNeighborhoodGraph.AddReserved(num);

            if (m_pSamples.R() != m_iDataSize + num || m_pNeighborhoodGraph.R() != m_iDataSize + num)
                std::cout << "Error m_iDataSize" << std::endl;
            m_iDataSize += num;

            for (int node = m_iDataSize - num; node < m_iDataSize; node++)
            {
                RefineRNGNode(node, space, true);
            }
        }

        template <typename T>
        void Index<T>::RefineRNGNode(const int node, COMMON::WorkSpace &space, bool updateNeighbors) {
            space.Reset(m_iMaxCheckForRefineGraph);
            COMMON::QueryResultSet<T> query((m_pSamples)[node], m_iCEF);
            space.CheckAndSet(node);
            for (int i = 0; i < m_iNeighborhoodSize; i++) {
                int index = m_pNeighborhoodGraph[node][i];
                if (index < 0 || space.CheckAndSet(index)) continue;
                space.m_NGQueue.insert(COMMON::HeapCell(index, m_fComputeDistance(query.GetTarget(), m_pSamples[index], m_iDataDimension)));
            }
            SearchIndex(query, space);
            RebuildRNGNodeNeighbors(m_pNeighborhoodGraph[node], query.GetResults(), m_iCEF);

            if (updateNeighbors) {
                // update neighbors
                for (int j = 0; j < m_iCEF; j++)
                {
                    BasicResult* item = query.GetResult(j);
                    if (item->Key < 0) continue;
                    COMMON::QueryResultSet<T> queryNbs(m_pSamples[item->Key], m_iNeighborhoodSize + 1);
                    queryNbs.AddPoint(node, item->Dist);
                    for (int k = 0; k < m_iNeighborhoodSize; k++)
                    {
                        int tmpNode = m_pNeighborhoodGraph[item->Key][k];
                        if (tmpNode < 0) break;
                        float distance = m_fComputeDistance(queryNbs.GetTarget(), m_pSamples[tmpNode], m_iDataDimension);
                        queryNbs.AddPoint(tmpNode, distance);
                    }
                    queryNbs.SortResult();
                    RebuildRNGNodeNeighbors(m_pNeighborhoodGraph[item->Key], queryNbs.GetResults(), m_iNeighborhoodSize + 1);
                }
            }
        }

        template <typename T>
        void Index<T>::RebuildRNGNodeNeighbors(int* nodes, const BasicResult* queryResults, int numResults) {
            int count = 0;
            for (int j = 0; j < numResults && count < m_iNeighborhoodSize; j++) {
                const BasicResult& item = queryResults[j];
                if (item.Key < 0) continue;

                bool good = true;
                for (int k = 0; k < count; k++) {
                    if (m_fComputeDistance((m_pSamples)[nodes[k]], (m_pSamples)[item.Key], m_iDataDimension) < item.Dist) {
                        good = false;
                        break;
                    }
                }
                if (good) nodes[count++] = item.Key;
            }
            for (int j = count; j < m_iNeighborhoodSize; j++)  nodes[j] = -1;
        }

        template <typename T>
        bool Index<T>::SaveDataPoints(std::string sDataPointsFileName)
        {
            std::cout << "Save Data Points To " << sDataPointsFileName << std::endl;

            FILE * fp = fopen(sDataPointsFileName.c_str(), "wb");
            if (fp == NULL) return false;

            int R = m_pSamples.R(), C = m_pSamples.C();
            fwrite(&R, sizeof(int), 1, fp);
            fwrite(&C, sizeof(int), 1, fp);

            // write point one by one in case for cache miss
            for (int i = 0; i < R; i++) {
                fwrite((m_pSamples)[i], sizeof(T), C, fp);
            }
            fclose(fp);

            std::cout << "Save Data Points (" << m_pSamples.R() << ", " << m_pSamples.C() << ") Finish!" << std::endl;
            return true;
        }

        template <typename T>
        bool Index<T>::SaveIndex()
        {
            if (!SaveDataPoints(m_sDataPointsFilename)) return false;
            if (!SaveKDT(m_sKDTFilename)) return false;
            if (!SaveRNG(m_sGraphFilename)) return false;
            return true;
        }

        template<typename T>
        ErrorCode
            Index<T>::SaveIndex(const std::string& p_folderPath)
        {
            std::string folderPath(p_folderPath);
            if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
            {
                folderPath += FolderSep;
            }

            if (!direxists(folderPath.c_str()))
            {
                mkdir(folderPath.c_str());
            }

            std::string loaderFilePath = folderPath + "indexloader.ini";

            std::ofstream loaderFile(loaderFilePath);
            if (!loaderFile.is_open())
            {
                return ErrorCode::FailedCreateFile;
            }

            m_sDataPointsFilename = "vectors.bin";
            m_sKDTFilename = "tree.bin";
            m_sGraphFilename = "graph.bin";

            if (!SaveDataPoints(folderPath + m_sDataPointsFilename)) return ErrorCode::Fail;
            if (!SaveKDT(folderPath + m_sKDTFilename)) return ErrorCode::Fail;
            if (!SaveRNG(folderPath + m_sGraphFilename)) return ErrorCode::Fail;

            loaderFile << "[Index]" << std::endl;

            loaderFile << "IndexAlgoType=" << Helper::Convert::ConvertToString(IndexAlgoType::KDT) << std::endl;
            loaderFile << "ValueType=" << Helper::Convert::ConvertToString(GetEnumValueType<T>()) << std::endl;
            loaderFile << std::endl;

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    loaderFile << RepresentStr << "=" << GetParameter(RepresentStr) << std::endl;

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter

            loaderFile << std::endl;

            if (nullptr != m_pMetadata)
            {
                std::string metadataFile = "metadata.bin";
                std::string metadataIndexFile = "metadataIndex.bin";

                m_pMetadata->SaveMetadata(folderPath + metadataFile, folderPath + metadataIndexFile);

                loaderFile << "[MetaData]" << std::endl;
                loaderFile << "MetaDataFilePath=" << metadataFile << std::endl;
                loaderFile << "MetaDataIndexPath=" << metadataIndexFile << std::endl;
                loaderFile << std::endl;
            }
            loaderFile.close();

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndex(const std::string& p_folderPath, const Helper::IniReader& p_configReader)
        {
            std::string folderPath(p_folderPath);
            if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
            {
                folderPath += FolderSep;
            }

            std::string metadataSection("MetaData");
            if (p_configReader.DoesSectionExist(metadataSection))
            {
                std::string metadataFilePath = p_configReader.GetParameter(metadataSection,
                    "MetaDataFilePath",
                    std::string());
                std::string metadataIndexFilePath = p_configReader.GetParameter(metadataSection,
                    "MetaDataIndexPath",
                    std::string());

                m_pMetadata.reset(new FileMetadataSet(folderPath + metadataFilePath, folderPath + metadataIndexFilePath));

                if (!m_pMetadata->Available())
                {
                    std::cerr << "Error: Failed to load metadata." << std::endl;
                    return ErrorCode::Fail;
                }
            }

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
            SetParameter(RepresentStr, \
                         p_configReader.GetParameter("Index", \
                                                     RepresentStr, \
                                                     std::string(#DefaultValue)).c_str()); \

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter

            if (DistCalcMethod::Undefined == m_iDistCalcMethod)
            {
                return ErrorCode::Fail;
            }

            if (!LoadDataPoints(folderPath + m_sDataPointsFilename)) return ErrorCode::Fail;
            if (!LoadKDT(folderPath + m_sKDTFilename)) return ErrorCode::Fail;
            if (!LoadGraph(folderPath + m_sGraphFilename)) return ErrorCode::Fail;

            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();

            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_iMaxCheck, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);

            return ErrorCode::Success;
        }
#pragma endregion
#pragma endregion

        template <typename T>
        ErrorCode
            Index<T>::SetParameter(const char* p_param, const char* p_value)
        {
            if (nullptr == p_param || nullptr == p_value)
            {
                return ErrorCode::Fail;
            }

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        fprintf(stderr, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter

            m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);
            return ErrorCode::Success;
        }


        template <typename T>
        std::string
            Index<T>::GetParameter(const char* p_param) const
        {
            if (nullptr == p_param)
            {
                return std::string();
            }

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        return SPTAG::Helper::Convert::ConvertToString(VarName); \
    } \

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter

            return std::string();
        }
    }
}

#define DefineVectorValueType(Name, Type) \
template class SPTAG::KDT::Index<Type>; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType


