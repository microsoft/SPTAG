#include "inc/Core/BKT/Index.h"
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
    namespace BKT
    {
#pragma region Load data points, kd-tree, neighborhood graph
        template <typename T>
        ErrorCode Index<T>::LoadIndexFromMemory(const std::vector<void*>& p_indexBlobs)
        {
            if (!LoadDataPoints((char*)p_indexBlobs[0])) return ErrorCode::FailedParseValue;
            if (!LoadBKT((char*)p_indexBlobs[1])) return ErrorCode::FailedParseValue;
            if (!LoadGraph((char*)p_indexBlobs[2])) return ErrorCode::FailedParseValue;
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndex(const std::string& p_folderPath)
        {
            std::string folderPath(p_folderPath);
            if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
            {
                folderPath += FolderSep;
            }

            Helper::IniReader p_configReader;
            if (ErrorCode::Success != p_configReader.LoadIniFile(folderPath + "/indexloader.ini"))
            {
                return ErrorCode::FailedOpenFile;
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

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) \
            SetParameter(RepresentStr, \
                         p_configReader.GetParameter("Index", \
                                                     RepresentStr, \
                                                     std::string(#DefaultValue)).c_str()); \

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

            if (DistCalcMethod::Undefined == m_iDistCalcMethod)
            {
                return ErrorCode::Fail;
            }

            if (!LoadDataPoints(folderPath + m_sDataPointsFilename)) return ErrorCode::Fail;
            if (!LoadBKT(folderPath + m_sBKTFilename)) return ErrorCode::Fail;
            if (!LoadGraph(folderPath + m_sGraphFilename)) return ErrorCode::Fail;

            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();
            m_dataUpdateLock.resize(m_iDataSize);

            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_iMaxCheck, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        bool Index<T>::LoadDataPoints(std::string sDataPointsFileName)
        {
            std::cout << "Load Data Points From " << sDataPointsFileName << std::endl;
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
        bool Index<T>::LoadBKT(std::string sBKTFilename)
        {
            std::cout << "Load BKT From " << sBKTFilename << std::endl;
            FILE *fp = fopen(sBKTFilename.c_str(), "rb");
            if (fp == NULL) return false;
            int realBKTNumber;
            fread(&realBKTNumber, sizeof(int), 1, fp);
            m_pBKTStart.resize(realBKTNumber);
            fread(m_pBKTStart.data(), sizeof(int), realBKTNumber, fp);
            if (realBKTNumber < m_iBKTNumber) m_iBKTNumber = realBKTNumber;
            int treeNodeSize;
            fread(&treeNodeSize, sizeof(int), 1, fp);
            m_pBKTRoots.resize(treeNodeSize);
            for (int i = 0; i < treeNodeSize; i++) {
                fread(&(m_pBKTRoots[i].centerid), sizeof(int), 1, fp);
                fread(&(m_pBKTRoots[i].childStart), sizeof(int), 1, fp);
                fread(&(m_pBKTRoots[i].childEnd), sizeof(int), 1, fp);
            }
            fclose(fp);
            std::cout << "Load BKT (" << m_iBKTNumber << ", " << treeNodeSize << ") Finish!" << std::endl;
            return true;
        }

        template <typename T>
        bool Index<T>::LoadBKT(char* pBKTMemFile)
        {
            int realBKTNumber = *((int*)pBKTMemFile);
            pBKTMemFile += sizeof(int);
            m_pBKTStart.resize(realBKTNumber);
            memcpy(m_pBKTStart.data(), pBKTMemFile, sizeof(int)*realBKTNumber);
            pBKTMemFile += sizeof(int)*realBKTNumber;
            if (realBKTNumber < m_iBKTNumber) m_iBKTNumber = realBKTNumber;

            int treeNodeSize = *((int*)pBKTMemFile);
            pBKTMemFile += sizeof(int);
            m_pBKTRoots.resize(treeNodeSize);
            for (int i = 0; i < treeNodeSize; i++) {
                m_pBKTRoots[i].centerid = *((int*)pBKTMemFile);
                pBKTMemFile += sizeof(int);
                m_pBKTRoots[i].childStart = *((int*)pBKTMemFile);
                pBKTMemFile += sizeof(int);
                m_pBKTRoots[i].childEnd = *((int*)pBKTMemFile);
                pBKTMemFile += sizeof(int);
            }
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
        void Index<T>::SearchIndex(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, const tbb::concurrent_unordered_set<int> &p_deleted) const
        {
            for (char i = 0; i < m_iBKTNumber; i++) {
                const BKTNode& node = m_pBKTRoots[m_pBKTStart[i]];
                if (node.childStart < 0) {
                    p_space.m_SPTQueue.insert(COMMON::HeapCell(m_pBKTStart[i], m_fComputeDistance(p_query.GetTarget(), (m_pSamples)[node.centerid], m_iDataDimension)));
                }
                else {
                    for (int begin = node.childStart; begin < node.childEnd; begin++) {
                        int index = m_pBKTRoots[begin].centerid;
                        p_space.m_SPTQueue.insert(COMMON::HeapCell(begin, m_fComputeDistance(p_query.GetTarget(), (m_pSamples)[index], m_iDataDimension)));
                    }
                }
            }
            int checkLimit = g_iNumberOfInitialDynamicPivots;
            const int checkPos = m_iNeighborhoodSize - 1;
            while (!p_space.m_SPTQueue.empty()) {
                do
                {
                    COMMON::HeapCell bcell = p_space.m_SPTQueue.pop();
                    const BKTNode& tnode = m_pBKTRoots[bcell.node];

                    if (tnode.childStart < 0) {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_iNumberOfCheckedLeaves++;
                            p_space.m_NGQueue.insert(COMMON::HeapCell(tnode.centerid, bcell.distance));
                        }
                        if (p_space.m_iNumberOfCheckedLeaves >= checkLimit) break;
                    }
                    else {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_NGQueue.insert(COMMON::HeapCell(tnode.centerid, bcell.distance));
                        }
                        for (int begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                            int index = m_pBKTRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(COMMON::HeapCell(begin, m_fComputeDistance(p_query.GetTarget(), (m_pSamples)[index], m_iDataDimension)));
                        }
                    }
                } while (!p_space.m_SPTQueue.empty());
                while (!p_space.m_NGQueue.empty()) {
                    COMMON::HeapCell gnode = p_space.m_NGQueue.pop();
                    const int *node = (m_pNeighborhoodGraph)[gnode.node];
                    _mm_prefetch((const char *)node, _MM_HINT_T0);
                    if (p_deleted.find(gnode.node) == p_deleted.end()) {
                      if (p_query.AddPoint(gnode.node, gnode.distance)) {
                             p_space.m_iNumOfContinuousNoBetterPropagation = 0;

                            int checkNode = node[checkPos];
                            if (checkNode < -1) {
                                const BKTNode& tnode = m_pBKTRoots[-2 - checkNode];
                                for (int i = -tnode.childStart; i < tnode.childEnd; i++) {
                                    if (p_deleted.find(m_pBKTRoots[i].centerid) == p_deleted.end()) {
                                        if (!p_query.AddPoint(m_pBKTRoots[i].centerid, gnode.distance)) break;
                                    }
                                }
                            }
                        }
                        else {
                            p_space.m_iNumOfContinuousNoBetterPropagation++;
                            if (p_space.m_iNumOfContinuousNoBetterPropagation > p_space.m_iContinuousLimit || p_space.m_iNumberOfCheckedLeaves > p_space.m_iMaxCheck) {
                                p_query.SortResult(); return;
                            }
                        }
                    }

#ifdef PREFETCH
                    for (int i = 0; i <= checkPos; i++) {
                        _mm_prefetch((const char *)(m_pSamples)[node[i]], _MM_HINT_T0);
                    }
#endif

                    for (int i = 0; i <= checkPos; i++)
                    {
                        int nn_index = node[i];

                        // do not check it if it has been checked
                        if (nn_index < 0) break;
                        if (p_space.CheckAndSet(nn_index)) continue;

                        // count the number of the computed nodes
                        float distance2leaf = m_fComputeDistance(p_query.GetTarget(), (m_pSamples)[nn_index], m_iDataDimension);
                        p_space.m_iNumberOfCheckedLeaves++;
                        p_space.m_NGQueue.insert(COMMON::HeapCell(nn_index, distance2leaf));
                    }
                    if (p_space.m_NGQueue.Top().distance > p_space.m_SPTQueue.Top().distance) {
                        checkLimit = g_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves;
                        break;
                    }
                }
            }
            p_query.SortResult();
        }

        template<typename T>
        ErrorCode
            Index<T>::SearchIndex(QueryResult &p_query) const
        {
            auto workSpace = m_workSpacePool->Rent();
            workSpace->Reset(m_iMaxCheck);

            SearchIndex(*((COMMON::QueryResultSet<T>*)&p_query), *workSpace, m_deletedID);
            m_workSpacePool->Return(workSpace);

            if (p_query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < p_query.GetResultNum(); ++i)
                {
                    for (int i = 0; i < p_query.GetResultNum(); ++i)
                    {
                        int result = p_query.GetResult(i)->VID;
                        p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadata(result));
                    }
                }
            }

            return ErrorCode::Success;
        }
#pragma endregion

#pragma region Build/Save kd-tree & neighborhood graphs
        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void* p_data, int p_vectorNum, int p_dimension)
        {
            m_pSamples.Initialize(p_vectorNum, p_dimension);
            std::memcpy(m_pSamples.GetData(), p_data, p_vectorNum * p_dimension * sizeof(T));
            m_iDataSize = m_pSamples.R();
            m_iDataDimension = m_pSamples.C();
            m_dataUpdateLock.resize(m_iDataSize);

            if (DistCalcMethod::Cosine == m_iDistCalcMethod)
            {
                int base = COMMON::Utils::GetBase<T>();
                for (int i = 0; i < m_iDataSize; i++) {
                    COMMON::Utils::Normalize(m_pSamples[i], m_iDataDimension, base);
                }
            }
            std::vector<int> indices(m_iDataSize);
            for (int i = 0; i < m_iDataSize; i++) indices[i] = i;
            BuildBKT(indices, m_pBKTStart, m_pBKTRoots);
            BuildRNG();

            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_iMaxCheck, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);
            return ErrorCode::Success;
        }

#pragma region Build/Save kd-tree
        template <typename T>
        bool Index<T>::SaveBKT(std::string sBKTFilename, std::vector<int>& newStart, std::vector<BKTNode>& newRoot) const
        {
            std::cout << "Save BKT to " << sBKTFilename << std::endl;
            FILE *fp = fopen(sBKTFilename.c_str(), "wb");
            if(fp == NULL) return false;
            fwrite(&m_iBKTNumber, sizeof(int), 1, fp);
            fwrite(newStart.data(), sizeof(int), m_iBKTNumber, fp);
            int treeNodeSize = (int)newRoot.size();
            fwrite(&treeNodeSize, sizeof(int), 1, fp);
            for (int i = 0; i < treeNodeSize; i++) {
                fwrite(&(newRoot[i].centerid), sizeof(int), 1, fp);
                fwrite(&(newRoot[i].childStart), sizeof(int), 1, fp);
                fwrite(&(newRoot[i].childEnd), sizeof(int), 1, fp);
            }
            fclose(fp);
            std::cout << "Save BKT Finish!" << std::endl;
            return true;
        }

        template <typename T>
        void Index<T>::BuildBKT(std::vector<int>& indices, std::vector<int>& newStart, std::vector<BKTNode>& newRoot)
        {
            omp_set_num_threads(m_iNumberOfThreads);
            struct  BKTStackItem {
                int index, first, last;
                BKTStackItem(int index_, int first_, int last_) : index(index_), first(first_), last(last_) {}
            };
            std::stack<BKTStackItem> ss;

            KmeansArgs<T> args(m_iBKTKmeansK, m_iDataDimension, (int)indices.size(), m_iNumberOfThreads);
            m_pSampleToCenter.clear();

            for (char i = 0; i < m_iBKTNumber; i++)
            {
                std::random_shuffle(indices.begin(), indices.end());

                newStart.push_back((int)newRoot.size());
                newRoot.push_back(BKTNode((int)indices.size()));
                std::cout << "Start to build tree " << i + 1 << std::endl;

                ss.push(BKTStackItem(newStart[i], 0, (int)indices.size()));
                while (!ss.empty()) {
                    BKTStackItem item = ss.top(); ss.pop();
                    int newBKTid = (int)newRoot.size();
                    newRoot[item.index].childStart = newBKTid;
                    if (item.last - item.first <= m_iBKTLeafSize) {
                        for (int j = item.first; j < item.last; j++) {
                            newRoot.push_back(BKTNode(indices[j]));
                        }
                    }
                    else { // clustering the data into BKTKmeansK clusters
                        int numClusters = KmeansClustering(indices, item.first, item.last, args);
                        if (numClusters <= 1) {
                            int end = min(item.last + 1, (int)indices.size());
                            std::sort(indices.begin() + item.first, indices.begin() + end);
                            newRoot[item.index].centerid = indices[item.first];
                            newRoot[item.index].childStart = -newRoot[item.index].childStart;
                            for (int j = item.first + 1; j < end; j++) {
                                newRoot.push_back(BKTNode(indices[j]));
                                m_pSampleToCenter[indices[j]] = newRoot[item.index].centerid;
                            }
                            m_pSampleToCenter[-1 - newRoot[item.index].centerid] = item.index;
                        }
                        else {
                            for (int k = 0; k < m_iBKTKmeansK; k++) {
                                if (args.counts[k] == 0) continue;
                                newRoot.push_back(BKTNode(indices[item.first + args.counts[k] - 1]));
                                if (args.counts[k] > 1) ss.push(BKTStackItem(newBKTid++, item.first, item.first + args.counts[k] - 1));
                                item.first += args.counts[k];
                            }
                        }
                    }
                    newRoot[item.index].childEnd = (int)newRoot.size();
                }
                std::cout << i + 1 << " trees built, " << newRoot.size() - newStart[i] << " " << indices.size() << std::endl;
            }
        }

        template <typename T>
        float Index<T>::KmeansAssign(std::vector<int>& indices, const int first, const int last, KmeansArgs<T>& args, bool updateCenters) {
            float currDist = 0;
            float lambda = (updateCenters) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / (100.0 * (last - first)) : 0;
            int subsize = (last - first - 1) / m_iNumberOfThreads + 1;

#pragma omp parallel for
            for (int tid = 0; tid < m_iNumberOfThreads; tid++)
            {
                int istart = first + tid * subsize;
                int iend = min(first + (tid + 1) * subsize, last);
                int *inewCounts = args.newCounts + tid * m_iBKTKmeansK;
                float *inewCenters = args.newCenters + tid * m_iBKTKmeansK * m_iDataDimension;
                int * iclusterIdx = args.clusterIdx + tid * m_iBKTKmeansK;
                float * iclusterDist = args.clusterDist + tid * m_iBKTKmeansK;
                float idist = 0;
                for (int i = istart; i < iend; i++) {
                    int clusterid = 0;
                    float smallestDist = MaxDist;
                    for (int k = 0; k < m_iBKTKmeansK; k++) {
                        float dist = m_fComputeDistance(m_pSamples[indices[i]], args.centers + k*m_iDataDimension, m_iDataDimension) + lambda*args.counts[k];
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k; smallestDist = dist;
                        }
                    }
                    args.label[i] = clusterid;
                    inewCounts[clusterid]++;
                    idist += smallestDist;
                    if (updateCenters) {
                        for (int j = 0; j < m_iDataDimension; j++) inewCenters[clusterid*m_iDataDimension + j] += m_pSamples[indices[i]][j];
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
                COMMON::Utils::atomic_float_add(&currDist, idist);
            }

            for (int i = 1; i < m_iNumberOfThreads; i++) {
                for (int k = 0; k < m_iBKTKmeansK; k++)
                    args.newCounts[k] += args.newCounts[i*m_iBKTKmeansK + k];
            }

            if (updateCenters) {
                for (int i = 1; i < m_iNumberOfThreads; i++) {
                    float* currCenter = args.newCenters + i*m_iBKTKmeansK*m_iDataDimension;
                    for (int j = 0; j < m_iBKTKmeansK * m_iDataDimension; j++) args.newCenters[j] += currCenter[j];
                }

                int maxcluster = 0;
                for (int k = 1; k < m_iBKTKmeansK; k++) if (args.newCounts[maxcluster] < args.newCounts[k]) maxcluster = k;

                int maxid = maxcluster;
                for (int tid = 1; tid < m_iNumberOfThreads; tid++) {
                    if (args.clusterDist[maxid] < args.clusterDist[tid * m_iBKTKmeansK + maxcluster]) maxid = tid * m_iBKTKmeansK + maxcluster;
                }
                if (args.clusterIdx[maxid] < 0 || args.clusterIdx[maxid] >= m_iDataSize)
                    std::cout << "first:" << first << " last:" << last << " maxcluster:" << maxcluster << "(" << args.newCounts[maxcluster] << ") Error maxid:" << maxid << " dist:" << args.clusterDist[maxid] << std::endl;
                maxid = args.clusterIdx[maxid];

                for (int k = 0; k < m_iBKTKmeansK; k++) {
                    T* TCenter = args.newTCenters + k * m_iDataDimension;
                    if (args.newCounts[k] == 0) {
                        //int nextid = Utils::rand_int(last, first);
                        //while (args.label[nextid] != maxcluster) nextid = Utils::rand_int(last, first);
                        int nextid = maxid;
                        std::memcpy(TCenter, m_pSamples[nextid], sizeof(T)*m_iDataDimension);
                    }
                    else {
                        float* currCenters = args.newCenters + k * m_iDataDimension;
                        for (int j = 0; j < m_iDataDimension; j++) currCenters[j] /= args.newCounts[k];

                        if (m_iDistCalcMethod == DistCalcMethod::Cosine) {
                            COMMON::Utils::Normalize(currCenters, m_iDataDimension, COMMON::Utils::GetBase<T>());
                        }
                        for (int j = 0; j < m_iDataDimension; j++) TCenter[j] = (T)(currCenters[j]);
                    }
                }
            }
            else {
                for (int i = 1; i < m_iNumberOfThreads; i++) {
                    for (int k = 0; k < m_iBKTKmeansK; k++) {
                        if (args.clusterIdx[i*m_iBKTKmeansK + k] != -1 && args.clusterDist[i*m_iBKTKmeansK + k] <= args.clusterDist[k]) {
                            args.clusterDist[k] = args.clusterDist[i*m_iBKTKmeansK + k];
                            args.clusterIdx[k] = args.clusterIdx[i*m_iBKTKmeansK + k];
                        }
                    }
                }
            }
            return currDist;
        }

        template <typename T>
        int Index<T>::KmeansClustering(std::vector<int>& indices, const int first, const int last, KmeansArgs<T>& args) {
            int iterLimit = 100;

            int batchEnd = min(first + m_iSamples, last);
            float currDiff, currDist, minClusterDist = MaxDist;
            for (int numKmeans = 0; numKmeans < 3; numKmeans++) {
                for (int k = 0; k < m_iBKTKmeansK; k++) {
                    int randid = COMMON::Utils::rand_int(last, first);
                    memcpy(args.centers + k*m_iDataDimension, m_pSamples[indices[randid]], sizeof(T)*m_iDataDimension);
                }
                args.ClearCounts();
                currDist = KmeansAssign(indices, first, batchEnd, args, false);
                if (currDist < minClusterDist) {
                    minClusterDist = currDist;
                    memcpy(args.newTCenters, args.centers, sizeof(T)*m_iBKTKmeansK*m_iDataDimension);
                    memcpy(args.counts, args.newCounts, sizeof(int) * m_iBKTKmeansK);
                }
            }

            minClusterDist = MaxDist;
            int noImprovement = 0;
            for (int iter = 0; iter < iterLimit; iter++) {
                std::memcpy(args.centers, args.newTCenters, sizeof(T)*m_iBKTKmeansK*m_iDataDimension);
                std::random_shuffle(indices.begin() + first, indices.begin() + last);

                args.ClearCenters();
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                currDist = KmeansAssign(indices, first, batchEnd, args, true);
                memcpy(args.counts, args.newCounts, sizeof(int)*m_iBKTKmeansK);

                currDiff = 0;
                for (int k = 0; k < m_iBKTKmeansK; k++) {
                    currDiff += m_fComputeDistance(args.centers + k*m_iDataDimension, args.newTCenters + k*m_iDataDimension, m_iDataDimension);
                }

                if (currDist < minClusterDist) {
                    noImprovement = 0;
                    minClusterDist = currDist;
                }
                else {
                    noImprovement++;
                }
                if (currDiff < 1e-3 || noImprovement >= 5) break;
            }

            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign(indices, first, last, args, false);
            memcpy(args.counts, args.newCounts, sizeof(int)*m_iBKTKmeansK);

            int numClusters = 0;
            for (int i = 0; i < m_iBKTKmeansK; i++) if (args.counts[i] > 0) numClusters++;

            if (numClusters <= 1) {
                //if (last - first > 1) std::cout << "large cluster:" << last - first << " dist:" << currDist << std::endl;
                return numClusters;
            }
            args.Shuffle(indices, first, last);
            return numClusters;
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
                int end = min(first + m_iSamples, last);
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
                std::sort(Variance.begin(), Variance.end(), COMMON::Compare);
                std::vector<int> index(m_numTopDimensionTpTreeSplit);
                std::vector<float> weight(m_numTopDimensionTpTreeSplit), bestweight(m_numTopDimensionTpTreeSplit);
                float bestvariance = Variance[m_iDataDimension - 1].Dist;
                for (int i = 0; i < m_numTopDimensionTpTreeSplit; i++)
                {
                    index[i] = Variance[m_iDataDimension - 1 - i].VID;
                    bestweight[i] = 0;
                }
                bestweight[0] = 1;
                float bestmean = Mean[index[0]];

                std::vector<float> Val(count);
                for (int i = 0; i < iIteration; i++)
                {
                    float sumweight = 0;
                    for (int j = 0; j < m_numTopDimensionTpTreeSplit; j++)
                    {
                        weight[j] = float(rand() % 10000) / 5000.0f - 1.0f;
                        sumweight += weight[j] * weight[j];
                    }
                    sumweight = sqrt(sumweight);
                    for (int j = 0; j < m_numTopDimensionTpTreeSplit; j++)
                    {
                        weight[j] /= sumweight;
                    }
                    float mean = 0;
                    for (int j = 0; j < count; j++)
                    {
                        Val[j] = 0;
                        for (int k = 0; k < m_numTopDimensionTpTreeSplit; k++)
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
                        for (int j = 0; j < m_numTopDimensionTpTreeSplit; j++)
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
                    for (int k = 0; k < m_numTopDimensionTpTreeSplit; k++)
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
            if (m_iGraphSize < 1000) {
                std::memset(m_pNeighborhoodGraph.GetData(), -1, m_iGraphSize * m_iNeighborhoodSize * sizeof(int));
                m_iNeighborhoodSize /= graphScale;
                RefineRNG();
                for (int i = 0; i < m_iGraphSize; i++) {
                    if (m_pSampleToCenter.find(-1 - i) != m_pSampleToCenter.end())
                        m_pNeighborhoodGraph[i][m_iNeighborhoodSize - 1] = -2 - m_pSampleToCenter[-1 - i];
                }
                std::cout << "Build RNG Graph end!" << std::endl;
                return;
            }

            {
                COMMON::Dataset<float> NeighborhoodDists(m_iGraphSize, m_iNeighborhoodSize);
                std::vector<std::vector<int>> TptreeDataIndices(m_iTptreeNumber, std::vector<int>(m_iGraphSize));
                std::vector<std::vector<std::pair<int, int>>> TptreeLeafNodes(m_iTptreeNumber, std::vector<std::pair<int, int>>());
                for (int i = 0; i < m_iGraphSize; i++)
                {
                    for (int j = 0; j < m_iNeighborhoodSize; j++)
                    {
                        (m_pNeighborhoodGraph)[i][j] = -1;
                        (NeighborhoodDists)[i][j] = MaxDist;
                    }
                    TptreeDataIndices[0][i] = i;
                }
                for (int i = 1; i < m_iTptreeNumber; i++) {
                    std::memcpy(TptreeDataIndices[i].data(), TptreeDataIndices[0].data(), sizeof(int) * m_iGraphSize);
                }

                std::cout << "Parallel TpTree Partition begin " << std::endl;
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < m_iTptreeNumber; i++)
                {
                    Sleep(i * 100); std::srand(clock());
                    std::random_shuffle(TptreeDataIndices[i].begin(), TptreeDataIndices[i].end());
                    PartitionByTptree(TptreeDataIndices[i], 0, m_iGraphSize - 1, TptreeLeafNodes[i]);
                    std::cout << "Finish Getting Leaves for Tree " << i << std::endl;
                }
                std::cout << "Parallel TpTree Partition done" << std::endl;

                for (int i = 0; i < m_iTptreeNumber; i++)
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
                                if (m_pSampleToCenter.find(p2) == m_pSampleToCenter.end())
                                    COMMON::Utils::AddNeighbor(p2, dist, (m_pNeighborhoodGraph)[p1], (NeighborhoodDists)[p1], m_iNeighborhoodSize);
                                else
                                    COMMON::Utils::AddNeighbor(m_pSampleToCenter[p2], dist, (m_pNeighborhoodGraph)[p1], (NeighborhoodDists)[p1], m_iNeighborhoodSize);
                                if (m_pSampleToCenter.find(p1) == m_pSampleToCenter.end())
                                    COMMON::Utils::AddNeighbor(p1, dist, (m_pNeighborhoodGraph)[p2], (NeighborhoodDists)[p2], m_iNeighborhoodSize);
                                else
                                    COMMON::Utils::AddNeighbor(m_pSampleToCenter[p1], dist, (m_pNeighborhoodGraph)[p2], (NeighborhoodDists)[p2], m_iNeighborhoodSize);
                            }
                        }
                    }
                    TptreeDataIndices[i].clear();
                    TptreeLeafNodes[i].clear();
                    std::cout << std::endl;
                }
                TptreeDataIndices.clear();
                TptreeLeafNodes.clear();

                for (int i = 0; i < m_iDataSize; i++) {
                    if (m_pSampleToCenter.find(-1 - i) != m_pSampleToCenter.end()) {
                        BKTNode& tnode = m_pBKTRoots[m_pSampleToCenter[-1 - i]];
                        for (int iter = -tnode.childStart; iter != tnode.childEnd; iter++) {
                            int node = m_pBKTRoots[iter].centerid;
                            for (int j = 0; j < m_iNeighborhoodSize; j++) {
                                int index = m_pNeighborhoodGraph[node][j];
                                if (index == i) continue;
                                COMMON::Utils::AddNeighbor(index, NeighborhoodDists[node][j], (m_pNeighborhoodGraph)[i], (NeighborhoodDists)[i], m_iNeighborhoodSize);
                            }
                        }
                    }
                }
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

                for (int i = 0; i < m_iGraphSize; i++) {
                    if (m_pSampleToCenter.find(-1 - i) != m_pSampleToCenter.end())
                        m_pNeighborhoodGraph[i][m_iNeighborhoodSize - 1] = -2 - m_pSampleToCenter[-1 - i];
                }
            }
            std::cout << "Build RNG Graph end!" << std::endl;
        }

        template <typename T>
        float Index<T>::GraphAccuracyEstimation(int NSample, bool rng) {
            int* correct = new int[NSample];

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < NSample; i++)
            {
                int x = COMMON::Utils::rand_int(m_iGraphSize);
                //int x = i;
                COMMON::QueryResultSet<T> query((m_pSamples)[x], m_iCEF);
                for (int y = 0; y < m_iGraphSize; y++)
                {
                    if (m_pSampleToCenter.find(y) != m_pSampleToCenter.end() || y == x) continue;
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
                        exact_rng[j] = query.GetResult(j)->VID;
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
        ErrorCode Index<T>::RefineIndex(const std::string& p_folderPath)
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
            tbb::concurrent_unordered_set<int> deleted(m_deletedID.begin(), m_deletedID.end());
            std::vector<int> indices;
            std::unordered_map<int, int> old2new;
            int newR = m_iDataSize;
            for (int i = 0; i < newR; i++) {
                if (deleted.find(i) == deleted.end()) {
                    indices.push_back(i);
                    old2new[i] = i;
                }
                else {
                    while (deleted.find(newR - 1) != deleted.end() && newR > i) newR--;
                    if (newR == i) break;
                    indices.push_back(newR - 1);
                    old2new[newR - 1] = i;
                    newR--;
                }
            }
            old2new[-1] = -1;

            std::cout << "Refine... from " << m_iDataSize << "->" << newR << std::endl;
            std::ofstream vecOut(folderPath + m_sDataPointsFilename, std::ios::binary);
            if (!vecOut.is_open()) return ErrorCode::FailedCreateFile;
            vecOut.write((char*)&newR, sizeof(int));
            vecOut.write((char*)&m_iDataDimension, sizeof(int));
            for (int i = 0; i < newR; i++) {
                vecOut.write((char*)m_pSamples[indices[i]], sizeof(T)*m_iDataDimension);
            }
            vecOut.close();

            if (nullptr != m_pMetadata)
            {
                std::ofstream metaOut(folderPath + "metadata.bin_tmp", std::ios::binary);
                std::ofstream metaIndexOut(folderPath + "metadataIndex.bin", std::ios::binary);
                if (!metaOut.is_open() || !metaIndexOut.is_open()) return ErrorCode::FailedCreateFile;
                metaIndexOut.write((char*)&newR, sizeof(int));
                std::uint64_t offset = 0;
                for (int i = 0; i < newR; i++) {
                    metaIndexOut.write((char*)&offset, sizeof(std::uint64_t));
                    ByteArray meta = m_pMetadata->GetMetadata(indices[i]);
                    metaOut.write((char*)meta.Data(), sizeof(uint8_t)*meta.Length());
                    offset += meta.Length();
                }
                metaOut.close();
                metaIndexOut.write((char*)&offset, sizeof(std::uint64_t));
                metaIndexOut.close();

                SPTAG::MetadataSet::MetaCopy(folderPath + "metadata.bin_tmp", folderPath + "metadata.bin");
            }

            std::vector<BKTNode> newRoot;
            std::vector<int> newStart;
            std::vector<int> tmpindices(indices.begin(), indices.end());
            BuildBKT(tmpindices, newStart, newRoot);
#pragma omp parallel for
            for (int i = 0; i < newRoot.size(); i++) {
                newRoot[i].centerid = old2new[newRoot[i].centerid];
            }
            SaveBKT(folderPath + m_sBKTFilename, newStart, newRoot);

            std::ofstream graphOut(folderPath + m_sGraphFilename, std::ios::binary);
            if (!graphOut.is_open()) return ErrorCode::FailedCreateFile;
            graphOut.write((char*)&newR, sizeof(int));
            graphOut.write((char*)&m_iNeighborhoodSize, sizeof(int));

            int *neighbors = new int[m_iNeighborhoodSize];
            COMMON::WorkSpace space;
            space.Initialize(m_iMaxCheckForRefineGraph, m_iDataSize);
            for (int i = 0; i < newR; i++) {
                space.Reset(m_iMaxCheckForRefineGraph);
                COMMON::QueryResultSet<T> query((m_pSamples)[indices[i]], m_iCEF);
                space.CheckAndSet(indices[i]);
                for (int j = 0; j < m_iNeighborhoodSize; j++) {
                    int index = m_pNeighborhoodGraph[indices[i]][j];
                    if (index < 0 || space.CheckAndSet(index)) continue;
                    space.m_NGQueue.insert(COMMON::HeapCell(index, m_fComputeDistance(query.GetTarget(), m_pSamples[index], m_iDataDimension)));
                }
                SearchIndex(query, space, deleted);
                RebuildRNGNodeNeighbors(neighbors, query.GetResults(), m_iCEF);
                for (int j = 0; j < m_iNeighborhoodSize; j++)
                    neighbors[j] = old2new[neighbors[j]];
                if (m_pSampleToCenter.find(-1 - indices[i]) != m_pSampleToCenter.end()) {
                    neighbors[m_iNeighborhoodSize - 1] = -2 - m_pSampleToCenter[-1 - indices[i]];
                }
                graphOut.write((char*)neighbors, sizeof(int) * m_iNeighborhoodSize);
            }
            delete[]neighbors;
            graphOut.close();

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::MergeIndex(const char* p_indexFilePath1, const char* p_indexFilePath2) {
            std::string folderPath1(p_indexFilePath1), folderPath2(p_indexFilePath2);
            if (!folderPath1.empty() && *(folderPath1.rbegin()) != FolderSep) folderPath1 += FolderSep;
            if (!folderPath2.empty() && *(folderPath2.rbegin()) != FolderSep) folderPath2 += FolderSep;

            Helper::IniReader p_configReader1, p_configReader2;
            if (ErrorCode::Success != p_configReader1.LoadIniFile(folderPath1 + "/indexloader.ini"))
                return ErrorCode::FailedOpenFile;

            if (ErrorCode::Success != p_configReader2.LoadIniFile(folderPath2 + "/indexloader.ini"))
                return ErrorCode::FailedOpenFile;

            std::string empty("");
            if (!COMMON::DataUtils::MergeIndex(folderPath1 + p_configReader1.GetParameter("Index", "VectorFilePath", empty),
                folderPath1 + p_configReader1.GetParameter("MetaData", "MetaDataFilePath", empty),
                folderPath1 + p_configReader1.GetParameter("MetaData", "MetaDataIndexPath", empty),
                folderPath2 + p_configReader1.GetParameter("Index", "VectorFilePath", empty),
                folderPath2 + p_configReader1.GetParameter("MetaData", "MetaDataFilePath", empty),
                folderPath2 + p_configReader1.GetParameter("MetaData", "MetaDataIndexPath", empty)))
                return ErrorCode::Fail;

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) \
            SetParameter(RepresentStr, \
                         p_configReader1.GetParameter("Index", \
                                                     RepresentStr, \
                                                     std::string(#DefaultValue)).c_str()); \

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

            if (!LoadDataPoints(folderPath1 + p_configReader1.GetParameter("Index", "VectorFilePath", empty))) return ErrorCode::FailedOpenFile;
            std::vector<int> indices(m_iDataSize);
            for (int i = 0; i < m_iDataSize; i++) indices[i] = i;
            BuildBKT(indices, m_pBKTStart, m_pBKTRoots);
            BuildRNG();

            SaveBKT(folderPath1 + p_configReader1.GetParameter("Index", "TreeFilePath", empty), m_pBKTStart, m_pBKTRoots);
            SaveRNG(folderPath1 + p_configReader1.GetParameter("Index", "GraphFilePath", empty));
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const void* p_vectors, int p_vectorNum) {
            const T* ptr_v = (const T*)p_vectors;
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < p_vectorNum; i++) {
                COMMON::QueryResultSet<T> query(ptr_v + i * m_iDataDimension, m_iCEF);
                SearchIndex(query);
                for (int i = 0; i < m_iCEF; i++) {
                    if (query.GetResult(i)->Dist < 1e-6) {
                        m_deletedID.insert(query.GetResult(i)->VID);
                    }
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::AddIndex(const void* p_vectors, int p_vectorNum, int p_dimension) {
            if (m_pBKTRoots.size() == 0) {
                return BuildIndex(p_vectors, p_vectorNum, p_dimension);
            }
            if (p_dimension != m_iDataDimension) return ErrorCode::FailedParseValue;

            int begin, end;
            {
                std::lock_guard<std::mutex> lock(m_dataAllocLock);
                
                m_pSamples.AddBatch((const T*)p_vectors, p_vectorNum);
                m_pNeighborhoodGraph.AddBatch(p_vectorNum);

                end = m_iDataSize + p_vectorNum;
                if (m_pSamples.R() != end || m_pNeighborhoodGraph.R() != end) {
                    std::cout << "Memory Error: Cannot alloc space for vectors" << std::endl;
                    m_pSamples.SetR(m_iDataSize);
                    m_pNeighborhoodGraph.SetR(m_iDataSize);
                    return ErrorCode::Fail;
                }
                begin = m_iDataSize;
                m_iDataSize = end;
                m_iGraphSize = end;
                m_dataUpdateLock.resize(m_iDataSize);
            }
            if (DistCalcMethod::Cosine == m_iDistCalcMethod)
            {
                int base = COMMON::Utils::GetBase<T>();
                for (int i = begin; i < end; i++) {
                    COMMON::Utils::Normalize((T*)m_pSamples[i], m_iDataDimension, base);
                }
            }

            auto space = m_workSpacePool->Rent();
            for (int node = begin; node < end; node++)
            {
                RefineRNGNode(node, *(space.get()), true);
            }
            m_workSpacePool->Return(space);
            std::cout << "Add " << p_vectorNum << " vectors" << std::endl;
            return ErrorCode::Success;
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
            SearchIndex(query, space, m_deletedID);
            RebuildRNGNodeNeighbors(m_pNeighborhoodGraph[node], query.GetResults(), m_iCEF);

            if (updateNeighbors) {
                // update neighbors
                for (int j = 0; j < m_iCEF; j++)
                {
                    BasicResult* item = query.GetResult(j);
                    if (item->VID < 0) break;

                    int insertID = node;
                    int* nodes = m_pNeighborhoodGraph[item->VID];
                    std::lock_guard<std::mutex> lock(m_dataUpdateLock[item->VID]);
                    for (int k = 0; k < m_iNeighborhoodSize; k++)
                    {
                        int tmpNode = nodes[k];
                        if (tmpNode < -1) continue;

                        if (tmpNode < 0)
                        {
                            bool good = true;
                            for (int t = 0; t < k; t++) {
                                if (m_fComputeDistance((m_pSamples)[insertID], (m_pSamples)[nodes[t]], m_iDataDimension) < item->Dist) {
                                    good = false;
                                    break;
                                }
                            }
                            if (good) {
                                nodes[k] = insertID;
                            }
                            break;
                        }
                        float tmpDist = m_fComputeDistance(m_pSamples[item->VID], m_pSamples[tmpNode], m_iDataDimension);
                        if (item->Dist < tmpDist || (item->Dist == tmpDist && insertID < tmpNode))
                        {
                            bool good = true;
                            for (int t = 0; t < k; t++) {
                                if (m_fComputeDistance((m_pSamples)[insertID], (m_pSamples)[nodes[t]], m_iDataDimension) < item->Dist) {
                                    good = false;
                                    break;
                                }
                            }
                            if (good) {
                                nodes[k] = insertID;
                                insertID = tmpNode;
                                item->Dist = tmpDist;
                            }
                            else {
                                break;
                            }
                        }
                    }
                }
            }
        }

        template <typename T>
        void Index<T>::RebuildRNGNodeNeighbors(int* nodes, const BasicResult* queryResults, int numResults) {
            int count = 0;
            for (int j = 0; j < numResults && count < m_iNeighborhoodSize; j++) {
                const BasicResult& item = queryResults[j];
                if (item.VID < 0) continue;

                bool good = true;
                for (int k = 0; k < count; k++) {
                    if (m_fComputeDistance((m_pSamples)[nodes[k]], (m_pSamples)[item.VID], m_iDataDimension) <= item.Dist) {
                        good = false;
                        break;
                    }
                }
                if (good) nodes[count++] = item.VID;
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
            m_sBKTFilename = "tree.bin";
            m_sGraphFilename = "graph.bin";
            std::string metadataFile = "metadata.bin";
            std::string metadataIndexFile = "metadataIndex.bin";

            loaderFile << "[Index]" << std::endl;
            loaderFile << "IndexAlgoType=" << Helper::Convert::ConvertToString(IndexAlgoType::BKT) << std::endl;
            loaderFile << "ValueType=" << Helper::Convert::ConvertToString(GetEnumValueType<T>()) << std::endl;
            loaderFile << std::endl;

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    loaderFile << RepresentStr << "=" << GetParameter(RepresentStr) << std::endl;

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

            loaderFile << std::endl;

            if (nullptr != m_pMetadata)
            {
                loaderFile << "[MetaData]" << std::endl;
                loaderFile << "MetaDataFilePath=" << metadataFile << std::endl;
                loaderFile << "MetaDataIndexPath=" << metadataIndexFile << std::endl;
                loaderFile << std::endl;
            }
            loaderFile.close();

            if (m_deletedID.size() > 0) {
                RefineIndex(folderPath);
            }
            else {            
                if (!SaveDataPoints(folderPath + m_sDataPointsFilename)) return ErrorCode::Fail;
                if (!SaveBKT(folderPath + m_sBKTFilename, m_pBKTStart, m_pBKTRoots)) return ErrorCode::Fail;
                if (!SaveRNG(folderPath + m_sGraphFilename)) return ErrorCode::Fail;
                if (nullptr != m_pMetadata)
                {
                    m_pMetadata->SaveMetadata(folderPath + metadataFile, folderPath + metadataIndexFile);
                }
            }
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

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        fprintf(stderr, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

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

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        return SPTAG::Helper::Convert::ConvertToString(VarName); \
    } \

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

            return std::string();
        }
    }
}

#define DefineVectorValueType(Name, Type) \
template class SPTAG::BKT::Index<Type>; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType


