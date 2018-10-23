#ifndef _SPTAG_BKT_INDEX_H_
#define _SPTAG_BKT_INDEX_H_

#include "../SearchQuery.h"
#include "../VectorIndex.h"
#include "../Common.h"

#include "../Common/CommonUtils.h"
#include "../Common/DistanceUtils.h"
#include "../Common/QueryResultSet.h"
#include "../Common/Heap.h"
#include "../Common/Dataset.h"
#include "../Common/WorkSpace.h"
#include "../Common/WorkSpacePool.h"

#include <functional>
#include <list>
#include <mutex>
#include <stack>

namespace SPTAG
{

namespace Helper
{
class IniReader;
}


namespace BKT
{
    // node type for storing BKT
    struct BKTNode
    {
        int centerid;
        int childStart;
        int childEnd;

        BKTNode(int cid = -1) : centerid(cid), childStart(-1) {}
    };

    template <typename T>
    struct KmeansArgs {
        int _K;
        int _D;
        int _T;
        T* centers;
        int* counts;
        float* newCenters;
        int* newCounts;
        char* label;
        int* clusterIdx;
        float* clusterDist;
        T* newTCenters;

        KmeansArgs(int k, int dim, int datasize, int threadnum): _K(k), _D(dim), _T(threadnum) {
            centers = new T[k * dim];
            counts = new int[k];
            newCenters = new float[threadnum * k * dim];
            newCounts = new int[threadnum * k];
            label = new char[datasize];
            clusterIdx = new int[threadnum * k];
            clusterDist = new float[threadnum * k];
            newTCenters = new T[k * dim];
        }

        ~KmeansArgs() {
            delete[] centers;
            delete[] counts;
            delete[] newCenters;
            delete[] newCounts;
            delete[] label;
            delete[] clusterIdx;
            delete[] clusterDist;
            delete[] newTCenters;
        }

        inline void ClearCounts() {
            memset(newCounts, 0, sizeof(int) * _T * _K);
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

        void Shuffle(std::vector<int>& indices, int first, int last) {
            int* pos = new int[_K];
            pos[0] = first;
            for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

            for (int k = 0; k < _K; k++) {
                if (newCounts[k] == 0) continue;
                int i = pos[k];
                while (newCounts[k] > 0) {
                    int swapid = pos[(int)(label[i])] + newCounts[(int)(label[i])] - 1;
                    newCounts[(int)(label[i])]--;
                    std::swap(indices[i], indices[swapid]);
                    std::swap(label[i], label[swapid]);
                }
                while (indices[i] != clusterIdx[k]) i++;
                std::swap(indices[i], indices[pos[k] + counts[k] - 1]);
            }
            delete[] pos;
        }
    };

    template<typename T>
    class Index : public VectorIndex
    {
    private:
        // Initial data points
        int m_iDataSize;
        int m_iDataDimension;
        COMMON::Dataset<T> m_pSamples;
        std::shared_ptr<MetadataSet> m_pMetadata;

        // BKT structures. 
        int m_iBKTNumber;
        std::vector<int> m_pBKTStart;
        std::vector<BKTNode> m_pBKTRoots;

        // Graph structure
        int m_iGraphSize;
        int m_iNeighborhoodSize;
        COMMON::Dataset<int> m_pNeighborhoodGraph;

        // Variables for building BKTs and TPTs 
        int m_iBKTKmeansK;
        int m_iBKTLeafSize;
        int m_iSamples;
        int m_iTptreeNumber;
        int m_iTPTLeafSize;
        int m_numTopDimensionTpTreeSplit;

        // Variables for building graph 
        int m_iRefineIter;
        int m_iCEF;
        int m_iMaxCheckForRefineGraph;
        int m_iMaxCheck;
        std::unordered_map<int, int> m_pSampleToCenter;

        // Load from files directly
        std::string m_sBKTFilename;
        std::string m_sGraphFilename;
        std::string m_sDataPointsFilename;

        // Load from memory mapped files
        char* m_pBKTMemoryFile;
        char* m_pGraphMemoryFile;
        char* m_pDataPointsMemoryFile;

        int m_iNumberOfThreads;
        DistCalcMethod m_iDistCalcMethod;
        float(*m_fComputeDistance)(const T* pX, const T* pY, int length);

        int m_iCacheSize;
        int m_iDebugLoad;

        int g_iThresholdOfNumberOfContinuousNoBetterPropagation;
        int g_iNumberOfInitialDynamicPivots;
        int g_iNumberOfOtherDynamicPivots;

        std::unique_ptr<COMMON::WorkSpacePool> m_workSpacePool;

    public:
        Index() : m_iBKTNumber(1),
            m_iBKTKmeansK(32),
            m_iBKTLeafSize(8),
            m_iSamples(1000),
            m_iNeighborhoodSize(32),
            m_iTptreeNumber(32),
            m_iTPTLeafSize(2000),
            m_numTopDimensionTpTreeSplit(5),
            m_iRefineIter(0),
            m_iCEF(1000),
            m_iMaxCheckForRefineGraph(10000),
            m_iMaxCheck(2048),
            m_pBKTMemoryFile(NULL),
            m_pGraphMemoryFile(NULL),
            m_pDataPointsMemoryFile(NULL),
            m_sBKTFilename("tree.bin"),
            m_sGraphFilename("graph.bin"),
            m_sDataPointsFilename("vectors.bin"),
            m_iNumberOfThreads(1),
            m_iDistCalcMethod(DistCalcMethod::Cosine),
            m_fComputeDistance(COMMON::DistanceCalcSelector<T>(DistCalcMethod::Cosine)),
            m_iCacheSize(-1),
            m_iDebugLoad(-1),
            g_iThresholdOfNumberOfContinuousNoBetterPropagation(3),
            g_iNumberOfInitialDynamicPivots(50),
            g_iNumberOfOtherDynamicPivots(4) {}

        ~Index() {
            m_pBKTRoots.clear();
        }
        int GetNumSamples() const { return m_pSamples.R(); }
        int GetFeatureDim() const { return m_pSamples.C(); }
        int GetNumThreads() const { return m_iNumberOfThreads; }
        int GetCurrMaxCheck() const { return m_iMaxCheck; }
        DistCalcMethod GetDistCalcMethod() const { return m_iDistCalcMethod; }
        IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::BKT; }
        VectorValueType AcceptableQueryValueType() const { return GetEnumValueType<T>(); }
        void SetMetadata(const std::string& metadataFilePath, const std::string& metadataIndexPath) {
            m_pMetadata.reset(new FileMetadataSet(metadataFilePath, metadataIndexPath));
        }
        ByteArray GetMetadata(IndexType p_vectorID) const {
            if (nullptr != m_pMetadata)
            {
                return m_pMetadata->GetMetadata(p_vectorID);
            }
            return ByteArray::c_empty;
        }

        bool BuildIndex();
        bool BuildIndex(void* p_data, int p_vectorNum, int p_dimension);
        ErrorCode BuildIndex(std::shared_ptr<VectorSet> p_vectorSet,
            std::shared_ptr<MetadataSet> p_metadataSet);

        bool LoadIndex();
        ErrorCode LoadIndex(const std::string& p_folderPath, const Helper::IniReader& p_configReader);

        bool SaveIndex();
        ErrorCode SaveIndex(const std::string& p_folderPath);

        void SearchIndex(COMMON::QueryResultSet<T> &query, COMMON::WorkSpace &space) const;
        ErrorCode SearchIndex(QueryResult &query) const;

        void AddNodes(const T* pData, int num, COMMON::WorkSpace &space);

        ErrorCode SetParameter(const char* p_param, const char* p_value);
        std::string GetParameter(const char* p_param) const;

        // This can be used for both building model files or searching with model files loaded.
        void SetParameters(std::string dataPointsFile,
            std::string BKTFile,
            std::string graphFile,
            int numBKT,
            int neighborhoodSize,
            int kmeansK,
            int BKTLeafSize,
            int numSamplesBKT,
            int numTPTrees,
            int TPTLeafSize,
            int maxCheckForRefineGraph,
            int numThreads,
            DistCalcMethod distCalcMethod,
            int cacheSize = -1,
            int numPoints = -1)
        {
            m_sDataPointsFilename = dataPointsFile;
            m_sBKTFilename = BKTFile;
            m_sGraphFilename = graphFile;
            m_iBKTNumber = numBKT;
            m_iNeighborhoodSize = neighborhoodSize;
            m_iBKTKmeansK = kmeansK;
            m_iBKTLeafSize = BKTLeafSize;
            m_iSamples = numSamplesBKT;
            m_iTptreeNumber = numTPTrees;
            m_iTPTLeafSize = TPTLeafSize;
            m_iMaxCheckForRefineGraph = maxCheckForRefineGraph;
            m_iNumberOfThreads = numThreads;
            m_iDistCalcMethod = distCalcMethod;
            m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);

            m_iCacheSize = cacheSize;
            m_iDebugLoad = numPoints;
        }

        // Only used for searching with memory mapped model files
        void SetParameters(char* pDataPointsMemFile,
            char* pBKTMemFile,
            char* pGraphMemFile,
            DistCalcMethod distCalcMethod,
            int maxCheck,
            int numBKT,
            int neighborhoodSize)
        {
            m_pDataPointsMemoryFile = pDataPointsMemFile;
            m_pBKTMemoryFile = pBKTMemFile;
            m_pGraphMemoryFile = pGraphMemFile;
            m_iMaxCheck = maxCheck;
            m_iBKTNumber = numBKT;
            m_iNeighborhoodSize = neighborhoodSize;
            m_iNumberOfThreads = 1;
            m_iDistCalcMethod = distCalcMethod;
            m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);
        }

    private:
        // Functions for loading models from files
        bool LoadDataPoints(std::string sDataPointsFileName);
        bool LoadBKT(std::string sBKTFilename);
        bool LoadGraph(std::string sGraphFilename);

        // Functions for loading models from memory mapped files
        bool LoadDataPoints(char* pDataPointsMemFile);
        bool LoadBKT(char*  pBKTMemFile);
        bool LoadGraph(char*  pGraphMemFile);

        bool SaveDataPoints(std::string sDataPointsFileName);

        // Functions for building balanced kmeans tree
        void BuildBKT();
        bool SaveBKT(std::string sBKTFilename) const;
        float KmeansAssign(std::vector<int>& indices, const int first, const int last, KmeansArgs<T>& args, bool updateCenters);
        int KmeansClustering(std::vector<int>& indices, const int first, const int last, KmeansArgs<T>& args);

        // Functions for building Graph
        void BuildRNG();
        bool SaveRNG(std::string sGraphFilename) const;
        void PartitionByTptree(std::vector<int> &indices,
            const int first,
            const int last,
            std::vector<std::pair<int, int>> &leaves);
        void RefineRNG();
        void RefineRNGNode(const int node, COMMON::WorkSpace &space, bool updateNeighbors);
        void RebuildRNGNodeNeighbors(int* nodes, const BasicResult* queryResults, int numResults);
        float GraphAccuracyEstimation(int NSample, bool rng);
    };
} // namespace BKT
} // namespace SPTAG

#endif // _SPTAG_BKT_INDEX_H_
