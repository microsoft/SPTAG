#ifndef _SPTAG_KDT_INDEX_H_
#define _SPTAG_KDT_INDEX_H_

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

    namespace KDT
    {
        // node type for storing KDT
        struct KDTNode
        {
            int left;
            int right;
            short split_dim;
            float split_value;
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

            // KDT structures. 
            int m_iKDTNumber;
            std::vector<int> m_pKDTStart;
            std::vector<KDTNode> m_pKDTRoots;
            int m_numTopDimensionKDTSplit;
            int m_numSamplesKDTSplitConsideration;

            // Graph structure
            int m_iGraphSize;
            int m_iNeighborhoodSize;
            COMMON::Dataset<int> m_pNeighborhoodGraph;

            // Variables for building TPTs 
            int m_iTPTNumber;
            int m_iTPTLeafSize;
            int m_numTopDimensionTPTSplit;
            int m_numSamplesTPTSplitConsideration;

            // Variables for building graph 
            int m_iRefineIter;
            int m_iCEF;
            int m_iMaxCheckForRefineGraph;
            int m_iMaxCheck;
            std::unordered_map<int, int> m_pSampleToCenter;

            // Load from files directly
            std::string m_sKDTFilename;
            std::string m_sGraphFilename;
            std::string m_sDataPointsFilename;

            // Load from memory mapped files
            char* m_pKDTMemoryFile;
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
            Index() : m_iKDTNumber(1),
                m_numTopDimensionKDTSplit(5),
                m_numSamplesKDTSplitConsideration(100),
                m_iNeighborhoodSize(32),
                m_iTPTNumber(32),
                m_iTPTLeafSize(2000),
                m_numTopDimensionTPTSplit(5),
                m_numSamplesTPTSplitConsideration(1000),
                m_iRefineIter(0),
                m_iCEF(1000),
                m_iMaxCheckForRefineGraph(10000),
                m_iMaxCheck(2048),
                m_pKDTMemoryFile(NULL),
                m_pGraphMemoryFile(NULL),
                m_pDataPointsMemoryFile(NULL),
                m_sKDTFilename("tree.bin"),
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
                m_pKDTRoots.clear();
            }
            int GetNumSamples() const { return m_pSamples.R(); }
            int GetFeatureDim() const { return m_pSamples.C(); }
            int GetNumThreads() const { return m_iNumberOfThreads; }
            int GetCurrMaxCheck() const { return m_iMaxCheck; }
            DistCalcMethod GetDistCalcMethod() const { return m_iDistCalcMethod; }
            IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::KDT; }
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
                std::string KDTFile,
                std::string graphFile,
                int numKDT,
                int neighborhoodSize,
                int numTPTrees,
                int TPTLeafSize,
                int maxCheckForRefineGraph,
                int numThreads,
                DistCalcMethod distCalcMethod,
                int cacheSize = -1,
                int numPoints = -1)
            {
                m_sDataPointsFilename = dataPointsFile;
                m_sKDTFilename = KDTFile;
                m_sGraphFilename = graphFile;
                m_iKDTNumber = numKDT;
                m_iNeighborhoodSize = neighborhoodSize;
                m_iTPTNumber = numTPTrees;
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
                char* pKDTMemFile,
                char* pGraphMemFile,
                DistCalcMethod distCalcMethod,
                int maxCheck,
                int numKDT,
                int neighborhoodSize)
            {
                m_pDataPointsMemoryFile = pDataPointsMemFile;
                m_pKDTMemoryFile = pKDTMemFile;
                m_pGraphMemoryFile = pGraphMemFile;
                m_iMaxCheck = maxCheck;
                m_iKDTNumber = numKDT;
                m_iNeighborhoodSize = neighborhoodSize;
                m_iNumberOfThreads = 1;
                m_iDistCalcMethod = distCalcMethod;
                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);
            }

        private:
            // Functions for loading models from files
            bool LoadDataPoints(std::string sDataPointsFileName);
            bool LoadKDT(std::string sKDTFilename);
            bool LoadGraph(std::string sGraphFilename);

            // Functions for loading models from memory mapped files
            bool LoadDataPoints(char* pDataPointsMemFile);
            bool LoadKDT(char*  pKDTMemFile);
            bool LoadGraph(char*  pGraphMemFile);

            bool SaveDataPoints(std::string sDataPointsFileName);

            // Functions for building kdtree
            void BuildKDT();
            bool SaveKDT(std::string sKDTFilename) const;
            void DivideTree(KDTNode* pTree, std::vector<int>& indices,int first, int last,
                int index, int &iTreeSize);
            void ChooseDivision(KDTNode& node, const std::vector<int>& indices, int first, int last);
            int SelectDivisionDimension(const std::vector<float>& varianceValues) const;
            int Subdivide(const KDTNode& node, std::vector<int>& indices, const int first, const int last);
            
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

            // Functions for hybrid search 
            void KDTSearch(const int node, const bool isInit, const float distBound,
                COMMON::WorkSpace& space, COMMON::QueryResultSet<T> &query) const;
        };
    } // namespace KDT
} // namespace SPTAG

#endif // _SPTAG_KDT_INDEX_H_
