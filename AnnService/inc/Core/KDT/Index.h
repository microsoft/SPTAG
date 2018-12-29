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
#include "../Common/FineGrainedLock.h"
#include "../Common/DataUtils.h"

#include <functional>
#include <mutex>
#include <tbb/concurrent_unordered_set.h>

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

            DistCalcMethod m_iDistCalcMethod;
            float(*m_fComputeDistance)(const T* pX, const T* pY, int length);

            int m_iCacheSize;
            int m_iDebugLoad;

            int g_iThresholdOfNumberOfContinuousNoBetterPropagation;
            int g_iNumberOfInitialDynamicPivots;
            int g_iNumberOfOtherDynamicPivots;

            int m_iNumberOfThreads;
            std::mutex m_dataAllocLock;
            COMMON::FineGrainedLock m_dataUpdateLock;
            tbb::concurrent_unordered_set<int> m_deletedID;
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
            VectorValueType GetVectorValueType() const { return GetEnumValueType<T>(); }

            ErrorCode BuildIndex(const void* p_data, int p_vectorNum, int p_dimension);
            ErrorCode LoadIndex(const std::string& p_folderPath);
            ErrorCode LoadIndexFromMemory(const std::vector<void*>& p_indexBlobs);

            ErrorCode SaveIndex(const std::string& p_folderPath);

            void SearchIndex(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, const tbb::concurrent_unordered_set<int> &p_deleted) const;
            ErrorCode SearchIndex(QueryResult &p_query) const;
           
            ErrorCode AddIndex(const void* p_vectors, int p_vectorNum, int p_dimension);
            ErrorCode DeleteIndex(const void* p_vectors, int p_vectorNum);
            ErrorCode RefineIndex(const std::string& p_folderPath);
            ErrorCode MergeIndex(const char* p_indexFilePath1, const char* p_indexFilePath2);

            ErrorCode SetParameter(const char* p_param, const char* p_value);
            std::string GetParameter(const char* p_param) const;

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
            void BuildKDT(std::vector<int>& indices, std::vector<int>& newStart, std::vector<KDTNode>& newRoot);
            bool SaveKDT(std::string sKDTFilename, std::vector<int>& newStart, std::vector<KDTNode>& newRoot) const;
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
                COMMON::WorkSpace& space, COMMON::QueryResultSet<T> &query, const tbb::concurrent_unordered_set<int> &deleted) const;
        };
    } // namespace KDT
} // namespace SPTAG

#endif // _SPTAG_KDT_INDEX_H_
