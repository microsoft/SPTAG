// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_BKT_INDEX_H_
#define _SPTAG_BKT_INDEX_H_

#include "../Common.h"
#include "../VectorIndex.h"

#include "../Common/CommonUtils.h"
#include "../Common/DistanceUtils.h"
#include "../Common/QueryResultSet.h"
#include "../Common/Dataset.h"
#include "../Common/WorkSpace.h"
#include "../Common/WorkSpacePool.h"
#include "../Common/RelativeNeighborhoodGraph.h"
#include "../Common/BKTree.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/ThreadPool.h"

#include <functional>
#include <mutex>

namespace SPTAG
{

    namespace Helper
    {
        class IniReader;
    }

    namespace BKT
    {
        template<typename T>
        class Index : public VectorIndex
        {
            class RebuildJob : public Helper::ThreadPool::Job {
            public:
                RebuildJob(VectorIndex* p_index, COMMON::BKTree* p_tree, COMMON::RelativeNeighborhoodGraph* p_graph) : m_index(p_index), m_tree(p_tree), m_graph(p_graph) {}
                void exec() {
                    SizeType numSamples = m_index->GetNumSamples();

                    std::vector<SizeType> localindices(numSamples);
                    for (SizeType i = 0; i < numSamples; i++) localindices[i] = i;

                    COMMON::BKTree newTrees(*m_tree);
                    newTrees.BuildTrees<T>(m_index, &localindices);
                    m_tree->swap(newTrees);

                    const std::unordered_map<SizeType, SizeType>& idmap = newTrees.GetSampleMap();
                    for (auto iter = idmap.begin(); iter != idmap.end(); iter++)
                        if (iter->first < 0)
                            m_graph->Update(-1 - iter->first, m_graph->m_iNeighborhoodSize - 1, -2 - iter->second);
                    std::cout << "Rebuild finish!" << std::endl;
                }
            private:
                VectorIndex* m_index;
                COMMON::BKTree* m_tree;
                COMMON::RelativeNeighborhoodGraph* m_graph;
            };

        private:
            // data points
            COMMON::Dataset<T> m_pSamples;
        
            // BKT structures. 
            COMMON::BKTree m_pTrees;

            // Graph structure
            COMMON::RelativeNeighborhoodGraph m_pGraph;

            std::string m_sBKTFilename;
            std::string m_sGraphFilename;
            std::string m_sDataPointsFilename;
            std::string m_sDeleteDataPointsFilename;

            std::mutex m_dataAddLock; // protect data and graph
            Helper::Concurrent::ConcurrentSet<SizeType> m_deletedID;
            float m_fDeletePercentageForRefine;
            std::unique_ptr<COMMON::WorkSpacePool> m_workSpacePool;
            int m_addCountForRebuild;
            Helper::ThreadPool m_threadPool;

            int m_iNumberOfThreads;
            DistCalcMethod m_iDistCalcMethod;
            float(*m_fComputeDistance)(const T* pX, const T* pY, DimensionType length);
        
            int m_iMaxCheck;        
            int m_iThresholdOfNumberOfContinuousNoBetterPropagation;
            int m_iNumberOfInitialDynamicPivots;
            int m_iNumberOfOtherDynamicPivots;
        public:
            Index()
            {
#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

                m_pSamples.SetName("Vector");
                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);
            }

            ~Index() {}

            inline SizeType GetNumSamples() const { return m_pSamples.R(); }
            inline DimensionType GetFeatureDim() const { return m_pSamples.C(); }
        
            inline int GetCurrMaxCheck() const { return m_iMaxCheck; }
            inline int GetNumThreads() const { return m_iNumberOfThreads; }
            inline DistCalcMethod GetDistCalcMethod() const { return m_iDistCalcMethod; }
            inline IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::BKT; }
            inline VectorValueType GetVectorValueType() const { return GetEnumValueType<T>(); }
            
            inline float ComputeDistance(const void* pX, const void* pY) const { return m_fComputeDistance((const T*)pX, (const T*)pY, m_pSamples.C()); }
            inline const void* GetSample(const SizeType idx) const { return (void*)m_pSamples[idx]; }
            inline bool ContainSample(const SizeType idx) const { return !m_deletedID.contains(idx); }
            inline bool NeedRefine() const { return m_deletedID.size() >= (size_t)(GetNumSamples() * m_fDeletePercentageForRefine); }
            std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const
            {
                std::shared_ptr<std::vector<std::uint64_t>> buffersize(new std::vector<std::uint64_t>);
                buffersize->push_back(m_pSamples.BufferSize());
                buffersize->push_back(m_pTrees.BufferSize());
                buffersize->push_back(m_pGraph.BufferSize());
                buffersize->push_back(m_deletedID.bufferSize());
                return std::move(buffersize);
            }

            ErrorCode SaveConfig(std::ostream& p_configout) const;
            ErrorCode SaveIndexData(const std::string& p_folderPath);
            ErrorCode SaveIndexData(const std::vector<std::ostream*>& p_indexStreams);

            ErrorCode LoadConfig(Helper::IniReader& p_reader);
            ErrorCode LoadIndexData(const std::string& p_folderPath);
            ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs);

            ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension);
            ErrorCode SearchIndex(QueryResult &p_query) const;
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false);
            ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum);
            ErrorCode DeleteIndex(const SizeType& p_id);

            ErrorCode SetParameter(const char* p_param, const char* p_value);
            std::string GetParameter(const char* p_param) const;

            ErrorCode RefineIndex(const std::string& p_folderPath);
            ErrorCode RefineIndex(const std::vector<std::ostream*>& p_indexStreams);

        private:
            void SearchIndexWithDeleted(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, const Helper::Concurrent::ConcurrentSet<SizeType> &p_deleted) const;
            void SearchIndexWithoutDeleted(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space) const;
        };
    } // namespace BKT
} // namespace SPTAG

#endif // _SPTAG_BKT_INDEX_H_
