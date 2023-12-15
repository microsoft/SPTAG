// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/KDT/Index.h"
#include <chrono>

#pragma warning(disable:4242)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4244)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4127)  // conditional expression is constant

namespace SPTAG
{
    template <typename T>
    thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;
    namespace KDT
    {

        template <typename T>
        ErrorCode Index<T>::LoadConfig(Helper::IniReader& p_reader)
        {
#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
            SetParameter(RepresentStr, \
                         p_reader.GetParameter("Index", \
                         RepresentStr, \
                         std::string(#DefaultValue)).c_str()); \

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter
            return ErrorCode::Success;
        }

        template <>
        void Index<std::uint8_t>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
        {
            m_pQuantizer = quantizer;
            m_pTrees.m_pQuantizer = quantizer;
            if (m_pQuantizer)
            {
                m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<std::uint8_t>(m_iDistCalcMethod);
                m_iBaseSquare = (m_iDistCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
            }
            else
            {
                m_fComputeDistance = COMMON::DistanceCalcSelector<std::uint8_t>(m_iDistCalcMethod);
                m_iBaseSquare = (m_iDistCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>() : 1;
            }
        }

        template <typename T>
        void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
        {
            m_pQuantizer = quantizer;
            m_pTrees.m_pQuantizer = quantizer;
            if (quantizer)
            {
                SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Set non-null quantizer for index with data type other than BYTE");
            }
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs)
        {
            if (p_indexBlobs.size() < 3) return ErrorCode::LackOfInputs;

            if (m_pSamples.Load((char*)p_indexBlobs[0].Data(), m_iDataBlockSize, m_iDataCapacity) != ErrorCode::Success) return ErrorCode::FailedParseValue;
            if (m_pTrees.LoadTrees((char*)p_indexBlobs[1].Data()) != ErrorCode::Success) return ErrorCode::FailedParseValue;
            if (m_pGraph.LoadGraph((char*)p_indexBlobs[2].Data(), m_iDataBlockSize, m_iDataCapacity) != ErrorCode::Success) return ErrorCode::FailedParseValue;
            if (p_indexBlobs.size() <= 3) m_deletedID.Initialize(m_pSamples.R(), m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains);
            else if (m_deletedID.Load((char*)p_indexBlobs[3].Data(), m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains) != ErrorCode::Success) return ErrorCode::FailedParseValue;

            if (m_pSamples.R() != m_pGraph.R() || m_pSamples.R() != m_deletedID.R())
            {
                SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Index data is corrupted, please rebuild the index. Samples: %i, Graph: %i, DeletedID: %i.", m_pSamples.R(), m_pGraph.R(), m_deletedID.R());
                return ErrorCode::FailedParseValue;
            }

            omp_set_num_threads(m_iNumberOfThreads);
            m_threadPool.init();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams)
        {
            if (p_indexStreams.size() < 4) return ErrorCode::LackOfInputs;

            ErrorCode ret = ErrorCode::Success;
            if (p_indexStreams[0] == nullptr || (ret = m_pSamples.Load(p_indexStreams[0], m_iDataBlockSize, m_iDataCapacity)) != ErrorCode::Success) return ret;
            if (p_indexStreams[1] == nullptr || (ret = m_pTrees.LoadTrees(p_indexStreams[1])) != ErrorCode::Success) return ret;
            if (p_indexStreams[2] == nullptr || (ret = m_pGraph.LoadGraph(p_indexStreams[2], m_iDataBlockSize, m_iDataCapacity)) != ErrorCode::Success) return ret;
            if (p_indexStreams[3] == nullptr) m_deletedID.Initialize(m_pSamples.R(), m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains);
            else if ((ret = m_deletedID.Load(p_indexStreams[3], m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains)) != ErrorCode::Success) return ret;

            if (m_pSamples.R() != m_pGraph.R() || m_pSamples.R() != m_deletedID.R())
            {
                SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error, "Index data is corrupted, please rebuild the index. Samples: %i, Graph: %i, DeletedID: %i.", m_pSamples.R(), m_pGraph.R(), m_deletedID.R());
                return ErrorCode::FailedParseValue;
            }

            omp_set_num_threads(m_iNumberOfThreads);
            m_threadPool.init();
            return ret;
        }

        template<typename T>
        ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
        {
            auto workspace = m_workSpaceFactory->GetWorkSpace();
            if (workspace)
            {
                m_iHashTableExp = workspace->HashTableExponent();
            }
            m_workSpaceFactory->ReturnWorkSpace(std::move(workspace));

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + GetParameter(RepresentStr) + std::string("\n")).c_str());

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter
            
            IOSTRING(p_configOut, WriteString, "\n");
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams)
        {
            if (p_indexStreams.size() < 4) return ErrorCode::LackOfInputs;

            std::lock_guard<std::mutex> lock(m_dataAddLock);
            std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

            ErrorCode ret = ErrorCode::Success;
            if ((ret = m_pSamples.Save(p_indexStreams[0])) != ErrorCode::Success) return ret;
            if ((ret = m_pTrees.SaveTrees(p_indexStreams[1])) != ErrorCode::Success) return ret;
            if ((ret = m_pGraph.SaveGraph(p_indexStreams[2])) != ErrorCode::Success) return ret;
            if ((ret = m_deletedID.Save(p_indexStreams[3])) != ErrorCode::Success) return ret;
            return ret;
        }

#pragma region K-NN search
        /*
#define Search(CheckDeleted) \
        std::shared_lock<std::shared_timed_mutex> lock(*(m_pTrees.m_lock)); \
        m_pTrees.InitSearchTrees<T,Q>(m_pSamples, m_fComputeDistance, p_query, p_space); \
        m_pTrees.SearchTrees<T,Q>(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfInitialDynamicPivots); \
        while (!p_space.m_NGQueue.empty()) { \
            NodeDistPair gnode = p_space.m_NGQueue.pop(); \
            const SizeType *node = m_pGraph[gnode.node]; \
            _mm_prefetch((const char *)node, _MM_HINT_T0); \
            for (DimensionType i = 0; i < m_pGraph.m_iNeighborhoodSize; i++) \
                _mm_prefetch((const char *)(m_pSamples)[node[i]], _MM_HINT_T0); \
            CheckDeleted { \
                if (!p_query.AddPoint(gnode.node, gnode.distance) && p_space.m_iNumberOfCheckedLeaves > p_space.m_iMaxCheck) { \
                    p_query.SortResult(); return; \
                } \
            } \
            float upperBound = max(p_query.worstDist(), gnode.distance); \
            bool bLocalOpt = true; \
            for (DimensionType i = 0; i < m_pGraph.m_iNeighborhoodSize; i++) { \
                SizeType nn_index = node[i]; \
                if (nn_index < 0) break; \
                if (p_space.CheckAndSet(nn_index)) continue; \
                float distance2leaf = m_fComputeDistance(p_query.GetQuantizedTarget(), (m_pSamples)[nn_index], GetFeatureDim()); \
                if (distance2leaf <= upperBound) bLocalOpt = false; \
                p_space.m_iNumberOfCheckedLeaves++; \
                p_space.m_NGQueue.insert(NodeDistPair(nn_index, distance2leaf)); \
            } \
            if (bLocalOpt) p_space.m_iNumOfContinuousNoBetterPropagation++; \
            else p_space.m_iNumOfContinuousNoBetterPropagation = 0; \
            if (p_space.m_iNumOfContinuousNoBetterPropagation > m_iThresholdOfNumberOfContinuousNoBetterPropagation) { \
                if (p_space.m_iNumberOfTreeCheckedLeaves <= p_space.m_iNumberOfCheckedLeaves / 10) { \
                    m_pTrees.SearchTrees<T,Q>(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves); \
                } else if (gnode.distance > p_query.worstDist()) { \
                    break; \
                } \
            } \
        } \
        p_query.SortResult(); \
*/
        template<typename T>
        template<typename Q, bool(*notDeleted)(const COMMON::Labelset&, SizeType)>
        void Index<T>::Search(COMMON::QueryResultSet<T>& p_query, COMMON::WorkSpace& p_space) const
        {
            std::shared_lock<std::shared_timed_mutex> lock(*(m_pTrees.m_lock));
            m_pTrees.InitSearchTrees<T, Q>(m_pSamples, m_fComputeDistance, p_query, p_space);
            m_pTrees.SearchTrees<T, Q>(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfInitialDynamicPivots);
            while (!p_space.m_NGQueue.empty()) 
            {
                NodeDistPair gnode = p_space.m_NGQueue.pop();
                const SizeType* node = m_pGraph[gnode.node];
                _mm_prefetch((const char*)node, _MM_HINT_T0);
                for (DimensionType i = 0; i < m_pGraph.m_iNeighborhoodSize; i++)
                {
                    auto futureNode = node[i];
                    if (futureNode < 0) break;
                    _mm_prefetch((const char*)(m_pSamples)[futureNode], _MM_HINT_T0);
                }
                    
                if (notDeleted(m_deletedID, gnode.node))
                {
                    if (!p_query.AddPoint(gnode.node, gnode.distance) && p_space.m_iNumberOfCheckedLeaves > p_space.m_iMaxCheck) 
                    {
                        p_query.SortResult();
                        return;
                    }
                }

                float upperBound = max(p_query.worstDist(), gnode.distance);
                bool bLocalOpt = true;
                for (DimensionType i = 0; i < m_pGraph.m_iNeighborhoodSize; i++) 
                {
                    SizeType nn_index = node[i];
                    if (nn_index < 0) break;

                    if (p_space.CheckAndSet(nn_index)) continue;
                    float distance2leaf = m_fComputeDistance(p_query.GetQuantizedTarget(), (m_pSamples)[nn_index], GetFeatureDim());
                    if (distance2leaf <= upperBound) 
                        bLocalOpt = false;
                    p_space.m_iNumberOfCheckedLeaves++;
                    p_space.m_NGQueue.insert(NodeDistPair(nn_index, distance2leaf));
                }
                if (bLocalOpt) 
                    p_space.m_iNumOfContinuousNoBetterPropagation++;
                else 
                    p_space.m_iNumOfContinuousNoBetterPropagation = 0;
                if (p_space.m_iNumOfContinuousNoBetterPropagation > m_iThresholdOfNumberOfContinuousNoBetterPropagation) 
                {
                    if (p_space.m_iNumberOfTreeCheckedLeaves <= p_space.m_iNumberOfCheckedLeaves / 10) 
                    {
                        m_pTrees.SearchTrees<T, Q>(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves);
                    }
                    else if (gnode.distance > p_query.worstDist()) 
                    {
                        break;
                    }
                }
            }
            p_query.SortResult();
        }

        namespace StaticDispatch
        {
            bool AlwaysTrue(const COMMON::Labelset& deletedIDs, SizeType node)
            {
                return true;
            }

            bool CheckIfNotDeleted(const COMMON::Labelset& deletedIDs, SizeType node)
            {
                return !deletedIDs.Contains(node);
            }
        };

        template <typename T>
        template <typename Q>
        void Index<T>::SearchIndex(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, bool p_searchDeleted) const
        {
            if (m_deletedID.Count() == 0 || p_searchDeleted) {
                Search<Q, StaticDispatch::AlwaysTrue>(p_query, p_space);
            }
            else {
                Search<Q, StaticDispatch::CheckIfNotDeleted>(p_query, p_space);
            }
        }

        template<typename T>
        ErrorCode
            Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
        {
            if (!m_bReady) return ErrorCode::EmptyIndex;

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace) {
                workSpace.reset(new COMMON::WorkSpace());
                workSpace->Initialize(max(m_iMaxCheck, m_pGraph.m_iMaxCheckForRefineGraph), m_iHashTableExp);
            }
            workSpace->Reset(m_iMaxCheck, p_query.GetResultNum());

            COMMON::QueryResultSet<T>* p_results = (COMMON::QueryResultSet<T>*) & p_query;

            if (m_pQuantizer)
            {
                if (!(p_results->HasQuantizedTarget())) {
                    p_results->SetTarget(p_results->GetTarget(), m_pQuantizer);
                }
                switch (m_pQuantizer->GetReconstructType())
                {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
                SearchIndex<Type>(*p_results, *workSpace, p_searchDeleted); \
                break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                default: break;
                }
            }
            else
            {            
                SearchIndex<T>(*p_results, *workSpace, p_searchDeleted);
            }

            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            
                if (p_query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < p_query.GetResultNum(); ++i)
                {
                    SizeType result = p_query.GetResult(i)->VID;
                    p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
                }
            }

            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SearchIndexWithFilter(QueryResult& p_query, std::function<bool(const ByteArray&)> filterFunc, int maxCheck, bool p_searchDeleted) const
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not Support Filter on KDT Index!\n");
            return ErrorCode::Fail;
        }

        template <typename T>
        ErrorCode Index<T>::RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted) const
        {
            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace) {
                workSpace.reset(new COMMON::WorkSpace());
                workSpace->Initialize(max(m_iMaxCheck, m_pGraph.m_iMaxCheckForRefineGraph), m_iHashTableExp);
            }
            workSpace->Reset(m_pGraph.m_iMaxCheckForRefineGraph, p_query.GetResultNum());

            COMMON::QueryResultSet<T>* p_results = (COMMON::QueryResultSet<T>*) & p_query;

            if (m_pQuantizer)
            {
                if (!(p_results->HasQuantizedTarget())) {
                    p_results->SetTarget(p_results->GetTarget(), m_pQuantizer);
                }
                switch (m_pQuantizer->GetReconstructType())
                {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
                SearchIndex<Type>(*p_results, *workSpace, p_searchDeleted); \
                break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                default: break;
                }
            }
            else
            {
                SearchIndex<T>(*p_results, *workSpace, p_searchDeleted);
            }
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SearchTree(QueryResult& p_query) const
        {
            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace) {
                workSpace.reset(new COMMON::WorkSpace());
                workSpace->Initialize(max(m_iMaxCheck, m_pGraph.m_iMaxCheckForRefineGraph), m_iHashTableExp);
            }
            workSpace->Reset(m_pGraph.m_iMaxCheckForRefineGraph, p_query.GetResultNum());

            COMMON::QueryResultSet<T>* p_results = (COMMON::QueryResultSet<T>*)&p_query;

            if (m_pQuantizer)
            {
                if (!(p_results->HasQuantizedTarget())) {
                    p_results->SetTarget(p_results->GetTarget(), m_pQuantizer);
                }
                switch (m_pQuantizer->GetReconstructType())
                {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
                    m_pTrees.InitSearchTrees<T, Type>(m_pSamples, m_fComputeDistance, *p_results, *workSpace); \
                    m_pTrees.SearchTrees<T, Type>(m_pSamples, m_fComputeDistance, *p_results, *workSpace, m_iNumberOfInitialDynamicPivots); \
                    break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                default: break;
                }
            }
            else
            {
                m_pTrees.InitSearchTrees<T, T>(m_pSamples, m_fComputeDistance, *p_results, *workSpace);
                m_pTrees.SearchTrees<T, T>(m_pSamples, m_fComputeDistance, *p_results, *workSpace, m_iNumberOfInitialDynamicPivots);
            }

            BasicResult * res = p_query.GetResults();
            for (int i = 0; i < p_query.GetResultNum(); i++)
            {
                auto& cell = workSpace->m_NGQueue.pop();
                res[i].VID = cell.node;
                res[i].Dist = cell.distance;
            }
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));

            return ErrorCode::Success;
        }
#pragma endregion

        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized, bool p_shareOwnership)
        {
            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;

            omp_set_num_threads(m_iNumberOfThreads);

            m_pSamples.Initialize(p_vectorNum, p_dimension, m_iDataBlockSize, m_iDataCapacity, (T*)p_data, p_shareOwnership);
            m_deletedID.Initialize(p_vectorNum, m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains);

            if (DistCalcMethod::Cosine == m_iDistCalcMethod && !p_normalized)
            {
                int base = m_pQuantizer ? m_pQuantizer->GetBase() : COMMON::Utils::GetBase<T>();
#pragma omp parallel for
                for (SizeType i = 0; i < GetNumSamples(); i++) {
                    COMMON::Utils::Normalize(m_pSamples[i], GetFeatureDim(), base);
                }
            }

            m_threadPool.init();

            auto t1 = std::chrono::high_resolution_clock::now();

            m_pTrees.BuildTrees<T>(m_pSamples, m_iNumberOfThreads);

            auto t2 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build Tree time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count());
            m_pGraph.BuildGraph<T>(this);
            auto t3 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build Graph time (s): %lld\n", std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count());

            m_bReady = true;
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex)
        {
            p_newIndex.reset(new Index<T>());
            Index<T>* ptr = (Index<T>*)p_newIndex.get();

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
            ptr->VarName =  VarName; \

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter

            std::lock_guard<std::mutex> lock(m_dataAddLock);
            std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

            SizeType newR = GetNumSamples();

            std::vector<SizeType> indices;
            std::vector<SizeType> reverseIndices(newR);
            for (SizeType i = 0; i < newR; i++) {
                if (!m_deletedID.Contains(i)) {
                    indices.push_back(i);
                    reverseIndices[i] = i;
                }
                else {
                    while (m_deletedID.Contains(newR - 1) && newR > i) newR--;
                    if (newR == i) break;
                    indices.push_back(newR - 1);
                    reverseIndices[newR - 1] = i;
                    newR--;
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Refine... from %d -> %d\n", GetNumSamples(), newR);
            if (newR == 0) return ErrorCode::EmptyIndex;

            ptr->m_threadPool.init();

            ErrorCode ret = ErrorCode::Success;
            if ((ret = m_pSamples.Refine(indices, ptr->m_pSamples)) != ErrorCode::Success) return ret;
            if (nullptr != m_pMetadata && (ret = m_pMetadata->RefineMetadata(indices, ptr->m_pMetadata, m_iDataBlockSize, m_iDataCapacity, m_iMetaRecordSize)) != ErrorCode::Success) return ret;

            ptr->m_deletedID.Initialize(newR, m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains);
            COMMON::KDTree* newtree = &(ptr->m_pTrees);

            (*newtree).BuildTrees<T>(ptr->m_pSamples, omp_get_num_threads());
            m_pGraph.RefineGraph<T>(this, indices, reverseIndices, nullptr, &(ptr->m_pGraph));
            if (HasMetaMapping()) ptr->BuildMetaMapping(false);
            ptr->m_bReady = true;
            return ret;
        }

        template <typename T>
        ErrorCode Index<T>::RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams, IAbortOperation* p_abort)
        {
            std::lock_guard<std::mutex> lock(m_dataAddLock);
            std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

            SizeType newR = GetNumSamples();

            std::vector<SizeType> indices;
            std::vector<SizeType> reverseIndices(newR);
            for (SizeType i = 0; i < newR; i++) {
                if (!m_deletedID.Contains(i)) {
                    indices.push_back(i);
                    reverseIndices[i] = i;
                }
                else {
                    while (m_deletedID.Contains(newR - 1) && newR > i) newR--;
                    if (newR == i) break;
                    indices.push_back(newR - 1);
                    reverseIndices[newR - 1] = i;
                    newR--;
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Refine... from %d -> %d\n", GetNumSamples(), newR);
            if (newR == 0) return ErrorCode::EmptyIndex;

            ErrorCode ret = ErrorCode::Success;
            if ((ret = m_pSamples.Refine(indices, p_indexStreams[0])) != ErrorCode::Success) return ret;

            if (p_abort != nullptr && p_abort->ShouldAbort()) return ErrorCode::ExternalAbort;

            COMMON::KDTree newTrees(m_pTrees);
            newTrees.BuildTrees<T>(m_pSamples, omp_get_num_threads(), &indices);
#pragma omp parallel for
            for (SizeType i = 0; i < newTrees.size(); i++) {
                if (newTrees[i].left < 0)
                    newTrees[i].left = -reverseIndices[-newTrees[i].left - 1] - 1;
                if (newTrees[i].right < 0)
                    newTrees[i].right = -reverseIndices[-newTrees[i].right - 1] - 1;
            }
            if ((ret = newTrees.SaveTrees(p_indexStreams[1])) != ErrorCode::Success) return ret;

            if (p_abort != nullptr && p_abort->ShouldAbort()) return ErrorCode::ExternalAbort;

            if ((ret = m_pGraph.RefineGraph<T>(this, indices, reverseIndices, p_indexStreams[2], nullptr)) != ErrorCode::Success) return ret;

            COMMON::Labelset newDeletedID;
            newDeletedID.Initialize(newR, m_iDataBlockSize, m_iDataCapacity, COMMON::Labelset::InvalidIDBehavior::AlwaysContains);
            if ((ret = newDeletedID.Save(p_indexStreams[3])) != ErrorCode::Success) return ret;

            if (nullptr != m_pMetadata) {
                if (p_indexStreams.size() < 6) return ErrorCode::LackOfInputs;
                if ((ret = m_pMetadata->RefineMetadata(indices, p_indexStreams[4], p_indexStreams[5])) != ErrorCode::Success) return ret;
            }
            return ret;
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const void* p_vectors, SizeType p_vectorNum) {
            const T* ptr_v = (const T*)p_vectors;
#pragma omp parallel for schedule(dynamic)
            for (SizeType i = 0; i < p_vectorNum; i++) {
                COMMON::QueryResultSet<T> query(ptr_v + i * GetFeatureDim(), m_pGraph.m_iCEF);
                SearchIndex(query);

                for (int j = 0; j < m_pGraph.m_iCEF; j++) {
                    if (query.GetResult(j)->Dist < 1e-6) {
                        DeleteIndex(query.GetResult(j)->VID);
                    }
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const SizeType& p_id) {
            if (!m_bReady) return ErrorCode::EmptyIndex;

            std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
            if (m_deletedID.Insert(p_id)) return ErrorCode::Success;
            return ErrorCode::VectorNotFound;
        }

        template <typename T>
        ErrorCode Index<T>::AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex, bool p_normalized)
        {
            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;

            SizeType begin, end;
            ErrorCode ret;
            {
                std::lock_guard<std::mutex> lock(m_dataAddLock);

                begin = GetNumSamples();
                end = begin + p_vectorNum;

                if (begin == 0) {
                    if (p_metadataSet != nullptr) {
                        m_pMetadata.reset(new MemMetadataSet(m_iDataBlockSize, m_iDataCapacity, m_iMetaRecordSize));
                        m_pMetadata->AddBatch(*p_metadataSet);
                        if (p_withMetaIndex) BuildMetaMapping(false);
                    }
                    if ((ret = BuildIndex(p_data, p_vectorNum, p_dimension, p_normalized)) != ErrorCode::Success) return ret;
                    return ErrorCode::Success;
                }

                if (p_dimension != GetFeatureDim()) return ErrorCode::DimensionSizeMismatch;

                if (m_pSamples.AddBatch((const T*)p_data, p_vectorNum) != ErrorCode::Success ||
                    m_pGraph.AddBatch(p_vectorNum) != ErrorCode::Success ||
                    m_deletedID.AddBatch(p_vectorNum) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Memory Error: Cannot alloc space for vectors!\n");
                    m_pSamples.SetR(begin);
                    m_pGraph.SetR(begin);
                    m_deletedID.SetR(begin);
                    return ErrorCode::MemoryOverFlow;
                }

                if (m_pMetadata != nullptr) {
                    if (p_metadataSet != nullptr) {
                        m_pMetadata->AddBatch(*p_metadataSet);
                        if (HasMetaMapping()) {
                            for (SizeType i = begin; i < end; i++) {
                                ByteArray meta = m_pMetadata->GetMetadata(i);
                                std::string metastr((char*)meta.Data(), meta.Length());
                                UpdateMetaMapping(metastr, i);
                            }
                        }
                    }
                    else {
                        for (SizeType i = begin; i < end; i++) m_pMetadata->Add(ByteArray::c_empty);
                    }
                }
            }

            if (DistCalcMethod::Cosine == m_iDistCalcMethod && !p_normalized)
            {
                int base = m_pQuantizer ? m_pQuantizer->GetBase() : COMMON::Utils::GetBase<T>();
                for (SizeType i = begin; i < end; i++) {
                    COMMON::Utils::Normalize((T*)m_pSamples[i], GetFeatureDim(), base);
                }
            }

            if (end - m_pTrees.sizePerTree() >= m_addCountForRebuild && m_threadPool.jobsize() == 0) {
                m_threadPool.add(new RebuildJob(&m_pSamples, &m_pTrees, &m_pGraph));
            }

            for (SizeType node = begin; node < end; node++)
            {
                m_pGraph.RefineNode<T>(this, node, true, true, m_pGraph.m_iAddCEF);
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
            Index<T>::UpdateIndex()
        {
            omp_set_num_threads(m_iNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
            Index<T>::SetParameter(const char* p_param, const char* p_value, const char* p_section)
        {
            if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter

            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod")) {
                m_fComputeDistance = m_pQuantizer ? m_pQuantizer->DistanceCalcSelector<T>(m_iDistCalcMethod) : COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);
                auto base = m_pQuantizer ? m_pQuantizer->GetBase() : COMMON::Utils::GetBase<T>();
                m_iBaseSquare = (m_iDistCalcMethod == DistCalcMethod::Cosine) ? base * base : 1;
            }
            return ErrorCode::Success;
        }


        template <typename T>
        std::string
            Index<T>::GetParameter(const char* p_param, const char* p_section) const
        {
            if (nullptr == p_param) return std::string();

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


