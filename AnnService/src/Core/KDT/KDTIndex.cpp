// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/KDT/Index.h"

#pragma warning(disable:4996)  // 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
#pragma warning(disable:4242)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4244)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4127)  // conditional expression is constant

namespace SPTAG
{
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

        template <typename T>
        ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs)
        {
            if (p_indexBlobs.size() < 3) return ErrorCode::LackOfInputs;

            if (!m_pSamples.Load((char*)p_indexBlobs[0].Data())) return ErrorCode::FailedParseValue;
            if (!m_pTrees.LoadTrees((char*)p_indexBlobs[1].Data())) return ErrorCode::FailedParseValue;
            if (!m_pGraph.LoadGraph((char*)p_indexBlobs[2].Data())) return ErrorCode::FailedParseValue;
            if (p_indexBlobs.size() > 3 && !m_deletedID.Load((char*)p_indexBlobs[3].Data())) return ErrorCode::FailedParseValue;

            omp_set_num_threads(m_iNumberOfThreads);
            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_iMaxCheck, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);
            m_threadPool.init();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexData(const std::string& p_folderPath)
        {
            if (!m_pSamples.Load(p_folderPath + m_sDataPointsFilename)) return ErrorCode::Fail;
            if (!m_pTrees.LoadTrees(p_folderPath + m_sKDTFilename)) return ErrorCode::Fail;
            if (!m_pGraph.LoadGraph(p_folderPath + m_sGraphFilename)) return ErrorCode::Fail;
            if (!m_deletedID.Load(p_folderPath + m_sDeleteDataPointsFilename)) return ErrorCode::Fail;

            omp_set_num_threads(m_iNumberOfThreads);
            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_iMaxCheck, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);
            m_threadPool.init();
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveConfig(std::ostream& p_configOut) const
        {
#define DefineKDTParameter(VarName, VarType, DefaultValue, RepresentStr) \
    p_configOut << RepresentStr << "=" << GetParameter(RepresentStr) << std::endl;

#include "inc/Core/KDT/ParameterDefinitionList.h"
#undef DefineKDTParameter
            p_configOut << std::endl;
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveIndexData(const std::string& p_folderPath)
        {
            std::lock_guard<std::mutex> lock(m_dataAddLock);
            std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

            if (!m_pSamples.Save(p_folderPath + m_sDataPointsFilename)) return ErrorCode::Fail;
            if (!m_pTrees.SaveTrees(p_folderPath + m_sKDTFilename)) return ErrorCode::Fail;
            if (!m_pGraph.SaveGraph(p_folderPath + m_sGraphFilename)) return ErrorCode::Fail;
            if (!m_deletedID.Save(p_folderPath + m_sDeleteDataPointsFilename)) return ErrorCode::Fail;
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveIndexData(const std::vector<std::ostream*>& p_indexStreams)
        {
            if (p_indexStreams.size() < 4) return ErrorCode::LackOfInputs;

            std::lock_guard<std::mutex> lock(m_dataAddLock);
            std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

            if (!m_pSamples.Save(*p_indexStreams[0])) return ErrorCode::Fail;
            if (!m_pTrees.SaveTrees(*p_indexStreams[1])) return ErrorCode::Fail;
            if (!m_pGraph.SaveGraph(*p_indexStreams[2])) return ErrorCode::Fail;
            if (!m_deletedID.Save(*p_indexStreams[3])) return ErrorCode::Fail;
            return ErrorCode::Success;
        }

#pragma region K-NN search

#define Search(CheckDeleted1) \
        std::shared_lock<std::shared_timed_mutex> lock(*(m_pTrees.m_lock)); \
        m_pTrees.InitSearchTrees(this, p_query, p_space); \
        m_pTrees.SearchTrees(this, p_query, p_space, m_iNumberOfInitialDynamicPivots); \
        while (!p_space.m_NGQueue.empty()) { \
            COMMON::HeapCell gnode = p_space.m_NGQueue.pop(); \
            const SizeType *node = m_pGraph[gnode.node]; \
            _mm_prefetch((const char *)node, _MM_HINT_T0); \
            CheckDeleted1 { \
                if (!p_query.AddPoint(gnode.node, gnode.distance) && p_space.m_iNumberOfCheckedLeaves > p_space.m_iMaxCheck) { \
                    p_query.SortResult(); return; \
                } \
            } \
            for (DimensionType i = 0; i < m_pGraph.m_iNeighborhoodSize; i++) \
                _mm_prefetch((const char *)(m_pSamples)[node[i]], _MM_HINT_T0); \
            float upperBound = max(p_query.worstDist(), gnode.distance); \
            bool bLocalOpt = true; \
            for (DimensionType i = 0; i < m_pGraph.m_iNeighborhoodSize; i++) { \
                SizeType nn_index = node[i]; \
                if (nn_index < 0) break; \
                if (p_space.CheckAndSet(nn_index)) continue; \
                float distance2leaf = m_fComputeDistance(p_query.GetTarget(), (m_pSamples)[nn_index], GetFeatureDim()); \
                if (distance2leaf <= upperBound) bLocalOpt = false; \
                p_space.m_iNumberOfCheckedLeaves++; \
                p_space.m_NGQueue.insert(COMMON::HeapCell(nn_index, distance2leaf)); \
            } \
            if (bLocalOpt) p_space.m_iNumOfContinuousNoBetterPropagation++; \
            else p_space.m_iNumOfContinuousNoBetterPropagation = 0; \
            if (p_space.m_iNumOfContinuousNoBetterPropagation > m_iThresholdOfNumberOfContinuousNoBetterPropagation) { \
                if (p_space.m_iNumberOfTreeCheckedLeaves <= p_space.m_iNumberOfCheckedLeaves / 10) { \
                    m_pTrees.SearchTrees(this, p_query, p_space, m_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves); \
                } else if (gnode.distance > p_query.worstDist()) { \
                    break; \
                } \
            } \
        } \
        p_query.SortResult(); \

        template <typename T>
        void Index<T>::SearchIndexWithoutDeleted(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space) const
        {
            Search(if (!m_deletedID.Contains(gnode.node)))
        }

        template <typename T>
        void Index<T>::SearchIndexWithDeleted(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space) const
        {
            Search(;)
        }

        template<typename T>
        ErrorCode
            Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
        {
            auto workSpace = m_workSpacePool->Rent();
            workSpace->Reset(m_iMaxCheck);

            if (m_deletedID.Count() == 0 || p_searchDeleted)
                SearchIndexWithDeleted(*((COMMON::QueryResultSet<T>*)&p_query), *workSpace);
            else
                SearchIndexWithoutDeleted(*((COMMON::QueryResultSet<T>*)&p_query), *workSpace);

            m_workSpacePool->Return(workSpace);

            if (p_query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < p_query.GetResultNum(); ++i)
                {
                    SizeType result = p_query.GetResult(i)->VID;
                    p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadata(result));
                }
            }
            return ErrorCode::Success;
        }
#pragma endregion

        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension)
        {
            omp_set_num_threads(m_iNumberOfThreads);

            m_pSamples.Initialize(p_vectorNum, p_dimension, (T*)p_data, false);
            m_deletedID.Initialize(p_vectorNum);

            if (DistCalcMethod::Cosine == m_iDistCalcMethod)
            {
                int base = COMMON::Utils::GetBase<T>();
#pragma omp parallel for
                for (SizeType i = 0; i < GetNumSamples(); i++) {
                    COMMON::Utils::Normalize(m_pSamples[i], GetFeatureDim(), base);
                }
            }

            m_workSpacePool.reset(new COMMON::WorkSpacePool(m_pGraph.m_iMaxCheckForRefineGraph, GetNumSamples()));
            m_workSpacePool->Init(m_iNumberOfThreads);
            m_threadPool.init();

            m_pTrees.BuildTrees<T>(this);
            m_pGraph.BuildGraph<T>(this);
            
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

            std::cout << "Refine... from " << GetNumSamples() << "->" << newR << std::endl;

            ptr->m_workSpacePool.reset(new COMMON::WorkSpacePool(ptr->m_pGraph.m_iMaxCheckForRefineGraph, newR));
            ptr->m_workSpacePool->Init(m_iNumberOfThreads);
            ptr->m_threadPool.init();

            if (false == m_pSamples.Refine(indices, ptr->m_pSamples)) return ErrorCode::Fail;
            if (nullptr != m_pMetadata && ErrorCode::Success != m_pMetadata->RefineMetadata(indices, ptr->m_pMetadata)) return ErrorCode::Fail;

            ptr->m_deletedID.Initialize(newR);
            COMMON::KDTree* newtree = &(ptr->m_pTrees);
            (*newtree).BuildTrees<T>(ptr);
            m_pGraph.RefineGraph<T>(this, indices, reverseIndices, nullptr, &(ptr->m_pGraph));
            if (m_pMetaToVec != nullptr) ptr->BuildMetaMapping();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::RefineIndex(const std::vector<std::ostream*>& p_indexStreams)
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

            std::cout << "Refine... from " << GetNumSamples() << "->" << newR << std::endl;

            if (false == m_pSamples.Refine(indices, *p_indexStreams[0])) return ErrorCode::Fail;
            if (nullptr != m_pMetadata && (p_indexStreams.size() < 6 || ErrorCode::Success != m_pMetadata->RefineMetadata(indices, *p_indexStreams[4], *p_indexStreams[5]))) return ErrorCode::Fail;

            COMMON::KDTree newTrees(m_pTrees);
            newTrees.BuildTrees<T>(this, &indices);
#pragma omp parallel for
            for (SizeType i = 0; i < newTrees.size(); i++) {
                if (newTrees[i].left < 0)
                    newTrees[i].left = -reverseIndices[-newTrees[i].left - 1] - 1;
                if (newTrees[i].right < 0)
                    newTrees[i].right = -reverseIndices[-newTrees[i].right - 1] - 1;
            }
            newTrees.SaveTrees(*p_indexStreams[1]);

            m_pGraph.RefineGraph<T>(this, indices, reverseIndices, p_indexStreams[2], nullptr);

            COMMON::Labelset newDeletedID;
            newDeletedID.Initialize(newR);
            newDeletedID.Save(*p_indexStreams[3]);
            return ErrorCode::Success;
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

            std::vector<std::ostream*> streams;
            streams.push_back(new std::ofstream(folderPath + m_sDataPointsFilename, std::ios::binary));
            streams.push_back(new std::ofstream(folderPath + m_sKDTFilename, std::ios::binary));
            streams.push_back(new std::ofstream(folderPath + m_sGraphFilename, std::ios::binary));
            streams.push_back(new std::ofstream(folderPath + m_sDeleteDataPointsFilename, std::ios::binary));
            if (nullptr != m_pMetadata)
            {
                streams.push_back(new std::ofstream(folderPath + m_sMetadataFile, std::ios::binary));
                streams.push_back(new std::ofstream(folderPath + m_sMetadataIndexFile, std::ios::binary));
            }

            for (size_t i = 0; i < streams.size(); i++) 
                if (!(((std::ofstream*)streams[i])->is_open())) return ErrorCode::FailedCreateFile;

            ErrorCode ret = RefineIndex(streams);
            
            for (size_t i = 0; i < streams.size(); i++) 
            {
                ((std::ofstream*)streams[i])->close();
                delete streams[i];
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

                for (int i = 0; i < m_pGraph.m_iCEF; i++) {
                    if (query.GetResult(i)->Dist < 1e-6) {
                        DeleteIndex(query.GetResult(i)->VID);
                    }
                }
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const SizeType& p_id) {
            std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
            m_deletedID.Insert(p_id);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex)
        {
            SizeType begin, end;
            ErrorCode ret;
            {
                std::lock_guard<std::mutex> lock(m_dataAddLock);

                begin = GetNumSamples();
                end = begin + p_vectorNum;

                if (begin == 0) {
                    if ((ret = BuildIndex(p_data, p_vectorNum, p_dimension)) != ErrorCode::Success) return ret;
                    m_pMetadata = std::move(p_metadataSet);
                    if (p_withMetaIndex && m_pMetadata != nullptr)
                    {
                        BuildMetaMapping();
                    }
                    return ErrorCode::Success;
                }

                if (p_dimension != GetFeatureDim()) return ErrorCode::FailedParseValue;

                if (m_pSamples.AddBatch((const T*)p_data, p_vectorNum) != ErrorCode::Success ||
                    m_pGraph.AddBatch(p_vectorNum) != ErrorCode::Success ||
                    m_deletedID.AddBatch(p_vectorNum) != ErrorCode::Success) {
                    std::cout << "Memory Error: Cannot alloc space for vectors" << std::endl;
                    m_pSamples.SetR(begin);
                    m_pGraph.SetR(begin);
                    m_deletedID.SetR(begin);
                    return ErrorCode::MemoryOverFlow;
                }
                if (DistCalcMethod::Cosine == m_iDistCalcMethod)
                {
                    int base = COMMON::Utils::GetBase<T>();
                    for (SizeType i = begin; i < end; i++) {
                        COMMON::Utils::Normalize((T*)m_pSamples[i], GetFeatureDim(), base);
                    }
                }

                if (m_pMetadata != nullptr) {
                    m_pMetadata->AddBatch(*p_metadataSet);

                    if (m_pMetaToVec != nullptr) {
                        for (SizeType i = begin; i < end; i++) {
                            ByteArray meta = m_pMetadata->GetMetadata(i);
                            std::string metastr((char*)meta.Data(), meta.Length());
                            auto iter = m_pMetaToVec->find(metastr);
                            if (iter != m_pMetaToVec->end()) DeleteIndex(iter->second);
                            (*m_pMetaToVec)[metastr] = i;
                        }
                    }
                }
            }

            if (end - m_pTrees.sizePerTree() >= m_addCountForRebuild && m_threadPool.jobsize() == 0) {
                m_threadPool.add(new RebuildJob(this, &m_pTrees, &m_pGraph));
            }

            for (SizeType node = begin; node < end; node++)
            {
                m_pGraph.RefineNode<T>(this, node, true, true);
            }
            //std::cout << "Add " << p_vectorNum << " vectors" << std::endl;
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
            Index<T>::SetParameter(const char* p_param, const char* p_value)
        {
            if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

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


