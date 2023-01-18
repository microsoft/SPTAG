// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraFullGraphSearcher.h"
#include "inc/Core/SPANN/ExtraRocksDBController.h"
#include <shared_mutex>
#include <chrono>
#include <random>

#pragma warning(disable:4242)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4244)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4127)  // conditional expression is constant

namespace SPTAG
{
    template <typename T>
    thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;
    namespace SPANN
    {
        std::atomic_int ExtraWorkSpace::g_spaceCount(0);
        EdgeCompare Selection::g_edgeComparer;

        std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskIO> { return std::shared_ptr<Helper::DiskIO>(new Helper::AsyncFileIO()); };

        template <typename T>
        bool Index<T>::CheckHeadIndexType() {
            SPTAG::VectorValueType v1 = m_index->GetVectorValueType(), v2 = GetEnumValueType<T>();
            if (v1 != v2) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
                    SPTAG::Helper::Convert::ConvertToString(v1).c_str(),
                    SPTAG::Helper::Convert::ConvertToString(v2).c_str()
                );
                if (!m_pQuantizer) return false;
            }
            return true;
        }

        template <typename T>
        void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
        {
            m_pQuantizer = quantizer;
            if (m_pQuantizer)
            {
                m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
            }
            else
            {
                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>() : 1;
            }  
            if (m_index)
            {
                m_index->SetQuantizer(quantizer);
            }
        }

        template <typename T>
        ErrorCode Index<T>::LoadConfig(Helper::IniReader& p_reader)
        {
            IndexAlgoType algoType = p_reader.GetParameter("Base", "IndexAlgoType", IndexAlgoType::Undefined);
            VectorValueType valueType = p_reader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
            if ((m_index = CreateInstance(algoType, valueType)) == nullptr) return ErrorCode::FailedParseValue;

            std::string sections[] = { "Base", "SelectHead", "BuildHead", "BuildSSDIndex" };
            for (int i = 0; i < 4; i++) {
                auto parameters = p_reader.GetParameters(sections[i].c_str());
                for (auto iter = parameters.begin(); iter != parameters.end(); iter++) {
                    SetParameter(iter->first.c_str(), iter->second.c_str(), sections[i].c_str());
                }
            }

            if (m_pQuantizer)
            {
                m_pQuantizer->SetEnableADC(m_options.m_enableADC);
            }

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs)
        {
            m_index->SetQuantizer(m_pQuantizer);
            if (m_index->LoadIndexDataFromMemory(p_indexBlobs) != ErrorCode::Success) return ErrorCode::Fail;

            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            //m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            //m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            m_index->SetReady(true);

            if (m_pQuantizer)
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<std::uint8_t>());
            }
            else
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<T>());
            }
            
            if (!m_extraSearcher->LoadIndex(m_options)) return ErrorCode::Fail;

            m_vectorTranslateMap.reset((std::uint64_t*)(p_indexBlobs.back().Data()), [=](std::uint64_t* ptr) {});
           
            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams)
        {
            m_index->SetQuantizer(m_pQuantizer);
            if (m_index->LoadIndexData(p_indexStreams) != ErrorCode::Success) return ErrorCode::Fail;

            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            //m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            //m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            m_index->SetReady(true);

            if (m_pQuantizer)
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<std::uint8_t>());
            }
            else
            {
                m_extraSearcher.reset(new ExtraFullGraphSearcher<T>());
            }

            if (!m_extraSearcher->LoadIndex(m_options)) return ErrorCode::Fail;

            m_versionMap.Load(m_options.m_fullDeletedIDFile, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);

            m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
            IOBINARY(p_indexStreams[m_index->GetIndexFiles()->size()], ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), reinterpret_cast<char*>(m_vectorTranslateMap.get()));

            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);

            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
        {
            IOSTRING(p_configOut, WriteString, "[Base]\n");
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

            IOSTRING(p_configOut, WriteString, "[SelectHead]\n");
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

            IOSTRING(p_configOut, WriteString, "[BuildHead]\n");
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

            m_index->SaveConfig(p_configOut);

            Helper::Convert::ConvertStringTo<int>(m_index->GetParameter("HashTableExponent").c_str(), m_options.m_hashExp);
            IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr) \
                IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) + std::string("\n")).c_str()); \

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

            IOSTRING(p_configOut, WriteString, "\n");
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams)
        {
            if (m_index == nullptr || m_vectorTranslateMap == nullptr) return ErrorCode::EmptyIndex;
            
            ErrorCode ret;
            if ((ret = m_index->SaveIndexData(p_indexStreams)) != ErrorCode::Success) return ret;

            IOBINARY(p_indexStreams[m_index->GetIndexFiles()->size()], WriteBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char*)(m_vectorTranslateMap.get()));
            m_versionMap.Save(m_options.m_fullDeletedIDFile);
            return ErrorCode::Success;
        }

#pragma region K-NN search

        template<typename T>
        ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
        {
            if (!m_bReady) return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T>* p_queryResults;
            if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum) 
                p_queryResults = (COMMON::QueryResultSet<T>*) & p_query;
            else
                p_queryResults = new COMMON::QueryResultSet<T>((const T*)p_query.GetTarget(), m_options.m_searchInternalResultNum);

            m_index->SearchIndex(*p_queryResults);
            
            if (m_extraSearcher != nullptr) {
                auto workSpace = m_workSpaceFactory->GetWorkSpace();
                if (!workSpace) {
                    workSpace.reset(new ExtraWorkSpace());
                    workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
                }
                else {
                    workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
                }
                workSpace->m_deduper.clear();
                workSpace->m_postingIDs.clear();

                float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
                for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
                {
                    auto res = p_queryResults->GetResult(i);
                    if (res->VID == -1) break;

                    auto postingID = res->VID;
                    res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                    if (res->VID == MaxSize) {
                        res->VID = -1;
                        res->Dist = MaxDist;
                    }

                    // Don't do disk reads for irrelevant pages
                    if (workSpace->m_postingIDs.size() >= m_options.m_searchInternalResultNum ||
                        (limitDist > 0.1 && res->Dist > limitDist) || 
                        !m_extraSearcher->CheckValidPosting(postingID)) 
                        continue;
                    workSpace->m_postingIDs.emplace_back(postingID);
                }

                p_queryResults->Reverse();
                m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, nullptr);
                m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
                p_queryResults->SortResult();
            }

            if (p_query.GetResultNum() < m_options.m_searchInternalResultNum) {
                std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(), p_query.GetResults());
                delete p_queryResults;
            }

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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not Support Filter on SPANN Index!\n");
            return ErrorCode::Fail;
        }

        template <typename T>
        ErrorCode Index<T>::SearchDiskIndex(QueryResult& p_query, SearchStats* p_stats) const
        {
            if (nullptr == m_extraSearcher) return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T>* p_queryResults = (COMMON::QueryResultSet<T>*) & p_query;

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace) {
                workSpace.reset(new ExtraWorkSpace());
                workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            else {
                workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            workSpace->m_deduper.clear();
            workSpace->m_postingIDs.clear();

            float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
            int i = 0;
            for (; i < m_options.m_searchInternalResultNum; ++i)
            {
                auto res = p_queryResults->GetResult(i);
                if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist)) break;
                if (m_extraSearcher->CheckValidPosting(res->VID)) 
                {
                    workSpace->m_postingIDs.emplace_back(res->VID);
                }
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (res->VID == MaxSize) 
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }

            for (; i < p_queryResults->GetResultNum(); ++i)
            {
                auto res = p_queryResults->GetResult(i);
                if (res->VID == -1) break;
                res->VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (res->VID == MaxSize) 
                {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }

            p_queryResults->Reverse();
            m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, p_stats);
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            p_queryResults->SortResult();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::DebugSearchDiskIndex(QueryResult& p_query, int p_subInternalResultNum, int p_internalResultNum,
            SearchStats* p_stats, std::set<int>* truth, std::map<int, std::set<int>>* found) const
        {
            if (nullptr == m_extraSearcher) return ErrorCode::EmptyIndex;

            COMMON::QueryResultSet<T> newResults(*((COMMON::QueryResultSet<T>*)&p_query));
            for (int i = 0; i < newResults.GetResultNum(); ++i)
            {
                auto res = newResults.GetResult(i);
                if (res->VID == -1) break;

                auto global_VID = static_cast<SizeType>((m_vectorTranslateMap.get())[res->VID]);
                if (truth && truth->count(global_VID)) (*found)[res->VID].insert(global_VID);
                res->VID = global_VID;
                if (res->VID == MaxSize) {
                    res->VID = -1;
                    res->Dist = MaxDist;
                }
            }
            newResults.Reverse();

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace) {
                workSpace.reset(new ExtraWorkSpace());
                workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            else {
                workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            workSpace->m_deduper.clear();

            int partitions = (p_internalResultNum + p_subInternalResultNum - 1) / p_subInternalResultNum;
            float limitDist = p_query.GetResult(0)->Dist * m_options.m_maxDistRatio;
            for (SizeType p = 0; p < partitions; p++) {
                int subInternalResultNum = min(p_subInternalResultNum, p_internalResultNum - p_subInternalResultNum * p);

                workSpace->m_postingIDs.clear();

                for (int i = p * p_subInternalResultNum; i < p * p_subInternalResultNum + subInternalResultNum; i++)
                {
                    auto res = p_query.GetResult(i);
                    if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist)) break;
                    if (!m_extraSearcher->CheckValidPosting(res->VID)) continue;
                    workSpace->m_postingIDs.emplace_back(res->VID);
                }

                m_extraSearcher->SearchIndex(workSpace.get(), newResults, m_index, p_stats, truth, found);
            }
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            newResults.SortResult();
            std::copy(newResults.GetResults(), newResults.GetResults() + newResults.GetResultNum(), p_query.GetResults());
            return ErrorCode::Success;
        }
#pragma endregion

        template <typename T>
        ErrorCode Index<T>::GetPostingDebug(SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs)
        {
            VIDs.clear();
            if (!m_extraSearcher) return ErrorCode::EmptyIndex;
            if (!m_extraSearcher->CheckValidPosting(vid)) return ErrorCode::Fail;

            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace) {
                workSpace.reset(new ExtraWorkSpace());
                workSpace->Initialize(m_options.m_maxCheck, m_options.m_hashExp, m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            else {
                workSpace->Clear(m_options.m_searchInternalResultNum, max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx, m_options.m_enableDataCompression);
            }
            workSpace->m_deduper.clear();


            auto out = m_extraSearcher->GetPostingDebug(workSpace.get(), m_index, vid, VIDs, vecs);
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            return out;
        }

        template <typename T>
        void Index<T>::SelectHeadAdjustOptions(int p_vectorCount) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Adjust Parameters...\n");

            if (m_options.m_headVectorCount != 0) m_options.m_ratio = m_options.m_headVectorCount * 1.0 / p_vectorCount;
            int headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
            if (headCnt == 0)
            {
                for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
                {
                    m_options.m_ratio = minCnt / p_vectorCount;
                    headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
            }

            if (m_options.m_iBKTKmeansK > headCnt)
            {
                m_options.m_iBKTKmeansK = headCnt;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n", headCnt);
            }

            if (m_options.m_selectThreshold == 0)
            {
                m_options.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / m_options.m_ratio));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", m_options.m_selectThreshold);
            }

            if (m_options.m_splitThreshold == 0)
            {
                m_options.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(m_options.m_selectThreshold * 2));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", m_options.m_splitThreshold);
            }

            if (m_options.m_splitFactor == 0)
            {
                m_options.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / m_options.m_ratio) + 0.5));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", m_options.m_splitFactor);
            }
        }

        template <typename T>
        int Index<T>::SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID, 
            const Options& p_opts, std::vector<int>& p_selected)
        {
            typedef std::pair<int, int> CSPair;
            std::vector<CSPair> children;
            int childrenSize = 1;
            const auto& node = (*p_tree)[p_nodeID];
            if (node.childStart >= 0)
            {
                children.reserve(node.childEnd - node.childStart);
                for (int i = node.childStart; i < node.childEnd; ++i)
                {
                    int cs = SelectHeadDynamicallyInternal(p_tree, i, p_opts, p_selected);
                    if (cs > 0)
                    {
                        children.emplace_back(i, cs);
                        childrenSize += cs;
                    }
                }
            }

            if (childrenSize >= p_opts.m_selectThreshold)
            {
                if (node.centerid < (*p_tree)[0].centerid)
                {
                    p_selected.push_back(node.centerid);
                }

                if (childrenSize > p_opts.m_splitThreshold)
                {
                    std::sort(children.begin(), children.end(), [](const CSPair& a, const CSPair& b)
                        {
                            return a.second > b.second;
                        });

                    size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
                    //if (selectCnt > 1) selectCnt -= 1;
                    for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
                    {
                        p_selected.push_back((*p_tree)[children[i].first].centerid);
                    }
                }

                return 0;
            }

            return childrenSize;
        }

        template <typename T>
        void Index<T>::SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount, std::vector<int>& p_selected) {
            p_selected.clear();
            p_selected.reserve(p_vectorCount);

            if (static_cast<int>(std::round(m_options.m_ratio * p_vectorCount)) >= p_vectorCount)
            {
                for (int i = 0; i < p_vectorCount; ++i)
                {
                    p_selected.push_back(i);
                }

                return;
            }
            Options opts = m_options;

            int selectThreshold = m_options.m_selectThreshold;
            int splitThreshold = m_options.m_splitThreshold;

            double minDiff = 100;
            for (int select = 2; select <= m_options.m_selectThreshold; ++select)
            {
                opts.m_selectThreshold = select;
                opts.m_splitThreshold = m_options.m_splitThreshold;

                int l = m_options.m_splitFactor;
                int r = m_options.m_splitThreshold;

                while (l < r - 1)
                {
                    opts.m_splitThreshold = (l + r) / 2;
                    p_selected.clear();

                    SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                    std::sort(p_selected.begin(), p_selected.end());
                    p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

                    double diff = static_cast<double>(p_selected.size()) / p_vectorCount - m_options.m_ratio;

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                        opts.m_selectThreshold,
                        opts.m_splitThreshold,
                        diff * 100.0);

                    if (minDiff > fabs(diff))
                    {
                        minDiff = fabs(diff);

                        selectThreshold = opts.m_selectThreshold;
                        splitThreshold = opts.m_splitThreshold;
                    }

                    if (diff > 0)
                    {
                        l = (l + r) / 2;
                    }
                    else
                    {
                        r = (l + r) / 2;
                    }
                }
            }

            opts.m_selectThreshold = selectThreshold;
            opts.m_splitThreshold = splitThreshold;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Final Select Threshold: %d, Split Threshold: %d.\n",
                opts.m_selectThreshold,
                opts.m_splitThreshold);

            p_selected.clear();
            SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
            std::sort(p_selected.begin(), p_selected.end());
            p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
        }

        template <typename T>
        template <typename InternalDataType>
        bool Index<T>::SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader) {
            std::shared_ptr<VectorSet> vectorset = p_reader->GetVectorSet();
            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized())
                vectorset->Normalize(m_options.m_iSelectHeadNumberOfThreads);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin initial data (%d,%d)...\n", vectorset->Count(), vectorset->Dimension());

            COMMON::Dataset<InternalDataType> data(vectorset->Count(), vectorset->Dimension(), vectorset->Count(), vectorset->Count() + 1, (InternalDataType*)vectorset->GetData());
            
            auto t1 = std::chrono::high_resolution_clock::now();
            SelectHeadAdjustOptions(data.R());
            std::vector<int> selected;
            if (data.R() == 1) {
                selected.push_back(0); 
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "Random")) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
                selected.resize(data.R());
                for (int i = 0; i < data.R(); i++) selected[i] = i;
                std::shuffle(selected.begin(), selected.end(), rg);
                int headCnt = static_cast<int>(std::round(m_options.m_ratio * data.R()));
                selected.resize(headCnt);
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "BKT")) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
                std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
                bkt->m_iBKTKmeansK = m_options.m_iBKTKmeansK;
                bkt->m_iBKTLeafSize = m_options.m_iBKTLeafSize;
                bkt->m_iSamples = m_options.m_iSamples;
                bkt->m_iTreeNumber = m_options.m_iTreeNumber;
                bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
                    bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber, m_options.m_iSelectHeadNumberOfThreads);

                bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads, nullptr, nullptr, true);
                auto t2 = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

                if (m_options.m_saveBKT) {                
                    std::stringstream bktFileNameBuilder;
                    bktFileNameBuilder << m_options.m_vectorPath << ".bkt." << m_options.m_iBKTKmeansK << "_"
                        << m_options.m_iBKTLeafSize << "_" << m_options.m_iTreeNumber << "_" << m_options.m_iSamples << "_"
                        << static_cast<int>(m_options.m_distCalcMethod) << ".bin";
                    bkt->SaveTrees(bktFileNameBuilder.str());
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...Select Head Dynamically...\n");
                SelectHeadDynamically(bkt, data.R(), selected);

                if (selected.empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
                    return false;
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Seleted Nodes: %u, about %.2lf%% of total.\n",
                static_cast<unsigned int>(selected.size()),
                selected.size() * 100.0 / data.R());

            if (!m_options.m_noOutput)
            {
                std::sort(selected.begin(), selected.end());

                std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
                if (output == nullptr || outputIDs == nullptr ||
                    !output->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(), std::ios::binary | std::ios::out) ||
                    !outputIDs->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n", 
                        (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(), 
                        (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                    return false;
                }

                SizeType val = static_cast<SizeType>(selected.size());
                if (output->WriteBinary(sizeof(val), reinterpret_cast<char*>(&val)) != sizeof(val)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                    return false;
                }
                DimensionType dt = data.C();
                if (output->WriteBinary(sizeof(dt), reinterpret_cast<char*>(&dt)) != sizeof(dt)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                    return false;
                }

                for (int i = 0; i < selected.size(); i++)
                {
                    uint64_t vid = static_cast<uint64_t>(selected[i]);
                    if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        return false;
                    }

                    if (output->WriteBinary(sizeof(InternalDataType) * data.C(), (char*)(data[vid])) != sizeof(InternalDataType) * data.C()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        return false;
                    }
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        template <typename T>
        ErrorCode Index<T>::BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader>& p_reader) {
            if (!m_options.m_indexDirectory.empty()) {
                if (!direxists(m_options.m_indexDirectory.c_str()))
                {
                    mkdir(m_options.m_indexDirectory.c_str());
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Select Head...\n");
            auto t1 = std::chrono::high_resolution_clock::now();
            if (m_options.m_selectHead) {
                omp_set_num_threads(m_options.m_iSelectHeadNumberOfThreads);
                bool success = false;
                if (m_pQuantizer)
                {
                    success = SelectHeadInternal<std::uint8_t>(p_reader);
                }
                else
                {
                    success = SelectHeadInternal<T>(p_reader);
                }
                if (!success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
                    return ErrorCode::Fail;
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
            if (m_options.m_buildHead) {
                auto valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
                auto dims = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;

                m_index = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
                m_index->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
                m_index->SetQuantizer(m_pQuantizer);
                for (const auto& iter : m_headParameters)
                {
                    m_index->SetParameter(iter.first.c_str(), iter.second.c_str());
                }

                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file.\n");
                    return ErrorCode::Fail;
                }
                {
                    auto headvectorset = vectorReader->GetVectorSet();
                    if (m_index->BuildIndex(headvectorset, nullptr, false, true, true) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                        return ErrorCode::Fail;
                    }
                    m_index->SetQuantizerFileName(m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
                    if (m_index->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save head index.\n");
                        return ErrorCode::Fail;
                    }
                }
                m_index.reset();
                if (LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            double buildHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs\n", selectHeadTime, buildHeadTime);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build SSDIndex...\n");
            if (m_options.m_enableSSD) {
                omp_set_num_threads(m_options.m_iSSDNumberOfThreads);

                if (m_index == nullptr && LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
                    return ErrorCode::Fail;
                }
                m_index->SetQuantizer(m_pQuantizer);
                if (!CheckHeadIndexType()) return ErrorCode::Fail;

                m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
                m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
                m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
                m_index->UpdateIndex();

                if (m_pQuantizer)
                {
                    m_extraSearcher.reset(new ExtraFullGraphSearcher<std::uint8_t>());
                }
                else if (m_options.m_useKV)
                {
                    if (m_options.m_inPlace) {
                        m_extraSearcher.reset(new ExtraRocksDBController<T>(m_options.m_KVPath.c_str(), m_options.m_dim, INT_MAX, m_options.m_useDirectIO, m_options.m_latencyLimit));
                    }
                    else {
                        m_extraSearcher.reset(new ExtraRocksDBController<T>(m_options.m_KVPath.c_str(), m_options.m_dim, m_options.m_postingPageLimit * PageSize / (sizeof(T)*m_options.m_dim + sizeof(int) + sizeof(uint8_t) ), m_options.m_useDirectIO, m_options.m_latencyLimit));
                    }
                } else {
                    m_extraSearcher.reset(new ExtraFullGraphSearcher<T>());
                }

                if (m_options.m_buildSsdIndex) {
                    if (!m_options.m_excludehead) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Include all vectors into SSD index...\n");
                        std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
                        if (ptr == nullptr || !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::out)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s for overwrite\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                            return ErrorCode::Fail;
                        }
                        std::uint64_t vid = (std::uint64_t)MaxSize;
                        for (int i = 0; i < m_index->GetNumSamples(); i++) {
                            IOBINARY(ptr, WriteBinary, sizeof(std::uint64_t), (char*)(&vid));
                        }
                    }

                    if (!m_extraSearcher->BuildIndex(p_reader, m_index, m_options)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BuildSSDIndex Failed!\n");
                        return ErrorCode::Fail;
                    }
                }
                if (!m_extraSearcher->LoadIndex(m_options)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot Load SSDIndex!\n");
                    if (m_options.m_buildSsdIndex) {
                        return ErrorCode::Fail;
                    }
                    else {
                        m_extraSearcher.reset();
                    }
                }

                if (m_extraSearcher != nullptr) {
                    m_vectorTranslateMap.reset(new std::uint64_t[m_index->GetNumSamples()], std::default_delete<std::uint64_t[]>());
                    std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n", (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                        return ErrorCode::Fail;
                    }
                    IOBINARY(ptr, ReadBinary, sizeof(std::uint64_t) * m_index->GetNumSamples(), (char*)(m_vectorTranslateMap.get()));
                }

                LOG(Helper::LogLevel::LL_Info, "DataBlockSize: %d, Capacity: %d\n", m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
                m_versionMap.Load(m_options.m_fullDeletedIDFile, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
                m_postingSizes.Load(m_options.m_ssdInfoFile, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);

                m_vectorNum.store(m_versionMap.GetVectorNum());

                LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", m_vectorNum.load());

                LOG(Helper::LogLevel::LL_Info, "Current posting num: %d.\n", m_postingSizes.GetPostingNum());

                CalculatePostingDistribution();
                if (m_options.m_preReassign) {
                    PreReassign(p_reader);
                    m_index->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder);
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
                    m_postingSizes.Save(m_options.m_ssdInfoFile);
                }

                if (m_options.m_update) {
                    m_inMemoryThread += m_options.m_reassignThreadNum;
                    m_inMemoryThread += m_options.m_insertThreadNum;
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize persistent buffer\n");
                    std::shared_ptr<Helper::KeyValueIO> db;
                    db.reset(new SPANN::RocksDBIO());
                    m_persistentBuffer = std::make_shared<PersistentBuffer>(m_options.m_persistentBufferPath, db);
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_options.m_appendThreadNum, m_options.m_reassignThreadNum);
                    m_splitThreadPool = std::make_shared<ThreadPool>();
                    m_splitThreadPool->init(m_options.m_appendThreadNum);
                    m_reassignThreadPool = std::make_shared<ThreadPool>();
                    m_reassignThreadPool->init(m_options.m_reassignThreadNum);
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");

                    // LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize dispatcher\n");
                    // m_dispatcher = std::make_shared<Dispatcher>(m_persistentBuffer, m_options.m_batch, m_splitThreadPool, m_reassignThreadPool, this);

                    // m_dispatcher->run();
                    LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
                }
            }
            auto t4 = std::chrono::high_resolution_clock::now();
            double buildSSDTime = std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs build ssd time: %.2lfs\n", selectHeadTime, buildHeadTime, buildSSDTime);

            if (m_options.m_deleteHeadVectors) {
                if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) && 
                    remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
                }
            }

            m_bReady = true;
            return ErrorCode::Success;
        }
        template <typename T>
        ErrorCode Index<T>::BuildIndex(bool p_normalized) 
        {
            SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
            SizeType dim = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
            std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, dim, m_options.m_vectorType, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, p_normalized));
            auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
            if (m_options.m_vectorPath.empty())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
            }
            else {
                if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                    return ErrorCode::Fail;
                }
                m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
            }
            return BuildIndexInternal(vectorReader);
        }

        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized, bool p_shareOwnership)
        {
            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;

            std::shared_ptr<VectorSet> vectorSet;
            if (p_shareOwnership) {
                vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                    GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }
            else {
                ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
                memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
                vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
            }


            if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized) {
                vectorSet->Normalize(m_options.m_iSSDNumberOfThreads);
            }
            SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
            std::shared_ptr<Helper::VectorSetReader> vectorReader(new Helper::MemoryVectorReader(std::make_shared<Helper::ReaderOptions>(valueType, p_dimension, VectorFileType::DEFAULT, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, true),
                vectorSet));
            
            m_options.m_valueType = GetEnumValueType<T>();
            m_options.m_dim = p_dimension;
            m_options.m_vectorSize = p_vectorNum;
            return BuildIndexInternal(vectorReader);
        }

        template <typename T>
        ErrorCode
            Index<T>::UpdateIndex()
        {
            omp_set_num_threads(m_options.m_iSSDNumberOfThreads);
            m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
            //m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
            //m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
            m_index->UpdateIndex();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
            Index<T>::SetParameter(const char* p_param, const char* p_value, const char* p_section)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") && !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute")) {
                if (m_index != nullptr) return m_index->SetParameter(p_param, p_value);
                else m_headParameters[p_param] = p_value;
            }
            else {
                m_options.SetParameter(p_section, p_param, p_value);
            }
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod")) {
                if (m_pQuantizer)
                {
                    m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                    m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
                }
                else
                {
                    m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
                    m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
                }
            }
            return ErrorCode::Success;
        }


        template <typename T>
        std::string
            Index<T>::GetParameter(const char* p_param, const char* p_section) const
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") && !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute")) {
                if (m_index != nullptr) return m_index->GetParameter(p_param);
                else {
                    auto iter = m_headParameters.find(p_param);
                    if (iter != m_headParameters.end()) return iter->second;
                    return "Undefined!";
                }
            }
            else {
                return m_options.GetParameter(p_section, p_param);
            }
        }

        // Add insert entry to persistent buffer
        template <typename T>
        ErrorCode Index<T>::AddIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension,
                                     std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex,
                                     bool p_normalized)
        {
            if (m_options.m_indexAlgoType != IndexAlgoType::BKT || m_extraSearcher == nullptr) {
                LOG(Helper::LogLevel::LL_Error, "Only Support BKT Update");
                return ErrorCode::Fail;
            }

            std::vector<QueryResult> p_queryResults(p_vectorNum, QueryResult(nullptr, m_options.m_internalResultNum, false));

            for (int k = 0; k < p_vectorNum; k++)
            {
                p_queryResults[k].SetTarget(reinterpret_cast<const T*>(reinterpret_cast<const char*>(p_data) + k * p_dimension));
                p_queryResults[k].Reset();
                auto VID = m_vectorNum++;
                {
                    std::lock_guard<std::mutex> lock(m_dataAddLock);
                    auto ret = m_versionMap.AddBatch(1);
                    if (ret == ErrorCode::MemoryOverFlow) {
                        LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: VID: %d, Map Size:%d\n", VID, m_versionMap.BufferSize());
                        exit(1);
                    }
                    //m_reassignedID.AddBatch(1);
                }

                m_index->SearchIndex(p_queryResults[k]);

                int replicaCount = 0;
                BasicResult* queryResults = p_queryResults[k].GetResults();
                std::vector<EdgeInsert> selections(static_cast<size_t>(m_options.m_replicaCount));
                for (int i = 0; i < p_queryResults[k].GetResultNum() && replicaCount < m_options.m_replicaCount; ++i)
                {
                    if (queryResults[i].VID == -1) {
                        break;
                    }
                    // RNG Check.
                    bool rngAccpeted = true;
                    for (int j = 0; j < replicaCount; ++j)
                    {
                        float nnDist = m_index->ComputeDistance(m_index->GetSample(queryResults[i].VID),
                                                                m_index->GetSample(selections[j].headID));
                        if (nnDist <= queryResults[i].Dist)
                        {
                            rngAccpeted = false;
                            break;
                        }
                    }
                    if (!rngAccpeted)
                        continue;
                    selections[replicaCount].headID = queryResults[i].VID;
                    selections[replicaCount].fullID = VID;
                    selections[replicaCount].distance = queryResults[i].Dist;
                    selections[replicaCount].order = (char)replicaCount;
                    ++replicaCount;
                }

                char insertCode = 0;
                uint8_t version = 0;
                m_versionMap.UpdateVersion(VID, version);
                std::string appendPosting;
                appendPosting += Helper::Convert::Serialize<int>(&VID, 1);
                appendPosting += Helper::Convert::Serialize<uint8_t>(&version, 1);
                appendPosting += Helper::Convert::Serialize<T>(p_queryResults[k].GetTarget(), m_options.m_dim);
                // std::shared_ptr<std::string> appendPosting_ptr = std::make_shared<std::string>(appendPosting);
                for (int i = 0; i < replicaCount; i++)
                {
                    // AppendAsync(selections[i].headID, 1, appendPosting_ptr);
                    Append(selections[i].headID, 1, appendPosting);
                }

                // std::string assignment;
                // assignment += Helper::Convert::Serialize<char>(&insertCode, 1);
                // assignment += Helper::Convert::Serialize<char>(&replicaCount, 1);
                // for (int i = 0; i < replicaCount; i++)
                // {
                //     // LOG(Helper::LogLevel::LL_Info, "VID: %d, HeadID: %d, Write To PersistentBuffer\n", VID, selections[i].headID);
                //     assignment += Helper::Convert::Serialize<int>(&selections[i].headID, 1);
                //     assignment += Helper::Convert::Serialize<int>(&VID, 1);
                //     assignment += Helper::Convert::Serialize<uint8_t>(&version, 1);
                //     // assignment += Helper::Convert::Serialize<float>(&selections[i].distance, 1);
                //     assignment += Helper::Convert::Serialize<T>(p_queryResults[k].GetTarget(), m_options.m_dim);
                // }
                // m_assignmentQueue.push(m_persistentBuffer->PutAssignment(assignment));
            }
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::DeleteIndex(const SizeType &p_id)
        {
            LOG(Helper::LogLevel::LL_Info, "Delete not support\n");
            exit(0);
            char deleteCode = 1;
            int VID = p_id;
            std::string assignment;
            assignment += Helper::Convert::Serialize<char>(&deleteCode, 1);
            assignment += Helper::Convert::Serialize<int>(&VID, 1);
            m_assignmentQueue.push(m_persistentBuffer->PutAssignment(assignment));
            return ErrorCode::Success;
        }

        template <typename T>
        void SPTAG::SPANN::Index<T>::Dispatcher::dispatch()
        {
            // int32_t vectorInfoSize = m_index->GetValueSize() + sizeof(int) + sizeof(uint8_t) + sizeof(float);
            // int32_t vectorInfoSize = m_index->GetValueSize() + sizeof(int) + sizeof(uint8_t);
            // while (running) {

            //     std::map<SizeType, std::shared_ptr<std::string>> newPart;
            //     newPart.clear();
            //     int i;
            //     for (i = 0; i < batch; i++) {
            //         std::string assignment;
            //         int assignId = m_index->GetNextAssignment();

            //         if (assignId == -1) break;

            //         m_persistentBuffer->GetAssignment(assignId, &assignment);
            //         if(assignment.empty()) {
            //             LOG(Helper::LogLevel::LL_Info, "Error: Get Assignment\n");
            //             exit(0);
            //         }
            //         char code = *(reinterpret_cast<char*>(assignment.data()));
            //         if (code == 0) {
            //             // insert
            //             char* replicaCount = assignment.data() + sizeof(char);
            //             // LOG(Helper::LogLevel::LL_Info, "dispatch: replica count: %d\n", *replicaCount);

            //             for (char index = 0; index < *replicaCount; index++) {
            //                 char* headPointer = assignment.data() + sizeof(char) + sizeof(char) + index * (vectorInfoSize + sizeof(int));
            //                 int32_t headID = *(reinterpret_cast<int*>(headPointer));
            //                 // LOG(Helper::LogLevel::LL_Info, "dispatch: headID: %d\n", headID);
            //                 int32_t vid = *(reinterpret_cast<int*>(headPointer + sizeof(int)));
            //                 // LOG(Helper::LogLevel::LL_Info, "dispatch: vid: %d\n", vid);
            //                 uint8_t version = *(reinterpret_cast<uint8_t*>(headPointer + sizeof(int) + sizeof(int)));
            //                 // LOG(Helper::LogLevel::LL_Info, "dispatch: version: %d\n", version);

            //                 if (m_index->CheckIdDeleted(vid) || !m_index->CheckVersionValid(vid, version)) {
            //                     // LOG(Helper::LogLevel::LL_Info, "Unvalid Vector: %d, version: %d, current version: %d\n", vid, version);
            //                     continue;
            //                 }
            //                 // LOG(Helper::LogLevel::LL_Info, "Vector: %d, Plan to append to: %d\n", vid, headID);
            //                 if (newPart.find(headID) == newPart.end()) {
            //                     newPart[headID] = std::make_shared<std::string>(assignment.substr(sizeof(char) + sizeof(char) + index * (vectorInfoSize + sizeof(int)) + sizeof(int), vectorInfoSize));
            //                 } else {
            //                     newPart[headID]->append(assignment.substr(sizeof(char) + sizeof(char) + index * (vectorInfoSize + sizeof(int)) + sizeof(int), vectorInfoSize));
            //                 }
            //             }
            //         } else {
            //             // delete
            //             char* vectorPointer = assignment.data() + sizeof(char);
            //             int VID = *(reinterpret_cast<int*>(vectorPointer));
            //             //LOG(Helper::LogLevel::LL_Info, "Scanner: delete: %d\n", VID);
            //             m_index->DeleteIndex(VID);
            //         }
            //     }

            //     for (auto & iter : newPart) {
            //         int appendNum = (*iter.second).size() / (vectorInfoSize);
            //         if (appendNum == 0) LOG(Helper::LogLevel::LL_Info, "Error!, headID :%d, appendNum :%d, size :%d\n", iter.first, appendNum, iter.second);
            //         m_index->AppendAsync(iter.first, appendNum, iter.second);
            //     }

            //     if (i == 0) {
            //         std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //     } else {
            //         // LOG(Helper::LogLevel::LL_Info, "Dispatcher: Process Append Assignments: %d, after batched: %d\n", i, newPart.size());
            //         std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //     }
            // }
        }

        // template <typename ValueType>
        // ErrorCode SPTAG::SPANN::Index<ValueType>::Split(const SizeType headID, int appendNum, std::string& appendPosting)
        template <typename ValueType>
        ErrorCode SPTAG::SPANN::Index<ValueType>::Split(const SizeType headID)
        {
            auto splitBegin = std::chrono::high_resolution_clock::now();
            std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);
            // if (m_postingSizes.GetSize(headID) + appendNum < m_extraSearcher->GetPostingSizeLimit()) {
            //     return ErrorCode::FailSplit;
            // }
            if (m_postingSizes.GetSize(headID) < m_extraSearcher->GetPostingSizeLimit()) {
                   return ErrorCode::FailSplit;    
            }
            std::string postingList;
            if (m_extraSearcher->SearchIndex(headID, postingList) != ErrorCode::Success) {
                LOG(Helper::LogLevel::LL_Info, "Split fail to get oversized postings\n");
                exit(0);
            }
            // postingList += appendPosting;
            // reinterpret postingList to vectors and IDs
            auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
            size_t vectorInfoSize = m_options.m_dim * sizeof(ValueType) + m_metaDataSize;
            size_t postVectorNum = postingList.size() / vectorInfoSize;
            COMMON::Dataset<ValueType> smallSample;  // smallSample[i] -> VID
            std::shared_ptr<uint8_t> vectorBuffer(new uint8_t[m_options.m_dim * sizeof(ValueType) * postVectorNum], std::default_delete<uint8_t[]>());
            std::vector<int> localIndicesInsert(postVectorNum);  // smallSample[i] = j <-> localindices[j] = i
            std::vector<uint8_t> localIndicesInsertVersion(postVectorNum);
            // std::vector<float> localIndicesInsertFloat(postVectorNum);
            std::vector<int> localIndices(postVectorNum);
            auto vectorBuf = vectorBuffer.get();
            size_t realVectorNum = postVectorNum;
            int index = 0;
            for (int j = 0; j < postVectorNum; j++)
            {
                uint8_t* vectorId = postingP + j * vectorInfoSize;
                //LOG(Helper::LogLevel::LL_Info, "vector index/total:id: %d/%d:%d\n", j, m_postingSizes[headID].load(), *(reinterpret_cast<int*>(vectorId)));
                uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                if (CheckIdDeleted(*(reinterpret_cast<int*>(vectorId))) || !CheckVersionValid(*(reinterpret_cast<int*>(vectorId)), version)) {
                    realVectorNum--;
                } else {
                    localIndicesInsert[index] = *(reinterpret_cast<int*>(vectorId));
                    localIndicesInsertVersion[index] = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    // localIndicesInsertFloat[index] = *(reinterpret_cast<float*>(vectorId + sizeof(int) + sizeof(uint8_t)));
                    localIndices[index] = index;
                    index++;
                    memcpy(vectorBuf, vectorId + m_metaDataSize, m_options.m_dim * sizeof(ValueType));
                    vectorBuf += m_options.m_dim * sizeof(ValueType);
                }
            }
            // double gcEndTime = sw.getElapsedMs();
            // m_splitGcCost += gcEndTime;
            if (realVectorNum < m_extraSearcher->GetPostingSizeLimit())
            {
                postingList.clear();
                for (int j = 0; j < realVectorNum; j++)
                {
                    postingList += Helper::Convert::Serialize<int>(&localIndicesInsert[j], 1);
                    postingList += Helper::Convert::Serialize<uint8_t>(&localIndicesInsertVersion[j], 1);
                    // postingList += Helper::Convert::Serialize<float>(&localIndicesInsertFloat[j], 1);
                    postingList += Helper::Convert::Serialize<ValueType>(vectorBuffer.get() + j * m_options.m_dim * sizeof(ValueType), m_options.m_dim);
                }
                m_postingSizes.UpdateSize(headID, realVectorNum);
                if (m_extraSearcher->OverrideIndex(headID, postingList) != ErrorCode::Success ) {
                    LOG(Helper::LogLevel::LL_Info, "Split Fail to write back postings\n");
                    exit(0);
                }
                m_garbageNum++;
                auto GCEnd = std::chrono::high_resolution_clock::now();
                double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(GCEnd - splitBegin).count();
                m_garbageCost += elapsedMSeconds;
                return ErrorCode::Success;
            }
            //LOG(Helper::LogLevel::LL_Info, "Resize\n");
            localIndicesInsert.resize(realVectorNum);
            localIndices.resize(realVectorNum);
            smallSample.Initialize(realVectorNum, m_options.m_dim, m_index->m_iDataBlockSize, m_index->m_iDataCapacity, reinterpret_cast<ValueType*>(vectorBuffer.get()), false);

            auto clusterBegin = std::chrono::high_resolution_clock::now();
            // k = 2, maybe we can change the split number, now it is fixed
            SPTAG::COMMON::KmeansArgs<ValueType> args(2, smallSample.C(), (SizeType)localIndicesInsert.size(), 1, m_index->GetDistCalcMethod());
            std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));

            int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr, m_options.m_virtualHead);

            auto clusterEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(clusterEnd - clusterBegin).count();
            m_clusteringCost += elapsedMSeconds;
            // int numClusters = ClusteringSPFresh(smallSample, localIndices, 0, localIndices.size(), args, 10, false, m_options.m_virtualHead);
            // exit(0);
            if (numClusters <= 1)
            {
                LOG(Helper::LogLevel::LL_Info, "Cluserting Failed (The same vector), Cut to limit\n");
                postingList.clear();
                for (int j = 0; j < m_extraSearcher->GetPostingSizeLimit(); j++)
                {
                    postingList += Helper::Convert::Serialize<int>(&localIndicesInsert[j], 1);
                    postingList += Helper::Convert::Serialize<uint8_t>(&localIndicesInsertVersion[j], 1);
                    // postingList += Helper::Convert::Serialize<float>(&localIndicesInsertFloat[j], 1);
                    postingList += Helper::Convert::Serialize<ValueType>(vectorBuffer.get() + j * m_options.m_dim * sizeof(ValueType), m_options.m_dim);
                }
                m_postingSizes.UpdateSize(headID, m_extraSearcher->GetPostingSizeLimit());
                if (m_extraSearcher->OverrideIndex(headID, postingList) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "Split fail to override postings cut to limit\n");
                    exit(0);
                }
                return ErrorCode::Success;
            }

            long long newHeadVID = -1;
            int first = 0;
            std::vector<SizeType> newHeadsID;
            std::vector<std::string> newPostingLists;
            bool theSameHead = false;
            for (int k = 0; k < 2; k++) {
                std::string postingList;
                if (args.counts[k] == 0)	continue;
                if (!theSameHead && m_index->ComputeDistance(args.centers + k * args._D, m_index->GetSample(headID)) < Epsilon) {
                    newHeadsID.push_back(headID);
                    newHeadVID = headID;
                    theSameHead = true;
                    for (int j = 0; j < args.counts[k]; j++)
                    {

                        postingList += Helper::Convert::Serialize<SizeType>(&localIndicesInsert[localIndices[first + j]], 1);
                        postingList += Helper::Convert::Serialize<uint8_t>(&localIndicesInsertVersion[localIndices[first + j]], 1);
                        // postingList += Helper::Convert::Serialize<float>(&localIndicesInsertFloat[localIndices[first + j]], 1);
                        postingList += Helper::Convert::Serialize<ValueType>(smallSample[localIndices[first + j]], m_options.m_dim);
                    }
                    if (m_extraSearcher->OverrideIndex(newHeadVID, postingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Fail to override postings\n");
                        exit(0);
                    }
                    m_theSameHeadNum++;
                }
                else {
                    int begin, end = 0;
                    m_index->AddIndexId(args.centers + k * args._D, 1, m_options.m_dim, begin, end);
                    newHeadVID = begin;
                    newHeadsID.push_back(begin);
                    for (int j = 0; j < args.counts[k]; j++)
                    {
                        // float dist = m_index->ComputeDistance(smallSample[args.clusterIdx[k]], smallSample[localIndices[first + j]]);
                        postingList += Helper::Convert::Serialize<SizeType>(&localIndicesInsert[localIndices[first + j]], 1);
                        postingList += Helper::Convert::Serialize<uint8_t>(&localIndicesInsertVersion[localIndices[first + j]], 1);
                        // postingList += Helper::Convert::Serialize<float>(&dist, 1);
                        postingList += Helper::Convert::Serialize<ValueType>(smallSample[localIndices[first + j]], m_options.m_dim);
                    }
                    if (m_extraSearcher->AddIndex(newHeadVID, postingList) != ErrorCode::Success) {
                        LOG(Helper::LogLevel::LL_Info, "Fail to add new postings\n");
                        exit(0);
                    }

                    auto updateHeadBegin = std::chrono::high_resolution_clock::now();
                    m_index->AddIndexIdx(begin, end);
                    auto updateHeadEnd = std::chrono::high_resolution_clock::now();
                    elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(updateHeadEnd - updateHeadBegin).count();
                    m_updateHeadCost += elapsedMSeconds;
                }
                newPostingLists.push_back(postingList);
                // LOG(Helper::LogLevel::LL_Info, "Head id: %d split into : %d, length: %d\n", headID, newHeadVID, args.counts[k]);
                first += args.counts[k];
                {
                    std::lock_guard<std::mutex> lock(m_dataAddLock);
                    auto ret = m_postingSizes.AddBatch(1);
                    if (ret == ErrorCode::MemoryOverFlow) {
                        LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: NnewHeadVID: %d, Map Size:%d\n", newHeadVID, m_postingSizes.BufferSize());
                        exit(1);
                    }
                }
                m_postingSizes.UpdateSize(newHeadVID, args.counts[k]);
            }
            if (!theSameHead) {
                m_index->DeleteIndex(headID);
                m_postingSizes.UpdateSize(headID, 0);
            }
            lock.unlock();
            int split_order = ++m_splitNum;
            // if (theSameHead) LOG(Helper::LogLevel::LL_Info, "The Same Head\n");
            // LOG(Helper::LogLevel::LL_Info, "head1:%d, head2:%d\n", newHeadsID[0], newHeadsID[1]);

            // QuantifySplit(headID, newPostingLists, newHeadsID, headID, split_order);
            // QuantifyAssumptionBrokenTotally();
            auto reassignScanBegin = std::chrono::high_resolution_clock::now();
            
            if (!m_options.m_disableReassign) ReAssign(headID, newPostingLists, newHeadsID);

            auto reassignScanEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanEnd - reassignScanBegin).count();

            m_reassignScanCost += elapsedMSeconds;
            
            // while (!ReassignFinished())
            // {
            //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // }
            
            
            // LOG(Helper::LogLevel::LL_Info, "After ReAssign\n");

            // QuantifySplit(headID, newPostingLists, newHeadsID, headID, split_order);
            auto splitEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - splitBegin).count();
            m_splitCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        template <typename ValueType>
        ErrorCode SPTAG::SPANN::Index<ValueType>::ReAssign(SizeType headID, std::vector<std::string>& postingLists, std::vector<SizeType>& newHeadsID) {
//            TimeUtils::StopW sw;
            auto headVector = reinterpret_cast<const ValueType*>(m_index->GetSample(headID));
            std::vector<SizeType> HeadPrevTopK;
            std::vector<float> HeadPrevToSplitHeadDist;
            if (m_options.m_reassignK > 0) {
                COMMON::QueryResultSet<ValueType> nearbyHeads(NULL, m_options.m_reassignK);
                nearbyHeads.SetTarget(headVector);
                nearbyHeads.Reset();
                m_index->SearchIndex(nearbyHeads);
                BasicResult* queryResults = nearbyHeads.GetResults();
                for (int i = 0; i < nearbyHeads.GetResultNum(); i++) {
                    std::string tempPostingList;
                    auto vid = queryResults[i].VID;
                    if (vid == -1) {
                        break;
                    }
                    if (find(newHeadsID.begin(), newHeadsID.end(), vid) == newHeadsID.end()) {
                        // m_extraSearcher->SearchIndex(vid, tempPostingList);
                        // postingLists.push_back(tempPostingList);
                        HeadPrevTopK.push_back(vid);
                        HeadPrevToSplitHeadDist.push_back(queryResults[i].Dist);
                    }
                }
                auto reassignScanIOBegin = std::chrono::high_resolution_clock::now();
                std::vector<std::string> tempPostingLists;
                if (m_extraSearcher->SearchIndexMulti(HeadPrevTopK, &tempPostingLists) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Info, "ReAssign can't get all the near postings\n");
                    exit(0);
                }
                auto reassignScanIOEnd = std::chrono::high_resolution_clock::now();
                auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanIOEnd - reassignScanIOBegin).count();

                m_reassignScanIOCost += elapsedMSeconds;
                for (int i = 0; i < HeadPrevTopK.size(); i++) {
                    postingLists.push_back(tempPostingLists[i]);
                }
            }

            int vectorInfoSize = m_options.m_dim * sizeof(ValueType) + m_metaDataSize;
            std::map<SizeType, ValueType*> reAssignVectorsTop0;
            std::map<SizeType, SizeType> reAssignVectorsHeadPrevTop0;
            std::map<SizeType, uint8_t> versionsTop0;
            std::map<SizeType, ValueType*> reAssignVectorsTopK;
            std::map<SizeType, SizeType> reAssignVectorsHeadPrevTopK;
            std::map<SizeType, uint8_t> versionsTopK;

            std::vector<float_t> newHeadDist;

            newHeadDist.push_back(m_index->ComputeDistance(m_index->GetSample(headID), m_index->GetSample(newHeadsID[0])));
            newHeadDist.push_back(m_index->ComputeDistance(m_index->GetSample(headID), m_index->GetSample(newHeadsID[1])));

            for (int i = 0; i < postingLists.size(); i++) {
                auto& postingList = postingLists[i];
                size_t postVectorNum = postingList.size() / vectorInfoSize;
                auto* postingP = reinterpret_cast<uint8_t*>(&postingList.front());
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    // float dist = *(reinterpret_cast<float*>(vectorId + sizeof(int) + sizeof(uint8_t)));
                    float dist;
                    if (i <= 1) {
                        if (!CheckIdDeleted(vid) && CheckVersionValid(vid, version)) {
                            m_reAssignScanNum++;
                            dist = m_index->ComputeDistance(m_index->GetSample(newHeadsID[i]), reinterpret_cast<ValueType*>(vectorId + m_metaDataSize));
                            if (CheckIsNeedReassign(newHeadsID, reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), headID, newHeadDist[i], dist, true, newHeadsID[i])) {
                                reAssignVectorsTop0[vid] = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                reAssignVectorsHeadPrevTop0[vid] = newHeadsID[i];
                                versionsTop0[vid] = version;
                            }
                        }
                    } else {
                        if ((reAssignVectorsTop0.find(vid) == reAssignVectorsTop0.end()))
                        {
                            if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !CheckIdDeleted(vid) && CheckVersionValid(vid, version)) {
                                m_reAssignScanNum++;
                                dist = m_index->ComputeDistance(m_index->GetSample(HeadPrevTopK[i-2]), reinterpret_cast<ValueType*>(vectorId + m_metaDataSize));
                                if (CheckIsNeedReassign(newHeadsID, reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), headID, HeadPrevToSplitHeadDist[i-2], dist, false, HeadPrevTopK[i-2])) {
                                    reAssignVectorsTopK[vid] = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                    reAssignVectorsHeadPrevTopK[vid] = HeadPrevTopK[i-2];
                                    versionsTopK[vid] = version;
                                }
                            }
                        }
                    }
                }
            }
            // LOG(Helper::LogLevel::LL_Info, "Scan: %d\n", m_reAssignScanNum.load());
            // exit(0);
            

            ReAssignVectors(reAssignVectorsTop0, reAssignVectorsHeadPrevTop0, versionsTop0);
            ReAssignVectors(reAssignVectorsTopK, reAssignVectorsHeadPrevTopK, versionsTopK);
            return ErrorCode::Success;
        }

        template <typename ValueType>
        void SPTAG::SPANN::Index<ValueType>::ReAssignVectors(std::map<SizeType, ValueType*>& reAssignVectors,
                             std::map<SizeType, SizeType>& HeadPrevs, std::map<SizeType, uint8_t>& versions)
        {
            for (auto it = reAssignVectors.begin(); it != reAssignVectors.end(); ++it) {

                auto vectorContain = std::make_shared<std::string>(Helper::Convert::Serialize<uint8_t>(it->second, m_options.m_dim));

                ReassignAsync(vectorContain, it->first, HeadPrevs[it->first], versions[it->first]);
            }
        }

        template <typename ValueType>
        bool SPTAG::SPANN::Index<ValueType>::ReAssignUpdate
                (const std::shared_ptr<std::string>& vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version)
        {
            m_reAssignNum++;

            bool isNeedReassign = true;
            auto selectBegin = std::chrono::high_resolution_clock::now();
            COMMON::QueryResultSet<ValueType> p_queryResults(NULL, m_options.m_internalResultNum);
            p_queryResults.SetTarget(reinterpret_cast<ValueType*>(&vectorContain->front()));
            p_queryResults.Reset();
            m_index->SearchIndex(p_queryResults);

            int replicaCount = 0;
            BasicResult* queryResults = p_queryResults.GetResults();
            std::vector<EdgeInsert> selections(static_cast<size_t>(m_options.m_replicaCount));

            int i;
            for (i = 0; i < p_queryResults.GetResultNum() && replicaCount < m_options.m_replicaCount; ++i) {
                if (queryResults[i].VID == -1) {
                    break;
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j) {
                    float nnDist = m_index->ComputeDistance(
                            m_index->GetSample(queryResults[i].VID),
                            m_index->GetSample(selections[j].headID));
                    if (m_options.m_rngFactor * nnDist <= queryResults[i].Dist) {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted)
                    continue;

                selections[replicaCount].headID = queryResults[i].VID;

                selections[replicaCount].fullID = VID;
                selections[replicaCount].distance = queryResults[i].Dist;
                selections[replicaCount].order = (char)replicaCount;
                if (selections[replicaCount].headID == HeadPrev) {
                    isNeedReassign = false;
                    break;
                }
                ++replicaCount;
            }
            auto selectEnd = std::chrono::high_resolution_clock::now();
            auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(selectEnd - selectBegin).count();
            m_selectCost += elapsedMSeconds;

            if (isNeedReassign && CheckVersionValid(VID, version)) {
                // LOG(Helper::LogLevel::LL_Info, "Update Version: VID: %d, version: %d, current version: %d\n", VID, version, m_versionMap.GetVersion(VID));
                m_versionMap.IncVersion(VID, &version);
            } else {
                isNeedReassign = false;
            }

            //LOG(Helper::LogLevel::LL_Info, "Reassign: oldVID:%d, replicaCount:%d, candidateNum:%d, dist0:%f\n", oldVID, replicaCount, i, selections[0].distance);
            auto reassignAppendBegin = std::chrono::high_resolution_clock::now();
            for (i = 0; isNeedReassign && i < replicaCount && CheckVersionValid(VID, version); i++) {
                std::string newPart;
                newPart += Helper::Convert::Serialize<int>(&VID, 1);
                newPart += Helper::Convert::Serialize<uint8_t>(&version, 1);
                // newPart += Helper::Convert::Serialize<float>(&selections[i].distance, 1);
                newPart += Helper::Convert::Serialize<ValueType>(p_queryResults.GetTarget(), m_options.m_dim);
                auto headID = selections[i].headID;
                //LOG(Helper::LogLevel::LL_Info, "Reassign: headID :%d, oldVID:%d, newVID:%d, posting length: %d, dist: %f, string size: %d\n", headID, oldVID, VID, m_postingSizes[headID].load(), selections[i].distance, newPart.size());
                if (ErrorCode::Undefined == Append(headID, -1, newPart)) {
                    // LOG(Helper::LogLevel::LL_Info, "Head Miss: VID: %d, current version: %d, another re-assign\n", VID, version);
                    isNeedReassign = false;
                }
            }
            auto reassignAppendEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignAppendEnd - reassignAppendBegin).count();
            m_reAssignAppendCost += elapsedMSeconds;

            return isNeedReassign;
        }

        template <typename ValueType>
        ErrorCode SPTAG::SPANN::Index<ValueType>::Append(SizeType headID, int appendNum, std::string& appendPosting)
        {
            auto appendBegin = std::chrono::high_resolution_clock::now();
            int reassignThreshold = 0;
            if (appendPosting.empty()) {
                LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }
            int vectorInfoSize = m_options.m_dim * sizeof(ValueType) + m_metaDataSize;

            if (appendNum == 0) {
                LOG(Helper::LogLevel::LL_Info, "Error!, headID :%d, appendNum:%d\n", headID, appendNum);
            } else if (appendNum == -1) {
                // for reassign
                reassignThreshold = 3;
                appendNum = 1;
            }

        checkDeleted:
            if (!m_index->ContainSample(headID)) {
                for (int i = 0; i < appendNum; i++)
                {
                    uint32_t idx = i * vectorInfoSize;
                    uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(int)]);
                    auto vectorContain = std::make_shared<std::string>(appendPosting.substr(idx + m_metaDataSize, m_options.m_dim * sizeof(ValueType)));
                    if (CheckVersionValid(*(int*)(&appendPosting[idx]), version)) {
                        // LOG(Helper::LogLevel::LL_Info, "Head Miss To ReAssign: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);
                        m_headMiss++;
                        ReassignAsync(vectorContain, *(int*)(&appendPosting[idx]), headID, version);
                    }
                    // LOG(Helper::LogLevel::LL_Info, "Head Miss Do Not To ReAssign: VID: %d, version: %d, current version: %d\n", *(int*)(&appendPosting[idx]), m_versionMap.GetVersion(*(int*)(&appendPosting[idx])), version);
                }
                return ErrorCode::Undefined;
            }
            // if (m_postingSizes.GetSize(headID) + appendNum > (m_extraSearcher->GetPostingSizeLimit() + reassignThreshold)) {
            //     if (Split(headID, appendNum, appendPosting) == ErrorCode::FailSplit) {
            //         goto checkDeleted;
            //     }
            //     auto splitEnd = std::chrono::high_resolution_clock::now();
            //     double elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - appendBegin).count();
            //     m_splitCost += elapsedMSeconds;
            //     return ErrorCode::Success;
            // } else {
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);
                if (!m_index->ContainSample(headID)) {
                    goto checkDeleted;
                }
                // for (int i = 0; i < appendNum; i++)
                // {
                //     uint32_t idx = i * vectorInfoSize;
                //     uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(int)]);
                //     LOG(Helper::LogLevel::LL_Info, "Append: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);

                // }
                // LOG(Helper::LogLevel::LL_Info, "Merge: headID: %d, appendNum:%d\n", headID, appendNum);
                if (!reassignThreshold) m_appendTaskNum++;
                auto appendIOBegin = std::chrono::high_resolution_clock::now();
                if (m_extraSearcher->AppendPosting(headID, appendPosting) != ErrorCode::Success) {
                    LOG(Helper::LogLevel::LL_Error, "Merge failed!\n");
                    exit(1);
                }
                auto appendIOEnd = std::chrono::high_resolution_clock::now();
                double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendIOEnd - appendIOBegin).count();
                if (!reassignThreshold) m_appendIOCost += elapsedMSeconds;
                m_postingSizes.IncSize(headID, appendNum);
            }
            if (m_postingSizes.GetSize(headID) + appendNum > (m_extraSearcher->GetPostingSizeLimit() + reassignThreshold)) {
                SplitAsync(headID);
            }
            // }
            auto appendEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
            if (!reassignThreshold) m_appendCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        template <typename T>
        void SPTAG::SPANN::Index<T>::ProcessAsyncReassign(std::shared_ptr<std::string> vectorContain, SizeType VID, SizeType HeadPrev, uint8_t version, std::function<void()> p_callback)
        {
            // return;
            if (m_versionMap.Contains(VID) || !CheckVersionValid(VID, version)) {
                // LOG(Helper::LogLevel::LL_Info, "ReassignID: %d, version: %d, current version: %d\n", VID, version, m_versionMap.GetVersion(VID));
                return;
            }

            
            // tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor VIDAccessor;
            // if (m_reassignMap.find(VIDAccessor, VID) && VIDAccessor->second < version) {
            //     return;
            // }
            // tbb::concurrent_hash_map<SizeType, SizeType>::value_type workPair(VID, version);
            // m_reassignMap.insert(workPair);
            auto reassignBegin = std::chrono::high_resolution_clock::now();

            ReAssignUpdate(vectorContain, VID, HeadPrev, version);

            auto reassignEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignEnd - reassignBegin).count();
            m_reAssignCost += elapsedMSeconds;
            //     m_reassignMap.erase(VID);

            if (p_callback != nullptr) {
                p_callback();
            }
        }
    }
}

#define DefineVectorValueType(Name, Type) \
template class SPTAG::SPANN::Index<Type>; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType


