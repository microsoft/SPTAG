// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"
#include <chrono>
#include <random>
#include <shared_mutex>
#include <filesystem>

#include "inc/Core/ResultIterator.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"

#ifdef SPDK
#include "ExtraSPDKController.h"
#endif

#ifdef ROCKSDB
#include "ExtraRocksDBController.h"
// enable rocksdb io_uring
extern "C" bool RocksDbIOUringEnable() { return true; }
#endif

#ifdef TIKV
#include "inc/Core/SPANN/ExtraTiKVController.h"
#endif

namespace SPTAG
{
template <typename T> thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;
namespace SPANN
{
EdgeCompare Selection::g_edgeComparer;

std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskIO> {
    return std::shared_ptr<Helper::DiskIO>(new Helper::AsyncFileIO());
};

template <typename T> bool Index<T>::CheckHeadIndexType()
{
    SPTAG::VectorValueType v1 = m_topIndex->GetVectorValueType(), v2 = GetEnumValueType<T>();
    if (v1 != v2)
    {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
            SPTAG::Helper::Convert::ConvertToString(v1).c_str(), SPTAG::Helper::Convert::ConvertToString(v2).c_str());
        if (!m_pQuantizer)
            return false;
    }
    return true;
}

template <typename T> void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
{
    m_pQuantizer = quantizer;
    if (m_pQuantizer)
    {
        m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
        m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                            ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase()
                            : 1;
    }
    else
    {
        m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
        m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                            ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>()
                            : 1;
    }
    if (m_topIndex)
    {
        m_topIndex->SetQuantizer(quantizer);
    }
}

template <typename T> ErrorCode Index<T>::LoadConfig(Helper::IniReader &p_reader)
{
    IndexAlgoType algoType = p_reader.GetParameter("Base", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = p_reader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
    if ((m_topIndex = CreateInstance(algoType, valueType)) == nullptr)
        return ErrorCode::FailedParseValue;

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
    for (int i = 0; i < 4; i++)
    {
        auto parameters = p_reader.GetParameters(sections[i].c_str());
        for (auto iter = parameters.begin(); iter != parameters.end(); iter++)
        {
            SetParameter(iter->first.c_str(), iter->second.c_str(), sections[i].c_str());
        }
    }

    if (m_pQuantizer)
    {
        m_pQuantizer->SetEnableADC(m_options.m_enableADC);
    }

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray> &p_indexBlobs)
{
    /** Need to modify **/
    m_topIndex->SetQuantizer(m_pQuantizer);
    if (!m_options.m_persistentBufferPath.empty() && !direxists(m_options.m_persistentBufferPath.c_str()))
        mkdir(m_options.m_persistentBufferPath.c_str());

    if (m_topIndex->LoadIndexDataFromMemory(p_indexBlobs) != ErrorCode::Success)
        return ErrorCode::Fail;

    m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    // m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    // m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_topIndex->UpdateIndex();
    m_topIndex->SetReady(true);

    m_topLocalToGlobalID.Initialize(m_topIndex->GetNumSamples(), 1, m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity,
                                    p_indexBlobs.back().Data(), false);
    m_topGlobalToLocalID.clear();
    for (int i = 0; i < m_topIndex->GetNumSamples(); i++)
    {
        SizeType globalID = *(m_topLocalToGlobalID[i]);
        m_topGlobalToLocalID[globalID] = i;
    }

    if (m_freeWorkSpaceIds == nullptr) {
        m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
        int maxIOThreads = (m_options.m_storage == Storage::STATIC) ? max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) : max(m_options.m_ioThreads, (2 * max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) +
                                    (m_options.m_layers + 1) * (m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum)));
        for (int i = 0; i < maxIOThreads; i++) {
            m_freeWorkSpaceIds->push(i);
        }
        m_workspaceCount = maxIOThreads;
    }

    if (m_options.m_shareDB) PrepareDB(m_db);
    m_extraSearchers.resize(m_options.m_layers);
    for (int i = m_options.m_layers - 1; i >= 0; i--) {
        if (m_options.m_storage == Storage::STATIC)
        {
            m_extraSearchers[i] = std::make_shared<ExtraStaticSearcher<T>>(i, this);
        }
        else
        {
            m_extraSearchers[i] = std::make_shared<ExtraDynamicSearcher<T>>(m_options, i, this, m_db);
        }

        if (!m_extraSearchers[i]->LoadIndex(m_options))
            return ErrorCode::Fail;
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    m_topIndex->SetQuantizer(m_pQuantizer);
    if (!m_options.m_persistentBufferPath.empty() && !direxists(m_options.m_persistentBufferPath.c_str()))
        mkdir(m_options.m_persistentBufferPath.c_str());

    auto headfiles = m_topIndex->GetIndexFiles();
    if (m_options.m_recovery)
    {
        std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
        auto headfiles = m_topIndex->GetIndexFiles();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading another in-memory index\n");
        std::string filename = m_options.m_persistentBufferPath + FolderSep + m_options.m_headIndexFolder;
        for (auto file : *headfiles)
        {
            files->push_back(filename + FolderSep + file);
        }
        std::vector<std::shared_ptr<Helper::DiskIO>> handles;
        for (std::string &f : *files)
        {
            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize(f.c_str(), std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file %s!\n", f.c_str());
                ptr = nullptr;
            }
            handles.push_back(std::move(ptr));
        }
        if (m_topIndex->LoadIndexData(handles) != ErrorCode::Success)
            return ErrorCode::Fail;
    }
    else if (m_topIndex->LoadIndexData(p_indexStreams) != ErrorCode::Success)
        return ErrorCode::Fail;

    m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_topIndex->UpdateIndex();
    m_topIndex->SetReady(true);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading headID map\n");
    m_topLocalToGlobalID.Load(p_indexStreams[m_topIndex->GetIndexFiles()->size()], m_topIndex->m_iDataBlockSize,
                              m_topIndex->m_iDataCapacity);
    m_topGlobalToLocalID.clear();
    for (int i = 0; i < m_topIndex->GetNumSamples(); i++)
    {
        SizeType globalID = *(m_topLocalToGlobalID[i]);
        m_topGlobalToLocalID[globalID] = i;
    }

    if (m_freeWorkSpaceIds == nullptr) {
        m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
        int maxIOThreads = (m_options.m_storage == Storage::STATIC) ? max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) : max(m_options.m_ioThreads, (2 * max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) +
                                    (m_options.m_layers + 1) * (m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum)));
        for (int i = 0; i < maxIOThreads; i++) {
            m_freeWorkSpaceIds->push(i);
        }
        m_workspaceCount = maxIOThreads;
    }

    if (m_options.m_shareDB) PrepareDB(m_db);
    m_extraSearchers.resize(m_options.m_layers);
    for (int i = m_options.m_layers - 1; i >= 0; i--) {
        if (m_options.m_storage == Storage::STATIC)
        {
            m_extraSearchers[i] = std::make_shared<ExtraStaticSearcher<T>>(i, this);
        }
        else
        {
            m_extraSearchers[i] = std::make_shared<ExtraDynamicSearcher<T>>(m_options, i, this, m_db);
        }

        if (m_options.m_storage != Storage::STATIC && !m_extraSearchers[i]->Available())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extrasearcher is not available and failed to initialize.\n");
            return ErrorCode::Fail;
        }

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading storage\n");
        if (!(m_extraSearchers[i]->LoadIndex(m_options)))
            return ErrorCode::Fail;
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
{
    IOSTRING(p_configOut, WriteString, "[Base]\n");
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr)                                             \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

    IOSTRING(p_configOut, WriteString, "[SelectHead]\n");
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr)                                        \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

    IOSTRING(p_configOut, WriteString, "[BuildHead]\n");
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr)                                         \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

    m_topIndex->SaveConfig(p_configOut);

    Helper::Convert::ConvertStringTo<int>(m_topIndex->GetParameter("HashTableExponent").c_str(), m_options.m_hashExp);
    IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

    IOSTRING(p_configOut, WriteString, "\n");
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    if (m_topIndex == nullptr) return ErrorCode::EmptyIndex;

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SaveIndexData: waiting for all background jobs to finish...\n");
    auto waitStart = std::chrono::steady_clock::now();
    int pollCount = 0;
    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (++pollCount % 250 == 0) { // every ~5 seconds
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - waitStart).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SaveIndexData: still waiting for background jobs (%lld s elapsed)\n", (long long)elapsed);
        }
    }
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - waitStart).count();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SaveIndexData: all background jobs finished (waited %lld s)\n", (long long)elapsed);
    }

    ErrorCode ret;
    if ((ret = m_topIndex->SaveIndexData(p_indexStreams)) != ErrorCode::Success)
        return ret;

    if ((ret = m_topLocalToGlobalID.Save(p_indexStreams[m_topIndex->GetIndexFiles()->size()])) != ErrorCode::Success)
        return ret;

    for (auto& searcher : m_extraSearchers) {
        if ((ret = searcher->Checkpoint(m_options.m_indexDirectory)) != ErrorCode::Success)
            return ret;
    }
    return ErrorCode::Success;
}

#pragma region K - NN search

template <typename T> ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_queryResults;
    if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
        p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    else
        p_queryResults =
            new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), m_options.m_searchInternalResultNum, p_query.WithMeta(), p_query.WithVec());

    ErrorCode ret;
    if ((ret = m_topIndex->SearchIndex(*p_queryResults)) != ErrorCode::Success)
        return ret;

    for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1) break;

        res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
    }

    if ((ret = SearchDiskIndex(*p_queryResults)) != ErrorCode::Success)
        return ret;

    if (p_query.GetResultNum() < m_options.m_searchInternalResultNum)
    {
        std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(),
                  p_query.GetResults());
        p_query.SetScanned(p_queryResults->GetScanned());
        delete p_queryResults;
    }

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            // if (result > m_pMetadata->Count()) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "vid return is beyond the
            // metadata set:(%d > (%d, %d))\n", result, GetNumSamples(), m_pMetadata->Count());
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterative(QueryResult &p_headQuery, QueryResult &p_query,
                                         COMMON::WorkSpace *p_indexWorkspace, ExtraWorkSpace *p_extraWorkspace,
                                         int p_batch, int &resultCount, bool first) const
{
    /*
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_headQueryResults = (COMMON::QueryResultSet<T> *)&p_headQuery;
    COMMON::QueryResultSet<T> *p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;

    if (first)
    {
        p_headQueryResults->SetResultNum(m_options.m_searchInternalResultNum);
        p_headQueryResults->Reset();
        m_topIndex->SearchIndexIterativeFromNeareast(*p_headQueryResults, p_indexWorkspace, true);
        p_extraWorkspace->m_loadPosting = true;
    }

    bool continueSearch = true;
    resultCount = 0;
    while (continueSearch && resultCount < p_batch)
    {
        bool oldRelaxedMono = p_extraWorkspace->m_relaxedMono;
        ErrorCode ret = SearchDiskIndexIterative(p_headQuery, p_query, p_extraWorkspace);
        bool found = (ret == ErrorCode::Success);
        if (!found && ret != ErrorCode::VectorNotFound) return ret;
        p_extraWorkspace->m_loadPosting = false;
        if (!found)
        {
            p_headQueryResults->SetResultNum(m_options.m_headBatch);
            p_headQueryResults->Reset();
            continueSearch = m_topIndex->SearchIndexIterativeFromNeareast(*p_headQueryResults, p_indexWorkspace, false);
            p_extraWorkspace->m_loadPosting = true;

            if (!oldRelaxedMono && p_extraWorkspace->m_relaxedMono)
                continueSearch = false;
        }
        else
            resultCount++;
    }
    p_queryResults->SortResult();

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < resultCount; ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
    */
   return ErrorCode::Undefined;
}

template <typename T>
std::shared_ptr<ResultIterator> Index<T>::GetIterator(const void *p_target, bool p_searchDeleted, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    if (!m_bReady)
        return nullptr;
    auto extraWorkspace = m_workSpaceFactory->GetWorkSpace();
    if (!extraWorkspace)
    {
        extraWorkspace.reset(new ExtraWorkSpace());
        InitWorkSpace(extraWorkspace.get(), false);
    }
    else
    {
        InitWorkSpace(extraWorkspace.get(), true);
    }
    extraWorkspace->m_filterFunc = p_filterFunc;
    extraWorkspace->m_relaxedMono = false;
    extraWorkspace->m_loadedPostingNum = 0;
    extraWorkspace->m_deduper.clear();
    extraWorkspace->m_postingIDs.clear();
    std::shared_ptr<ResultIterator> resultIterator = std::make_shared<SPANNResultIterator<T>>(
        this, m_topIndex.get(), p_target, std::move(extraWorkspace),
        max(m_options.m_headBatch, m_options.m_searchInternalResultNum), p_maxCheck);
    return resultIterator;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeNext(QueryResult &p_query, COMMON::WorkSpace *workSpace, int p_batch,
                                             int &resultCount, bool p_isFirst, bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ITERATIVE NOT SUPPORT FOR SPANN");
    return ErrorCode::Undefined;
}

template <typename T> ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> space) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SearchIndexIterativeEnd NOT SUPPORT FOR SPANN");
    return ErrorCode::Fail;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<SPANN::ExtraWorkSpace> extraWorkspace) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    if (extraWorkspace != nullptr)
        m_workSpaceFactory->ReturnWorkSpace(std::move(extraWorkspace));

    return ErrorCode::Success;
}

template <typename T>
bool Index<T>::SearchIndexIterativeFromNeareast(QueryResult &p_query, COMMON::WorkSpace *p_space, bool p_isFirst,
                                                bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SearchIndexIterativeFromNeareast NOT SUPPORT FOR SPANN");
    return false;
}

template <typename T>
ErrorCode Index<T>::SearchIndexWithFilter(QueryResult &p_query, std::function<bool(const ByteArray &)> filterFunc,
                                          int maxCheck, bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not Support Filter on SPANN Index!\n");
    return ErrorCode::Fail;
}

template <typename T> 
ErrorCode Index<T>::SearchHeadIndex(QueryResult& p_query, int p_tolayer, ExtraWorkSpace *p_exWorkSpace) const
{
    COMMON::QueryResultSet<T> *p_queryResults;
    if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
        p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    else
        p_queryResults =
            new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), m_options.m_searchInternalResultNum, p_query.WithMeta(), p_query.WithVec());

    ErrorCode ret;
    if ((ret = m_topIndex->SearchIndex(*p_queryResults)) != ErrorCode::Success)
        return ret;

    for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1) break;

        res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
    }

    if (p_tolayer < m_extraSearchers.size() && (ret = SearchDiskIndex(*p_queryResults, nullptr, p_tolayer, p_exWorkSpace)) != ErrorCode::Success)
        return ret;

    if (p_query.GetResultNum() < m_options.m_searchInternalResultNum)
    {
        std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(),
                  p_query.GetResults());
        p_query.SetScanned(p_queryResults->GetScanned());
        delete p_queryResults;
    }
    return ErrorCode::Success;
}

template <typename T> 
ErrorCode Index<T>::SearchDiskIndex(QueryResult &p_query, SearchStats *p_stats, int p_tolayer, ExtraWorkSpace *p_exWorkSpace) const
{
    if (m_extraSearchers.size() == 0) return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    std::unique_ptr<ExtraWorkSpace> workSpace;
    if (p_exWorkSpace == nullptr) {
        workSpace = m_workSpaceFactory->GetWorkSpace();
        if (!workSpace)
        {
            workSpace.reset(new ExtraWorkSpace());
            InitWorkSpace(workSpace.get(), false);
        }
        else
        {
            InitWorkSpace(workSpace.get(), true);
        }
        p_exWorkSpace = workSpace.get();
    }

    COMMON::OptHashPosVector resultDedup;
    resultDedup.Init(m_options.m_maxCheck, m_options.m_hashExp);
    COMMON::QueryResultSet<T> localResults((const T *)p_query.GetTarget(), m_options.m_searchInternalResultNum, p_query.WithMeta(), p_query.WithVec());
    int k = 0;
    for (int i = 0, j = 0; i < p_queryResults->GetResultNum(); ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1) break;

        resultDedup.CheckAndSet(res->VID);
        if (j < m_options.m_searchInternalResultNum) *(localResults.GetResult(j++)) = *res;
        if (m_extraSearchers[p_tolayer]->ContainSample(res->VID)) {
            if (k < i) {
                *(p_queryResults->GetResult(k++)) = *res;
                res->VID = -1;
                res->Dist = MaxDist;
                res->Vec = ByteArray::c_empty;
            }
            else k++;
        } else {
            res->VID = -1;
            res->Dist = MaxDist;
            res->Vec = ByteArray::c_empty;
        }
    }
    p_queryResults->Reverse();

    ErrorCode ret;
    for (int layer = m_extraSearchers.size() - 1; layer >= p_tolayer; layer--) {
        auto& searcher = m_extraSearchers[layer];
        if (!searcher) return ErrorCode::Fail;

        p_exWorkSpace->m_deduper.clear();
        p_exWorkSpace->m_postingIDs.clear();

        for (int i = 0; i < m_options.m_searchInternalResultNum; ++i)
        {
            auto res = localResults.GetResult(i);
            if (res->VID == -1) continue;

            p_exWorkSpace->m_postingIDs.emplace_back(res->VID);

            if (resultDedup.CheckAndSet(res->VID)) continue;
            if (!m_extraSearchers[p_tolayer]->ContainSample(res->VID)) continue;
            p_queryResults->AddPoint(res->VID, res->Dist, res->Vec);
        }
        localResults.Reset();
        if ((ret = searcher->SearchIndex(p_exWorkSpace, localResults, p_stats)) !=
            ErrorCode::Success)
            return ret;
    }

    for (int i = 0; i < m_options.m_searchInternalResultNum; ++i)
    {
        auto res = localResults.GetResult(i);
        if (res->VID == -1) continue;

        if (resultDedup.CheckAndSet(res->VID)) continue;
        p_queryResults->AddPoint(res->VID, res->Dist, res->Vec);
    }
    p_queryResults->SortResult();
    if (workSpace != nullptr) m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchDiskIndexIterative(QueryResult &p_headQuery, QueryResult &p_query,
                                        ExtraWorkSpace *extraWorkspace) const
{
    /*
    if (extraWorkspace->m_loadPosting)
    {
        COMMON::QueryResultSet<T> *p__headQueryResults = (COMMON::QueryResultSet<T> *)&p_headQuery;
        // std::shared_ptr<ExtraWorkSpace> workSpace = m_workSpacePool->Rent();
        // workSpace->m_deduper.clear();
        extraWorkspace->m_postingIDs.clear();

        // float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;

        for (int i = 0; i < p__headQueryResults->GetResultNum(); ++i)
        {
            auto res = p__headQueryResults->GetResult(i);
            // break or continue
            if (res->VID == -1)
                break;

            extraWorkspace->m_postingIDs.emplace_back(res->VID);

            if (m_topLocalToGlobalID.R() != 0)
                res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
            else
            {
                res->VID = -1;
                res->Dist = MaxDist;
            }
            if (res->VID == MaxSize)
            {
                res->VID = -1;
                res->Dist = MaxDist;
            }
        }
        extraWorkspace->m_loadedPostingNum += (int)(extraWorkspace->m_postingIDs.size());
    }

    ErrorCode ret = m_extraSearcher->SearchIterativeNext(extraWorkspace, p_headQuery, p_query, m_topIndex, this);
    if (ret == ErrorCode::VectorNotFound && extraWorkspace->m_loadedPostingNum >= m_options.m_searchInternalResultNum)
        extraWorkspace->m_relaxedMono = true;
    return ret;
    */
   return ErrorCode::Undefined;
}

template <typename T> std::unique_ptr<COMMON::WorkSpace> Index<T>::RentWorkSpace(int batch, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RentWorkSpace NOT SUPPORT FOR SPANN");
    return nullptr;
}

template <typename T>
ErrorCode Index<T>::DebugSearchDiskIndex(QueryResult &p_query, int p_subInternalResultNum, int p_internalResultNum,
                                         SearchStats *p_stats, std::set<SizeType> *truth,
                                         std::map<SizeType, std::set<SizeType>> *found) const
{
    if (m_extraSearchers.size() == 0)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> newResults(*((COMMON::QueryResultSet<T> *)&p_query));
    for (int i = 0; i < newResults.GetResultNum(); ++i)
    {
        auto res = newResults.GetResult(i);
        if (res->VID == -1) break;

        auto global_VID = -1;
        if (m_topLocalToGlobalID.R() != 0)
            global_VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
        if (truth && truth->count(global_VID))
            (*found)[res->VID].insert(global_VID);
        res->VID = global_VID;
    }
    newResults.Reverse();

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();

    int partitions = (p_internalResultNum + p_subInternalResultNum - 1) / p_subInternalResultNum;
    float limitDist = p_query.GetResult(0)->Dist * m_options.m_maxDistRatio;
    for (int p = 0; p < partitions; p++)
    {
        int subInternalResultNum = min(p_subInternalResultNum, p_internalResultNum - p_subInternalResultNum * p);

        workSpace->m_postingIDs.clear();

        for (int i = p * p_subInternalResultNum; i < p * p_subInternalResultNum + subInternalResultNum; i++)
        {
            auto res = p_query.GetResult(i);
            if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                break;

            if (m_topLocalToGlobalID.R() > 0) res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
            workSpace->m_postingIDs.emplace_back(res->VID);
        }

        m_extraSearchers.back()->SearchIndex(workSpace.get(), newResults, p_stats, truth, found);
    }
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    newResults.SortResult();
    std::copy(newResults.GetResults(), newResults.GetResults() + newResults.GetResultNum(), p_query.GetResults());
    return ErrorCode::Success;
}
#pragma endregion

template <typename T> void Index<T>::SelectHeadAdjustOptions(int p_vectorCount)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Adjust Parameters...\n");

    if (m_options.m_headVectorCount != 0)
        m_options.m_ratio = m_options.m_headVectorCount * 1.0 / p_vectorCount;
    int headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
    if (headCnt == 0)
    {
        for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
        {
            m_options.m_ratio = minCnt / p_vectorCount;
            headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
        }

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
    }

    if (m_options.m_iBKTKmeansK > headCnt)
    {
        m_options.m_iBKTKmeansK = headCnt;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n",
                     headCnt);
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
                                            const Options &p_opts, std::vector<int> &p_selected)
{
    typedef std::pair<int, int> CSPair;
    std::vector<CSPair> children;
    int childrenSize = 1;
    const auto &node = (*p_tree)[p_nodeID];
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
            std::sort(children.begin(), children.end(),
                      [](const CSPair &a, const CSPair &b) { return a.second > b.second; });

            size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
            // if (selectCnt > 1) selectCnt -= 1;
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
void Index<T>::SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount,
                                     std::vector<int> &p_selected)
{
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

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                         opts.m_selectThreshold, opts.m_splitThreshold, diff * 100.0);

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

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Final Select Threshold: %d, Split Threshold: %d.\n",
                 opts.m_selectThreshold, opts.m_splitThreshold);

    p_selected.clear();
    SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
    std::sort(p_selected.begin(), p_selected.end());
    p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
}

template <typename T>
template <typename InternalDataType>
bool Index<T>::SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader)
{
    std::shared_ptr<VectorSet> vectorset = p_reader->GetVectorSet();
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized())
        vectorset->Normalize(m_options.m_iSelectHeadNumberOfThreads);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin initial data (%d,%d)...\n", vectorset->Count(),
                 vectorset->Dimension());

    COMMON::Dataset<InternalDataType> data(vectorset->Count(), vectorset->Dimension(), vectorset->Count(),
                                           vectorset->Count() + 1, (InternalDataType *)vectorset->GetData());

    auto t1 = std::chrono::high_resolution_clock::now();
    SelectHeadAdjustOptions(data.R());
    std::vector<int> selected;
    if (data.R() == 1)
    {
        selected.push_back(0);
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "Random"))
    {
        std::mt19937 rg;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
        selected.resize(data.R());
        for (int i = 0; i < data.R(); i++)
            selected[i] = i;
        std::shuffle(selected.begin(), selected.end(), rg);
        int headCnt = static_cast<int>(std::round(m_options.m_ratio * data.R()));
        selected.resize(headCnt);
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "BKT"))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
        std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
        bkt->m_iBKTKmeansK = m_options.m_iBKTKmeansK;
        bkt->m_iBKTLeafSize = m_options.m_iBKTLeafSize;
        bkt->m_iSamples = m_options.m_iSamples;
        bkt->m_iTreeNumber = m_options.m_iTreeNumber;
        bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
        bkt->m_parallelBuild = m_options.m_parallelBKTBuild;
        bkt->m_pQuantizer = m_pQuantizer;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d, ParallelBuild: %s.\n",
            bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber,
            m_options.m_iSelectHeadNumberOfThreads, m_options.m_parallelBKTBuild ? "true" : "false");

        if (bkt->m_parallelBuild) {
            bkt->BuildTreesParallel<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads,
                                              nullptr, nullptr, true);
        } else {
            bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads,
                                              nullptr, nullptr, true);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n",
                     elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

        if (m_options.m_saveBKT)
        {
            std::stringstream bktFileNameBuilder;
            bktFileNameBuilder << m_options.m_vectorPath << ".bkt." << m_options.m_iBKTKmeansK << "_"
                               << m_options.m_iBKTLeafSize << "_" << m_options.m_iTreeNumber << "_"
                               << m_options.m_iSamples << "_" << static_cast<int>(m_options.m_distCalcMethod) << ".bin";
            bkt->SaveTrees(bktFileNameBuilder.str());
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...Select Head Dynamically...\n");
        SelectHeadDynamically(bkt, data.R(), selected);

        if (selected.empty())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
            return false;
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Seleted Nodes: %u, about %.2lf%% of total.\n",
                 static_cast<unsigned int>(selected.size()), selected.size() * 100.0 / data.R());

    if (!m_options.m_noOutput)
    {
        std::sort(selected.begin(), selected.end());

        mkdir(m_options.m_indexDirectory.c_str());
        std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
        if (output == nullptr || outputIDs == nullptr ||
            !output->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(),
                                std::ios::binary | std::ios::out) ||
            !outputIDs->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                   std::ios::binary | std::ios::out))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(),
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
            return false;
        }

        SizeType val = static_cast<SizeType>(selected.size());
        if (output->WriteBinary(sizeof(val), reinterpret_cast<char *>(&val)) != sizeof(val))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
            return false;
        }
        if (outputIDs->WriteBinary(sizeof(val), reinterpret_cast<char *>(&val)) != sizeof(val))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputID file!\n");
            return false;
        }
        DimensionType dt = data.C();
        if (output->WriteBinary(sizeof(dt), reinterpret_cast<char *>(&dt)) != sizeof(dt))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
            return false;
        }
        dt = 1;
        if (outputIDs->WriteBinary(sizeof(dt), reinterpret_cast<char *>(&dt)) != sizeof(dt))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputID file!\n");
            return false;
        }
        for (int i = 0; i < selected.size(); i++)
        {
            SizeType vid = selected[i];
            if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char *>(&vid)) != sizeof(vid))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                return false;
            }

            if (output->WriteBinary(sizeof(InternalDataType) * data.C(), (char *)(data[vid])) !=
                sizeof(InternalDataType) * data.C())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                return false;
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n",
                 elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
    return true;
}

template <typename T> ErrorCode Index<T>::BuildIndexInternalLayer(std::shared_ptr<Helper::VectorSetReader> &p_reader)
{
    int currentLayer = static_cast<int>(m_extraSearchers.size());
    COMMON::Dataset<SizeType> localToGlobalID;
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading headIDFile for layer %d...\n", currentLayer - 1);
        std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
        if (ptr == nullptr ||
            !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                std::ios::binary | std::ios::in))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "No headIDFile file:%s\n",
                            (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
        }
        else {
            localToGlobalID.Load(ptr, this->m_iDataBlockSize, this->m_iDataCapacity);
            SizeType vectorCount = p_reader->GetVectorSet()->Count();
            if (localToGlobalID.R() != vectorCount)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "HeadIDFile count %lld doesn't match head vector file count %lld!\n", (int64_t)localToGlobalID.R(), (int64_t)vectorCount);
                localToGlobalID.SetR(0);
            }
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Select Head...\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    if (m_options.m_selectHead)
    {
        bool success = false;
        if (m_pQuantizer)
        {
            success = SelectHeadInternal<std::uint8_t>(p_reader);
        }
        else
        {
            success = SelectHeadInternal<T>(p_reader);
        }
        if (!success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
            return ErrorCode::Fail;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
    if (m_options.m_buildHead)
    {
        auto valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
        auto dims = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;

        m_topIndex = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
        m_topIndex->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
        m_topIndex->SetParameter("ParallelBKTBuild", m_options.m_parallelBKTBuild ? "true" : "false");
        m_topIndex->SetQuantizer(m_pQuantizer);
        for (const auto &iter : m_topParameters)
        {
            m_topIndex->SetParameter(iter.first.c_str(), iter.second.c_str());
        }

        std::shared_ptr<Helper::ReaderOptions> vectorOptions(
            new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
        if (ErrorCode::Success !=
            vectorReader->LoadFile(m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file.\n");
            return ErrorCode::Fail;
        }
        {
            auto headvectorset = vectorReader->GetVectorSet();
            if (m_topIndex->BuildIndex(headvectorset, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                return ErrorCode::Fail;
            }
            if (!m_options.m_quantizerFilePath.empty())
                m_topIndex->SetQuantizerFileName(
                    m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
            if (m_topIndex->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder) !=
                ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save head index.\n");
                return ErrorCode::Fail;
            }
        }
        m_topIndex.reset();
        if (LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_topIndex) !=
            ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double buildHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs\n", selectHeadTime,
                 buildHeadTime);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build SSDIndex...\n");
    if (m_options.m_enableSSD)
    {
        if (m_topIndex == nullptr && LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder,
                                            m_topIndex) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
            return ErrorCode::Fail;
        }
        m_topIndex->SetQuantizer(m_pQuantizer);
        if (!CheckHeadIndexType())
            return ErrorCode::Fail;

        m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
        m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
        m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));

        m_topIndex->UpdateIndex();

        if (m_options.m_storage == Storage::STATIC)
        {
            m_extraSearchers.emplace_back(std::make_shared<ExtraStaticSearcher<T>>(m_extraSearchers.size(), this));
        }
        else
        {
            m_extraSearchers.emplace_back(std::make_shared<ExtraDynamicSearcher<T>>(m_options, m_extraSearchers.size(), this, m_db));
        }

        {
            std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
            if (ptr == nullptr ||
                !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                 std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n",
                             (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                return ErrorCode::Fail;
            }
            m_topLocalToGlobalID.Load(ptr, m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity);
            m_topGlobalToLocalID.clear();
            for (int i = 0; i < m_topIndex->GetNumSamples(); i++)
            {
                SizeType globalID = *(m_topLocalToGlobalID[i]);
                m_topGlobalToLocalID[globalID] = i;
            }
        }

        if (m_options.m_buildSsdIndex)
        {
            if (m_options.m_storage != Storage::STATIC && !m_extraSearchers.back()->Available())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extrasearcher is not available and failed to initialize.\n");
                return ErrorCode::Fail;
            }
            if (!m_extraSearchers.back()->BuildIndex(p_reader, m_topIndex, m_options, m_topLocalToGlobalID, m_topGlobalToLocalID, localToGlobalID))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BuildSSDIndex Failed!\n");
                return ErrorCode::Fail;
            }
        }

        if (!m_extraSearchers.back()->LoadIndex(m_options))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot Load SSDIndex!\n");
            if (m_options.m_buildSsdIndex)
            {
                return ErrorCode::Fail;
            }
            else
            {
                m_extraSearchers.pop_back();
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double buildSSDTime = std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs build ssd time: %.2lfs\n",
                 selectHeadTime, buildHeadTime, buildSSDTime);

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader> &vectorReader, std::shared_ptr<Helper::ReaderOptions> &vectorOptions)
{
    if (!(m_options.m_indexDirectory.empty()) && !(direxists(m_options.m_indexDirectory.c_str())))
    {
        mkdir(m_options.m_indexDirectory.c_str());
    }
    if (!(m_options.m_persistentBufferPath.empty()) && !(direxists(m_options.m_persistentBufferPath.c_str())))
    {
        mkdir(m_options.m_persistentBufferPath.c_str());
    }

    if (m_freeWorkSpaceIds == nullptr) {
        m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
        int maxIOThreads = (m_options.m_storage == Storage::STATIC) ? max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) : max(m_options.m_ioThreads, (2 * max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) +
                                    (m_options.m_layers + 1) * (m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum)));
        for (int i = 0; i < maxIOThreads; i++) {
            m_freeWorkSpaceIds->push(i);
        }
        m_workspaceCount = maxIOThreads;
    }

    if (m_db == nullptr && m_options.m_shareDB) PrepareDB(m_db);

    int resumeLayer = m_options.m_resumeLayer;

    if (resumeLayer >= 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Resuming build from layer %d (loading layers 0..%d from existing data)\n",
                     resumeLayer, resumeLayer - 1);

        // Save original file path options
        std::string origHeadIDFile = m_options.m_headIDFile;
        std::string origHeadVectorFile = m_options.m_headVectorFile;
        std::string origHeadIndexFolder = m_options.m_headIndexFolder;

        auto setLayerPaths = [&](int layer) {
            if (m_options.m_layers > 1 && layer < m_options.m_layers - 1) {
                std::string suffix = "_layer" + std::to_string(layer);
                m_options.m_headIDFile = origHeadIDFile + suffix;
                m_options.m_headVectorFile = origHeadVectorFile + suffix;
                m_options.m_headIndexFolder = origHeadIndexFolder + suffix;
            } else {
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
            }
        };

        // For resume, we need the head index that corresponds to the layer ABOVE the resume layer.
        // For 2-layer: resumeLayer=1 means load Layer 1's head index (the last layer, default paths).
        setLayerPaths(resumeLayer);
        std::string headIndexPath = m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder;
        std::string headIDPath = m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;

        // Load head index from disk
        if (LoadIndex(headIndexPath, m_topIndex) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index for resume from %s\n", headIndexPath.c_str());
            m_options.m_headIDFile = origHeadIDFile;
            m_options.m_headVectorFile = origHeadVectorFile;
            m_options.m_headIndexFolder = origHeadIndexFolder;
            return ErrorCode::Fail;
        }
        m_topIndex->SetQuantizer(m_pQuantizer);
        m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
        m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
        m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
        m_topIndex->UpdateIndex();

        // Load head ID mapping
        {
            std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
            if (ptr != nullptr && ptr->Initialize(headIDPath.c_str(),
                    std::ios::binary | std::ios::in)) {
                m_topLocalToGlobalID.Load(ptr, m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity);
                m_topGlobalToLocalID.clear();
                for (int i = 0; i < m_topIndex->GetNumSamples(); i++) {
                    SizeType globalID = *(m_topLocalToGlobalID[i]);
                    m_topGlobalToLocalID[globalID] = i;
                }
            }
        }

        // Load existing layers below resumeLayer
        for (int i = 0; i < resumeLayer; i++) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading existing data for layer %d\n", i);
            if (m_options.m_storage == Storage::STATIC) {
                m_extraSearchers.push_back(std::make_shared<ExtraStaticSearcher<T>>(i, this));
            } else {
                m_extraSearchers.push_back(std::make_shared<ExtraDynamicSearcher<T>>(m_options, i, this, m_db));
            }
            if (!m_extraSearchers.back()->LoadIndex(m_options)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to load existing index for layer %d\n", i);
                return ErrorCode::Fail;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Layer %d loaded successfully\n", i);
        }

        // For the resume layer and above, build SSD only (skip SelectHead + BuildHead)
        bool origSelectHead = m_options.m_selectHead;
        bool origBuildHead = m_options.m_buildHead;
        m_options.m_selectHead = false;
        m_options.m_buildHead = false;

        for (int layer = resumeLayer; layer < m_options.m_layers; layer++) {
            setLayerPaths(layer);

            if (layer == 0) {
                // Layer 0 uses the original base vectors
                auto ret = BuildIndexInternalLayer(vectorReader);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to resume build for layer 0.\n");
                    m_options.m_selectHead = origSelectHead;
                    m_options.m_buildHead = origBuildHead;
                    m_options.m_headIDFile = origHeadIDFile;
                    m_options.m_headVectorFile = origHeadVectorFile;
                    m_options.m_headIndexFolder = origHeadIndexFolder;
                    return ret;
                }
            } else {
                // Layer N uses the head vectors from the previous layer as input
                setLayerPaths(layer - 1);
                std::string prevHeadVectors = m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile;
                setLayerPaths(layer);

                vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(prevHeadVectors)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read headvector file for layer %d from %s.\n", layer, prevHeadVectors.c_str());
                    m_options.m_selectHead = origSelectHead;
                    m_options.m_buildHead = origBuildHead;
                    m_options.m_headIDFile = origHeadIDFile;
                    m_options.m_headVectorFile = origHeadVectorFile;
                    m_options.m_headIndexFolder = origHeadIndexFolder;
                    return ErrorCode::Fail;
                }
                auto ret = BuildIndexInternalLayer(vectorReader);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to resume build for layer %d.\n", layer);
                    m_options.m_selectHead = origSelectHead;
                    m_options.m_buildHead = origBuildHead;
                    m_options.m_headIDFile = origHeadIDFile;
                    m_options.m_headVectorFile = origHeadVectorFile;
                    m_options.m_headIndexFolder = origHeadIndexFolder;
                    return ret;
                }
            }
        }

        m_options.m_selectHead = origSelectHead;
        m_options.m_buildHead = origBuildHead;
        m_options.m_headIDFile = origHeadIDFile;
        m_options.m_headVectorFile = origHeadVectorFile;
        m_options.m_headIndexFolder = origHeadIndexFolder;
    } else {
        // Normal build path
        // Save original file path options for per-layer suffixing
        std::string origHeadIDFile = m_options.m_headIDFile;
        std::string origHeadVectorFile = m_options.m_headVectorFile;
        std::string origHeadIndexFolder = m_options.m_headIndexFolder;

        auto setLayerPaths = [&](int layer) {
            if (m_options.m_layers > 1 && layer < m_options.m_layers - 1) {
                std::string suffix = "_layer" + std::to_string(layer);
                m_options.m_headIDFile = origHeadIDFile + suffix;
                m_options.m_headVectorFile = origHeadVectorFile + suffix;
                m_options.m_headIndexFolder = origHeadIndexFolder + suffix;
            } else {
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
            }
        };

        // Layer 0
        setLayerPaths(0);
        auto ret = BuildIndexInternalLayer(vectorReader);
        if (ret != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build index layer 0.\n");
            m_options.m_headIDFile = origHeadIDFile;
            m_options.m_headVectorFile = origHeadVectorFile;
            m_options.m_headIndexFolder = origHeadIndexFolder;
            return ret;
        }
        for (int layer = 1; layer < m_options.m_layers; layer++)
        {
            // Load previous layer's head vectors as input for this layer
            std::string prevHeadVectors = m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile;
            std::string prevHeadIDFile = m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
            setLayerPaths(layer);

            // Copy previous layer's headIDFile to current layer's path so that
            // BuildIndexInternalLayer can load it as localToGlobalID (mapping from
            // the previous layer's local head IDs to original global vector IDs).
            // SelectHead will overwrite this file later with the current layer's head IDs.
            std::string curHeadIDFile = m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
            std::error_code ec;
            std::filesystem::copy_file(prevHeadIDFile, curHeadIDFile, std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Could not copy previous layer headIDFile %s to %s: %s\n",
                             prevHeadIDFile.c_str(), curHeadIDFile.c_str(), ec.message().c_str());
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Copied localToGlobalID from %s to %s for layer %d\n",
                             prevHeadIDFile.c_str(), curHeadIDFile.c_str(), layer);
            }

            vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
            if (ErrorCode::Success != vectorReader->LoadFile(prevHeadVectors))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read headvector file for layer %d from %s.\n", layer, prevHeadVectors.c_str());
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
                return ErrorCode::Fail;
            }
            auto ret = BuildIndexInternalLayer(vectorReader);
            if (ret != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build index layer %d.\n", layer);
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
                return ret;
            }
        }

        // Restore original options
        m_options.m_headIDFile = origHeadIDFile;
        m_options.m_headVectorFile = origHeadVectorFile;
        m_options.m_headIndexFolder = origHeadIndexFolder;
    } // end of normal vs resume path

    // Clean up last layer's head vector file (not needed at runtime)
    if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str())) 
    {
        if (remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to delete head vector file: %s\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str());
        }
    }
    m_bReady = true;
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::BuildIndex(bool p_normalized)
{
    SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
    SizeType dim = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
    std::shared_ptr<Helper::ReaderOptions> vectorOptions(
        new Helper::ReaderOptions(valueType, dim, m_options.m_vectorType, m_options.m_vectorDelimiter,
                                  m_options.m_iSSDNumberOfThreads, p_normalized));
    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (m_options.m_vectorPath.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
    }
    else
    {
        if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            return ErrorCode::Fail;
        }
        m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
    }

    return BuildIndexInternal(vectorReader, vectorOptions);
}

template <typename T>
ErrorCode Index<T>::BuildIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized,
                               bool p_shareOwnership)
{
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;

    std::shared_ptr<VectorSet> vectorSet;
    if (p_shareOwnership)
    {
        vectorSet.reset(
            new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                               GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }
    else
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
    {
        vectorSet->Normalize(m_options.m_iSSDNumberOfThreads);
    }
    SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
    std::shared_ptr<Helper::ReaderOptions> vectorOptions(
        new Helper::ReaderOptions(valueType, p_dimension, VectorFileType::DEFAULT, m_options.m_vectorDelimiter,
                                  m_options.m_iSSDNumberOfThreads, true));
    
    std::shared_ptr<Helper::VectorSetReader> vectorReader(new Helper::MemoryVectorReader(
        vectorOptions,
        vectorSet));

    m_options.m_valueType = GetEnumValueType<T>();
    m_options.m_dim = p_dimension;
    m_options.m_vectorSize = p_vectorNum;

    return BuildIndexInternal(vectorReader, vectorOptions);
}

template <typename T> ErrorCode Index<T>::UpdateIndex()
{
    m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    // m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    // m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_topIndex->UpdateIndex();
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams,
                                IAbortOperation *p_abort, std::vector<SizeType> *p_mapping)
{
    if (m_topIndex == nullptr)
        return ErrorCode::EmptyIndex;

    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::lock_guard<std::mutex> lock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);
    ErrorCode ret;
    for (int layer = 0; layer < m_extraSearchers.size(); layer++)
    {
        if ((ret = m_extraSearchers[layer]->RefineIndex()) != ErrorCode::Success)
            return ret;
    }

    std::vector<SizeType> headOldtoNew;

    if ((ret = m_topIndex->RefineIndex(p_indexStreams, nullptr, &headOldtoNew)) != ErrorCode::Success)
        return ret;

    COMMON::Dataset<SizeType> new_topLocalToGlobalID(m_topIndex->GetNumSamples() - m_topIndex->GetNumDeleted(), 1,
                                                          m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity);
    for (int i = 0; i < m_topLocalToGlobalID.R(); i++)
    {
        if (m_topIndex->ContainSample(i))
        {
            *(new_topLocalToGlobalID[headOldtoNew[i]]) = *(m_topLocalToGlobalID[i]);
        }
    }
    new_topLocalToGlobalID.Save(p_indexStreams[m_topIndex->GetIndexFiles()->size()]);
    
    if ((ret = VectorIndex::LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder,
                                                m_topIndex)) != ErrorCode::Success)
        return ret;

    if ((ret = m_topLocalToGlobalID.Load(
                             m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile, m_topIndex->m_iDataBlockSize,
                             m_topIndex->m_iDataCapacity)) != ErrorCode::Success)
                        return ret;
    m_topGlobalToLocalID.clear();
    for (int i = 0; i < m_topLocalToGlobalID.R(); i++)
    {    
        SizeType globalID = *(m_topLocalToGlobalID[i]);
        m_topGlobalToLocalID[globalID] = i;
    }

    for (int i = 0; i < p_indexStreams.size(); i++)
    {
        p_indexStreams[i]->ShutDown();
    }
    return ret;
}

template <typename T> ErrorCode Index<T>::SetParameter(const char *p_param, const char *p_value, const char *p_section)
{
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_topIndex != nullptr)
            return m_topIndex->SetParameter(p_param, p_value);
        else
            m_topParameters[p_param] = p_value;
    }
    else
    {
        m_options.SetParameter(p_section, p_param, p_value);
    }
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod"))
    {
        if (m_pQuantizer)
        {
            m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
            m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                                ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase()
                                : 1;
        }
        else
        {
            m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
            m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                                ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>()
                                : 1;
        }
    }
    return ErrorCode::Success;
}

template <typename T> std::string Index<T>::GetParameter(const char *p_param, const char *p_section) const
{
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_topIndex != nullptr)
            return m_topIndex->GetParameter(p_param);
        else
        {
            auto iter = m_topParameters.find(p_param);
            if (iter != m_topParameters.end())
                return iter->second;
            return "Undefined!";
        }
    }
    else
    {
        return m_options.GetParameter(p_section, p_param);
    }
}

// Add insert entry to persistent buffer
template <typename T>
ErrorCode Index<T>::AddIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension,
                             std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex, bool p_normalized, SizeType* VID)
{
    if ((m_options.m_storage == Storage::STATIC) || m_extraSearchers.size() == 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
        return ErrorCode::Fail;
    }

    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;
    if (p_dimension != GetFeatureDim())
        return ErrorCode::DimensionSizeMismatch;

    std::shared_lock<std::shared_timed_mutex> lock(m_checkPointLock);

    SizeType begin, end;
    {
        std::lock_guard<std::mutex> lock(m_dataAddLock);

        begin = m_options.m_vectorSize;
        end = begin + p_vectorNum;

        if (begin == 0)
        {
            return ErrorCode::EmptyIndex;
        }

        for (int layer = 0; layer < m_extraSearchers.size(); layer++)
        {
            if (m_extraSearchers[layer]->AddIDCapacity(p_vectorNum, layer == 0? false : true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MemoryOverFlow for layer %d: add VID: %d\n", layer, begin);
                return ErrorCode::MemoryOverFlow;
            }
        }

        if (m_pMetadata != nullptr)
        {
            if (p_metadataSet != nullptr)
            {
                m_pMetadata->AddBatch(*p_metadataSet);
                if (HasMetaMapping())
                {
                    for (SizeType i = begin; i < end; i++)
                    {
                        ByteArray meta = m_pMetadata->GetMetadata(i);
                        std::string metastr((char *)meta.Data(), meta.Length());
                        UpdateMetaMapping(metastr, i);
                    }
                }
            }
            else
            {
                for (SizeType i = begin; i < end; i++)
                    m_pMetadata->Add(ByteArray::c_empty);
            }
        }
        m_options.m_vectorSize = end;
    }

    if (VID != nullptr)
    {
        for (int i = 0; i < p_vectorNum; i++) VID[i] = begin + i;
    }

    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; i++)
        {
            COMMON::Utils::Normalize((T *)(vectorSet->GetVector(i)), p_dimension, base);
        }
    }
    else
    {
        vectorSet.reset(
            new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                               GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();
    return m_extraSearchers[0]->AddIndex(workSpace.get(), vectorSet, begin);
}

template <typename T>
ErrorCode Index<T>::Check()
{
    std::atomic<ErrorCode> ret = ErrorCode::Success;
    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    //if ((ret = m_topIndex->Check()) != ErrorCode::Success)
    //    return ret;

    std::vector<std::thread> mythreads;
    mythreads.reserve(m_options.m_iSSDNumberOfThreads);
    for (int layer = m_extraSearchers.size() - 1; layer >= 0; layer--)
    {
        std::vector<SizeType> globalIDs;
        GetHeadIndexMapping(layer + 1, globalIDs);
        SizeType currentPostingNum = globalIDs.size();
        std::atomic_size_t sent(0);
        std::vector<std::uint8_t> checked(m_extraSearchers[layer]->GetNumBlocks(), false);
        for (int tid = 0; tid < m_options.m_iSSDNumberOfThreads; tid++)
        {
            mythreads.emplace_back([&, tid]() {
                auto workSpace = m_workSpaceFactory->GetWorkSpace();
                if (!workSpace)
                {
                    workSpace.reset(new ExtraWorkSpace());
                    InitWorkSpace(workSpace.get(), false);
                }
                else
                {
                    InitWorkSpace(workSpace.get(), true);
                }
                size_t i = 0;
                while (true)
                {
                    i = sent.fetch_add(1);
                    if (i < currentPostingNum)
                    {
                        if (ContainSample(i, layer + 1))
                        {
                            if (m_extraSearchers[layer]->CheckPosting(globalIDs[i], &checked, workSpace.get()) != ErrorCode::Success)
                            {
                                ret = ErrorCode::Fail;
                                return;
                            }
                        }
                    }
                    else
                    {
                        return;
                    }
                }
                m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            });
        }
        for (auto &t : mythreads)
        {
            t.join();
        }
        mythreads.clear();
    }
    return ret.load();
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const SizeType &p_id, int tolayer)
{
    std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
    if (tolayer >= m_extraSearchers.size()) {
        auto it = m_topGlobalToLocalID.find(p_id);
        if (it != m_topGlobalToLocalID.end())
        {
            m_topIndex->DeleteIndex(it->second);
        } else {
            return ErrorCode::VectorNotFound;
        }
        return ErrorCode::Success;
    }
    return m_extraSearchers[tolayer]->DeleteIndex(p_id);
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const void *p_vectors, SizeType p_vectorNum)
{
    // TODO: Support batch delete
    DimensionType p_dimension = GetFeatureDim();
    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_vectors, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; i++)
        {
            COMMON::Utils::Normalize((T *)(vectorSet->GetVector(i)), p_dimension, base);
        }
    }
    else
    {
        vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t *)p_vectors, sizeof(T) * p_vectorNum * p_dimension, false),
                                           GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    for (int i = 0; i < p_vectorNum; i++) {
        QueryResult queryResults(vectorSet->GetVector(i), 1, false);
        SearchHeadIndex(queryResults, 0);
        if (queryResults.GetResult(0)->Dist < Epsilon)
        {
            DeleteIndex(queryResults.GetResult(0)->VID);
        }
        else {
            return ErrorCode::VectorNotFound;
        }
    }
    return ErrorCode::Success;
}

template <typename T> void Index<T>::PrepareDB(std::shared_ptr<Helper::KeyValueIO>& db, int layer)
{
    if(m_options.m_storage == Storage::FILEIO) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseFileIO\n");
        db.reset(new FileIO(m_options, layer));
    }
    else if (m_options.m_storage == Storage::SPDKIO) {
#ifdef SPDK
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseSPDK\n");
        db.reset(new SPDKIO(m_options));
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex:SPDK unsupport! Use -DSPDK to enable SPDK when doing cmake.\n");
        return;
#endif
    } 
    else if (m_options.m_storage == Storage::ROCKSDBIO) {
#ifdef ROCKSDB
        std::string indexDir = (m_options.m_recovery)? m_options.m_persistentBufferPath + FolderSep: m_options.m_indexDirectory + FolderSep;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseKV\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:dbPath:%s\n", (indexDir + m_options.m_KVFile + "_" + std::to_string(layer)).c_str());
        db.reset(new RocksDBIO((indexDir + m_options.m_KVFile + "_" + std::to_string(layer)).c_str(), m_options.m_useDirectIO, m_options.m_enableWAL, m_options.m_recovery));
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex:RocksDB unsupport! Use -DROCKSDB to enable RocksDB when doing cmake.\n");
        return;
#endif
    }
    else if (m_options.m_storage == Storage::TIKVIO) {
#ifdef TIKV
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseTiKV\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:PD addresses:%s, prefix:%s\n",
                     m_options.m_tikvPDAddresses.c_str(), m_options.m_tikvKeyPrefix.c_str());
        db.reset(new TiKVIO(m_options.m_tikvPDAddresses, m_options.m_tikvKeyPrefix));
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex:TiKV unsupport! Use -DTIKV to enable TiKV when doing cmake.\n");
        return;
#endif
    }
}
} // namespace SPANN
} // namespace SPTAG

#define DefineVectorValueType(Name, Type) template class SPTAG::SPANN::Index<Type>;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
