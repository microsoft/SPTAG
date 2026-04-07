// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CoreInterface.h"
#include "inc/Helper/StringConvert.h"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "inc/Core/SPANN/Options.h"
#include "inc/Core/SPANN/ExtraFileController.h"
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>

namespace {

bool EnsureDir(const std::string& path)
{
    if (path.empty()) return false;

    std::string cmd = "mkdir -p \"" + path + "\"";
    return std::system(cmd.c_str()) == 0;
}

bool RemovePathRecursive(const std::string& path)
{
    if (path.empty()) return false;

    std::string cmd = "rm -rf \"" + path + "\"";
    return std::system(cmd.c_str()) == 0;
}

bool CopyDirRecursive(const std::string& src, const std::string& dst)
{
    if (src.empty() || dst.empty()) return false;

    if (!RemovePathRecursive(dst)) return false;

    std::string cmd = "cp -a \"" + src + "\" \"" + dst + "\"";
    return std::system(cmd.c_str()) == 0;
}

} // namespace

AnnIndex::AnnIndex(DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::BKT), m_inputValueType(SPTAG::VectorValueType::Float), m_dimension(p_dimension)
{
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::AnnIndex(const char *p_algoType, const char *p_valueType, DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::Undefined), m_inputValueType(SPTAG::VectorValueType::Undefined),
      m_dimension(p_dimension)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_inputValueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::AnnIndex(const std::shared_ptr<SPTAG::VectorIndex> &p_index)
    : m_algoType(p_index->GetIndexAlgoType()), m_inputValueType(p_index->GetVectorValueType()),
      m_dimension(p_index->GetFeatureDim()), m_index(p_index)
{
    m_inputVectorSize = p_index->m_pQuantizer ? p_index->m_pQuantizer->GetNumSubvectors()
                                              : SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::~AnnIndex()
{
}

bool AnnIndex::BuildSPANN(bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index)
        return false;

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

bool AnnIndex::BuildSPANNWithMetaData(ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index)
        return false;

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex)
        m_index->BuildMetaMapping(false);

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

// Build SPANN index with both vector data and metadata (for attribute filtering support)
bool AnnIndex::BuildSPANNWithDataAndMeta(ByteArray p_data, ByteArray p_meta, SizeType p_num,
                                          bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
        return false;

    // Set metadata first (before build, so it's available during search)
    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex)
        m_index->BuildMetaMapping(false);

    // Build with in-memory vector data
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                              (SPTAG::DimensionType)m_dimension, p_normalized));
}

bool AnnIndex::Build(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                             (SPTAG::DimensionType)m_dimension, p_normalized));
}

bool AnnIndex::BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex,
                                 bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    auto vectorType = m_index->m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_inputValueType;
    auto vectorSize = m_index->m_pQuantizer ? m_index->m_pQuantizer->GetNumSubvectors() : m_dimension;
    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(
        p_data, vectorType, static_cast<SPTAG::DimensionType>(vectorSize), static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize));
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(vectors, meta, p_withMetaIndex, p_normalized));
}

void AnnIndex::SetBuildParam(const char *p_name, const char *p_value, const char *p_section)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return;
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    m_index->SetParameter(p_name, p_value, p_section);
}

void AnnIndex::SetSearchParam(const char *p_name, const char *p_value, const char *p_section)
{
    if (nullptr != m_index)
        m_index->SetParameter(p_name, p_value, p_section);
}

std::shared_ptr<ResultIterator> AnnIndex::GetIterator(ByteArray p_target)
{
    if (nullptr != m_index)
        return m_index->GetIterator(p_target.Data());
    return nullptr;
}

bool AnnIndex::LoadQuantizer(const char *p_quantizerFile)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return false;
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }

    auto ret = (m_index->LoadQuantizer(p_quantizerFile) == SPTAG::ErrorCode::Success);
    if (ret)
    {
        m_inputVectorSize = m_index->m_pQuantizer->QuantizeSize();
    }
    return ret;
}

void AnnIndex::SetQuantizerADC(bool p_adc)
{
    if (nullptr != m_index)
        return m_index->SetQuantizerADC(p_adc);
}

ByteArray AnnIndex::QuantizeVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr)
    {
        size_t outsize = m_index->GetQuantizer()->GetNumSubvectors() * (size_t)p_num;
        std::uint8_t *outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success !=
            m_index->QuantizeVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false)))
            return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}

ByteArray AnnIndex::ReconstructVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr)
    {
        size_t outsize = m_index->GetQuantizer()->ReconstructSize() * (size_t)p_num;
        std::uint8_t *outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success !=
            m_index->ReconstructVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false)))
            return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}

std::shared_ptr<QueryResult> AnnIndex::Search(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, false);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::SearchWithMetaData(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::BatchSearch(ByteArray p_data, int p_vectorNum, int p_resultNum,
                                                   bool p_withMetaData)
{
    std::shared_ptr<QueryResult> results =
        std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, p_withMetaData);
    if (nullptr != m_index)
    {
        m_index->SearchIndex(p_data.Data(), p_vectorNum, p_resultNum, p_withMetaData, results->GetResults());
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::SearchWithTenantFilter(ByteArray p_data, int p_resultNum, const char* p_tenantId)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);
    
    if (nullptr != m_index && nullptr != p_tenantId)
    {
        // Create filter function that checks if metadata exactly matches tenantId
        std::string tenantId(p_tenantId);
        auto filterFunc = [tenantId](const SPTAG::ByteArray& metadata) -> bool {
            if (metadata.Length() == 0) return false;
            std::string meta(reinterpret_cast<const char*>(metadata.Data()), metadata.Length());
            // Trim trailing whitespace/newline
            while (!meta.empty() && (meta.back() == '\n' || meta.back() == '\r' || meta.back() == ' '))
                meta.pop_back();
            return meta == tenantId;
        };
        
        m_index->SearchIndexWithFilter(*results, filterFunc, 0, false);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::BatchSearchWithTenantFilter(ByteArray p_data, int p_vectorNum, 
                                                                    int p_resultNum, const char* p_tenantId)
{
    std::shared_ptr<QueryResult> results = 
        std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, true);
    
    if (nullptr != m_index && nullptr != p_tenantId && nullptr != p_data.Data())
    {
        // For batch search with filter, we need to process each vector separately
        // since the batch SearchIndex doesn't support filtering
        std::string tenantId(p_tenantId);
        auto filterFunc = [tenantId](const SPTAG::ByteArray& metadata) -> bool {
            if (metadata.Length() == 0) return false;
            std::string meta(reinterpret_cast<const char*>(metadata.Data()), metadata.Length());
            return meta.find(tenantId) != std::string::npos;
        };
        
        SPTAG::BasicResult* results_array = results->GetResults();
        const char* data = reinterpret_cast<const char*>(p_data.Data());
        size_t vectorSize = p_data.Length() / p_vectorNum;
        
        for (int i = 0; i < p_vectorNum; i++)
        {
            SPTAG::QueryResult singleQuery(data + i * vectorSize, p_resultNum, true);
            m_index->SearchIndexWithFilter(singleQuery, filterFunc, 0, false);
            
            // Copy results
            for (int j = 0; j < p_resultNum && j < singleQuery.GetResultNum(); j++)
            {
                auto* one = singleQuery.GetResult(j);
                if (one != nullptr)
                {
                    results_array[i * p_resultNum + j] = *one;
                }
            }
        }
    }
    return std::move(results);
}

bool AnnIndex::ReadyToServe() const
{
    return m_index != nullptr;
}

void AnnIndex::UpdateIndex()
{
    m_index->UpdateIndex();
}

bool AnnIndex::Save(const char *p_savefile) const
{
    return SPTAG::ErrorCode::Success == m_index->SaveIndex(p_savefile);
}

bool AnnIndex::Add(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->AddIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                           (SPTAG::DimensionType)m_dimension, nullptr, false,
                                                           p_normalized));
}

bool AnnIndex::AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex,
                               bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(
        p_data, m_inputValueType, static_cast<SPTAG::DimensionType>(m_dimension), static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num));
    return (SPTAG::ErrorCode::Success == m_index->AddIndex(vectors, meta, p_withMetaIndex, p_normalized));
}

bool AnnIndex::Delete(ByteArray p_data, SizeType p_num)
{
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_data.Data(), (SPTAG::SizeType)p_num));
}

bool AnnIndex::DeleteByMetaData(ByteArray p_meta)
{
    if (nullptr == m_index)
        return false;

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_meta));
}

uint64_t AnnIndex::CalculateBufferSize()
{
    if (nullptr == m_index)
        return 0;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    uint64_t total = sizeof(int) + sizeof(uint64_t) * buffersize->size();
    for (uint64_t bs : *buffersize)
    {
        total += bs;
    }
    return total;
}

ByteArray AnnIndex::Dump(ByteArray p_blobs)
{
    if (nullptr == m_index)
        return ByteArray::c_empty;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    std::uint8_t *ptr = p_blobs.Data(), *pdata = ptr + sizeof(int) + sizeof(uint64_t) * buffersize->size();
    *((int *)ptr) = (int)(buffersize->size());
    ptr += sizeof(int);

    std::vector<SPTAG::ByteArray> indexBlobs;
    for (size_t i = 0; i < buffersize->size(); i++)
    {
        *((uint64_t *)ptr) = buffersize->at(i);
        ptr += sizeof(uint64_t);
        indexBlobs.push_back(SPTAG::ByteArray(pdata, buffersize->at(i), false));
        pdata += buffersize->at(i);
    }

    std::string config;
    if (SPTAG::ErrorCode::Success != m_index->SaveIndex(config, indexBlobs))
    {
        return ByteArray::c_empty;
    }
    std::uint8_t *newdata = new std::uint8_t[config.size()];
    memcpy(newdata, config.c_str(), config.size());
    return ByteArray(newdata, config.size(), false);
}

AnnIndex AnnIndex::LoadFromDump(ByteArray p_config, ByteArray p_blobs)
{
    if (p_config.Length() == 0)
        return AnnIndex(0);

    std::uint8_t *ptr = p_blobs.Data();
    int streamNum = *((int *)ptr);
    ptr += sizeof(int);
    std::uint8_t *pdata = ptr + sizeof(uint64_t) * streamNum;

    std::vector<SPTAG::ByteArray> p_indexBlobs;
    for (int i = 0; i < streamNum; i++)
    {
        std::uint64_t streamSize = *((uint64_t *)ptr);
        ptr += sizeof(uint64_t);
        p_indexBlobs.push_back(SPTAG::ByteArray((std::uint8_t *)pdata, streamSize, false));
        pdata += streamSize;
    }

    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    std::string config((char *)p_config.Data(), p_config.Length());
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(config, p_indexBlobs, vecIndex) ||
        nullptr == vecIndex)
    {
        return AnnIndex(0);
    }
    return AnnIndex(vecIndex);
}

AnnIndex AnnIndex::Load(const char *p_loaderFile)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    auto ret = SPTAG::VectorIndex::LoadIndex(p_loaderFile, vecIndex);
    if (SPTAG::ErrorCode::Success != ret || nullptr == vecIndex)
    {
        return AnnIndex(0);
    }

    return AnnIndex(vecIndex);
}

AnnIndex AnnIndex::Merge(const char *p_indexFilePath1, const char *p_indexFilePath2)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex, addIndex;
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath1, vecIndex) ||
        SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath2, addIndex) ||
        SPTAG::ErrorCode::Success !=
            vecIndex->MergeIndex(addIndex.get(), std::atoi(vecIndex->GetParameter("NumberOfThreads").c_str()), nullptr))
        return AnnIndex(0);

    return AnnIndex(vecIndex);
}

// ============================================================================
// TenantIndexManager Implementation
// ============================================================================

TenantIndexManager::TenantIndexManager(DimensionType p_dimension, const char* p_algoType, const char* p_valueType)
    : m_dimension(p_dimension), m_algoType(SPTAG::IndexAlgoType::Undefined), 
    m_valueType(SPTAG::VectorValueType::Undefined),
    m_headIndexCacheLimitBytes(1024*1024*1024),  // Default 1GB cache limit
    m_headCache(nullptr)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_valueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_valueType) * m_dimension;

    // Initialize shared AIO pool (4 contexts, 64 events each) — created once, never destroyed
    SPTAG::Helper::SharedAIOPool::Instance().Initialize(4, 64);
}

TenantIndexManager::~TenantIndexManager()
{
    if (m_headCache) m_headCache->Clear();
    m_tenantIndices.clear();
    m_tenantVectorCounts.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantPostingOffsets.clear();
    m_tenantHeadCounts.clear();
}

bool TenantIndexManager::BuildFromData(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum,
                                       bool p_withMetaIndex, bool p_normalized)
{
    if (p_vectorNum == 0 || m_dimension == 0 || p_vectors.Length() != p_vectorNum * m_inputVectorSize)
    {
        return false;
    }

    m_tenantIndices.clear();
    m_tenantVectorCounts.clear();
    m_tenantSpannWorkDirs.clear();
    
    
    

    std::map<int, std::vector<std::pair<const uint8_t*, size_t>>> tenantVectorRanges;
    std::map<int, std::vector<std::string>> tenantMetadataLines;

    const char* metaPtr = reinterpret_cast<const char*>(p_metadata.Data());
    const char* metaEnd = metaPtr + p_metadata.Length();
    const uint8_t* vectorPtr = p_vectors.Data();

    SizeType globalIdx = 0;
    while (metaPtr < metaEnd && globalIdx < p_vectorNum)
    {
        const char* lineEnd = metaPtr;
        while (lineEnd < metaEnd && *lineEnd != '\n')
        {
            lineEnd++;
        }

        if (lineEnd == metaPtr)
        {
            return false;
        }

        std::string metaLine(metaPtr, lineEnd - metaPtr);
        int tenantId = RegisterTenantId(metaLine.c_str());

        tenantVectorRanges[tenantId].push_back({vectorPtr, m_inputVectorSize});
        tenantMetadataLines[tenantId].push_back(metaLine);
        vectorPtr += m_inputVectorSize;
        metaPtr = (lineEnd < metaEnd) ? (lineEnd + 1) : lineEnd;
        globalIdx++;
    }

    std::string algoTypeStr = SPTAG::Helper::Convert::ConvertToString(m_algoType);
    std::string valueTypeStr = SPTAG::Helper::Convert::ConvertToString(m_valueType);

    for (auto& tenantEntry : tenantVectorRanges)
    {
        int tenantId = tenantEntry.first;
        std::vector<std::pair<const uint8_t*, size_t>>& vectorRanges = tenantEntry.second;
        if (vectorRanges.empty())
        {
            continue;
        }

        size_t totalVectorSize = vectorRanges.size() * m_inputVectorSize;
        uint8_t* tenantVectorBuffer = new uint8_t[totalVectorSize];
        uint8_t* out = tenantVectorBuffer;
        for (const auto& vec : vectorRanges)
        {
            memcpy(out, vec.first, vec.second);
            out += vec.second;
        }
        ByteArray tenantVectors(tenantVectorBuffer, totalVectorSize, true);

        std::string metaStr;
        for (size_t i = 0; i < tenantMetadataLines[tenantId].size(); ++i)
        {
            if (i > 0) metaStr.push_back('\n');
            metaStr += tenantMetadataLines[tenantId][i];
        }
        metaStr.push_back('\n');

        uint8_t* metaBuffer = new uint8_t[metaStr.size()];
        memcpy(metaBuffer, metaStr.data(), metaStr.size());
        ByteArray tenantMetadata(metaBuffer, metaStr.size(), true);

        auto tenantIndex = std::make_shared<AnnIndex>(algoTypeStr.c_str(), valueTypeStr.c_str(), m_dimension);
        bool buildOk = false;
        SizeType tenantVecCount = static_cast<SizeType>(vectorRanges.size());

        // Choose index type based on tenant size (hybrid strategy)
        TenantIndexType indexType = ChooseIndexType(tenantVecCount);
        m_tenantIndexTypes[tenantId] = indexType;

        if (indexType == TenantIndexType::SPANN)
        {
            std::string spannWorkDir = "/tmp/sptag_spann_tenant_" + std::to_string(tenantId);
            EnsureDir(spannWorkDir);
            tenantIndex->SetBuildParam("IndexDirectory", spannWorkDir.c_str(), "Base");
            tenantIndex->SetBuildParam("DistCalcMethod", "Cosine", "Base");
            tenantIndex->SetBuildParam("isExecute", "true", "SelectHead");
            tenantIndex->SetBuildParam("isExecute", "true", "BuildHead");
            tenantIndex->SetBuildParam("isExecute", "true", "BuildSSDIndex");
            tenantIndex->SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex");
            tenantIndex->SetBuildParam("Storage", m_storageBackend.c_str(), "BuildSSDIndex");

            // Scale DataCapacity and SSD file size to tenant size
            // Block pool uses 4KB pages; each vector with replication needs multiple blocks
            // Use generous multiplier to avoid "cannot expand beyond cap" errors
            int dataCapacity = std::max(tenantVecCount * 8, (SizeType)4096);
            tenantIndex->SetBuildParam("DataCapacity", std::to_string(dataCapacity).c_str(), "Base");
            tenantIndex->SetBuildParam("DataBlockSize", std::to_string(std::min(dataCapacity, 1024 * 1024)).c_str(), "Base");

            // Scale SSD file size: each posting can hold ~PostingVectorLimit(118) vectors
            // Each vector in posting: dim*sizeof(float) + metadata overhead ~= dim*4+64 bytes
            // With ReplicaCount=8, total data ~ N * replica * vec_bytes / page_size blocks
            int64_t estimatedBytes = (int64_t)tenantVecCount * (int64_t)(m_dimension * 4 + 64) * 10;
            int startFileSizeGB = std::max(1, (int)(estimatedBytes / (1024LL * 1024LL * 1024LL)) + 1);
            tenantIndex->SetBuildParam("StartFileSizeGB", std::to_string(startFileSizeGB).c_str(), "BuildSSDIndex");
            tenantIndex->SetBuildParam("MaxFileSizeGB", std::to_string(std::max(startFileSizeGB * 3, 10)).c_str(), "BuildSSDIndex");

            // Scale graph build parameters by tenant size to avoid fixed overhead on small tenants
            // TPTNumber: controls how many random partition trees for initial KNN graph
            // RefineIterations: controls graph refinement passes
            int tptNumber = 32;
            int refineIter = 2;
            if (tenantVecCount < 10000) {
                tptNumber = 8;
                refineIter = 2;
            } else if (tenantVecCount < 50000) {
                tptNumber = 8;
                refineIter = 2;
            } else if (tenantVecCount < 200000) {
                tptNumber = 16;
                refineIter = 2;
            } else if (tenantVecCount < 500000) {
                tptNumber = 16;
                refineIter = 2;
            }
            tenantIndex->SetBuildParam("TPTNumber", std::to_string(tptNumber).c_str(), "BuildHead");
            tenantIndex->SetBuildParam("RefineIterations", std::to_string(refineIter).c_str(), "BuildHead");

            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            m_tenantSpannWorkDirs[tenantId] = spannWorkDir;
            fprintf(stderr, "[INFO] Tenant %d: SPANN build (%d vectors)\n", tenantId, tenantVecCount);
        }
        else if (indexType == TenantIndexType::BKT)
        {
            // Medium tenant: build in-memory BKT index
            tenantIndex = std::make_shared<AnnIndex>("BKT", valueTypeStr.c_str(), m_dimension);
            tenantIndex->SetBuildParam("DistCalcMethod", "Cosine", "Index");
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            fprintf(stderr, "[INFO] Tenant %d: BKT build (%d vectors)\n", tenantId, tenantVecCount);
        }
        else // BRUTEFORCE
        {
            // Small tenant: build trivial BKT index (effectively brute force at this scale)
            tenantIndex = std::make_shared<AnnIndex>("BKT", valueTypeStr.c_str(), m_dimension);
            tenantIndex->SetBuildParam("DistCalcMethod", "Cosine", "Index");
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            fprintf(stderr, "[INFO] Tenant %d: BruteForce build (%d vectors)\n", tenantId, tenantVecCount);
        }

        if (!buildOk)
        {
            return false;
        }

        m_tenantVectorCounts[tenantId] = static_cast<int>(vectorRanges.size());

        // For SPANN: save the index to its work dir right away, then release the
        // AnnIndex object.  This closes the SSD file descriptor and frees the
        // HeadIndex memory, preventing fd exhaustion when building many tenants.
        if (indexType == TenantIndexType::SPANN)
        {
            std::string workDir = m_tenantSpannWorkDirs[tenantId];
            tenantIndex->Save(workDir.c_str());
            fprintf(stderr, "[INFO] Tenant %d: built & released (%d vectors, dir=%s)\n",
                tenantId, (int)vectorRanges.size(), workDir.c_str());
            tenantIndex.reset();
            continue;
        }

        m_tenantIndices[tenantId] = tenantIndex;
    }

    // For SPANN tenants: compute posting offsets and record head counts
    if (m_algoType == SPTAG::IndexAlgoType::SPANN)
    {
        m_tenantPostingOffsets.clear();
        m_tenantHeadCounts.clear();
        m_totalPostingCount = 0;

        // Iterate all tenants (not just loaded ones — released tenants have work dirs)
        for (const auto& kv : m_tenantVectorCounts)
        {
            int tenantId = kv.first;
            auto typeIt = m_tenantIndexTypes.find(tenantId);
            if (typeIt == m_tenantIndexTypes.end() || typeIt->second != TenantIndexType::SPANN)
            {
                // Non-SPANN tenants don't have postings in the shared SSD
                m_tenantPostingOffsets[tenantId] = -1;
                m_tenantHeadCounts[tenantId] = 0;
                continue;
            }
            // Get head count from SPTAGHeadVectorIDs.bin file size
            // Each entry is sizeof(uint64_t) = 8 bytes
            std::string headIDFile = m_tenantSpannWorkDirs[tenantId] + "/SPTAGHeadVectorIDs.bin";
            int headCount = 0;

            if (fileexists(headIDFile.c_str()))
            {
                int64_t fsize = filesize(headIDFile.c_str());
                headCount = static_cast<int>(fsize / sizeof(uint64_t));
            }

            if (headCount <= 0)
            {
                // Fallback: read vector file
                std::string headVecFile = m_tenantSpannWorkDirs[tenantId] + "/SPTAGHeadVectors.bin";
                if (fileexists(headVecFile.c_str()))
                {
                    int64_t fsize = filesize(headVecFile.c_str());
                    headCount = static_cast<int>(fsize / (m_dimension * SPTAG::GetValueTypeSize(m_valueType)));
                }
            }

            if (headCount <= 0)
            {
                fprintf(stderr, "[ERROR] Cannot determine head count for tenant %d\n", tenantId);
                return false;
            }

            m_tenantPostingOffsets[tenantId] = m_totalPostingCount;
            m_tenantHeadCounts[tenantId] = headCount;
            m_totalPostingCount += headCount;
            fprintf(stderr, "[INFO] Tenant %d: headCount=%d, postingOffset=%d\n",
                tenantId, headCount, m_tenantPostingOffsets[tenantId]);
        }

        fprintf(stderr, "[INFO] Total posting count across %d tenants: %d\n",
            (int)m_tenantVectorCounts.size(), m_totalPostingCount);
    }

    return !m_tenantVectorCounts.empty();
}

std::shared_ptr<QueryResult> TenantIndexManager::Search(ByteArray p_queryVector, int p_tenantId, int p_resultNum)
{
    if (!EnsureTenantLoaded(p_tenantId))
    {
        return nullptr;
    }

    // Get index under shared lock (concurrent reads safe)
    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;  // shared_ptr copy under lock, search outside lock
    }

    return indexPtr->Search(p_queryVector, p_resultNum);
}

std::shared_ptr<QueryResult> TenantIndexManager::BatchSearch(ByteArray p_queryVectors, int p_vectorNum,
                                                              int p_tenantId, int p_resultNum)
{
    if (!EnsureTenantLoaded(p_tenantId))
    {
        return nullptr;
    }

    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;
    }
    return indexPtr->BatchSearch(p_queryVectors, p_vectorNum, p_resultNum, false);
}

void TenantIndexManager::GetTenantIds(int* p_tenants, int* p_count) const
{
    int idx = 0;
    for (const auto& [tenantId, _] : m_tenantVectorCounts)
    {
        p_tenants[idx++] = tenantId;
    }
    *p_count = (int)m_tenantVectorCounts.size();
}

int TenantIndexManager::GetTenantCount() const
{
    return (int)m_tenantVectorCounts.size();
}

int TenantIndexManager::GetTenantVectorCount(int p_tenantId) const
{
    auto it = m_tenantVectorCounts.find(p_tenantId);
    if (it != m_tenantVectorCounts.end())
    {
        return it->second;
    }
    return 0;  // Tenant not found
}

bool TenantIndexManager::SaveAll(const char* p_baseDir)
{
    m_baseStoragePath = std::string(p_baseDir);
    if (!EnsureDir(m_baseStoragePath))
    {
        return false;
    }

    // For SPANN: copy shared SSD infrastructure once
    // For other algos: save each tenant's data sequentially without directory overhead
    
    if (!SaveUnifiedStorage(p_baseDir))
    {
        return false;
    }

    // Write manifest for all tenants
    std::string manifestPath = m_baseStoragePath + "/manifest.txt";
    FILE* manifestFile = fopen(manifestPath.c_str(), "w");
    if (!manifestFile)
    {
        return false;
    }

    fprintf(manifestFile, "dimension %d\n", static_cast<int>(m_dimension));
    fprintf(manifestFile, "algorithm %s\n", m_algoType == SPTAG::IndexAlgoType::SPANN ? "SPANN" : 
            (m_algoType == SPTAG::IndexAlgoType::BKT ? "BKT" : "KDT"));
    fprintf(manifestFile, "unified_storage 1\n");
    fprintf(manifestFile, "total_postings %d\n", m_totalPostingCount);
    
    for (const auto& kv : m_tenantVectorCounts)
    {
        int tenantId = kv.first;
        int count = kv.second;
        int postingOffset = 0;
        int headCount = 0;
        auto offIt = m_tenantPostingOffsets.find(tenantId);
        if (offIt != m_tenantPostingOffsets.end()) postingOffset = offIt->second;
        auto hcIt = m_tenantHeadCounts.find(tenantId);
        if (hcIt != m_tenantHeadCounts.end()) headCount = hcIt->second;
        int typeInt = 0;
        auto typeIt = m_tenantIndexTypes.find(tenantId);
        if (typeIt != m_tenantIndexTypes.end()) typeInt = static_cast<int>(typeIt->second);
        // Format: tenant <id> <vecCount> <postingOffset> <headCount> <indexType>
        fprintf(manifestFile, "tenant %d %d %d %d %d\n", tenantId, count, postingOffset, headCount, typeInt);
    }

    // Save string tenant ID ↔ internal ID mapping
    {
        std::lock_guard<std::mutex> lock(m_tenantIdMutex);
        for (const auto& kv : m_tenantStrToInt)
        {
            // Format: tenant_mapping <internalId> <stringId>
            fprintf(manifestFile, "tenant_mapping %d %s\n", kv.second, kv.first.c_str());
        }
    }

    fclose(manifestFile);

    return true;
}

bool TenantIndexManager::LoadAll(const char* p_baseDir)
{
    m_tenantIndices.clear();
    m_tenantVectorCounts.clear();
    m_tenantIndexPaths.clear();
    m_tenantSpannWorkDirs.clear();
    
    
    

    // Clear tenant ID mapping
    {
        std::lock_guard<std::mutex> lock(m_tenantIdMutex);
        m_tenantStrToInt.clear();
        m_tenantIntToStr.clear();
        m_nextInternalId = 0;
    }

    std::string baseDir(p_baseDir);
    m_baseStoragePath = baseDir;
    std::string manifestPath = baseDir + "/manifest.txt";
    std::ifstream in(manifestPath.c_str());
    if (!in)
    {
        return false;
    }

    // Read manifest
    std::string line;
    bool unifiedStorage = false;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "dimension")
        {
            int dim = 0;
            if (!(iss >> dim) || dim != m_dimension)
            {
                return false;
            }
        }
        else if (key == "unified_storage")
        {
            int val = 0;
            if (iss >> val)
            {
                unifiedStorage = (val != 0);
            }
        }
        else if (key == "total_postings")
        {
            int val = 0;
            if (iss >> val) m_totalPostingCount = val;
        }
        else if (key == "tenant")
        {
            int tenantId = 0;
            int count = 0;
            int postingOffset = 0;
            int headCount = 0;
            int typeInt = 0;
            if (!(iss >> tenantId >> count))
            {
                return false;
            }
            // Optional fields: postingOffset, headCount, indexType
            iss >> postingOffset >> headCount >> typeInt;
            m_tenantVectorCounts[tenantId] = count;
            m_tenantPostingOffsets[tenantId] = postingOffset;
            m_tenantHeadCounts[tenantId] = headCount;
            m_tenantIndexTypes[tenantId] = static_cast<TenantIndexType>(typeInt);
        }
        else if (key == "tenant_mapping")
        {
            int internalId = 0;
            std::string strId;
            if (iss >> internalId >> strId)
            {
                std::lock_guard<std::mutex> lock(m_tenantIdMutex);
                m_tenantStrToInt[strId] = internalId;
                m_tenantIntToStr[internalId] = strId;
                if (internalId >= m_nextInternalId)
                    m_nextInternalId = internalId + 1;
            }
        }
    }
    in.close();

    // Load tenant indices based on storage type
    if (unifiedStorage)
    {
        return LoadUnifiedStorage(p_baseDir);
    }
    else
    {
        // Legacy: load from tenant_XX directories for backward compatibility
        for (const auto& kv : m_tenantVectorCounts)
        {
            int tenantId = kv.first;
            m_tenantSpannWorkDirs[tenantId] = baseDir + "/tenant_" + std::to_string(tenantId) + "/index";
        }
        return true;
    }
}

bool TenantIndexManager::SaveUnifiedStorage(const char* p_baseDir)
{
    std::string baseDir(p_baseDir);

    // Save tenants that are still in memory
    for (const auto& kv : m_tenantIndices)
    {
        int tenantId = kv.first;
        std::string dstTenantDir = baseDir + "/tenant_" + std::to_string(tenantId);
        if (!EnsureDir(dstTenantDir))
            return false;

        auto typeIt = m_tenantIndexTypes.find(tenantId);
        TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;

        if (indexType == TenantIndexType::SPANN)
        {
            if (!kv.second->Save(dstTenantDir.c_str()))
            {
                fprintf(stderr, "[ERROR] Failed to save SPANN index for tenant %d\n", tenantId);
                return false;
            }
            fprintf(stderr, "[INFO] Tenant %d: saved full SPANN index\n", tenantId);
        }
        else
        {
            std::string indexPath = dstTenantDir + "/index";
            if (!kv.second->Save(indexPath.c_str()))
            {
                fprintf(stderr, "[ERROR] Failed to save BKT/BF index for tenant %d\n", tenantId);
                return false;
            }
            fprintf(stderr, "[INFO] Tenant %d: saved BKT/BF index\n", tenantId);
        }
    }

    // Copy tenants that were already saved-and-released during build
    // (they exist in m_tenantSpannWorkDirs but not in m_tenantIndices)
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantIndices.count(tenantId)) continue;  // Already saved above

        std::string srcDir = kv.second;
        std::string dstDir = baseDir + "/tenant_" + std::to_string(tenantId);

        if (srcDir == dstDir) {
            // Already in the right place (saved directly to output dir)
            fprintf(stderr, "[INFO] Tenant %d: already saved in place\n", tenantId);
            continue;
        }

        if (!EnsureDir(dstDir)) return false;
        if (!CopyDirRecursive(srcDir, dstDir))
        {
            fprintf(stderr, "[ERROR] Failed to copy tenant %d from %s to %s\n", tenantId, srcDir.c_str(), dstDir.c_str());
            return false;
        }
        // Update work dir to point to final location
        m_tenantSpannWorkDirs[tenantId] = dstDir;
        fprintf(stderr, "[INFO] Tenant %d: copied from build dir\n", tenantId);
    }

    int totalSaved = (int)m_tenantIndices.size();
    for (const auto& kv : m_tenantSpannWorkDirs)
        if (!m_tenantIndices.count(kv.first)) totalSaved++;
    fprintf(stderr, "[INFO] Unified storage saved: %d tenants (%d SPANN)\n",
        totalSaved,
        (int)std::count_if(m_tenantIndexTypes.begin(), m_tenantIndexTypes.end(),
            [](const auto& kv) { return kv.second == TenantIndexType::SPANN; }));

    return true;
}

bool TenantIndexManager::LoadUnifiedStorage(const char* p_baseDir)
{
    std::string baseDir(p_baseDir);
    m_sharedSpannWorkDir = baseDir + "/shared_ssd";

    for (const auto& kv : m_tenantVectorCounts)
    {
        int tenantId = kv.first;
        auto typeIt = m_tenantIndexTypes.find(tenantId);
        TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;

        std::string tenantDir = baseDir + "/tenant_" + std::to_string(tenantId);

        if (indexType == TenantIndexType::SPANN)
        {
            m_tenantSpannWorkDirs[tenantId] = tenantDir;
        }
        else
        {
            // BKT / BruteForce: store index path for lazy loading
            m_tenantIndexPaths[tenantId] = tenantDir + "/index";
        }
    }

    return true;
}

void TenantIndexManager::SetBuildParam(const char* p_name, const char* p_value, const char* p_section)
{
    for (auto& tenantEntry : m_tenantIndices)
    {
        tenantEntry.second->SetBuildParam(p_name, p_value, p_section);
    }
}

void TenantIndexManager::SetSearchParam(const char* p_name, const char* p_value, const char* p_section)
{
    for (auto& tenantEntry : m_tenantIndices)
    {
        tenantEntry.second->SetSearchParam(p_name, p_value, p_section);
    }
}

bool TenantIndexManager::EnsureTenantLoaded(int p_tenantId)
{
    // Fast path: shared lock check (hot cache)
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        if (m_tenantIndices.count(p_tenantId))
        {
            auto it = m_lruMap.find(p_tenantId);
            if (it != m_lruMap.end())
                m_lruList.splice(m_lruList.end(), m_lruList, it->second);
            return true;
        }
    }

    // Slow path: exclusive lock
    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    // Double-check hot cache
    if (m_tenantIndices.count(p_tenantId))
    {
        auto it = m_lruMap.find(p_tenantId);
        if (it != m_lruMap.end())
            m_lruList.splice(m_lruList.end(), m_lruList, it->second);
        return true;
    }

    // Estimate HeadIndex size
    uint64_t estimatedBytes = 0;
    auto vcIt = m_tenantVectorCounts.find(p_tenantId);
    if (vcIt != m_tenantVectorCounts.end())
        estimatedBytes = static_cast<uint64_t>(vcIt->second) * 128;
    else
        estimatedBytes = 1024 * 1024;

    // Soft-evict LRU hot tenants until we have room
    if (m_headIndexCacheLimitBytes > 0)
    {
        while (m_loadedHeadIndexBytes + estimatedBytes > m_headIndexCacheLimitBytes
               && !m_lruList.empty())
        {
            int evictId = m_lruList.front();
            if (evictId == p_tenantId) break;
            UnloadTenantLocked(evictId);  // With shared AIO pool: ~1ms (just close fd)
        }
    }

    // Full load from disk
    auto typeIt = m_tenantIndexTypes.find(p_tenantId);
    TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;
    std::string loadPath;
    if (indexType == TenantIndexType::SPANN)
    {
        auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (workIt == m_tenantSpannWorkDirs.end()) return false;
        loadPath = workIt->second;
    }
    else
    {
        auto pathIt = m_tenantIndexPaths.find(p_tenantId);
        if (pathIt == m_tenantIndexPaths.end()) return false;
        loadPath = pathIt->second;
    }

    // Check SQ8 cache: if we have compressed head vectors, dequantize them
    // to a temp file to avoid reading the large vectors.bin from disk.
    auto sq8It = m_sq8Cache.find(p_tenantId);
    bool usedSQ8 = false;
    std::string tempVectorsPath;
    if (sq8It != m_sq8Cache.end() && indexType == TenantIndexType::SPANN)
    {
        tempVectorsPath = loadPath + "/HeadIndex/vectors.bin.sq8tmp";
        auto& sq8 = sq8It->second;
        std::vector<float> decompressed = sq8->Decompress();

        // Write decompressed vectors to temp file
        FILE* f = fopen(tempVectorsPath.c_str(), "wb");
        if (f)
        {
            // vectors.bin format: 4-byte num_vectors, 4-byte dimension, then float data
            int32_t nv = sq8->num_vectors;
            int32_t dm = sq8->dimension;
            fwrite(&nv, sizeof(int32_t), 1, f);
            fwrite(&dm, sizeof(int32_t), 1, f);
            fwrite(decompressed.data(), sizeof(float), nv * dm, f);
            fclose(f);

            // Rename: original vectors.bin → vectors.bin.orig, temp → vectors.bin
            std::string origPath = loadPath + "/HeadIndex/vectors.bin";
            std::string backupPath = loadPath + "/HeadIndex/vectors.bin.orig";
            rename(origPath.c_str(), backupPath.c_str());
            rename(tempVectorsPath.c_str(), origPath.c_str());
            usedSQ8 = true;
        }
        m_sq8Cache.erase(sq8It);  // Free SQ8 buffer
    }

    AnnIndex loadedIndex = AnnIndex::Load(loadPath.c_str());

    // Restore original vectors.bin if we used SQ8
    if (usedSQ8)
    {
        std::string origPath = loadPath + "/HeadIndex/vectors.bin";
        std::string backupPath = loadPath + "/HeadIndex/vectors.bin.orig";
        remove(origPath.c_str());
        rename(backupPath.c_str(), origPath.c_str());
    }

    if (!loadedIndex.ReadyToServe())
    {
        fprintf(stderr, "[ERROR] Failed to load tenant %d from %s\n", p_tenantId, loadPath.c_str());
        return false;
    }

    auto indexPtr = std::make_shared<AnnIndex>(loadedIndex);
    m_tenantIndices[p_tenantId] = indexPtr;
    m_loadedHeadIndexBytes += estimatedBytes;
    m_lruList.push_back(p_tenantId);
    m_lruMap[p_tenantId] = std::prev(m_lruList.end());
    return true;
}

void TenantIndexManager::InitCache()
{
    SPTAG::Cache::HeadIndexCache::Config cfg;
    cfg.capacity_bytes = m_headIndexCacheLimitBytes;
    cfg.ttl = std::chrono::seconds(600);
    cfg.load_timeout = std::chrono::milliseconds(30000);
    m_headCache = std::make_unique<SPTAG::Cache::HeadIndexCache>(cfg);
}

void TenantIndexManager::SetHeadIndexCacheLimit(uint64_t p_bytesLimit)
{
    m_headIndexCacheLimitBytes = p_bytesLimit;
    fprintf(stderr, "[INFO] HeadIndex cache limit set to %lu bytes (%.1f MB)\n",
            (unsigned long)p_bytesLimit, p_bytesLimit / (1024.0 * 1024.0));
}

uint64_t TenantIndexManager::GetHeadIndexCacheUsage() const
{
    std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
    return m_loadedHeadIndexBytes;
}

bool TenantIndexManager::UnloadTenant(int p_tenantId)
{
    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    return UnloadTenantLocked(p_tenantId);
}

bool TenantIndexManager::UnloadTenantLocked(int p_tenantId)
{
    // Must be called under exclusive lock (m_tenantIndicesMutex)
    auto it = m_tenantIndices.find(p_tenantId);
    if (it == m_tenantIndices.end()) return false;

    // Estimate size being freed
    uint64_t freedBytes = 0;
    auto vcIt = m_tenantVectorCounts.find(p_tenantId);
    if (vcIt != m_tenantVectorCounts.end())
        freedBytes = static_cast<uint64_t>(vcIt->second) * 128;
    else
        freedBytes = 1024 * 1024;

    // With SharedAIOPool: destruction only does close(fd) + free memory (~1ms).
    // AIO contexts are shared and never destroyed.

    // Before destroying: compress head vectors to SQ8 for fast reload.
    // Read vectors directly from HeadIndex/vectors.bin (known format: int32 rows, int32 cols, float32 data).
    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt != m_tenantSpannWorkDirs.end())
    {
        std::string vecPath = wdIt->second + "/HeadIndex/vectors.bin";
        FILE* vf = fopen(vecPath.c_str(), "rb");
        if (vf)
        {
            int32_t rows = 0, cols = 0;
            fread(&rows, sizeof(int32_t), 1, vf);
            fread(&cols, sizeof(int32_t), 1, vf);
            if (rows > 0 && cols > 0 && cols == static_cast<int32_t>(m_dimension))
            {
                std::vector<float> vecs(rows * cols);
                fread(vecs.data(), sizeof(float), rows * cols, vf);
                auto sq8 = SPTAG::Cache::SQ8CompressedVectors::Compress(vecs.data(), rows, cols);
                m_sq8Cache[p_tenantId] = sq8;
            }
            fclose(vf);
        }
    }

    it->second.reset();
    m_tenantIndices.erase(it);

    // Update cache accounting
    if (m_loadedHeadIndexBytes >= freedBytes)
        m_loadedHeadIndexBytes -= freedBytes;
    else
        m_loadedHeadIndexBytes = 0;

    // Remove from LRU
    auto lruIt = m_lruMap.find(p_tenantId);
    if (lruIt != m_lruMap.end())
    {
        m_lruList.erase(lruIt->second);
        m_lruMap.erase(lruIt);
    }

    // Drop OS page cache for this tenant's HeadIndex files.
    // This ensures next load hits real disk IO, not page cache.
    if (m_dropPageCacheOnEvict)
    {
        std::string hiDir;
        auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (wdIt != m_tenantSpannWorkDirs.end())
            hiDir = wdIt->second + "/HeadIndex";
        if (!hiDir.empty())
        {
            DIR* dir = opendir(hiDir.c_str());
            if (dir) {
                struct dirent* ent;
                while ((ent = readdir(dir)) != nullptr) {
                    if (ent->d_name[0] == '.') continue;
                    std::string fp = hiDir + "/" + ent->d_name;
                    int fd = open(fp.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        struct stat st;
                        fstat(fd, &st);
                        posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
                        close(fd);
                    }
                }
                closedir(dir);
            }
        }
    }

    return true;
}

void TenantIndexManager::TouchLRU(int p_tenantId)
{
    // No-op: S3-FIFO handles promotion internally via freq counter
}

void TenantIndexManager::EvictIfNeeded()
{
    // No-op: HeadIndexCache handles eviction internally
}

bool TenantIndexManager::EnsureTenantCached(int p_tenantId)
{
    return EnsureTenantLoaded(p_tenantId);
}

TenantIndexType TenantIndexManager::ChooseIndexType(int vectorCount) const
{
    // All tenants use SPANN: HeadIndex in memory, postings on SSD
    (void)vectorCount;
    return TenantIndexType::SPANN;
}

// --- String tenant ID mapping ---

int TenantIndexManager::RegisterTenantId(const char* p_tenantStr)
{
    if (p_tenantStr == nullptr) return -1;
    std::string key(p_tenantStr);
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantStrToInt.find(key);
    if (it != m_tenantStrToInt.end())
    {
        return it->second;
    }
    int id = m_nextInternalId++;
    m_tenantStrToInt[key] = id;
    m_tenantIntToStr[id] = key;
    return id;
}

int TenantIndexManager::GetInternalTenantId(const char* p_tenantStr) const
{
    if (p_tenantStr == nullptr) return -1;
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantStrToInt.find(std::string(p_tenantStr));
    return (it != m_tenantStrToInt.end()) ? it->second : -1;
}

const char* TenantIndexManager::GetTenantIdStr(int p_internalId) const
{
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantIntToStr.find(p_internalId);
    return (it != m_tenantIntToStr.end()) ? it->second.c_str() : nullptr;
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchByTenant(
    ByteArray p_queryVector, const char* p_tenantStr, int p_resultNum)
{
    int internalId = GetInternalTenantId(p_tenantStr);
    if (internalId < 0) return nullptr;
    return Search(p_queryVector, internalId, p_resultNum);
}

bool TenantIndexManager::InitSharedFileIO()
{
    if (m_sharedFileIO) return true;
    if (m_sharedSpannWorkDir.empty()) return false;

    SPTAG::SPANN::Options sharedOpts;
    sharedOpts.m_indexDirectory = m_sharedSpannWorkDir;
    sharedOpts.m_ssdMappingFile = "ssdmapping";
    sharedOpts.m_storage = SPTAG::Storage::FILEIO;
    sharedOpts.m_datasetCapacity = std::max(m_totalPostingCount * 8, 4096);
    sharedOpts.m_datasetRowsInBlock = std::min(sharedOpts.m_datasetCapacity, 1024 * 1024);

    // Estimate file size from total posting count
    int64_t totalEstBytes = (int64_t)m_totalPostingCount * (int64_t)(m_dimension * 4 + 64) * 10;
    int startGB = std::max(1, (int)(totalEstBytes / (1024LL * 1024LL * 1024LL)) + 1);
    sharedOpts.m_startFileSize = startGB;
    sharedOpts.m_maxFileSize = std::max(startGB * 3, 10);
    sharedOpts.m_spdkBatchSize = 64;

    m_sharedFileIO = std::make_shared<SPTAG::SPANN::FileIO>(sharedOpts);
    if (!m_sharedFileIO->Available())
    {
        fprintf(stderr, "[ERROR] Failed to initialize shared FileIO at %s\n", m_sharedSpannWorkDir.c_str());
        m_sharedFileIO.reset();
        return false;
    }
    fprintf(stderr, "[INFO] Shared FileIO initialized: %s (%d total postings)\n",
        m_sharedSpannWorkDir.c_str(), m_totalPostingCount);
    return true;
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchSharedSPANN(
    ByteArray p_queryVector, int p_tenantId, int p_resultNum)
{
    // Ensure shared FileIO is initialized
    if (!InitSharedFileIO()) return nullptr;

    // Ensure tenant's HeadIndex is loaded
    auto headIt = m_tenantHeadIndices.find(p_tenantId);
    if (headIt == m_tenantHeadIndices.end())
    {
        // Load HeadIndex from saved directory
        auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (workIt == m_tenantSpannWorkDirs.end()) return nullptr;

        std::string headIdxDir = workIt->second + "/HeadIndex";
        std::shared_ptr<SPTAG::VectorIndex> headIdx;
        if (SPTAG::VectorIndex::LoadIndex(headIdxDir, headIdx) != SPTAG::ErrorCode::Success || !headIdx)
        {
            fprintf(stderr, "[ERROR] Failed to load HeadIndex for tenant %d from %s\n",
                p_tenantId, headIdxDir.c_str());
            return nullptr;
        }
        headIdx->SetReady(true);
        m_tenantHeadIndices[p_tenantId] = headIdx;
        headIt = m_tenantHeadIndices.find(p_tenantId);
    }

    auto& headIdx = headIt->second;
    int postingOffset = 0;
    auto offIt = m_tenantPostingOffsets.find(p_tenantId);
    if (offIt != m_tenantPostingOffsets.end()) postingOffset = offIt->second;

    // Step 1: Search HeadIndex for top candidate postings
    int internalResultNum = std::min(64, headIdx->GetNumSamples());
    SPTAG::QueryResult headQuery(p_queryVector.Data(), internalResultNum, false);
    headIdx->SearchIndex(headQuery);

    // Collect valid posting IDs (apply offset)
    std::vector<int> postingIDs;
    for (int i = 0; i < internalResultNum; i++)
    {
        auto res = headQuery.GetResult(i);
        if (res->VID == -1) break;
        postingIDs.push_back(res->VID + postingOffset);
    }

    if (postingIDs.empty())
    {
        fprintf(stderr, "[DEBUG] SearchSharedSPANN tenant %d: HeadIndex returned 0 results\n", p_tenantId);
        return nullptr;
    }
    fprintf(stderr, "[DEBUG] SearchSharedSPANN tenant %d: %d posting IDs (first=%d, offset=%d)\n",
        p_tenantId, (int)postingIDs.size(), postingIDs[0], postingOffset);

    // Step 2: Read postings from shared FileIO and compute distances
    const float* queryVec = reinterpret_cast<const float*>(p_queryVector.Data());
    size_t vectorSize = m_dimension * sizeof(float);
    int metaDataSize = sizeof(SPTAG::SizeType) + sizeof(uint8_t);  // 4 + 1 = 5
    int vectorInfoSize = (int)vectorSize + metaDataSize;

    // Priority queue for top-K results
    struct Result { int vid; float dist; };
    auto cmp = [](const Result& a, const Result& b) { return a.dist < b.dist; };
    std::priority_queue<Result, std::vector<Result>, decltype(cmp)> topK(cmp);

    int totalVecsSeen = 0;
    int getFailures = 0;
    for (int globalPostingId : postingIDs)
    {
        std::string postingData;
        auto ret = m_sharedFileIO->Get(globalPostingId, &postingData,
            SPTAG::MaxTimeout, nullptr);
        if (ret != SPTAG::ErrorCode::Success || postingData.empty()) { getFailures++; continue; }

        int numVectors = (int)postingData.size() / vectorInfoSize;
        totalVecsSeen += numVectors;
        const char* ptr = postingData.data();

        for (int j = 0; j < numVectors; j++)
        {
            SPTAG::SizeType vid;
            memcpy(&vid, ptr, sizeof(vid));
            const float* vec = reinterpret_cast<const float*>(ptr + metaDataSize);

            // Cosine distance (SPTAG uses negative inner product for cosine on normalized vectors)
            float dist = 0;
            for (int d = 0; d < m_dimension; d++)
            {
                dist -= queryVec[d] * vec[d];
            }

            if ((int)topK.size() < p_resultNum)
            {
                topK.push({vid, dist});
            }
            else if (dist < topK.top().dist)
            {
                topK.pop();
                topK.push({vid, dist});
            }

            ptr += vectorInfoSize;
        }
    }

    if (totalVecsSeen == 0)
    {
        fprintf(stderr, "[DEBUG] SearchSharedSPANN tenant %d: 0 vectors read from %d postings (%d Get failures)\n",
            p_tenantId, (int)postingIDs.size(), getFailures);
    }

    // Step 3: Build QueryResult
    auto result = std::make_shared<SPTAG::QueryResult>(p_queryVector.Data(), p_resultNum, false);
    int count = (int)topK.size();
    std::vector<Result> sorted;
    while (!topK.empty()) { sorted.push_back(topK.top()); topK.pop(); }
    std::reverse(sorted.begin(), sorted.end());

    for (int i = 0; i < p_resultNum; i++)
    {
        if (i < count)
        {
            result->SetResult(i, sorted[i].vid, sorted[i].dist);
        }
        else
        {
            result->SetResult(i, -1, SPTAG::MaxDist);
        }
    }

    return result;
}
