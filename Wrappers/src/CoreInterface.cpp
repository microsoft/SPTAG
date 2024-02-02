// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CoreInterface.h"
#include "inc/Helper/StringConvert.h"


AnnIndex::AnnIndex(DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::BKT),
      m_inputValueType(SPTAG::VectorValueType::Float),
      m_dimension(p_dimension)
{
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}


AnnIndex::AnnIndex(const char* p_algoType, const char* p_valueType, DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::Undefined),
      m_inputValueType(SPTAG::VectorValueType::Undefined),
      m_dimension(p_dimension)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_inputValueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}


AnnIndex::AnnIndex(const std::shared_ptr<SPTAG::VectorIndex>& p_index)
    : m_algoType(p_index->GetIndexAlgoType()),
      m_inputValueType(p_index->GetVectorValueType()),
      m_dimension(p_index->GetFeatureDim()),
      m_index(p_index)
{
    m_inputVectorSize = p_index->m_pQuantizer ? p_index->m_pQuantizer->GetNumSubvectors() : SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}


AnnIndex::~AnnIndex()
{
}



bool
AnnIndex::BuildSPANN(bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index) return false;

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

bool
AnnIndex::BuildSPANNWithMetaData(ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index) return false;

    std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n')) return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(p_meta, ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex) m_index->BuildMetaMapping(false);

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}


bool
AnnIndex::Build(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num, (SPTAG::DimensionType)m_dimension, p_normalized));
}


bool
AnnIndex::BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized)
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
    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(p_data,
        vectorType,
        static_cast<SPTAG::DimensionType>(vectorSize),
        static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n')) return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(p_meta, ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize));
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(vectors, meta, p_withMetaIndex, p_normalized));
}


void
AnnIndex::SetBuildParam(const char* p_name, const char* p_value, const char* p_section)
{
    if (nullptr == m_index) 
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || 
            SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return;    
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);

    }
    m_index->SetParameter(p_name, p_value, p_section);
}


void
AnnIndex::SetSearchParam(const char* p_name, const char* p_value, const char* p_section)
{
    if (nullptr != m_index) m_index->SetParameter(p_name, p_value, p_section);
}


std::shared_ptr<ResultIterator> 
AnnIndex::GetIterator(ByteArray p_target)
{
    if (nullptr != m_index) return m_index->GetIterator(p_target.Data());
    return nullptr;
}


bool
AnnIndex::LoadQuantizer(const char* p_quantizerFile)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType ||
            SPTAG::VectorValueType::Undefined == m_inputValueType)
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


void
AnnIndex::SetQuantizerADC(bool p_adc)
{
    if (nullptr != m_index) return m_index->SetQuantizerADC(p_adc);
}


ByteArray 
AnnIndex::QuantizeVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr) {
        size_t outsize = m_index->GetQuantizer()->GetNumSubvectors() * (size_t)p_num;
        std::uint8_t* outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success != m_index->QuantizeVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false))) return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}


ByteArray 
AnnIndex::ReconstructVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr) {
        size_t outsize = m_index->GetQuantizer()->ReconstructSize() * (size_t)p_num;
        std::uint8_t* outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success != m_index->ReconstructVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false))) return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}


std::shared_ptr<QueryResult>
AnnIndex::Search(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, false);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult>
AnnIndex::SearchWithMetaData(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult>
AnnIndex::BatchSearch(ByteArray p_data, int p_vectorNum, int p_resultNum, bool p_withMetaData)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, p_withMetaData);
    if (nullptr != m_index)
    {
        m_index->SearchIndex(p_data.Data(), p_vectorNum, p_resultNum, p_withMetaData, results->GetResults());
    }
    return std::move(results);
}

bool
AnnIndex::ReadyToServe() const
{
    return m_index != nullptr;
}


void
AnnIndex::UpdateIndex()
{
    m_index->UpdateIndex();
}


bool
AnnIndex::Save(const char* p_savefile) const
{
    return SPTAG::ErrorCode::Success == m_index->SaveIndex(p_savefile);
}


bool 
AnnIndex::Add(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->AddIndex(p_data.Data(), (SPTAG::SizeType)p_num, (SPTAG::DimensionType)m_dimension, nullptr, false, p_normalized));
}


bool
AnnIndex::AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(p_data,
        m_inputValueType,
        static_cast<SPTAG::DimensionType>(m_dimension),
        static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n')) return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(p_meta, ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num));
    return (SPTAG::ErrorCode::Success == m_index->AddIndex(vectors, meta, p_withMetaIndex, p_normalized));
}


bool
AnnIndex::Delete(ByteArray p_data, SizeType p_num)
{
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_data.Data(), (SPTAG::SizeType)p_num));
}


bool
AnnIndex::DeleteByMetaData(ByteArray p_meta)
{
    if (nullptr == m_index) return false;
    
    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_meta));
}


uint64_t
AnnIndex::CalculateBufferSize()
{
    if (nullptr == m_index) return 0;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    uint64_t total = sizeof(int) + sizeof(uint64_t) * buffersize->size();
    for (uint64_t bs : *buffersize) {
        total += bs;
    }
    return total;
}


ByteArray
AnnIndex::Dump(ByteArray p_blobs)
{
    if (nullptr == m_index) return ByteArray::c_empty;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    std::uint8_t *ptr = p_blobs.Data(), *pdata = ptr + sizeof(int) + sizeof(uint64_t) * buffersize->size();
    *((int*)ptr) = (int)(buffersize->size());
    ptr += sizeof(int);

    std::vector<SPTAG::ByteArray> indexBlobs;
    for (size_t i = 0; i < buffersize->size(); i++) {
        *((uint64_t*)ptr) = buffersize->at(i);
        ptr += sizeof(uint64_t);
        indexBlobs.push_back(SPTAG::ByteArray(pdata, buffersize->at(i), false));
        pdata += buffersize->at(i);
    }

    std::string config;
    if (SPTAG::ErrorCode::Success != m_index->SaveIndex(config, indexBlobs))
    {
        return ByteArray::c_empty;
    }
    std::uint8_t* newdata = new std::uint8_t[config.size()];
    memcpy(newdata, config.c_str(), config.size());
    return ByteArray(newdata, config.size(), false);
}


AnnIndex
AnnIndex::LoadFromDump(ByteArray p_config, ByteArray p_blobs)
{
    if (p_config.Length() == 0) return AnnIndex(0);

    std::uint8_t* ptr = p_blobs.Data();
    int streamNum = *((int*)ptr);
    ptr += sizeof(int);
    std::uint8_t* pdata = ptr + sizeof(uint64_t) * streamNum;

    std::vector<SPTAG::ByteArray> p_indexBlobs;
    for (int i = 0; i < streamNum; i++)
    {
        std::uint64_t streamSize = *((uint64_t*)ptr);
        ptr += sizeof(uint64_t);
        p_indexBlobs.push_back(SPTAG::ByteArray((std::uint8_t*)pdata, streamSize, false));
        pdata += streamSize;
    }

    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    std::string config((char*)p_config.Data(), p_config.Length());
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(config, p_indexBlobs, vecIndex) || nullptr == vecIndex)
    {
        return AnnIndex(0);
    }
    return AnnIndex(vecIndex);
}


AnnIndex
AnnIndex::Load(const char* p_loaderFile)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    auto ret = SPTAG::VectorIndex::LoadIndex(p_loaderFile, vecIndex);
    if (SPTAG::ErrorCode::Success != ret || nullptr == vecIndex)
    {
        return AnnIndex(0);
    }

    return AnnIndex(vecIndex);
}


AnnIndex
AnnIndex::Merge(const char* p_indexFilePath1, const char* p_indexFilePath2)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex, addIndex;
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath1, vecIndex) ||
        SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath2, addIndex) ||
        SPTAG::ErrorCode::Success != vecIndex->MergeIndex(addIndex.get(), std::atoi(vecIndex->GetParameter("NumberOfThreads").c_str()), nullptr))
        return AnnIndex(0);

    return AnnIndex(vecIndex);
}
