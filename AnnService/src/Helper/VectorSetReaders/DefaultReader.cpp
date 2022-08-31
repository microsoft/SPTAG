// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReaders/DefaultReader.h"
#include "inc/Helper/CommonHelper.h"

using namespace SPTAG;
using namespace SPTAG::Helper;

DefaultVectorReader::DefaultVectorReader(std::shared_ptr<ReaderOptions> p_options)
    : VectorSetReader(p_options)
{
    m_vectorOutput = "";
    m_metadataConentOutput = "";
    m_metadataIndexOutput = "";
}


DefaultVectorReader::~DefaultVectorReader()
{
}


ErrorCode
DefaultVectorReader::LoadFile(const std::string& p_filePaths)
{
    const auto& files = SPTAG::Helper::StrUtils::SplitString(p_filePaths, ",");
    m_vectorOutput = files[0];
    if (files.size() >= 3) {
        m_metadataConentOutput = files[1];
        m_metadataIndexOutput = files[2];
    }
    return ErrorCode::Success;
}


std::shared_ptr<VectorSet>
DefaultVectorReader::GetVectorSet(SizeType start, SizeType end) const
{
    auto ptr = f_createIO();
    if (ptr == nullptr || !ptr->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read file %s.\n", m_vectorOutput.c_str());
        throw std::runtime_error("Failed read file");
    }

    SizeType row;
    DimensionType col;
    if (ptr->ReadBinary(sizeof(SizeType), (char*)&row) != sizeof(SizeType)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        throw std::runtime_error("Failed read file");
    }
    if (ptr->ReadBinary(sizeof(DimensionType), (char*)&col) != sizeof(DimensionType)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        throw std::runtime_error("Failed read file");
    }
    
    if (start > row) start = row;
    if (end < 0 || end > row) end = row;
    std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(m_options->m_inputValueType)) * (end - start) * col;
    ByteArray vectorSet;
    if (totalRecordVectorBytes > 0) {
        vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
        char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());
        std::uint64_t offset = ((std::uint64_t)GetValueTypeSize(m_options->m_inputValueType)) * start * col + sizeof(SizeType) + sizeof(DimensionType);
        if (ptr->ReadBinary(totalRecordVectorBytes, vecBuf, offset) != totalRecordVectorBytes) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
            throw std::runtime_error("Failed read file");
        }
    }

    LOG(Helper::LogLevel::LL_Info, "Load Vector(%d,%d)\n", end - start, col);
    return std::make_shared<BasicVectorSet>(vectorSet,
                                            m_options->m_inputValueType,
                                            col,
                                            end - start);
}


std::shared_ptr<MetadataSet>
DefaultVectorReader::GetMetadataSet() const
{
    if (fileexists(m_metadataIndexOutput.c_str()) && fileexists(m_metadataConentOutput.c_str()))
        return std::shared_ptr<MetadataSet>(new FileMetadataSet(m_metadataConentOutput, m_metadataIndexOutput));
    return nullptr;
}
