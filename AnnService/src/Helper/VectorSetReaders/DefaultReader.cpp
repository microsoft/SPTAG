// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReaders/DefaultReader.h"
#include "inc/Helper/CommonHelper.h"

#include <fstream>

using namespace SPTAG;
using namespace SPTAG::Helper;

DefaultVectorReader::DefaultVectorReader(std::shared_ptr<ReaderOptions> p_options)
    : VectorSetReader(std::move(p_options))
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
DefaultVectorReader::GetVectorSet() const
{
    std::ifstream inputStream(m_vectorOutput, std::ifstream::binary);
    if (!inputStream.is_open()) {
        fprintf(stderr, "Failed to read file %s.\n", m_vectorOutput.c_str());
        exit(1);
    }

    SizeType row;
    DimensionType col;
    inputStream.read((char*)&row, sizeof(SizeType));
    inputStream.read((char*)&col, sizeof(DimensionType));
    std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(m_options->m_inputValueType)) * row * col;
    ByteArray vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
    char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());
    inputStream.read(vecBuf, totalRecordVectorBytes);
    inputStream.close();

    return std::shared_ptr<VectorSet>(new BasicVectorSet(vectorSet,
                                                         m_options->m_inputValueType,
                                                         col,
                                                         row));
}


std::shared_ptr<MetadataSet>
DefaultVectorReader::GetMetadataSet() const
{
    if (fileexists(m_metadataIndexOutput.c_str()) && fileexists(m_metadataConentOutput.c_str()))
        return std::shared_ptr<MetadataSet>(new FileMetadataSet(m_metadataConentOutput, m_metadataIndexOutput));
    return nullptr;
}
