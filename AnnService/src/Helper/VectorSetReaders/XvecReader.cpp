// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReaders/XvecReader.h"
#include "inc/Helper/CommonHelper.h"

#include <time.h>

using namespace SPTAG;
using namespace SPTAG::Helper;

XvecVectorReader::XvecVectorReader(std::shared_ptr<ReaderOptions> p_options)
    : VectorSetReader(std::move(p_options))
{
    std::string tempFolder("tempfolder");
    if (!direxists(tempFolder.c_str()))
    {
        mkdir(tempFolder.c_str());
    }
    std::srand(clock());
    m_vectorOutput = tempFolder + FolderSep + "vectorset.bin." + std::to_string(std::rand());
}


XvecVectorReader::~XvecVectorReader()
{
    if (fileexists(m_vectorOutput.c_str()))
    {
        remove(m_vectorOutput.c_str());
    }
}


ErrorCode
XvecVectorReader::LoadFile(const std::string& p_filePaths)
{
    const auto& files = Helper::StrUtils::SplitString(p_filePaths, ",");
    auto fp = f_createIO();
    if (fp == nullptr || !fp->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::out)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to write file: %s \n", m_vectorOutput.c_str());
        return ErrorCode::FailedCreateFile;
    }
    SizeType vectorCount = 0;
    IOBINARY(fp, WriteBinary, sizeof(vectorCount), (char*)&vectorCount);
    IOBINARY(fp, WriteBinary, sizeof(m_options->m_dimension), (char*)&(m_options->m_dimension));
    
    size_t vectorDataSize = GetValueTypeSize(m_options->m_inputValueType) * m_options->m_dimension;
    std::unique_ptr<char[]> buffer(new char[vectorDataSize]);
    for (std::string file : files)
    {
        auto ptr = f_createIO();
        if (ptr == nullptr || !ptr->Initialize(file.c_str(), std::ios::binary | std::ios::in)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read file: %s \n", file.c_str());
            return ErrorCode::FailedOpenFile;
        }
        while (true)
        {
            DimensionType dim;
            if (ptr->ReadBinary(sizeof(DimensionType), (char*)&dim) == 0) break;

            if (dim != m_options->m_dimension) {
                LOG(Helper::LogLevel::LL_Error, "Xvec file %s has No.%d vector whose dims are not as many as expected. Expected: %d, Fact: %d\n", file.c_str(), vectorCount, m_options->m_dimension, dim);
                return ErrorCode::DimensionSizeMismatch;
            }
            IOBINARY(ptr, ReadBinary, vectorDataSize, buffer.get());
            IOBINARY(fp, WriteBinary, vectorDataSize, buffer.get());
            vectorCount++;
        }
    }
    IOBINARY(fp, WriteBinary, sizeof(vectorCount), (char*)&vectorCount, 0);
    return ErrorCode::Success;
}


std::shared_ptr<VectorSet>
XvecVectorReader::GetVectorSet() const
{
    auto ptr = f_createIO();
    if (ptr == nullptr || !ptr->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read file %s.\n", m_vectorOutput.c_str());
        exit(1);
    }

    SizeType row;
    DimensionType col;
    if (ptr->ReadBinary(sizeof(SizeType), (char*)&row) != sizeof(SizeType)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        exit(1);
    }
    if (ptr->ReadBinary(sizeof(DimensionType), (char*)&col) != sizeof(DimensionType)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        exit(1);
    }

    std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(m_options->m_inputValueType)) * row * col;
    ByteArray vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
    char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());

    if (ptr->ReadBinary(totalRecordVectorBytes, vecBuf) != totalRecordVectorBytes) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        exit(1);
    }

    return std::shared_ptr<VectorSet>(new BasicVectorSet(vectorSet,
        m_options->m_inputValueType,
        col,
        row));
}


std::shared_ptr<MetadataSet>
XvecVectorReader::GetMetadataSet() const
{
    return nullptr;
}
