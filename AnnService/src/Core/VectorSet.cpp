// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/VectorSet.h"
#include "inttypes.h"
#include <fstream>
#include <memory>
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/CommonUtils.h"

using namespace SPTAG;

#pragma warning(disable:4996)  // 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

VectorSet::VectorSet()
{
}


VectorSet::~VectorSet()
{
}


BasicVectorSet::BasicVectorSet(const ByteArray& p_bytesArray,
                               VectorValueType p_valueType,
                               DimensionType p_dimension,
                               SizeType p_vectorCount)
    : m_data(p_bytesArray),
      m_valueType(p_valueType),
      m_dimension(p_dimension),
      m_vectorCount(p_vectorCount),
      m_perVectorDataSize(static_cast<SizeType>(p_dimension * GetValueTypeSize(p_valueType)))
{
}

void BasicVectorSet::readXvec(std::string p_filePath, VectorValueType p_valueType,
    DimensionType p_dimension, SizeType p_vectorCount) 
{
    if (p_filePath == "")
    {
        fprintf(stderr, "BasicVectorSet init error: lack filename(s) for xvec file.\n");
        exit(-1);
    }

    std::vector< std::ifstream* > files;
    std::string::size_type offset = 0, pos;
    while (offset < p_filePath.size()) {
        std::string file_name;
        pos = p_filePath.find(",", offset);
        if (pos != std::string::npos)
        {
            file_name = p_filePath.substr(offset, pos - offset);
        }
        else
        {
            file_name = p_filePath.substr(offset);
        }

        std::ifstream* cur_in = new std::ifstream(file_name.c_str(), std::ifstream::binary);
        if (!*cur_in)
        {
            fprintf(stderr, "Error: Failed to read input file: %s \n", file_name.c_str());
            exit(-1);
        }

        files.push_back(cur_in);

        if (pos != std::string::npos) {
            offset = pos + 1;
        }
        else {
            break;
        }
    }

    size_t vectorDataSize = GetValueTypeSize(p_valueType) * p_dimension;
    size_t totalRecordVectorBytes = static_cast<size_t>(vectorDataSize) * p_vectorCount;
    ByteArray l_data = ByteArray::Alloc(totalRecordVectorBytes);
    char* vecBuf = reinterpret_cast<char*>(l_data.Data());
    char* curAddr = vecBuf;

    DimensionType dim = p_dimension;

    // only read first readLine lines of a series files
    size_t readLine = 0;
    for (size_t i = 0; i < files.size(); i++)
    {
        while (readLine < p_vectorCount)
        {
            files[i]->read((char*)&dim, 4);
            if (files[i]->eof())
            {
                break;
            }
            if (dim != p_dimension) {
                fprintf(stderr, "Error: Xvec file %s has No.%zd vector whose dims are not as many as expected. Expected: %d, Fact: %d\n", p_filePath.c_str(), i, p_dimension, dim);
                exit(-1);
            }
            files[i]->read(curAddr, vectorDataSize);
            curAddr += vectorDataSize;
            readLine++;
        }

        delete files[i];
    }

    m_data = std::move(l_data);
    m_valueType = p_valueType;
    m_dimension = p_dimension;
    m_vectorCount = p_vectorCount;
    m_perVectorDataSize = vectorDataSize;
}

void BasicVectorSet::readDefault(std::string p_filePath, VectorValueType p_valueType)
{
    if (p_filePath == "")
    {
        fprintf(stderr, "BasicVectorSet init error: lack filename(s) for default file.\n");
        exit(-1);
    }
    std::vector< std::ifstream* > files;
    std::vector<SizeType> offsets = { 0 };
    std::vector<SizeType> rows;
    std::string::size_type offset = 0, pos;
    SizeType row = 0;
    DimensionType col = -1;
    while (offset < p_filePath.size()) {
        std::string file_name;
        pos = p_filePath.find(",", offset);
        if (pos != std::string::npos)
        {
            file_name = p_filePath.substr(offset, pos - offset);
        }
        else
        {
            file_name = p_filePath.substr(offset);
        }
        
        std::ifstream* cur_in = new std::ifstream(file_name.c_str(), std::ifstream::binary);
        if (!*cur_in)
        {
            fprintf(stderr, "Error: Failed to read input file: %s \n", file_name.c_str());
            exit(-1);
        }

        SizeType cur_row;
        DimensionType cur_col;
        cur_in->read((char*)&cur_row, 4);
        cur_in->read((char*)&cur_col, 4);

        auto err_and_exit = [](const std::string t) {
            fprintf(stderr, "Error: Number of %s can't be negative.\n", t.c_str());
            exit(-1);
        };

        if (cur_row < 0) {
            err_and_exit("row");
        }
        else if (cur_col < 0) {
            err_and_exit("column");
        }
        
        if (col == -1)
        {
            col = cur_col;
        }
        else if (col != cur_col) {
            fprintf(stderr, "Error: Input files don't have the same dimension. \n");
            exit(-1);
        }

        row += cur_row;
        files.push_back(cur_in);
        rows.push_back(cur_row);
        offsets.push_back(cur_row + offsets.back());

        if (pos != std::string::npos) { 
            offset = pos + 1; 
        }
        else {
            break;
        }
    }

    size_t vectorDataSize = GetValueTypeSize(p_valueType) * col;
    std::size_t ULLVectorDataSize = static_cast<std::size_t>(vectorDataSize);
    std::size_t totalRecordVectorBytes = ULLVectorDataSize * row;
    ByteArray l_data = ByteArray::Alloc(totalRecordVectorBytes);
    char* vecBuf = reinterpret_cast<char*>(l_data.Data());
    
    for (size_t i = 0; i < files.size(); i++)
    {
        files[i]->read(vecBuf + ULLVectorDataSize * offsets[i], ULLVectorDataSize * rows[i]);
        delete files[i];
    }

    m_data = std::move(l_data);
    m_valueType = p_valueType;
    m_dimension = col;
    m_vectorCount = row;
    m_perVectorDataSize = vectorDataSize;
}

void BasicVectorSet::readTxt(std::string p_filePath, VectorValueType p_valueType, DimensionType p_dimension, std::string p_delimiter) {
    std::shared_ptr<SPTAG::Helper::ReaderOptions> options = std::make_shared<SPTAG::Helper::ReaderOptions>(p_valueType, p_dimension, p_delimiter, 1);
    auto vectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile(p_filePath))
    {
        fprintf(stderr, "Failed to read input file.\n");
        exit(1);
    }

    std::size_t totalRecordVectorBytes = 
        static_cast<std::size_t>(vectorReader->GetVectorSet()->Count()) * vectorReader->GetVectorSet()->PerVectorDataSize();
    ByteArray l_data = std::move(ByteArray::Alloc(totalRecordVectorBytes));
    memcpy(reinterpret_cast<void *>(l_data.Data()), 
        vectorReader->GetVectorSet()->GetData(), 
        totalRecordVectorBytes);

    m_data = std::move(l_data);
    m_valueType = vectorReader->GetVectorSet()->GetValueType();
    m_dimension = vectorReader->GetVectorSet()->Dimension();
    m_vectorCount = vectorReader->GetVectorSet()->Count();
    m_perVectorDataSize = vectorReader->GetVectorSet()->PerVectorDataSize();
}

// copied from src/IndexBuilder/main.cpp
BasicVectorSet::BasicVectorSet(std::string p_filePath, VectorValueType p_valueType,
    DimensionType p_dimension, SizeType p_vectorCount, VectorFileType p_fileType, std::string p_delimiter, DistCalcMethod p_distCalcMethod)
{
    if (p_fileType == VectorFileType::XVEC)
    {
        readXvec(p_filePath, p_valueType, p_dimension, p_vectorCount);
    }
    else if (p_fileType == VectorFileType::DEFAULT)
    {
        readDefault(p_filePath, p_valueType);
    }
    else if (p_fileType == VectorFileType::TXT) 
    {
        readTxt(p_filePath, p_valueType, p_dimension, p_delimiter);
    }
    else
    {
        fprintf(stderr, "VectorFileType Unsupported.\n");
        exit(-1);
    }

    if (p_distCalcMethod == DistCalcMethod::Cosine) {
#pragma omp parallel for
        for (int64_t i = 0; i < m_vectorCount; i++)
        {
            int64_t offset = i * m_perVectorDataSize;
            switch (p_valueType)
            {
#define DefineVectorValueType(Name, Type) \
case SPTAG::VectorValueType::Name: \
SPTAG::COMMON::Utils::Normalize<Type>(reinterpret_cast<Type *>(m_data.Data() + offset), p_dimension, SPTAG::COMMON::Utils::GetBase<Type>()); \
break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
            default:
                break;
            }
        }
    }
}

BasicVectorSet::~BasicVectorSet()
{
}


VectorValueType
BasicVectorSet::GetValueType() const
{
    return m_valueType;
}


void*
BasicVectorSet::GetVector(SizeType p_vectorID) const
{
    if (p_vectorID < 0 || p_vectorID >= m_vectorCount)
    {
        return nullptr;
    }

    return reinterpret_cast<void*>(m_data.Data() + ((size_t)p_vectorID) * m_perVectorDataSize);
}


void*
BasicVectorSet::GetData() const
{
    return reinterpret_cast<void*>(m_data.Data());
}

DimensionType
BasicVectorSet::Dimension() const
{
    return m_dimension;
}


SizeType
BasicVectorSet::Count() const
{
    return m_vectorCount;
}


bool
BasicVectorSet::Available() const
{
    return m_data.Data() != nullptr;
}


ErrorCode 
BasicVectorSet::Save(const std::string& p_vectorFile) const
{
    FILE * fp = fopen(p_vectorFile.c_str(), "wb");
    if (fp == NULL) return ErrorCode::FailedOpenFile;

    fwrite(&m_vectorCount, sizeof(SizeType), 1, fp);
    fwrite(&m_dimension, sizeof(DimensionType), 1, fp);

    fwrite((const void*)(m_data.Data()), m_data.Length(), 1, fp);
    fclose(fp);
    return ErrorCode::Success;
}

SizeType BasicVectorSet::PerVectorDataSize() const {
    return (SizeType)m_perVectorDataSize;
}