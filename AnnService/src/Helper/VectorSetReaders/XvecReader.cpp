// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReaders/XvecReader.h"
#include "inc/Helper/CommonHelper.h"

#include <fstream>

using namespace SPTAG;
using namespace SPTAG::Helper;

XvecReader::XvecReader(std::shared_ptr<ReaderOptions> p_options)
    : VectorSetReader(std::move(p_options))
{
    std::string tempFolder("tempfolder");
    if (!direxists(tempFolder.c_str()))
    {
        mkdir(tempFolder.c_str());
    }
    m_vectorOutput = tempFolder + FolderSep + "vectorset.bin";
}


XvecReader::~XvecReader()
{
    if (fileexists(m_vectorOutput.c_str()))
    {
        remove(m_vectorOutput.c_str());
    }
}


ErrorCode
XvecReader::LoadFile(const std::string& p_filePaths)
{
    const auto& files = Helper::StrUtils::SplitString(p_filePaths, ",");
    std::ofstream outputStream(m_vectorOutput, std::ofstream::binary);
    if (!outputStream.is_open()) {
        fprintf(stderr, "Failed to write file: %s \n", m_vectorOutput.c_str());
        exit(1);
    }
    SizeType vectorCount = 0;
    outputStream.write(reinterpret_cast<char*>(&vectorCount), sizeof(vectorCount));
    outputStream.write(reinterpret_cast<char*>(&(m_options->m_dimension)), sizeof(m_options->m_dimension));
    
    size_t vectorDataSize = GetValueTypeSize(m_options->m_inputValueType) * m_options->m_dimension;
    std::unique_ptr<char[]> buffer(new char[vectorDataSize]);
    for (std::string file : files)
    {
        std::ifstream fin(file, std::ifstream::binary);
        if (!fin.is_open()) {
            fprintf(stderr, "Failed to read file: %s \n", file.c_str());
            exit(-1);
        }
        while (true)
        {
            DimensionType dim;
            fin.read((char*)&dim, sizeof(DimensionType));
            if (fin.eof()) break;

            if (dim != m_options->m_dimension) {
                fprintf(stderr, "Xvec file %s has No.%d vector whose dims are not as many as expected. Expected: %d, Fact: %d\n", file.c_str(), vectorCount, m_options->m_dimension, dim);
                exit(-1);
            }
            fin.read(buffer.get(), vectorDataSize);
            outputStream.write(buffer.get(), vectorDataSize);
            vectorCount++;
        }
        fin.close();
    }

    outputStream.seekp(0, std::ios_base::beg);
    outputStream.write(reinterpret_cast<char*>(&vectorCount), sizeof(vectorCount));
    outputStream.close();
}


std::shared_ptr<VectorSet>
XvecReader::GetVectorSet() const
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
XvecReader::GetMetadataSet() const
{
    return nullptr;
}
