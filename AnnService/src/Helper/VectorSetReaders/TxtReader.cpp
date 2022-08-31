// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReaders/TxtReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/CommonHelper.h"

#include <omp.h>

using namespace SPTAG;
using namespace SPTAG::Helper;

TxtVectorReader::TxtVectorReader(std::shared_ptr<ReaderOptions> p_options)
    : VectorSetReader(p_options),
    m_subTaskBlocksize(0)
{
    omp_set_num_threads(m_options->m_threadNum);

    std::string tempFolder("tempfolder");
    if (!direxists(tempFolder.c_str()))
    {
        mkdir(tempFolder.c_str());
    }

    tempFolder += FolderSep;
    std::srand(clock());
    std::string randstr = std::to_string(std::rand());
    m_vectorOutput = tempFolder + "vectorset.bin." + randstr;
    m_metadataConentOutput = tempFolder + "metadata.bin." + randstr;
    m_metadataIndexOutput = tempFolder + "metadataindex.bin." + randstr;
}


TxtVectorReader::~TxtVectorReader()
{
    if (fileexists(m_vectorOutput.c_str()))
    {
        remove(m_vectorOutput.c_str());
    }

    if (fileexists(m_metadataIndexOutput.c_str()))
    {
        remove(m_metadataIndexOutput.c_str());
    }

    if (fileexists(m_metadataConentOutput.c_str()))
    {
        remove(m_metadataConentOutput.c_str());
    }
}


ErrorCode
TxtVectorReader::LoadFile(const std::string& p_filePaths)
{
    const auto& files = GetFileSizes(p_filePaths);
    std::vector<std::function<ErrorCode()>> subWorks;
    subWorks.reserve(files.size() * m_options->m_threadNum);

    m_subTaskCount = 0;
    for (const auto& fileInfo : files)
    {
        if (fileInfo.second == (std::numeric_limits<std::size_t>::max)())
        {
            LOG(Helper::LogLevel::LL_Error, "File %s not exists or can't access.\n", fileInfo.first.c_str());
            return ErrorCode::FailedOpenFile;
        }

        std::uint32_t fileTaskCount = 0;
        std::size_t blockSize = m_subTaskBlocksize;
        if (0 == blockSize)
        {
            fileTaskCount = m_options->m_threadNum;
            if(fileTaskCount == 0) fileTaskCount = 1;
            blockSize = (fileInfo.second + fileTaskCount - 1) / fileTaskCount;
        }
        else
        {
            fileTaskCount = static_cast<std::uint32_t>((fileInfo.second + blockSize - 1) / blockSize);
        }

        for (std::uint32_t i = 0; i < fileTaskCount; ++i)
        {
            subWorks.emplace_back(std::bind(&TxtVectorReader::LoadFileInternal,
                                            this,
                                            fileInfo.first,
                                            m_subTaskCount++,
                                            i,
                                            blockSize));
        }
    }

    m_totalRecordCount = 0;
    m_totalRecordVectorBytes = 0;
    m_subTaskRecordCount.clear();
    m_subTaskRecordCount.resize(m_subTaskCount, 0);

    m_waitSignal.Reset(m_subTaskCount);

#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < (int64_t)subWorks.size(); i++)
    {
        ErrorCode code = subWorks[i]();
        if (ErrorCode::Success != code)
        {
            throw std::runtime_error("LoadFileInternal failed");
        }

    }

    m_waitSignal.Wait();

    return MergeData();
}


std::shared_ptr<VectorSet>
TxtVectorReader::GetVectorSet(SizeType start, SizeType end) const
{
    auto ptr = f_createIO();
    if (ptr == nullptr || !ptr->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read file %s.\n", m_vectorOutput.c_str());
        throw std::runtime_error("Failed to read vectorset file");
    }

    SizeType row;
    DimensionType col;
    if (ptr->ReadBinary(sizeof(SizeType), (char*)&row) != sizeof(SizeType)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        throw std::runtime_error("Failed to read vectorset file");
    }
    if (ptr->ReadBinary(sizeof(DimensionType), (char*)&col) != sizeof(DimensionType)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
        throw std::runtime_error("Failed to read vectorset file");
    }

    if (start > row) start = row;
    if (end < 0 || end > row) end = row;
    std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(m_options->m_inputValueType)) * (end - start) * col;
    ByteArray vectorSet;
    if (totalRecordVectorBytes > 0) {
        vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
        char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());
        std::uint64_t offset = ((std::uint64_t)GetValueTypeSize(m_options->m_inputValueType)) * start * col + +sizeof(SizeType) + sizeof(DimensionType);
        if (ptr->ReadBinary(totalRecordVectorBytes, vecBuf, offset) != totalRecordVectorBytes) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
            throw std::runtime_error("Failed to read vectorset file");
        }
    }
    return std::shared_ptr<VectorSet>(new BasicVectorSet(vectorSet,
        m_options->m_inputValueType,
        col,
        end - start));
}


std::shared_ptr<MetadataSet>
TxtVectorReader::GetMetadataSet() const
{
    if (fileexists(m_metadataIndexOutput.c_str()) && fileexists(m_metadataConentOutput.c_str()))
        return std::shared_ptr<MetadataSet>(new FileMetadataSet(m_metadataConentOutput, m_metadataIndexOutput));
    return nullptr;
}


ErrorCode
TxtVectorReader::LoadFileInternal(const std::string& p_filePath,
                                std::uint32_t p_subTaskID,
                                std::uint32_t p_fileBlockID,
                                std::size_t p_fileBlockSize)
{
    std::uint64_t lineBufferSize = 1 << 16;
    std::unique_ptr<char[]> currentLine(new char[lineBufferSize]);

    SizeType recordCount = 0;
    std::uint64_t metaOffset = 0;
    std::size_t totalRead = 0;
    std::streamoff startpos = p_fileBlockID * p_fileBlockSize;

    std::shared_ptr<Helper::DiskIO> input = f_createIO(), output = f_createIO(), meta = f_createIO(), metaIndex = f_createIO();
    if (input == nullptr || !input->Initialize(p_filePath.c_str(), std::ios::in | std::ios::binary))
    {
        LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n",p_filePath.c_str());
        return ErrorCode::FailedOpenFile;
    }

    LOG(Helper::LogLevel::LL_Info, "Begin Subtask: %u, start offset position: %lld\n", p_subTaskID, startpos);

    std::string subFileSuffix("_");
    subFileSuffix += std::to_string(p_subTaskID);
    subFileSuffix += ".tmp";

    if (output == nullptr || !output->Initialize((m_vectorOutput + subFileSuffix).c_str(), std::ios::binary | std::ios::out) ||
        meta == nullptr || !meta->Initialize((m_metadataConentOutput + subFileSuffix).c_str(), std::ios::binary | std::ios::out) ||
        metaIndex == nullptr || !metaIndex->Initialize((m_metadataIndexOutput + subFileSuffix).c_str(), std::ios::binary | std::ios::out))
    {
        LOG(Helper::LogLevel::LL_Error, "Unable to create files: %s %s %s\n", (m_vectorOutput + subFileSuffix).c_str(), (m_metadataConentOutput + subFileSuffix).c_str(), (m_metadataIndexOutput + subFileSuffix).c_str());
        return ErrorCode::FailedCreateFile;
    }

    if (p_fileBlockID != 0)
    {
        totalRead += input->ReadString(lineBufferSize, currentLine, '\n', startpos);
    }

    std::size_t vectorByteSize = GetValueTypeSize(m_options->m_inputValueType) * m_options->m_dimension;
    std::unique_ptr<std::uint8_t[]> vector;
    vector.reset(new std::uint8_t[vectorByteSize]);

    while (totalRead <= p_fileBlockSize)
    {
        std::uint64_t lineLength = input->ReadString(lineBufferSize, currentLine);
        if (lineLength == 0) break;
        totalRead += lineLength;

        std::size_t tabIndex = lineLength - 1;
        while (tabIndex > 0 && currentLine[tabIndex] != '\t')
        {
            --tabIndex;
        }

        if (0 == tabIndex && currentLine[tabIndex] != '\t')
        {
            LOG(Helper::LogLevel::LL_Error, "Subtask: %u cannot parsing line:%s\n", p_subTaskID, currentLine.get());
            return ErrorCode::FailedParseValue;
        }

        bool parseSuccess = false;
        switch (m_options->m_inputValueType)
        {
#define DefineVectorValueType(Name, Type) \
        case VectorValueType::Name: \
            parseSuccess = TranslateVector(currentLine.get() + tabIndex + 1, reinterpret_cast<Type*>(vector.get())); \
            break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default:
            parseSuccess = false;
            break;
        }

        if (!parseSuccess)
        {
            LOG(Helper::LogLevel::LL_Error, "Subtask: %u cannot parsing vector:%s\n", p_subTaskID, currentLine.get());
            return ErrorCode::FailedParseValue;
        }

        ++recordCount;
        if (output->WriteBinary(vectorByteSize, (char*)vector.get()) != vectorByteSize ||
            meta->WriteBinary(tabIndex, currentLine.get()) != tabIndex ||
            metaIndex->WriteBinary(sizeof(metaOffset), (const char*)&metaOffset) != sizeof(metaOffset)) {
            LOG(Helper::LogLevel::LL_Error, "Subtask: %u cannot write line:%s\n", p_subTaskID, currentLine.get());
            return ErrorCode::DiskIOFail;
        }
        metaOffset += tabIndex;
    }
    if (metaIndex->WriteBinary(sizeof(metaOffset), (const char*)&metaOffset) != sizeof(metaOffset)) {
        LOG(Helper::LogLevel::LL_Error, "Subtask: %u cannot write final offset!\n", p_subTaskID);
        return ErrorCode::DiskIOFail;
    }

    m_totalRecordCount += recordCount;
    m_subTaskRecordCount[p_subTaskID] = recordCount;
    m_totalRecordVectorBytes += recordCount * vectorByteSize;

    m_waitSignal.FinishOne();
    return ErrorCode::Success;
}


ErrorCode
TxtVectorReader::MergeData()
{
    const std::size_t bufferSize = 1 << 30;
    const std::size_t bufferSizeTrim64 = (bufferSize / sizeof(std::uint64_t)) * sizeof(std::uint64_t);

    std::shared_ptr<Helper::DiskIO> input = f_createIO(), output = f_createIO(), meta = f_createIO(), metaIndex = f_createIO();

    if (output == nullptr || !output->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::out) ||
        meta == nullptr || !meta->Initialize(m_metadataConentOutput.c_str(), std::ios::binary | std::ios::out) ||
        metaIndex == nullptr || !metaIndex->Initialize(m_metadataIndexOutput.c_str(), std::ios::binary | std::ios::out))
    {
        LOG(Helper::LogLevel::LL_Error, "Unable to create files: %s %s %s\n", m_vectorOutput.c_str(), m_metadataConentOutput.c_str(), m_metadataIndexOutput.c_str());
        return ErrorCode::FailedCreateFile;
    }

    std::unique_ptr<char[]> bufferHolder(new char[bufferSize]);
    char* buf = bufferHolder.get();

    SizeType totalRecordCount = m_totalRecordCount;
    if (output->WriteBinary(sizeof(totalRecordCount), (char*)(&totalRecordCount)) != sizeof(totalRecordCount)) {
        LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_vectorOutput.c_str());
        return ErrorCode::DiskIOFail;
    }
    if (output->WriteBinary(sizeof(m_options->m_dimension), (char*)&(m_options->m_dimension)) != sizeof(m_options->m_dimension)) {
        LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_vectorOutput.c_str());
        return ErrorCode::DiskIOFail;
    }

    for (std::uint32_t i = 0; i < m_subTaskCount; ++i)
    {
        std::string file = m_vectorOutput;
        file += "_";
        file += std::to_string(i);
        file += ".tmp";

        if (input == nullptr || !input->Initialize(file.c_str(), std::ios::binary | std::ios::in))
        {
            LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", file.c_str());
            return ErrorCode::FailedOpenFile;
        }

        std::uint64_t readSize;
        while ((readSize = input->ReadBinary(bufferSize, bufferHolder.get()))) {
            if (output->WriteBinary(readSize, bufferHolder.get()) != readSize) {
                LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_vectorOutput.c_str());
                return ErrorCode::DiskIOFail;
            }
        }
        input->ShutDown();
        remove(file.c_str());
    }

    for (std::uint32_t i = 0; i < m_subTaskCount; ++i)
    {
        std::string file = m_metadataConentOutput;
        file += "_";
        file += std::to_string(i);
        file += ".tmp";

        if (input == nullptr || !input->Initialize(file.c_str(), std::ios::binary | std::ios::in))
        {
            LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", file.c_str());
            return ErrorCode::FailedOpenFile;
        }

        std::uint64_t readSize;
        while ((readSize = input->ReadBinary(bufferSize, bufferHolder.get()))) {
            if (meta->WriteBinary(readSize, bufferHolder.get()) != readSize) {
                LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_metadataConentOutput.c_str());
                return ErrorCode::DiskIOFail;
            }
        }
        input->ShutDown();
        remove(file.c_str());
    }

    if (metaIndex->WriteBinary(sizeof(totalRecordCount), (char*)(&totalRecordCount)) != sizeof(totalRecordCount)) {
        LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_metadataIndexOutput.c_str());
        return ErrorCode::DiskIOFail;
    }

    std::uint64_t totalOffset = 0;
    for (std::uint32_t i = 0; i < m_subTaskCount; ++i)
    {
        std::string file = m_metadataIndexOutput;
        file += "_";
        file += std::to_string(i);
        file += ".tmp";

        if (input == nullptr || !input->Initialize(file.c_str(), std::ios::binary | std::ios::in))
        {
            LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", file.c_str());
            return ErrorCode::FailedOpenFile;
        }

        for (SizeType remains = m_subTaskRecordCount[i]; remains > 0;)
        {
            std::size_t readBytesCount = min(remains * sizeof(std::uint64_t), bufferSizeTrim64);
            if (input->ReadBinary(readBytesCount, buf) != readBytesCount) {
                LOG(Helper::LogLevel::LL_Error, "Unable to read file: %s\n", file.c_str());
                return ErrorCode::DiskIOFail;
            }
            std::uint64_t* offset = reinterpret_cast<std::uint64_t*>(buf);
            for (std::uint64_t i = 0; i < readBytesCount / sizeof(std::uint64_t); ++i)
            {
                offset[i] += totalOffset;
            }

            if (metaIndex->WriteBinary(readBytesCount, buf) != readBytesCount) {
                LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_metadataIndexOutput.c_str());
                return ErrorCode::DiskIOFail;
            }
            remains -= static_cast<SizeType>(readBytesCount / sizeof(std::uint64_t));
        }
        if (input->ReadBinary(sizeof(std::uint64_t), buf) != sizeof(std::uint64_t)) {
            LOG(Helper::LogLevel::LL_Error, "Unable to read file: %s\n", file.c_str());
            return ErrorCode::DiskIOFail;
        }
        totalOffset += *(reinterpret_cast<std::uint64_t*>(buf));

        input->ShutDown();
        remove(file.c_str());
    }

    if (metaIndex->WriteBinary(sizeof(totalOffset), (char*)&totalOffset) != sizeof(totalOffset)) {
        LOG(Helper::LogLevel::LL_Error, "Unable to write file: %s\n", m_metadataIndexOutput.c_str());
        return ErrorCode::DiskIOFail;
    }
    return ErrorCode::Success;
}


std::vector<TxtVectorReader::FileInfoPair>
TxtVectorReader::GetFileSizes(const std::string& p_filePaths)
{
    const auto& files = Helper::StrUtils::SplitString(p_filePaths, ",");
    std::vector<TxtVectorReader::FileInfoPair> res;
    res.reserve(files.size());

    for (const auto& filePath : files)
    {
        if (!fileexists(filePath.c_str()))
        {
            res.emplace_back(filePath, (std::numeric_limits<std::size_t>::max)());
            continue;
        }
#ifndef _MSC_VER
        struct stat stat_buf;
        stat(filePath.c_str(), &stat_buf);
#else
        struct _stat64 stat_buf;
        _stat64(filePath.c_str(), &stat_buf);
#endif
        std::size_t fileSize = stat_buf.st_size;
        res.emplace_back(filePath, static_cast<std::size_t>(fileSize));
    }

    return res;
}


