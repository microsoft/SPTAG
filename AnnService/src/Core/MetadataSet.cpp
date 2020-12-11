// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/MetadataSet.h"

#include <string.h>
#include <shared_mutex>

using namespace SPTAG;

ErrorCode
MetadataSet::RefineMetadata(std::vector<SizeType>& indices, std::shared_ptr<MetadataSet>& p_newMetadata,
    std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize) const
{
    p_newMetadata.reset(new MemMetadataSet(p_blockSize, p_capacity, p_metaSize));
    for (SizeType& t : indices) {
        p_newMetadata->Add(GetMetadata(t));
    }
    return ErrorCode::Success;
}

ErrorCode
MetadataSet::RefineMetadata(std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut) const
{
    SizeType R = (SizeType)indices.size();
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(SizeType), (char*)&R);
    std::uint64_t offset = 0;
    for (SizeType i = 0; i < R; i++) {
        IOBINARY(p_metaIndexOut, WriteBinary, sizeof(std::uint64_t), (char*)&offset);
        ByteArray meta = GetMetadata(indices[i]);
        offset += meta.Length();
    }
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(std::uint64_t), (char*)&offset);

    for (SizeType i = 0; i < R; i++) {
        ByteArray meta = GetMetadata(indices[i]);
        IOBINARY(p_metaOut, WriteBinary, sizeof(uint8_t) * meta.Length(), (char*)meta.Data());
    }
    LOG(Helper::LogLevel::LL_Info, "Save MetaIndex(%d) Meta(%llu)\n", R, offset);
    return ErrorCode::Success;
}


ErrorCode 
MetadataSet::RefineMetadata(std::vector<SizeType>& indices, const std::string& p_metaFile, const std::string& p_metaindexFile) const
{
    {
        std::shared_ptr<Helper::DiskPriorityIO> ptrMeta = f_createIO(), ptrMetaIndex = f_createIO();
        if (ptrMeta == nullptr || ptrMetaIndex == nullptr || !ptrMeta->Initialize((p_metaFile + "_tmp").c_str(), std::ios::binary | std::ios::out) || !ptrMetaIndex->Initialize((p_metaindexFile + "_tmp").c_str(), std::ios::binary | std::ios::out))
        return ErrorCode::FailedCreateFile;

        ErrorCode ret = RefineMetadata(indices, ptrMeta, ptrMetaIndex);
        if (ret != ErrorCode::Success) return ret;
    }

    if (fileexists(p_metaFile.c_str())) std::remove(p_metaFile.c_str());
    if (fileexists(p_metaindexFile.c_str())) std::remove(p_metaindexFile.c_str());
    std::rename((p_metaFile + "_tmp").c_str(), p_metaFile.c_str());
    std::rename((p_metaindexFile + "_tmp").c_str(), p_metaindexFile.c_str());
    return ErrorCode::Success;
}


void
MetadataSet::AddBatch(MetadataSet& data)
{
    for (SizeType i = 0; i < data.Count(); i++)
    {
        Add(data.GetMetadata(i));
    }
}


MetadataSet::MetadataSet()
{
}


MetadataSet:: ~MetadataSet()
{
}


FileMetadataSet::FileMetadataSet(const std::string& p_metafile, const std::string& p_metaindexfile, std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize)
    : m_metaFile(p_metafile),
      m_metaindexFile(p_metaindexfile)
{
    m_fp = f_createIO();
    auto fpidx = f_createIO();
    if (m_fp == nullptr || fpidx == nullptr || !m_fp->Initialize(p_metafile.c_str(), std::ios::binary | std::ios::in) || !fpidx->Initialize(p_metaindexfile.c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open meta files %s or %s!\n", p_metafile.c_str(), p_metaindexfile.c_str());
        exit(1);
    }

    if (fpidx->ReadBinary(sizeof(m_count), (char*)&m_count) != sizeof(m_count)) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read FileMetadataSet!\n");
        exit(1);
    }

    m_offsets.Initialize(p_blockSize, p_capacity);
    if (!m_offsets.Load(fpidx, m_count + 1)) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read FileMetadataSet!\n");
        exit(1);
    }
    m_newdata.Initialize(p_blockSize * p_metaSize, p_capacity * p_metaSize);
    LOG(Helper::LogLevel::LL_Info, "Load MetaIndex(%d) Meta(%llu)\n", m_count, m_offsets[m_count]);
}


FileMetadataSet::~FileMetadataSet()
{
}


ByteArray
FileMetadataSet::GetMetadata(SizeType p_vectorID) const
{
    std::uint64_t startoff = m_offsets[p_vectorID];
    std::uint64_t bytes = m_offsets[p_vectorID + 1] - startoff;
    if (p_vectorID < m_count) {
        ByteArray b = ByteArray::Alloc(bytes);
        m_fp->ReadBinary(bytes, (char*)b.Data(), startoff);
        return b;
    }
    else {
        return m_newdata.At(startoff - m_offsets[m_count], bytes);
    }
}


ByteArray
FileMetadataSet::GetMetadataCopy(SizeType p_vectorID) const
{
    return GetMetadata(p_vectorID);
}


SizeType
FileMetadataSet::Count() const
{
    return static_cast<SizeType>(m_offsets.Size() - 1);
}


bool
FileMetadataSet::Available() const
{
    return m_fp != nullptr && m_offsets.Size() > 1;
}


std::pair<std::uint64_t, std::uint64_t> 
FileMetadataSet::BufferSize() const
{
    return std::make_pair(m_offsets[m_offsets.Size() - 1], 
        sizeof(SizeType) + sizeof(std::uint64_t) * m_offsets.Size());
}


void
FileMetadataSet::Add(const ByteArray& data)
{
    m_newdata.Append(data.Data(), data.Length());
    m_offsets.Append(m_offsets[m_offsets.Size() - 1] + data.Length());
}


ErrorCode
FileMetadataSet::SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut)
{
    SizeType count = Count();
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(SizeType), (char*)&count);
    if (!m_offsets.Save(p_metaIndexOut)) return ErrorCode::DiskIOFail;

    std::uint64_t bufsize = 1000000;
    char* buf = new char[bufsize];
    auto readsize = m_fp->ReadBinary(bufsize, buf, 0);
    while (readsize > 0) {
        IOBINARY(p_metaOut, WriteBinary, readsize, buf);
        readsize = m_fp->ReadBinary(bufsize, buf);
    }
    delete[] buf;
    
    if (m_newdata.Size() > 0 && !m_newdata.Save(p_metaOut)) return ErrorCode::DiskIOFail;
    LOG(Helper::LogLevel::LL_Info, "Save MetaIndex(%llu) Meta(%llu)\n", m_offsets.Size() - 1, m_offsets[m_offsets.Size() - 1]);
    return ErrorCode::Success;
}


ErrorCode
FileMetadataSet::SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile)
{
    {
        std::shared_ptr<Helper::DiskPriorityIO> metaOut = f_createIO(), metaIndexOut = f_createIO();
        if (metaOut == nullptr || metaIndexOut == nullptr || !metaOut->Initialize((p_metaFile + "_tmp").c_str(), std::ios::binary | std::ios::out) || !metaIndexOut->Initialize((p_metaindexFile + "_tmp").c_str(), std::ios::binary | std::ios::out))
            return ErrorCode::FailedCreateFile;

        ErrorCode ret = SaveMetadata(metaOut, metaIndexOut);
        if (ret != ErrorCode::Success) return ret;
    }

    m_fp->ShutDown();
    if (fileexists(p_metaFile.c_str())) std::remove(p_metaFile.c_str());
    if (fileexists(p_metaindexFile.c_str())) std::remove(p_metaindexFile.c_str());
    std::rename((p_metaFile + "_tmp").c_str(), p_metaFile.c_str());
    std::rename((p_metaindexFile + "_tmp").c_str(), p_metaindexFile.c_str());
    if (!m_fp->Initialize(p_metaFile.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
    m_count = Count();
    m_newdata.Clear();
    return ErrorCode::Success;
}


MemMetadataSet::MemMetadataSet(std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize): m_count(0), m_metadataHolder(ByteArray::c_empty)
{
    m_offsets.Initialize(p_blockSize, p_capacity);
    m_offsets.Append(0);
    m_newdata.Initialize(p_blockSize * p_metaSize, p_capacity * p_metaSize);
}


ErrorCode
MemMetadataSet::Init(std::shared_ptr<Helper::DiskPriorityIO> p_metain, std::shared_ptr<Helper::DiskPriorityIO> p_metaindexin,
    std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize)
{
    IOBINARY(p_metaindexin, ReadBinary, sizeof(m_count), (char*)&m_count);
    m_offsets.Initialize(p_blockSize, p_capacity);
    if (!m_offsets.Load(p_metaindexin, m_count + 1)) return ErrorCode::DiskIOFail;

    m_metadataHolder = ByteArray::Alloc(m_offsets[m_count]);
    IOBINARY(p_metain, ReadBinary, m_metadataHolder.Length(), (char *)m_metadataHolder.Data());

    m_newdata.Initialize(p_blockSize * p_metaSize, p_capacity * p_metaSize);
    LOG(Helper::LogLevel::LL_Info, "Load MetaIndex(%d) Meta(%llu)\n", m_count, m_offsets[m_count]);
    return ErrorCode::Success;
}


MemMetadataSet::MemMetadataSet(std::shared_ptr<Helper::DiskPriorityIO> p_metain, std::shared_ptr<Helper::DiskPriorityIO> p_metaindexin,
    std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize)
{
    if (Init(p_metain, p_metaindexin, p_blockSize, p_capacity, p_metaSize) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read MemMetadataSet!\n");
        exit(1);
    }
}

MemMetadataSet::MemMetadataSet(const std::string& p_metafile, const std::string& p_metaindexfile,
    std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize)
{
    std::shared_ptr<Helper::DiskPriorityIO> ptrMeta = f_createIO(), ptrMetaIndex = f_createIO();
    if (ptrMeta == nullptr || ptrMetaIndex == nullptr || !ptrMeta->Initialize(p_metafile.c_str(), std::ios::binary | std::ios::in) || !ptrMetaIndex->Initialize(p_metaindexfile.c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open meta files %s or %s!\n", p_metafile.c_str(),  p_metaindexfile.c_str());
        exit(1);
    }
    if (Init(ptrMeta, ptrMetaIndex, p_blockSize, p_capacity, p_metaSize) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read MemMetadataSet!\n");
        exit(1);
    }
}


MemMetadataSet::MemMetadataSet(ByteArray p_metadata, ByteArray p_offsets, SizeType p_count,
    std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize)
    : m_count(p_count), m_metadataHolder(std::move(p_metadata))
{
    m_offsets.Initialize(p_blockSize, p_capacity);
    m_offsets.Append((std::uint64_t*)(p_offsets.Data()), p_count + 1);
    m_newdata.Initialize(p_blockSize * p_metaSize, p_capacity * p_metaSize);
}


MemMetadataSet::~MemMetadataSet()
{
}


ByteArray
MemMetadataSet::GetMetadata(SizeType p_vectorID) const
{
    std::uint64_t startoff = m_offsets[p_vectorID];
    std::uint64_t bytes = m_offsets[p_vectorID + 1] - startoff;
    if (p_vectorID < m_count) {
        return ByteArray(m_metadataHolder.Data() + startoff, bytes, false);
    } else {
        return m_newdata.At(startoff - m_offsets[m_count], bytes);
    }
}


ByteArray
MemMetadataSet::GetMetadataCopy(SizeType p_vectorID) const
{
    std::uint64_t startoff = m_offsets[p_vectorID];
    std::uint64_t bytes = m_offsets[p_vectorID + 1] - startoff;
    if (p_vectorID < m_count) {
        ByteArray b = ByteArray::Alloc(bytes);
        memcpy(b.Data(), m_metadataHolder.Data() + startoff, bytes);
        return b;
    } else {
        return m_newdata.At(startoff - m_offsets[m_count], bytes);
    }
}


SizeType
MemMetadataSet::Count() const
{
    return static_cast<SizeType>(m_offsets.Size() - 1);
}


bool
MemMetadataSet::Available() const
{
    return m_offsets.Size() > 1;
}


std::pair<std::uint64_t, std::uint64_t>
MemMetadataSet::BufferSize() const
{
    return std::make_pair(m_offsets[m_offsets.Size() - 1],
        sizeof(SizeType) + sizeof(std::uint64_t) * m_offsets.Size());
}


void
MemMetadataSet::Add(const ByteArray& data)
{
    m_newdata.Append(data.Data(), data.Length());
    m_offsets.Append(m_offsets[m_offsets.Size() - 1]+ data.Length());
}


ErrorCode
MemMetadataSet::SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut)
{
    SizeType count = Count();
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(SizeType), (char*)&count);
    if (!m_offsets.Save(p_metaIndexOut)) return ErrorCode::DiskIOFail;

    IOBINARY(p_metaOut, WriteBinary, m_metadataHolder.Length(), reinterpret_cast<const char*>(m_metadataHolder.Data()));
    if (m_newdata.Size() > 0 && !m_newdata.Save(p_metaOut)) return ErrorCode::DiskIOFail;
    LOG(Helper::LogLevel::LL_Info, "Save MetaIndex(%llu) Meta(%llu)\n", m_offsets.Size() - 1, m_offsets[m_offsets.Size() - 1]);
    return ErrorCode::Success;
}



ErrorCode
MemMetadataSet::SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile)
{
    {
        std::shared_ptr<Helper::DiskPriorityIO> metaOut = f_createIO(), metaIndexOut = f_createIO();
        if (metaOut == nullptr || metaIndexOut == nullptr || !metaOut->Initialize((p_metaFile + "_tmp").c_str(), std::ios::binary | std::ios::out) || !metaIndexOut->Initialize((p_metaindexFile + "_tmp").c_str(), std::ios::binary | std::ios::out))
            return ErrorCode::FailedCreateFile;

        ErrorCode ret = SaveMetadata(metaOut, metaIndexOut);
        if (ret != ErrorCode::Success) return ret;
    }
    if (fileexists(p_metaFile.c_str())) std::remove(p_metaFile.c_str());
    if (fileexists(p_metaindexFile.c_str())) std::remove(p_metaindexFile.c_str());
    std::rename((p_metaFile + "_tmp").c_str(), p_metaFile.c_str());
    std::rename((p_metaindexFile + "_tmp").c_str(), p_metaindexFile.c_str());
    return ErrorCode::Success;
}

