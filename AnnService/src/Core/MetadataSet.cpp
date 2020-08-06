// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/MetadataSet.h"

#include <string.h>
#include <shared_mutex>

using namespace SPTAG;

ErrorCode
MetadataSet::RefineMetadata(std::vector<SizeType>& indices, std::shared_ptr<MetadataSet>& p_newMetadata) const
{
    p_newMetadata.reset(new MemMetadataSet());
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


FileMetadataSet::FileMetadataSet(const std::string& p_metafile, const std::string& p_metaindexfile)
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

    m_pOffsets.resize(m_count + 1);
    if (fpidx->ReadBinary(sizeof(std::uint64_t) * (m_count + 1), (char*)m_pOffsets.data()) != sizeof(std::uint64_t) * (m_count + 1)) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read FileMetadataSet!\n");
        exit(1);
    }
    LOG(Helper::LogLevel::LL_Info, "Load MetaIndex(%zu) Meta(%llu)\n", m_pOffsets.size() - 1, m_pOffsets.back());
}


FileMetadataSet::~FileMetadataSet()
{
}


ByteArray
FileMetadataSet::GetMetadata(SizeType p_vectorID) const
{
    std::uint64_t startoff = m_pOffsets[p_vectorID];
    std::uint64_t bytes = m_pOffsets[p_vectorID + 1] - startoff;
    if (p_vectorID < m_count) {
        ByteArray b = ByteArray::Alloc(bytes);
        m_fp->ReadBinary(bytes, (char*)b.Data(), startoff);
        return b;
    }
    else {
        startoff -= m_pOffsets[m_count];
        return ByteArray((std::uint8_t*)m_newdata.data() + startoff, bytes, false);
    }
}


ByteArray
FileMetadataSet::GetMetadataCopy(SizeType p_vectorID) const
{
    std::uint64_t startoff = m_pOffsets[p_vectorID];
    std::uint64_t bytes = m_pOffsets[p_vectorID + 1] - startoff;
    ByteArray b = ByteArray::Alloc(bytes);
    if (p_vectorID < m_count) {
        m_fp->ReadBinary(bytes, (char*)b.Data(), startoff);
    }
    else {
        memcpy(b.Data(), m_newdata.data() + (startoff - m_pOffsets[m_count]), bytes);
    }
    return b;
}


SizeType
FileMetadataSet::Count() const
{
    return static_cast<SizeType>(m_pOffsets.size() - 1);
}


bool
FileMetadataSet::Available() const
{
    return m_fp != nullptr && m_pOffsets.size() > 1;
}


std::pair<std::uint64_t, std::uint64_t> 
FileMetadataSet::BufferSize() const
{
    return std::make_pair(m_pOffsets[m_pOffsets.size() - 1], 
        sizeof(SizeType) + sizeof(std::uint64_t) * m_pOffsets.size());
}


void
FileMetadataSet::Add(const ByteArray& data)
{
    m_newdata.insert(m_newdata.end(), data.Data(), data.Data() + data.Length());
    m_pOffsets.push_back(m_pOffsets.back() + data.Length());
}


ErrorCode
FileMetadataSet::SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut)
{
    SizeType count = Count();
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(SizeType), (char*)&count);
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(std::uint64_t) * m_pOffsets.size(), (char*)m_pOffsets.data());

    std::uint64_t bufsize = 1000000;
    char* buf = new char[bufsize];
    auto readsize = m_fp->ReadBinary(bufsize, buf, 0);
    while (readsize > 0) {
        IOBINARY(p_metaOut, WriteBinary, readsize, buf);
        readsize = m_fp->ReadBinary(bufsize, buf);
    }
    delete[] buf;
    
    if (m_newdata.size() > 0) {
        IOBINARY(p_metaOut, WriteBinary, m_newdata.size(), (char*)m_newdata.data());
    }
    LOG(Helper::LogLevel::LL_Info, "Save MetaIndex(%zu) Meta(%llu)\n", m_pOffsets.size() - 1, m_pOffsets.back());
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
    m_newdata.clear();
    return ErrorCode::Success;
}


MemMetadataSet::MemMetadataSet(): m_count(0), m_metadataHolder(ByteArray::c_empty)
{
    m_offsets.push_back(0);
    m_lock.reset(new std::shared_timed_mutex, std::default_delete<std::shared_timed_mutex>());
}


ErrorCode
MemMetadataSet::Init(std::shared_ptr<Helper::DiskPriorityIO> p_metain, std::shared_ptr<Helper::DiskPriorityIO> p_metaindexin)
{
    IOBINARY(p_metaindexin, ReadBinary, sizeof(m_count), (char*)&m_count);
    m_offsets.resize(m_count + 1);
    IOBINARY(p_metaindexin, ReadBinary, sizeof(std::uint64_t) * (m_count + 1), (char *)m_offsets.data());

    m_metadataHolder = ByteArray::Alloc(m_offsets[m_count]);
    IOBINARY(p_metain, ReadBinary, m_metadataHolder.Length(), (char *)m_metadataHolder.Data());
    m_lock.reset(new std::shared_timed_mutex, std::default_delete<std::shared_timed_mutex>());
    LOG(Helper::LogLevel::LL_Info, "Load MetaIndex(%zu) Meta(%llu)\n", m_offsets.size() - 1, m_offsets.back());
    return ErrorCode::Success;
}


MemMetadataSet::MemMetadataSet(std::shared_ptr<Helper::DiskPriorityIO> p_metain, std::shared_ptr<Helper::DiskPriorityIO> p_metaindexin)
{
    if (Init(p_metain, p_metaindexin) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read MemMetadataSet!\n");
        exit(1);
    }
}

MemMetadataSet::MemMetadataSet(const std::string& p_metafile, const std::string& p_metaindexfile)
{
    std::shared_ptr<Helper::DiskPriorityIO> ptrMeta = f_createIO(), ptrMetaIndex = f_createIO();
    if (ptrMeta == nullptr || ptrMetaIndex == nullptr || !ptrMeta->Initialize(p_metafile.c_str(), std::ios::binary | std::ios::in) || !ptrMetaIndex->Initialize(p_metaindexfile.c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open meta files %s or %s!\n", p_metafile.c_str(),  p_metaindexfile.c_str());
        exit(1);
    }
    if (Init(ptrMeta, ptrMetaIndex) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot read MemMetadataSet!\n");
        exit(1);
    }
}


MemMetadataSet::MemMetadataSet(ByteArray p_metadata, ByteArray p_offsets, SizeType p_count)
    : m_metadataHolder(std::move(p_metadata)),
      m_count(p_count)
{
    const std::uint64_t* newdata = reinterpret_cast<const std::uint64_t*>(p_offsets.Data());
    m_offsets.assign(newdata, newdata + p_count + 1);
    m_lock.reset(new std::shared_timed_mutex, std::default_delete<std::shared_timed_mutex>());
}


MemMetadataSet::~MemMetadataSet()
{
}


ByteArray
MemMetadataSet::GetMetadata(SizeType p_vectorID) const
{
    std::shared_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    if (p_vectorID < m_count)
    {
        return ByteArray(m_metadataHolder.Data() + m_offsets[p_vectorID],
                         m_offsets[p_vectorID + 1] - m_offsets[p_vectorID],
                         false);
    }
    else if (p_vectorID < (SizeType)(m_offsets.size() - 1)) {
        return ByteArray((std::uint8_t*)m_newdata.data() + m_offsets[p_vectorID] - m_offsets[m_count],
            m_offsets[p_vectorID + 1] - m_offsets[p_vectorID],
            false);
    }

    return ByteArray::c_empty;
}


ByteArray
MemMetadataSet::GetMetadataCopy(SizeType p_vectorID) const
{
    std::shared_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    std::uint64_t startoff = m_offsets[p_vectorID];
    std::uint64_t bytes = m_offsets[p_vectorID + 1] - startoff;
    ByteArray b = ByteArray::Alloc(bytes);
    if (p_vectorID < m_count) {
        memcpy(b.Data(), m_metadataHolder.Data() + startoff, bytes);
    }
    else {
        memcpy(b.Data(), m_newdata.data() + (startoff - m_offsets[m_count]), bytes);
    }
    return b;
}


SizeType
MemMetadataSet::Count() const
{
    std::shared_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    return static_cast<SizeType>(m_offsets.size() - 1);
}


bool
MemMetadataSet::Available() const
{
    std::shared_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    return m_offsets.size() > 1;
}


std::pair<std::uint64_t, std::uint64_t>
MemMetadataSet::BufferSize() const
{
    std::shared_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    return std::make_pair(m_offsets.back(),
        sizeof(SizeType) + sizeof(std::uint64_t) * m_offsets.size());
}


void
MemMetadataSet::Add(const ByteArray& data)
{
    std::unique_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    m_newdata.insert(m_newdata.end(), data.Data(), data.Data() + data.Length());
    m_offsets.push_back(m_offsets.back() + data.Length());
}


ErrorCode
MemMetadataSet::SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut)
{
    std::shared_lock<std::shared_timed_mutex> lock(*static_cast<std::shared_timed_mutex*>(m_lock.get()));
    SizeType count = Count();
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(SizeType), (char*)&count);
    IOBINARY(p_metaIndexOut, WriteBinary, sizeof(std::uint64_t) * m_offsets.size(), (char*)m_offsets.data());

    IOBINARY(p_metaOut, WriteBinary, m_metadataHolder.Length(), reinterpret_cast<const char*>(m_metadataHolder.Data()));
    if (m_newdata.size() > 0) {
        IOBINARY(p_metaOut, WriteBinary, m_newdata.size(), (char*)m_newdata.data());
    }
    LOG(Helper::LogLevel::LL_Info, "Save MetaIndex(%zu) Meta(%llu)\n", m_offsets.size() - 1, m_offsets.back());
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

