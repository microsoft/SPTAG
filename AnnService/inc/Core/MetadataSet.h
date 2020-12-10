// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_METADATASET_H_
#define _SPTAG_METADATASET_H_

#include "CommonDataStructure.h"
#include "../Helper/LockFree.h"

namespace SPTAG
{

class MetadataSet
{
public:
    MetadataSet();

    virtual ~MetadataSet();

    virtual ByteArray GetMetadata(SizeType p_vectorID) const = 0;

    virtual ByteArray GetMetadataCopy(SizeType p_vectorID) const = 0;

    virtual SizeType Count() const = 0;

    virtual bool Available() const = 0;

    virtual std::pair<std::uint64_t, std::uint64_t> BufferSize() const = 0;

    virtual void Add(const ByteArray& data) = 0;

    virtual ErrorCode SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut) = 0;

    virtual ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) = 0;
 
    virtual void AddBatch(MetadataSet& data);
    
    virtual ErrorCode RefineMetadata(std::vector<SizeType>& indices, std::shared_ptr<MetadataSet>& p_newMetadata, std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize) const;

    virtual ErrorCode RefineMetadata(std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut) const;

    virtual ErrorCode RefineMetadata(std::vector<SizeType>& indices, const std::string& p_metaFile, const std::string& p_metaindexFile) const;
};


class FileMetadataSet : public MetadataSet
{
public:
    FileMetadataSet(const std::string& p_metaFile, const std::string& p_metaindexFile, 
        std::uint64_t p_blockSize = 1024 * 1024, std::uint64_t p_capacity = MaxSize, std::uint64_t p_metaSize = 10);
    
    ~FileMetadataSet();

    ByteArray GetMetadata(SizeType p_vectorID) const;

    ByteArray GetMetadataCopy(SizeType p_vectorID) const;

    SizeType Count() const;

    bool Available() const;

    std::pair<std::uint64_t, std::uint64_t> BufferSize() const;
    
    void Add(const ByteArray& data);

    ErrorCode SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut);

    ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile);

private:
    std::shared_ptr<Helper::DiskPriorityIO> m_fp = nullptr;

    Helper::LockFree::LockFreeVector<std::uint64_t> m_offsets;

    SizeType m_count;

    std::string m_metaFile;

    std::string m_metaindexFile;

    Helper::LockFree::LockFreeVector<std::uint8_t> m_newdata;
};


class MemMetadataSet : public MetadataSet
{
public:
    MemMetadataSet(std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize);

    MemMetadataSet(ByteArray p_metadata, ByteArray p_offsets, SizeType p_count, 
        std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize);

    MemMetadataSet(const std::string& p_metafile, const std::string& p_metaindexfile, 
        std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize);

    MemMetadataSet(std::shared_ptr<Helper::DiskPriorityIO> p_metain, std::shared_ptr<Helper::DiskPriorityIO> p_metaindexin, 
        std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize);

    ~MemMetadataSet();

    ByteArray GetMetadata(SizeType p_vectorID) const;
    
    ByteArray GetMetadataCopy(SizeType p_vectorID) const;

    SizeType Count() const;

    bool Available() const;

    std::pair<std::uint64_t, std::uint64_t> BufferSize() const;

    void Add(const ByteArray& data);

    ErrorCode SaveMetadata(std::shared_ptr<Helper::DiskPriorityIO> p_metaOut, std::shared_ptr<Helper::DiskPriorityIO> p_metaIndexOut);

    ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile);

private:
    ErrorCode Init(std::shared_ptr<Helper::DiskPriorityIO> p_metain, std::shared_ptr<Helper::DiskPriorityIO> p_metaindexin,
        std::uint64_t p_blockSize, std::uint64_t p_capacity, std::uint64_t p_metaSize);

    Helper::LockFree::LockFreeVector<std::uint64_t> m_offsets;

    SizeType m_count;

    ByteArray m_metadataHolder;

    Helper::LockFree::LockFreeVector<std::uint8_t> m_newdata;
};


} // namespace SPTAG

#endif // _SPTAG_METADATASET_H_
