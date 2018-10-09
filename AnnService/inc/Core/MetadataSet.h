#ifndef _SPTAG_METADATASET_H_
#define _SPTAG_METADATASET_H_

#include "Common.h"
#include "CommonDataStructure.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace SPTAG
{

class MetadataSet
{
public:
    MetadataSet();

    virtual ~MetadataSet();

    virtual ByteArray GetMetadata(IndexType p_vectorID) const = 0;

    virtual SizeType Count() const = 0;

    virtual bool Available() const = 0;

    virtual ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const = 0;
};


class FileMetadataSet : public MetadataSet
{
public:
    FileMetadataSet(const std::string& p_metaFile, const std::string& p_metaindexFile);
    
    ~FileMetadataSet();

    ByteArray GetMetadata(IndexType p_vectorID) const;

    SizeType Count() const;

    bool Available() const;

    virtual ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const;

private:
    std::ifstream* m_fp = nullptr;

    const long long *m_pOffsets = nullptr;

    const int m_iCount = 0;

    std::string m_metaFile;

    std::string m_metaindexFile;
};


class MemMetadataSet : public MetadataSet
{
public:
    MemMetadataSet(ByteArray p_metadata, ByteArray p_offsets, SizeType p_count);

    ~MemMetadataSet();

    ByteArray GetMetadata(IndexType p_vectorID) const;

    SizeType Count() const;

    bool Available() const;

    virtual ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const;

private:
    const std::uint64_t *m_offsets;

    ByteArray m_metadataHolder;

    ByteArray m_offsetHolder;

    SizeType m_count;
};


class MetadataSetFileTransfer : public MetadataSet
{
public:
    MetadataSetFileTransfer(const std::string& p_metaFile, const std::string& p_metaindexFile);

    virtual ~MetadataSetFileTransfer();

    virtual ByteArray GetMetadata(IndexType p_vectorID) const;

    virtual SizeType Count() const;

    virtual bool Available() const;

    virtual ErrorCode SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const;

private:
    std::string m_metaFile;

    std::string m_metaindexFile;
};


} // namespace SPTAG

#endif // _SPTAG_METADATASET_H_
