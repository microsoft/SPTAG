#include "inc/Core/MetadataSet.h"

#include <fstream>
#include <iostream>

using namespace SpaceV;

namespace
{
namespace Local
{

ErrorCode
CopyFile(const std::string& p_src, const std::string& p_dst)
{
    std::ifstream src(p_src, std::ios::binary);
    if (!src.is_open())
    {
        std::cerr << "ERROR: Can't open " << p_src << std::endl;
        return ErrorCode::FailedOpenFile;
    }

    std::ofstream dst(p_dst, std::ios::binary);
    if (!dst.is_open())
    {
        std::cerr << "ERROR: Can't create " << p_dst << std::endl;
        src.close();
        return ErrorCode::FailedCreateFile;
    }

    int bufsize = 1000000;
    char* buf = new char[bufsize];
    while (!src.eof()) {
        src.read(buf, bufsize);
        dst.write(buf, src.gcount());
    }
    delete[] buf;

    src.close();
    dst.close();

    return ErrorCode::Success;
}

} // namespace Local
} // namespace


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
    m_fp = new std::ifstream(p_metafile, std::ifstream::binary);
    std::ifstream fpidx(p_metaindexfile, std::ifstream::binary);
    if (!m_fp->is_open() || !fpidx.is_open())
    {
        std::cerr << "ERROR: Cannot open meta files " << p_metafile << " and " << p_metaindexfile << "!" << std::endl;
        return;
    }

    fpidx.read((char *)&m_iCount, sizeof(int));
    m_pOffsets = new long long[m_iCount + 1];
    fpidx.read((char *)m_pOffsets, sizeof(long long) * (m_iCount + 1));
    fpidx.close();
}


FileMetadataSet::~FileMetadataSet()
{
    if (m_fp)
    {
        m_fp->close();
        delete m_fp;
    }
    delete[] m_pOffsets;
}


ByteArray
FileMetadataSet::GetMetadata(IndexType p_vectorID) const
{
    long long startoff = m_pOffsets[p_vectorID];
    long long bytes = m_pOffsets[p_vectorID + 1] - startoff;
    m_fp->seekg(startoff, std::ios_base::beg);
    ByteArray b = ByteArray::Alloc((SizeType)bytes);
    m_fp->read((char*)b.Data(), bytes);
    return b;
}


SizeType
FileMetadataSet::Count() const
{
    return m_iCount;
}


bool
FileMetadataSet::Available() const
{
    return m_fp && m_fp->is_open() && m_pOffsets;
}


ErrorCode
FileMetadataSet::SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const
{
    ErrorCode ret = ErrorCode::Success;
    if (p_metaFile != m_metaFile) 
    {
        m_fp->close();
        ret = Local::CopyFile(m_metaFile, p_metaFile);
        if (ErrorCode::Success != ret)
        {
            return ret;
        }
        m_fp->open(p_metaFile, std::ifstream::binary);
    }
    
    if (p_metaindexFile != m_metaindexFile) 
    {
        ret = Local::CopyFile(m_metaindexFile, p_metaindexFile);
    }

    return ret;
}


MemMetadataSet::MemMetadataSet(ByteArray p_metadata, ByteArray p_offsets, SizeType p_count)
    : m_metadataHolder(std::move(p_metadata)),
      m_offsetHolder(std::move(p_offsets)),
      m_count(p_count)
{
    m_offsets = reinterpret_cast<const std::uint64_t*>(m_offsetHolder.Data());
}


MemMetadataSet::~MemMetadataSet()
{
}


ByteArray
MemMetadataSet::GetMetadata(IndexType p_vectorID) const
{
    if (static_cast<SizeType>(p_vectorID) < m_count)
    {
        return ByteArray(m_metadataHolder.Data() + m_offsets[p_vectorID],
                         static_cast<SizeType>(m_offsets[p_vectorID + 1] - m_offsets[p_vectorID]),
                         m_metadataHolder.DataHolder());
    }

    return ByteArray::c_empty;
}


SizeType
MemMetadataSet::Count() const
{
    return m_count;
}


bool
MemMetadataSet::Available() const
{
    return m_metadataHolder.Length() > 0 && m_offsetHolder.Length() > 0;
}


ErrorCode
MemMetadataSet::SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const
{
    std::ofstream outputStream;
    outputStream.open(p_metaFile, std::ios::binary);
    if (!outputStream.is_open())
    {
        std::cerr << "Error: Failed to create file " << p_metaFile << "." << std::endl;
        return ErrorCode::FailedCreateFile;
    }

    outputStream.write(reinterpret_cast<const char*>(m_metadataHolder.Data()), m_metadataHolder.Length());
    outputStream.close();

    outputStream.open(p_metaindexFile, std::ios::binary);
    if (!outputStream.is_open())
    {
        std::cerr << "Error: Failed to create file " << p_metaindexFile << "." << std::endl;
        return ErrorCode::FailedCreateFile;
    }

    outputStream.write(reinterpret_cast<const char*>(&m_count), sizeof(m_count));
    outputStream.write(reinterpret_cast<const char*>(m_offsetHolder.Data()), m_offsetHolder.Length());
    outputStream.close();

    return ErrorCode::Success;
}


MetadataSetFileTransfer::MetadataSetFileTransfer(const std::string& p_metaFile, const std::string& p_metaindexFile)
    : m_metaFile(p_metaFile),
      m_metaindexFile(p_metaindexFile)
{
}


MetadataSetFileTransfer::~MetadataSetFileTransfer()
{
}


ByteArray
MetadataSetFileTransfer::GetMetadata(IndexType p_vectorID) const
{
    return ByteArray::c_empty;
}


SizeType
MetadataSetFileTransfer::Count() const
{
    return 0;
}


bool
MetadataSetFileTransfer::Available() const
{
    return false;
}


ErrorCode
MetadataSetFileTransfer::SaveMetadata(const std::string& p_metaFile, const std::string& p_metaindexFile) const
{
    auto ret = Local::CopyFile(m_metaFile, p_metaFile);
    if (ErrorCode::Success != ret)
    {
        return ret;
    }

    return Local::CopyFile(m_metaindexFile, p_metaindexFile);
}
