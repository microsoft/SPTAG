#include "inc/SSDServing/VectorSearch/DiskFileReader.h"

#include <cstdio>

using namespace SPTAG::SSDServing::VectorSearch;

DiskFileReadRequest::DiskFileReadRequest()
    : m_offset(UINT64_MAX),
    m_readSize(0),
    m_buffer(nullptr)
{
}


IDiskFileReader::IDiskFileReader()
{
}


IDiskFileReader::~IDiskFileReader()
{
}

