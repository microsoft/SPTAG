// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_VECTORSETREADER_H_
#define _SPTAG_HELPER_VECTORSETREADER_H_

#include "inc/Core/Common.h"
#include "inc/Core/VectorSet.h"
#include "inc/Core/MetadataSet.h"
#include "inc/Helper/ArgumentsParser.h"

#include <memory>

namespace SPTAG
{
namespace Helper
{

class ReaderOptions : public ArgumentsParser
{
public:
    ReaderOptions(VectorValueType p_valueType, DimensionType p_dimension, VectorFileType p_fileType, std::string p_vectorDelimiter = "|", std::uint32_t p_threadNum = 32, bool p_normalized = false);

    ~ReaderOptions();

    SPTAG::VectorValueType m_inputValueType;
    
    DimensionType m_dimension;   
  
    SPTAG::VectorFileType m_inputFileType;

    std::string m_vectorDelimiter;    
    
    std::uint32_t m_threadNum;

    bool m_normalized;
};

class VectorSetReader
{
public:
    VectorSetReader(std::shared_ptr<ReaderOptions> p_options);

    virtual ~VectorSetReader();

    virtual ErrorCode LoadFile(const std::string& p_filePath) = 0;

    virtual std::shared_ptr<VectorSet> GetVectorSet(SizeType start = 0, SizeType end = -1) const = 0;

    virtual std::shared_ptr<MetadataSet> GetMetadataSet() const = 0;

    virtual bool IsNormalized() const { return m_options->m_normalized; }

    static std::shared_ptr<VectorSetReader> CreateInstance(std::shared_ptr<ReaderOptions> p_options);

protected:
    std::shared_ptr<ReaderOptions> m_options;
};



} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_VECTORSETREADER_H_
