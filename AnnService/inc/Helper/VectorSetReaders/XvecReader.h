// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_VECTORSETREADERS_XVECREADER_H_
#define _SPTAG_HELPER_VECTORSETREADERS_XVECREADER_H_

#include "inc/Helper/VectorSetReader.h"

namespace SPTAG
{
namespace Helper
{

class XvecVectorReader : public VectorSetReader
{
public:
    XvecVectorReader(std::shared_ptr<ReaderOptions> p_options);

    virtual ~XvecVectorReader();

    virtual ErrorCode LoadFile(const std::string& p_filePaths);

    virtual std::shared_ptr<VectorSet> GetVectorSet(SizeType start = 0, SizeType end = -1) const;

    virtual std::shared_ptr<MetadataSet> GetMetadataSet() const;

private:
    std::string m_vectorOutput;
};



} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_VECTORSETREADERS_XVECREADER_H_
