// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_VECTORSETREADERS_XVECREADER_H_
#define _SPTAG_HELPER_VECTORSETREADERS_XVECREADER_H_

#include "../VectorSetReader.h"

namespace SPTAG
{
namespace Helper
{

class XvecReader : public VectorSetReader
{
public:
    XvecReader(std::shared_ptr<ReaderOptions> p_options);

    virtual ~XvecReader();

    virtual ErrorCode LoadFile(const std::string& p_filePaths);

    virtual std::shared_ptr<VectorSet> GetVectorSet() const;

    virtual std::shared_ptr<MetadataSet> GetMetadataSet() const;

private:
    std::string m_vectorOutput;
};



} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_VECTORSETREADERS_XVECREADER_H_
