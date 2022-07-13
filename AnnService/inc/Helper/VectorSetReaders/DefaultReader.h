// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_VECTORSETREADERS_DEFAULTREADER_H_
#define _SPTAG_HELPER_VECTORSETREADERS_DEFAULTREADER_H_

#include "inc/Helper/VectorSetReader.h"

namespace SPTAG
{
namespace Helper
{

class DefaultVectorReader : public VectorSetReader
{
public:
    DefaultVectorReader(std::shared_ptr<ReaderOptions> p_options);

    virtual ~DefaultVectorReader();

    virtual ErrorCode LoadFile(const std::string& p_filePaths);

    virtual std::shared_ptr<VectorSet> GetVectorSet(SizeType start = 0, SizeType end = -1) const;

    virtual std::shared_ptr<MetadataSet> GetMetadataSet() const;

private:
    std::string m_vectorOutput;

    std::string m_metadataConentOutput;

    std::string m_metadataIndexOutput;
};



} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_VECTORSETREADERS_DEFAULTREADER_H_
