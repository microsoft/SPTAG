// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_VECTORSETREADERS_MEMORYREADER_H_
#define _SPTAG_HELPER_VECTORSETREADERS_MEMORYREADER_H_

#include "inc/Helper/VectorSetReader.h"

namespace SPTAG
{
    namespace Helper
    {

        class MemoryVectorReader : public VectorSetReader
        {
        public:
            MemoryVectorReader(std::shared_ptr<ReaderOptions> p_options, std::shared_ptr<VectorSet> p_vectors) :
                VectorSetReader(p_options), m_vectors(p_vectors)
            {}

            virtual ~MemoryVectorReader() {}

            virtual ErrorCode LoadFile(const std::string& p_filePaths) { return ErrorCode::Success; }

            virtual std::shared_ptr<VectorSet> GetVectorSet(SizeType start = 0, SizeType end = -1) const 
            {
                if (end < 0 || end > m_vectors->Count()) end = m_vectors->Count();
                return std::shared_ptr<VectorSet>(new BasicVectorSet(ByteArray((std::uint8_t*)(m_vectors->GetVector(start)), (end - start) * m_vectors->PerVectorDataSize(), false),
                    m_vectors->GetValueType(),
                    m_vectors->Dimension(),
                    end - start));
            }

            virtual std::shared_ptr<MetadataSet> GetMetadataSet() const { return nullptr; }

        private:
            std::shared_ptr<VectorSet> m_vectors;
        };

    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_VECTORSETREADERS_MEMORYREADER_H_
