// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/VectorSet.h"
#include "inttypes.h"
#include <fstream>
#include <memory>
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/CommonUtils.h"

using namespace SPTAG;

#pragma warning(disable:4996)  // 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

VectorSet::VectorSet()
{
}


VectorSet::~VectorSet()
{
}


BasicVectorSet::BasicVectorSet(const ByteArray& p_bytesArray,
                               VectorValueType p_valueType,
                               DimensionType p_dimension,
                               SizeType p_vectorCount)
    : m_data(p_bytesArray),
      m_valueType(p_valueType),
      m_dimension(p_dimension),
      m_vectorCount(p_vectorCount),
      m_perVectorDataSize(static_cast<SizeType>(p_dimension * GetValueTypeSize(p_valueType)))
{
}


void
BasicVectorSet::Normalize(DistCalcMethod p_distCalcMethod)
{
    if (p_distCalcMethod == DistCalcMethod::Cosine) {
#pragma omp parallel for
        for (int64_t i = 0; i < m_vectorCount; i++)
        {
            int64_t offset = i * m_perVectorDataSize;
            switch (m_valueType)
            {
#define DefineVectorValueType(Name, Type) \
case SPTAG::VectorValueType::Name: \
SPTAG::COMMON::Utils::Normalize<Type>(reinterpret_cast<Type *>(m_data.Data() + offset), p_dimension, SPTAG::COMMON::Utils::GetBase<Type>()); \
break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
            default:
                break;
            }
        }
    }
}

BasicVectorSet::~BasicVectorSet()
{
}


VectorValueType
BasicVectorSet::GetValueType() const
{
    return m_valueType;
}


void*
BasicVectorSet::GetVector(SizeType p_vectorID) const
{
    if (p_vectorID < 0 || p_vectorID >= m_vectorCount)
    {
        return nullptr;
    }

    return reinterpret_cast<void*>(m_data.Data() + ((size_t)p_vectorID) * m_perVectorDataSize);
}


void*
BasicVectorSet::GetData() const
{
    return reinterpret_cast<void*>(m_data.Data());
}

DimensionType
BasicVectorSet::Dimension() const
{
    return m_dimension;
}


SizeType
BasicVectorSet::Count() const
{
    return m_vectorCount;
}


bool
BasicVectorSet::Available() const
{
    return m_data.Data() != nullptr;
}


ErrorCode 
BasicVectorSet::Save(const std::string& p_vectorFile) const
{
    FILE * fp = fopen(p_vectorFile.c_str(), "wb");
    if (fp == NULL) return ErrorCode::FailedOpenFile;

    fwrite(&m_vectorCount, sizeof(SizeType), 1, fp);
    fwrite(&m_dimension, sizeof(DimensionType), 1, fp);

    fwrite((const void*)(m_data.Data()), m_data.Length(), 1, fp);
    fclose(fp);
    return ErrorCode::Success;
}

SizeType BasicVectorSet::PerVectorDataSize() const {
    return (SizeType)m_perVectorDataSize;
}