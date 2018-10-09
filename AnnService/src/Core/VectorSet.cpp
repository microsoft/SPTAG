#include "inc/Core/VectorSet.h"

using namespace SpaceV;


VectorSet::VectorSet()
{
}


VectorSet::~VectorSet()
{
}


BasicVectorSet::BasicVectorSet(const ByteArray& p_bytesArray,
                               VectorValueType p_valueType,
                               SizeType p_dimension,
                               SizeType p_vectorCount)
    : m_data(p_bytesArray),
      m_valueType(p_valueType),
      m_dimension(p_dimension),
      m_vectorCount(p_vectorCount),
      m_perVectorDataSize(static_cast<SizeType>(p_dimension * GetValueTypeSize(p_valueType)))
{
}


BasicVectorSet::~BasicVectorSet()
{
}


void*
BasicVectorSet::GetVector(IndexType p_vectorID) const
{
    if (p_vectorID < 0 || static_cast<SizeType>(p_vectorID) >= m_vectorCount)
    {
        return nullptr;
    }

    SizeType offset = static_cast<SizeType>(p_vectorID) * m_perVectorDataSize;
    return reinterpret_cast<void*>(m_data.Data() + offset);
}


void*
BasicVectorSet::GetData() const
{
    return reinterpret_cast<void*>(m_data.Data());
}


VectorValueType
BasicVectorSet::ValueType() const
{
    return m_valueType;
}


SizeType
BasicVectorSet::Dimension() const
{
    return m_dimension;
}


SizeType
BasicVectorSet::Count() const
{
    return m_vectorCount;
}
