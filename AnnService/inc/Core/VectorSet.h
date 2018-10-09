#ifndef _SPTAG_VECTORSET_H_
#define _SPTAG_VECTORSET_H_

#include "Common.h"
#include "CommonDataStructure.h"

#include <memory>
#include <string>

namespace SPTAG
{

class VectorSet
{
public:
    VectorSet();

    virtual ~VectorSet();

    virtual void* GetVector(IndexType p_vectorID) const = 0;

    virtual void* GetData() const = 0;

    virtual VectorValueType ValueType() const = 0;

    virtual SizeType Dimension() const = 0;

    virtual SizeType Count() const = 0;
};


class BasicVectorSet : public VectorSet
{
public:
    BasicVectorSet(const ByteArray& p_bytesArray,
                   VectorValueType p_valueType,
                   SizeType p_dimension,
                   SizeType p_vectorCount);

    virtual ~BasicVectorSet();

    virtual void* GetVector(IndexType p_vectorID) const;

    virtual void* GetData() const;

    virtual VectorValueType ValueType() const;

    virtual SizeType Dimension() const;

    virtual SizeType Count() const;

private:
    ByteArray m_data;

    VectorValueType m_valueType;

    SizeType m_dimension;

    SizeType m_vectorCount;

    SizeType m_perVectorDataSize;
};

} // namespace SPTAG

#endif // _SPTAG_VECTORSET_H_
