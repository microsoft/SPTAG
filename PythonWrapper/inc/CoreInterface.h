// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_PW_COREINTERFACE_H_
#define _SPTAG_PW_COREINTERFACE_H_

#ifndef SWIG

#include "TransferDataType.h"
#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"

#else
%module SPTAG

%{
#include "inc/CoreInterface.h"
%}

%include <std_shared_ptr.i>
%shared_ptr(AnnIndex)
%shared_ptr(QueryResult)
%include "PyByteArray.i"

%{
#define SWIG_FILE_WITH_INIT
%}

#endif // SWIG


typedef unsigned int SizeType;

class AnnIndex
{
public:
    AnnIndex(SizeType p_dimension);

    AnnIndex(const char* p_algoType, const char* p_valueType, SizeType p_dimension);

    ~AnnIndex();

    void SetBuildParam(const char* p_name, const char* p_value);

    void SetSearchParam(const char* p_name, const char* p_value);

    bool Build(ByteArray p_data, SizeType p_num);

    bool BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num);

    std::shared_ptr<QueryResult> Search(ByteArray p_data, SizeType p_resultNum);

    std::shared_ptr<QueryResult> SearchWithMetaData(ByteArray p_data, SizeType p_resultNum);

    bool ReadyToServe() const;

    bool Save(const char* p_saveFile) const;

    bool Add(ByteArray p_data, SizeType p_num);

    bool AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num);

    bool Delete(ByteArray p_data, SizeType p_num);

    static AnnIndex Load(const char* p_loaderFile);

private:
    AnnIndex(const std::shared_ptr<SPTAG::VectorIndex>& p_index);
    
    std::shared_ptr<SPTAG::VectorIndex> m_index;

    SizeType m_inputVectorSize;
    
    SizeType m_dimension;

    SPTAG::IndexAlgoType m_algoType;

    SPTAG::VectorValueType m_inputValueType;
};

#endif // _SPTAG_PW_COREINTERFACE_H_
