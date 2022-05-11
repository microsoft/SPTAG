// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_PW_COREINTERFACE_H_
#define _SPTAG_PW_COREINTERFACE_H_

#include "TransferDataType.h"
#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"

typedef int SizeType;
typedef int DimensionType;

class AnnIndex
{
public:
    AnnIndex(DimensionType p_dimension);

    AnnIndex(const char* p_algoType, const char* p_valueType, DimensionType p_dimension);

    ~AnnIndex();

    void SetBuildParam(const char* p_name, const char* p_value, const char* p_section);

    void SetSearchParam(const char* p_name, const char* p_value, const char* p_section);

    bool LoadQuantizer(const char* p_quantizerFile);

    void SetQuantizerADC(bool p_adc);

    bool BuildSPANN(bool p_normalized);

    bool BuildSPANNWithMetaData(ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized);

    bool Build(ByteArray p_data, SizeType p_num, bool p_normalized);

    bool BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized);

    std::shared_ptr<QueryResult> Search(ByteArray p_data, int p_resultNum);

    std::shared_ptr<QueryResult> SearchWithMetaData(ByteArray p_data, int p_resultNum);

    std::shared_ptr<QueryResult> BatchSearch(ByteArray p_data, int p_vectorNum, int p_resultNum, bool p_withMetaData);

    bool ReadyToServe() const;

    void UpdateIndex();

    bool Save(const char* p_saveFile) const;

    bool Add(ByteArray p_data, SizeType p_num, bool p_normalized);

    bool AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized);

    bool Delete(ByteArray p_data, SizeType p_num);

    bool DeleteByMetaData(ByteArray p_meta);

    static AnnIndex Load(const char* p_loaderFile);

    static AnnIndex Merge(const char* p_indexFilePath1, const char* p_indexFilePath2);

private:
    AnnIndex(const std::shared_ptr<SPTAG::VectorIndex>& p_index);
    
    std::shared_ptr<SPTAG::VectorIndex> m_index;

    size_t m_inputVectorSize;
    
    DimensionType m_dimension;

    SPTAG::IndexAlgoType m_algoType;

    SPTAG::VectorValueType m_inputValueType;
};

#endif // _SPTAG_PW_COREINTERFACE_H_
