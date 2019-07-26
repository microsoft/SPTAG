// This is the main DLL file.

#include "inc/CLRCoreInterface.h"


namespace CLI {
    AnnIndex::AnnIndex(std::shared_ptr<SPTAG::VectorIndex> p_index):
        ManagedObject(p_index)
    {
        m_dimension = p_index->GetFeatureDim();
        m_inputVectorSize = SPTAG::GetValueTypeSize(p_index->GetVectorValueType()) * m_dimension;
    }

    AnnIndex::AnnIndex(String^ p_algoType, String^ p_valueType, int p_dimension): 
        ManagedObject(SPTAG::VectorIndex::CreateInstance(string_to<SPTAG::IndexAlgoType>(p_algoType), string_to<SPTAG::VectorValueType>(p_valueType)))
    {
        m_dimension = p_dimension;
        m_inputVectorSize = SPTAG::GetValueTypeSize((*m_Instance)->GetVectorValueType()) * m_dimension;
    }

    void AnnIndex::SetBuildParam(String^ p_name, String^ p_value)
    {
        if (m_Instance != nullptr)
            (*m_Instance)->SetParameter(string_to_char_array(p_name), string_to_char_array(p_value));
    }

    void AnnIndex::SetSearchParam(String^ p_name, String^ p_value)
    {
        if (m_Instance != nullptr)
            (*m_Instance)->SetParameter(string_to_char_array(p_name), string_to_char_array(p_value));
    }

    bool AnnIndex::Build(array<Byte>^ p_data, int p_num)
    {
        if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
            return false;

        pin_ptr<Byte> ptr = &p_data[0];
        return (SPTAG::ErrorCode::Success == (*m_Instance)->BuildIndex(ptr, p_num, m_dimension));
    }

    bool AnnIndex::BuildWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex)
    {
        if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
            return false;

        pin_ptr<Byte> dataptr = &p_data[0];
        std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(SPTAG::ByteArray(dataptr, p_data->LongLength, false), (*m_Instance)->GetVectorValueType(), m_dimension, p_num));
        
        pin_ptr<Byte> metaptr = &p_meta[0];
        std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
        int current = 0;
        for (long long i = 0; i < p_meta->LongLength; i++) {
            if (((char)metaptr[i]) == '\n')
                offsets[++current] = (std::uint64_t)(i + 1);
        }
        std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(SPTAG::ByteArray(metaptr, p_meta->LongLength, false), SPTAG::ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), p_num));
        return (SPTAG::ErrorCode::Success == (*m_Instance)->BuildIndex(vectors, meta, p_withMetaIndex));
    }

    array<Result^>^ AnnIndex::Search(array<Byte>^ p_data, int p_resultNum)
    {
        array<Result^>^ res;
        if (m_Instance == nullptr || m_dimension == 0 || p_data->LongLength != m_inputVectorSize)
            return res;

        pin_ptr<Byte> ptr = &p_data[0];
        SPTAG::BasicResult* results = new SPTAG::BasicResult[p_resultNum];
        (*m_Instance)->SearchIndex(ptr, p_resultNum, false, results);

        res = gcnew array<Result^>(p_resultNum);
        for (int i = 0; i < p_resultNum; i++)
            res[i] = gcnew Result(&results[i]);

        return res;
    }

    array<Result^>^ AnnIndex::SearchWithMetaData(array<Byte>^ p_data, int p_resultNum)
    {
        array<Result^>^ res;
        if (m_Instance == nullptr || m_dimension == 0 || p_data->LongLength != m_inputVectorSize)
            return res;

        pin_ptr<Byte> ptr = &p_data[0];
        SPTAG::BasicResult* results = new SPTAG::BasicResult[p_resultNum];
        (*m_Instance)->SearchIndex(ptr, p_resultNum, true, results);

        res = gcnew array<Result^>(p_resultNum);
        for (int i = 0; i < p_resultNum; i++)
            res[i] = gcnew Result(&results[i]);

        return res;
    }

    bool AnnIndex::Save(String^ p_saveFile)
    {
        return SPTAG::ErrorCode::Success == (*m_Instance)->SaveIndex(string_to_char_array(p_saveFile));
    }

    bool AnnIndex::Add(array<Byte>^ p_data, int p_num)
    {
        if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
            return false;

        pin_ptr<Byte> ptr = &p_data[0];
        return (SPTAG::ErrorCode::Success == (*m_Instance)->AddIndex(ptr, p_num, m_dimension));
    }

    bool AnnIndex::AddWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num)
    {
        if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
            return false;

        pin_ptr<Byte> dataptr = &p_data[0];
        std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(SPTAG::ByteArray(dataptr, p_data->LongLength, false), (*m_Instance)->GetVectorValueType(), m_dimension, p_num));

        pin_ptr<Byte> metaptr = &p_meta[0];
        std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
        int current = 0;
        for (long long i = 0; i < p_meta->LongLength; i++) {
            if (((char)metaptr[i]) == '\n')
                offsets[++current] = (std::uint64_t)(i + 1);
        }
        std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(SPTAG::ByteArray(metaptr, p_meta->LongLength, false), SPTAG::ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), p_num));
        return (SPTAG::ErrorCode::Success == (*m_Instance)->AddIndex(vectors, meta));
    }

    bool AnnIndex::Delete(array<Byte>^ p_data, int p_num)
    {
        if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
            return false;

        pin_ptr<Byte> ptr = &p_data[0];
        return (SPTAG::ErrorCode::Success == (*m_Instance)->DeleteIndex(ptr, p_num));
    }

    bool AnnIndex::DeleteByMetaData(array<Byte>^ p_meta)
    {
        if (m_Instance == nullptr)
            return false;

        pin_ptr<Byte> metaptr = &p_meta[0];
        return (SPTAG::ErrorCode::Success == (*m_Instance)->DeleteIndex(SPTAG::ByteArray(metaptr, p_meta->LongLength, false)));
    }

    AnnIndex^ AnnIndex::Load(String^ p_loaderFile)
    {
        std::shared_ptr<SPTAG::VectorIndex> vecIndex;
        AnnIndex^ res;
        if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(string_to_char_array(p_loaderFile), vecIndex) || nullptr == vecIndex)
        {
            res = gcnew AnnIndex(nullptr);
        }
        else {
            res = gcnew AnnIndex(vecIndex);
        }
        return res;
    }

    bool AnnIndex::Merge(String^ p_indexFilePath1, String^ p_indexFilePath2)
    {
        return (SPTAG::ErrorCode::Success == SPTAG::VectorIndex::MergeIndex(string_to_char_array(p_indexFilePath1), string_to_char_array(p_indexFilePath2)));
    }
}