// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CLRCoreInterface.h"


namespace Microsoft
{
    namespace ANN
    {
        namespace SPTAGManaged
        {
            RIterator::RIterator(std::shared_ptr<ResultIterator> result_iterator) :
                ManagedObject(result_iterator)
            {
            }

            array<BasicResult^>^ RIterator::Next(int p_batch)
            {
                array<BasicResult^>^ res;
                if (m_Instance == nullptr)
                    return res;

                std::shared_ptr<QueryResult> results = (*m_Instance)->Next(p_batch);

                res = gcnew array<BasicResult^>(results->GetResultNum());
                for (int i = 0; i < results->GetResultNum(); i++)
                    res[i] = gcnew BasicResult(new SPTAG::BasicResult(*(results->GetResult(i))));

                return res;
            }

            bool RIterator::GetRelaxedMono()
            {
                return (*m_Instance)->GetRelaxedMono();
            }

            void RIterator::Close()
            {
                (*m_Instance)->Close();
            }

            AnnIndex::AnnIndex(std::shared_ptr<SPTAG::VectorIndex> p_index) :
                ManagedObject(p_index)
            {
                m_dimension = p_index->GetFeatureDim();
                m_inputVectorSize = SPTAG::GetValueTypeSize(p_index->GetVectorValueType()) * m_dimension;
            }

            AnnIndex::AnnIndex(String^ p_algoType, String^ p_valueType, int p_dimension) :
                ManagedObject(SPTAG::VectorIndex::CreateInstance(string_to<SPTAG::IndexAlgoType>(p_algoType), string_to<SPTAG::VectorValueType>(p_valueType)))
            {
                m_dimension = p_dimension;
                m_inputVectorSize = SPTAG::GetValueTypeSize((*m_Instance)->GetVectorValueType()) * m_dimension;
            }

            void AnnIndex::SetBuildParam(String^ p_name, String^ p_value, String^ p_section)
            {
                if (m_Instance != nullptr)
                    (*m_Instance)->SetParameter(string_to_char_array(p_name), string_to_char_array(p_value), string_to_char_array(p_section));
            }

            void AnnIndex::SetSearchParam(String^ p_name, String^ p_value, String^ p_section)
            {
                if (m_Instance != nullptr)
                    (*m_Instance)->SetParameter(string_to_char_array(p_name), string_to_char_array(p_value), string_to_char_array(p_section));
            }

            bool AnnIndex::LoadQuantizer(String^ p_quantizerFile)
            {
                if (m_Instance == nullptr) return false;
                
                auto ret = ((*m_Instance)->LoadQuantizer(string_to_char_array(p_quantizerFile)) == SPTAG::ErrorCode::Success);
                if (ret)
                {
                    m_inputVectorSize = (*m_Instance)->m_pQuantizer->QuantizeSize();
                }
                return ret;
            }

            void AnnIndex::SetQuantizerADC(bool p_adc)
            {
                if (m_Instance != nullptr)
                    (*m_Instance)->SetQuantizerADC(p_adc);
            }

            array<Byte>^ AnnIndex::QuantizeVector(array<Byte>^ p_data, int p_num) {
                array<Byte>^ res;
                if (m_Instance != nullptr && (*m_Instance)->GetQuantizer() != nullptr) {
                    res = gcnew array<Byte>((*m_Instance)->GetQuantizer()->GetNumSubvectors() * (size_t)p_num);
                    pin_ptr<Byte> ptr = &res[0], optr = &p_data[0];
                    (*m_Instance)->QuantizeVector(optr, p_num, SPTAG::ByteArray(ptr, res->LongLength, false));
                }
                return res;
            }

            array<Byte>^ AnnIndex::ReconstructVector(array<Byte>^ p_data, int p_num) {
                array<Byte>^ res;
                if (m_Instance != nullptr && (*m_Instance)->GetQuantizer() != nullptr) {
                    res = gcnew array<Byte>((*m_Instance)->GetQuantizer()->ReconstructSize() * (size_t)p_num);
                    pin_ptr<Byte> ptr = &res[0], optr = &p_data[0];
                    (*m_Instance)->ReconstructVector(optr, p_num, SPTAG::ByteArray(ptr, res->LongLength, false));
                }
                return res;
            }

            bool AnnIndex::BuildSPANN(bool p_normalized)
            {
                if (m_Instance == nullptr || (*m_Instance)->GetIndexAlgoType() != SPTAG::IndexAlgoType::SPANN) return false;

                return (*m_Instance)->BuildIndex(p_normalized) == SPTAG::ErrorCode::Success;
            }

            bool AnnIndex::BuildSPANNWithMetaData(array<Byte>^ p_meta, int p_num, bool p_withMetaIndex, bool p_normalized)
            {
                if (m_Instance == nullptr || (*m_Instance)->GetIndexAlgoType() != SPTAG::IndexAlgoType::SPANN) return false;

                pin_ptr<Byte> metaptr = &p_meta[0];
                std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
                if (!SPTAG::MetadataSet::GetMetadataOffsets(metaptr, p_meta->LongLength, offsets, p_num + 1, '\n')) return false;
                (*m_Instance)->SetMetadata(new SPTAG::MemMetadataSet(SPTAG::ByteArray(metaptr, p_meta->LongLength, false),
                    SPTAG::ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), p_num,
                    (*m_Instance)->m_iDataBlockSize, (*m_Instance)->m_iDataCapacity, (*m_Instance)->m_iMetaRecordSize));

                if (p_withMetaIndex) (*m_Instance)->BuildMetaMapping(false);

                return (SPTAG::ErrorCode::Success == (*m_Instance)->BuildIndex(p_normalized));
            }

            bool AnnIndex::Build(array<Byte>^ p_data, int p_num)
            {
                return Build(p_data, p_num, false);
            }

            bool AnnIndex::Build(array<Byte>^ p_data, int p_num, bool p_normalized)
            {
                if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
                    return false;

                pin_ptr<Byte> ptr = &p_data[0];
                return (SPTAG::ErrorCode::Success == (*m_Instance)->BuildIndex(ptr, p_num, m_dimension, p_normalized));
            }

            bool AnnIndex::BuildWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex)
            {
                return BuildWithMetaData(p_data, p_meta, p_num, p_withMetaIndex, false);
            }

            bool AnnIndex::BuildWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex, bool p_normalized)
            {
                if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
                    return false;

                pin_ptr<Byte> dataptr = &p_data[0];
                auto vectorType = (*m_Instance)->m_pQuantizer ? SPTAG::VectorValueType::UInt8 : (*m_Instance)->GetVectorValueType();
                auto vectorSize = (*m_Instance)->m_pQuantizer ? (*m_Instance)->m_pQuantizer->GetNumSubvectors() : m_dimension;
                std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(SPTAG::ByteArray(dataptr, p_data->LongLength, false), vectorType, vectorSize, p_num));

                pin_ptr<Byte> metaptr = &p_meta[0];
                std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
                if (!SPTAG::MetadataSet::GetMetadataOffsets(metaptr, p_meta->LongLength, offsets, p_num + 1, '\n')) return false;
                std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(SPTAG::ByteArray(metaptr, p_meta->LongLength, false), 
                    SPTAG::ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), p_num, 
                    (*m_Instance)->m_iDataBlockSize, (*m_Instance)->m_iDataCapacity, (*m_Instance)->m_iMetaRecordSize));
                return (SPTAG::ErrorCode::Success == (*m_Instance)->BuildIndex(vectors, meta, p_withMetaIndex, p_normalized));
            }

            array<BasicResult^>^ AnnIndex::Search(array<Byte>^ p_data, int p_resultNum)
            {
                array<BasicResult^>^ res;
                if (m_Instance == nullptr)
                    return res;

                pin_ptr<Byte> ptr = &p_data[0];
                SPTAG::QueryResult results(ptr, p_resultNum, false);
                (*m_Instance)->SearchIndex(results);

                res = gcnew array<BasicResult^>(p_resultNum);
                for (int i = 0; i < p_resultNum; i++)
                    res[i] = gcnew BasicResult(new SPTAG::BasicResult(*(results.GetResult(i))));

                return res;
            }

            array<BasicResult^>^ AnnIndex::SearchWithMetaData(array<Byte>^ p_data, int p_resultNum)
            {
                array<BasicResult^>^ res;
                if (m_Instance == nullptr)
                    return res;

                pin_ptr<Byte> ptr = &p_data[0];
                SPTAG::QueryResult results(ptr, p_resultNum, true);
                (*m_Instance)->SearchIndex(results);

                res = gcnew array<BasicResult^>(p_resultNum);
                for (int i = 0; i < p_resultNum; i++)
                    res[i] = gcnew BasicResult(new SPTAG::BasicResult(*(results.GetResult(i))));

                return res;
            }

            RIterator^ AnnIndex::GetIterator(array<Byte>^ p_data)
            {
                RIterator^ res;
                if (m_Instance == nullptr || m_dimension == 0 || p_data->LongLength != m_inputVectorSize)
                    return res;

                pin_ptr<Byte> ptr = &p_data[0];
                std::shared_ptr<ResultIterator> result_iterator = (*m_Instance)->GetIterator(ptr);

                res = gcnew RIterator(result_iterator);
                return res;
            }

            void AnnIndex::UpdateIndex()
            {
                if (m_Instance != nullptr) (*m_Instance)->UpdateIndex();
            }

            bool AnnIndex::Save(String^ p_saveFile)
            {
                return SPTAG::ErrorCode::Success == (*m_Instance)->SaveIndex(string_to_char_array(p_saveFile));
            }

            array<array<Byte>^>^ AnnIndex::Dump()
            {
                std::shared_ptr<std::vector<std::uint64_t>> buffersize = (*m_Instance)->CalculateBufferSize();
                array<array<Byte>^>^ res = gcnew array<array<Byte>^>(buffersize->size() + 1);
                std::vector<SPTAG::ByteArray> indexBlobs;
                for (int i = 1; i < res->Length; i++)
                {
                    res[i] = gcnew array<Byte>(buffersize->at(i-1));
                    pin_ptr<Byte> ptr = &res[i][0];
                    indexBlobs.push_back(SPTAG::ByteArray((std::uint8_t*)ptr, res[i]->LongLength, false));
                }
                std::string config;
                if (SPTAG::ErrorCode::Success != (*m_Instance)->SaveIndex(config, indexBlobs)) 
                {
                    array<array<Byte>^>^ null;
                    return null;
                }
                res[0] = gcnew array<Byte>(config.size());
                Marshal::Copy(IntPtr(&config[0]), res[0], 0, config.size());
                return res;
            }

            bool AnnIndex::Add(array<Byte>^ p_data, int p_num)
            {
                return Add(p_data, p_num, false);
            }


            bool AnnIndex::Add(array<Byte>^ p_data, int p_num, bool p_normalized)
            {
                if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
                    return false;

                pin_ptr<Byte> ptr = &p_data[0];
                return (SPTAG::ErrorCode::Success == (*m_Instance)->AddIndex(ptr, p_num, m_dimension, nullptr, false, p_normalized));
            }


            bool AnnIndex::AddWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex)
            {
                return AddWithMetaData(p_data, p_meta, p_num, p_withMetaIndex, false);
            }

            bool AnnIndex::AddWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex, bool p_normalized)
            {
                if (m_Instance == nullptr || p_num == 0 || m_dimension == 0 || p_data->LongLength != p_num * m_inputVectorSize)
                    return false;

                pin_ptr<Byte> dataptr = &p_data[0];
                std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(SPTAG::ByteArray(dataptr, p_data->LongLength, false), (*m_Instance)->GetVectorValueType(), m_dimension, p_num));

                pin_ptr<Byte> metaptr = &p_meta[0];
                std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
                if (!SPTAG::MetadataSet::GetMetadataOffsets(metaptr, p_meta->LongLength, offsets, p_num + 1, '\n')) return false;
                std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(SPTAG::ByteArray(metaptr, p_meta->LongLength, false), SPTAG::ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), p_num));
                return (SPTAG::ErrorCode::Success == (*m_Instance)->AddIndex(vectors, meta, p_withMetaIndex, p_normalized));
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

            AnnIndex^ AnnIndex::Load(array<array<Byte>^>^ p_index)
            {
                std::vector<SPTAG::ByteArray> p_indexBlobs;
                for (int i = 1; i < p_index->Length; i++)
                {
                    pin_ptr<Byte> ptr = &p_index[i][0];
                    p_indexBlobs.push_back(SPTAG::ByteArray((std::uint8_t*)ptr, p_index[i]->LongLength, false));
                }
                pin_ptr<Byte> configptr = &p_index[0][0];

                std::shared_ptr<SPTAG::VectorIndex> vecIndex;
                if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(std::string((char*)configptr, p_index[0]->LongLength), p_indexBlobs, vecIndex) || nullptr == vecIndex)
                {
                    return gcnew AnnIndex(nullptr);
                }
                return gcnew AnnIndex(vecIndex);
            }

            AnnIndex^ AnnIndex::Merge(String^ p_indexFilePath1, String^ p_indexFilePath2)
            {
                AnnIndex^ res = Load(p_indexFilePath1);
                AnnIndex^ add = Load(p_indexFilePath2);
                if (*(res->m_Instance) == nullptr || *(add->m_Instance) == nullptr || 
                    SPTAG::ErrorCode::Success != (*(res->m_Instance))->MergeIndex(add->m_Instance->get(), std::atoi((*(res->m_Instance))->GetParameter("NumberOfThreads").c_str()), nullptr))
                {
                    return gcnew AnnIndex(nullptr);
                }
                return res;
            }

            MultiIndexScan::MultiIndexScan(std::shared_ptr<SPTAG::MultiIndexScan> multi_index_scan) :
                ManagedObject(multi_index_scan)
            {
            }

            MultiIndexScan::MultiIndexScan(array<AnnIndex^>^ indice, array<array<Byte>^>^ p_data, array<float>^ weight, int p_resultNum,
                bool useTimer, int termCondVal, int searchLimit) : ManagedObject(std::make_shared<SPTAG::MultiIndexScan>())
            {
                std::vector<std::shared_ptr<SPTAG::VectorIndex>> vecIndices;
                for (int i = 0; i < indice->Length; i++)
                {
                    std::shared_ptr<SPTAG::VectorIndex> index = *(indice[i]->GetInstance());
                    vecIndices.push_back(index);
                }

                std::vector<SPTAG::ByteArray> data_array;
                for (int i = 0; i < p_data->Length; i++)
                {
                    pin_ptr<Byte> ptr = &p_data[i][0];
                    SPTAG::ByteArray byte_target = SPTAG::ByteArray::Alloc(p_data[i]->LongLength);
                    byte_target.Set((std::uint8_t*)ptr, p_data[i]->LongLength, false);
                    data_array.push_back(byte_target);
                }

                std::vector<float> weight_array;
                for (int i = 0; i < weight->Length; i++)
                {
                    float w = weight[i];
                    weight_array.push_back(w);
                }

                (*m_Instance)->Init(vecIndices, data_array, weight_array, p_resultNum, useTimer, termCondVal, searchLimit);
            }

            BasicResult^ MultiIndexScan::Next()
            {
                SPTAG::BasicResult* result = new SPTAG::BasicResult();
                (*m_Instance)->Next(*result);
                return gcnew BasicResult(result);
            }

            void MultiIndexScan::Close()
            {
                (*m_Instance)->Close();
            }

        }
    }
}