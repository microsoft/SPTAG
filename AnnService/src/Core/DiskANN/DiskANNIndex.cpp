// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/DiskANN/Index.h"

#include "DiskANN/include/utils.h"
#include "DiskANN/include/memory_mapper.h"

#include <chrono>

#pragma warning(disable:4242)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4244)  // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable:4127)  // conditional expression is constant

namespace SPTAG
{
    namespace DiskANN
    {
        template <typename T>
        ErrorCode Index<T>::LoadConfig(Helper::IniReader& p_reader)
        {
#define DefineDiskANNParameter(VarName, VarType, DefaultValue, RepresentStr) \
            SetParameter(RepresentStr, \
                         p_reader.GetParameter("Index", \
                         RepresentStr, \
                         std::string(#DefaultValue)).c_str()); \

#include "inc/Core/DiskANN/ParameterDefinitionList.h"
#undef DefineDiskANNParameter
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskPriorityIO>>& p_indexStreams)
        {
            if (p_indexStreams.size() < 2) return ErrorCode::LackOfInputs;

            const std::size_t bufferSize = 1 << 30;
            std::unique_ptr<char[]> bufferHolder(new char[bufferSize]);
            char* buf = bufferHolder.get();

            std::shared_ptr<Helper::DiskPriorityIO> output = f_createIO();
            if (output == nullptr || !output->Initialize(m_sDataPointsFilename.c_str(), std::ios::binary | std::ios::out))
            {
                LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", m_sDataPointsFilename.c_str());
                return ErrorCode::FailedOpenFile;
            }

            std::uint64_t readSize;
            while ((readSize = p_indexStreams[0]->ReadBinary(bufferSize, bufferHolder.get()))) {
                if (output->WriteBinary(readSize, bufferHolder.get()) != readSize) {
                    return ErrorCode::DiskIOFail;
                }
            }
            output->ShutDown();

            diskann::Metric metric = (m_distCalcMethod == DistCalcMethod::Cosine) ? diskann::Metric::INNER_PRODUCT : diskann::Metric::FAST_L2;
            m_index.reset(new diskann::Index<T>(metric, m_sDataPointsFilename.c_str()));

            
            if (output == nullptr || !output->Initialize(m_sGraphFilename.c_str(), std::ios::binary | std::ios::out))
            {
                LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", m_sGraphFilename.c_str());
                return ErrorCode::FailedOpenFile;
            }

            while ((readSize = p_indexStreams[1]->ReadBinary(bufferSize, bufferHolder.get()))) {
                if (output->WriteBinary(readSize, bufferHolder.get()) != readSize) {
                    return ErrorCode::DiskIOFail;
                }
            }
            output->ShutDown();

            m_index->load(m_sGraphFilename.c_str());
            if (metric == diskann::Metric::FAST_L2) m_index->optimize_graph();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskPriorityIO> p_configOut)
        {
#define DefineDiskANNParameter(VarName, VarType, DefaultValue, RepresentStr) \
    IOSTRING(p_configOut, WriteString, (RepresentStr + std::string("=") + GetParameter(RepresentStr) + std::string("\n")).c_str());

#include "inc/Core/DiskANN/ParameterDefinitionList.h"
#undef DefineDiskANNParameter

            IOSTRING(p_configOut, WriteString, "\n");
            return ErrorCode::Success;
        }

        template<typename T>
        ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskPriorityIO>>& p_indexStreams)
        {
            if (p_indexStreams.size() < 2) return ErrorCode::LackOfInputs;

            const std::size_t bufferSize = 1 << 30;
            std::unique_ptr<char[]> bufferHolder(new char[bufferSize]);
            char* buf = bufferHolder.get();

            std::shared_ptr<Helper::DiskPriorityIO> input = f_createIO();
            if (input == nullptr || !input->Initialize(m_sDataPointsFilename.c_str(), std::ios::binary | std::ios::in))
            {
                LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", m_sDataPointsFilename.c_str());
                return ErrorCode::FailedOpenFile;
            }

            std::uint64_t readSize;
            while ((readSize = input->ReadBinary(bufferSize, bufferHolder.get()))) {
                if (p_indexStreams[0]->WriteBinary(readSize, bufferHolder.get()) != readSize) {
                    return ErrorCode::DiskIOFail;
                }
            }
            input->ShutDown();
            remove(m_sDataPointsFilename.c_str());
            
            m_index->save(m_sGraphFilename.c_str());
            if (input == nullptr || !input->Initialize(m_sGraphFilename.c_str(), std::ios::binary | std::ios::in))
            {
                LOG(Helper::LogLevel::LL_Error, "Unable to open file: %s\n", m_sGraphFilename.c_str());
                return ErrorCode::FailedOpenFile;
            }

            while ((readSize = input->ReadBinary(bufferSize, bufferHolder.get()))) {
                if (p_indexStreams[1]->WriteBinary(readSize, bufferHolder.get()) != readSize) {
                    return ErrorCode::DiskIOFail;
                }
            }
            input->ShutDown();
            remove(m_sGraphFilename.c_str());
            return ErrorCode::Success;
        }

#pragma region K-NN search

        template<typename T>
        ErrorCode Index<T>::SearchIndex(QueryResult& p_query, bool p_searchDeleted) const
        {
            if (!m_bReady) return ErrorCode::EmptyIndex;

            diskann::Parameters paras;

            std::vector<uint32_t> query_result_ids(p_query.GetResultNum());
            std::vector<float>    query_result_dists(p_query.GetResultNum());

            if (m_distCalcMethod == DistCalcMethod::L2) {
                m_index->search_with_opt_graph((const T*)p_query.GetTarget(), p_query.GetResultNum(), L, query_result_ids.data(), query_result_dists.data());
            }
            else {
                m_index->search((const T*)p_query.GetTarget(), p_query.GetResultNum(), L, query_result_ids.data(), query_result_dists.data());
            }

            for (int i = 0; i < p_query.GetResultNum(); i++) {
                auto res = p_query.GetResult(i);
                res->VID = query_result_ids[i];
                res->Dist = query_result_dists[i];
            }

            if (p_query.WithMeta() && nullptr != m_pMetadata)
            {
                for (int i = 0; i < p_query.GetResultNum(); ++i)
                {
                    SizeType result = p_query.GetResult(i)->VID;
                    p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
                }
            }
            return ErrorCode::Success;
        }
#pragma endregion

        template <typename T>
        ErrorCode Index<T>::BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized)
        {
            if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0) return ErrorCode::EmptyData;

            std::shared_ptr<VectorSet> vectorSet(new BasicVectorSet(ByteArray((std::uint8_t*)p_data, p_vectorNum * p_dimension * sizeof(T), false),
                GetEnumValueType<T>(), p_dimension, p_vectorNum));
            vectorSet->Save(m_sDataPointsFilename);
            vectorSet.reset();

            diskann::Parameters paras;
            paras.Set<unsigned>("R", R);
            paras.Set<unsigned>("L", L);
            paras.Set<unsigned>("C", C);
            paras.Set<float>("alpha", alpha);
            paras.Set<bool>("saturate_graph", saturate_graph);
            paras.Set<unsigned>("num_threads", m_iNumberOfThreads);

            diskann::Metric metric = (m_distCalcMethod == DistCalcMethod::Cosine) ? diskann::Metric::INNER_PRODUCT : diskann::Metric::FAST_L2;
            m_index.reset(new diskann::Index<T>(metric, m_sDataPointsFilename.c_str()));
            auto s = std::chrono::high_resolution_clock::now();
            m_index->build(paras);
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
            LOG(Helper::LogLevel::LL_Info, "DiskANN Indexing time:%lld\n", diff.count());
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
            Index<T>::UpdateIndex()
        {
            omp_set_num_threads(m_iNumberOfThreads);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode
            Index<T>::SetParameter(const char* p_param, const char* p_value, const char* p_section)
        {
            if (nullptr == p_param || nullptr == p_value) return ErrorCode::Fail;

#define DefineDiskANNParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value); \
        VarType tmp; \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp)) \
        { \
            VarName = tmp; \
        } \
    } \

#include "inc/Core/DiskANN/ParameterDefinitionList.h"
#undef DefineDiskANNParameter

            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod")) {
                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_distCalcMethod);
                m_iBaseSquare = (m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
            }
            return ErrorCode::Success;
        }


        template <typename T>
        std::string
            Index<T>::GetParameter(const char* p_param, const char* p_section) const
        {
            if (nullptr == p_param) return std::string();

#define DefineDiskANNParameter(VarName, VarType, DefaultValue, RepresentStr) \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr)) \
    { \
        return SPTAG::Helper::Convert::ConvertToString(VarName); \
    } \

#include "inc/Core/DiskANN/ParameterDefinitionList.h"
#undef DefineDiskANNParameter

            return std::string();
        }
    }
}

#define DefineVectorValueType(Name, Type) \
template class SPTAG::DiskANN::Index<Type>; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType


