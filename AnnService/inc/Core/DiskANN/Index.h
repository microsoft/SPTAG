// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_DISKANN_INDEX_H_
#define _SPTAG_DISKANN_INDEX_H_

#include "../Common.h"
#include "../VectorIndex.h"
#include "../Common/CommonUtils.h"
#include "../Common/DistanceUtils.h"

#include "DiskANN/include/index.h"

namespace SPTAG
{

    namespace Helper
    {
        class IniReader;
    }

    namespace DiskANN
    {
        template<typename T>
        class Index : public VectorIndex
        {
        private:
            std::shared_ptr<diskann::Index<T>> m_index;

            unsigned R;
            unsigned L;
            unsigned C;
            float alpha;
            bool saturate_graph;

            std::string m_sGraphFilename;
            std::string m_sDataPointsFilename;

            unsigned m_iNumberOfThreads;
            DistCalcMethod m_distCalcMethod;
            float(*m_fComputeDistance)(const T* pX, const T* pY, DimensionType length);
            int m_iBaseSquare;

        public:
            Index()
            {
#define DefineDiskANNParameter(VarName, VarType, DefaultValue, RepresentStr) \
                VarName = DefaultValue; \

#include "inc/Core/DiskANN/ParameterDefinitionList.h"
#undef DefineDiskANNParameter

                m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_distCalcMethod);
                m_iBaseSquare = (m_distCalcMethod == DistCalcMethod::Cosine) ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() : 1;
            }

            ~Index() {}

            inline SizeType GetNumSamples() const { return (SizeType)m_index->get_vector_number(); }
            inline DimensionType GetFeatureDim() const { return (DimensionType)m_index->get_dimension(); }
            inline int GetNumThreads() const { return m_iNumberOfThreads; }
            inline DistCalcMethod GetDistCalcMethod() const { return m_distCalcMethod; }
            inline IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::DISKANN; }
            inline VectorValueType GetVectorValueType() const { return GetEnumValueType<T>(); }

            inline float AccurateDistance(const void* pX, const void* pY) const {
                if (m_distCalcMethod == DistCalcMethod::L2) return m_fComputeDistance((const T*)pX, (const T*)pY, GetFeatureDim());

                float xy = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pY, GetFeatureDim());
                float xx = m_iBaseSquare - m_fComputeDistance((const T*)pX, (const T*)pX, GetFeatureDim());
                float yy = m_iBaseSquare - m_fComputeDistance((const T*)pY, (const T*)pY, GetFeatureDim());
                return 1.0f - xy / (sqrt(xx) * sqrt(yy));
            }
            inline float ComputeDistance(const void* pX, const void* pY) const { return m_fComputeDistance((const T*)pX, (const T*)pY, GetFeatureDim()); }
            inline bool ContainSample(const SizeType idx) const { return idx < GetNumSamples(); }

            std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const
            {
                std::shared_ptr<std::vector<std::uint64_t>> buffersize(new std::vector<std::uint64_t>);
                buffersize->push_back(sizeof(T) * GetNumSamples() * GetFeatureDim());
                buffersize->push_back(0); // GraphSize
                return std::move(buffersize);
            }

            std::shared_ptr<std::vector<std::string>> GetIndexFiles() const
            {
                std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
                files->push_back(m_sDataPointsFilename);
                files->push_back(m_sGraphFilename);
                return std::move(files);
            }

            ErrorCode SaveConfig(std::shared_ptr<Helper::DiskPriorityIO> p_configout); // *
            ErrorCode SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskPriorityIO>>& p_indexStreams); // *

            ErrorCode LoadConfig(Helper::IniReader& p_reader); // *
            ErrorCode LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskPriorityIO>>& p_indexStreams); // *
            ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs) { return ErrorCode::Undefined; }

            ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized = false); // *
            ErrorCode SearchIndex(QueryResult& p_query, bool p_searchDeleted = false) const; // *
            ErrorCode UpdateIndex();

            ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr); // *
            std::string GetParameter(const char* p_param, const char* p_section = nullptr) const; // *

            inline const void* GetSample(const SizeType idx) const { return nullptr; }
            inline SizeType GetNumDeleted() const { return 0; }
            inline bool NeedRefine() const { return false; }

            ErrorCode RefineSearchIndex(QueryResult& p_query, bool p_searchDeleted = false) const { return ErrorCode::Undefined; }
            ErrorCode SearchTree(QueryResult& p_query) const { return ErrorCode::Undefined; }
            ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false) { return ErrorCode::Undefined; }
            ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum) { return ErrorCode::Undefined; }
            ErrorCode DeleteIndex(const SizeType& p_id) { return ErrorCode::Undefined; }
            ErrorCode RefineIndex(const std::vector<std::shared_ptr<Helper::DiskPriorityIO>>& p_indexStreams, IAbortOperation* p_abort) { return ErrorCode::Undefined; }
            ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) { return ErrorCode::Undefined; }
        };
    } // namespace DiskANN
} // namespace SPTAG
#endif // _SPTAG_DISKANN_INDEX_H_