// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_VECTORINDEX_H_
#define _SPTAG_VECTORINDEX_H_

#include "Common.h"
#include "Common/WorkSpace.h"
#include "SearchQuery.h"
#include "VectorSet.h"
#include "MetadataSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include <unordered_set>
#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Common/WorkSpace.h"

class ResultIterator;

namespace SPTAG
{

class IAbortOperation
{
public:
    virtual bool ShouldAbort() = 0;
};

class VectorIndex
{
public:
    VectorIndex();

    virtual ~VectorIndex();

    virtual ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized = false, bool p_shareOwnership = false) = 0;
    
    virtual ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false) = 0;
    virtual ErrorCode AddIndexId(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, int& beginHead, int& endHead) { return ErrorCode::Undefined; }
    virtual ErrorCode AddIndexIdx(SizeType begin, SizeType end) { return ErrorCode::Undefined; }


    virtual ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum) = 0;

    virtual ErrorCode SearchIndex(QueryResult& p_results, bool p_searchDeleted = false) const = 0;
    
    virtual std::shared_ptr<ResultIterator> GetIterator(const void* p_target, bool p_searchDeleted = false) const = 0;

    virtual ErrorCode SearchIndexIterativeNext(QueryResult& p_query, COMMON::WorkSpace* workSpace, int p_batch, int& resultCount, bool p_isFirst, bool p_searchDeleted) const = 0;

    virtual ErrorCode SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> workSpace) const = 0;

    virtual bool SearchIndexIterativeFromNeareast(QueryResult& p_query, COMMON::WorkSpace* p_space, bool p_isFirst, bool p_searchDeleted = false) const = 0;

    virtual std::unique_ptr<COMMON::WorkSpace> RentWorkSpace(int batch) const = 0;

    virtual ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const = 0;

    virtual ErrorCode SearchIndexWithFilter(QueryResult& p_query, std::function<bool(const ByteArray&)> filterFunc, int maxCheck = 0, bool p_searchDeleted = false) const = 0;

    virtual ErrorCode SearchTree(QueryResult &p_query) const = 0;

    virtual ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) = 0;

    virtual float AccurateDistance(const void* pX, const void* pY) const = 0;
    virtual float ComputeDistance(const void* pX, const void* pY) const = 0;
    virtual float GetDistance(const void* target, const SizeType idx) const = 0;
    virtual const void* GetSample(const SizeType idx) const = 0;
    virtual bool ContainSample(const SizeType idx) const = 0;
    virtual bool NeedRefine() const = 0;
   
    virtual DimensionType GetFeatureDim() const = 0;
    virtual SizeType GetNumSamples() const = 0;
    virtual SizeType GetNumDeleted() const = 0;

    virtual DistCalcMethod GetDistCalcMethod() const = 0;
    virtual IndexAlgoType GetIndexAlgoType() const = 0;
    virtual VectorValueType GetVectorValueType() const = 0;

    virtual std::string GetParameter(const char* p_param, const char* p_section = nullptr) const = 0;
    virtual ErrorCode SetParameter(const char* p_param, const char* p_value, const char* p_section = nullptr) = 0;
    virtual ErrorCode UpdateIndex() = 0;

    virtual bool IsReady() const { return m_bReady; }
    virtual void SetReady(bool p_ready) { m_bReady = p_ready; }

    virtual std::shared_ptr<std::vector<std::uint64_t>> CalculateBufferSize() const;

    virtual ErrorCode SaveIndex(std::string& p_config, const std::vector<ByteArray>& p_indexBlobs);

    virtual ErrorCode SaveIndex(const std::string& p_folderPath);

    virtual ErrorCode SaveIndexToFile(const std::string& p_file, IAbortOperation* p_abort = nullptr);

    virtual ErrorCode BuildIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false, bool p_shareOwnership = false);
    
    virtual ErrorCode BuildIndex(bool p_normalized = false) { return ErrorCode::Undefined; }

    virtual ErrorCode AddIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false, bool p_normalized = false);

    virtual ErrorCode DeleteIndex(ByteArray p_meta);

    virtual ErrorCode MergeIndex(VectorIndex* p_addindex, int p_threadnum, IAbortOperation* p_abort);
    
    virtual const void* GetSample(ByteArray p_meta, bool& deleteFlag);

    virtual ErrorCode SearchIndex(const void* p_vector, int p_vectorCount, int p_neighborCount, bool p_withMeta, BasicResult* p_results) const;

    virtual void ApproximateRNG(std::shared_ptr<VectorSet>& fullVectors, std::unordered_set<SizeType>& exceptIDS, int candidateNum, Edge* selections, int replicaCount, int numThreads, int numTrees, int leafSize, float RNGFactor, int numGPUs);

    static void SortSelections(std::vector<Edge>* selections);

    virtual std::string GetParameter(const std::string& p_param, const std::string& p_section = "Index") const;
    virtual ErrorCode SetParameter(const std::string& p_param, const std::string& p_value, const std::string& p_section = "Index");

    virtual ByteArray GetMetadata(SizeType p_vectorID) const;
    virtual MetadataSet* GetMetadata() const;
    virtual void SetMetadata(MetadataSet* p_new);

    virtual std::string GetIndexName() const 
    { 
        if (m_sIndexName == "") return Helper::Convert::ConvertToString(GetIndexAlgoType());
        return m_sIndexName; 
    }
    virtual void SetIndexName(std::string p_name) { m_sIndexName = p_name; }

    virtual void SetQuantizerFileName(std::string p_QuantizerFileName) { m_sQuantizerFile = p_QuantizerFileName; }

    virtual void SetQuantizerADC(bool enableADC) {
        if (m_pQuantizer) m_pQuantizer->SetEnableADC(enableADC);
    }

    virtual void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer) = 0;

    virtual ErrorCode LoadQuantizer(std::string p_quantizerFile);

    virtual std::shared_ptr<SPTAG::COMMON::IQuantizer> GetQuantizer() {
        return m_pQuantizer;
    }

    virtual ErrorCode QuantizeVector(const void* p_data, SizeType p_num, ByteArray p_out) {
        if (m_pQuantizer != nullptr && p_out.Length() >= m_pQuantizer->GetNumSubvectors() * (size_t)p_num) {
            for (int i = 0; i < p_num; i++) 
                m_pQuantizer->QuantizeVector(((std::uint8_t*)p_data) + i * (size_t)(m_pQuantizer->ReconstructSize()), p_out.Data() + i * (size_t)(m_pQuantizer->GetNumSubvectors()), false);
            return ErrorCode::Success;
        }
        return ErrorCode::Fail;
    }

    virtual ErrorCode ReconstructVector(const void* p_data, SizeType p_num, ByteArray p_out) {
        if (m_pQuantizer != nullptr && p_out.Length() >= m_pQuantizer->ReconstructSize() * (size_t)p_num) {
            for (int i = 0; i < p_num; i++)
                m_pQuantizer->ReconstructVector(((std::uint8_t*)p_data) + i * (size_t)(m_pQuantizer->GetNumSubvectors()), p_out.Data() + i * (size_t)(m_pQuantizer->ReconstructSize()));
            return ErrorCode::Success;
        }
        return ErrorCode::Fail;
    }

    static std::shared_ptr<VectorIndex> CreateInstance(IndexAlgoType p_algo, VectorValueType p_valuetype);

    static ErrorCode LoadIndex(const std::string& p_loaderFilePath, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static ErrorCode LoadIndexFromFile(const std::string& p_file, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static ErrorCode LoadIndex(const std::string& p_config, const std::vector<ByteArray>& p_indexBlobs, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static std::uint64_t EstimatedVectorCount(std::uint64_t p_memory, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_vectorsInBlock, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize);

    static std::uint64_t EstimatedMemoryUsage(std::uint64_t p_vectorCount, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_vectorsInBlock, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize);

    virtual std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const = 0;

    virtual std::shared_ptr<std::vector<std::string>> GetIndexFiles() const = 0;

    virtual ErrorCode SaveConfig(std::shared_ptr<Helper::DiskIO> p_configout) = 0;

    virtual ErrorCode SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams) = 0;

    virtual ErrorCode LoadConfig(Helper::IniReader& p_reader) = 0;

    virtual ErrorCode LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams) = 0;

    virtual ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs) = 0;

    virtual ErrorCode DeleteIndex(const SizeType& p_id) = 0;

    virtual ErrorCode RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>>& p_indexStreams, IAbortOperation* p_abort) = 0;

    virtual ErrorCode SetWorkSpaceFactory(std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>> up_workSpaceFactory) = 0;

    inline bool HasMetaMapping() const { return nullptr != m_pMetaToVec; }

    inline SizeType GetMetaMapping(std::string& meta) const;

    void UpdateMetaMapping(const std::string& meta, SizeType i);

    void BuildMetaMapping(bool p_checkDeleted = true);

private:
    ErrorCode LoadIndexConfig(Helper::IniReader& p_reader);

    ErrorCode SaveIndexConfig(std::shared_ptr<Helper::DiskIO> p_configOut);

protected:
    bool m_bReady = false;
    std::string m_sIndexName = "";
    std::string m_sMetadataFile = "metadata.bin";
    std::string m_sMetadataIndexFile = "metadataIndex.bin";
    std::string m_sQuantizerFile = "quantizer.bin";
    std::shared_ptr<MetadataSet> m_pMetadata;
    std::shared_ptr<void> m_pMetaToVec;

public:
    int m_iDataBlockSize = 1024 * 1024;
    int m_iDataCapacity = MaxSize;
    int m_iMetaRecordSize = 10;
    std::shared_ptr<SPTAG::COMMON::IQuantizer> m_pQuantizer = nullptr;
};


} // namespace SPTAG

#endif // _SPTAG_VECTORINDEX_H_
