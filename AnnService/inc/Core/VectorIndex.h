// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_VECTORINDEX_H_
#define _SPTAG_VECTORINDEX_H_

#include "Common.h"
#include "SearchQuery.h"
#include "VectorSet.h"
#include "MetadataSet.h"
#include "inc/Helper/SimpleIniReader.h"

namespace SPTAG
{
class VectorIndex
{
public:
    VectorIndex();

    virtual ~VectorIndex();

    virtual ErrorCode BuildIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension) = 0;
    
    virtual ErrorCode AddIndex(const void* p_data, SizeType p_vectorNum, DimensionType p_dimension, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false) = 0;

    virtual ErrorCode DeleteIndex(const void* p_vectors, SizeType p_vectorNum) = 0;

    virtual ErrorCode SearchIndex(QueryResult& p_results, bool p_searchDeleted = false) const = 0;
    
    virtual ErrorCode RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted = false) const = 0;

    virtual ErrorCode SearchTree(QueryResult &p_query) const = 0;

    virtual ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_newIndex) = 0;

    virtual float AccurateDistance(const void* pX, const void* pY) const = 0;
    virtual float ComputeDistance(const void* pX, const void* pY) const = 0;
    virtual const void* GetSample(const SizeType idx) const = 0;
    virtual bool ContainSample(const SizeType idx) const = 0;
    virtual bool NeedRefine() const = 0;
   
    virtual DimensionType GetFeatureDim() const = 0;
    virtual SizeType GetNumSamples() const = 0;
    virtual SizeType GetNumDeleted() const = 0;

    virtual DistCalcMethod GetDistCalcMethod() const = 0;
    virtual IndexAlgoType GetIndexAlgoType() const = 0;
    virtual VectorValueType GetVectorValueType() const = 0;

    virtual std::string GetParameter(const char* p_param) const = 0;
    virtual ErrorCode SetParameter(const char* p_param, const char* p_value) = 0;

    virtual bool IsReady() const { return m_bReady; }

    virtual std::shared_ptr<std::vector<std::uint64_t>> CalculateBufferSize() const;

    virtual ErrorCode SaveIndex(std::string& p_config, const std::vector<ByteArray>& p_indexBlobs);

    virtual ErrorCode SaveIndex(const std::string& p_folderPath);

    virtual ErrorCode SaveIndexToFile(const std::string& p_file);

    virtual ErrorCode BuildIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false);
    
    virtual ErrorCode AddIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex = false);

    virtual ErrorCode DeleteIndex(ByteArray p_meta);

    virtual ErrorCode MergeIndex(VectorIndex* p_addindex, int p_threadnum);
    
    virtual const void* GetSample(ByteArray p_meta, bool& deleteFlag);

    virtual ErrorCode SearchIndex(const void* p_vector, int p_vectorCount, int p_neighborCount, bool p_withMeta, BasicResult* p_results) const;

    virtual std::string GetParameter(const std::string& p_param) const;
    virtual ErrorCode SetParameter(const std::string& p_param, const std::string& p_value);

    virtual ByteArray GetMetadata(SizeType p_vectorID) const;
    virtual MetadataSet* GetMetadata() const;
    virtual void SetMetadata(MetadataSet* p_new);

    virtual std::string GetIndexName() const 
    { 
        if (m_sIndexName == "") return Helper::Convert::ConvertToString(GetIndexAlgoType());
        return m_sIndexName; 
    }
    virtual void SetIndexName(std::string p_name) { m_sIndexName = p_name; }

    static std::shared_ptr<VectorIndex> CreateInstance(IndexAlgoType p_algo, VectorValueType p_valuetype);

    static ErrorCode LoadIndex(const std::string& p_loaderFilePath, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static ErrorCode LoadIndexFromFile(const std::string& p_file, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static ErrorCode LoadIndex(const std::string& p_config, const std::vector<ByteArray>& p_indexBlobs, std::shared_ptr<VectorIndex>& p_vectorIndex);

    static std::uint64_t EstimatedVectorCount(std::uint64_t p_memory, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize);

    static std::uint64_t EstimatedMemoryUsage(std::uint64_t p_vectorCount, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize);

protected:
    virtual std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const = 0;

    virtual ErrorCode SaveConfig(std::ostream& p_configout) const = 0;

    virtual ErrorCode SaveIndexData(const std::string& p_folderPath) = 0;

    virtual ErrorCode SaveIndexData(const std::vector<std::ostream*>& p_indexStreams) = 0;

    virtual ErrorCode LoadConfig(Helper::IniReader& p_reader) = 0;

    virtual ErrorCode LoadIndexData(const std::string& p_folderPath) = 0;

    virtual ErrorCode LoadIndexData(const std::vector<std::istream*>& p_indexStreams) = 0;

    virtual ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray>& p_indexBlobs) = 0;

    virtual ErrorCode DeleteIndex(const SizeType& p_id) = 0;

    virtual ErrorCode RefineIndex(const std::string& p_folderPath) = 0;

    virtual ErrorCode RefineIndex(const std::vector<std::ostream*>& p_indexStreams) = 0;

    inline bool HasMetaMapping() const { return nullptr != m_pMetaToVec; }

    inline SizeType GetMetaMapping(std::string& meta) const;

    void UpdateMetaMapping(std::string& meta, SizeType i);

    void BuildMetaMapping(bool p_checkDeleted = true);

private:
    ErrorCode LoadIndexConfig(Helper::IniReader& p_reader);

    ErrorCode SaveIndexConfig(std::ostream& p_configOut);

protected:
    bool m_bReady = false;
    std::string m_sIndexName = "";
    std::string m_sMetadataFile = "metadata.bin";
    std::string m_sMetadataIndexFile = "metadataIndex.bin";
    std::shared_ptr<MetadataSet> m_pMetadata;
    std::shared_ptr<void> m_pMetaToVec;
};


} // namespace SPTAG

#endif // _SPTAG_VECTORINDEX_H_
