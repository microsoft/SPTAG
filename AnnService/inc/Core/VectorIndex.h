#ifndef _SPTAG_VECTORINDEX_H_
#define _SPTAG_VECTORINDEX_H_

#include "Common.h"
#include "SearchQuery.h"
#include "VectorSet.h"
#include "MetadataSet.h"

#include <memory>
#include <string>
#include <vector>

namespace SPTAG
{

namespace Helper
{
class IniReader;
}


class VectorIndex
{
public:
    VectorIndex();

    virtual ~VectorIndex();

    virtual ErrorCode SaveIndex(const std::string& p_folderPath) = 0;

    virtual ErrorCode LoadIndex(const std::string& p_folderPath, const Helper::IniReader& p_configReader) = 0;

    virtual ErrorCode BuildIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet) = 0;

    virtual ErrorCode SearchIndex(QueryResult& p_results) const = 0;

    virtual std::string GetParameter(const char* p_param) const = 0;
    virtual ErrorCode SetParameter(const char* p_param, const char* p_value) = 0;

    virtual std::string GetParameter(const std::string& p_param) const;
    virtual ErrorCode SetParameter(const std::string& p_param, const std::string& p_value);

    virtual ByteArray GetMetadata(IndexType p_vectorID) const = 0;

    virtual VectorValueType AcceptableQueryValueType() const = 0;

    virtual int GetFeatureDim() const = 0;

    virtual IndexAlgoType GetIndexAlgoType() const = 0;

    void SetIndexName(const std::string& p_indexName);

    const std::string& GetIndexName() const;

    static std::shared_ptr<VectorIndex> CreateInstance(IndexAlgoType p_algo, VectorValueType p_valuetype);

    static ErrorCode LoadIndex(const std::string& p_loaderFilePath, std::shared_ptr<VectorIndex>& p_vectorIndex);

private:
    std::string m_indexName;
};


} // namespace SPTAG

#endif // _SPTAG_VECTORINDEX_H_
