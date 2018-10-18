#include "inc/Core/VectorIndex.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/SimpleIniReader.h"

#include "inc/Core/BKT/Index.h"

#include <fstream>


using namespace SPTAG;


VectorIndex::VectorIndex()
{
}


VectorIndex::~VectorIndex()
{
}


void
VectorIndex::SetIndexName(const std::string& p_indexName)
{
    m_indexName = p_indexName;
}


const std::string&
VectorIndex::GetIndexName() const
{
    return m_indexName;
}


std::string 
VectorIndex::GetParameter(const std::string& p_param) const
{
    return GetParameter(p_param.c_str());
}


ErrorCode
VectorIndex::SetParameter(const std::string& p_param, const std::string& p_value)
{
    return SetParameter(p_param.c_str(), p_value.c_str());
}


std::shared_ptr<VectorIndex>
VectorIndex::CreateInstance(IndexAlgoType p_algo, VectorValueType p_valuetype)
{
    if (IndexAlgoType::Undefined == p_algo || VectorValueType::Undefined == p_valuetype)
    {
        return nullptr;
    }

    if (p_algo == IndexAlgoType::BKT) {
        switch (p_valuetype)
        {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        return std::shared_ptr<VectorIndex>(new BKT::Index<Type>); \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default: break;
        }
    }
    return nullptr;
}


ErrorCode
VectorIndex::LoadIndex(const std::string& p_loaderFilePath, std::shared_ptr<VectorIndex>& p_vectorIndex)
{
    Helper::IniReader iniReader;

    if (ErrorCode::Success != iniReader.LoadIniFile(p_loaderFilePath + "/indexloader.ini"))
    {
        return ErrorCode::FailedOpenFile;
    }

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);
    if (IndexAlgoType::Undefined == algoType || VectorValueType::Undefined == valueType)
    {
        return ErrorCode::Fail;
    }

    if (algoType == IndexAlgoType::BKT) {
        switch (valueType)
        {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        p_vectorIndex.reset(new BKT::Index<Type>); \
        p_vectorIndex->LoadIndex(p_loaderFilePath, iniReader); \
        break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default: break;
        }
    }
    return ErrorCode::Success;
}
