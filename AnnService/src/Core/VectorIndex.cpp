// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/DataUtils.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/SimpleIniReader.h"

#include "inc/Core/BKT/Index.h"
#include "inc/Core/KDT/Index.h"
#include <fstream>


using namespace SPTAG;


VectorIndex::VectorIndex()
{
}


VectorIndex::~VectorIndex()
{
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


void 
VectorIndex::SetMetadata(const std::string& p_metadataFilePath, const std::string& p_metadataIndexPath) {
    m_pMetadata.reset(new FileMetadataSet(p_metadataFilePath, p_metadataIndexPath));
}


ByteArray 
VectorIndex::GetMetadata(SizeType p_vectorID) const {
    if (nullptr != m_pMetadata)
    {
        return m_pMetadata->GetMetadata(p_vectorID);
    }
    return ByteArray::c_empty;
}


ErrorCode 
VectorIndex::LoadIndex(const std::string& p_folderPath)
{
    std::string folderPath(p_folderPath);
    if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
    {
        folderPath += FolderSep;
    }

    Helper::IniReader p_configReader;
    if (ErrorCode::Success != p_configReader.LoadIniFile(folderPath + "/indexloader.ini"))
    {
        return ErrorCode::FailedOpenFile;
    }

    std::string metadataSection("MetaData");
    if (p_configReader.DoesSectionExist(metadataSection))
    {
        std::string metadataFilePath = p_configReader.GetParameter(metadataSection,
            "MetaDataFilePath",
            std::string());
        std::string metadataIndexFilePath = p_configReader.GetParameter(metadataSection,
            "MetaDataIndexPath",
            std::string());

        m_pMetadata.reset(new FileMetadataSet(folderPath + metadataFilePath, folderPath + metadataIndexFilePath));

        if (!m_pMetadata->Available())
        {
            std::cerr << "Error: Failed to load metadata." << std::endl;
            return ErrorCode::Fail;
        }

        std::string metadataToVectorIndex = p_configReader.GetParameter(metadataSection,
            "MetaDataToVectorIndex",
            std::string());

        if (metadataToVectorIndex == "true")
        {
            m_pMetaToVec.reset(new std::unordered_map<std::string, SizeType>);
            for (SizeType i = 0; i < m_pMetadata->Count(); i++) {
                ByteArray meta = m_pMetadata->GetMetadata(i);
                m_pMetaToVec->emplace(std::string((char*)meta.Data(), meta.Length()), i);
            }
        }
    }
    if (DistCalcMethod::Undefined == p_configReader.GetParameter("Index", "DistCalcMethod", DistCalcMethod::Undefined))
    {
        std::cerr << "Error: Failed to load parameter DistCalcMethod." << std::endl;
        return ErrorCode::Fail;
    }

    return LoadIndex(folderPath, p_configReader);
}


ErrorCode VectorIndex::SaveIndex(const std::string& p_folderPath)
{
    std::string folderPath(p_folderPath);
    if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
    {
        folderPath += FolderSep;
    }

    if (!direxists(folderPath.c_str()))
    {
        mkdir(folderPath.c_str());
    }

    std::string loaderFilePath = folderPath + "indexloader.ini";

    std::ofstream loaderFile(loaderFilePath);
    if (!loaderFile.is_open())
    {
        return ErrorCode::FailedCreateFile;
    }

    if (nullptr != m_pMetadata)
    {
        std::string metadataFile = "metadata.bin";
        std::string metadataIndexFile = "metadataIndex.bin";
        loaderFile << "[MetaData]" << std::endl;
        loaderFile << "MetaDataFilePath=" << metadataFile << std::endl;
        loaderFile << "MetaDataIndexPath=" << metadataIndexFile << std::endl;
        if (nullptr != m_pMetaToVec) loaderFile << "MetaDataToVectorIndex=true" << std::endl;
        loaderFile << std::endl;

        m_pMetadata->SaveMetadata(folderPath + metadataFile, folderPath + metadataIndexFile);
    }

    loaderFile << "[Index]" << std::endl;
    loaderFile << "IndexAlgoType=" << Helper::Convert::ConvertToString(GetIndexAlgoType()) << std::endl;
    loaderFile << "ValueType=" << Helper::Convert::ConvertToString(GetVectorValueType()) << std::endl;
    loaderFile << std::endl;

    ErrorCode ret = SaveIndex(folderPath, loaderFile);
    loaderFile.close();
    return ret;
}

ErrorCode
VectorIndex::BuildIndex(std::shared_ptr<VectorSet> p_vectorSet,
    std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex)
{
    if (nullptr == p_vectorSet || p_vectorSet->Count() == 0 || p_vectorSet->Dimension() == 0 || p_vectorSet->GetValueType() != GetVectorValueType())
    {
        return ErrorCode::Fail;
    }

    BuildIndex(p_vectorSet->GetData(), p_vectorSet->Count(), p_vectorSet->Dimension());
    m_pMetadata = std::move(p_metadataSet);
    if (p_withMetaIndex && m_pMetadata != nullptr) {
        m_pMetaToVec.reset(new std::unordered_map<std::string, SizeType>);
        for (SizeType i = 0; i < p_vectorSet->Count(); i++) {
            ByteArray meta = m_pMetadata->GetMetadata(i);
            m_pMetaToVec->emplace(std::string((char*)meta.Data(), meta.Length()), i);
        }
    }
    return ErrorCode::Success;
}


ErrorCode
VectorIndex::SearchIndex(const void* p_vector, int p_neighborCount, bool p_withMeta, BasicResult* p_results) const {
    QueryResult res(p_vector, p_neighborCount, p_withMeta, p_results);
    SearchIndex(res);
    return ErrorCode::Success;
}


ErrorCode 
VectorIndex::AddIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet) {
    if (nullptr == p_vectorSet || p_vectorSet->Count() == 0 || p_vectorSet->Dimension() == 0 || p_vectorSet->GetValueType() != GetVectorValueType())
    {
        return ErrorCode::Fail;
    }

    SizeType currStart;
    ErrorCode ret = AddIndex(p_vectorSet->GetData(), p_vectorSet->Count(), p_vectorSet->Dimension(), &currStart);
    if (ret != ErrorCode::Success) return ret;

    if (m_pMetadata == nullptr) {
        if (currStart == 0)
            m_pMetadata = std::move(p_metadataSet);
        else
            return ErrorCode::Success;
    }
    else {
        m_pMetadata->AddBatch(*p_metadataSet);
    }
    
    if (m_pMetaToVec != nullptr) {
        for (SizeType i = 0; i < p_vectorSet->Count(); i++) {
            ByteArray meta = m_pMetadata->GetMetadata(currStart + i);
            DeleteIndex(meta);
            m_pMetaToVec->emplace(std::string((char*)meta.Data(), meta.Length()), currStart + i);
        }
    }
    return ErrorCode::Success;
}


ErrorCode
VectorIndex::DeleteIndex(ByteArray p_meta) {
    if (m_pMetaToVec == nullptr) return ErrorCode::Fail;

    std::string meta((char*)p_meta.Data(), p_meta.Length());
    auto iter = m_pMetaToVec->find(meta);
    if (iter != m_pMetaToVec->end()) DeleteIndex(iter->second);
    return ErrorCode::Success;
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
    else if (p_algo == IndexAlgoType::KDT) {
        switch (p_valuetype)
        {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        return std::shared_ptr<VectorIndex>(new KDT::Index<Type>); \

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
        p_vectorIndex->LoadIndex(p_loaderFilePath); \
        break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default: break;
        }
    }
    else if (algoType == IndexAlgoType::KDT) {
        switch (valueType)
        {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        p_vectorIndex.reset(new KDT::Index<Type>); \
        p_vectorIndex->LoadIndex(p_loaderFilePath); \
        break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default: break;
        }
    }
    return ErrorCode::Success;
}


ErrorCode
VectorIndex::MergeIndex(const char* p_indexFilePath1, const char* p_indexFilePath2)
{
    std::string folderPath1(p_indexFilePath1), folderPath2(p_indexFilePath2);

    std::shared_ptr<VectorIndex> index1, index2;
    LoadIndex(folderPath1, index1);
    LoadIndex(folderPath2, index2);

    std::shared_ptr<VectorSet> p_vectorSet;
    std::shared_ptr<MetadataSet> p_metaSet;
    size_t vectorSize = GetValueTypeSize(index2->GetVectorValueType()) * index2->GetFeatureDim();
    std::uint64_t offsets[2] = { 0 };
    ByteArray metaoffset((std::uint8_t*)offsets, 2 * sizeof(std::uint64_t), false);
    for (SizeType i = 0; i < index2->GetNumSamples(); i++)
        if (index2->ContainSample(i))
        {
            p_vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)index2->GetSample(i), vectorSize, false),
                index2->GetVectorValueType(), index2->GetFeatureDim(), 1));
            ByteArray meta = index2->GetMetadata(i);
            offsets[1] = meta.Length();
            p_metaSet.reset(new MemMetadataSet(meta, metaoffset, 1));
            index1->AddIndex(p_vectorSet, p_metaSet);
        }

    index1->SaveIndex(folderPath1);
    return ErrorCode::Success;
}
