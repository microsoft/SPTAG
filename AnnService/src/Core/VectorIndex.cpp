// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/DataUtils.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/BufferStream.h"

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


std::shared_ptr<std::vector<std::uint64_t>> VectorIndex::CalculateBufferSize() const
{
    std::shared_ptr<std::vector<std::uint64_t>> ret = BufferSize();
    if (m_pMetadata != nullptr)
    {
        auto metasize = m_pMetadata->BufferSize();
        ret->push_back(metasize.first);
        ret->push_back(metasize.second);
    }
    return std::move(ret);
}


ErrorCode
VectorIndex::LoadIndexConfig(Helper::IniReader& p_reader)
{
    std::string metadataSection("MetaData");
    if (p_reader.DoesSectionExist(metadataSection))
    {
        m_sMetadataFile = p_reader.GetParameter(metadataSection, "MetaDataFilePath", std::string());
        m_sMetadataIndexFile = p_reader.GetParameter(metadataSection, "MetaDataIndexPath", std::string());
    }

    if (DistCalcMethod::Undefined == p_reader.GetParameter("Index", "DistCalcMethod", DistCalcMethod::Undefined))
    {
        std::cerr << "Error: Failed to load parameter DistCalcMethod." << std::endl;
        return ErrorCode::Fail;
    }
    return LoadConfig(p_reader);
}


ErrorCode
VectorIndex::SaveIndexConfig(std::ostream& p_configOut)
{
    if (nullptr != m_pMetadata)
    {
        p_configOut << "[MetaData]" << std::endl;
        p_configOut << "MetaDataFilePath=" << m_sMetadataFile << std::endl;
        p_configOut << "MetaDataIndexPath=" << m_sMetadataIndexFile << std::endl;
        if (nullptr != m_pMetaToVec) p_configOut << "MetaDataToVectorIndex=true" << std::endl;
        p_configOut << std::endl;
    }

    p_configOut << "[Index]" << std::endl;
    p_configOut << "IndexAlgoType=" << Helper::Convert::ConvertToString(GetIndexAlgoType()) << std::endl;
    p_configOut << "ValueType=" << Helper::Convert::ConvertToString(GetVectorValueType()) << std::endl;
    p_configOut << std::endl;

    return SaveConfig(p_configOut);
}


void
VectorIndex::BuildMetaMapping()
{
    m_pMetaToVec.reset(new std::unordered_map<std::string, SizeType>);
    for (SizeType i = 0; i < m_pMetadata->Count(); i++) {
        ByteArray meta = m_pMetadata->GetMetadata(i);
        m_pMetaToVec->emplace(std::string((char*)meta.Data(), meta.Length()), i);
    }
}


ErrorCode 
VectorIndex::LoadIndex(const std::string& p_config, const std::vector<ByteArray>& p_indexBlobs)
{
    SPTAG::Helper::IniReader p_reader;
    std::istringstream p_configin(p_config);
    if (SPTAG::ErrorCode::Success != p_reader.LoadIni(p_configin)) return ErrorCode::FailedParseValue;
    LoadIndexConfig(p_reader);
    
    if (p_reader.DoesSectionExist("MetaData") && p_indexBlobs.size() > 4)
    {
        ByteArray pMetaIndex = p_indexBlobs[p_indexBlobs.size() - 1];
        m_pMetadata.reset(new MemMetadataSet(p_indexBlobs[p_indexBlobs.size() - 2],
            ByteArray(pMetaIndex.Data() + sizeof(SizeType), pMetaIndex.Length() - sizeof(SizeType), false),
            *((SizeType*)pMetaIndex.Data())));

        if (!m_pMetadata->Available())
        {
            std::cerr << "Error: Failed to load metadata." << std::endl;
            return ErrorCode::Fail;
        }

        if (p_reader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
        {
            BuildMetaMapping();
        }
    }
    return LoadIndexDataFromMemory(p_indexBlobs);
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
    if (ErrorCode::Success != p_configReader.LoadIniFile(folderPath + "/indexloader.ini")) return ErrorCode::FailedOpenFile;
    LoadIndexConfig(p_configReader);
    
    if (p_configReader.DoesSectionExist("MetaData"))
    {
        m_pMetadata.reset(new FileMetadataSet(folderPath + m_sMetadataFile, folderPath + m_sMetadataIndexFile));

        if (!m_pMetadata->Available())
        {
            std::cerr << "Error: Failed to load metadata." << std::endl;
            return ErrorCode::Fail;
        }

        if (p_configReader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
        {
            BuildMetaMapping();
        }
    }
    return LoadIndexData(folderPath);
}


ErrorCode
VectorIndex::SaveIndex(std::string& p_config, const std::vector<ByteArray>& p_indexBlobs)
{
    std::ostringstream p_configStream;
    SaveIndexConfig(p_configStream);
    p_config = p_configStream.str();
    
    std::vector<std::ostream*> p_indexStreams;
    for (size_t i = 0; i < p_indexBlobs.size(); i++)
    {
        p_indexStreams.push_back(new Helper::obufferstream(new Helper::streambuf((char*)p_indexBlobs[i].Data(), p_indexBlobs[i].Length()), true));
    }

    ErrorCode ret = ErrorCode::Success;
    if (NeedRefine()) 
    {
        ret = RefineIndex(p_indexStreams);
    }
    else 
    {
        if (m_pMetadata != nullptr && p_indexStreams.size() > 5)
        {
            ret = m_pMetadata->SaveMetadata(*p_indexStreams[p_indexStreams.size() - 2], *p_indexStreams[p_indexStreams.size() - 1]);
        }
        if (ErrorCode::Success == ret) ret = SaveIndexData(p_indexStreams);
    }
    for (size_t i = 0; i < p_indexStreams.size(); i++)
    {
        delete p_indexStreams[i];
    }
    return ret;
}


ErrorCode
VectorIndex::SaveIndex(const std::string& p_folderPath)
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

    std::ofstream configFile(folderPath + "indexloader.ini");
    if (!configFile.is_open()) return ErrorCode::FailedCreateFile;
    SaveIndexConfig(configFile);
    configFile.close();
    
    if (NeedRefine()) return RefineIndex(p_folderPath);

    if (m_pMetadata != nullptr)
    {
        ErrorCode ret = m_pMetadata->SaveMetadata(folderPath + m_sMetadataFile, folderPath + m_sMetadataIndexFile);
        if (ErrorCode::Success != ret) return ret;
    }
    return SaveIndexData(folderPath);
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
    if (p_withMetaIndex && m_pMetadata != nullptr) 
    {
        BuildMetaMapping();
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
VectorIndex::AddIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex) {
    if (nullptr == p_vectorSet || p_vectorSet->Count() == 0 || p_vectorSet->Dimension() == 0 || p_vectorSet->GetValueType() != GetVectorValueType())
    {
        return ErrorCode::Fail;
    }

    return AddIndex(p_vectorSet->GetData(), p_vectorSet->Count(), p_vectorSet->Dimension(), p_metadataSet, p_withMetaIndex);
}


ErrorCode
VectorIndex::DeleteIndex(ByteArray p_meta) {
    if (m_pMetaToVec == nullptr) return ErrorCode::Fail;

    std::string meta((char*)p_meta.Data(), p_meta.Length());
    auto iter = m_pMetaToVec->find(meta);
    if (iter != m_pMetaToVec->end()) DeleteIndex(iter->second);
    return ErrorCode::Success;
}


const void* VectorIndex::GetSample(ByteArray p_meta)
{
    if (m_pMetaToVec == nullptr) return nullptr;

    std::string meta((char*)p_meta.Data(), p_meta.Length());
    auto iter = m_pMetaToVec->find(meta);
    if (iter != m_pMetaToVec->end()) return GetSample(iter->second);
    return nullptr;
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
    if (ErrorCode::Success != iniReader.LoadIniFile(p_loaderFilePath + "/indexloader.ini")) return ErrorCode::FailedOpenFile;

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);

    p_vectorIndex = CreateInstance(algoType, valueType);
    if (p_vectorIndex == nullptr) return ErrorCode::FailedParseValue;

    return p_vectorIndex->LoadIndex(p_loaderFilePath);
}



ErrorCode
VectorIndex::LoadIndex(const std::string& p_config, const std::vector<ByteArray>& p_indexBlobs, std::shared_ptr<VectorIndex>& p_vectorIndex)
{
    SPTAG::Helper::IniReader iniReader;
    std::istringstream p_configin(p_config);
    if (SPTAG::ErrorCode::Success != iniReader.LoadIni(p_configin)) return ErrorCode::FailedParseValue;

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);

    p_vectorIndex = CreateInstance(algoType, valueType);
    if (p_vectorIndex == nullptr) return ErrorCode::FailedParseValue;

    return p_vectorIndex->LoadIndex(p_config, p_indexBlobs);
}


ErrorCode
VectorIndex::MergeIndex(const char* p_indexFilePath1, const char* p_indexFilePath2, std::shared_ptr<VectorIndex>& p_vectorIndex)
{
    std::string folderPath1(p_indexFilePath1), folderPath2(p_indexFilePath2);
    LoadIndex(folderPath1, p_vectorIndex);

    if (!folderPath2.empty() && *(folderPath2.rbegin()) != FolderSep) folderPath2 += FolderSep;

    Helper::IniReader iniReader;
    if (ErrorCode::Success != iniReader.LoadIniFile(folderPath2 + "/indexloader.ini")) return ErrorCode::FailedOpenFile;

    std::shared_ptr<VectorIndex> addIndex = CreateInstance(iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined), 
        iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined));
    addIndex->LoadConfig(iniReader);
    addIndex->LoadIndexData(folderPath2);

    std::shared_ptr<MetadataSet> pMetadata;
    if (iniReader.DoesSectionExist("MetaData"))
    {
        pMetadata.reset(new MemMetadataSet(folderPath2 + iniReader.GetParameter("MetaData", "MetaDataFilePath", std::string()), 
            folderPath2 + iniReader.GetParameter("MetaData", "MetaDataIndexPath", std::string())));
    }
    
#pragma omp parallel for schedule(dynamic,128)
    for (SizeType i = 0; i < addIndex->GetNumSamples(); i++)
        if (addIndex->ContainSample(i))
        {
            std::shared_ptr<MetadataSet> p_metaSet;
            if (pMetadata != nullptr) {
                ByteArray meta = pMetadata->GetMetadata(i);
                std::uint64_t offsets[2] = { 0, meta.Length() };
                p_metaSet.reset(new MemMetadataSet(meta, ByteArray((std::uint8_t*)offsets, 2 * sizeof(std::uint64_t), false), 1));
            }
            p_vectorIndex->AddIndex(addIndex->GetSample(i), 1, addIndex->GetFeatureDim(), p_metaSet);
        }
    return ErrorCode::Success;
}
