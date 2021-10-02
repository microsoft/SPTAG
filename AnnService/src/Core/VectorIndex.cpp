// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/VectorIndex.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/SimpleIniReader.h"

#include "inc/Core/BKT/Index.h"
#include "inc/Core/KDT/Index.h"

#ifndef _MSC_VER
#include "inc/Helper/ConcurrentSet.h"
typedef typename SPTAG::Helper::Concurrent::ConcurrentMap<std::string, SPTAG::SizeType> MetadataMap;
#else
#include <concurrent_unordered_map.h>
typedef typename Concurrency::concurrent_unordered_map<std::string, SPTAG::SizeType> MetadataMap;
#endif

using namespace SPTAG;

#ifdef DEBUG
std::shared_ptr<Helper::Logger> SPTAG::g_pLogger(new Helper::SimpleLogger(Helper::LogLevel::LL_Debug));
#else
std::shared_ptr<Helper::Logger> SPTAG::g_pLogger(new Helper::SimpleLogger(Helper::LogLevel::LL_Info));
#endif



std::shared_ptr<Helper::DiskPriorityIO>(*SPTAG::f_createIO)() = []() -> std::shared_ptr<Helper::DiskPriorityIO> { return std::shared_ptr<Helper::DiskPriorityIO>(new Helper::SimpleFileIO()); };

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
VectorIndex::SetMetadata(MetadataSet* p_new) {
    m_pMetadata.reset(p_new);
}


MetadataSet*
VectorIndex::GetMetadata() const {
    return m_pMetadata.get();
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
        LOG(Helper::LogLevel::LL_Error, "Error: Failed to load parameter DistCalcMethod.\n");
        return ErrorCode::Fail;
    }
    return LoadConfig(p_reader);
}


ErrorCode
VectorIndex::SaveIndexConfig(std::shared_ptr<Helper::DiskPriorityIO> p_configOut)
{
    if (nullptr != m_pMetadata)
    {
        IOSTRING(p_configOut, WriteString, "[MetaData]\n");
        IOSTRING(p_configOut, WriteString, ("MetaDataFilePath=" + m_sMetadataFile + "\n").c_str());
        IOSTRING(p_configOut, WriteString, ("MetaDataIndexPath=" + m_sMetadataIndexFile + "\n").c_str());
        if (nullptr != m_pMetaToVec) IOSTRING(p_configOut, WriteString, "MetaDataToVectorIndex=true\n");
        IOSTRING(p_configOut, WriteString, "\n");
    }

    IOSTRING(p_configOut, WriteString, "[Index]\n");
    IOSTRING(p_configOut, WriteString, ("IndexAlgoType=" + Helper::Convert::ConvertToString(GetIndexAlgoType()) + "\n").c_str());
    IOSTRING(p_configOut, WriteString, ("ValueType=" + Helper::Convert::ConvertToString(GetVectorValueType()) + "\n").c_str());
    IOSTRING(p_configOut, WriteString, "\n");

    return SaveConfig(p_configOut);
}


SizeType
VectorIndex::GetMetaMapping(std::string& meta) const
{
    MetadataMap* ptr = static_cast<MetadataMap*>(m_pMetaToVec.get());
    auto iter = ptr->find(meta);
    if (iter != ptr->end()) return iter->second;
    return -1;
}


void
VectorIndex::UpdateMetaMapping(const std::string& meta, SizeType i)
{
    MetadataMap* ptr = static_cast<MetadataMap*>(m_pMetaToVec.get());
    auto iter = ptr->find(meta);
    if (iter != ptr->end()) DeleteIndex(iter->second);;
    (*ptr)[meta] = i;
}


void
VectorIndex::BuildMetaMapping(bool p_checkDeleted)
{
    MetadataMap* ptr = new MetadataMap(m_iDataBlockSize);
    for (SizeType i = 0; i < m_pMetadata->Count(); i++) {
        if (!p_checkDeleted || ContainSample(i)) {
            ByteArray meta = m_pMetadata->GetMetadata(i);
            (*ptr)[std::string((char*)meta.Data(), meta.Length())] = i;
        }
    }
    m_pMetaToVec.reset(ptr, std::default_delete<MetadataMap>());
}


ErrorCode
VectorIndex::SaveIndex(std::string& p_config, const std::vector<ByteArray>& p_indexBlobs)
{
    if (GetNumSamples() - GetNumDeleted() == 0) return ErrorCode::EmptyIndex;

    ErrorCode ret = ErrorCode::Success;
    {
        std::shared_ptr<Helper::DiskPriorityIO> p_configStream(new Helper::SimpleBufferIO());
        if (p_configStream == nullptr || !p_configStream->Initialize(nullptr, std::ios::out)) return ErrorCode::EmptyDiskIO;
        if ((ret = SaveIndexConfig(p_configStream)) != ErrorCode::Success) return ret;
        p_config.resize(p_configStream->TellP());
        IOBINARY(p_configStream, ReadBinary, p_config.size(), (char*)p_config.c_str(), 0);
    }

    std::vector<std::shared_ptr<Helper::DiskPriorityIO>> p_indexStreams;
    for (size_t i = 0; i < p_indexBlobs.size(); i++)
    {
        std::shared_ptr<Helper::DiskPriorityIO> ptr(new Helper::SimpleBufferIO());
        if (ptr == nullptr || !ptr->Initialize((char*)p_indexBlobs[i].Data(), std::ios::binary | std::ios::out, p_indexBlobs[i].Length())) return ErrorCode::EmptyDiskIO;
        p_indexStreams.push_back(std::move(ptr));
    }

    if (NeedRefine())
    {
        ret = RefineIndex(p_indexStreams, nullptr);
    }
    else 
    {
        if (m_pMetadata != nullptr && p_indexStreams.size() > 5)
        {
            ret = m_pMetadata->SaveMetadata(p_indexStreams[p_indexStreams.size() - 2], p_indexStreams[p_indexStreams.size() - 1]);
        }
        if (ErrorCode::Success == ret) ret = SaveIndexData(p_indexStreams);
    }
    return ret;
}


ErrorCode
VectorIndex::SaveIndex(const std::string& p_folderPath)
{
    if (GetNumSamples() - GetNumDeleted() == 0) return ErrorCode::EmptyIndex;

    std::string folderPath(p_folderPath);
    if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep)
    {
        folderPath += FolderSep;
    }

    if (!direxists(folderPath.c_str()))
    {
        mkdir(folderPath.c_str());
    }

    ErrorCode ret = ErrorCode::Success;
    {
        auto configFile = SPTAG::f_createIO();
        if (configFile == nullptr || !configFile->Initialize((folderPath + "indexloader.ini").c_str(), std::ios::out)) return ErrorCode::FailedCreateFile;
        if ((ret = SaveIndexConfig(configFile)) != ErrorCode::Success) return ret;
    }

    std::shared_ptr<std::vector<std::string>> indexfiles = GetIndexFiles();
    if (nullptr != m_pMetadata) {
        indexfiles->push_back(m_sMetadataFile);
        indexfiles->push_back(m_sMetadataIndexFile);
    }
    std::vector<std::shared_ptr<Helper::DiskPriorityIO>> handles;
    for (std::string& f : *indexfiles) {
        auto ptr = SPTAG::f_createIO();
        if (ptr == nullptr || !ptr->Initialize((folderPath + f).c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
        handles.push_back(std::move(ptr));
    }

    if (NeedRefine()) 
    {
        ret = RefineIndex(handles, nullptr);
    }
    else 
    {
        if (m_pMetadata != nullptr) ret = m_pMetadata->SaveMetadata(handles[handles.size() - 2], handles[handles.size() - 1]);
        if (ErrorCode::Success == ret) ret = SaveIndexData(handles);
    }
    return ret;
}


ErrorCode
VectorIndex::SaveIndexToFile(const std::string& p_file, IAbortOperation* p_abort)
{
    if (GetNumSamples() - GetNumDeleted() == 0) return ErrorCode::EmptyIndex;

    auto fp = SPTAG::f_createIO();
    if (fp == nullptr || !fp->Initialize(p_file.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;

    ErrorCode ret = ErrorCode::Success;
    
    std::uint64_t configSize = 0;
    IOBINARY(fp, WriteBinary, sizeof(configSize), (char*)&configSize);
    if ((ret = SaveIndexConfig(fp)) != ErrorCode::Success) return ret;
    configSize = fp->TellP() - sizeof(configSize);

    if (p_abort != nullptr && p_abort->ShouldAbort()) ret = ErrorCode::ExternalAbort;
    else {
        std::uint64_t blobs = CalculateBufferSize()->size();
        IOBINARY(fp, WriteBinary, sizeof(blobs), (char*)&blobs);
        std::vector<std::shared_ptr<Helper::DiskPriorityIO>> p_indexStreams(blobs, fp);

        if (NeedRefine())
        {
            ret = RefineIndex(p_indexStreams, p_abort);
        }
        else
        {
            ret = SaveIndexData(p_indexStreams);

            if (p_abort != nullptr && p_abort->ShouldAbort()) ret = ErrorCode::ExternalAbort;

            if (ErrorCode::Success == ret && m_pMetadata != nullptr) ret = m_pMetadata->SaveMetadata(fp, fp);
        }
    }
    IOBINARY(fp, WriteBinary, sizeof(configSize), (char*)&configSize, 0);
    fp->ShutDown();

    if (ret == ErrorCode::ExternalAbort) std::remove(p_file.c_str());
    return ret;
}


ErrorCode
VectorIndex::BuildIndex(std::shared_ptr<VectorSet> p_vectorSet,
    std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex)
{
    if (nullptr == p_vectorSet || p_vectorSet->GetValueType() != GetVectorValueType())
    {
        return ErrorCode::Fail;
    }
    m_pMetadata = std::move(p_metadataSet);
    if (p_withMetaIndex && m_pMetadata != nullptr)
    {
        BuildMetaMapping(false);
    }
    BuildIndex(p_vectorSet->GetData(), p_vectorSet->Count(), p_vectorSet->Dimension());
    return ErrorCode::Success;
}


ErrorCode
VectorIndex::SearchIndex(const void* p_vector, int p_vectorCount, int p_neighborCount, bool p_withMeta, BasicResult* p_results) const {
    size_t vectorSize = GetValueTypeSize(GetVectorValueType()) * GetFeatureDim();
#pragma omp parallel for schedule(dynamic,10)
    for (int i = 0; i < p_vectorCount; i++) {
        QueryResult res((char*)p_vector + i * vectorSize, p_neighborCount, p_withMeta, p_results + i * p_neighborCount);
        SearchIndex(res);
    }
    return ErrorCode::Success;
}


ErrorCode 
VectorIndex::AddIndex(std::shared_ptr<VectorSet> p_vectorSet, std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex) {
    if (nullptr == p_vectorSet || p_vectorSet->GetValueType() != GetVectorValueType())
    {
        return ErrorCode::Fail;
    }

    return AddIndex(p_vectorSet->GetData(), p_vectorSet->Count(), p_vectorSet->Dimension(), p_metadataSet, p_withMetaIndex);
}


ErrorCode
VectorIndex::DeleteIndex(ByteArray p_meta) {
    if (m_pMetaToVec == nullptr) return ErrorCode::VectorNotFound;

    std::string meta((char*)p_meta.Data(), p_meta.Length());
    SizeType vid = GetMetaMapping(meta);
    if (vid >= 0) return DeleteIndex(vid);
    return ErrorCode::VectorNotFound;
}


ErrorCode
VectorIndex::MergeIndex(VectorIndex* p_addindex, int p_threadnum, IAbortOperation* p_abort)
{
    ErrorCode ret = ErrorCode::Success;
    if (p_addindex->m_pMetadata != nullptr) {
#pragma omp parallel for num_threads(p_threadnum) schedule(dynamic,128)
        for (SizeType i = 0; i < p_addindex->GetNumSamples(); i++)
        {
            if (ret == ErrorCode::ExternalAbort) continue;

            if (p_addindex->ContainSample(i))
            {
                ByteArray meta = p_addindex->GetMetadata(i);
                std::uint64_t offsets[2] = { 0, meta.Length() };
                std::shared_ptr<MetadataSet> p_metaSet(new MemMetadataSet(meta, ByteArray((std::uint8_t*)offsets, sizeof(offsets), false), 1));
                AddIndex(p_addindex->GetSample(i), 1, p_addindex->GetFeatureDim(), p_metaSet);
            }

            if (p_abort != nullptr && p_abort->ShouldAbort()) 
            {
                ret = ErrorCode::ExternalAbort;
            }
        }
    }
    else {
#pragma omp parallel for num_threads(p_threadnum) schedule(dynamic,128)
        for (SizeType i = 0; i < p_addindex->GetNumSamples(); i++) 
        {
            if (ret == ErrorCode::ExternalAbort) continue;

            if (p_addindex->ContainSample(i))
            {
                AddIndex(p_addindex->GetSample(i), 1, p_addindex->GetFeatureDim(), nullptr);
            }

            if (p_abort != nullptr && p_abort->ShouldAbort())
            {
                ret = ErrorCode::ExternalAbort;
            }
        }
    }
    return ret;
}


const void* VectorIndex::GetSample(ByteArray p_meta, bool& deleteFlag)
{
    if (m_pMetaToVec == nullptr) return nullptr;

    std::string meta((char*)p_meta.Data(), p_meta.Length());
    SizeType vid = GetMetaMapping(meta);
    if (vid >= 0) {
        deleteFlag = !ContainSample(vid);
        return GetSample(vid);
    }
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
    std::string folderPath(p_loaderFilePath);
    if (!folderPath.empty() && *(folderPath.rbegin()) != FolderSep) folderPath += FolderSep;

    Helper::IniReader iniReader;
    {
        auto fp = SPTAG::f_createIO();
        if (fp == nullptr || !fp->Initialize((folderPath + "indexloader.ini").c_str(), std::ios::in)) return ErrorCode::FailedOpenFile;
        if (ErrorCode::Success != iniReader.LoadIni(fp)) return ErrorCode::FailedParseValue;
    }

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);
    if ((p_vectorIndex = CreateInstance(algoType, valueType)) == nullptr) return ErrorCode::FailedParseValue;

    ErrorCode ret = ErrorCode::Success;
    if ((ret = p_vectorIndex->LoadIndexConfig(iniReader)) != ErrorCode::Success) return ret;

    std::shared_ptr<std::vector<std::string>> indexfiles = p_vectorIndex->GetIndexFiles();
    if (iniReader.DoesSectionExist("MetaData")) {
        indexfiles->push_back(p_vectorIndex->m_sMetadataFile);
        indexfiles->push_back(p_vectorIndex->m_sMetadataIndexFile);
    }
    std::vector<std::shared_ptr<Helper::DiskPriorityIO>> handles;
    for (std::string& f : *indexfiles) {
        auto ptr = SPTAG::f_createIO();
        if (ptr == nullptr || !ptr->Initialize((folderPath + f).c_str(), std::ios::binary | std::ios::in)) {
            LOG(Helper::LogLevel::LL_Error, "Cannot open file %s!\n", (folderPath + f).c_str());
            ptr = nullptr;
        }
        handles.push_back(std::move(ptr));
    }

    if ((ret = p_vectorIndex->LoadIndexData(handles)) != ErrorCode::Success) return ret;

    if (iniReader.DoesSectionExist("MetaData"))
    {
        p_vectorIndex->SetMetadata(new MemMetadataSet(handles[handles.size() - 2], handles[handles.size() - 1], 
            p_vectorIndex->m_iDataBlockSize, p_vectorIndex->m_iDataCapacity, p_vectorIndex->m_iMetaRecordSize));

        if (!(p_vectorIndex->GetMetadata()->Available()))
        {
            LOG(Helper::LogLevel::LL_Error, "Error: Failed to load metadata.\n");
            return ErrorCode::Fail;
        }

        if (iniReader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
        {
            p_vectorIndex->BuildMetaMapping();
        }
    }
    p_vectorIndex->m_bReady = true;
    return ErrorCode::Success;
}


ErrorCode
VectorIndex::LoadIndexFromFile(const std::string& p_file, std::shared_ptr<VectorIndex>& p_vectorIndex)
{
    auto fp = SPTAG::f_createIO();
    if (fp == nullptr || !fp->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;

    SPTAG::Helper::IniReader iniReader;
    {
        std::uint64_t configSize;
        IOBINARY(fp, ReadBinary, sizeof(configSize), (char*)&configSize);
        std::vector<char> config(configSize + 1, '\0');
        IOBINARY(fp, ReadBinary, configSize, config.data());

        std::shared_ptr<Helper::DiskPriorityIO> bufferhandle(new Helper::SimpleBufferIO());
        if (bufferhandle == nullptr || !bufferhandle->Initialize(config.data(), std::ios::in, configSize)) return ErrorCode::EmptyDiskIO;
        if (SPTAG::ErrorCode::Success != iniReader.LoadIni(bufferhandle)) return ErrorCode::FailedParseValue;
    }

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);
    if ((p_vectorIndex = CreateInstance(algoType, valueType)) == nullptr) return ErrorCode::FailedParseValue;

    ErrorCode ret = ErrorCode::Success;
    if ((ret = p_vectorIndex->LoadIndexConfig(iniReader)) != ErrorCode::Success) return ret;

    std::uint64_t blobs;
    IOBINARY(fp, ReadBinary, sizeof(blobs), (char*)&blobs);
   
    std::vector<std::shared_ptr<Helper::DiskPriorityIO>> p_indexStreams(blobs, fp);
    if ((ret = p_vectorIndex->LoadIndexData(p_indexStreams)) != ErrorCode::Success) return ret;

    if (iniReader.DoesSectionExist("MetaData"))
    {
        p_vectorIndex->SetMetadata(new MemMetadataSet(fp, fp, p_vectorIndex->m_iDataBlockSize, p_vectorIndex->m_iDataCapacity, p_vectorIndex->m_iMetaRecordSize));

        if (!(p_vectorIndex->GetMetadata()->Available()))
        {
            LOG(Helper::LogLevel::LL_Error, "Error: Failed to load metadata.\n");
            return ErrorCode::Fail;
        }

        if (iniReader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
        {
            p_vectorIndex->BuildMetaMapping();
        }
    }
    p_vectorIndex->m_bReady = true;
    return ErrorCode::Success;
}


ErrorCode
VectorIndex::LoadIndex(const std::string& p_config, const std::vector<ByteArray>& p_indexBlobs, std::shared_ptr<VectorIndex>& p_vectorIndex)
{
    SPTAG::Helper::IniReader iniReader;
    std::shared_ptr<Helper::DiskPriorityIO> fp(new Helper::SimpleBufferIO());
    if (fp == nullptr || !fp->Initialize(p_config.c_str(), std::ios::in, p_config.size())) return ErrorCode::EmptyDiskIO;
    if (SPTAG::ErrorCode::Success != iniReader.LoadIni(fp)) return ErrorCode::FailedParseValue;

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);
    if ((p_vectorIndex = CreateInstance(algoType, valueType)) == nullptr) return ErrorCode::FailedParseValue;

    ErrorCode ret = ErrorCode::Success;
    if ((p_vectorIndex->LoadIndexConfig(iniReader)) != ErrorCode::Success) return ret;

    if ((ret = p_vectorIndex->LoadIndexDataFromMemory(p_indexBlobs)) != ErrorCode::Success) return ret;

    if (iniReader.DoesSectionExist("MetaData") && p_indexBlobs.size() > 4)
    {
        ByteArray pMetaIndex = p_indexBlobs[p_indexBlobs.size() - 1];
        p_vectorIndex->SetMetadata(new MemMetadataSet(p_indexBlobs[p_indexBlobs.size() - 2],
            ByteArray(pMetaIndex.Data() + sizeof(SizeType), pMetaIndex.Length() - sizeof(SizeType), false),
            *((SizeType*)pMetaIndex.Data()), 
            p_vectorIndex->m_iDataBlockSize, p_vectorIndex->m_iDataCapacity, p_vectorIndex->m_iMetaRecordSize));

        if (!(p_vectorIndex->GetMetadata()->Available()))
        {
            LOG(Helper::LogLevel::LL_Error, "Error: Failed to load metadata.\n");
            return ErrorCode::Fail;
        }

        if (iniReader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
        {
            p_vectorIndex->BuildMetaMapping();
        }
    }
    p_vectorIndex->m_bReady = true;
    return ErrorCode::Success;
}


std::uint64_t VectorIndex::EstimatedVectorCount(std::uint64_t p_memory, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_vectorsInBlock, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize)
{
    size_t treeNodeSize;
    if (p_algo == IndexAlgoType::BKT) {
        treeNodeSize = sizeof(SizeType) * 3;
    }
    else if (p_algo == IndexAlgoType::KDT) {
        treeNodeSize = sizeof(SizeType) * 2 + sizeof(DimensionType) + sizeof(float);
    }
    else {
        return 0;
    }
    std::uint64_t unit = GetValueTypeSize(p_valuetype) * p_dimension + p_maxmeta + sizeof(std::uint64_t) + sizeof(SizeType) * p_neighborhoodSize + 1 + treeNodeSize * p_treeNumber;
    return ((p_memory / unit) / p_vectorsInBlock) * p_vectorsInBlock;
}


std::uint64_t VectorIndex::EstimatedMemoryUsage(std::uint64_t p_vectorCount, DimensionType p_dimension, VectorValueType p_valuetype, SizeType p_vectorsInBlock, SizeType p_maxmeta, IndexAlgoType p_algo, int p_treeNumber, int p_neighborhoodSize)
{
    p_vectorCount = ((p_vectorCount + p_vectorsInBlock - 1) / p_vectorsInBlock) * p_vectorsInBlock;
    size_t treeNodeSize;
    if (p_algo == IndexAlgoType::BKT) {
        treeNodeSize = sizeof(SizeType) * 3;
    }
    else if (p_algo == IndexAlgoType::KDT) {
        treeNodeSize = sizeof(SizeType) * 2 + sizeof(DimensionType) + sizeof(float);
    }
    else {
        return 0;
    }
    std::uint64_t ret = GetValueTypeSize(p_valuetype) * p_dimension * p_vectorCount; //Vector Size
    ret += p_maxmeta * p_vectorCount; // MetaData Size
    ret += sizeof(std::uint64_t) * p_vectorCount; // MetaIndex Size
    ret += sizeof(SizeType) * p_neighborhoodSize * p_vectorCount; // Graph Size
    ret += p_vectorCount; // DeletedFlag Size
    ret += treeNodeSize * p_treeNumber * p_vectorCount; // Tree Size
    return ret;
}



#if defined(GPU)

#include "inc/Core/Common/cuda/TailNeighbors.hxx"
struct test_EdgeCompare {
    bool operator()(const Edge& a, int b) const
    {
        return a.node < b;
    };

    bool operator()(int a, const Edge& b) const
    {
        return a < b.node;
    };

    __host__ __device__ bool operator()(const Edge& a, const Edge& b) {
        if (a.node == b.node)
        {
            if (a.distance == b.distance)
            {
                return a.tonode < b.tonode;
            }

            return a.distance < b.distance;
        }
        return a.node < b.node;
    }
} g_edgeComparer;




void VectorIndex::SortSelections(std::vector<Edge>* selections) {
  LOG(Helper::LogLevel::LL_Debug, "Starting sort of final input on GPU\n");
  GPU_SortSelections(selections);
}



void VectorIndex::ApproximateRNG(std::shared_ptr<VectorSet>& fullVectors, std::unordered_set<int>& exceptIDS, int candidateNum, Edge* selections, int replicaCount, int numThreads, int numTrees, int leafSize, float RNGFactor, int numGPUs)
{

    LOG(Helper::LogLevel::LL_Info, "Starting GPU SSD Index build stage...\n");

    int metric = (GetDistCalcMethod() == SPTAG::DistCalcMethod::Cosine);

    if(GetVectorValueType() != VectorValueType::Float) {
        typedef int32_t SUMTYPE;
        DistPair<SUMTYPE>* results = new DistPair<SUMTYPE>[((size_t)fullVectors->Count())*((size_t)replicaCount)];

        switch (GetVectorValueType())
        {
#define DefineVectorValueType(Name, Type) \
        case VectorValueType::Name: \
            if(fullVectors->Dimension() <= 64) { \
                getTailNeighborsTPT<Type, float, SUMTYPE, 64>((Type*)fullVectors->GetData(), fullVectors->Count(), this, exceptIDS, 64, (DistPair<SUMTYPE>*)results, replicaCount, numThreads, numTrees, leafSize, metric, numGPUs); \
            } else if (fullVectors->Dimension() <= 100) { \
                getTailNeighborsTPT<Type, float, SUMTYPE, 100>((Type*)fullVectors->GetData(), fullVectors->Count(), this, exceptIDS, 100, (DistPair<SUMTYPE>*)results, replicaCount, numThreads, numTrees, leafSize, metric, numGPUs); \
            } else { \
                LOG(Helper::LogLevel::LL_Error, "Datasets of >100 dimensions not currently supported for GPU Index build\n"); \
                exit(1); \
            } \
            break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default: break;
        }

        SizeType resIdx = 0;
        //#pragma omp parallel for
        for (SizeType vecIdx = 0; vecIdx < fullVectors->Count(); vecIdx++) {
            if (exceptIDS.count(vecIdx) == 0) {
                size_t vecOffset = vecIdx * (size_t)replicaCount;
                size_t resOffset = resIdx * (size_t)replicaCount;
                for (int resNum = 0; resNum < replicaCount && results[resOffset + resNum].idx != -1; resNum++) {
                    selections[vecOffset + resNum].node = results[resOffset + resNum].idx;
                    selections[vecOffset + resNum].distance = (float)results[resOffset + resNum].dist;
                }
                resIdx++;
            }
        }

        delete[] results;
    }
    else {
        typedef float SUMTYPE;
        DistPair<SUMTYPE>* results = new DistPair<SUMTYPE>[((size_t)fullVectors->Count())*((size_t)replicaCount)];

        if (fullVectors->Dimension() <= 64) {
            getTailNeighborsTPT<float, float, SUMTYPE, 64>((float*)fullVectors->GetData(), fullVectors->Count(), this, exceptIDS, 64, (DistPair<SUMTYPE>*)results, replicaCount, numThreads, numTrees, leafSize, metric, numGPUs);
        }
        else if (fullVectors->Dimension() <= 100) {
            getTailNeighborsTPT<float, float, SUMTYPE, 100>((float*)fullVectors->GetData(), fullVectors->Count(), this, exceptIDS, 100, (DistPair<SUMTYPE>*)results, replicaCount, numThreads, numTrees, leafSize, metric, numGPUs);
        }
        else {
            LOG(Helper::LogLevel::LL_Error, "Datasets of >100 dimensions not currently supported for GPU Index build\n");
            exit(1);
        }

        SizeType resIdx = 0;
        //#pragma omp parallel for
        for (SizeType vecIdx = 0; vecIdx < fullVectors->Count(); vecIdx++) {
            if (exceptIDS.count(vecIdx) == 0) {
                size_t vecOffset = vecIdx * (size_t)replicaCount;
                size_t resOffset = resIdx * (size_t)replicaCount;
                for (int resNum = 0; resNum < replicaCount && results[resOffset + resNum].idx != -1; resNum++) {
                    selections[vecOffset + resNum].node = results[resOffset + resNum].idx;
                    selections[vecOffset + resNum].distance = (float)results[resOffset + resNum].dist;
                }
                resIdx++;
            }
        }
        delete[] results;
    }
}
#else

struct EdgeCompare
{
    bool operator()(const Edge& a, int b) const
    {
        return a.node < b;
    };

    bool operator()(int a, const Edge& b) const
    {
        return a < b.node;
    };

    bool operator()(const Edge& a, const Edge& b) const
    {
        if (a.node == b.node)
        {
            if (a.distance == b.distance)
            {
                return a.tonode < b.tonode;
            }

            return a.distance < b.distance;
        }

        return a.node < b.node;
    };
} test_edgeComparer;

void VectorIndex::SortSelections(std::vector<Edge>* selections) {
  std::sort(selections->begin(), selections->end(), test_edgeComparer);
}

void VectorIndex::ApproximateRNG(std::shared_ptr<VectorSet>& fullVectors, std::unordered_set<int>& exceptIDS, int candidateNum, Edge* selections, int replicaCount, int numThreads, int numTrees, int leafSize, float RNGFactor, int numGPUs)
{
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    std::atomic_int nextFullID(0);
    std::atomic_size_t rngFailedCountTotal(0);

    for (int tid = 0; tid < numThreads; ++tid)
    {
        threads.emplace_back([&, tid]()
            {
                QueryResult resultSet(NULL, candidateNum, false);

                size_t rngFailedCount = 0;

                while (true)
                {
                    int fullID = nextFullID.fetch_add(1);
                    if (fullID >= fullVectors->Count())
                    {
                        break;
                    }

                    if (exceptIDS.count(fullID) > 0)
                    {
                        continue;
                    }

                    resultSet.SetTarget(fullVectors->GetVector(fullID));
                    resultSet.Reset();

                    SearchIndex(resultSet);

                    size_t selectionOffset = static_cast<size_t>(fullID)* replicaCount;

                    BasicResult* queryResults = resultSet.GetResults();
                    int currReplicaCount = 0;
                    for (int i = 0; i < candidateNum && currReplicaCount < replicaCount; ++i)
                    {
                        if (queryResults[i].VID == -1)
                        {
                            break;
                        }

                        // RNG Check.
                        bool rngAccpeted = true;
                        for (int j = 0; j < currReplicaCount; ++j)
                        {
                            float nnDist = ComputeDistance(GetSample(queryResults[i].VID), GetSample(selections[selectionOffset+j].node));

                            if (RNGFactor * nnDist < queryResults[i].Dist)
                            {
                                rngAccpeted = false;
                                break;
                            }
                        }

                        if (!rngAccpeted)
                        {
                            ++rngFailedCount;
                            continue;
                        }

                        selections[selectionOffset + currReplicaCount].node = queryResults[i].VID;
                        selections[selectionOffset + currReplicaCount].distance = queryResults[i].Dist;
                        ++currReplicaCount;
                    }
                }
                rngFailedCountTotal += rngFailedCount;
            });
    }

    for (int tid = 0; tid < numThreads; ++tid)
    {
        threads[tid].join();
    }
    LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. RNG failed count: %llu\n", static_cast<uint64_t>(rngFailedCountTotal.load()));
}
#endif
