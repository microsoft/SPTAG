#include "inc/Helper/ArgumentsParser.h"
#include "inc/Core/VectorIndex.h"

using namespace SPTAG;

class ParserOptions : public Helper::ArgumentsParser
{
public:
    ParserOptions()  
    {
        AddRequiredOption(m_indexFolder, "-i", "--indexfolder", "Input index folder.");
        AddRequiredOption(m_valueType, "-v", "--valueType", "Value type.");
        AddOptionalOption(m_step, "-s", "--step", "Running step.");
        AddOptionalOption(m_ssdFile, "-f", "--SSDFile", "SSD file.");
        AddOptionalOption(m_headIDFile, "-h", "--HeadIDFile", "HeadID file.");
        AddOptionalOption(m_headIndex, "-x", "--HeadIndex", "Head index.");
        AddOptionalOption(m_metaFile, "-m", "--MetaFile", "Meta file.");
        AddOptionalOption(m_metaIdxFile, "-mi", "--MetaIndexFile", "Meta index file.");
        AddOptionalOption(m_output, "-o", "--output", "Output file.");
    }

    ~ParserOptions() {}

    std::string m_indexFolder;

    VectorValueType m_valueType;
    
    std::string m_step = "Parse";

    std::string m_headIndex = "HeadIndex";

    std::string m_ssdFile = "SPTAGFullList.bin";

    std::string m_headIDFile = "SPTAGHeadVectorIDs.bin";

    std::string m_metaFile = "";

    std::string m_metaIdxFile = "";

    std::string m_output = "output.txt";

} options;

template <typename ValueType>
class SSDIndex {
public:
    struct ListInfo
    {
        int listEleCount = 0;

        std::uint16_t listPageCount = 0;

        std::uint64_t listOffset = 0;

        std::uint16_t pageOffset = 0;
    };

    int LoadingHeadInfo(std::shared_ptr<Helper::DiskPriorityIO>& ptr, std::vector<ListInfo>& m_listInfos)
    {
        int m_listCount;
        int m_totalDocumentCount;
        int m_iDataDimension;
        int m_listPageOffset;

        if (ptr->ReadBinary(sizeof(m_listCount), reinterpret_cast<char*>(&m_listCount)) != sizeof(m_listCount)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
            exit(1);
        }
        if (ptr->ReadBinary(sizeof(m_totalDocumentCount), reinterpret_cast<char*>(&m_totalDocumentCount)) != sizeof(m_totalDocumentCount)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
            exit(1);
        }
        if (ptr->ReadBinary(sizeof(m_iDataDimension), reinterpret_cast<char*>(&m_iDataDimension)) != sizeof(m_iDataDimension)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
            exit(1);
        }
        if (ptr->ReadBinary(sizeof(m_listPageOffset), reinterpret_cast<char*>(&m_listPageOffset)) != sizeof(m_listPageOffset)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
            exit(1);
        }

        if (m_vectorInfoSize == 0) m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
        else if (m_vectorInfoSize != m_iDataDimension * sizeof(ValueType) + sizeof(int)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read head info file! DataDimension and ValueType are not match!\n");
            exit(1);
        }

        m_listInfos.resize(m_listCount);

        size_t totalListElementCount = 0;

        std::map<int, int> pageCountDist;

        size_t biglistCount = 0;
        size_t biglistElementCount = 0;
        int pageNum;
        for (int i = 0; i < m_listCount; ++i)
        {
            if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&(pageNum))) != sizeof(pageNum)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }
            if (ptr->ReadBinary(sizeof(m_listInfos[i].pageOffset), reinterpret_cast<char*>(&(m_listInfos[i].pageOffset))) != sizeof(m_listInfos[i].pageOffset)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }
            if (ptr->ReadBinary(sizeof(m_listInfos[i].listEleCount), reinterpret_cast<char*>(&(m_listInfos[i].listEleCount))) != sizeof(m_listInfos[i].listEleCount)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }
            if (ptr->ReadBinary(sizeof(m_listInfos[i].listPageCount), reinterpret_cast<char*>(&(m_listInfos[i].listPageCount))) != sizeof(m_listInfos[i].listPageCount)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }

            m_listInfos[i].listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
            m_listInfos[i].listPageCount = static_cast<std::uint16_t>(ceil((m_vectorInfoSize * m_listInfos[i].listEleCount + m_listInfos[i].pageOffset) * 1.0 / (1 << PageSizeEx)));
            totalListElementCount += m_listInfos[i].listEleCount;
            int pageCount = m_listInfos[i].listPageCount;

            if (pageCount > 1)
            {
                ++biglistCount;
                biglistElementCount += m_listInfos[i].listEleCount;
            }

            if (pageCountDist.count(pageCount) == 0)
            {
                pageCountDist[pageCount] = 1;
            }
            else
            {
                pageCountDist[pageCount] += 1;
            }
        }

        LOG(Helper::LogLevel::LL_Info,
            "Finish reading header info, list count %d, total doc count %d, dimension %d, list page offset %d.\n",
            m_listCount,
            m_totalDocumentCount,
            m_iDataDimension,
            m_listPageOffset);


        LOG(Helper::LogLevel::LL_Info,
            "Big page (>4K): list count %zu, total element count %zu.\n",
            biglistCount,
            biglistElementCount);

        LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n", totalListElementCount);

        for (auto& ele : pageCountDist)
        {
            LOG(Helper::LogLevel::LL_Info, "Page Count Dist: %d %d\n", ele.first, ele.second);
        }

        return m_listCount;
    }

    bool LoadIndex(ParserOptions& p_opt) {
        std::string extraFullGraphFile = p_opt.m_indexFolder + FolderSep + p_opt.m_ssdFile;
        std::string curFile = extraFullGraphFile;
        do {
            auto curIndexFile = f_createIO();
            if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in)) {
                LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                return false;
            }

            m_indexFiles.emplace_back(curIndexFile);
            m_listInfos.emplace_back(0);
            m_totalListCount += LoadingHeadInfo(curIndexFile, m_listInfos.back());

            curFile = extraFullGraphFile + "_" + std::to_string(m_indexFiles.size());
        } while (fileexists(curFile.c_str()));
        m_listPerFile = static_cast<int>((m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());
        return true;
    }

    std::vector<std::vector<ListInfo>> m_listInfos;

    std::vector<std::shared_ptr<Helper::DiskPriorityIO>> m_indexFiles;

    int m_vectorInfoSize = 0;

    int m_totalListCount = 0;

    int m_listPerFile = 0;
};

template <typename ValueType>
ErrorCode MergeSSDFiles() {
    return ErrorCode::Success;
}

template <typename ValueType>
ErrorCode ParsePosting() {
    std::shared_ptr<VectorIndex> headIndex;
    ErrorCode ret;
    if ((ret = VectorIndex::LoadIndex(options.m_indexFolder + FolderSep + options.m_headIndex, headIndex)) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "Cannot load index %s\n", options.m_indexFolder.c_str());
        return ret;
    }

    std::shared_ptr<std::uint64_t> headMapping;
    headMapping.reset(new std::uint64_t[headIndex->GetNumSamples()], std::default_delete<std::uint64_t[]>());
    std::shared_ptr<Helper::DiskPriorityIO> ptr = SPTAG::f_createIO();
    if (ptr == nullptr || !ptr->Initialize((options.m_indexFolder + FolderSep + options.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n", (options.m_indexFolder + FolderSep + options.m_headIDFile).c_str());
        return ErrorCode::Fail;
    }
    IOBINARY(ptr, ReadBinary, sizeof(std::uint64_t) * headIndex->GetNumSamples(), (char*)(headMapping.get()));
    ptr->ShutDown();

    SSDIndex<ValueType> tailIndex;
    tailIndex.LoadIndex(options);
    
    if (ptr == nullptr || !ptr->Initialize((options.m_indexFolder + FolderSep + options.m_output).c_str(), std::ios::out)) {
        LOG(Helper::LogLevel::LL_Error, "Cannot open file %s to write!", (options.m_indexFolder + FolderSep + options.m_output).c_str());
        return ErrorCode::Fail;
    }

    std::shared_ptr<MetadataSet> metaIndex;
    if (options.m_metaFile != "") metaIndex.reset(new MemMetadataSet(options.m_indexFolder + FolderSep + options.m_metaFile, options.m_indexFolder + FolderSep + options.m_metaIdxFile, 1024 * 1024, MaxSize, 10));
    if (metaIndex == nullptr) LOG(Helper::LogLevel::LL_Info, "Output vector ids\n");
    else LOG(Helper::LogLevel::LL_Info, "Output vector metas\n");

    std::vector<char> buffer;
    for (int curPostingID = 0; curPostingID < headIndex->GetNumSamples(); curPostingID++) {
        int fileid = curPostingID / tailIndex.m_listPerFile;
        auto listInfo = &(tailIndex.m_listInfos[fileid][curPostingID % tailIndex.m_listPerFile]);
        Helper::DiskPriorityIO* indexFile = tailIndex.m_indexFiles[fileid].get();

        if (metaIndex != nullptr) {
            ByteArray meta = metaIndex->GetMetadata((SizeType)(headMapping.get()[curPostingID]));
            ptr->WriteBinary(meta.Length(), (const char*)meta.Data());
        }
        else {
            ptr->WriteString(std::to_string(headMapping.get()[curPostingID]).c_str());
        }

        if (listInfo->listEleCount == 0)
        {
            ptr->WriteString("\n");
            continue;
        }

        size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
        buffer.resize(totalBytes);

        auto numRead = indexFile->ReadBinary(totalBytes, buffer.data(), listInfo->listOffset);
        if (numRead != totalBytes) {
            LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", options.m_ssdFile.c_str(), totalBytes, numRead);
            exit(-1);
        }
        
        for (char* vectorInfo = buffer.data() + listInfo->pageOffset, *vectorInfoEnd = vectorInfo + listInfo->listEleCount * tailIndex.m_vectorInfoSize; 
            vectorInfo < vectorInfoEnd; vectorInfo += tailIndex.m_vectorInfoSize) {
            int vecid = *(reinterpret_cast<int*>(vectorInfo));
            ptr->WriteString(",");
            if (metaIndex != nullptr) {
                ByteArray vmeta = metaIndex->GetMetadata(vecid);
                ptr->WriteBinary(vmeta.Length(), (const char*)vmeta.Data());
            }
            else {
                ptr->WriteString(std::to_string(vecid).c_str());
            }
        }
        ptr->WriteString("\n");
    }
    ptr->ShutDown();
}

int main(int argc, char* argv[]) {
    if (!options.Parse(argc - 1, argv + 1))
    {
        exit(1);
    }

    if (options.m_step == "Parse") {
        switch (options.m_valueType)
        {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        ParsePosting<Type>(); \
        break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

        default: break;
        }
    }
    return 0;
}