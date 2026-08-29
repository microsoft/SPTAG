// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASTATICSEARCHER_H_
#define _SPTAG_SPANN_EXTRASTATICSEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/Common/TruthSet.h"
#include "Compressor.h"

#include <atomic>
#include <cstring>
#include <map>
#include <cmath>
#include <climits>
#include <future>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>

namespace SPTAG
{
    namespace SPANN
    {
        extern std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO;

        struct Selection {
            std::string m_tmpfile;
            size_t m_totalsize;
            size_t m_start;
            size_t m_end;
            std::vector<Edge> m_selections;
            static EdgeCompare g_edgeComparer;

            Selection(size_t totalsize, std::string tmpdir) : m_tmpfile(tmpdir + FolderSep + "selection_tmp"), m_totalsize(totalsize), m_start(0), m_end(totalsize) { remove(m_tmpfile.c_str()); m_selections.resize(totalsize); }

            ErrorCode SaveBatch()
            {
                auto f_out = f_createIO();
                if (f_out == nullptr || !f_out->Initialize(m_tmpfile.c_str(), std::ios::out | std::ios::binary | (fileexists(m_tmpfile.c_str()) ? std::ios::in : 0))) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to save selection for batching!\n", m_tmpfile.c_str());
                    return ErrorCode::FailedOpenFile;
                }
                if (f_out->WriteBinary(sizeof(Edge) * (m_end - m_start), (const char*)m_selections.data(), sizeof(Edge) * m_start) != sizeof(Edge) * (m_end - m_start)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot write to %s!\n", m_tmpfile.c_str());
                    return ErrorCode::DiskIOFail;
                }
                std::vector<Edge> batch_selection;
                m_selections.swap(batch_selection);
                m_start = m_end = 0;
                return ErrorCode::Success;
            }

            ErrorCode LoadBatch(size_t start, size_t end)
            {
                auto f_in = f_createIO();
                if (f_in == nullptr || !f_in->Initialize(m_tmpfile.c_str(), std::ios::in | std::ios::binary)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s to load selection batch!\n", m_tmpfile.c_str());
                    return ErrorCode::FailedOpenFile;
                }

                size_t readsize = end - start;
                m_selections.resize(readsize);
                if (f_in->ReadBinary(readsize * sizeof(Edge), (char*)m_selections.data(), start * sizeof(Edge)) != readsize * sizeof(Edge)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot read from %s! start:%zu size:%zu\n", m_tmpfile.c_str(), start, readsize);
                    return ErrorCode::DiskIOFail;
                }
                m_start = start;
                m_end = end;
                return ErrorCode::Success;
            }

            size_t lower_bound(SizeType node)
            {
                auto ptr = std::lower_bound(m_selections.begin(), m_selections.end(), node, g_edgeComparer);
                return m_start + (ptr - m_selections.begin());
            }

            Edge& operator[](size_t offset)
            {
                if (offset < m_start || offset >= m_end) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error read offset in selections:%zu\n", offset);
                }
                return m_selections[offset - m_start];
            }
        };

#define DecompressPosting(){\
        p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer(); \
        if (listInfo->listEleCount != 0) { \
            std::size_t sizePostingListFullData;\
            try {\
                sizePostingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);\
            }\
            catch (std::runtime_error& err) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", listInfo - m_listInfos.data(), err.what());\
                return;\
            }\
            if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", listInfo - m_listInfos.data(), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
                return;\
            }\
        }\
}\

#define DecompressPostingIterative(){\
        p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer(); \
        if (listInfo->listEleCount != 0) { \
            std::size_t sizePostingListFullData;\
            try {\
                sizePostingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);\
                if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", listInfo - m_listInfos.data(), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
                }\
             }\
            catch (std::runtime_error& err) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", listInfo - m_listInfos.data(), err.what());\
            }\
        }\
}\

#define ProcessPosting() \
        for (int i = 0; i < listInfo->listEleCount; i++) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);\
            SizeType vectorID = *(reinterpret_cast<SizeType*>(p_postingListFullData + offsetVectorID));\
            if (p_exWorkSpace->Deduper().CheckAndSet(vectorID)) { listElements--; continue; } \
            (this->*m_parseEncoding)(listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = m_headIndex->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf, queryResults.WithVec()? ByteArray((std::uint8_t*)(p_postingListFullData + offsetVector), sizeof(ValueType) * m_opt->m_dim, false) : ByteArray::c_empty); \
        } \

#define ProcessPostingOffset() \
        while (p_exWorkSpace->m_offset < listInfo->listEleCount) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, p_exWorkSpace->m_offset, listInfo->listEleCount);\
            p_exWorkSpace->m_offset++;\
            SizeType vectorID = *(reinterpret_cast<SizeType*>(p_postingListFullData + offsetVectorID));\
            if (p_exWorkSpace->Deduper().CheckAndSet(vectorID)) continue; \
            if (p_exWorkSpace->m_filterFunc != nullptr && !p_exWorkSpace->m_filterFunc(m_headIndex->GetMetadata(vectorID))) continue; \
            (this->*m_parseEncoding)(listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = m_headIndex->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
            foundResult = true;\
            break;\
        } \
        if (p_exWorkSpace->m_offset == listInfo->listEleCount) { \
            p_exWorkSpace->m_pi++; \
            p_exWorkSpace->m_offset = 0; \
        } \

        template <typename ValueType>
        class ExtraStaticSearcher : public IExtraSearcher
        {
        public:
            ExtraStaticSearcher(int layer, SPANN::Index<ValueType>* headIndex)
            {
                m_layer = layer;
                m_headIndex = headIndex;
                m_enableDeltaEncoding = false;
                m_enablePostingListRearrange = false;
                m_enableDataCompression = false;
                m_enableDictTraining = true;
            }

            virtual ~ExtraStaticSearcher()
            {
            }

            virtual SizeType GetNumSamples() const override
            {
                return m_opt->m_vectorSize;
            }

            virtual bool Available() override
            {
                return m_available;
            }

            virtual bool LoadIndex(Options& p_opt) override {
                m_opt = &p_opt;
                m_enableDeltaEncoding = p_opt.m_enableDeltaEncoding;
                m_enablePostingListRearrange = p_opt.m_enablePostingListRearrange;
                m_enableDataCompression = p_opt.m_enableDataCompression;
                m_enableDictTraining = p_opt.m_enableDictTraining;
                m_indexFiles.clear();
                m_indexFilePaths.clear();
                m_extendedSidecars.clear();
                m_rawSidecars.clear();
                m_sidecarReadLocks.clear();
                m_listInfos.clear();
                m_globalVectorIDToHeadMap.clear();
                m_totalListCount = 0;
                m_vectorInfoSize = 0;
                m_iDataDimension = 0;
                if (!ConfigurePostingFormat(p_opt))
                {
                    return false;
                }

                m_extraFullGraphFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                std::string curFile = m_extraFullGraphFile + "_" + std::to_string(m_layer);
                int loadFileID = 0;
                const size_t minimumPostingBytes = UseRaBitQBatchPosting()
                    ? RaBitQBatchPostingBytes(static_cast<size_t>(p_opt.m_postingVectorLimit))
                    : static_cast<size_t>(p_opt.m_postingVectorLimit) *
                        (UseRaBitQPosting()
                            ? ExpectedRaBitQVectorInfoSize()
                            : static_cast<size_t>(p_opt.m_dim) * sizeof(ValueType) +
                                sizeof(SizeType));
                p_opt.m_searchPostingPageLimit = max(
                    p_opt.m_searchPostingPageLimit,
                    static_cast<int>(
                        (minimumPostingBytes + PageSize - 1) / PageSize));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load index with posting page limit:%d\n", p_opt.m_searchPostingPageLimit);
                do {
                    auto curIndexFile = f_createAsyncIO();
                    if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(),
#ifndef _MSC_VER
#ifdef BATCH_READ
                        O_RDONLY | O_DIRECT, p_opt.m_searchInternalResultNum, 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                        O_RDONLY | O_DIRECT, p_opt.m_searchInternalResultNum * p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
#else
                        GENERIC_READ, (p_opt.m_searchPostingPageLimit + 1) * PageSize, 2, 2, (std::uint16_t)p_opt.m_ioThreads
#endif
                    )) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                        return false;
                    }

                    m_indexFiles.emplace_back(curIndexFile);
                    m_indexFilePaths.emplace_back(curFile);
                    try {
                        const size_t firstList = m_listInfos.size();
                        const int loadedListCount = LoadingHeadInfo(
                            curFile, p_opt.m_searchPostingPageLimit, m_listInfos);
                        if (UseRaBitQBatchPosting() &&
                            !LoadRaBitQBatchSidecars(
                                curFile,
                                static_cast<int>(m_indexFiles.size()) - 1,
                                firstList,
                                static_cast<size_t>(loadedListCount)))
                        {
                            return false;
                        }
                        m_totalListCount += loadedListCount;
                    } 
                    catch (std::exception& e)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error occurs when loading HeadInfo:%s\n", e.what());
                        return false;
                    }

                    ++loadFileID;
                    if (UseRaBitQBatchPosting())
                    {
                        if (loadFileID >= max(1, p_opt.m_ssdIndexFileNum)) break;
                        curFile = m_extraFullGraphFile + "_" +
                            std::to_string(m_layer) + ".part" +
                            std::to_string(loadFileID);
                    }
                    else
                    {
                        curFile = m_extraFullGraphFile + "_" +
                            std::to_string(m_indexFiles.size());
                    }
                } while (UseRaBitQBatchPosting() || fileexists(curFile.c_str()));
                m_oneContext = (m_indexFiles.size() == 1);

                if (m_enablePostingListRearrange) m_parsePosting = &ExtraStaticSearcher<ValueType>::ParsePostingListRearrange;
                else m_parsePosting = &ExtraStaticSearcher<ValueType>::ParsePostingList;
                if (m_enableDeltaEncoding) m_parseEncoding = &ExtraStaticSearcher<ValueType>::ParseDeltaEncoding;
                else m_parseEncoding = &ExtraStaticSearcher<ValueType>::ParseEncoding;
                
                m_listPerFile = static_cast<int>((m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", m_opt->m_vectorSize);

#ifndef _MSC_VER
                Helper::AIOTimeout.tv_nsec = p_opt.m_iotimeout * 1000;
#endif

                m_available = true;
                return true;
            }

            virtual ErrorCode SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                SearchStats* p_stats,
                std::set<SizeType>* truth, std::map<SizeType, std::set<SizeType>>* found,
                bool)
            {
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
                if (postingListCount > p_exWorkSpace->m_pageBuffers.size() ||
                    postingListCount > p_exWorkSpace->m_diskRequests.size()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static search workspace is too small: postings=%u buffers=%zu requests=%zu.\n",
                                 postingListCount, p_exWorkSpace->m_pageBuffers.size(),
                                 p_exWorkSpace->m_diskRequests.size());
                    return ErrorCode::Fail;
                }

                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
                QueryDiskStats diskStats;
                int listElements = 0;
                int missingPostingIDs = 0;
                std::atomic<int> scanRet(static_cast<int>(ErrorCode::Success));
                std::vector<BatchCandidate> batchCandidates;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    auto it = m_globalVectorIDToHeadMap.find(curPostingID);
                    if (it == m_globalVectorIDToHeadMap.end()) {
                        ++missingPostingIDs;
                        auto& request = p_exWorkSpace->m_diskRequests[pi];
                        request.m_readSize = 0;
                        request.m_success = false;
                        request.m_callback = nullptr;
                        continue;
                    }
                    curPostingID = it->second;
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext? 0: curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    listElements += listInfo->listEleCount;

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                    if (totalBytes > p_exWorkSpace->m_pageBuffers[pi].GetPageSize()) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Static posting %d requires %zu bytes but its workspace buffer has %zu bytes.\n",
                                     curPostingID, totalBytes,
                                     p_exWorkSpace->m_pageBuffers[pi].GetPageSize());
                        return ErrorCode::DiskIOFail;
                    }
                    diskStats.RecordRead(totalBytes);

#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_buffer = reinterpret_cast<char*>(p_exWorkSpace->m_pageBuffers[pi].GetBuffer());
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_status = (fileid << 16) | (request.m_status & 0xffff);
                    request.m_payload = (void*)listInfo; 
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    request.m_callback = [&p_exWorkSpace, &queryResults, &request, &listElements, &scanRet, &batchCandidates, &diskStats, this](bool success)
                    {
                        if (!success)
                        {
                            scanRet.store(
                                static_cast<int>(ErrorCode::DiskIOFail),
                                std::memory_order_relaxed);
                            return;
                        }
                        char* buffer = request.m_buffer;
                        ListInfo* listInfo = (ListInfo*)(request.m_payload);
                        const SizeType localPostingID = static_cast<SizeType>(listInfo - m_listInfos.data());

                        // decompress posting list
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            DecompressPosting();
                        }

                        if (UseRaBitQPosting())
                        {
                            auto ret = ProcessRaBitQPostingList(
                                localPostingID,
                                p_exWorkSpace,
                                queryResults,
                                listInfo,
                                p_postingListFullData,
                                nullptr,
                                m_opt->m_postingRaBitQRerank > 0
                                    ? &batchCandidates
                                    : nullptr,
                                &diskStats);
                            if (ret != ErrorCode::Success)
                            {
                                scanRet.store(static_cast<int>(ret), std::memory_order_relaxed);
                            }
                            return;
                        }

                        ProcessPosting();
                    };
#else // async read
                    request.m_callback = [&p_exWorkSpace, &request](bool success)
                    {
                        p_exWorkSpace->m_processIocp.push(&request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                        unprocessed--;
                    }
#endif
#else // sync read
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        throw std::runtime_error("File read mismatch");
                    }
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    if (UseRaBitQPosting())
                    {
                        auto ret = ProcessRaBitQPostingList(
                            curPostingID,
                            p_exWorkSpace,
                            queryResults,
                            listInfo,
                            p_postingListFullData,
                            nullptr,
                            m_opt->m_postingRaBitQRerank > 0
                                ? &batchCandidates
                                : nullptr,
                            &diskStats);
                        if (ret != ErrorCode::Success) return ret;
                    }
                    else
                    {
                        ProcessPosting();
                    }
#endif
                }

                if (missingPostingIDs > 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "Static search skipped %d of %u posting IDs that are absent from the head map.\n",
                                 missingPostingIDs, postingListCount);
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                {
                    int retry = 0;
                    bool readSuccess = false;
                    while (retry < 2 && !readSuccess)
                    {
                        readSuccess = BatchReadFileAsync(
                            m_indexFiles,
                            (p_exWorkSpace->m_diskRequests).data(),
                            postingListCount);
                        ++retry;
                    }
                    if (!readSuccess) return ErrorCode::DiskIOFail;
                }
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    if (UseRaBitQPosting())
                    {
                        auto ret = ProcessRaBitQPostingList(
                            static_cast<SizeType>(listInfo - m_listInfos.data()),
                            p_exWorkSpace,
                            queryResults,
                            listInfo,
                            p_postingListFullData,
                            nullptr,
                            m_opt->m_postingRaBitQRerank > 0
                                ? &batchCandidates
                                : nullptr,
                            &diskStats);
                        if (ret != ErrorCode::Success)
                        {
                            scanRet.store(static_cast<int>(ret), std::memory_order_relaxed);
                        }
                    }
                    else
                    {
                        ProcessPosting();
                    }
                }
#endif
#endif
                if (scanRet.load(std::memory_order_relaxed) != static_cast<int>(ErrorCode::Success))
                {
                    return static_cast<ErrorCode>(scanRet.load(std::memory_order_relaxed));
                }
                if (UseRaBitQBatchPosting() &&
                    m_opt->m_postingRaBitQRerank > 0)
                {
                    const ErrorCode rerankStatus = FinalizeRaBitQBatchRerank(
                        batchCandidates,
                        p_exWorkSpace,
                        queryResults,
                        nullptr,
                        &diskStats);
                    if (rerankStatus != ErrorCode::Success) return rerankStatus;
                }
                if (truth && found) {
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        const auto globalPostingID =
                            p_exWorkSpace->m_postingIDs[pi];
                        const auto posting = m_globalVectorIDToHeadMap.find(
                            globalPostingID);
                        if (posting == m_globalVectorIDToHeadMap.end())
                        {
                            continue;
                        }
                        const auto curPostingID = posting->second;

                        ListInfo* listInfo = &(m_listInfos[curPostingID]);
                        char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer();
                            if (listInfo->listEleCount != 0)
                            {
                                try {
                                    m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);
                                }
                                catch (std::runtime_error& err) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", curPostingID, err.what());
                                    continue;
                                }
                            }
                        }

                        if (UseRaBitQBatchPosting())
                        {
                            size_t vectorCount = 0;
                            const size_t batchCount =
                                RaBitQBatchCount(listInfo->listEleCount);
                            for (size_t batch = 0; batch < batchCount; ++batch)
                            {
                                const char* record = p_postingListFullData +
                                    batch * ExpectedRaBitQBatchRecordSize();
                                BatchRecordHeader recordHeader;
                                std::memcpy(
                                    &recordHeader, record, sizeof(recordHeader));
                                if (recordHeader.reserved != 0 ||
                                    recordHeader.validCount >
                                        m_splitBatchLayout.batchSize ||
                                    recordHeader.validCount >
                                        static_cast<size_t>(
                                            listInfo->listEleCount) -
                                            vectorCount)
                                {
                                    SPTAGLIB_LOG(
                                        Helper::LogLevel::LL_Error,
                                        "Corrupt STATIC RaBitQ batch metadata during truth instrumentation.\n");
                                    return ErrorCode::DiskIOFail;
                                }
                                const char* ids =
                                    record + sizeof(BatchRecordHeader);
                                for (size_t lane = 0;
                                     lane < recordHeader.validCount;
                                     ++lane)
                                {
                                    SizeType vectorID = -1;
                                    std::memcpy(
                                        &vectorID,
                                        ids + lane * sizeof(SizeType),
                                        sizeof(vectorID));
                                    if (truth->count(vectorID))
                                    {
                                        (*found)[globalPostingID].insert(
                                            vectorID);
                                    }
                                }
                                vectorCount += recordHeader.validCount;
                            }
                            if (vectorCount !=
                                static_cast<size_t>(listInfo->listEleCount))
                            {
                                SPTAGLIB_LOG(
                                    Helper::LogLevel::LL_Error,
                                    "Incomplete STATIC RaBitQ batch metadata during truth instrumentation.\n");
                                return ErrorCode::DiskIOFail;
                            }
                            continue;
                        }

                        for (size_t i = 0; i < listInfo->listEleCount; ++i) {
                            uint64_t offsetVectorID = m_enablePostingListRearrange ? (m_vectorInfoSize - sizeof(SizeType)) * listInfo->listEleCount + sizeof(SizeType) * i : m_vectorInfoSize * i;
                            SizeType vectorID = -1;
                            std::memcpy(
                                &vectorID,
                                p_postingListFullData + offsetVectorID,
                                sizeof(vectorID));
                            if (truth->count(vectorID))
                            {
                                (*found)[globalPostingID].insert(vectorID);
                            }
                        }
                    }
                }

                if (p_stats) 
                {
                    p_stats->m_totalListElementsCount = listElements;
                    p_stats->m_diskIOCount +=
                        diskStats.operationCount.load(std::memory_order_relaxed);
                    p_stats->m_diskAccessCount +=
                        diskStats.pageCount.load(std::memory_order_relaxed);
                }
                queryResults.SetScanned(listElements);
                return ErrorCode::Success;
            }

            virtual ErrorCode SearchIndexIterativeScan(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::vector<BasicResult>& p_results,
                bool) override
            {
                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
                std::vector<BatchCandidate> batchCandidates;

                auto appendPosting = [&](char* buffer, ListInfo* listInfo) -> ErrorCode {
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer();
                        if (listInfo->listEleCount != 0) {
                            std::size_t sizePostingListFullData;
                            try {
                                sizePostingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes,
                                    p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);
                            }
                            catch (std::runtime_error& err) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", listInfo - m_listInfos.data(), err.what());
                                return ErrorCode::DiskIOFail;
                            }
                            if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", listInfo - m_listInfos.data(), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);
                                return ErrorCode::DiskIOFail;
                            }
                        }
                    }

                    if (UseRaBitQPosting())
                    {
                        return ProcessRaBitQPostingList(
                            static_cast<SizeType>(listInfo - m_listInfos.data()),
                            p_exWorkSpace,
                            queryResults,
                            listInfo,
                            p_postingListFullData,
                            &p_results,
                            m_opt->m_postingRaBitQRerank > 0
                                ? &batchCandidates
                                : nullptr);
                    }

                    for (int i = 0; i < listInfo->listEleCount; i++) {
                        uint64_t offsetVectorID, offsetVector;
                        (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);
                        SizeType vectorID = *(reinterpret_cast<SizeType*>(p_postingListFullData + offsetVectorID));
                        if (p_exWorkSpace->Deduper().CheckAndSet(vectorID))
                            continue;
                        (this->*m_parseEncoding)(listInfo, (ValueType*)(p_postingListFullData + offsetVector));
                        auto distance2leaf = m_headIndex->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector);
                        p_results.emplace_back(vectorID, distance2leaf, ByteArray::c_empty,
                            queryResults.WithVec() ? ByteArray::Alloc((std::uint8_t*)(p_postingListFullData + offsetVector), sizeof(ValueType) * m_opt->m_dim) : ByteArray::c_empty);
                    }
                    return ErrorCode::Success;
                };

                std::atomic<int> scanRet(
                    static_cast<int>(ErrorCode::Success));

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    auto it = m_globalVectorIDToHeadMap.find(curPostingID);
                    if (it == m_globalVectorIDToHeadMap.end()) {
                        auto& request = p_exWorkSpace->m_diskRequests[pi];
                        request.m_readSize = 0;
                        request.m_success = false;
                        request.m_callback = nullptr;
                        continue;
                    }
                    curPostingID = it->second;
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext ? 0 : curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);

#ifdef ASYNC_READ
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_buffer = reinterpret_cast<char*>(p_exWorkSpace->m_pageBuffers[pi].GetBuffer());
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_status = (fileid << 16) | (request.m_status & 0xffff);
                    request.m_payload = (void*)listInfo;
                    request.m_success = false;

#ifdef BATCH_READ
                    request.m_callback = [&appendPosting, &request, &scanRet](bool success)
                    {
                        if (!success) {
                            scanRet.store(
                                static_cast<int>(ErrorCode::DiskIOFail),
                                std::memory_order_relaxed);
                            return;
                        }
                        ErrorCode ret = appendPosting(request.m_buffer, static_cast<ListInfo*>(request.m_payload));
                        if (ret != ErrorCode::Success)
                            scanRet.store(
                                static_cast<int>(ret),
                                std::memory_order_relaxed);
                    };
#else
                    request.m_callback = [&p_exWorkSpace, &request](bool success)
                    {
                        p_exWorkSpace->m_processIocp.push(&request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                        unprocessed--;
                    }
#endif
#else
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        return ErrorCode::DiskIOFail;
                    }

                    ErrorCode ret = appendPosting(buffer, listInfo);
                    if (ret != ErrorCode::Success) return ret;
#endif
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                int retry = 0;
                bool success = false;
                while (retry < 2 && !success)
                {
                    success = BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
                    retry++;
                }
                if (!success) return ErrorCode::DiskIOFail;
                if (scanRet.load(std::memory_order_relaxed) !=
                    static_cast<int>(ErrorCode::Success))
                {
                    return static_cast<ErrorCode>(
                        scanRet.load(std::memory_order_relaxed));
                }
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    ErrorCode ret = appendPosting(buffer, listInfo);
                    if (ret != ErrorCode::Success) return ret;
                }
#endif
#endif
                if (UseRaBitQBatchPosting() &&
                    m_opt->m_postingRaBitQRerank > 0)
                {
                    return FinalizeRaBitQBatchRerank(
                        batchCandidates,
                        p_exWorkSpace,
                        queryResults,
                        &p_results);
                }
                return ErrorCode::Success;
            }

            virtual ErrorCode SearchIndexWithoutParsing(ExtraWorkSpace *p_exWorkSpace) override
            {
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                int diskRead = 0;
                int diskIO = 0;
                int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext ? 0 : curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    diskRead += listInfo->listPageCount;
                    diskIO += 1;
                    listElements += listInfo->listEleCount;

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                    
#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_buffer = reinterpret_cast<char*>(p_exWorkSpace->m_pageBuffers[pi].GetBuffer());
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_status = (fileid << 16) | (request.m_status & 0xffff);
                    request.m_payload = (void*)listInfo;
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    request.m_callback = [this](bool success)
                    {
                        //char* buffer = request.m_buffer;
                        //ListInfo* listInfo = (ListInfo*)(request.m_payload);

                        // decompress posting list
                        /*
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            DecompressPosting();
                        }

                        ProcessPosting();
                        */
                    };
#else // async read
                    request.m_callback = [&p_exWorkSpace, &request](bool success)
                    {
                        p_exWorkSpace->m_processIocp.push(&request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                        unprocessed--;
                    }
#endif
#else // sync read
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        return ErrorCode::DiskIOFail;
                    }
                    // decompress posting list
                    /*
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
                    */
#endif
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                int retry = 0;
                bool success = false;
                while (retry < 2 && !success)
                {
                    success = BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
                    retry++;
                }
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    // decompress posting list
                    /*
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
                    */
                }
#endif
#endif
                return (success)? ErrorCode::Success: ErrorCode::DiskIOFail;
            }

            virtual ErrorCode SearchNextInPosting(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
                QueryResult& p_queryResults) override
            {
                /*
                COMMON::QueryResultSet<ValueType>& headResults = *((COMMON::QueryResultSet<ValueType>*) & p_headResults);
                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
                bool foundResult = false;
                BasicResult* head = headResults.GetResult(p_exWorkSpace->m_ri);
                while (!foundResult && p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) {
                    if (head && head->VID != -1 && p_exWorkSpace->m_ri <= p_exWorkSpace->m_pi &&
                       (p_exWorkSpace->m_filterFunc == nullptr || p_exWorkSpace->m_filterFunc(p_spann->GetMetadata(head->VID)))) {
                        queryResults.AddPoint(head->VID, head->Dist);
                        head = headResults.GetResult(++p_exWorkSpace->m_ri);
                        foundResult = true;
                        continue;
                    }
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[p_exWorkSpace->m_pi]).GetBuffer());
                    ListInfo* listInfo = static_cast<ListInfo*>(p_exWorkSpace->m_diskRequests[p_exWorkSpace->m_pi].m_payload);
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression && p_exWorkSpace->m_offset == 0)
                    {
                        DecompressPostingIterative();
                    }
                    ProcessPostingOffset();
                }
                if (!foundResult && head && head->VID != -1 &&
                (p_exWorkSpace->m_filterFunc == nullptr || p_exWorkSpace->m_filterFunc(p_spann->GetMetadata(head->VID)))) {
                    queryResults.AddPoint(head->VID, head->Dist);
                    head = headResults.GetResult(++p_exWorkSpace->m_ri);
                    foundResult = true;
                }
                if (foundResult) p_queryResults.SetScanned(p_queryResults.GetScanned() + 1);
                return (foundResult)? ErrorCode::Success : ErrorCode::VectorNotFound;
                */
                return ErrorCode::Undefined;
            }

            virtual ErrorCode SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
                QueryResult& p_query) override
            {
                /*
                if (p_exWorkSpace->m_loadPosting) {
                    ErrorCode ret = SearchIndexWithoutParsing(p_exWorkSpace);
                    if (ret != ErrorCode::Success) return ret;
                    p_exWorkSpace->m_ri = 0;
                    p_exWorkSpace->m_pi = 0;
                    p_exWorkSpace->m_offset = 0;
                    p_exWorkSpace->m_loadPosting = false;
                }

                return SearchNextInPosting(p_exWorkSpace, p_headResults, p_query, p_index, p_spann);
                */
                return ErrorCode::Undefined;
            }

            std::string GetPostingListFullData(
                int postingListId,
                size_t p_postingListSize,
                Selection &p_selections,
                std::shared_ptr<VectorSet> p_fullVectors,
                COMMON::Dataset<SizeType>& p_localToGlobal,
                bool p_enableDeltaEncoding = false,
                bool p_enablePostingListRearrange = false,
                const ValueType *headVector = nullptr)
            {
                if (UseRaBitQBatchPosting())
                {
                    return BuildRaBitQBatchPosting(
                        postingListId,
                        p_postingListSize,
                        p_selections,
                        p_fullVectors,
                        p_localToGlobal).binary;
                }

                if (UseRaBitQPosting())
                {
                    if (p_enableDeltaEncoding || p_enablePostingListRearrange)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "STATIC PostingQuantizer=RaBitQ does not support delta encoding or posting rearrangement.\n");
                        throw std::runtime_error("Unsupported STATIC RaBitQ posting encoding");
                    }

                    const auto* localCentroid = reinterpret_cast<const float*>(
                        m_headIndex->GetMemoryIndex()->GetSample(static_cast<SizeType>(postingListId)));
                    if (localCentroid == nullptr)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Failed to read head centroid for posting list %d.\n",
                                     postingListId);
                        throw std::runtime_error("Missing STATIC RaBitQ posting centroid");
                    }

                    std::string postingListFullData(
                        p_postingListSize * static_cast<size_t>(m_vectorInfoSize), '\0');
                    size_t selectIdx = p_selections.lower_bound(postingListId);
                    for (int i = 0; i < p_postingListSize; ++i)
                    {
                        if (p_selections[selectIdx].node != postingListId)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Selection ID NOT MATCH! node:%d offset:%zu\n",
                                         postingListId, selectIdx);
                            throw std::runtime_error("Selection ID mismatch");
                        }

                        const SizeType localVid = p_selections[selectIdx++].tonode;
                        SizeType vid = localVid;
                        if (p_localToGlobal.R() > 0) vid = *(p_localToGlobal[localVid]);

                        char* record = postingListFullData.data() +
                            static_cast<size_t>(i) * static_cast<size_t>(m_vectorInfoSize);
                        std::memcpy(record, &vid, sizeof(SizeType));
                        auto* binaryCode = reinterpret_cast<std::uint8_t*>(record + sizeof(SizeType));
                        auto* extendedCode = (m_splitCodeLayout.extendedBytes == 0)
                            ? nullptr
                            : binaryCode + m_splitCodeLayout.binaryBytes;
                        if (m_postingQuantizer->QuantizeSplitVector(
                                p_fullVectors->GetVector(localVid),
                                localCentroid,
                                binaryCode,
                                extendedCode) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Failed to encode STATIC RaBitQ posting vector %d.\n",
                                         vid);
                            throw std::runtime_error("Failed STATIC RaBitQ posting encoding");
                        }
                    }
                    return postingListFullData;
                }

                std::string postingListFullData("");
                std::string vectors("");
                std::string vectorIDs("");
                size_t selectIdx = p_selections.lower_bound(postingListId);
                // iterate over all the vectors in the posting list
                for (int i = 0; i < p_postingListSize; ++i)
                {
                    if (p_selections[selectIdx].node != postingListId)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH! node:%d offset:%zu\n", postingListId, selectIdx);
                        throw std::runtime_error("Selection ID mismatch");
                    }
                    std::string vectorID("");
                    std::string vector("");

                    const SizeType localVid = p_selections[selectIdx++].tonode;
                    SizeType vid = localVid;
                    if (p_localToGlobal.R() > 0) vid = *(p_localToGlobal[localVid]);
                    vectorID.append(reinterpret_cast<char *>(&vid), sizeof(SizeType));

                    ValueType *p_vector = reinterpret_cast<ValueType *>(p_fullVectors->GetVector(localVid));
                    if (p_enableDeltaEncoding)
                    {
                        DimensionType n = p_fullVectors->Dimension();
                        std::vector<ValueType> p_vector_delta(n);
                        for (auto j = 0; j < n; j++)
                        {
                            p_vector_delta[j] = p_vector[j] - headVector[j];
                        }
                        vector.append(reinterpret_cast<char *>(&p_vector_delta[0]), p_fullVectors->PerVectorDataSize());
                    }
                    else
                    {
                        vector.append(reinterpret_cast<char *>(p_vector), p_fullVectors->PerVectorDataSize());
                    }

                    if (p_enablePostingListRearrange)
                    {
                        vectorIDs += vectorID;
                        vectors += vector;
                    }
                    else
                    {
                        postingListFullData += (vectorID + vector);
                    }
                }
                if (p_enablePostingListRearrange)
                {
                    return vectors + vectorIDs;
                }
                return postingListFullData;
            }

            bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt, COMMON::Dataset<SizeType>& p_headToLocal, Helper::Concurrent::ConcurrentMap<SizeType, SizeType>& p_headGlobaltoLocal, COMMON::Dataset<SizeType>& p_localToGlobal, SizeType upperBound = -1) {
                std::string outputFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex + "_" + std::to_string(m_layer);
                if (outputFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Output file can't be empty!\n");
                    return false;
                }

                m_opt = &p_opt;
                if (!ConfigurePostingFormat(p_opt))
                {
                    return false;
                }
                if (UseRaBitQPosting() && !PersistPostingQuantizerCopy(p_opt))
                {
                    return false;
                }
                int numThreads = p_opt.m_iSSDNumberOfThreads;
                const int candidateNum = std::min(
                    p_opt.m_internalResultNum,
                    static_cast<int>(p_headIndex->GetNumSamples()));
                if (candidateNum <= 0)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Cannot build STATIC postings without head candidates.\n");
                    return false;
                }
                std::unordered_map<SizeType, SizeType> headVectorIDS;
                if (p_opt.m_headIDFile.empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                    return false;
                }

                if (m_layer > 0 && p_localToGlobal.R() == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty localToGlobal for non-leaf layer!\n");
                    return false;
                }

                for (int i = 0; i < p_headToLocal.R(); i++)
                {
                    const SizeType headID = *(p_headToLocal[i]);
                    if (p_localToGlobal.R() > 0 &&
                        (headID < 0 || headID >= p_localToGlobal.R()))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Head ID %lld is outside the previous layer mapping of size %lld.\n",
                            static_cast<std::int64_t>(headID),
                            static_cast<std::int64_t>(p_localToGlobal.R()));
                        return false;
                    }
                    headVectorIDS[headID] = i;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded %lld Vector IDs\n", static_cast<std::int64_t>(headVectorIDS.size()));

                SizeType fullCount = 0;
                size_t vectorInfoSize = 0;
                {
                    auto fullVectors = p_reader->GetVectorSet();
                    fullCount = fullVectors->Count();
                    m_iDataDimension = static_cast<int>(fullVectors->Dimension());
                    vectorInfoSize = UseRaBitQBatchPosting()
                        ? ExpectedRaBitQBatchRecordSize()
                        : UseRaBitQPosting()
                            ? ExpectedRaBitQVectorInfoSize()
                        : fullVectors->PerVectorDataSize() + sizeof(SizeType);
                    m_vectorInfoSize = static_cast<int>(vectorInfoSize);
                }
                if (upperBound > 0) fullCount = upperBound;

                Selection selections(static_cast<size_t>(fullCount) * p_opt.m_replicaCount, p_opt.m_tmpdir);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
                std::vector<std::atomic_int> replicaCount(fullCount);
                std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
                for (auto& pls : postingListSize) pls = 0;
                std::unordered_set<SizeType> emptySet;
                SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (p_opt.m_batches > 1)
                {
                    if (selections.SaveBatch() != ErrorCode::Success)
                    {
                        return false;
                    }
                }
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                    SizeType sampleSize = p_opt.m_samples;
                    std::vector<SizeType> samples(sampleSize, 0);
                    for (int i = 0; i < p_opt.m_batches; i++) {
                        SizeType start = i * batchSize;
                        SizeType end = min(start + batchSize, fullCount);
                        auto fullVectors = p_reader->GetVectorSet(start, end);
                        if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                        if (p_opt.m_batches > 1) {
                            if (selections.LoadBatch(static_cast<size_t>(start) * p_opt.m_replicaCount, static_cast<size_t>(end) * p_opt.m_replicaCount) != ErrorCode::Success)
                            {
                                return false;
                            }
                            emptySet.clear();
                            for (auto& pair : headVectorIDS) {
                                if (pair.first >= start && pair.first < end) emptySet.insert(pair.first - start);
                            }
                        }
                        else {
                            for (auto& pair : headVectorIDS) {
                                emptySet.insert(pair.first);
                            }
                        }

                        int sampleNum = 0;
                        for (SizeType j = start; j < end && sampleNum < sampleSize; j++)
                        {
                            if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                        }

                        float acc = 0;
                        for (int j = 0; j < sampleNum; j++)
                        {
                            COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                        }
                        if (sampleNum > 0) acc = acc / sampleNum;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%lld,%lld) loaded with %lld vectors (%zu) HeadIndex acc @%d:%f.\n", i, (std::int64_t)start, (std::int64_t)end, (std::int64_t)(fullVectors->Count()), selections.m_selections.size(), candidateNum, acc);

                        p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != MaxSize; resNum++) {
                                    ++postingListSize[selections[vecOffset + resNum].node];
                                    selections[vecOffset + resNum].tonode = j;
                                    ++replicaCount[j];
                                }
                            } else if (!p_opt.m_excludehead) {
                                selections[vecOffset].node = headVectorIDS[j];
                                selections[vecOffset].tonode = j;
                                ++postingListSize[selections[vecOffset].node];
                                ++replicaCount[j];
                            }
                        }

                        if (p_opt.m_batches > 1)
                        {
                            if (selections.SaveBatch() != ErrorCode::Success)
                            {
                                return false;
                            }
                        }
                    }
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

                if (p_opt.m_batches > 1)
                {
                    if (selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount) != ErrorCode::Success)
                    {
                        return false;
                    }
                }

                // Sort results either in CPU or GPU
                VectorIndex::SortSelections(&selections.m_selections);

                auto t3 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

                int postingSizeLimit = INT_MAX;
                if (p_opt.m_postingPageLimit > 0)
                {
                    const size_t minimumBytes = UseRaBitQBatchPosting()
                        ? RaBitQBatchPostingBytes(
                            static_cast<size_t>(p_opt.m_postingVectorLimit))
                        : static_cast<size_t>(p_opt.m_postingVectorLimit) *
                            vectorInfoSize;
                    p_opt.m_postingPageLimit = max(
                        p_opt.m_postingPageLimit,
                        static_cast<int>((minimumBytes + PageSize - 1) / PageSize));
                    p_opt.m_searchPostingPageLimit = p_opt.m_postingPageLimit;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build index with posting page limit:%d\n", p_opt.m_postingPageLimit);
                    postingSizeLimit = UseRaBitQBatchPosting()
                        ? static_cast<int>(
                            p_opt.m_postingPageLimit * PageSize / vectorInfoSize *
                            m_splitBatchLayout.batchSize)
                        : static_cast<int>(
                            p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);

                {
                    std::vector<SizeType> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    for (SizeType i = 0; i < replicaCount.size(); ++i)
                    {
                        ++replicaCountDist[replicaCount[i]];
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %lld\n", i, (std::int64_t)replicaCountDist[i]);
                    }
                }
                {
                    std::vector<std::thread> mythreads;
                    mythreads.reserve(m_opt->m_iSSDNumberOfThreads);
                    std::atomic_size_t sent(0);
                    for (int tid = 0; tid < m_opt->m_iSSDNumberOfThreads; tid++)
                    {
                        mythreads.emplace_back([&, tid]() {
                            size_t i = 0;
                            while (true)
                            {
                                i = sent.fetch_add(1);
                                if (i < postingListSize.size())
                                {
                                    if (postingListSize[i] <= postingSizeLimit)
                                        continue;

                                    std::size_t selectIdx =
                                        std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(),
                                                         i, Selection::g_edgeComparer) -
                                        selections.m_selections.begin();

                                    for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                                    {
                                        int tonode = selections.m_selections[selectIdx + dropID].tonode;
                                        --replicaCount[tonode];
                                    }
                                    postingListSize[i] = postingSizeLimit;
                                }
                                else
                                {
                                    return;
                                }
                            }
                        });
                    }
                    for (auto &t : mythreads)
                    {
                        t.join();
                    }
                    mythreads.clear();
                }
                if (p_opt.m_outputEmptyReplicaID)
                {
                    std::vector<SizeType> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                        return false;
                    }
                    for (SizeType i = 0; i < replicaCount.size(); ++i)
                    {
                        ++replicaCountDist[replicaCount[i]];
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %lld\n", i, (std::int64_t)(replicaCountDist[i]));
                    }
                }

                auto t4 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

                // number of posting lists per file
                size_t postingFileSize = (postingListSize.size() + p_opt.m_ssdIndexFileNum - 1) / p_opt.m_ssdIndexFileNum;
                std::vector<size_t> selectionsBatchOffset(p_opt.m_ssdIndexFileNum + 1, 0);
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    selectionsBatchOffset[i + 1] = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), (SizeType)curPostingListEnd, Selection::g_edgeComparer) - selections.m_selections.begin();
                }

                if (p_opt.m_ssdIndexFileNum > 1)
                {
                    if (selections.SaveBatch() != ErrorCode::Success)
                    {
                        return false;
                    }
                }

                auto fullVectors = p_reader->GetVectorSet();
                if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                // iterate over files
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListOffSet = i * postingFileSize;
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    // postingListSize: number of vectors in the posting list, type vector<int>
                    std::vector<int> curPostingListSizes(
                        postingListSize.begin() + curPostingListOffSet,
                        postingListSize.begin() + curPostingListEnd);

                    std::vector<size_t> curPostingListBytes(curPostingListSizes.size());
                    
                    if (p_opt.m_ssdIndexFileNum > 1)
                    {
                        if (selections.LoadBatch(selectionsBatchOffset[i], selectionsBatchOffset[i + 1]) != ErrorCode::Success)
                        {
                            return false;
                        }
                    }
                    // create compressor
                    if (p_opt.m_enableDataCompression && i == 0)
                    {
                        m_pCompressor = std::make_unique<Compressor>(p_opt.m_zstdCompressLevel, p_opt.m_dictBufferCapacity);
                        // train dict
                        if (p_opt.m_enableDictTraining) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Training dictionary...\n");
                            std::string samplesBuffer("");
                            std::vector<size_t> samplesSizes;
                            for (int j = 0; j < curPostingListSizes.size(); j++) {
                                if (curPostingListSizes[j] == 0) {
                                    continue;
                                }
                                ValueType* headVector = nullptr;
                                if (p_opt.m_enableDeltaEncoding)
                                {
                                    headVector = (ValueType*)p_headIndex->GetSample(j);
                                }
                                std::string postingListFullData = GetPostingListFullData(
                                    j, curPostingListSizes[j], selections, fullVectors, p_localToGlobal, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);

                                samplesBuffer += postingListFullData;
                                samplesSizes.push_back(postingListFullData.size());
                                if (samplesBuffer.size() > p_opt.m_minDictTraingBufferSize) break;
                            }
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using the first %zu postingLists to train dictionary... \n", samplesSizes.size());
                            std::size_t dictSize = m_pCompressor->TrainDict(samplesBuffer, &samplesSizes[0], (unsigned int)samplesSizes.size());
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Dictionary trained, dictionary size: %zu \n", dictSize);
                        }
                    }

                    if (p_opt.m_enableDataCompression) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Getting compressed size of each posting list...\n");
                        std::vector<std::thread> mythreads;
                        mythreads.reserve(m_opt->m_iSSDNumberOfThreads);
                        std::atomic_size_t sent(0);
                        for (int tid = 0; tid < m_opt->m_iSSDNumberOfThreads; tid++)
                        {
                            mythreads.emplace_back([&, tid]() {
                                size_t j = 0;
                                while (true)
                                {
                                    j = sent.fetch_add(1);
                                    if (j < curPostingListSizes.size())
                                    {
                                        SizeType postingListId = j + (SizeType)curPostingListOffSet;
                                        // do not compress if no data
                                        if (postingListSize[postingListId] == 0)
                                        {
                                            curPostingListBytes[j] = 0;
                                            continue;
                                        }
                                        ValueType *headVector = nullptr;
                                        if (p_opt.m_enableDeltaEncoding)
                                        {
                                            headVector = (ValueType *)p_headIndex->GetSample(postingListId);
                                        }
                                        std::string postingListFullData =
                                            GetPostingListFullData(postingListId, postingListSize[postingListId],
                                                                   selections, fullVectors, p_localToGlobal, p_opt.m_enableDeltaEncoding,
                                                                   p_opt.m_enablePostingListRearrange, headVector);
                                        size_t sizeToCompress = postingListSize[postingListId] * vectorInfoSize;
                                        if (sizeToCompress != postingListFullData.size())
                                        {
                                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                         "Size to compress NOT MATCH! PostingListFullData size: %zu "
                                                         "sizeToCompress: %zu \n",
                                                         postingListFullData.size(), sizeToCompress);
                                        }
                                        curPostingListBytes[j] = m_pCompressor->GetCompressedSize(
                                            postingListFullData, p_opt.m_enableDictTraining);
                                        if (postingListId % 10000 == 0 ||
                                            curPostingListBytes[j] >
                                                static_cast<uint64_t>(p_opt.m_postingPageLimit) * PageSize)
                                        {
                                            SPTAGLIB_LOG(
                                                Helper::LogLevel::LL_Info,
                                                "Posting list %lld/%zu, compressed size: %d, compression ratio: %.4f\n",
                                                (std::int64_t)postingListId, postingListSize.size(), curPostingListBytes[j],
                                                curPostingListBytes[j] / float(sizeToCompress));
                                        }
                                    }
                                    else
                                    {
                                        return;
                                    }
                                }
                            });
                        }
                        for (auto &t : mythreads)
                        {
                            t.join();
                        }
                        mythreads.clear();

                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Getted compressed size for all the %d posting lists in SSD Index file %d.\n", curPostingListBytes.size(), i);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mean compressed size: %.4f \n", std::accumulate(curPostingListBytes.begin(), curPostingListBytes.end(), 0.0) / curPostingListBytes.size());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mean compression ratio: %.4f \n", std::accumulate(curPostingListBytes.begin(), curPostingListBytes.end(), 0.0) / (std::accumulate(curPostingListSizes.begin(), curPostingListSizes.end(), 0.0) * vectorInfoSize));
                    }
                    else {
                        for (int j = 0; j < curPostingListSizes.size(); j++)
                        {
                            curPostingListBytes[j] = UseRaBitQBatchPosting()
                                ? RaBitQBatchPostingBytes(
                                    static_cast<size_t>(curPostingListSizes[j]))
                                : static_cast<size_t>(curPostingListSizes[j]) *
                                    vectorInfoSize;
                        }
                    }

                    std::unique_ptr<int[]> postPageNum;
                    std::unique_ptr<std::uint16_t[]> postPageOffset;
                    std::vector<int> postingOrderInIndex;
                    SelectPostingOffset(curPostingListBytes, postPageNum, postPageOffset, postingOrderInIndex);

                    OutputSSDIndexFile(
                        (i == 0)
                            ? outputFile
                            : UseRaBitQBatchPosting()
                                ? outputFile + ".part" + std::to_string(i)
                                : outputFile + "_" + std::to_string(i),
                        p_opt.m_enableDeltaEncoding,
                        p_opt.m_enablePostingListRearrange,
                        p_opt.m_enableDataCompression,
                        p_opt.m_enableDictTraining,
                        vectorInfoSize,
                        curPostingListSizes,
                        curPostingListBytes,
                        p_headIndex,
                        selections,
                        postPageNum,
                        postPageOffset,
                        postingOrderInIndex,
                        fullVectors, p_headToLocal, p_localToGlobal,
                        curPostingListOffSet,
                        i);
                }

                if (p_localToGlobal.R() > 0)
                {
                    p_headGlobaltoLocal.clear();
                    for (SizeType i = 0; i < p_headToLocal.R(); ++i)
                    {
                        const SizeType localID = *(p_headToLocal[i]);
                        if (localID < 0 || localID >= p_localToGlobal.R())
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Head ID %lld is outside the previous layer mapping of size %lld.\n",
                                static_cast<std::int64_t>(localID),
                                static_cast<std::int64_t>(p_localToGlobal.R()));
                            return false;
                        }

                        const SizeType globalID = *(p_localToGlobal[localID]);
                        *(p_headToLocal[i]) = globalID;
                        if (p_headGlobaltoLocal.find(globalID) !=
                            p_headGlobaltoLocal.end())
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Duplicate global head ID %lld in layer %d.\n",
                                static_cast<std::int64_t>(globalID),
                                m_layer);
                            return false;
                        }
                        p_headGlobaltoLocal[globalID] = i;
                    }

                    if (p_headToLocal.Save(
                            p_opt.m_indexDirectory + FolderSep +
                            p_opt.m_headIDFile) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Failed to save global head IDs for layer %d.\n",
                            m_layer);
                        return false;
                    }
                }

                auto t5 = std::chrono::high_resolution_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
                return true;
            }

            virtual ErrorCode CheckPosting(SizeType postingID, std::vector<std::uint8_t> *visited = nullptr,
                                           ExtraWorkSpace *p_exWorkSpace = nullptr) override
            {
                if (postingID < 0 || postingID >= m_totalListCount)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: Error postingID %d (should be 0 ~ %d)\n",
                                 postingID, m_totalListCount);
                    return ErrorCode::Key_OverFlow;
                }
                if (m_listInfos[postingID].listEleCount < 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: postingID %d has wrong size:%d\n",
                                 postingID, m_listInfos[postingID].listEleCount);
                    return ErrorCode::Posting_SizeError;
                }
                return ErrorCode::Success;
            }

        private:
            struct ListInfo
            {
                std::size_t listTotalBytes = 0;
                
                int listEleCount = 0;

                std::uint16_t listPageCount = 0;

                std::uint64_t listOffset = 0;

                std::uint16_t pageOffset = 0;

                int fileID = 0;

                std::uint64_t extendedOffset = 0;

                std::uint64_t extendedBytes = 0;

                std::uint64_t rawOffset = 0;

                std::uint64_t rawBytes = 0;

                std::vector<float> centroid;
            };

            enum class StaticPostingFormat : std::uint32_t
            {
                LegacyRaw = 0,
                RaBitQSplit = 1,
                RaBitQBatch = 2,
            };

            struct StaticPostingHeader
            {
                std::uint32_t magic = 0;
                std::uint32_t version = 0;
                std::uint32_t format = 0;
                std::uint32_t reserved = 0;
                SizeType listCount = 0;
                SizeType totalDocumentCount = 0;
                int dataDimension = 0;
                SizeType listPageOffset = 0;
                int vectorInfoSize = 0;
                std::uint64_t postingQuantizerFingerprint = 0;
            };

            struct StaticBatchPostingHeader
            {
                StaticPostingHeader base;
                std::uint32_t batchSize = 0;
                std::uint32_t batchRecordBytes = 0;
                std::uint32_t extendedBytesPerVector = 0;
                std::uint32_t flags = 0;
            };

            struct BatchRecordHeader
            {
                std::uint32_t validCount = 0;
                std::uint32_t reserved = 0;
            };

            enum class BatchSidecarKind : std::uint32_t
            {
                Extended = 1,
                Raw = 2,
            };

            struct BatchSidecarHeader
            {
                std::uint32_t magic = 0;
                std::uint32_t version = 0;
                std::uint32_t kind = 0;
                std::uint32_t layer = 0;
                std::uint32_t fileID = 0;
                std::uint32_t listCount = 0;
                std::uint32_t dimension = 0;
                std::uint32_t batchSize = 0;
                std::uint64_t postingQuantizerFingerprint = 0;
                std::uint64_t dataBytesPerRecord = 0;
            };

            struct BatchSidecarListInfo
            {
                std::uint64_t vectorCount = 0;
                std::uint64_t batchCount = 0;
                std::uint64_t centroidOffset = 0;
                std::uint64_t dataOffset = 0;
                std::uint64_t dataBytes = 0;
            };

            struct BatchPostingData
            {
                std::string binary;
                std::string extended;
                std::string raw;
                std::vector<float> centroid;
            };

            struct BatchCandidate
            {
                SizeType vectorID = -1;
                float distance = std::numeric_limits<float>::infinity();
                float upperBound = std::numeric_limits<float>::infinity();
                int fileID = 0;
                std::uint64_t rawOffset = 0;
            };

            struct QueryDiskStats
            {
                void RecordRead(size_t p_bytes)
                {
                    operationCount.fetch_add(1, std::memory_order_relaxed);
                    const size_t pages =
                        p_bytes / PageSize + (p_bytes % PageSize != 0);
                    pageCount.fetch_add(
                        static_cast<int>(pages), std::memory_order_relaxed);
                }

                std::atomic<int> operationCount{0};
                std::atomic<int> pageCount{0};
            };

            static constexpr std::uint32_t kStaticPostingHeaderMagic = 0x32545053U; // SPT2
            static constexpr std::uint32_t kStaticPostingHeaderVersion = 2U;
            static constexpr std::uint32_t kStaticBatchPostingHeaderVersion = 3U;
            static constexpr std::uint32_t kBatchSidecarMagic = 0x53425152U; // RQBS
            static constexpr std::uint32_t kBatchSidecarVersion = 1U;
            static constexpr const char* kPostingQuantizerIndexFile = "SPTAGPostingRaBitQQuantizer.bin";
            static constexpr const char* kBatchExtendedSuffix = ".rabitq.ext";
            static constexpr const char* kBatchRawSuffix = ".rabitq.raw";
            static constexpr std::uint32_t kBatchPostingRawFlag = 1U;

            bool UseRaBitQPosting() const
            {
                return m_postingFormat == StaticPostingFormat::RaBitQSplit ||
                    m_postingFormat == StaticPostingFormat::RaBitQBatch;
            }

            bool UseRaBitQBatchPosting() const
            {
                return m_postingFormat == StaticPostingFormat::RaBitQBatch;
            }

            size_t ExpectedRaBitQVectorInfoSize() const
            {
                return sizeof(SizeType) + m_splitCodeLayout.totalBytes;
            }

            size_t ExpectedRaBitQBatchRecordSize() const
            {
                return sizeof(BatchRecordHeader) +
                    m_splitBatchLayout.batchSize * sizeof(SizeType) +
                    m_splitBatchLayout.binaryBytes;
            }

            size_t RaBitQBatchCount(size_t p_vectorCount) const
            {
                return (p_vectorCount + m_splitBatchLayout.batchSize - 1) /
                    m_splitBatchLayout.batchSize;
            }

            size_t RaBitQBatchPostingBytes(size_t p_vectorCount) const
            {
                return RaBitQBatchCount(p_vectorCount) * ExpectedRaBitQBatchRecordSize();
            }

            std::string ResolvePostingQuantizerPath(const Options& p_opt) const
            {
                std::filesystem::path quantizerPath(p_opt.m_postingQuantizerFile);
                if (quantizerPath.is_absolute())
                {
                    return p_opt.m_postingQuantizerFile;
                }

                const auto indexLocalPath =
                    (std::filesystem::path(p_opt.m_indexDirectory) / quantizerPath).string();
                if (fileexists(indexLocalPath.c_str()))
                {
                    return indexLocalPath;
                }

                if (fileexists(p_opt.m_postingQuantizerFile.c_str()))
                {
                    return p_opt.m_postingQuantizerFile;
                }

                return indexLocalPath;
            }

            bool ReadPostingQuantizerBytes(const std::string& p_quantizerPath, std::vector<char>& p_bytes) const
            {
                std::error_code ec;
                const auto fileSize = std::filesystem::file_size(p_quantizerPath, ec);
                if (ec || fileSize == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Failed to stat PostingQuantizerFile %s.\n",
                                 p_quantizerPath.c_str());
                    return false;
                }

                auto quantizerFile = f_createIO();
                if (quantizerFile == nullptr ||
                    !quantizerFile->Initialize(p_quantizerPath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Failed to open PostingQuantizerFile %s.\n",
                                 p_quantizerPath.c_str());
                    return false;
                }

                p_bytes.resize(static_cast<size_t>(fileSize));
                if (quantizerFile->ReadBinary(fileSize, p_bytes.data()) != fileSize)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Failed to read PostingQuantizerFile %s.\n",
                                 p_quantizerPath.c_str());
                    p_bytes.clear();
                    return false;
                }

                return true;
            }

            static std::uint64_t ComputePostingQuantizerFingerprint(const std::vector<char>& p_bytes)
            {
                std::uint64_t fingerprint = 14695981039346656037ULL;
                for (char byte : p_bytes)
                {
                    fingerprint ^= static_cast<std::uint8_t>(byte);
                    fingerprint *= 1099511628211ULL;
                }
                return fingerprint;
            }

            bool LoadRaBitQBatchSidecars(
                const std::string& p_mainFile,
                int p_fileID,
                size_t p_firstList,
                size_t p_listCount)
            {
                auto loadSidecar =
                    [&](const std::string& p_path,
                        BatchSidecarKind p_kind,
                        std::uint64_t p_expectedRecordBytes,
                        std::shared_ptr<Helper::DiskIO>& p_input,
                        std::vector<BatchSidecarListInfo>& p_infos) -> bool
                {
                    std::error_code ec;
                    const std::uint64_t fileSize =
                        std::filesystem::file_size(p_path, ec);
                    const std::uint64_t metadataBytes =
                        sizeof(BatchSidecarHeader) +
                        p_listCount * sizeof(BatchSidecarListInfo);
                    if (ec || fileSize < metadataBytes)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Missing or truncated STATIC RaBitQ batch sidecar %s.\n",
                            p_path.c_str());
                        return false;
                    }

                    p_input = SPTAG::f_createIO();
                    if (p_input == nullptr ||
                        !p_input->Initialize(
                            p_path.c_str(), std::ios::binary | std::ios::in))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Failed to open STATIC RaBitQ batch sidecar %s.\n",
                            p_path.c_str());
                        return false;
                    }

                    BatchSidecarHeader header;
                    if (p_input->ReadBinary(
                            sizeof(header), reinterpret_cast<char*>(&header)) !=
                        sizeof(header) ||
                        header.magic != kBatchSidecarMagic ||
                        header.version != kBatchSidecarVersion ||
                        header.kind != static_cast<std::uint32_t>(p_kind) ||
                        header.layer != static_cast<std::uint32_t>(m_layer) ||
                        header.fileID != static_cast<std::uint32_t>(p_fileID) ||
                        header.listCount != p_listCount ||
                        header.dimension !=
                            static_cast<std::uint32_t>(m_opt->m_dim) ||
                        header.batchSize != m_splitBatchLayout.batchSize ||
                        header.postingQuantizerFingerprint !=
                            m_postingQuantizerFingerprint ||
                        header.dataBytesPerRecord != p_expectedRecordBytes)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Invalid STATIC RaBitQ batch sidecar header in %s.\n",
                            p_path.c_str());
                        return false;
                    }

                    p_infos.resize(p_listCount);
                    if (!p_infos.empty() &&
                        p_input->ReadBinary(
                            p_infos.size() * sizeof(BatchSidecarListInfo),
                            reinterpret_cast<char*>(p_infos.data())) !=
                            p_infos.size() * sizeof(BatchSidecarListInfo))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Failed to read STATIC RaBitQ batch sidecar metadata from %s.\n",
                            p_path.c_str());
                        return false;
                    }

                    for (size_t i = 0; i < p_infos.size(); ++i)
                    {
                        const auto& info = p_infos[i];
                        const ListInfo& list = m_listInfos[p_firstList + i];
                        const std::uint64_t expectedBatchCount =
                            RaBitQBatchCount(info.vectorCount);
                        const std::uint64_t expectedDataBytes =
                            p_kind == BatchSidecarKind::Extended
                            ? expectedBatchCount * m_splitBatchLayout.extendedBytes
                            : info.vectorCount *
                                static_cast<std::uint64_t>(m_opt->m_dim) *
                                sizeof(float);
                        if (info.vectorCount <
                                static_cast<std::uint64_t>(list.listEleCount) ||
                            info.batchCount != expectedBatchCount ||
                            info.dataBytes != expectedDataBytes ||
                            (info.dataBytes > 0 && info.dataOffset < metadataBytes) ||
                            info.dataOffset > fileSize ||
                            info.dataBytes > fileSize - info.dataOffset)
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Corrupt STATIC RaBitQ batch sidecar entry %zu in %s.\n",
                                i,
                                p_path.c_str());
                            return false;
                        }
                        if (p_kind == BatchSidecarKind::Extended &&
                            info.vectorCount > 0)
                        {
                            const std::uint64_t centroidBytes =
                                static_cast<std::uint64_t>(m_opt->m_dim) *
                                sizeof(float);
                            if (info.centroidOffset < metadataBytes ||
                                info.dataOffset != info.centroidOffset + centroidBytes ||
                                info.centroidOffset > fileSize ||
                                centroidBytes >
                                    fileSize - info.centroidOffset)
                            {
                                SPTAGLIB_LOG(
                                    Helper::LogLevel::LL_Error,
                                    "Corrupt STATIC RaBitQ centroid entry %zu in %s.\n",
                                    i,
                                    p_path.c_str());
                                return false;
                            }
                        }
                    }
                    std::vector<std::pair<std::uint64_t, std::uint64_t>> payloadExtents;
                    payloadExtents.reserve(p_infos.size());
                    for (const auto& info : p_infos)
                    {
                        if (info.vectorCount == 0)
                        {
                            continue;
                        }
                        const std::uint64_t begin =
                            p_kind == BatchSidecarKind::Extended
                            ? info.centroidOffset
                            : info.dataOffset;
                        const std::uint64_t centroidBytes =
                            p_kind == BatchSidecarKind::Extended
                            ? static_cast<std::uint64_t>(m_opt->m_dim) * sizeof(float)
                            : 0;
                        payloadExtents.emplace_back(
                            begin, info.dataOffset + info.dataBytes);
                        if (payloadExtents.back().second <
                            payloadExtents.back().first + centroidBytes)
                        {
                            return false;
                        }
                    }
                    std::sort(payloadExtents.begin(), payloadExtents.end());
                    for (size_t i = 1; i < payloadExtents.size(); ++i)
                    {
                        if (payloadExtents[i].first < payloadExtents[i - 1].second)
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Overlapping STATIC RaBitQ batch payloads in %s.\n",
                                p_path.c_str());
                            return false;
                        }
                    }
                    return true;
                };

                std::shared_ptr<Helper::DiskIO> extended;
                std::vector<BatchSidecarListInfo> extendedInfos;
                if (!loadSidecar(
                        p_mainFile + kBatchExtendedSuffix,
                        BatchSidecarKind::Extended,
                        m_splitBatchLayout.extendedBytes,
                        extended,
                        extendedInfos))
                {
                    return false;
                }

                std::shared_ptr<Helper::DiskIO> raw;
                std::vector<BatchSidecarListInfo> rawInfos;
                if (m_opt->m_postingRaBitQRerank > 0 &&
                    !loadSidecar(
                        p_mainFile + kBatchRawSuffix,
                        BatchSidecarKind::Raw,
                        static_cast<std::uint64_t>(m_opt->m_dim) * sizeof(float),
                        raw,
                        rawInfos))
                {
                    return false;
                }

                for (size_t i = 0; i < p_listCount; ++i)
                {
                    ListInfo& list = m_listInfos[p_firstList + i];
                    const auto& extendedInfo = extendedInfos[i];
                    list.extendedOffset = extendedInfo.dataOffset;
                    list.extendedBytes = extendedInfo.dataBytes;
                    if (extendedInfo.vectorCount > 0)
                    {
                        list.centroid.resize(static_cast<size_t>(m_opt->m_dim));
                        const size_t centroidBytes =
                            list.centroid.size() * sizeof(float);
                        if (extended->ReadBinary(
                                centroidBytes,
                                reinterpret_cast<char*>(list.centroid.data()),
                                extendedInfo.centroidOffset) != centroidBytes)
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Failed to read STATIC RaBitQ centroid from %s.\n",
                                p_mainFile.c_str());
                            return false;
                        }
                    }
                    if (raw != nullptr)
                    {
                        list.rawOffset = rawInfos[i].dataOffset;
                        list.rawBytes = rawInfos[i].dataBytes;
                    }
                }

                m_extendedSidecars.emplace_back(std::move(extended));
                m_rawSidecars.emplace_back(std::move(raw));
                m_sidecarReadLocks.emplace_back(std::make_unique<std::mutex>());
                return true;
            }

            bool LoadPostingQuantizer(const std::string& p_quantizerPath)
            {
                std::vector<char> quantizerBytes;
                if (!ReadPostingQuantizerBytes(p_quantizerPath, quantizerBytes))
                {
                    return false;
                }

                auto quantizerFile = f_createIO();
                if (quantizerFile == nullptr ||
                    !quantizerFile->Initialize(
                        p_quantizerPath.c_str(),
                        std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Failed to open PostingQuantizerFile %s for checked loading.\n",
                        p_quantizerPath.c_str());
                    return false;
                }
                auto quantizer =
                    COMMON::IQuantizer::LoadIQuantizer(quantizerFile);
                if (quantizer == nullptr ||
                    quantizer->GetQuantizerType() != QuantizerType::RaBitQQuantizer)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "PostingQuantizerFile %s is not a supported RaBitQ model.\n",
                                 p_quantizerPath.c_str());
                    return false;
                }

                m_postingQuantizer = std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(quantizer);
                if (m_postingQuantizer == nullptr || !m_postingQuantizer->Ready())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "PostingQuantizerFile %s did not load as a ready RaBitQ model.\n",
                                 p_quantizerPath.c_str());
                    m_postingQuantizer.reset();
                    return false;
                }

                m_postingQuantizerBytes = std::move(quantizerBytes);
                m_postingQuantizerFingerprint = ComputePostingQuantizerFingerprint(m_postingQuantizerBytes);
                return true;
            }

            bool PersistPostingQuantizerCopy(Options& p_opt)
            {
                if (!UseRaBitQPosting())
                {
                    return true;
                }

                if (m_postingQuantizerBytes.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ has no serialized model to persist.\n");
                    return false;
                }

                std::error_code ec;
                if (!p_opt.m_indexDirectory.empty())
                {
                    std::filesystem::create_directories(p_opt.m_indexDirectory, ec);
                }
                if (ec)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Failed to create index directory %s for PostingQuantizer copy.\n",
                                 p_opt.m_indexDirectory.c_str());
                    return false;
                }

                const auto targetPath =
                    (std::filesystem::path(p_opt.m_indexDirectory) / kPostingQuantizerIndexFile).string();
                auto output = f_createIO();
                if (output == nullptr ||
                    !output->Initialize(targetPath.c_str(), std::ios::binary | std::ios::out))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Failed to create index-local PostingQuantizerFile %s.\n",
                                 targetPath.c_str());
                    return false;
                }

                const auto serializedBytes = static_cast<std::uint64_t>(m_postingQuantizerBytes.size());
                if (output->WriteBinary(serializedBytes, m_postingQuantizerBytes.data()) != serializedBytes)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Failed to persist index-local PostingQuantizerFile %s.\n",
                                 targetPath.c_str());
                    return false;
                }

                p_opt.m_postingQuantizerFile = kPostingQuantizerIndexFile;
                return true;
            }

            bool ConfigurePostingFormat(Options& p_opt)
            {
                m_postingFormat = StaticPostingFormat::LegacyRaw;
                m_postingQuantizer.reset();
                m_postingQuantizerBytes.clear();
                m_postingQuantizerFingerprint = 0;
                m_splitCodeLayout = {};
                m_splitBatchLayout = {};

                const bool useLegacyRaBitQ = Helper::StrUtils::StrEqualIgnoreCase(
                    p_opt.m_postingQuantizer.c_str(), "RaBitQ");
                const bool useBatchRaBitQ = Helper::StrUtils::StrEqualIgnoreCase(
                    p_opt.m_postingQuantizer.c_str(), "RaBitQBatch");
                if (!useLegacyRaBitQ && !useBatchRaBitQ)
                {
                    if (p_opt.m_postingRaBitQRerank > 0)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "PostingRaBitQRerank requires STATIC PostingQuantizer=RaBitQBatch.\n");
                        return false;
                    }
                    return true;
                }

                if (GetEnumValueType<ValueType>() != VectorValueType::Float ||
                    p_opt.m_valueType != VectorValueType::Float)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ requires Float base/head vectors.\n");
                    return false;
                }

                if (!p_opt.m_quantizerFilePath.empty() ||
                    m_headIndex == nullptr ||
                    m_headIndex->m_pQuantizer != nullptr)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ does not support a global quantizer.\n");
                    return false;
                }

                if (p_opt.m_enableDeltaEncoding)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ does not support delta encoding.\n");
                    return false;
                }

                if (p_opt.m_enablePostingListRearrange)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ does not support posting rearrangement.\n");
                    return false;
                }

                if (useBatchRaBitQ && p_opt.m_enableDataCompression)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "STATIC PostingQuantizer=RaBitQBatch does not support posting compression.\n");
                    return false;
                }

                if (p_opt.m_rerank > 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC RaBitQ posting formats do not alter legacy Rerank semantics; set Rerank=0 and use PostingRaBitQRerank with RaBitQBatch.\n");
                    return false;
                }

                if (p_opt.m_postingRaBitQRerank < 0 ||
                    (p_opt.m_postingRaBitQRerank > 0 &&
                     p_opt.m_postingRaBitQRerank < p_opt.m_searchInternalResultNum))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "PostingRaBitQRerank must be 0 or at least SearchInternalResultNum (%d).\n",
                        p_opt.m_searchInternalResultNum);
                    return false;
                }
                if (useLegacyRaBitQ && p_opt.m_postingRaBitQRerank > 0)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "PostingRaBitQRerank is supported only by PostingQuantizer=RaBitQBatch.\n");
                    return false;
                }

                if (p_opt.m_postingQuantizerFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ requires PostingQuantizerFile to point to a v3 RaBitQ model.\n");
                    return false;
                }

                const std::string postingQuantizerPath = ResolvePostingQuantizerPath(p_opt);
                if (!LoadPostingQuantizer(postingQuantizerPath)) return false;

                if (m_postingQuantizer->Dimension() != p_opt.m_dim)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "STATIC PostingQuantizer=RaBitQ dimension mismatch: model=%d index=%d.\n",
                        m_postingQuantizer->Dimension(),
                        p_opt.m_dim);
                    return false;
                }

                if (m_postingQuantizer->GetMetric() != p_opt.m_distCalcMethod)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "STATIC PostingQuantizer=RaBitQ metric mismatch: model=%s index=%s.\n",
                        Helper::Convert::ConvertToString(m_postingQuantizer->GetMetric()).c_str(),
                        Helper::Convert::ConvertToString(p_opt.m_distCalcMethod).c_str());
                    return false;
                }

                m_splitCodeLayout = m_postingQuantizer->GetSplitCodeLayout();
                if (m_splitCodeLayout.binaryBytes == 0 || m_splitCodeLayout.totalBytes == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ produced an empty split-code layout.\n");
                    return false;
                }

                if (useBatchRaBitQ)
                {
                    m_splitBatchLayout = m_postingQuantizer->GetSplitBatchLayout();
                    if (m_splitBatchLayout.batchSize == 0 ||
                        m_splitBatchLayout.binaryBytes == 0 ||
                        ExpectedRaBitQBatchRecordSize() >
                            static_cast<size_t>((std::numeric_limits<int>::max)()))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "STATIC PostingQuantizer=RaBitQBatch produced an invalid batch layout.\n");
                        return false;
                    }
                    m_postingFormat = StaticPostingFormat::RaBitQBatch;
                }
                else
                {
                    m_postingFormat = StaticPostingFormat::RaBitQSplit;
                }
                return true;
            }

            BatchPostingData BuildRaBitQBatchPosting(
                int p_postingListID,
                size_t p_postingListSize,
                Selection& p_selections,
                const std::shared_ptr<VectorSet>& p_fullVectors,
                COMMON::Dataset<SizeType>& p_localToGlobal) const
            {
                if (!UseRaBitQBatchPosting() || p_fullVectors == nullptr)
                {
                    throw std::runtime_error("Invalid STATIC RaBitQ batch posting build");
                }

                BatchPostingData result;
                result.centroid.assign(static_cast<size_t>(m_opt->m_dim), 0.0F);
                std::vector<SizeType> storedIDs(p_postingListSize);
                std::vector<float> vectors(
                    p_postingListSize * static_cast<size_t>(m_opt->m_dim));
                size_t selectionIndex = p_selections.lower_bound(p_postingListID);
                for (size_t i = 0; i < p_postingListSize; ++i)
                {
                    const Edge& selection = p_selections[selectionIndex++];
                    if (selection.node != p_postingListID)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Selection ID NOT MATCH! node:%d offset:%zu\n",
                            p_postingListID,
                            selectionIndex - 1);
                        throw std::runtime_error("Selection ID mismatch");
                    }

                    const SizeType localVID = selection.tonode;
                    const auto* vector = reinterpret_cast<const float*>(
                        p_fullVectors->GetVector(localVID));
                    if (vector == nullptr)
                    {
                        throw std::runtime_error("Missing STATIC RaBitQ batch source vector");
                    }
                    std::copy(
                        vector,
                        vector + m_opt->m_dim,
                        vectors.begin() + i * static_cast<size_t>(m_opt->m_dim));
                    for (DimensionType dim = 0; dim < m_opt->m_dim; ++dim)
                    {
                        result.centroid[static_cast<size_t>(dim)] += vector[dim];
                    }

                    storedIDs[i] = localVID;
                    if (p_localToGlobal.R() > 0)
                    {
                        storedIDs[i] = *(p_localToGlobal[localVID]);
                    }
                }

                if (p_postingListSize > 0)
                {
                    const float scale = 1.0F / static_cast<float>(p_postingListSize);
                    for (float& value : result.centroid) value *= scale;
                }

                const size_t batchCount = RaBitQBatchCount(p_postingListSize);
                result.binary.reserve(batchCount * ExpectedRaBitQBatchRecordSize());
                result.extended.reserve(batchCount * m_splitBatchLayout.extendedBytes);
                if (m_opt->m_postingRaBitQRerank > 0)
                {
                    result.raw.assign(
                        reinterpret_cast<const char*>(vectors.data()),
                        vectors.size() * sizeof(float));
                }

                for (size_t batch = 0; batch < batchCount; ++batch)
                {
                    const size_t first = batch * m_splitBatchLayout.batchSize;
                    const size_t validCount = std::min(
                        m_splitBatchLayout.batchSize, p_postingListSize - first);
                    std::vector<std::uint8_t> binary(
                        m_splitBatchLayout.binaryBytes, 0);
                    std::vector<std::uint8_t> extended(
                        m_splitBatchLayout.extendedBytes, 0);
                    size_t encodedCount = 0;
                    if (m_postingQuantizer->QuantizeSplitBatch(
                            vectors.data() + first * static_cast<size_t>(m_opt->m_dim),
                            validCount,
                            result.centroid.data(),
                            binary.data(),
                            extended.empty() ? nullptr : extended.data(),
                            encodedCount) != ErrorCode::Success ||
                        encodedCount != validCount)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Failed to encode STATIC RaBitQ batch posting %d batch %zu.\n",
                            p_postingListID,
                            batch);
                        throw std::runtime_error("Failed STATIC RaBitQ batch encoding");
                    }

                    BatchRecordHeader recordHeader;
                    recordHeader.validCount = static_cast<std::uint32_t>(validCount);
                    result.binary.append(
                        reinterpret_cast<const char*>(&recordHeader),
                        sizeof(recordHeader));
                    for (size_t i = 0; i < m_splitBatchLayout.batchSize; ++i)
                    {
                        const SizeType id = i < validCount ? storedIDs[first + i] : -1;
                        result.binary.append(
                            reinterpret_cast<const char*>(&id), sizeof(id));
                    }
                    result.binary.append(
                        reinterpret_cast<const char*>(binary.data()), binary.size());
                    result.extended.append(
                        reinterpret_cast<const char*>(extended.data()), extended.size());
                }
                return result;
            }

            ErrorCode ProcessRaBitQPostingList(
                SizeType p_localPostingID,
                ExtraWorkSpace* p_exWorkSpace,
                COMMON::QueryResultSet<ValueType>& p_queryResults,
                ListInfo* p_listInfo,
                char* p_postingListFullData,
                std::vector<BasicResult>* p_results = nullptr,
                std::vector<BatchCandidate>* p_batchCandidates = nullptr,
                QueryDiskStats* p_diskStats = nullptr) const
            {
                if (!UseRaBitQPosting() || m_postingQuantizer == nullptr || p_exWorkSpace == nullptr ||
                    p_listInfo == nullptr || p_postingListFullData == nullptr)
                {
                    return ErrorCode::Fail;
                }
                if (UseRaBitQBatchPosting())
                {
                    return ProcessRaBitQBatchPostingList(
                        p_exWorkSpace,
                        p_queryResults,
                        p_listInfo,
                        p_postingListFullData,
                        p_results,
                        p_batchCandidates,
                        p_diskStats);
                }
                if (p_queryResults.WithVec())
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "STATIC PostingQuantizer=RaBitQ does not support returning reconstructed vectors.\n");
                    return ErrorCode::Undefined;
                }

                auto memoryIndex = m_headIndex->GetMemoryIndex();
                if (memoryIndex == nullptr)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ could not access the head index.\n");
                    return ErrorCode::Fail;
                }

                const auto* localCentroid =
                    reinterpret_cast<const float*>(memoryIndex->GetSample(p_localPostingID));
                if (localCentroid == nullptr)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ could not find centroid for posting %d.\n",
                                 p_localPostingID);
                    return ErrorCode::Fail;
                }

                COMMON::RaBitQQuantizer::SplitQueryContext queryContext;
                if (m_postingQuantizer->PrepareSplitQueryContext(
                        reinterpret_cast<const float*>(p_queryResults.GetTarget()),
                        localCentroid,
                        queryContext) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "STATIC PostingQuantizer=RaBitQ failed to prepare query context for posting %d.\n",
                                 p_localPostingID);
                    return ErrorCode::Fail;
                }

                for (int i = 0; i < p_listInfo->listEleCount; ++i)
                {
                    const char* record = p_postingListFullData +
                        static_cast<size_t>(i) * static_cast<size_t>(m_vectorInfoSize);
                    SizeType vectorID = -1;
                    std::memcpy(&vectorID, record, sizeof(SizeType));

                    const auto* binaryCode = reinterpret_cast<const std::uint8_t*>(record + sizeof(SizeType));
                    const auto oneBitEstimate =
                        m_postingQuantizer->EstimateSplitDistance(queryContext, binaryCode);

                    {
                        std::lock_guard<std::mutex> guard(p_exWorkSpace->RaBitQPruningLock());
                        if (p_exWorkSpace->Deduper().CheckAndSet(vectorID)) continue;
                        if (p_results == nullptr &&
                            p_exWorkSpace->CanPruneRaBitQCandidateUnlocked(oneBitEstimate.lowerBound))
                        {
                            continue;
                        }
                    }

                    auto estimate = oneBitEstimate;
                    if (m_splitCodeLayout.extendedBytes > 0)
                    {
                        const auto* extendedCode =
                            binaryCode + m_splitCodeLayout.binaryBytes;
                        estimate = m_postingQuantizer->EstimateSplitDistance(
                            queryContext, binaryCode, extendedCode);
                    }

                    std::lock_guard<std::mutex> guard(p_exWorkSpace->RaBitQPruningLock());
                    if (p_results == nullptr)
                    {
                        if (p_queryResults.AddPoint(vectorID, estimate.distance, ByteArray::c_empty))
                        {
                            p_exWorkSpace->RecordRaBitQCandidateUnlocked(
                                vectorID, estimate.distance, estimate.upperBound);
                        }
                    }
                    else
                    {
                        p_results->emplace_back(
                            vectorID, estimate.distance, ByteArray::c_empty, ByteArray::c_empty);
                    }
                }

                return ErrorCode::Success;
            }

            ErrorCode ReadBatchSidecar(
                const std::vector<std::shared_ptr<Helper::DiskIO>>& p_sidecars,
                int p_fileID,
                std::uint64_t p_offset,
                size_t p_bytes,
                char* p_output,
                QueryDiskStats* p_diskStats) const
            {
                if (p_fileID < 0 ||
                    static_cast<size_t>(p_fileID) >= p_sidecars.size() ||
                    p_sidecars[static_cast<size_t>(p_fileID)] == nullptr ||
                    p_output == nullptr)
                {
                    return ErrorCode::DiskIOFail;
                }
                std::lock_guard<std::mutex> guard(
                    *m_sidecarReadLocks[static_cast<size_t>(p_fileID)]);
                if (p_diskStats != nullptr)
                {
                    p_diskStats->RecordRead(p_bytes);
                }
                return p_sidecars[static_cast<size_t>(p_fileID)]->ReadBinary(
                           p_bytes, p_output, p_offset) == p_bytes
                    ? ErrorCode::Success
                    : ErrorCode::DiskIOFail;
            }

            ErrorCode ProcessRaBitQBatchPostingList(
                ExtraWorkSpace* p_exWorkSpace,
                COMMON::QueryResultSet<ValueType>& p_queryResults,
                ListInfo* p_listInfo,
                const char* p_postingListFullData,
                std::vector<BasicResult>* p_results,
                std::vector<BatchCandidate>* p_batchCandidates,
                QueryDiskStats* p_diskStats) const
            {
                if (p_listInfo->listEleCount == 0)
                {
                    return ErrorCode::Success;
                }
                if (p_listInfo->centroid.size() !=
                    static_cast<size_t>(m_opt->m_dim))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Missing STATIC RaBitQ batch centroid for posting.\n");
                    return ErrorCode::DiskIOFail;
                }
                if (p_queryResults.WithVec() &&
                    m_opt->m_postingRaBitQRerank <= 0)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "STATIC PostingQuantizer=RaBitQBatch requires PostingRaBitQRerank>0 for WithVec.\n");
                    return ErrorCode::Undefined;
                }

                COMMON::RaBitQQuantizer::SplitBatchQueryContext queryContext;
                if (m_postingQuantizer->PrepareSplitBatchQueryContext(
                        reinterpret_cast<const float*>(p_queryResults.GetTarget()),
                        p_listInfo->centroid.data(),
                        queryContext) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Failed to prepare STATIC RaBitQ batch query context.\n");
                    return ErrorCode::Fail;
                }

                const size_t batchCount =
                    RaBitQBatchCount(p_listInfo->listEleCount);
                size_t remaining = static_cast<size_t>(p_listInfo->listEleCount);
                for (size_t batch = 0; batch < batchCount; ++batch)
                {
                    const char* record = p_postingListFullData +
                        batch * ExpectedRaBitQBatchRecordSize();
                    BatchRecordHeader recordHeader;
                    std::memcpy(&recordHeader, record, sizeof(recordHeader));
                    const size_t expectedValid = std::min(
                        m_splitBatchLayout.batchSize, remaining);
                    if (recordHeader.validCount != expectedValid ||
                        recordHeader.reserved != 0)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Corrupt STATIC RaBitQ batch valid-count metadata.\n");
                        return ErrorCode::DiskIOFail;
                    }
                    remaining -= expectedValid;

                    const char* ids = record + sizeof(BatchRecordHeader);
                    const auto* binary = reinterpret_cast<const std::uint8_t*>(
                        ids + m_splitBatchLayout.batchSize * sizeof(SizeType));
                    COMMON::RaBitQQuantizer::SplitBatchDistanceEstimates estimates;
                    if (m_postingQuantizer->EstimateSplitBatchDistances(
                            queryContext,
                            binary,
                            expectedValid,
                            estimates) != ErrorCode::Success)
                    {
                        return ErrorCode::Fail;
                    }

                    struct Survivor
                    {
                        size_t index;
                        SizeType vectorID;
                    };
                    std::vector<Survivor> survivors;
                    survivors.reserve(expectedValid);
                    for (size_t i = 0; i < expectedValid; ++i)
                    {
                        SizeType vectorID = -1;
                        std::memcpy(
                            &vectorID,
                            ids + i * sizeof(SizeType),
                            sizeof(SizeType));
                        std::lock_guard<std::mutex> guard(
                            p_exWorkSpace->RaBitQPruningLock());
                        if (vectorID < 0 ||
                            p_exWorkSpace->Deduper().CheckAndSet(vectorID))
                        {
                            continue;
                        }
                        if (p_results == nullptr &&
                            p_exWorkSpace->CanPruneRaBitQCandidateUnlocked(
                                estimates.lowerBounds[i]))
                        {
                            continue;
                        }
                        survivors.push_back({i, vectorID});
                    }

                    if (survivors.empty()) continue;

                    std::vector<std::uint8_t> extended;
                    if (m_splitBatchLayout.extendedBytes > 0)
                    {
                        const std::uint64_t relativeOffset =
                            batch * m_splitBatchLayout.extendedBytes;
                        if (relativeOffset > p_listInfo->extendedBytes ||
                            m_splitBatchLayout.extendedBytes >
                                p_listInfo->extendedBytes - relativeOffset)
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Corrupt STATIC RaBitQ extended sidecar bounds.\n");
                            return ErrorCode::DiskIOFail;
                        }
                        extended.resize(m_splitBatchLayout.extendedBytes);
                        if (ReadBatchSidecar(
                                m_extendedSidecars,
                                p_listInfo->fileID,
                                p_listInfo->extendedOffset + relativeOffset,
                                extended.size(),
                                reinterpret_cast<char*>(extended.data()),
                                p_diskStats) !=
                            ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Failed to read STATIC RaBitQ extended batch.\n");
                            return ErrorCode::DiskIOFail;
                        }
                    }

                    for (const auto& survivor : survivors)
                    {
                        COMMON::RaBitQQuantizer::SplitDistanceEstimate estimate;
                        if (m_postingQuantizer->BoostSplitBatchDistance(
                                queryContext,
                                binary,
                                extended.empty() ? nullptr : extended.data(),
                                expectedValid,
                                survivor.index,
                                estimate) != ErrorCode::Success)
                        {
                            return ErrorCode::Fail;
                        }

                        std::lock_guard<std::mutex> guard(
                            p_exWorkSpace->RaBitQPruningLock());
                        if (p_batchCandidates != nullptr)
                        {
                            BatchCandidate candidate;
                            candidate.vectorID = survivor.vectorID;
                            candidate.distance = estimate.distance;
                            candidate.upperBound = estimate.upperBound;
                            candidate.fileID = p_listInfo->fileID;
                            candidate.rawOffset = p_listInfo->rawOffset +
                                (batch * m_splitBatchLayout.batchSize +
                                 survivor.index) *
                                    static_cast<std::uint64_t>(m_opt->m_dim) *
                                    sizeof(float);
                            p_batchCandidates->emplace_back(candidate);
                        }
                        else if (p_results != nullptr)
                        {
                            p_results->emplace_back(
                                survivor.vectorID,
                                estimate.distance,
                                ByteArray::c_empty,
                                ByteArray::c_empty);
                        }
                        else if (p_queryResults.AddPoint(
                                     survivor.vectorID,
                                     estimate.distance,
                                     ByteArray::c_empty))
                        {
                            p_exWorkSpace->RecordRaBitQCandidateUnlocked(
                                survivor.vectorID,
                                estimate.distance,
                                estimate.upperBound);
                        }
                    }
                }
                return ErrorCode::Success;
            }

            ErrorCode FinalizeRaBitQBatchRerank(
                std::vector<BatchCandidate>& p_candidates,
                ExtraWorkSpace* p_exWorkSpace,
                COMMON::QueryResultSet<ValueType>& p_queryResults,
                std::vector<BasicResult>* p_results = nullptr,
                QueryDiskStats* p_diskStats = nullptr) const
            {
                if (m_opt->m_postingRaBitQRerank <= 0) return ErrorCode::Success;
                std::sort(
                    p_candidates.begin(),
                    p_candidates.end(),
                    [](const BatchCandidate& left, const BatchCandidate& right)
                    {
                        return left.distance < right.distance ||
                            (left.distance == right.distance &&
                             left.vectorID < right.vectorID);
                    });
                if (p_candidates.size() >
                    static_cast<size_t>(m_opt->m_postingRaBitQRerank))
                {
                    p_candidates.resize(
                        static_cast<size_t>(m_opt->m_postingRaBitQRerank));
                }

                std::vector<float> raw(static_cast<size_t>(m_opt->m_dim));
                const auto* query =
                    reinterpret_cast<const float*>(p_queryResults.GetTarget());
                for (const auto& candidate : p_candidates)
                {
                    if (ReadBatchSidecar(
                            m_rawSidecars,
                            candidate.fileID,
                            candidate.rawOffset,
                            raw.size() * sizeof(float),
                            reinterpret_cast<char*>(raw.data()),
                            p_diskStats) !=
                        ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Failed to read STATIC RaBitQ raw rerank vector.\n");
                        return ErrorCode::DiskIOFail;
                    }
                    const float exactDistance =
                        COMMON::DistanceUtils::ComputeDistance(
                            query,
                            raw.data(),
                            m_opt->m_dim,
                            m_opt->m_distCalcMethod);
                    ByteArray vector = p_queryResults.WithVec()
                        ? ByteArray::Alloc(
                            reinterpret_cast<std::uint8_t*>(raw.data()),
                            raw.size() * sizeof(float))
                        : ByteArray::c_empty;
                    if (p_results != nullptr)
                    {
                        p_results->emplace_back(
                            candidate.vectorID,
                            exactDistance,
                            ByteArray::c_empty,
                            vector);
                    }
                    else if (p_queryResults.AddPoint(
                                 candidate.vectorID, exactDistance, vector))
                    {
                        p_exWorkSpace->RecordRaBitQCandidate(
                            candidate.vectorID, exactDistance, exactDistance);
                    }
                }
                return ErrorCode::Success;
            }

            int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, std::vector<ListInfo>& p_listInfos)
            {
                std::error_code fileSizeError;
                const std::uintmax_t rawFileSize =
                    std::filesystem::file_size(p_file, fileSizeError);
                if (fileSizeError || rawFileSize == 0 ||
                    rawFileSize > std::numeric_limits<std::uint64_t>::max())
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Failed to stat posting file %s.\n",
                        p_file.c_str());
                    throw std::runtime_error(
                        "Failed stat file in LoadingHeadInfo");
                }
                const std::uint64_t fileSize =
                    static_cast<std::uint64_t>(rawFileSize);

                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                    throw std::runtime_error("Failed open file in LoadingHeadInfo");
                }
                m_pCompressor = std::make_unique<Compressor>(); // no need compress level to decompress

                SizeType m_listCount;
                SizeType m_totalDocumentCount;
                SizeType m_listPageOffset;
                if (UseRaBitQPosting())
                {
                    StaticPostingHeader header;
                    if (ptr->ReadBinary(sizeof(header), reinterpret_cast<char*>(&header)) != sizeof(header))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read RaBitQ posting header.\n");
                        throw std::runtime_error("Failed read RaBitQ posting header");
                    }

                    const std::uint32_t expectedVersion = UseRaBitQBatchPosting()
                        ? kStaticBatchPostingHeaderVersion
                        : kStaticPostingHeaderVersion;
                    if (header.magic != kStaticPostingHeaderMagic ||
                        header.version != expectedVersion ||
                        header.format != static_cast<std::uint32_t>(m_postingFormat) ||
                        header.reserved != 0)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Unsupported STATIC RaBitQ posting file format in %s.\n",
                                     p_file.c_str());
                        throw std::runtime_error("Unsupported STATIC RaBitQ posting file format");
                    }

                    if (UseRaBitQBatchPosting())
                    {
                        StaticBatchPostingHeader batchHeader;
                        batchHeader.base = header;
                        constexpr size_t extraHeaderBytes =
                            sizeof(StaticBatchPostingHeader) -
                            sizeof(StaticPostingHeader);
                        if (ptr->ReadBinary(
                                extraHeaderBytes,
                                reinterpret_cast<char*>(&batchHeader) +
                                    sizeof(StaticPostingHeader)) !=
                            extraHeaderBytes)
                        {
                            throw std::runtime_error(
                                "Failed read STATIC RaBitQ batch posting header");
                        }
                        const std::uint32_t expectedFlags =
                            m_opt->m_postingRaBitQRerank > 0
                            ? kBatchPostingRawFlag
                            : 0U;
                        if (batchHeader.batchSize != m_splitBatchLayout.batchSize ||
                            batchHeader.batchRecordBytes !=
                                ExpectedRaBitQBatchRecordSize() ||
                            batchHeader.extendedBytesPerVector !=
                                m_splitBatchLayout.extendedBytesPerVector ||
                            batchHeader.flags != expectedFlags)
                        {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "STATIC RaBitQ batch layout/options mismatch in %s.\n",
                                p_file.c_str());
                            throw std::runtime_error(
                                "STATIC RaBitQ batch layout/options mismatch");
                        }
                    }

                    m_listCount = header.listCount;
                    m_totalDocumentCount = header.totalDocumentCount;
                    m_iDataDimension = header.dataDimension;
                    m_listPageOffset = header.listPageOffset;
                    const int expectedVectorInfoSize = static_cast<int>(
                        UseRaBitQBatchPosting()
                            ? ExpectedRaBitQBatchRecordSize()
                            : ExpectedRaBitQVectorInfoSize());
                    if (header.dataDimension != m_opt->m_dim ||
                        header.dataDimension != m_postingQuantizer->Dimension())
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "STATIC PostingQuantizer=RaBitQ dimension mismatch while loading %s: file=%d index=%d model=%d.\n",
                            p_file.c_str(),
                            header.dataDimension,
                            m_opt->m_dim,
                            m_postingQuantizer->Dimension());
                        throw std::runtime_error("STATIC RaBitQ posting dimension mismatch");
                    }
                    if (header.vectorInfoSize != expectedVectorInfoSize)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "STATIC PostingQuantizer=RaBitQ record size mismatch while loading %s: file=%d expected=%d.\n",
                            p_file.c_str(),
                            header.vectorInfoSize,
                            expectedVectorInfoSize);
                        throw std::runtime_error("STATIC RaBitQ posting record size mismatch");
                    }
                    if (m_vectorInfoSize == 0) m_vectorInfoSize = header.vectorInfoSize;
                    else if (m_vectorInfoSize != header.vectorInfoSize)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "STATIC PostingQuantizer=RaBitQ record size changed across files.\n");
                        throw std::runtime_error("STATIC RaBitQ posting record size mismatch");
                    }
                    if (header.postingQuantizerFingerprint != m_postingQuantizerFingerprint)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "STATIC PostingQuantizer=RaBitQ fingerprint mismatch while loading %s: file=%llu model=%llu.\n",
                            p_file.c_str(),
                            static_cast<unsigned long long>(header.postingQuantizerFingerprint),
                            static_cast<unsigned long long>(m_postingQuantizerFingerprint));
                        throw std::runtime_error("STATIC RaBitQ posting quantizer fingerprint mismatch");
                    }
                }
                else
                {
                    if (ptr->ReadBinary(sizeof(m_listCount), reinterpret_cast<char*>(&m_listCount)) != sizeof(m_listCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(m_totalDocumentCount), reinterpret_cast<char*>(&m_totalDocumentCount)) != sizeof(m_totalDocumentCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(m_iDataDimension), reinterpret_cast<char*>(&m_iDataDimension)) != sizeof(m_iDataDimension)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(m_listPageOffset), reinterpret_cast<char*>(&m_listPageOffset)) != sizeof(m_listPageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }

                    if (m_vectorInfoSize == 0) m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(SizeType);
                    else if (m_vectorInfoSize != m_iDataDimension * sizeof(ValueType) + sizeof(SizeType)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file! DataDimension and ValueType are not match!\n");
                        throw std::runtime_error("DataDimension and ValueType don't match in LoadingHeadInfo");
                    }
                }

                const std::uint64_t headerBytes = UseRaBitQBatchPosting()
                    ? sizeof(StaticBatchPostingHeader)
                    : UseRaBitQPosting()
                        ? sizeof(StaticPostingHeader)
                        : sizeof(SizeType) * 3 + sizeof(int);
                const std::uint64_t metadataRecordBytes =
                    (m_enableDataCompression ? sizeof(size_t) : 0) +
                    sizeof(int) + sizeof(std::uint16_t) + sizeof(int) +
                    sizeof(std::uint16_t) + sizeof(SizeType);
                if (m_listCount < 0 || m_totalDocumentCount < 0 ||
                    m_listPageOffset < 0 || m_iDataDimension <= 0 ||
                    m_iDataDimension != m_opt->m_dim ||
                    m_listCount > m_totalDocumentCount)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Invalid posting header metadata in %s.\n",
                        p_file.c_str());
                    throw std::runtime_error(
                        "Invalid posting header in LoadingHeadInfo");
                }
                const std::uint64_t listCount =
                    static_cast<std::uint64_t>(m_listCount);
                const std::uint64_t listPageOffset =
                    static_cast<std::uint64_t>(m_listPageOffset);
                if (listPageOffset >
                    (std::numeric_limits<std::uint64_t>::max() >> PageSizeEx))
                {
                    throw std::runtime_error(
                        "Posting data offset overflow in LoadingHeadInfo");
                }
                const std::uint64_t postingDataOffset =
                    listPageOffset << PageSizeEx;
                if (postingDataOffset > fileSize ||
                    listCount >
                        (std::numeric_limits<std::uint64_t>::max() -
                         headerBytes) /
                            metadataRecordBytes ||
                    headerBytes + listCount * metadataRecordBytes >
                        postingDataOffset)
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Posting metadata exceeds file bounds in %s.\n",
                        p_file.c_str());
                    throw std::runtime_error(
                        "Posting metadata bounds failure in LoadingHeadInfo");
                }

                const size_t totalListCount = p_listInfos.size();
                if (listCount >
                    p_listInfos.max_size() - totalListCount)
                {
                    throw std::runtime_error(
                        "Posting list count overflow in LoadingHeadInfo");
                }
                p_listInfos.resize(
                    totalListCount + static_cast<size_t>(listCount));
                std::uint64_t metadataBytesRead = headerBytes;

                size_t totalListElementCount = 0;

                std::map<int, int> pageCountDist;

                size_t biglistCount = 0;
                size_t biglistElementCount = 0;
                for (int i = 0; i < m_listCount; ++i)
                {
                    ListInfo* listInfo = &(p_listInfos[totalListCount + i]);
                    listInfo->fileID = static_cast<int>(m_indexFiles.size()) - 1;

                    size_t storedListTotalBytes = 0;
                    if (m_enableDataCompression)
                    {
                        if (ptr->ReadBinary(sizeof(storedListTotalBytes), reinterpret_cast<char*>(&storedListTotalBytes)) != sizeof(storedListTotalBytes)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                            throw std::runtime_error("Failed read file in LoadingHeadInfo");
                        }
                    }
                    int pageNum = 0;
                    std::uint16_t pageOffset = 0;
                    int listEleCount = 0;
                    std::uint16_t listPageCount = 0;
                    SizeType globalVectorID = -1;
                    if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&pageNum)) != sizeof(pageNum)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(pageOffset), reinterpret_cast<char*>(&pageOffset)) != sizeof(pageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listEleCount), reinterpret_cast<char*>(&listEleCount)) != sizeof(listEleCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listPageCount), reinterpret_cast<char*>(&listPageCount)) != sizeof(listPageCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(globalVectorID), reinterpret_cast<char*>(&(globalVectorID))) != sizeof(globalVectorID)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    metadataBytesRead += metadataRecordBytes;

                    if (pageNum < 0 || pageOffset >= PageSize ||
                        listEleCount < 0 || globalVectorID < 0 ||
                        (listEleCount == 0 &&
                         (pageOffset != 0 || listPageCount != 0)) ||
                        (listEleCount > 0 && listPageCount == 0))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Invalid posting-list metadata in %s at list %d.\n",
                            p_file.c_str(),
                            i);
                        throw std::runtime_error(
                            "Invalid list metadata in LoadingHeadInfo");
                    }
                    const std::uint64_t pageNumber =
                        static_cast<std::uint64_t>(pageNum);
                    if (pageNumber >
                        (std::numeric_limits<std::uint64_t>::max() >>
                         PageSizeEx) -
                            listPageOffset)
                    {
                        throw std::runtime_error(
                            "Posting list offset overflow in LoadingHeadInfo");
                    }
                    const std::uint64_t listOffset =
                        (listPageOffset + pageNumber) << PageSizeEx;
                    const std::uint64_t listExtent =
                        static_cast<std::uint64_t>(listPageCount) <<
                        PageSizeEx;
                    if (listOffset > fileSize ||
                        listExtent > fileSize - listOffset)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Posting-list extent exceeds %s at list %d.\n",
                            p_file.c_str(),
                            i);
                        throw std::runtime_error(
                            "Posting list extent failure in LoadingHeadInfo");
                    }
                    const size_t requiredPostingBytes =
                        m_enableDataCompression
                        ? storedListTotalBytes
                        : UseRaBitQBatchPosting()
                            ? RaBitQBatchPostingBytes(
                                  static_cast<size_t>(listEleCount))
                            : static_cast<size_t>(listEleCount) *
                                  static_cast<size_t>(m_vectorInfoSize);
                    if (requiredPostingBytes >
                        listExtent - pageOffset)
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Posting-list payload exceeds its extent in %s at list %d.\n",
                            p_file.c_str(),
                            i);
                        throw std::runtime_error(
                            "Posting list payload failure in LoadingHeadInfo");
                    }
                    if (!m_globalVectorIDToHeadMap.emplace(
                            globalVectorID, totalListCount + i).second)
                    {
                        throw std::runtime_error(
                            "Duplicate posting head ID in LoadingHeadInfo");
                    }
                    listInfo->listTotalBytes = storedListTotalBytes;
                    listInfo->pageOffset = pageOffset;
                    listInfo->listEleCount = listEleCount;
                    listInfo->listPageCount = listPageCount;
                    listInfo->listOffset = listOffset;
                    if (!m_enableDataCompression)
                    {
                        if (UseRaBitQBatchPosting())
                        {
                            listInfo->listTotalBytes =
                                RaBitQBatchPostingBytes(listInfo->listEleCount);
                            const size_t readableBytes =
                                (static_cast<size_t>(min(
                                     static_cast<int>(listInfo->listPageCount),
                                     p_postingPageLimit))
                                 << PageSizeEx);
                            const size_t readableRecords =
                                readableBytes > listInfo->pageOffset
                                ? (readableBytes - listInfo->pageOffset) /
                                    static_cast<size_t>(m_vectorInfoSize)
                                : 0;
                            listInfo->listEleCount = min(
                                listInfo->listEleCount,
                                static_cast<int>(
                                    readableRecords *
                                    m_splitBatchLayout.batchSize));
                            const size_t postingBytes =
                                RaBitQBatchPostingBytes(listInfo->listEleCount);
                            listInfo->listPageCount =
                                static_cast<std::uint16_t>(
                                    (postingBytes + listInfo->pageOffset +
                                     PageSize - 1) /
                                    PageSize);
                        }
                        else
                        {
                            listInfo->listTotalBytes =
                                listInfo->listEleCount * m_vectorInfoSize;
                            listInfo->listEleCount = min(
                                listInfo->listEleCount,
                                (min(
                                     static_cast<int>(listInfo->listPageCount),
                                     p_postingPageLimit)
                                 << PageSizeEx) /
                                    m_vectorInfoSize);
                            listInfo->listPageCount =
                                static_cast<std::uint16_t>(ceil(
                                    (m_vectorInfoSize * listInfo->listEleCount +
                                     listInfo->pageOffset) *
                                    1.0 / (1 << PageSizeEx)));
                        }
                    }
                    totalListElementCount += listInfo->listEleCount;
                    int pageCount = listInfo->listPageCount;

                    if (pageCount > 1)
                    {
                        ++biglistCount;
                        biglistElementCount += listInfo->listEleCount;
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

                if (m_enableDataCompression && m_enableDictTraining)
                {
                    if (metadataBytesRead + sizeof(size_t) >
                        postingDataOffset)
                    {
                        throw std::runtime_error(
                            "Compression dictionary metadata exceeds posting header");
                    }
                    size_t dictBufferSize = 0;
                    if (ptr->ReadBinary(sizeof(size_t), reinterpret_cast<char*>(&dictBufferSize)) != sizeof(dictBufferSize)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    metadataBytesRead += sizeof(size_t);
                    if (dictBufferSize >
                        postingDataOffset - metadataBytesRead)
                    {
                        throw std::runtime_error(
                            "Compression dictionary exceeds posting metadata bounds");
                    }
                    std::vector<char> dictBuffer(dictBufferSize);
                    if (ptr->ReadBinary(dictBufferSize, dictBuffer.data()) != dictBufferSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    try {
                        m_pCompressor->SetDictBuffer(
                            std::string(dictBuffer.data(), dictBufferSize));
                    }
                    catch (std::runtime_error& err) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file: %s \n", err.what());
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "Finish reading header info, list count %lld, total doc count %lld, dimension %d, list page offset %lld.\n",
                    (std::int64_t)m_listCount,
                    (std::int64_t)m_totalDocumentCount,
                    m_iDataDimension,
                    (std::int64_t)m_listPageOffset);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "Big page (>4K): list count %zu, total element count %zu.\n",
                    biglistCount,
                    biglistElementCount);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n", totalListElementCount);

                for (auto& ele : pageCountDist)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Page Count Dist: %d %d\n", ele.first, ele.second);
                }

                return m_listCount;
            }

            inline void ParsePostingListRearrange(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                offsetVectorID = (m_vectorInfoSize - sizeof(SizeType)) * eleCount + sizeof(SizeType) * i;
                offsetVector = (m_vectorInfoSize - sizeof(SizeType)) * i;
            }

            inline void ParsePostingList(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                offsetVectorID = m_vectorInfoSize * i;
                offsetVector = offsetVectorID + sizeof(SizeType);
            }

            inline void ParseDeltaEncoding(ListInfo* p_info, ValueType* vector)
            {
                ValueType* headVector = (ValueType*)m_headIndex->GetSample((SizeType)(p_info - m_listInfos.data()));
                COMMON::SIMDUtils::ComputeSum(vector, headVector, m_iDataDimension);
            }

            inline void ParseEncoding(ListInfo* p_info, ValueType* vector) { }

            void SelectPostingOffset(
                const std::vector<size_t>& p_postingListBytes,
                std::unique_ptr<int[]>& p_postPageNum,
                std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                std::vector<int>& p_postingOrderInIndex)
            {
                p_postPageNum.reset(new int[p_postingListBytes.size()]);
                p_postPageOffset.reset(new std::uint16_t[p_postingListBytes.size()]);

                struct PageModWithID
                {
                    int id;

                    std::uint16_t rest;
                };

                struct PageModeWithIDCmp
                {
                    bool operator()(const PageModWithID& a, const PageModWithID& b) const
                    {
                        return a.rest == b.rest ? a.id < b.id : a.rest > b.rest;
                    }
                };

                std::set<PageModWithID, PageModeWithIDCmp> listRestSize;

                p_postingOrderInIndex.clear();
                p_postingOrderInIndex.reserve(p_postingListBytes.size());

                PageModWithID listInfo;
                for (size_t i = 0; i < p_postingListBytes.size(); ++i)
                {
                    if (p_postingListBytes[i] == 0)
                    {
                        continue;
                    }

                    listInfo.id = static_cast<int>(i);
                    listInfo.rest = static_cast<std::uint16_t>(p_postingListBytes[i] % PageSize);

                    listRestSize.insert(listInfo);
                }

                listInfo.id = -1;

                int currPageNum = 0;
                std::uint16_t currOffset = 0;

                while (!listRestSize.empty())
                {
                    listInfo.rest = PageSize - currOffset;
                    auto iter = listRestSize.lower_bound(listInfo); // avoid page-crossing
                    if (iter == listRestSize.end() || (listInfo.rest != PageSize && iter->rest == 0))
                    {
                        ++currPageNum;
                        currOffset = 0;
                    }
                    else
                    {
                        p_postPageNum[iter->id] = currPageNum;
                        p_postPageOffset[iter->id] = currOffset;

                        p_postingOrderInIndex.push_back(iter->id);

                        currOffset += iter->rest;
                        if (currOffset > PageSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Crossing extra pages\n");
                            throw std::runtime_error("Read too many pages");
                        }

                        if (currOffset == PageSize)
                        {
                            ++currPageNum;
                            currOffset = 0;
                        }

                        currPageNum += static_cast<int>(p_postingListBytes[iter->id] / PageSize);

                        listRestSize.erase(iter);
                    }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "TotalPageNumbers: %d, IndexSize: %llu\n", currPageNum, static_cast<uint64_t>(currPageNum) * PageSize + currOffset);
            }

            void OutputSSDIndexFile(const std::string& p_outputFile,
                bool p_enableDeltaEncoding,
                bool p_enablePostingListRearrange,
                bool p_enableDataCompression,
                bool p_enableDictTraining,
                size_t p_spacePerVector,
                const std::vector<int>& p_postingListSizes,
                const std::vector<size_t>& p_postingListBytes,
                std::shared_ptr<VectorIndex> p_headIndex,
                Selection& p_postingSelections,
                const std::unique_ptr<int[]>& p_postPageNum,
                const std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                const std::vector<int>& p_postingOrderInIndex,
                std::shared_ptr<VectorSet> p_fullVectors, COMMON::Dataset<SizeType>& p_headToLocal, COMMON::Dataset<SizeType>& p_localToGlobal,
                size_t p_postingListOffset,
                int p_fileID)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                auto t1 = std::chrono::high_resolution_clock::now();

                auto ptr = SPTAG::f_createIO();
                int retry = 3;
                // open file 
                while (retry > 0 && (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open file %s, retrying...\n", p_outputFile.c_str());
                    retry--;
                }

                if (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open file %s\n", p_outputFile.c_str());
                    throw std::runtime_error("Failed to open file for SSD index save");
                }

                std::shared_ptr<Helper::DiskIO> extendedOutput;
                std::shared_ptr<Helper::DiskIO> rawOutput;
                BatchSidecarHeader extendedHeader;
                BatchSidecarHeader rawHeader;
                std::vector<BatchSidecarListInfo> extendedInfos;
                std::vector<BatchSidecarListInfo> rawInfos;
                if (UseRaBitQBatchPosting())
                {
                    auto initializeSidecar =
                        [&](const std::string& p_path,
                            BatchSidecarKind p_kind,
                            std::uint64_t p_dataBytesPerRecord,
                            std::shared_ptr<Helper::DiskIO>& p_output,
                            BatchSidecarHeader& p_header,
                            std::vector<BatchSidecarListInfo>& p_infos)
                    {
                        p_output = SPTAG::f_createIO();
                        if (p_output == nullptr ||
                            !p_output->Initialize(
                                p_path.c_str(), std::ios::binary | std::ios::out))
                        {
                            throw std::runtime_error(
                                "Failed to create STATIC RaBitQ batch sidecar");
                        }
                        p_header.magic = kBatchSidecarMagic;
                        p_header.version = kBatchSidecarVersion;
                        p_header.kind = static_cast<std::uint32_t>(p_kind);
                        p_header.layer = static_cast<std::uint32_t>(m_layer);
                        p_header.fileID = static_cast<std::uint32_t>(p_fileID);
                        p_header.listCount =
                            static_cast<std::uint32_t>(p_postingListSizes.size());
                        p_header.dimension =
                            static_cast<std::uint32_t>(m_opt->m_dim);
                        p_header.batchSize =
                            static_cast<std::uint32_t>(m_splitBatchLayout.batchSize);
                        p_header.postingQuantizerFingerprint =
                            m_postingQuantizerFingerprint;
                        p_header.dataBytesPerRecord = p_dataBytesPerRecord;
                        p_infos.resize(p_postingListSizes.size());
                        const size_t metadataBytes = sizeof(BatchSidecarHeader) +
                            sizeof(BatchSidecarListInfo) * p_infos.size();
                        std::vector<char> zeros(metadataBytes, 0);
                        if (p_output->WriteBinary(zeros.size(), zeros.data()) !=
                            zeros.size())
                        {
                            throw std::runtime_error(
                                "Failed to reserve STATIC RaBitQ batch sidecar metadata");
                        }
                    };

                    initializeSidecar(
                        p_outputFile + kBatchExtendedSuffix,
                        BatchSidecarKind::Extended,
                        m_splitBatchLayout.extendedBytes,
                        extendedOutput,
                        extendedHeader,
                        extendedInfos);
                    if (m_opt->m_postingRaBitQRerank > 0)
                    {
                        initializeSidecar(
                            p_outputFile + kBatchRawSuffix,
                            BatchSidecarKind::Raw,
                            static_cast<std::uint64_t>(m_opt->m_dim) *
                                sizeof(float),
                            rawOutput,
                            rawHeader,
                            rawInfos);
                    }
                }
                // meta size of global info
                std::uint64_t listOffset = UseRaBitQBatchPosting()
                    ? sizeof(StaticBatchPostingHeader)
                    : UseRaBitQPosting()
                        ? sizeof(StaticPostingHeader)
                    : sizeof(SizeType) * 3 + sizeof(int);
                // meta size of the posting lists
                listOffset += (sizeof(int) + sizeof(std::uint16_t) + sizeof(int) + sizeof(std::uint16_t) + sizeof(SizeType)) * p_postingListSizes.size();
                // write listTotalBytes only when enabled data compression
                if (p_enableDataCompression)
                {
                    listOffset += sizeof(size_t) * p_postingListSizes.size();
                }

                // compression dict
                if (p_enableDataCompression && p_enableDictTraining)
                {
                    listOffset += sizeof(size_t);
                    listOffset += m_pCompressor->GetDictBuffer().size();
                }

                std::unique_ptr<char[]> paddingVals(new char[PageSize]);
                memset(paddingVals.get(), 0, sizeof(char) * PageSize);
                // paddingSize: bytes left in the last page
                std::uint64_t paddingSize = PageSize - (listOffset % PageSize);
                if (paddingSize == PageSize)
                {
                    paddingSize = 0;
                }
                else
                {
                    listOffset += paddingSize;
                }

                if (UseRaBitQPosting())
                {
                    StaticPostingHeader header;
                    header.magic = kStaticPostingHeaderMagic;
                    header.version = UseRaBitQBatchPosting()
                        ? kStaticBatchPostingHeaderVersion
                        : kStaticPostingHeaderVersion;
                    header.format = static_cast<std::uint32_t>(m_postingFormat);
                    header.reserved = 0;
                    header.listCount = static_cast<SizeType>(p_postingListSizes.size());
                    header.totalDocumentCount = static_cast<SizeType>(p_fullVectors->Count());
                    header.dataDimension = static_cast<int>(p_fullVectors->Dimension());
                    header.listPageOffset = static_cast<SizeType>(listOffset / PageSize);
                    header.vectorInfoSize = static_cast<int>(p_spacePerVector);
                    header.postingQuantizerFingerprint = m_postingQuantizerFingerprint;
                    if (UseRaBitQBatchPosting())
                    {
                        StaticBatchPostingHeader batchHeader;
                        batchHeader.base = header;
                        batchHeader.batchSize =
                            static_cast<std::uint32_t>(m_splitBatchLayout.batchSize);
                        batchHeader.batchRecordBytes =
                            static_cast<std::uint32_t>(ExpectedRaBitQBatchRecordSize());
                        batchHeader.extendedBytesPerVector =
                            static_cast<std::uint32_t>(
                                m_splitBatchLayout.extendedBytesPerVector);
                        batchHeader.flags = m_opt->m_postingRaBitQRerank > 0
                            ? kBatchPostingRawFlag
                            : 0U;
                        if (ptr->WriteBinary(
                                sizeof(batchHeader),
                                reinterpret_cast<char*>(&batchHeader)) !=
                            sizeof(batchHeader))
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                    }
                    else if (ptr->WriteBinary(
                                 sizeof(header), reinterpret_cast<char*>(&header)) !=
                             sizeof(header))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }
                else
                {
                    // Number of posting lists
                    SizeType iVal = static_cast<SizeType>(p_postingListSizes.size());
                    if (ptr->WriteBinary(sizeof(iVal), reinterpret_cast<char*>(&iVal)) != sizeof(iVal)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }

                    // Number of vectors
                    iVal = static_cast<SizeType>(p_fullVectors->Count());
                    if (ptr->WriteBinary(sizeof(iVal), reinterpret_cast<char*>(&iVal)) != sizeof(iVal)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }

                    // Vector dimension
                    int i32Val = static_cast<int>(p_fullVectors->Dimension());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }

                    // Page offset of list content section
                    iVal = static_cast<SizeType>(listOffset / PageSize);
                    if (ptr->WriteBinary(sizeof(iVal), reinterpret_cast<char*>(&iVal)) != sizeof(iVal)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                // Meta of each posting list
                for (int i = 0; i < p_postingListSizes.size(); ++i)
                {
                    size_t postingListByte = 0;
                    int pageNum = 0; // starting page number
                    std::uint16_t pageOffset = 0;
                    int listEleCount = 0;
                    std::uint16_t listPageCount = 0;

                    if (p_postingListSizes[i] > 0)
                    {
                        pageNum = p_postPageNum[i];
                        pageOffset = static_cast<std::uint16_t>(p_postPageOffset[i]);
                        listEleCount = static_cast<int>(p_postingListSizes[i]);
                        postingListByte = p_postingListBytes[i];
                        listPageCount = static_cast<std::uint16_t>(postingListByte / PageSize);
                        if (0 != (postingListByte % PageSize))
                        {
                            ++listPageCount;
                        }
                    }
                    // Total bytes of the posting list, write only when enabled data compression
                    if (p_enableDataCompression && ptr->WriteBinary(sizeof(postingListByte), reinterpret_cast<char*>(&postingListByte)) != sizeof(postingListByte)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page number of the posting list
                    if (ptr->WriteBinary(sizeof(pageNum), reinterpret_cast<char*>(&pageNum)) != sizeof(pageNum)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page offset
                    if (ptr->WriteBinary(sizeof(pageOffset), reinterpret_cast<char*>(&pageOffset)) != sizeof(pageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Number of vectors in the posting list
                    if (ptr->WriteBinary(sizeof(listEleCount), reinterpret_cast<char*>(&listEleCount)) != sizeof(listEleCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page count of the posting list
                    if (ptr->WriteBinary(sizeof(listPageCount), reinterpret_cast<char*>(&listPageCount)) != sizeof(listPageCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }

                    SizeType globalVectorID = *(p_headToLocal[p_postingListOffset + i]);
                    if (p_localToGlobal.R() > 0) globalVectorID = *(p_localToGlobal[globalVectorID]);
                    if (ptr->WriteBinary(sizeof(globalVectorID), reinterpret_cast<char*>(&(globalVectorID))) != sizeof(globalVectorID)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!\n");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }
                // compression dict
                if (p_enableDataCompression && p_enableDictTraining)
                {
                    std::string dictBuffer = m_pCompressor->GetDictBuffer();
                    // dict size
                    size_t dictBufferSize = dictBuffer.size();
                    if (ptr->WriteBinary(sizeof(size_t), reinterpret_cast<char *>(&dictBufferSize)) != sizeof(dictBufferSize))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // dict
                    if (ptr->WriteBinary(dictBuffer.size(), const_cast<char *>(dictBuffer.data())) != dictBuffer.size())
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                // Write padding vals
                if (paddingSize > 0)
                {
                    if (ptr->WriteBinary(paddingSize, reinterpret_cast<char*>(paddingVals.get())) != paddingSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                if (static_cast<uint64_t>(ptr->TellP()) != listOffset)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "List offset not match!\n");
                    throw std::runtime_error("List offset mismatch");
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SubIndex Size: %llu bytes, %llu MBytes\n", listOffset, listOffset >> 20);

                listOffset = 0;

                std::uint64_t paddedSize = 0;
                // iterate over all the posting lists
                for (auto id : p_postingOrderInIndex)
                {
                    std::uint64_t targetOffset = static_cast<uint64_t>(p_postPageNum[id]) * PageSize + p_postPageOffset[id];
                    if (targetOffset < listOffset)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "List offset not match, targetOffset < listOffset!\n");
                        throw std::runtime_error("List offset mismatch");
                    }
                    // write padding vals before the posting list
                    if (targetOffset > listOffset)
                    {
                        if (targetOffset - listOffset > PageSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Padding size greater than page size!\n");
                            throw std::runtime_error("Padding size mismatch with page size");
                        }

                        if (ptr->WriteBinary(targetOffset - listOffset, reinterpret_cast<char*>(paddingVals.get())) != targetOffset - listOffset) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }

                        paddedSize += targetOffset - listOffset;

                        listOffset = targetOffset;
                    }

                    if (p_postingListSizes[id] == 0)
                    {
                        continue;
                    }
                    int postingListId = id + (int)p_postingListOffset;
                    // get posting list full content and write it at once
                    ValueType *headVector = nullptr;
                    if (p_enableDeltaEncoding)
                    {
                        headVector = (ValueType *)p_headIndex->GetSample(postingListId);
                    }
                    BatchPostingData batchPosting;
                    std::string postingListFullData;
                    if (UseRaBitQBatchPosting())
                    {
                        batchPosting = BuildRaBitQBatchPosting(
                            postingListId,
                            static_cast<size_t>(p_postingListSizes[id]),
                            p_postingSelections,
                            p_fullVectors,
                            p_localToGlobal);
                        postingListFullData = batchPosting.binary;

                        auto& extendedInfo = extendedInfos[id];
                        extendedInfo.vectorCount =
                            static_cast<std::uint64_t>(p_postingListSizes[id]);
                        extendedInfo.batchCount = static_cast<std::uint64_t>(
                            RaBitQBatchCount(
                                static_cast<size_t>(p_postingListSizes[id])));
                        extendedInfo.centroidOffset =
                            static_cast<std::uint64_t>(extendedOutput->TellP());
                        const size_t centroidBytes =
                            batchPosting.centroid.size() * sizeof(float);
                        if (extendedOutput->WriteBinary(
                                centroidBytes,
                                reinterpret_cast<const char*>(
                                    batchPosting.centroid.data())) != centroidBytes)
                        {
                            throw std::runtime_error(
                                "Failed to write STATIC RaBitQ batch centroid");
                        }
                        extendedInfo.dataOffset =
                            static_cast<std::uint64_t>(extendedOutput->TellP());
                        extendedInfo.dataBytes = batchPosting.extended.size();
                        if (!batchPosting.extended.empty() &&
                            extendedOutput->WriteBinary(
                                batchPosting.extended.size(),
                                batchPosting.extended.data()) !=
                                batchPosting.extended.size())
                        {
                            throw std::runtime_error(
                                "Failed to write STATIC RaBitQ extended sidecar");
                        }

                        if (rawOutput != nullptr)
                        {
                            auto& rawInfo = rawInfos[id];
                            rawInfo.vectorCount = extendedInfo.vectorCount;
                            rawInfo.batchCount = extendedInfo.batchCount;
                            rawInfo.dataOffset =
                                static_cast<std::uint64_t>(rawOutput->TellP());
                            rawInfo.dataBytes = batchPosting.raw.size();
                            if (rawOutput->WriteBinary(
                                    batchPosting.raw.size(),
                                    batchPosting.raw.data()) !=
                                batchPosting.raw.size())
                            {
                                throw std::runtime_error(
                                    "Failed to write STATIC RaBitQ raw sidecar");
                            }
                        }
                    }
                    else
                    {
                        postingListFullData = GetPostingListFullData(
                            postingListId,
                            p_postingListSizes[id],
                            p_postingSelections,
                            p_fullVectors,
                            p_localToGlobal,
                            p_enableDeltaEncoding,
                            p_enablePostingListRearrange,
                            headVector);
                    }
                    size_t postingListFullSize = UseRaBitQBatchPosting()
                        ? RaBitQBatchPostingBytes(
                            static_cast<size_t>(p_postingListSizes[id]))
                        : static_cast<size_t>(p_postingListSizes[id]) *
                            p_spacePerVector;
                    if (postingListFullSize != postingListFullData.size())
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "posting list full data size NOT MATCH! postingListFullData.size(): %zu postingListFullSize: %zu \n", postingListFullData.size(), postingListFullSize);
                        throw std::runtime_error("Posting list full size mismatch");
                    }
                    if (p_enableDataCompression)
                    {
                        std::string compressedData = m_pCompressor->Compress(postingListFullData, p_enableDictTraining);
                        size_t compressedSize = compressedData.size();
                        if (compressedSize != p_postingListBytes[id])
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Compressed size NOT MATCH! compressed size:%zu, pre-calculated compressed size:%zu\n", compressedSize, p_postingListBytes[id]);
                            throw std::runtime_error("Compression size mismatch");
                        }
                        if (ptr->WriteBinary(compressedSize, compressedData.data()) != compressedSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                        listOffset += compressedSize;
                    }
                    else
                    {
                        if (ptr->WriteBinary(postingListFullSize, postingListFullData.data()) != postingListFullSize)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                        listOffset += postingListFullSize;
                    }
                }

                paddingSize = PageSize - (listOffset % PageSize);
                if (paddingSize == PageSize)
                {
                    paddingSize = 0;
                }
                else
                {
                    listOffset += paddingSize;
                    paddedSize += paddingSize;
                }

                if (paddingSize > 0)
                {
                    if (ptr->WriteBinary(paddingSize, reinterpret_cast<char *>(paddingVals.get())) != paddingSize)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                if (UseRaBitQBatchPosting())
                {
                    auto finalizeSidecar =
                        [](const std::shared_ptr<Helper::DiskIO>& p_output,
                           BatchSidecarHeader& p_header,
                           std::vector<BatchSidecarListInfo>& p_infos)
                    {
                        if (p_output->WriteBinary(
                                sizeof(p_header),
                                reinterpret_cast<const char*>(&p_header),
                                0) != sizeof(p_header) ||
                            (!p_infos.empty() &&
                             p_output->WriteBinary(
                                 sizeof(BatchSidecarListInfo) * p_infos.size(),
                                 reinterpret_cast<const char*>(p_infos.data()),
                                 sizeof(BatchSidecarHeader)) !=
                                 sizeof(BatchSidecarListInfo) * p_infos.size()))
                        {
                            throw std::runtime_error(
                                "Failed to finalize STATIC RaBitQ batch sidecar");
                        }
                    };
                    finalizeSidecar(
                        extendedOutput, extendedHeader, extendedInfos);
                    if (rawOutput != nullptr)
                    {
                        finalizeSidecar(rawOutput, rawHeader, rawInfos);
                    }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Padded Size: %llu, final total size: %llu.\n", paddedSize, listOffset);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Output done...\n");
                auto t2 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to write results:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000);
            }

            ErrorCode GetWritePosting(ExtraWorkSpace* p_exWorkSpace, SizeType pid, std::string& posting, bool write = false) override {
                if (write) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Unsupport write\n");
                    return ErrorCode::Undefined;
                }
                auto it = m_globalVectorIDToHeadMap.find(pid);
                if (it == m_globalVectorIDToHeadMap.end()) return ErrorCode::Key_NotFound;
                pid = it->second;
                ListInfo* listInfo = &(m_listInfos[pid]);
                size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                size_t realBytes = UseRaBitQBatchPosting()
                    ? RaBitQBatchPostingBytes(listInfo->listEleCount)
                    : static_cast<size_t>(listInfo->listEleCount) *
                        m_vectorInfoSize;
                posting.resize(totalBytes);
                int fileid = m_oneContext? 0: pid / m_listPerFile;
                Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
                auto numRead = indexFile->ReadBinary(totalBytes, (char*)posting.data(), listInfo->listOffset);
                if (numRead != totalBytes) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                    return ErrorCode::DiskIOFail;
                }
                char* ptr = (char*)(posting.c_str());
                memcpy(ptr, posting.c_str() + listInfo->pageOffset, realBytes);
                posting.resize(realBytes);
                return ErrorCode::Success;
            }

            ErrorCode Checkpoint(std::string p_prefix) override
            {
                if (p_prefix.empty()) return ErrorCode::FailedOpenFile;
                std::error_code ec;
                std::filesystem::create_directories(p_prefix, ec);
                if (ec) return ErrorCode::FailedOpenFile;

                auto copyFile = [&](const std::string& p_source) -> bool
                {
                    const std::filesystem::path source(p_source);
                    const std::filesystem::path destination =
                        std::filesystem::path(p_prefix) / source.filename();
                    std::error_code compareError;
                    if (std::filesystem::equivalent(
                            source, destination, compareError) &&
                        !compareError)
                    {
                        return true;
                    }
                    std::error_code copyError;
                    return std::filesystem::copy_file(
                        source,
                        destination,
                        std::filesystem::copy_options::overwrite_existing,
                        copyError);
                };

                for (const auto& indexFile : m_indexFilePaths)
                {
                    if (!copyFile(indexFile)) return ErrorCode::DiskIOFail;
                    if (UseRaBitQBatchPosting())
                    {
                        if (!copyFile(indexFile + kBatchExtendedSuffix))
                        {
                            return ErrorCode::DiskIOFail;
                        }
                        if (m_opt->m_postingRaBitQRerank > 0 &&
                            !copyFile(indexFile + kBatchRawSuffix))
                        {
                            return ErrorCode::DiskIOFail;
                        }
                    }
                }
                if (UseRaBitQPosting() &&
                    !copyFile(ResolvePostingQuantizerPath(*m_opt)))
                {
                    return ErrorCode::DiskIOFail;
                }
                return ErrorCode::Success;
            }

        private:
            bool m_available = false;

            std::string m_extraFullGraphFile;

            std::vector<ListInfo> m_listInfos;
            bool m_oneContext;
            Options* m_opt;
            int m_layer;

            SPANN::Index<ValueType>* m_headIndex;
            std::vector<std::shared_ptr<Helper::DiskIO>> m_indexFiles;
            std::unique_ptr<Compressor> m_pCompressor;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableDataCompression;
            bool m_enableDictTraining;
            StaticPostingFormat m_postingFormat = StaticPostingFormat::LegacyRaw;
            std::shared_ptr<COMMON::RaBitQQuantizer> m_postingQuantizer;
            std::vector<char> m_postingQuantizerBytes;
            std::uint64_t m_postingQuantizerFingerprint = 0;
            COMMON::RaBitQQuantizer::SplitCodeLayout m_splitCodeLayout;
            COMMON::RaBitQQuantizer::SplitBatchLayout m_splitBatchLayout;
            std::vector<std::shared_ptr<Helper::DiskIO>> m_extendedSidecars;
            std::vector<std::shared_ptr<Helper::DiskIO>> m_rawSidecars;
            std::vector<std::unique_ptr<std::mutex>> m_sidecarReadLocks;
            std::vector<std::string> m_indexFilePaths;
            
            void (ExtraStaticSearcher<ValueType>::*m_parsePosting)(uint64_t&, uint64_t&, int, int);
            void (ExtraStaticSearcher<ValueType>::*m_parseEncoding)(ListInfo*, ValueType*);

            int m_vectorInfoSize = 0;
            int m_iDataDimension = 0;
            int m_totalListCount = 0;

            int m_listPerFile = 0;

            std::unordered_map<SizeType, SizeType> m_globalVectorIDToHeadMap;

        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_EXTRASTATICSEARCHER_H_
