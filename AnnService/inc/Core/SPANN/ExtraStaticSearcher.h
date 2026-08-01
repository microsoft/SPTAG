// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASTATICSEARCHER_H_
#define _SPTAG_SPANN_EXTRASTATICSEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "inc/Core/Common/TruthSet.h"
#include "Compressor.h"
#include "PipePQ.h"

#include <atomic>
#include <map>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef _MSC_VER
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
        { \
        bool postingMatched = false; \
        bool postingContributedUnique = false; \
        const int staticScanBegin = this->StaticScanBegin(p_exWorkSpace, staticPostingSlot, listInfo); \
        const int staticScanEnd = this->StaticScanEnd(p_exWorkSpace, staticPostingSlot, listInfo); \
        for (int i = staticScanBegin; i < staticScanEnd; i++) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);\
            const char* record = p_postingListFullData + offsetVectorID;\
            int vectorID;\
            std::memcpy(&vectorID, record, sizeof(vectorID));\
            if (!this->StaticRecordMatchesFilter(p_exWorkSpace, record)) { listElements--; continue; } \
            postingMatched = true; \
            ++p_exWorkSpace->m_postingProbeStats.m_matchedVectors; \
            if (collectPostingContributionStats && uniqueMatchedVIDs.insert(vectorID).second) { \
                postingContributedUnique = true; \
                ++p_exWorkSpace->m_postingProbeStats.m_uniqueMatchedVectors; \
            } \
            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) { listElements--; continue; } \
            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
        } \
        if (postingMatched) ++p_exWorkSpace->m_postingProbeStats.m_matchedPostings; \
        if (postingContributedUnique) ++p_exWorkSpace->m_postingProbeStats.m_uniqueMatchedPostings; \
        } \

#define ProcessPostingOffset() \
        while (p_exWorkSpace->m_offset < this->StaticScanEnd(p_exWorkSpace, p_exWorkSpace->m_pi, listInfo)) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, p_exWorkSpace->m_offset, listInfo->listEleCount);\
            p_exWorkSpace->m_offset++;\
            const char* record = p_postingListFullData + offsetVectorID;\
            int vectorID;\
            std::memcpy(&vectorID, record, sizeof(vectorID));\
            if (!this->StaticRecordMatchesFilter(p_exWorkSpace, record)) continue; \
            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue; \
            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
            foundResult = true;\
            break;\
        } \
        if (p_exWorkSpace->m_offset == this->StaticScanEnd(p_exWorkSpace, p_exWorkSpace->m_pi, listInfo)) { \
            p_exWorkSpace->m_pi++; \
            if (p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) { \
                SizeType nextPostingID = p_exWorkSpace->m_postingIDs[p_exWorkSpace->m_pi]; \
                ListInfo* nextListInfo = &(this->m_listInfos[nextPostingID]); \
                p_exWorkSpace->m_offset = this->StaticScanBegin( \
                    p_exWorkSpace, p_exWorkSpace->m_pi, nextListInfo); \
            } else { \
                p_exWorkSpace->m_offset = 0; \
            } \
        } \

        template <typename ValueType>
        class ExtraStaticSearcher : public IExtraSearcher
        {
        public:
            ExtraStaticSearcher()
            {
                m_enableDeltaEncoding = false;
                m_enablePostingListRearrange = false;
                m_enableDataCompression = false;
                m_enableDictTraining = true;
            }

            virtual ~ExtraStaticSearcher()
            {
                CloseStaticPipePQCodes();
            }

            virtual bool Available() override
            {
                return m_available;
            }

            void SetVectorTags(const uint32_t* p_tags, int p_numVectors,
                               int p_numTagsPerVec) override
            {
                m_staticBuildTags.clear();
                m_staticBuildNumTagsPerVec = 0;
                if (p_tags == nullptr || p_numVectors <= 0 || p_numTagsPerVec <= 0) return;

                m_staticBuildTags.assign(
                    p_tags,
                    p_tags + static_cast<size_t>(p_numVectors) * static_cast<size_t>(p_numTagsPerVec));
                m_staticBuildNumTagsPerVec = p_numTagsPerVec;
            }

            void SetNodeVectorAssignments(
                const std::vector<std::vector<SizeType>>& p_assignments) override
            {
                m_staticNodeVectorAssignments = p_assignments;
            }

            void SetPrimaryNodeVectorAssignments(
                const std::vector<std::vector<SizeType>>& p_assignments) override
            {
                m_staticPrimaryNodeVectorAssignments = p_assignments;
            }

            void SetHeadVectorOwners(
                const std::unordered_map<SizeType, int>& p_owners) override
            {
                m_staticHeadVectorOwners = p_owners;
                m_staticHeadVectorOwnersView = nullptr;
            }

            void SetHeadVectorOwnersView(
                const std::unordered_map<SizeType, int>* p_owners) override
            {
                m_staticHeadVectorOwnersView = p_owners;
            }

            void SetHeadBundleBuildView(
                const std::vector<std::shared_ptr<VectorIndex>>& p_indexes,
                const std::vector<std::vector<SizeType>>* p_localToGlobalHIDs,
                const std::vector<std::vector<SizeType>>* p_nodeHeadVectorIDs) override
            {
                m_staticHeadBundleIndexes = p_indexes;
                m_staticHeadBundleLocalToGlobalHIDs = p_localToGlobalHIDs;
                m_staticHeadBundleNodeHeadVectorIDs = p_nodeHeadVectorIDs;
            }

            using StaticCrossGraphSearch = std::function<bool(
                const ValueType*,
                int,
                int,
                std::vector<std::pair<SizeType, float>>&)>;

            void SetStaticCrossGraphSearch(StaticCrossGraphSearch p_search)
            {
                m_staticCrossGraphSearch = std::move(p_search);
            }

            void InitWorkSpace(ExtraWorkSpace* p_exWorkSpace, bool clear = false) override
            {
                if (clear) {
                    p_exWorkSpace->Clear(m_opt->m_searchInternalResultNum, max(m_opt->m_postingPageLimit, m_opt->m_searchPostingPageLimit + 1) << PageSizeEx, false, m_opt->m_enableDataCompression);
                }
                else {
                    p_exWorkSpace->Initialize(m_opt->m_maxCheck, m_opt->m_hashExp, m_opt->m_searchInternalResultNum, max(m_opt->m_postingPageLimit, m_opt->m_searchPostingPageLimit + 1) << PageSizeEx, false, m_opt->m_enableDataCompression);
                    int wid = 0;
                    if (m_freeWorkSpaceIds == nullptr || !m_freeWorkSpaceIds->try_pop(wid))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FreeWorkSpaceIds is not initalized or the workspace number is not enough! Please increase iothread number.\n");
                        wid = m_workspaceCount.fetch_add(1);
                    }
                    for (auto & req : p_exWorkSpace->m_diskRequests)
                    {
                        req.m_status = wid;
                    }
                    p_exWorkSpace->m_callback = [m_freeWorkSpaceIds = m_freeWorkSpaceIds, wid] () {
                        if (m_freeWorkSpaceIds) m_freeWorkSpaceIds->push(wid);
                    };
                }
            }

            virtual bool LoadIndex(Options& p_opt, COMMON::VersionLabel& p_versionMap, COMMON::Dataset<std::uint64_t>& p_vectorTranslateMap,  std::shared_ptr<VectorIndex> m_index) {
                m_extraFullGraphFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                m_opt = &p_opt;
                if (!ConfigureStaticPipePQ(p_opt, 0, false)) {
                    return false;
                }
                m_staticHasMetadata = false;
                m_staticNumTagsPerVec = 0;
                m_staticACLTagCols = 0;
                m_staticMetadataBytes = sizeof(int);
                m_staticMaxListPageCount = 0;
                std::string curFile = m_extraFullGraphFile;
                const size_t configuredTagBytes =
                    (!m_staticPipePQ && p_opt.m_numTagsPerVec > 0)
                        ? static_cast<size_t>(p_opt.m_numTagsPerVec) * sizeof(uint32_t)
                        : 0;
                if (configuredTagBytes > 0 &&
                    (p_opt.m_enableDeltaEncoding || p_opt.m_enablePostingListRearrange ||
                     p_opt.m_enableDataCompression)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static metadata snapshots require raw postings without delta encoding, "
                        "rearrangement, or compression.\n");
                    return false;
                }
                const size_t recordBytes = m_staticPipePQ
                    ? sizeof(int) + static_cast<size_t>(m_staticPipePQCodeBytes)
                    : p_opt.m_dim * sizeof(ValueType) + sizeof(int) + configuredTagBytes;
                const int tailPageBudget = p_opt.m_tailReplicaCount > 0
                    ? p_opt.m_unfilterTailBufferLength
                    : 0;
                p_opt.m_searchPostingPageLimit = max(
                    p_opt.m_searchPostingPageLimit,
                    static_cast<int>((p_opt.m_postingVectorLimit * recordBytes + PageSize - 1) / PageSize) +
                        (std::max)(0, tailPageBudget));
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
                    try {
                        m_totalListCount += LoadingHeadInfo(curFile, p_opt.m_searchPostingPageLimit, m_listInfos);
                    } 
                    catch (std::exception& e)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error occurs when loading HeadInfo:%s\n", e.what());
                        return false;
                    }

                    curFile = m_extraFullGraphFile + "_" + std::to_string(m_indexFiles.size());
                } while (fileexists(curFile.c_str()));
                if (m_staticTailPageBudget < 0 && m_staticMaxListPageCount > 0) {
                    p_opt.m_searchPostingPageLimit = (std::max)(
                        p_opt.m_searchPostingPageLimit, m_staticMaxListPageCount);
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Static unbounded tail: expanded search posting page limit to %d.\n",
                        p_opt.m_searchPostingPageLimit);
                }
                m_oneContext = (m_indexFiles.size() == 1);

                if (m_staticHasMetadata && p_opt.m_numTagsPerVec > 0 &&
                    p_opt.m_numTagsPerVec != m_staticNumTagsPerVec) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static metadata tag-column mismatch: index=%d runtime=%d\n",
                        m_staticNumTagsPerVec, p_opt.m_numTagsPerVec);
                    return false;
                }
                m_enableDeltaEncoding = p_opt.m_enableDeltaEncoding;
                m_enablePostingListRearrange = p_opt.m_enablePostingListRearrange;
                m_enableDataCompression = p_opt.m_enableDataCompression;
                m_enableDictTraining = p_opt.m_enableDictTraining;

                if (m_enablePostingListRearrange) m_parsePosting = &ExtraStaticSearcher<ValueType>::ParsePostingListRearrange;
                else m_parsePosting = &ExtraStaticSearcher<ValueType>::ParsePostingList;
                if (m_enableDeltaEncoding) m_parseEncoding = &ExtraStaticSearcher<ValueType>::ParseDeltaEncoding;
                else m_parseEncoding = &ExtraStaticSearcher<ValueType>::ParseEncoding;

                if (!LoadOrderedPageStarts(p_opt)) {
                    return false;
                }
                
                m_listPerFile = static_cast<int>((m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());

                p_versionMap.Load(p_opt.m_indexDirectory + FolderSep + p_opt.m_deleteIDFile, p_opt.m_datasetRowsInBlock, p_opt.m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", p_versionMap.Count());
                if (m_staticPipePQ && !OpenStaticRerankFile(p_opt, p_versionMap.Count())) {
                    return false;
                }

#ifndef _MSC_VER
                Helper::AIOTimeout.tv_nsec = p_opt.m_iotimeout * 1000;
#endif

                m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
                int maxIOThreads = max(p_opt.m_searchThreadNum, p_opt.m_iSSDNumberOfThreads);
                for (int i = 0; i < maxIOThreads; i++) {
                    m_freeWorkSpaceIds->push(i);
                }
                m_workspaceCount = maxIOThreads;
                m_available = true;
                return true;
            }

            virtual ErrorCode SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats,
                std::set<int>* truth, std::map<int, std::set<int>>* found)
            {
                if (RejectUnsupportedStaticFilter(p_exWorkSpace)) return ErrorCode::Fail;
                if (m_staticPipePQ) {
                    return SearchIndexPipePQ(p_exWorkSpace, p_queryResults, p_index, p_stats, truth, found);
                }
                if (HasStaticMetadataFilter(p_exWorkSpace)) {
                    auto& postingIDs = p_exWorkSpace->m_postingIDs;
                    postingIDs.erase(
                        std::remove_if(
                            postingIDs.begin(), postingIDs.end(),
                            [this, p_exWorkSpace](SizeType p_postingID) {
                                return p_postingID < 0 || p_postingID >= m_totalListCount ||
                                    StaticScanLimit(p_exWorkSpace, &m_listInfos[p_postingID]) == 0;
                            }),
                        postingIDs.end());
                }
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
                p_exWorkSpace->m_postingReadRanges.resize(postingListCount);

                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
 
                int diskRead = 0;
                int diskIO = 0;
                int listElements = 0;
                int scannedListElements = 0;
                const bool collectPostingContributionStats =
                    m_opt != nullptr && m_opt->m_collectPostingContributionStats;
                std::unordered_set<SizeType> uniqueMatchedVIDs;
                const bool profilePhases =
                    p_stats != nullptr && m_opt != nullptr && m_opt->m_logPhaseTime;
                std::atomic<std::int64_t> scanMicros{0};
                const auto retrievalStart = profilePhases
                    ? std::chrono::high_resolution_clock::now()
                    : std::chrono::high_resolution_clock::time_point{};

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext? 0: curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    auto& readRange = p_exWorkSpace->m_postingReadRanges[pi];
                    readRange = BuildStaticPostingReadRange(
                        p_exWorkSpace, curPostingID, listInfo);
                    const int readPageCount = readRange.m_readPageCount;
                    diskRead += readPageCount;
                    diskIO += 1;
                    const int scanCount = readRange.m_scanEnd - readRange.m_scanBegin;
                    listElements += scanCount;
                    scannedListElements += scanCount;

                    size_t totalBytes = static_cast<size_t>(readPageCount) << PageSizeEx;

#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset +
                        (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx);
                    request.m_readSize = totalBytes;
                    request.m_buffer = reinterpret_cast<char*>(
                        p_exWorkSpace->m_pageBuffers[pi].GetBuffer() +
                        (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx));
                    request.m_status = (fileid << 16) | (request.m_status & 0xffff);
                    request.m_payload = (void*)listInfo; 
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    request.m_callback = [&p_exWorkSpace, &queryResults, &p_index, &request, &listElements,
                                          &collectPostingContributionStats, &uniqueMatchedVIDs, &scanMicros,
                                          profilePhases, this](bool success)
                    {
                        if (!success) return;
                        const auto scanStart = profilePhases
                            ? std::chrono::high_resolution_clock::now()
                            : std::chrono::high_resolution_clock::time_point{};
                        const int staticPostingSlot = static_cast<int>(
                            &request - p_exWorkSpace->m_diskRequests.data());
                        char* buffer = reinterpret_cast<char*>(
                            p_exWorkSpace->m_pageBuffers[staticPostingSlot].GetBuffer());
                        ListInfo* listInfo = (ListInfo*)(request.m_payload);

                        // decompress posting list
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            DecompressPosting();
                        }

                        ProcessPosting();
                        if (profilePhases) {
                            scanMicros.fetch_add(
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::high_resolution_clock::now() - scanStart).count(),
                                std::memory_order_relaxed);
                        }
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
                    const int staticPostingSlot = static_cast<int>(pi);
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());
                    char* readBuffer = buffer +
                        (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx);
                    auto numRead = indexFile->ReadBinary(
                        totalBytes,
                        readBuffer,
                        listInfo->listOffset +
                            (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx));
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

                    const auto scanStart = profilePhases
                        ? std::chrono::high_resolution_clock::now()
                        : std::chrono::high_resolution_clock::time_point{};
                    ProcessPosting();
                    if (profilePhases) {
                        scanMicros.fetch_add(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::high_resolution_clock::now() - scanStart).count(),
                            std::memory_order_relaxed);
                    }
#endif
                }

                if (collectPostingContributionStats) {
                    uniqueMatchedVIDs.reserve(static_cast<size_t>((std::max)(scannedListElements, 0)));
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                if (!BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount)) {
                    return ErrorCode::DiskIOFail;
                }
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    const int staticPostingSlot = static_cast<int>(
                        request - p_exWorkSpace->m_diskRequests.data());
                    char* buffer = reinterpret_cast<char*>(
                        p_exWorkSpace->m_pageBuffers[staticPostingSlot].GetBuffer());
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    const auto scanStart = profilePhases
                        ? std::chrono::high_resolution_clock::now()
                        : std::chrono::high_resolution_clock::time_point{};
                    ProcessPosting();
                    if (profilePhases) {
                        scanMicros.fetch_add(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::high_resolution_clock::now() - scanStart).count(),
                            std::memory_order_relaxed);
                    }
                }
#endif
#endif
                if (profilePhases) {
                    const double retrievalMicros =
                        std::chrono::duration<double, std::micro>(
                            std::chrono::high_resolution_clock::now() - retrievalStart).count();
                    const double computationMicros =
                        static_cast<double>(scanMicros.load(std::memory_order_relaxed));
                    p_stats->m_diskReadLatency =
                        (std::max)(0.0, retrievalMicros - computationMicros) / 1000.0;
                    p_stats->m_compLatency = computationMicros / 1000.0;
                }
                if (truth) {
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

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

                        const int staticPostingSlot = static_cast<int>(pi);
                        const int scanBegin = StaticScanBegin(p_exWorkSpace, staticPostingSlot, listInfo);
                        const int scanEnd = StaticScanEnd(p_exWorkSpace, staticPostingSlot, listInfo);
                        for (int i = scanBegin; i < scanEnd; ++i) {
                            uint64_t offsetVectorID = m_enablePostingListRearrange ? (m_vectorInfoSize - sizeof(int)) * listInfo->listEleCount + sizeof(int) * i : m_vectorInfoSize * i;
                            const char* record = p_postingListFullData + offsetVectorID;
                            int vectorID;
                            std::memcpy(&vectorID, record, sizeof(vectorID));
                            if (!StaticRecordMatchesFilter(p_exWorkSpace, record)) continue;
                            if (truth && truth->count(vectorID)) (*found)[curPostingID].insert(vectorID);
                        }
                    }
                }

                if (p_stats) 
                {
                    p_stats->m_totalListElementsCount = listElements;
                    p_stats->m_diskIOCount = diskIO;
                    p_stats->m_diskAccessCount = diskRead;
                }
                p_exWorkSpace->m_postingProbeStats.m_readPostings += postingListCount;
                p_exWorkSpace->m_postingProbeStats.m_scannedVectors +=
                    static_cast<std::uint64_t>((std::max)(scannedListElements, 0));
                p_exWorkSpace->m_postingProbeStats.m_postingPageReads +=
                    static_cast<std::uint64_t>((std::max)(diskRead, 0));
                p_exWorkSpace->m_postingProbeStats.m_postingLogicalBytes +=
                    static_cast<std::uint64_t>((std::max)(scannedListElements, 0)) *
                    static_cast<std::uint64_t>(m_vectorInfoSize);
                p_exWorkSpace->m_postingProbeStats.m_postingPhysicalBytes +=
                    static_cast<std::uint64_t>((std::max)(diskRead, 0)) * PageSize;
                queryResults.SetScanned(listElements);
                return ErrorCode::Success;
            }

            virtual ErrorCode SearchIndexWithoutParsing(ExtraWorkSpace *p_exWorkSpace) override
            {
                if (RejectUnsupportedStaticFilter(p_exWorkSpace)) return ErrorCode::Fail;
                if (HasStaticMetadataFilter(p_exWorkSpace)) {
                    auto& postingIDs = p_exWorkSpace->m_postingIDs;
                    postingIDs.erase(
                        std::remove_if(
                            postingIDs.begin(), postingIDs.end(),
                            [this, p_exWorkSpace](SizeType p_postingID) {
                                return p_postingID < 0 || p_postingID >= m_totalListCount ||
                                    StaticScanLimit(p_exWorkSpace, &m_listInfos[p_postingID]) == 0;
                            }),
                        postingIDs.end());
                }
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
                p_exWorkSpace->m_postingReadRanges.resize(postingListCount);

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

                    auto& readRange = p_exWorkSpace->m_postingReadRanges[pi];
                    readRange = BuildStaticPostingReadRange(
                        p_exWorkSpace, curPostingID, listInfo);
                    const int readPageCount = readRange.m_readPageCount;
                    diskRead += readPageCount;
                    diskIO += 1;
                    listElements += readRange.m_scanEnd - readRange.m_scanBegin;

                    size_t totalBytes = static_cast<size_t>(readPageCount) << PageSizeEx;
                    
#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset +
                        (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx);
                    request.m_readSize = totalBytes;
                    request.m_buffer = reinterpret_cast<char*>(
                        p_exWorkSpace->m_pageBuffers[pi].GetBuffer() +
                        (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx));
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
                    auto numRead = indexFile->ReadBinary(
                        totalBytes,
                        buffer + (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx),
                        listInfo->listOffset +
                            (static_cast<std::uint64_t>(readRange.m_readStartPage) << PageSizeEx));
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
                QueryResult& p_queryResults,
		        std::shared_ptr<VectorIndex>& p_index, const VectorIndex* p_spann) override
            {
                if (RejectUnsupportedStaticFilter(p_exWorkSpace)) return ErrorCode::Fail;
                COMMON::QueryResultSet<ValueType>& headResults = *((COMMON::QueryResultSet<ValueType>*) & p_headResults);
                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
                bool foundResult = false;
                BasicResult* head = headResults.GetResult(p_exWorkSpace->m_ri);
                while (!foundResult && p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) {
                    if (head && head->VID != -1 && p_exWorkSpace->m_ri <= p_exWorkSpace->m_pi) {
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
                if (!foundResult && head && head->VID != -1) {
                    queryResults.AddPoint(head->VID, head->Dist);
                    head = headResults.GetResult(++p_exWorkSpace->m_ri);
                    foundResult = true;
                }
                if (foundResult) p_queryResults.SetScanned(p_queryResults.GetScanned() + 1);
                return (foundResult)? ErrorCode::Success : ErrorCode::VectorNotFound;
            }

            virtual ErrorCode SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
                QueryResult& p_query,
		        std::shared_ptr<VectorIndex> p_index, const VectorIndex* p_spann) override
            {
                if (RejectUnsupportedStaticFilter(p_exWorkSpace)) return ErrorCode::Fail;
                if (p_exWorkSpace->m_loadPosting) {
                    ErrorCode ret = SearchIndexWithoutParsing(p_exWorkSpace);
                    if (ret != ErrorCode::Success) return ret;
                    p_exWorkSpace->m_ri = 0;
                    p_exWorkSpace->m_pi = 0;
                    p_exWorkSpace->m_offset = p_exWorkSpace->m_postingIDs.empty()
                        ? 0
                        : StaticScanBegin(
                            p_exWorkSpace,
                            0,
                            &(m_listInfos[p_exWorkSpace->m_postingIDs.front()]));
                    p_exWorkSpace->m_loadPosting = false;
                }

                return SearchNextInPosting(p_exWorkSpace, p_headResults, p_query, p_index, p_spann);
            }

            std::string GetPostingListFullData(
                int postingListId,
                size_t p_postingListSize,
                Selection &p_selections,
                std::shared_ptr<VectorSet> p_fullVectors,
                bool p_enableDeltaEncoding = false,
                bool p_enablePostingListRearrange = false,
                const ValueType *headVector = nullptr,
                const std::vector<int>* p_orderedPageStartAttrs = nullptr,
                int p_pureCount = -1)
            {
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

                    int vid = p_selections[selectIdx++].tonode;
                    vectorID.append(reinterpret_cast<char *>(&vid), sizeof(int));
                    if (m_staticHasMetadata) {
                        if (vid < 0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Static metadata tag VID out of range: %d\n", vid);
                            throw std::runtime_error("Static metadata tag VID out of range");
                        }
                        const size_t tagOffset =
                            static_cast<size_t>(vid) * static_cast<size_t>(m_staticNumTagsPerVec);
                        if (tagOffset + static_cast<size_t>(m_staticNumTagsPerVec) >
                                           m_staticBuildTags.size()) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Static metadata tag VID out of range: %d\n", vid);
                            throw std::runtime_error("Static metadata tag VID out of range");
                        }
                        vectorID.append(
                            reinterpret_cast<const char*>(m_staticBuildTags.data() + tagOffset),
                            static_cast<size_t>(m_staticNumTagsPerVec) * sizeof(uint32_t));
                    }

                    if (m_staticPipePQ)
                    {
                        if (vid < 0 || static_cast<size_t>(vid) >= m_staticPipePQN) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Static PipePQ code VID out of range: %d (N=%zu)\n",
                                         vid, m_staticPipePQN);
                            throw std::runtime_error("Static PipePQ code VID out of range");
                        }
                        vector.append(reinterpret_cast<const char*>(
                                          m_staticPipePQCodes + static_cast<size_t>(vid) * m_staticPipePQCodeBytes),
                                      static_cast<size_t>(m_staticPipePQCodeBytes));
                    }
                    else if (p_enableDeltaEncoding)
                    {
                        ValueType *p_vector = reinterpret_cast<ValueType *>(p_fullVectors->GetVector(vid));
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
                        ValueType *p_vector = reinterpret_cast<ValueType *>(p_fullVectors->GetVector(vid));
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
                if (p_orderedPageStartAttrs != nullptr && !p_orderedPageStartAttrs->empty()) {
                    SortStaticPurePostingByAttrs(
                        postingListFullData,
                        p_pureCount < 0 ? static_cast<int>(p_postingListSize) : p_pureCount,
                        *p_orderedPageStartAttrs);
                }
                return postingListFullData;
            }

            bool AppendUnfilterTail(
                Selection& p_selections,
                std::vector<std::atomic_int>& p_postingListSize,
                const std::unordered_map<SizeType, SizeType>& p_headVectorIDs,
                std::shared_ptr<VectorSet> p_fullVectors,
                std::shared_ptr<VectorIndex> p_headIndex,
                SizeType p_fullCount,
                Options& p_opt,
                std::vector<int>& p_pureCountPerHead)
            {
                const int requestedReplicaCount = p_opt.m_tailReplicaCount;
                if (requestedReplicaCount <= 0) return true;
                if (m_staticPipePQ) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ does not support unfilter tails.\n");
                    return false;
                }
                if (p_fullVectors == nullptr || p_headIndex == nullptr ||
                    p_headIndex->GetNumSamples() != p_postingListSize.size()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static unfilter-tail head/posting cardinality mismatch.\n");
                    return false;
                }

                const SizeType headCount = p_headIndex->GetNumSamples();
                const int replicaCount = (std::min)(requestedReplicaCount, static_cast<int>(headCount));
                p_pureCountPerHead.resize(static_cast<size_t>(headCount));
                for (SizeType h = 0; h < headCount; ++h) {
                    p_pureCountPerHead[static_cast<size_t>(h)] =
                        p_postingListSize[static_cast<size_t>(h)].load();
                }

                // The post-cut selection array still includes dropped pure candidates.
                // Compact it before appending tails so only persisted pure records consume
                // tail capacity and the resulting prefixes remain contiguous.
                std::vector<Edge> pureSelections;
                pureSelections.reserve(p_selections.m_selections.size());
                size_t read = 0;
                while (read < p_selections.m_selections.size()) {
                    const int head = p_selections.m_selections[read].node;
                    size_t end = read + 1;
                    while (end < p_selections.m_selections.size() &&
                           p_selections.m_selections[end].node == head) {
                        ++end;
                    }
                    if (head >= 0 && static_cast<SizeType>(head) < headCount) {
                        const int pureCount = (std::max)(
                            0,
                            (std::min)(
                                p_pureCountPerHead[static_cast<size_t>(head)],
                                static_cast<int>(end - read)));
                        pureSelections.insert(
                            pureSelections.end(),
                            p_selections.m_selections.begin() + read,
                            p_selections.m_selections.begin() + read + pureCount);
                        p_postingListSize[static_cast<size_t>(head)] = pureCount;
                    }
                    read = end;
                }
                p_selections.m_selections.swap(pureSelections);
                // The old selection vector may contain dropped replicas. Release it
                // before tail collection instead of carrying both copies through
                // Phase 4.
                std::vector<Edge>().swap(pureSelections);
                p_selections.m_start = 0;
                p_selections.m_end = p_selections.m_selections.size();
                p_selections.m_totalsize = p_selections.m_end;
                const size_t pureSelectionCount = p_selections.m_selections.size();

                const int recordBytes = m_vectorInfoSize;
                const bool unboundedTail = p_opt.m_unfilterTailBufferLength < 0;
                const int extraTailPages = unboundedTail
                    ? -1
                    : (std::max)(0, p_opt.m_unfilterTailBufferLength);
                auto recordsForPages = [recordBytes](int p_pages) {
                    return (std::max)(0, (p_pages * PageSize) / (std::max)(1, recordBytes));
                };
                auto pagesForRecords = [recordBytes](int p_records) {
                    return p_records <= 0 ? 0 : (p_records * recordBytes + PageSize - 1) / PageSize;
                };
                auto tailHardCapForHead = [&](SizeType p_head) {
                    const int pure = p_pureCountPerHead[static_cast<size_t>(p_head)];
                    if (unboundedTail) return (std::numeric_limits<int>::max)();
                    return (std::max)(pure, recordsForPages(pagesForRecords(pure) + extraTailPages));
                };
                auto sparseTailLastPageKeep = [recordBytes](int p_pure, int p_keep) {
                    if (p_keep <= p_pure) return p_pure;
                    const int totalBytes = p_keep * recordBytes;
                    const int totalPages = (totalBytes + PageSize - 1) / PageSize;
                    if (totalPages <= 1) return p_keep;
                    const int lastPageStart = (totalPages - 1) * PageSize;
                    const int pureBytes = p_pure * recordBytes;
                    if (pureBytes > lastPageStart) return p_keep;
                    const int lastPageBytes = totalBytes - lastPageStart;
                    if (lastPageBytes >= (PageSize + 9) / 10) return p_keep;
                    return (std::max)(p_pure, lastPageStart / recordBytes);
                };

                const std::vector<Edge>& pure = p_selections.m_selections;
                const bool haveCrossBundleOwners =
                    m_staticBuildVectorOwners.size() == static_cast<size_t>(p_fullCount) &&
                    m_staticBuildHeadOwners.size() == static_cast<size_t>(headCount);
                const bool useSingleSeedCrossGraphTail =
                    haveCrossBundleOwners && static_cast<bool>(m_staticCrossGraphSearch);
                const bool useBundleFanoutRNGTail =
                    !useSingleSeedCrossGraphTail &&
                    haveCrossBundleOwners &&
                    m_staticHeadBundleLocalToGlobalHIDs != nullptr &&
                    m_staticHeadBundleIndexes.size() == m_staticHeadBundleLocalToGlobalHIDs->size() &&
                    !m_staticHeadBundleIndexes.empty();
                const bool useGlobalRNGTail = !useSingleSeedCrossGraphTail &&
                    haveCrossBundleOwners && !useBundleFanoutRNGTail;
                const auto& headOwners = m_staticHeadVectorOwnersView != nullptr
                    ? *m_staticHeadVectorOwnersView
                    : m_staticHeadVectorOwners;
                const char* tailSource = useSingleSeedCrossGraphTail
                    ? "single-seed-cross-graph-RNG"
                    : (useBundleFanoutRNGTail
                        ? "bundle-fanout-RNG-cross-bundle"
                        : (useGlobalRNGTail ? "global-RNG-cross-bundle" : "nearest-head"));
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Static Phase 4 (unfilter-tail): K_replica=%d, source=%s, recordBytes=%d, "
                    "extraTailPages=%d, scanning %d base vectors against %d heads\n",
                    replicaCount, tailSource,
                    recordBytes, extraTailPages, p_fullCount, headCount);
                std::atomic_size_t nextVector(0);
                std::atomic_size_t skippedDuplicate(0);
                std::atomic_size_t skippedCapacity(0);
                std::atomic<bool> tailSearchFailed(false);
                constexpr size_t kTailLockShards = 256;
                std::vector<std::vector<Edge>> tailCandidatesByHead(static_cast<size_t>(headCount));
                std::vector<std::mutex> tailCandidateLocks(kTailLockShards);
                const int threadCount = (std::max)(1, p_opt.m_iSSDNumberOfThreads);
                auto tailCandidateLess = [](const Edge& p_left, const Edge& p_right) {
                    if (p_left.distance != p_right.distance) {
                        return p_left.distance < p_right.distance;
                    }
                    return p_left.tonode < p_right.tonode;
                };
                auto offerTailCandidate = [&](SizeType p_vectorID, SizeType p_head, float p_distance) {
                    if (p_head < 0 || p_head >= headCount) return;

                    const size_t pureBegin = std::lower_bound(
                        pure.begin(), pure.end(), p_head, Selection::g_edgeComparer) - pure.begin();
                    const size_t pureEnd = (std::min)(
                        pure.size(),
                        pureBegin + static_cast<size_t>(
                            p_pureCountPerHead[static_cast<size_t>(p_head)]));
                    for (size_t i = pureBegin; i < pureEnd && pure[i].node == p_head; ++i) {
                        if (pure[i].tonode == p_vectorID) {
                            ++skippedDuplicate;
                            return;
                        }
                    }

                    Edge edge;
                    edge.node = p_head;
                    edge.tonode = p_vectorID;
                    edge.distance = p_distance;
                    const int capacity = (std::max)(
                        0,
                        tailHardCapForHead(p_head) -
                            p_pureCountPerHead[static_cast<size_t>(p_head)]);
                    if (capacity == 0) {
                        ++skippedCapacity;
                        return;
                    }

                    std::lock_guard<std::mutex> lock(
                        tailCandidateLocks[static_cast<size_t>(p_head) % kTailLockShards]);
                    auto& candidates = tailCandidatesByHead[static_cast<size_t>(p_head)];
                    auto existing = std::find_if(
                        candidates.begin(), candidates.end(),
                        [p_vectorID](const Edge& p_candidate) {
                            return p_candidate.tonode == p_vectorID;
                        });
                    if (existing != candidates.end()) {
                        if (tailCandidateLess(edge, *existing)) *existing = edge;
                        return;
                    }
                    if (static_cast<int>(candidates.size()) < capacity) {
                        candidates.push_back(edge);
                        return;
                    }

                    auto worst = std::max_element(
                        candidates.begin(), candidates.end(), tailCandidateLess);
                    if (worst != candidates.end() && tailCandidateLess(edge, *worst)) {
                        *worst = edge;
                    }
                    else {
                        ++skippedCapacity;
                    }
                };
                auto collectTailCandidates = [&]() {
                    COMMON::QueryResultSet<ValueType> nearbyHeads(nullptr, replicaCount);
                    std::vector<Edge> globalSelections(static_cast<size_t>(
                        (std::max)(replicaCount, m_opt->m_replicaCount)));
                    std::vector<std::pair<SizeType, float>> crossCandidates;
                    crossCandidates.reserve(static_cast<size_t>(
                        (std::max)(1, m_opt->m_internalResultNum)));
                    while (!tailSearchFailed.load(std::memory_order_acquire)) {
                        const SizeType vectorID = nextVector.fetch_add(1);
                        if (vectorID >= p_fullCount) break;
                        if (headOwners.count(vectorID) != 0 ||
                            p_headVectorIDs.count(vectorID) != 0) {
                            continue;
                        }

                        const ValueType* vector = static_cast<const ValueType*>(
                            p_fullVectors->GetVector(vectorID));
                        if (vector == nullptr) continue;
                        const int vectorOwner =
                            m_staticBuildVectorOwners[static_cast<size_t>(vectorID)];
                        if (useSingleSeedCrossGraphTail) {
                            crossCandidates.clear();
                            if (!m_staticCrossGraphSearch(
                                    vector,
                                    vectorOwner,
                                    (std::max)(1, m_opt->m_internalResultNum),
                                    crossCandidates)) {
                                tailSearchFailed.store(true, std::memory_order_release);
                                return;
                            }
                            int globalCount = 0;
                            if (!StaticCrossGraphRNGSelection(
                                    globalSelections,
                                    p_headIndex.get(),
                                    crossCandidates,
                                    vectorID,
                                    globalCount)) {
                                tailSearchFailed.store(true, std::memory_order_release);
                                return;
                            }
                            for (int rank = 0;
                                 rank < globalCount && rank < replicaCount;
                                 ++rank) {
                                const Edge& candidate =
                                    globalSelections[static_cast<size_t>(rank)];
                                if (candidate.node < 0 || candidate.node >= headCount ||
                                    m_staticBuildHeadOwners[static_cast<size_t>(candidate.node)] ==
                                        vectorOwner) {
                                    continue;
                                }
                                offerTailCandidate(vectorID, candidate.node, candidate.distance);
                            }
                            continue;
                        }
                        if (useBundleFanoutRNGTail) {
                            int globalCount = 0;
                            if (!StaticBundleFanoutRNGSelection(
                                    globalSelections,
                                    vector,
                                    vectorID,
                                    vectorOwner,
                                    globalCount)) {
                                continue;
                            }
                            for (int rank = 0;
                                 rank < globalCount && rank < replicaCount;
                                 ++rank) {
                                const Edge& candidate =
                                    globalSelections[static_cast<size_t>(rank)];
                                if (candidate.node < 0 || candidate.node >= headCount ||
                                    m_staticBuildHeadOwners[static_cast<size_t>(candidate.node)] ==
                                        vectorOwner) {
                                    continue;
                                }
                                offerTailCandidate(vectorID, candidate.node, candidate.distance);
                            }
                            continue;
                        }
                        if (useGlobalRNGTail) {
                            int globalCount = 0;
                            if (!StaticGlobalRNGSelection(
                                    globalSelections,
                                    vector,
                                    p_headIndex.get(),
                                    vectorID,
                                    globalCount)) {
                                continue;
                            }
                            for (int rank = 0;
                                 rank < globalCount && rank < replicaCount;
                                 ++rank) {
                                const Edge& candidate =
                                    globalSelections[static_cast<size_t>(rank)];
                                if (candidate.node < 0 || candidate.node >= headCount ||
                                    m_staticBuildHeadOwners[static_cast<size_t>(candidate.node)] ==
                                        vectorOwner) {
                                    continue;
                                }
                                offerTailCandidate(vectorID, candidate.node, candidate.distance);
                            }
                            continue;
                        }

                        nearbyHeads.SetTarget(vector, p_headIndex->m_pQuantizer);
                        nearbyHeads.Reset();
                        if (p_headIndex->SearchIndex(nearbyHeads) != ErrorCode::Success) continue;

                        BasicResult* results = nearbyHeads.GetResults();
                        for (int rank = 0; rank < replicaCount; ++rank) {
                            offerTailCandidate(
                                vectorID, results[rank].VID, results[rank].Dist);
                        }
                    }
                };

                std::vector<std::thread> workers;
                workers.reserve(threadCount);
                for (int t = 0; t < threadCount; ++t) workers.emplace_back(collectTailCandidates);
                for (auto& worker : workers) worker.join();
                if (tailSearchFailed.load(std::memory_order_acquire)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Single-seed cross-graph tail candidate search failed.\n");
                    return false;
                }

                size_t admittedTailCount = 0;
                for (SizeType head = 0; head < headCount; ++head) {
                    auto& candidates = tailCandidatesByHead[static_cast<size_t>(head)];
                    std::sort(candidates.begin(), candidates.end(), tailCandidateLess);
                    admittedTailCount += candidates.size();
                }

                std::vector<Edge> mergedSelections;
                mergedSelections.reserve(pureSelectionCount + admittedTailCount);
                size_t pureRead = 0;
                for (SizeType head = 0; head < headCount; ++head) {
                    const size_t pureBegin = pureRead;
                    while (pureRead < pure.size() && pure[pureRead].node == head) ++pureRead;
                    mergedSelections.insert(
                        mergedSelections.end(), pure.begin() + pureBegin, pure.begin() + pureRead);

                    auto& candidates = tailCandidatesByHead[static_cast<size_t>(head)];
                    for (const Edge& candidate : candidates) {
                        Edge tail = candidate;
                        tail.distance = (std::numeric_limits<float>::max)();
                        mergedSelections.push_back(tail);
                    }
                    p_postingListSize[static_cast<size_t>(head)] =
                        static_cast<int>((pureRead - pureBegin) + candidates.size());
                    std::vector<Edge>().swap(candidates);
                }
                std::vector<std::vector<Edge>>().swap(tailCandidatesByHead);
                if (pureRead != pure.size()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static tail merge encountered an invalid pure posting ID.\n");
                    return false;
                }

                if (unboundedTail) {
                    p_selections.m_selections.swap(mergedSelections);
                    p_selections.m_start = 0;
                    p_selections.m_end = p_selections.m_selections.size();
                    p_selections.m_totalsize = p_selections.m_end;
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Static Phase 4 done: pure=%zu tail=%zu skippedDuplicate=%zu "
                        "skippedCapacity=%zu trimmed=0 final=%zu cap=unbounded\n",
                        pureSelectionCount,
                        p_selections.m_selections.size() - pureSelectionCount,
                        skippedDuplicate.load(),
                        skippedCapacity.load(),
                        p_selections.m_selections.size());
                    return true;
                }

                std::vector<Edge> trimmedSelections;
                trimmedSelections.reserve(mergedSelections.size());
                size_t trimRead = 0;
                size_t tailRecordsTrimmed = 0;
                while (trimRead < mergedSelections.size()) {
                    const SizeType head = mergedSelections[trimRead].node;
                    size_t trimEnd = trimRead + 1;
                    while (trimEnd < mergedSelections.size() &&
                           mergedSelections[trimEnd].node == head) {
                        ++trimEnd;
                    }
                    const int pureCount = p_pureCountPerHead[static_cast<size_t>(head)];
                    const int totalCount = static_cast<int>(trimEnd - trimRead);
                    int keepCount = unboundedTail
                        ? totalCount
                        : (std::min)(totalCount, tailHardCapForHead(head));
                    if (!unboundedTail) {
                        keepCount = sparseTailLastPageKeep(pureCount, keepCount);
                    }
                    keepCount = (std::max)(pureCount, (std::min)(keepCount, totalCount));
                    trimmedSelections.insert(
                        trimmedSelections.end(),
                        mergedSelections.begin() + trimRead,
                        mergedSelections.begin() + trimRead + keepCount);
                    tailRecordsTrimmed += static_cast<size_t>(totalCount - keepCount);
                    p_postingListSize[static_cast<size_t>(head)] = keepCount;
                    trimRead = trimEnd;
                }

                for (SizeType head = 0; head < headCount; ++head) {
                    const int pureCount = p_pureCountPerHead[static_cast<size_t>(head)];
                    const int totalCount = p_postingListSize[static_cast<size_t>(head)].load();
                    const int maxPages = unboundedTail
                        ? (std::numeric_limits<int>::max)()
                        : pagesForRecords(pureCount) + extraTailPages;
                    if (totalCount < pureCount ||
                        (!unboundedTail && pagesForRecords(totalCount) > maxPages)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Static tail page-budget violation: head=%d pure=%d total=%d maxPages=%d\n",
                            head, pureCount, totalCount, maxPages);
                        return false;
                    }
                }

                p_selections.m_selections.swap(trimmedSelections);
                p_selections.m_start = 0;
                p_selections.m_end = p_selections.m_selections.size();
                p_selections.m_totalsize = p_selections.m_end;
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Static Phase 4 done: pure=%zu tail=%zu skippedDuplicate=%zu "
                    "skippedCapacity=%zu trimmed=%zu final=%zu cap=%s\n",
                    pureSelectionCount,
                    p_selections.m_selections.size() - pureSelectionCount,
                    skippedDuplicate.load(),
                    skippedCapacity.load(),
                    tailRecordsTrimmed,
                    p_selections.m_selections.size(),
                    unboundedTail ? "unbounded" :
                        ("purePages+" + std::to_string(extraTailPages)).c_str());
                return true;
            }

            bool StaticRNGSelection(std::vector<Edge>& p_selections,
                                    const ValueType* p_queryVector,
                                    VectorIndex* p_index,
                                    SizeType p_fullID,
                                    int& p_replicaCount,
                                    const std::vector<uint8_t>& p_allowedHeads,
                                    SizeType p_allowedHeadCount)
            {
                if (p_queryVector == nullptr || p_index == nullptr || p_index->m_pQuantizer != nullptr) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Node-aware static placement requires unquantized full-float head vectors.\n");
                    return false;
                }

                std::vector<std::pair<float, SizeType>> candidates;
                candidates.reserve(static_cast<size_t>((std::max)(1, p_allowedHeadCount)));
                for (SizeType head = 0; head < static_cast<SizeType>(p_allowedHeads.size()); ++head) {
                    if (p_allowedHeads[static_cast<size_t>(head)] == 0) continue;
                    const void* headVector = p_index->GetSample(head);
                    if (headVector == nullptr) continue;
                    candidates.emplace_back(p_index->ComputeDistance(p_queryVector, headVector), head);
                }
                if (candidates.empty()) {
                    p_replicaCount = 0;
                    return true;
                }

                const auto byDistance = [](const std::pair<float, SizeType>& p_left,
                                           const std::pair<float, SizeType>& p_right) {
                    return p_left.first == p_right.first
                        ? p_left.second < p_right.second
                        : p_left.first < p_right.first;
                };
                const size_t keepCount = (std::min)(
                    candidates.size(),
                    static_cast<size_t>((std::max)(1, m_opt->m_internalResultNum)));
                if (candidates.size() > keepCount) {
                    std::nth_element(
                        candidates.begin(), candidates.begin() + keepCount, candidates.end(), byDistance);
                    candidates.resize(keepCount);
                }
                std::sort(candidates.begin(), candidates.end(), byDistance);

                p_replicaCount = 0;
                for (const auto& candidate : candidates) {
                    if (p_replicaCount >= m_opt->m_replicaCount) break;
                    bool accepted = true;
                    for (int i = 0; i < p_replicaCount; ++i) {
                        const float headDistance = p_index->ComputeDistance(
                            p_index->GetSample(candidate.second),
                            p_index->GetSample(p_selections[static_cast<size_t>(i)].node));
                        if (m_opt->m_rngFactor * headDistance <= candidate.first) {
                            accepted = false;
                            break;
                        }
                    }
                    if (!accepted) continue;
                    Edge& selection = p_selections[static_cast<size_t>(p_replicaCount)];
                    selection.node = candidate.second;
                    selection.tonode = p_fullID;
                    selection.distance = candidate.first;
                    ++p_replicaCount;
                }
                return true;
            }

            bool StaticBundleRNGSelection(
                std::vector<Edge>& p_selections,
                const ValueType* p_queryVector,
                VectorIndex* p_index,
                const std::vector<SizeType>& p_localToGlobalHIDs,
                const std::vector<SizeType>& p_nodeHeadVectorIDs,
                SizeType p_fullID,
                int& p_replicaCount)
            {
                if (p_queryVector == nullptr || p_index == nullptr || p_index->m_pQuantizer != nullptr ||
                    p_localToGlobalHIDs.size() != p_nodeHeadVectorIDs.size()) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Bundle-aware static placement requires aligned unquantized bundle head vectors.\n");
                    return false;
                }

                p_replicaCount = 0;
                std::vector<const void*> selectedHeadVectors;
                selectedHeadVectors.reserve(p_selections.size());
                auto addCandidate = [&](SizeType p_localHeadID, float p_distance) {
                    if (p_replicaCount >= static_cast<int>(p_selections.size()) ||
                        p_localHeadID < 0 ||
                        static_cast<size_t>(p_localHeadID) >= p_localToGlobalHIDs.size()) {
                        return;
                    }
                    const void* candidateSample = p_index->GetSample(p_localHeadID);
                    if (candidateSample == nullptr) return;
                    for (const void* selectedSample : selectedHeadVectors) {
                        const float headDistance = p_index->ComputeDistance(candidateSample, selectedSample);
                        if (m_opt->m_rngFactor * headDistance <= p_distance) {
                            return;
                        }
                    }

                    Edge& selection = p_selections[static_cast<size_t>(p_replicaCount)];
                    selection.node = p_localToGlobalHIDs[static_cast<size_t>(p_localHeadID)];
                    selection.tonode = p_fullID;
                    selection.distance = p_distance;
                    selectedHeadVectors.push_back(candidateSample);
                    ++p_replicaCount;
                };

                const auto self = std::lower_bound(
                    p_nodeHeadVectorIDs.begin(), p_nodeHeadVectorIDs.end(), p_fullID);
                if (self != p_nodeHeadVectorIDs.end() && *self == p_fullID) {
                    if (m_opt->m_excludehead) return true;
                    addCandidate(static_cast<SizeType>(self - p_nodeHeadVectorIDs.begin()), 0.0f);
                }

                COMMON::QueryResultSet<ValueType> queryResults(
                    p_queryVector, (std::max)(1, m_opt->m_internalResultNum));
                if (p_index->SearchIndex(queryResults) != ErrorCode::Success) {
                    return false;
                }

                for (int i = 0; i < queryResults.GetResultNum() &&
                                p_replicaCount < static_cast<int>(p_selections.size()); ++i) {
                    BasicResult* result = queryResults.GetResult(i);
                    if (result == nullptr || result->VID == -1) break;
                    addCandidate(result->VID, result->Dist);
                }
                return true;
            }

            bool StaticGlobalRNGSelection(std::vector<Edge>& p_selections,
                                          const ValueType* p_queryVector,
                                          VectorIndex* p_index,
                                          SizeType p_fullID,
                                          int& p_replicaCount)
            {
                if (p_queryVector == nullptr || p_index == nullptr || p_index->m_pQuantizer != nullptr) {
                    return false;
                }

                COMMON::QueryResultSet<ValueType> queryResults(
                    p_queryVector, (std::max)(1, m_opt->m_internalResultNum));
                if (p_index->SearchIndex(queryResults) != ErrorCode::Success) {
                    return false;
                }

                p_replicaCount = 0;
                for (int i = 0; i < queryResults.GetResultNum() &&
                                p_replicaCount < m_opt->m_replicaCount; ++i) {
                    BasicResult* result = queryResults.GetResult(i);
                    if (result->VID == -1) break;

                    bool accepted = true;
                    for (int j = 0; j < p_replicaCount; ++j) {
                        const float headDistance = p_index->ComputeDistance(
                            p_index->GetSample(result->VID),
                            p_index->GetSample(p_selections[static_cast<size_t>(j)].node));
                        if (m_opt->m_rngFactor * headDistance <= result->Dist) {
                            accepted = false;
                            break;
                        }
                    }
                    if (!accepted) continue;
                    Edge& selection = p_selections[static_cast<size_t>(p_replicaCount)];
                    selection.node = result->VID;
                    selection.tonode = p_fullID;
                    selection.distance = result->Dist;
                    ++p_replicaCount;
                }
                return true;
            }

            bool StaticCrossGraphRNGSelection(
                std::vector<Edge>& p_selections,
                VectorIndex* p_index,
                const std::vector<std::pair<SizeType, float>>& p_candidates,
                SizeType p_fullID,
                int& p_replicaCount)
            {
                if (p_index == nullptr || p_index->m_pQuantizer != nullptr) {
                    return false;
                }

                p_replicaCount = 0;
                for (const auto& candidate : p_candidates) {
                    if (p_replicaCount >= static_cast<int>(p_selections.size()) ||
                        candidate.first < 0 || candidate.first >= p_index->GetNumSamples()) {
                        continue;
                    }

                    const void* candidateSample = p_index->GetSample(candidate.first);
                    if (candidateSample == nullptr) return false;

                    bool accepted = true;
                    for (int rank = 0; rank < p_replicaCount; ++rank) {
                        const SizeType selectedHead =
                            p_selections[static_cast<size_t>(rank)].node;
                        const void* selectedSample = p_index->GetSample(selectedHead);
                        if (selectedSample == nullptr) return false;
                        const float headDistance =
                            p_index->ComputeDistance(candidateSample, selectedSample);
                        if (m_opt->m_rngFactor * headDistance <= candidate.second) {
                            accepted = false;
                            break;
                        }
                    }
                    if (!accepted) continue;

                    Edge& selection =
                        p_selections[static_cast<size_t>(p_replicaCount)];
                    selection.node = candidate.first;
                    selection.tonode = p_fullID;
                    selection.distance = candidate.second;
                    ++p_replicaCount;
                }
                return true;
            }

            bool StaticBundleFanoutRNGSelection(
                std::vector<Edge>& p_selections,
                const ValueType* p_queryVector,
                SizeType p_fullID,
                int p_vectorOwner,
                int& p_replicaCount)
            {
                if (p_queryVector == nullptr || p_vectorOwner < 0 ||
                    m_staticHeadBundleLocalToGlobalHIDs == nullptr ||
                    m_staticHeadBundleIndexes.size() != m_staticHeadBundleLocalToGlobalHIDs->size()) {
                    return false;
                }

                struct Candidate {
                    float distance;
                    SizeType globalHeadID;
                    const void* sample;
                    VectorIndex* index;
                };

                std::vector<Candidate> candidates;
                const int candidateCount = (std::max)(1, m_opt->m_internalResultNum);
                for (size_t nodeId = 0; nodeId < m_staticHeadBundleIndexes.size(); ++nodeId) {
                    const auto& nodeIndex = m_staticHeadBundleIndexes[nodeId];
                    const auto& localToGlobal =
                        (*m_staticHeadBundleLocalToGlobalHIDs)[nodeId];
                    if (nodeIndex == nullptr || localToGlobal.empty() ||
                        nodeIndex->m_pQuantizer != nullptr) {
                        return false;
                    }

                    COMMON::QueryResultSet<ValueType> nodeResults(
                        p_queryVector,
                        (std::min)(candidateCount, static_cast<int>(localToGlobal.size())));
                    if (nodeIndex->SearchIndex(nodeResults) != ErrorCode::Success) {
                        return false;
                    }
                    for (int i = 0; i < nodeResults.GetResultNum(); ++i) {
                        BasicResult* result = nodeResults.GetResult(i);
                        if (result == nullptr || result->VID == -1) break;
                        if (result->VID < 0 ||
                            static_cast<size_t>(result->VID) >= localToGlobal.size()) {
                            continue;
                        }
                        const void* sample = nodeIndex->GetSample(result->VID);
                        if (sample == nullptr) continue;
                        candidates.push_back(
                            { result->Dist,
                              localToGlobal[static_cast<size_t>(result->VID)],
                              sample,
                              nodeIndex.get() });
                    }
                }

                std::sort(candidates.begin(), candidates.end(),
                          [](const Candidate& p_left, const Candidate& p_right) {
                              return p_left.distance == p_right.distance
                                  ? p_left.globalHeadID < p_right.globalHeadID
                                  : p_left.distance < p_right.distance;
                          });

                p_replicaCount = 0;
                std::vector<const void*> selectedHeadVectors;
                selectedHeadVectors.reserve(p_selections.size());
                for (const Candidate& candidate : candidates) {
                    if (p_replicaCount >= static_cast<int>(p_selections.size())) break;

                    bool accepted = true;
                    for (const void* selectedSample : selectedHeadVectors) {
                        const float headDistance =
                            candidate.index->ComputeDistance(candidate.sample, selectedSample);
                        if (m_opt->m_rngFactor * headDistance <= candidate.distance) {
                            accepted = false;
                            break;
                        }
                    }
                    if (!accepted) continue;

                    bool duplicate = false;
                    for (int i = 0; i < p_replicaCount; ++i) {
                        if (p_selections[static_cast<size_t>(i)].node == candidate.globalHeadID) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (duplicate) continue;

                    Edge& selection = p_selections[static_cast<size_t>(p_replicaCount)];
                    selection.node = candidate.globalHeadID;
                    selection.tonode = p_fullID;
                    selection.distance = candidate.distance;
                    selectedHeadVectors.push_back(candidate.sample);
                    ++p_replicaCount;
                }
                return true;
            }

            bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt, COMMON::VersionLabel& p_versionMap, COMMON::Dataset<std::uint64_t>& p_vectorTranslateMap, SizeType upperBound = -1) {
                std::string outputFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                if (outputFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Output file can't be empty!\n");
                    return false;
                }

                m_opt = &p_opt;
                m_staticBuildVectorOwners.clear();
                m_staticBuildHeadOwners.clear();
                int numThreads = p_opt.m_iSSDNumberOfThreads;
                int candidateNum = p_opt.m_internalResultNum;
                std::unordered_map<SizeType, SizeType> headVectorIDS;
                if (p_opt.m_headIDFile.empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                    return false;
                }

                SizeType fullCount = 0;
                size_t vectorInfoSize = 0;
                size_t fullVectorDataSize = 0;
                {
                    auto fullVectors = p_reader->GetVectorSet();
                    fullCount = fullVectors->Count();
                    fullVectorDataSize = fullVectors->PerVectorDataSize();
                    vectorInfoSize = fullVectorDataSize + sizeof(int);
                }
                if (upperBound > 0) fullCount = upperBound;
                if (!ConfigureStaticPipePQ(p_opt, fullCount, true)) {
                    return false;
                }
                if (!ConfigureStaticMetadataForBuild(p_opt, fullCount)) {
                    return false;
                }
                if (m_staticHasMetadata) {
                    vectorInfoSize = fullVectorDataSize + static_cast<size_t>(m_staticMetadataBytes);
                }
                else if (m_staticPipePQ) {
                    vectorInfoSize = sizeof(int) + static_cast<size_t>(m_staticPipePQCodeBytes);
                }
                m_vectorInfoSize = static_cast<int>(vectorInfoSize);

                p_versionMap.Initialize(fullCount, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);

                const std::vector<std::vector<SizeType>>& placementSource =
                    !m_staticPrimaryNodeVectorAssignments.empty()
                        ? m_staticPrimaryNodeVectorAssignments
                        : m_staticNodeVectorAssignments;
                bool useNodeAwareBuild = !placementSource.empty();
                std::vector<std::vector<SizeType>> plannedNodeVectors;
                std::vector<int> vectorOwner(static_cast<size_t>(fullCount), -1);
                size_t plannedAssignmentCount = static_cast<size_t>(fullCount);
                if (useNodeAwareBuild) {
                    if (p_opt.m_batches > 1 || p_headIndex->m_pQuantizer != nullptr) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Node-aware static placement requires Batches=1 and unquantized heads.\n");
                        return false;
                    }
                    plannedNodeVectors.resize(placementSource.size());
                    std::vector<uint8_t> claimed(static_cast<size_t>(fullCount), 0);
                    plannedAssignmentCount = 0;
                    for (size_t nodeId = 0; nodeId < placementSource.size(); ++nodeId) {
                        for (SizeType vectorId : placementSource[nodeId]) {
                            if (vectorId < 0 || vectorId >= fullCount ||
                                claimed[static_cast<size_t>(vectorId)] != 0) {
                                continue;
                            }
                            claimed[static_cast<size_t>(vectorId)] = 1;
                            vectorOwner[static_cast<size_t>(vectorId)] = static_cast<int>(nodeId);
                            plannedNodeVectors[nodeId].push_back(vectorId);
                            ++plannedAssignmentCount;
                        }
                    }
                    if (plannedAssignmentCount != static_cast<size_t>(fullCount)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Node-aware static placement covers %zu/%d vectors.\n",
                            plannedAssignmentCount, fullCount);
                        return false;
                    }
                }

                bool useBundleLocalNodeAwareBuild = useNodeAwareBuild &&
                    m_staticHeadBundleLocalToGlobalHIDs != nullptr &&
                    m_staticHeadBundleNodeHeadVectorIDs != nullptr &&
                    m_staticHeadBundleIndexes.size() == plannedNodeVectors.size() &&
                    m_staticHeadBundleLocalToGlobalHIDs->size() == plannedNodeVectors.size() &&
                    m_staticHeadBundleNodeHeadVectorIDs->size() == plannedNodeVectors.size();
                if (useBundleLocalNodeAwareBuild) {
                    for (size_t nodeId = 0; nodeId < plannedNodeVectors.size(); ++nodeId) {
                        const auto& localToGlobal =
                            (*m_staticHeadBundleLocalToGlobalHIDs)[nodeId];
                        const auto& nodeHeadVectorIDs =
                            (*m_staticHeadBundleNodeHeadVectorIDs)[nodeId];
                        if (m_staticHeadBundleIndexes[nodeId] == nullptr ||
                            localToGlobal.empty() ||
                            localToGlobal.size() != nodeHeadVectorIDs.size() ||
                            !std::is_sorted(nodeHeadVectorIDs.begin(), nodeHeadVectorIDs.end())) {
                            useBundleLocalNodeAwareBuild = false;
                            break;
                        }
                    }
                }

                if (!useBundleLocalNodeAwareBuild) {
                    for (int i = 0; i < p_vectorTranslateMap.R(); i++) {
                        headVectorIDS[static_cast<SizeType>(*(p_vectorTranslateMap[i]))] = i;
                    }
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Loaded %u Vector IDs for global static placement\n",
                        static_cast<uint32_t>(headVectorIDS.size()));
                } else {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Using bundle-local BKT placement; skipping global head-ID hash and masks.\n");
                }
                const auto& staticHeadOwners = m_staticHeadVectorOwnersView != nullptr
                    ? *m_staticHeadVectorOwnersView
                    : m_staticHeadVectorOwners;
                const auto isHeadVector = [&](SizeType p_vectorID) {
                    return headVectorIDS.count(p_vectorID) != 0 ||
                        staticHeadOwners.count(p_vectorID) != 0;
                };

                Selection selections(
                    (useNodeAwareBuild ? plannedAssignmentCount : static_cast<size_t>(fullCount)) *
                        p_opt.m_replicaCount,
                    p_opt.m_tmpdir);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
                std::vector<std::atomic_int> replicaCount(fullCount);
                std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
                for (auto& rc : replicaCount) rc = 0;
                for (auto& pls : postingListSize) pls = 0;
                std::unordered_set<SizeType> emptySet;
                SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (useNodeAwareBuild) {
                    auto fullVectors = p_reader->GetVectorSet();
                    if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) {
                        fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);
                    }

                    std::vector<int> headToNode(p_headIndex->GetNumSamples(), -1);
                    std::vector<std::vector<uint8_t>> allowedHeadMasks;
                    std::vector<SizeType> allowedHeadCounts;
                    if (useBundleLocalNodeAwareBuild) {
                        for (size_t nodeId = 0; nodeId < plannedNodeVectors.size(); ++nodeId) {
                            const auto& localToGlobal =
                                (*m_staticHeadBundleLocalToGlobalHIDs)[nodeId];
                            for (SizeType globalHeadID : localToGlobal) {
                                if (globalHeadID < 0 ||
                                    globalHeadID >= p_headIndex->GetNumSamples() ||
                                    headToNode[static_cast<size_t>(globalHeadID)] != -1) {
                                    SPTAGLIB_LOG(
                                        Helper::LogLevel::LL_Error,
                                        "Bundle-aware static placement has an invalid or duplicate global head ID.\n");
                                    return false;
                                }
                                headToNode[static_cast<size_t>(globalHeadID)] =
                                    static_cast<int>(nodeId);
                            }
                        }
                    } else {
                        for (const auto& pair : headVectorIDS) {
                            if (pair.first < 0 || pair.first >= fullCount) continue;
                            int owner = -1;
                            const auto ownerIt = staticHeadOwners.find(pair.first);
                            if (ownerIt != staticHeadOwners.end()) {
                                owner = ownerIt->second;
                            } else {
                                owner = vectorOwner[static_cast<size_t>(pair.first)];
                            }
                            if (owner < 0 || owner >= static_cast<int>(plannedNodeVectors.size())) {
                                SPTAGLIB_LOG(
                                    Helper::LogLevel::LL_Error,
                                    "Node-aware static placement cannot resolve owner for head vector %d.\n",
                                    pair.first);
                                return false;
                            }
                            headToNode[static_cast<size_t>(pair.second)] = owner;
                        }

                        allowedHeadMasks.assign(
                            plannedNodeVectors.size(),
                            std::vector<uint8_t>(p_headIndex->GetNumSamples(), 0));
                        allowedHeadCounts.assign(plannedNodeVectors.size(), 0);
                        for (SizeType head = 0; head < p_headIndex->GetNumSamples(); ++head) {
                            const int owner = headToNode[static_cast<size_t>(head)];
                            if (owner >= 0 && owner < static_cast<int>(allowedHeadMasks.size())) {
                                allowedHeadMasks[static_cast<size_t>(owner)][static_cast<size_t>(head)] = 1;
                                ++allowedHeadCounts[static_cast<size_t>(owner)];
                            }
                        }
                    }
                    for (size_t nodeId = 0; nodeId < plannedNodeVectors.size(); ++nodeId) {
                        const SizeType headCount = useBundleLocalNodeAwareBuild
                            ? static_cast<SizeType>(
                                (*m_staticHeadBundleLocalToGlobalHIDs)[nodeId].size())
                            : allowedHeadCounts[nodeId];
                        if (!plannedNodeVectors[nodeId].empty() && headCount == 0) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Node-aware static placement node %zu has vectors but no heads.\n",
                                nodeId);
                            return false;
                        }
                    }

                    std::vector<std::pair<int, SizeType>> assignments;
                    assignments.reserve(plannedAssignmentCount);
                    for (size_t nodeId = 0; nodeId < plannedNodeVectors.size(); ++nodeId) {
                        for (SizeType vectorId : plannedNodeVectors[nodeId]) {
                            assignments.emplace_back(static_cast<int>(nodeId), vectorId);
                        }
                    }

                    std::atomic_size_t sent(0);
                    std::vector<std::thread> threads;
                    threads.reserve(numThreads);
                    for (int tid = 0; tid < numThreads; ++tid) {
                        threads.emplace_back([&]() {
                            std::vector<Edge> localSelections(
                                static_cast<size_t>(p_opt.m_replicaCount));
                            while (true) {
                                const size_t assignmentIndex = sent.fetch_add(1);
                                if (assignmentIndex >= assignments.size()) return;

                                const int nodeId = assignments[assignmentIndex].first;
                                const SizeType vectorId = assignments[assignmentIndex].second;
                                const size_t selectionOffset =
                                    assignmentIndex * static_cast<size_t>(p_opt.m_replicaCount);
                                replicaCount[static_cast<size_t>(vectorId)] = 0;
                                int assignedCount = 0;

                                std::fill(localSelections.begin(), localSelections.end(), Edge());
                                int localCount = 0;
                                if (useBundleLocalNodeAwareBuild) {
                                    const auto& nodeIndex =
                                        m_staticHeadBundleIndexes[static_cast<size_t>(nodeId)];
                                    const auto& localToGlobal =
                                        (*m_staticHeadBundleLocalToGlobalHIDs)[static_cast<size_t>(nodeId)];
                                    const auto& nodeHeadVectorIDs =
                                        (*m_staticHeadBundleNodeHeadVectorIDs)[static_cast<size_t>(nodeId)];
                                    if (!StaticBundleRNGSelection(
                                            localSelections,
                                            static_cast<const ValueType*>(fullVectors->GetVector(vectorId)),
                                            nodeIndex.get(),
                                            localToGlobal,
                                            nodeHeadVectorIDs,
                                            vectorId,
                                            localCount)) {
                                        continue;
                                    }
                                } else {
                                    const auto headIt = headVectorIDS.find(vectorId);
                                    if (headIt != headVectorIDS.end() && p_opt.m_excludehead) {
                                        continue;
                                    }
                                    if (!p_opt.m_excludehead && headIt != headVectorIDS.end() &&
                                        headToNode[static_cast<size_t>(headIt->second)] == nodeId) {
                                        Edge& self = selections.m_selections[selectionOffset];
                                        self.node = headIt->second;
                                        self.tonode = vectorId;
                                        self.distance = 0.0f;
                                        ++postingListSize[static_cast<size_t>(self.node)];
                                        ++replicaCount[static_cast<size_t>(vectorId)];
                                        assignedCount = 1;
                                    }
                                    if (!StaticRNGSelection(
                                            localSelections,
                                            static_cast<const ValueType*>(fullVectors->GetVector(vectorId)),
                                            p_headIndex.get(),
                                            vectorId,
                                            localCount,
                                            allowedHeadMasks[static_cast<size_t>(nodeId)],
                                            allowedHeadCounts[static_cast<size_t>(nodeId)])) {
                                        continue;
                                    }
                                }
                                for (int i = 0; i < localCount &&
                                                assignedCount < p_opt.m_replicaCount; ++i) {
                                    const Edge& candidate = localSelections[static_cast<size_t>(i)];
                                    bool duplicate = false;
                                    for (int j = 0; j < assignedCount; ++j) {
                                        if (selections.m_selections[
                                                selectionOffset + static_cast<size_t>(j)].node ==
                                            candidate.node) {
                                            duplicate = true;
                                            break;
                                        }
                                    }
                                    if (duplicate) continue;
                                    selections.m_selections[
                                        selectionOffset + static_cast<size_t>(assignedCount)] = candidate;
                                    ++postingListSize[static_cast<size_t>(candidate.node)];
                                    ++replicaCount[static_cast<size_t>(vectorId)];
                                    ++assignedCount;
                                }
                            }
                        });
                    }
                    for (auto& thread : threads) thread.join();
                    m_staticBuildVectorOwners = std::move(vectorOwner);
                    m_staticBuildHeadOwners = std::move(headToNode);
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Node-aware static candidate search finished with %zu assignments across %zu nodes.\n",
                        assignments.size(), plannedNodeVectors.size());
                } else {
                    if (p_opt.m_batches > 1)
                    {
                        if (selections.SaveBatch() != ErrorCode::Success)
                        {
                            return false;
                        }
                    }
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
                        for (int j = start; j < end && sampleNum < sampleSize; j++)
                        {
                            if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                        }

                        float acc = 0;
                        for (int j = 0; j < sampleNum; j++)
                        {
                            COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                        }
                        acc = acc / sampleNum;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%d,%d) loaded with %d vectors (%zu) HeadIndex acc @%d:%f.\n", i, start, end, fullVectors->Count(), selections.m_selections.size(), candidateNum, acc);

                        p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
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

                if (!useNodeAwareBuild && p_opt.m_batches > 1)
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
                    p_opt.m_postingPageLimit = max(p_opt.m_postingPageLimit, static_cast<int>((p_opt.m_postingVectorLimit * vectorInfoSize + PageSize - 1) / PageSize));
                    p_opt.m_searchPostingPageLimit = p_opt.m_postingPageLimit;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build index with posting page limit:%d\n", p_opt.m_postingPageLimit);
                    postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);

                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (isHeadVector(i)) continue;
                        ++replicaCountDist[replicaCount[i]];
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
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
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                        return false;
                    }
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (isHeadVector(i)) continue;

                        ++replicaCountDist[replicaCount[i]];

                        if (replicaCount[i] < 2)
                        {
                            long long vid = i;
                            if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
                                return false;
                            }
                        }
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

                auto t4 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

                auto fullVectors = p_reader->GetVectorSet();
                if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine &&
                    !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) {
                    fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);
                }
                std::vector<int> orderedPageStartAttrs;
                std::vector<std::uint32_t> orderedPageStartBases;
                if (p_opt.m_enableOrderedPageStart &&
                    !ParseOrderedPageStartAttrs(
                        p_opt.m_orderedPageStartAttrs,
                        m_staticNumTagsPerVec,
                        orderedPageStartAttrs)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Invalid OrderedPageStartAttrs=%s\n",
                                 p_opt.m_orderedPageStartAttrs.c_str());
                    return false;
                }
                if (!p_opt.m_enableOrderedPageStart &&
                    !p_opt.m_orderedPageStartAttrs.empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Ordered page-starts disabled; ignoring OrderedPageStartAttrs=%s.\n",
                                 p_opt.m_orderedPageStartAttrs.c_str());
                }
                if (!orderedPageStartAttrs.empty()) {
                    if (!m_staticHasMetadata || p_opt.m_enablePostingListRearrange ||
                        p_opt.m_enableDataCompression || p_opt.m_enableDeltaEncoding ||
                        p_opt.m_ssdIndexFileNum != 1) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "OrderedPageStartAttrs requires one STM1 raw static snapshot "
                            "(no rearrange, compression, delta encoding, or sharding).\n");
                        return false;
                    }
                    orderedPageStartBases.assign(
                        orderedPageStartAttrs.size(),
                        (std::numeric_limits<std::uint32_t>::max)());
                    for (SizeType vid = 0; vid < fullCount; ++vid) {
                        for (size_t attrIndex = 0; attrIndex < orderedPageStartAttrs.size(); ++attrIndex) {
                            const int attr = orderedPageStartAttrs[attrIndex];
                            const std::uint32_t tag =
                                m_staticBuildTags[static_cast<size_t>(vid) * m_staticNumTagsPerVec + attr];
                            orderedPageStartBases[attrIndex] =
                                (std::min)(orderedPageStartBases[attrIndex], tag);
                        }
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Ordered page-start attrs enabled: %s\n",
                                 p_opt.m_orderedPageStartAttrs.c_str());
                }
                std::vector<int> pureCountPerHead;
                if (!AppendUnfilterTail(
                        selections,
                        postingListSize,
                        headVectorIDS,
                        fullVectors,
                        p_headIndex,
                        fullCount,
                        p_opt,
                        pureCountPerHead)) {
                    return false;
                }
                if (m_staticHasMetadata && pureCountPerHead.empty()) {
                    pureCountPerHead.resize(postingListSize.size());
                    for (size_t h = 0; h < postingListSize.size(); ++h) {
                        pureCountPerHead[h] = postingListSize[h].load();
                    }
                }

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

                // iterate over files
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListOffSet = i * postingFileSize;
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    // postingListSize: number of vectors in the posting list, type vector<int>
                    std::vector<int> curPostingListSizes(
                        postingListSize.begin() + curPostingListOffSet,
                        postingListSize.begin() + curPostingListEnd);
                    std::vector<int> curPureCounts;
                    const std::vector<int>* curPureCountsPtr = nullptr;
                    if (!pureCountPerHead.empty()) {
                        curPureCounts.assign(
                            pureCountPerHead.begin() + curPostingListOffSet,
                            pureCountPerHead.begin() + curPostingListEnd);
                        curPureCountsPtr = &curPureCounts;
                    }

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
                                    j, curPostingListSizes[j], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);

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
                                                                   selections, fullVectors, p_opt.m_enableDeltaEncoding,
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
                                                "Posting list %d/%d, compressed size: %d, compression ratio: %.4f\n",
                                                postingListId, postingListSize.size(), curPostingListBytes[j],
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
                            curPostingListBytes[j] = curPostingListSizes[j] * vectorInfoSize;
                        }
                    }

                    std::unique_ptr<int[]> postPageNum;
                    std::unique_ptr<std::uint16_t[]> postPageOffset;
                    std::vector<int> postingOrderInIndex;
                    SelectPostingOffset(curPostingListBytes, postPageNum, postPageOffset, postingOrderInIndex);

                    OutputSSDIndexFile((i == 0) ? outputFile : outputFile + "_" + std::to_string(i),
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
                        fullVectors,
                        curPostingListOffSet,
                        curPureCountsPtr,
                        p_opt.m_unfilterTailBufferLength,
                        orderedPageStartAttrs,
                        orderedPageStartBases);
                }

                p_versionMap.Save(p_opt.m_indexDirectory + FolderSep + p_opt.m_deleteIDFile);

                auto t5 = std::chrono::high_resolution_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
                return true;
            }

            virtual bool CheckValidPosting(SizeType postingID)
            {
                return m_listInfos[postingID].listEleCount != 0;
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

            virtual ErrorCode GetPostingDebug(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorIndex> p_index, SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs)
            {
                VIDs.clear();

                SizeType curPostingID = vid;
                ListInfo* listInfo = &(m_listInfos[curPostingID]);
                VIDs.resize(listInfo->listEleCount);
                ByteArray vector_array = ByteArray::Alloc(sizeof(ValueType) * listInfo->listEleCount * m_iDataDimension);
                vecs.reset(new BasicVectorSet(vector_array, GetEnumValueType<ValueType>(), m_iDataDimension, listInfo->listEleCount));

                int fileid = m_oneContext ? 0 : curPostingID / m_listPerFile;

#ifndef BATCH_READ
                Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);

#ifdef ASYNC_READ       
                auto& request = p_exWorkSpace->m_diskRequests[0];
                request.m_offset = listInfo->listOffset;
                request.m_readSize = totalBytes;
                request.m_status = (fileid << 16) | (request.m_status & 0xffff);
                request.m_payload = (void*)listInfo;
                request.m_success = false;

#ifdef BATCH_READ // async batch read
                request.m_callback = [&p_exWorkSpace, &vecs, &VIDs, &p_index, &request, this](bool success)
                {
                    char* buffer = request.m_buffer;
                    ListInfo* listInfo = (ListInfo*)(request.m_payload);

                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    for (int i = 0; i < listInfo->listEleCount; i++) 
                    {
                            uint64_t offsetVectorID, offsetVector; 
                            (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount); 
                            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID)); 
                            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector)); 
                            VIDs[i] = vectorID;
                            auto outVec = vecs->GetVector(i);
                            memcpy(outVec, (void*)(p_postingListFullData + offsetVector), sizeof(ValueType) * m_iDataDimension);
                    } 
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
                char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[0]).GetBuffer());
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

                for (int i = 0; i < listInfo->listEleCount; i++) 
                {
                    uint64_t offsetVectorID, offsetVector;
                    (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);
                    int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID));
                    (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));
                    VIDs[i] = vectorID;
                    auto outVec = vecs->GetVector(i);
                    memcpy(outVec, (void*)(p_postingListFullData + offsetVector), sizeof(ValueType) * m_iDataDimension);
                }
#endif
                return ErrorCode::Success;
            }

        private:
            struct ListInfo
            {
                std::size_t listTotalBytes = 0;

                int listEleCount = 0;

                int pureEleCount = 0;

                std::uint16_t listPageCount = 0;

                std::uint64_t listOffset = 0;

                std::uint16_t pageOffset = 0;
            };

            static constexpr std::uint32_t kOrderedPageStartMagic = 0x3153504FU; // "OPS1"
            static constexpr std::int32_t kOrderedPageStartVersion = 1;
            static constexpr std::int32_t kOrderedPageStartEmpty = (std::numeric_limits<std::int32_t>::max)();

            struct OrderedPageStartHeader
            {
                std::uint32_t magic;
                std::int32_t version;
                std::int32_t listCount;
                std::int32_t recordBytes;
                std::int32_t attrCount;
            };

            static bool ParseOrderedPageStartAttrs(const std::string& p_value,
                                                   int p_numTags,
                                                   std::vector<int>& p_attrs)
            {
                p_attrs.clear();
                if (p_value.empty()) return true;

                std::stringstream input(p_value);
                std::string token;
                while (std::getline(input, token, ',')) {
                    if (token.empty()) return false;
                    char* end = nullptr;
                    const long value = std::strtol(token.c_str(), &end, 10);
                    if (end == token.c_str() || *end != '\0' || value < 0 || value >= p_numTags) {
                        return false;
                    }
                    if (std::find(p_attrs.begin(), p_attrs.end(), static_cast<int>(value)) != p_attrs.end()) {
                        return false;
                    }
                    p_attrs.push_back(static_cast<int>(value));
                }
                std::sort(p_attrs.begin(), p_attrs.end());
                return !p_attrs.empty() && p_attrs.size() <= 2;
            }

            static std::int32_t OrderedPageStartBit(std::uint32_t p_tag, std::uint32_t p_levelBase)
            {
                if (p_tag < p_levelBase ||
                    static_cast<std::uint64_t>(p_tag) - p_levelBase >=
                        static_cast<std::uint64_t>(kOrderedPageStartEmpty)) {
                    return kOrderedPageStartEmpty;
                }
                return static_cast<std::int32_t>(p_tag - p_levelBase);
            }

            void SortStaticPurePostingByAttrs(std::string& p_posting,
                                               int p_pureCount,
                                               const std::vector<int>& p_attrs) const
            {
                if (p_attrs.empty() || !m_staticHasMetadata || p_pureCount <= 1) return;
                const int pureCount = (std::min)(
                    p_pureCount, static_cast<int>(p_posting.size() / static_cast<size_t>(m_vectorInfoSize)));
                if (pureCount <= 1) return;

                std::vector<int> order(static_cast<size_t>(pureCount));
                std::iota(order.begin(), order.end(), 0);
                const int sortColumns = *std::max_element(p_attrs.begin(), p_attrs.end()) + 1;
                auto tagAt = [this, &p_posting](int p_record, int p_column) {
                    std::uint32_t tag = 0;
                    const size_t offset = static_cast<size_t>(p_record) * m_vectorInfoSize + sizeof(int) +
                        static_cast<size_t>(p_column) * sizeof(tag);
                    std::memcpy(&tag, p_posting.data() + offset, sizeof(tag));
                    return tag;
                };
                std::stable_sort(order.begin(), order.end(), [&tagAt, sortColumns](int p_left, int p_right) {
                    for (int column = 0; column < sortColumns; ++column) {
                        const std::uint32_t leftTag = tagAt(p_left, column);
                        const std::uint32_t rightTag = tagAt(p_right, column);
                        if (leftTag != rightTag) return leftTag < rightTag;
                    }
                    return p_left < p_right;
                });

                bool changed = false;
                for (int i = 0; i < pureCount; ++i) {
                    if (order[static_cast<size_t>(i)] != i) {
                        changed = true;
                        break;
                    }
                }
                if (!changed) return;

                std::string sorted = p_posting;
                for (int i = 0; i < pureCount; ++i) {
                    std::memcpy(sorted.data() + static_cast<size_t>(i) * m_vectorInfoSize,
                                p_posting.data() +
                                    static_cast<size_t>(order[static_cast<size_t>(i)]) * m_vectorInfoSize,
                                m_vectorInfoSize);
                }
                p_posting.swap(sorted);
            }

            bool BuildOrderedPageStartsForPosting(const std::string& p_posting,
                                                   int p_pureCount,
                                                   std::uint16_t p_pageOffset,
                                                   std::uint16_t p_pageCount,
                                                   const std::vector<int>& p_attrs,
                                                   const std::vector<std::uint32_t>& p_bases,
                                                   std::vector<std::int32_t>& p_starts) const
            {
                p_starts.assign(static_cast<size_t>(p_attrs.size()) * p_pageCount, kOrderedPageStartEmpty);
                if (p_attrs.empty() || !m_staticHasMetadata || p_pureCount <= 0) return true;
                const int pureCount = (std::min)(
                    p_pureCount, static_cast<int>(p_posting.size() / static_cast<size_t>(m_vectorInfoSize)));

                for (size_t attrIndex = 0; attrIndex < p_attrs.size(); ++attrIndex) {
                    const int attr = p_attrs[attrIndex];
                    std::int32_t previousRecord = (std::numeric_limits<std::int32_t>::min)();
                    for (int record = 0; record < pureCount; ++record) {
                        std::uint32_t tag = 0;
                        const size_t tagOffset =
                            static_cast<size_t>(record) * m_vectorInfoSize + sizeof(int) +
                            static_cast<size_t>(attr) * sizeof(tag);
                        std::memcpy(&tag, p_posting.data() + tagOffset, sizeof(tag));
                        const std::int32_t bit =
                            OrderedPageStartBit(tag, p_bases[attrIndex]);
                        if (bit < previousRecord) return false;
                        previousRecord = bit;
                    }

                    std::int32_t previous = (std::numeric_limits<std::int32_t>::min)();
                    for (int page = 0; page < p_pageCount; ++page) {
                        const std::int64_t pageByte = static_cast<std::int64_t>(page) * PageSize;
                        const std::int64_t relative = pageByte - p_pageOffset;
                        int record = relative <= 0 ? 0 :
                            static_cast<int>((relative + m_vectorInfoSize - 1) / m_vectorInfoSize);
                        if (record >= pureCount) continue;

                        std::uint32_t tag = 0;
                        const size_t tagOffset = static_cast<size_t>(record) * m_vectorInfoSize + sizeof(int) +
                            static_cast<size_t>(attr) * sizeof(tag);
                        std::memcpy(&tag, p_posting.data() + tagOffset, sizeof(tag));
                        const std::int32_t start =
                            OrderedPageStartBit(tag, p_bases[attrIndex]);
                        if (start < previous) return false;
                        p_starts[attrIndex * static_cast<size_t>(p_pageCount) + page] = start;
                        previous = start;
                    }
                }
                return true;
            }

            bool SaveOrderedPageStarts(const std::string& p_path,
                                       const std::vector<std::vector<std::int32_t>>& p_starts,
                                       const std::vector<int>& p_attrs,
                                       const std::vector<std::uint32_t>& p_bases) const
            {
                std::ofstream output(p_path, std::ios::binary | std::ios::trunc);
                if (!output) return false;

                const OrderedPageStartHeader header{
                    kOrderedPageStartMagic,
                    kOrderedPageStartVersion,
                    static_cast<std::int32_t>(p_starts.size()),
                    m_vectorInfoSize,
                    static_cast<std::int32_t>(p_attrs.size()),
                };
                output.write(reinterpret_cast<const char*>(&header), sizeof(header));
                for (size_t i = 0; i < p_attrs.size(); ++i) {
                    const std::int32_t attr = p_attrs[i];
                    output.write(reinterpret_cast<const char*>(&attr), sizeof(attr));
                    output.write(reinterpret_cast<const char*>(&p_bases[i]), sizeof(p_bases[i]));
                }
                for (const auto& starts : p_starts) {
                    const std::int32_t count = static_cast<std::int32_t>(starts.size());
                    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
                    if (!starts.empty()) {
                        output.write(reinterpret_cast<const char*>(starts.data()),
                                     static_cast<std::streamsize>(starts.size() * sizeof(starts[0])));
                    }
                }
                return static_cast<bool>(output);
            }

            bool LoadOrderedPageStarts(const Options& p_opt)
            {
                m_orderedPageStartAttrs.clear();
                m_orderedPageStartBases.clear();
                m_orderedPageStartOffsets.clear();
                m_orderedPageStartBits.clear();
                if (!p_opt.m_enableOrderedPageStart) return true;

                std::vector<int> requestedAttrs;
                if (!ParseOrderedPageStartAttrs(
                        p_opt.m_orderedPageStartAttrs, m_staticNumTagsPerVec, requestedAttrs)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Invalid OrderedPageStartAttrs=%s\n",
                                 p_opt.m_orderedPageStartAttrs.c_str());
                    return false;
                }
                if (requestedAttrs.empty()) return true;
                if (!m_staticHasMetadata) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "OrderedPageStartAttrs requires an STM1 metadata snapshot.\n");
                    return false;
                }

                const std::string path = p_opt.m_indexDirectory + FolderSep + "ordered_page_starts.bin";
                std::ifstream input(path, std::ios::binary);
                OrderedPageStartHeader header{};
                if (!input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
                    header.magic != kOrderedPageStartMagic ||
                    header.version != kOrderedPageStartVersion ||
                    header.listCount != m_totalListCount ||
                    header.recordBytes != m_vectorInfoSize ||
                    header.attrCount != static_cast<std::int32_t>(requestedAttrs.size())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Invalid ordered page-start directory: %s\n", path.c_str());
                    return false;
                }

                m_orderedPageStartAttrs.resize(requestedAttrs.size());
                m_orderedPageStartBases.resize(requestedAttrs.size());
                for (size_t i = 0; i < requestedAttrs.size(); ++i) {
                    std::int32_t attr = -1;
                    if (!input.read(reinterpret_cast<char*>(&attr), sizeof(attr)) ||
                        !input.read(reinterpret_cast<char*>(&m_orderedPageStartBases[i]),
                                    sizeof(m_orderedPageStartBases[i])) ||
                        attr != requestedAttrs[i]) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Ordered page-start attributes do not match %s\n", path.c_str());
                        return false;
                    }
                    m_orderedPageStartAttrs[i] = attr;
                }

                m_orderedPageStartOffsets.resize(static_cast<size_t>(m_totalListCount) + 1, 0);
                for (int list = 0; list < m_totalListCount; ++list) {
                    std::int32_t count = 0;
                    if (!input.read(reinterpret_cast<char*>(&count), sizeof(count)) || count < 0 ||
                        count != static_cast<std::int32_t>(
                            static_cast<size_t>(m_orderedPageStartAttrs.size()) *
                            m_listInfos[static_cast<size_t>(list)].listPageCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Invalid ordered page-start count for list %d in %s\n",
                                     list, path.c_str());
                        return false;
                    }
                    m_orderedPageStartOffsets[static_cast<size_t>(list)] = m_orderedPageStartBits.size();
                    const size_t oldSize = m_orderedPageStartBits.size();
                    m_orderedPageStartBits.resize(oldSize + static_cast<size_t>(count));
                    if (count > 0 &&
                        !input.read(reinterpret_cast<char*>(m_orderedPageStartBits.data() + oldSize),
                                    static_cast<std::streamsize>(static_cast<size_t>(count) *
                                                                 sizeof(std::int32_t)))) {
                        return false;
                    }
                }
                m_orderedPageStartOffsets[static_cast<size_t>(m_totalListCount)] =
                    m_orderedPageStartBits.size();
                return true;
            }

            bool HasStaticFlatTagFilter(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return p_exWorkSpace != nullptr && p_exWorkSpace->m_queryTags != nullptr &&
                    p_exWorkSpace->m_numQueryTags > 0;
            }

            bool HasStaticDNFFilter(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return p_exWorkSpace != nullptr && p_exWorkSpace->m_dnf != nullptr &&
                    !p_exWorkSpace->m_dnf->Empty();
            }

            bool HasStaticMetadataFilter(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return HasStaticFlatTagFilter(p_exWorkSpace) || HasStaticDNFFilter(p_exWorkSpace);
            }

            bool RejectUnsupportedStaticFilter(const ExtraWorkSpace* p_exWorkSpace) const
            {
                if (p_exWorkSpace == nullptr) return false;
                if (p_exWorkSpace->m_filterFunc) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static snapshots do not support arbitrary metadata callbacks.\n");
                    return true;
                }
                if (HasStaticDNFFilter(p_exWorkSpace)) {
                    if (m_staticHasMetadata) return false;
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static DNF filtering requires an STM1 metadata snapshot.\n");
                    return true;
                }
                if (!HasStaticFlatTagFilter(p_exWorkSpace) || m_staticHasMetadata) return false;

                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Static raw/STT1 snapshots do not contain per-vector tags; "
                    "use an STM1 metadata snapshot for ACL filtering.\n");
                return true;
            }

            int StaticScanLimit(const ExtraWorkSpace* p_exWorkSpace, const ListInfo* p_listInfo) const
            {
                if (p_listInfo == nullptr) return 0;
                if (!HasStaticMetadataFilter(p_exWorkSpace)) {
                    return p_listInfo->listEleCount;
                }
                return (std::max)(0, (std::min)(p_listInfo->pureEleCount, p_listInfo->listEleCount));
            }

            int StaticReadPageCount(const ExtraWorkSpace* p_exWorkSpace,
                                    const ListInfo* p_listInfo) const
            {
                const int scanCount = StaticScanLimit(p_exWorkSpace, p_listInfo);
                if (scanCount <= 0) return 0;
                if (scanCount >= p_listInfo->listEleCount) return p_listInfo->listPageCount;
                const size_t bytes = static_cast<size_t>(p_listInfo->pageOffset) +
                    static_cast<size_t>(scanCount) * static_cast<size_t>(m_vectorInfoSize);
                return static_cast<int>((bytes + PageSize - 1) >> PageSizeEx);
            }

            bool TryOrderedPageStartQuery(const ExtraWorkSpace* p_exWorkSpace,
                                          size_t& p_attrIndex,
                                          std::int32_t& p_queryBit) const
            {
                if (m_opt == nullptr || !m_opt->m_enableOrderedPageStart ||
                    !HasStaticDNFFilter(p_exWorkSpace) || m_orderedPageStartAttrs.empty() ||
                    p_exWorkSpace->m_dnf->clauses.size() != 1) {
                    return false;
                }

                const auto& clause = p_exWorkSpace->m_dnf->clauses.front();
                if (clause.lits.size() < 2) return false;
                for (size_t reverse = m_orderedPageStartAttrs.size(); reverse > 0; --reverse) {
                    const size_t attrIndex = reverse - 1;
                    const int attr = m_orderedPageStartAttrs[attrIndex];
                    for (const auto& literal : clause.lits) {
                        if (literal.kind != 0 || literal.op != SPTAG::Cache::DNF_EQ ||
                            static_cast<int>(literal.col) != attr) {
                            continue;
                        }
                        const std::int32_t bit =
                            OrderedPageStartBit(literal.val, m_orderedPageStartBases[attrIndex]);
                        if (bit == kOrderedPageStartEmpty) return false;
                        p_attrIndex = attrIndex;
                        p_queryBit = bit;
                        return true;
                    }
                }
                return false;
            }

            ExtraWorkSpace::PostingReadRange BuildStaticPostingReadRange(
                const ExtraWorkSpace* p_exWorkSpace,
                SizeType p_postingId,
                const ListInfo* p_listInfo) const
            {
                ExtraWorkSpace::PostingReadRange range;
                if (p_listInfo == nullptr) return range;

                const int scanLimit = StaticScanLimit(p_exWorkSpace, p_listInfo);
                range.m_scanBegin = 0;
                range.m_scanEnd = scanLimit;
                range.m_readStartPage = 0;
                range.m_readPageCount = StaticReadPageCount(p_exWorkSpace, p_listInfo);

                size_t attrIndex = 0;
                std::int32_t queryBit = kOrderedPageStartEmpty;
                if (scanLimit <= 0 || p_postingId < 0 ||
                    !TryOrderedPageStartQuery(p_exWorkSpace, attrIndex, queryBit) ||
                    static_cast<size_t>(p_postingId + 1) >= m_orderedPageStartOffsets.size()) {
                    return range;
                }

                const int pageCount = p_listInfo->listPageCount;
                const size_t expectedCount =
                    m_orderedPageStartAttrs.size() * static_cast<size_t>(pageCount);
                const size_t base = m_orderedPageStartOffsets[static_cast<size_t>(p_postingId)];
                const size_t end = m_orderedPageStartOffsets[static_cast<size_t>(p_postingId + 1)];
                if (pageCount <= 0 || end < base || end - base != expectedCount) return range;

                const std::int32_t* starts =
                    m_orderedPageStartBits.data() + base + attrIndex * static_cast<size_t>(pageCount);
                const std::int32_t* upper = std::upper_bound(starts, starts + pageCount, queryBit);
                const std::int32_t* lower = std::lower_bound(starts, starts + pageCount, queryBit);
                const int lowerPage = static_cast<int>(lower - starts);
                const int upperPage = static_cast<int>(upper - starts);

                // Page starts bound the matching run, but fixed-size records can straddle
                // either boundary. Include one physical page on both sides and scan only
                // records fully present in the resulting buffer.
                const int readStart = (std::max)(0, lowerPage - 1);
                const int readEnd = (std::min)(pageCount, (std::max)(lowerPage + 1, upperPage + 1));
                if (readEnd <= readStart) return range;

                const std::int64_t recordBytes = m_vectorInfoSize;
                const std::int64_t firstBytes =
                    static_cast<std::int64_t>(readStart) * PageSize - p_listInfo->pageOffset;
                const std::int64_t endBytes =
                    static_cast<std::int64_t>(readEnd) * PageSize - p_listInfo->pageOffset;
                const int scanBegin = firstBytes <= 0 ? 0 :
                    static_cast<int>((firstBytes + recordBytes - 1) / recordBytes);
                const int scanEnd = endBytes <= 0 ? 0 :
                    static_cast<int>(endBytes / recordBytes);
                const int clampedBegin = (std::max)(0, (std::min)(scanBegin, scanLimit));
                const int clampedEnd = (std::max)(clampedBegin, (std::min)(scanEnd, scanLimit));
                if (clampedEnd == clampedBegin) return range;

                range.m_scanBegin = clampedBegin;
                range.m_scanEnd = clampedEnd;
                range.m_readStartPage = readStart;
                range.m_readPageCount = readEnd - readStart;
                return range;
            }

            int StaticScanBegin(const ExtraWorkSpace* p_exWorkSpace,
                                int p_slot,
                                const ListInfo* p_listInfo) const
            {
                if (p_exWorkSpace != nullptr && p_slot >= 0 &&
                    static_cast<size_t>(p_slot) < p_exWorkSpace->m_postingReadRanges.size()) {
                    return p_exWorkSpace->m_postingReadRanges[static_cast<size_t>(p_slot)].m_scanBegin;
                }
                return 0;
            }

            int StaticScanEnd(const ExtraWorkSpace* p_exWorkSpace,
                              int p_slot,
                              const ListInfo* p_listInfo) const
            {
                if (p_exWorkSpace != nullptr && p_slot >= 0 &&
                    static_cast<size_t>(p_slot) < p_exWorkSpace->m_postingReadRanges.size()) {
                    const int end = p_exWorkSpace->m_postingReadRanges[static_cast<size_t>(p_slot)].m_scanEnd;
                    if (end >= 0) return end;
                }
                return StaticScanLimit(p_exWorkSpace, p_listInfo);
            }

            bool StaticRecordMatchesFilter(const ExtraWorkSpace* p_exWorkSpace,
                                           const char* p_record) const
            {
                if (!HasStaticMetadataFilter(p_exWorkSpace)) return true;
                if (!m_staticHasMetadata || p_record == nullptr) return false;

                const char* tags = p_record + sizeof(int);
                if (HasStaticDNFFilter(p_exWorkSpace)) {
                    for (const auto& clause : p_exWorkSpace->m_dnf->clauses) {
                        if (clause.lits.empty()) continue;
                        bool all = true;
                        for (const auto& literal : clause.lits) {
                            if (literal.col >= static_cast<std::uint32_t>(m_staticNumTagsPerVec)) {
                                all = false;
                                break;
                            }
                            std::uint32_t tag = 0;
                            std::memcpy(
                                &tag,
                                tags + static_cast<size_t>(literal.col) * sizeof(tag),
                                sizeof(tag));
                            if (!SPTAG::Cache::DNFEvalOp(literal.op, tag, literal.val)) {
                                all = false;
                                break;
                            }
                        }
                        if (all) return true;
                    }
                    return false;
                }
                const int filterTagCols = m_staticACLTagCols > 0
                    ? m_staticACLTagCols
                    : m_staticNumTagsPerVec;
                for (int ti = 0; ti < filterTagCols; ++ti) {
                    std::uint32_t vectorTag = 0;
                    std::memcpy(&vectorTag, tags + static_cast<size_t>(ti) * sizeof(vectorTag),
                                sizeof(vectorTag));
                    for (int qi = 0; qi < p_exWorkSpace->m_numQueryTags; ++qi) {
                        if (vectorTag == p_exWorkSpace->m_queryTags[qi]) return true;
                    }
                }
                return false;
            }

            static constexpr std::uint32_t kStaticPipePQMagic = 0x31515053U; // "SPQ1"
            static constexpr int kStaticPipePQVersion = 1;
            static constexpr int kStaticPipePQHeaderInts = 8;
            static constexpr std::uint32_t kStaticTailMagic = 0x31545453U; // "STT1"
            static constexpr int kStaticTailVersion = 1;
            static constexpr int kStaticTailHeaderInts = 8;
            static constexpr std::uint32_t kStaticMetadataMagic = 0x314D5453U; // "STM1"
            static constexpr int kStaticMetadataVersion = 1;
            static constexpr int kStaticMetadataHeaderInts = 9;

            std::string ResolveStaticPath(const std::string& p_path, const Options& p_opt) const
            {
                if (p_path.empty() || p_path[0] == '/') return p_path;
                return p_opt.m_indexDirectory + FolderSep + p_path;
            }

            void CloseStaticPipePQCodes()
            {
#ifndef _MSC_VER
                if (m_staticPipePQCodeMap != nullptr) {
                    munmap(m_staticPipePQCodeMap, m_staticPipePQCodeMapBytes);
                    m_staticPipePQCodeMap = nullptr;
                }
                if (m_staticPipePQCodeFd >= 0) {
                    close(m_staticPipePQCodeFd);
                    m_staticPipePQCodeFd = -1;
                }
#endif
                m_staticPipePQCodes = nullptr;
                m_staticPipePQN = 0;
                m_staticPipePQCodeMapBytes = 0;
            }

            bool OpenStaticPipePQCodes(const std::string& p_path, int p_codeBytes, size_t p_expectedCount)
            {
#ifdef _MSC_VER
                (void)p_path;
                (void)p_codeBytes;
                (void)p_expectedCount;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Static PipePQ code mmap is currently supported on Linux only.\n");
                return false;
#else
                CloseStaticPipePQCodes();
                m_staticPipePQCodeFd = open(p_path.c_str(), O_RDONLY);
                if (m_staticPipePQCodeFd < 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Cannot open static PipePQ code sidecar: %s\n", p_path.c_str());
                    return false;
                }
                struct stat st {};
                if (fstat(m_staticPipePQCodeFd, &st) != 0 || st.st_size <= 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Cannot stat static PipePQ code sidecar: %s\n", p_path.c_str());
                    CloseStaticPipePQCodes();
                    return false;
                }
                m_staticPipePQCodeMapBytes = static_cast<size_t>(st.st_size);
                m_staticPipePQCodeMap = mmap(nullptr, m_staticPipePQCodeMapBytes, PROT_READ, MAP_SHARED,
                                              m_staticPipePQCodeFd, 0);
                if (m_staticPipePQCodeMap == MAP_FAILED) {
                    m_staticPipePQCodeMap = nullptr;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Cannot mmap static PipePQ code sidecar: %s\n", p_path.c_str());
                    CloseStaticPipePQCodes();
                    return false;
                }

                const size_t rawBytes = p_expectedCount * static_cast<size_t>(p_codeBytes);
                size_t dataOffset = 0;
                if (m_staticPipePQCodeMapBytes == rawBytes) {
                    dataOffset = 0;
                }
                else if (m_staticPipePQCodeMapBytes == rawBytes + 2 * sizeof(std::uint32_t)) {
                    std::uint32_t header[2] = {0, 0};
                    std::memcpy(header, m_staticPipePQCodeMap, sizeof(header));
                    if (header[0] != p_expectedCount || header[1] != static_cast<std::uint32_t>(p_codeBytes)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Static PipePQ sidecar header mismatch: N=%u M=%u, expected N=%zu M=%d\n",
                                     header[0], header[1], p_expectedCount, p_codeBytes);
                        CloseStaticPipePQCodes();
                        return false;
                    }
                    dataOffset = sizeof(header);
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ sidecar size mismatch: %zu, expected %zu or %zu bytes\n",
                                 m_staticPipePQCodeMapBytes, rawBytes, rawBytes + 2 * sizeof(std::uint32_t));
                    CloseStaticPipePQCodes();
                    return false;
                }

                m_staticPipePQCodes = static_cast<const std::uint8_t*>(m_staticPipePQCodeMap) + dataOffset;
                m_staticPipePQN = p_expectedCount;
                return true;
#endif
            }

            bool ConfigureStaticPipePQ(Options& p_opt, size_t p_expectedCount, bool p_needCodes)
            {
                m_staticPipePQ = false;
                m_staticPipePQCodeBytes = 0;
                m_staticPipePQDimension = p_opt.m_dim;
                const bool noQuantizer =
                    p_opt.m_postingQuantizer.empty() ||
                    Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "None");
                if (noQuantizer) return true;

                if (!Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "PipePQ") &&
                    !Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "PQ")) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Storage=STATIC currently supports PipePQ only; got PostingQuantizer=%s\n",
                                 p_opt.m_postingQuantizer.c_str());
                    return false;
                }
                if (p_opt.m_postingQuantM <= 0 || p_opt.m_pipePQPivotsFile.empty() ||
                    p_opt.m_fullVectorFile.empty() || p_opt.m_quantADCOnly || p_opt.m_rerankL <= 0 ||
                    p_opt.m_enableDeltaEncoding || p_opt.m_enablePostingListRearrange ||
                    p_opt.m_enableDataCompression) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ requires M, pivots, FullVectorFile, positive RerankL, "
                                 "and disables delta/rearrange/compression.\n");
                    return false;
                }
                if (p_opt.m_tailReplicaCount > 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ does not yet support unfilter tails.\n");
                    return false;
                }
                if (!p_opt.m_quantizerFilePath.empty()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ does not support a global quantizer because exact rerank "
                                 "requires the original vector scalar type.\n");
                    return false;
                }

                const std::string pivots = ResolveStaticPath(p_opt.m_pipePQPivotsFile, p_opt);
                m_staticPipePQTable.reset(new PipePQTable());
                if (!m_staticPipePQTable->Load(pivots, p_opt.m_postingQuantM) ||
                    m_staticPipePQTable->Dim() != p_opt.m_dim) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Cannot load static PipePQ pivots: %s\n", pivots.c_str());
                    return false;
                }

                m_staticPipePQ = true;
                m_staticPipePQCodeBytes = p_opt.m_postingQuantM;
                if (p_needCodes) {
                    const std::string codes = ResolveStaticPath(p_opt.m_postingQuantFile, p_opt);
                    if (codes.empty() || !OpenStaticPipePQCodes(codes, m_staticPipePQCodeBytes, p_expectedCount)) {
                        return false;
                    }
                }
                return true;
            }

            bool ConfigureStaticMetadataForBuild(const Options& p_opt, size_t p_expectedCount)
            {
                m_staticHasMetadata = false;
                m_staticNumTagsPerVec = 0;
                m_staticACLTagCols = 0;
                m_staticMetadataBytes = sizeof(int);
                if (p_opt.m_numTagsPerVec <= 0) return true;

                const size_t expectedTagCount =
                    p_expectedCount * static_cast<size_t>(p_opt.m_numTagsPerVec);
                if (m_staticBuildNumTagsPerVec != p_opt.m_numTagsPerVec ||
                    m_staticBuildTags.size() != expectedTagCount) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static metadata build requires %zu tags across %d columns; got %zu across %d.\n",
                        expectedTagCount,
                        p_opt.m_numTagsPerVec,
                        m_staticBuildTags.size(),
                        m_staticBuildNumTagsPerVec);
                    return false;
                }
                if (m_staticPipePQ || p_opt.m_enableDeltaEncoding ||
                    p_opt.m_enablePostingListRearrange || p_opt.m_enableDataCompression) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static metadata currently requires raw full-float postings without "
                        "PipePQ, delta encoding, rearrangement, or compression.\n");
                    return false;
                }

                m_staticHasMetadata = true;
                m_staticNumTagsPerVec = p_opt.m_numTagsPerVec;
                m_staticACLTagCols = p_opt.m_staticACLTagCols > 0
                    ? p_opt.m_staticACLTagCols
                    : m_staticNumTagsPerVec;
                if (m_staticACLTagCols > m_staticNumTagsPerVec) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "StaticACLTagCols=%d exceeds NumTagsPerVec=%d.\n",
                                 m_staticACLTagCols, m_staticNumTagsPerVec);
                    return false;
                }
                m_staticMetadataBytes = static_cast<int>(
                    sizeof(int) + static_cast<size_t>(m_staticNumTagsPerVec) * sizeof(uint32_t));
                return true;
            }

            bool OpenStaticRerankFile(Options& p_opt, size_t p_expectedCount)
            {
#ifdef _MSC_VER
                (void)p_opt;
                (void)p_expectedCount;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Static PipePQ exact rerank is currently supported on Linux only.\n");
                return false;
#else
                const std::string fullPath = ResolveStaticPath(p_opt.m_fullVectorFile, p_opt);
                auto header = f_createIO();
                int32_t dimensions[2] = {0, 0};
                if (header == nullptr ||
                    !header->Initialize(fullPath.c_str(), std::ios::binary | std::ios::in) ||
                    header->ReadBinary(sizeof(dimensions), reinterpret_cast<char*>(dimensions), 0) != sizeof(dimensions) ||
                    dimensions[0] <= 0 || dimensions[1] != p_opt.m_dim ||
                    (p_expectedCount > 0 && static_cast<size_t>(dimensions[0]) != p_expectedCount)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ FullVectorFile mismatch: %s\n", fullPath.c_str());
                    return false;
                }
                const std::uint64_t recordBytes = static_cast<std::uint64_t>(p_opt.m_dim) * sizeof(ValueType);
                const std::uint64_t vectorCount = static_cast<std::uint64_t>(dimensions[0]);
                if (recordBytes == 0 ||
                    vectorCount > (std::numeric_limits<std::uint64_t>::max() - sizeof(dimensions)) / recordBytes) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ FullVectorFile size overflows: %s\n", fullPath.c_str());
                    return false;
                }
                struct stat fileStatus {};
                const std::uint64_t requiredBytes = sizeof(dimensions) + vectorCount * recordBytes;
                if (stat(fullPath.c_str(), &fileStatus) != 0 || fileStatus.st_size < 0 ||
                    static_cast<std::uint64_t>(fileStatus.st_size) < requiredBytes) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ FullVectorFile is truncated: %s\n", fullPath.c_str());
                    return false;
                }

                m_staticRerankFile = f_createAsyncIO();
                const int rerankContexts = max(
                    max(1, p_opt.m_ioThreads),
                    max(p_opt.m_searchThreadNum, p_opt.m_iSSDNumberOfThreads));
                if (m_staticRerankFile == nullptr ||
                    !m_staticRerankFile->Initialize(fullPath.c_str(), O_RDONLY | O_DIRECT,
                                                     std::max(64, p_opt.m_rerankL), 2, 2,
                                                     static_cast<std::uint16_t>(rerankContexts))) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Cannot open static PipePQ FullVectorFile: %s\n", fullPath.c_str());
                    return false;
                }
                m_staticRerankHandlers.clear();
                m_staticRerankHandlers.emplace_back(m_staticRerankFile);
                m_staticRerankCount = static_cast<size_t>(dimensions[0]);
                return true;
#endif
            }

            bool RerankStaticPipePQ(const std::vector<int>& p_vids, const ValueType* p_query,
                                    std::shared_ptr<VectorIndex> p_index,
                                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                                    ExtraWorkSpace::PostingProbeStats* p_probeStats,
                                    int p_ioContext)
            {
#ifdef _MSC_VER
                (void)p_vids;
                (void)p_query;
                (void)p_index;
                (void)p_queryResults;
                (void)p_probeStats;
                return false;
#else
                if (p_vids.empty() || m_staticRerankFile == nullptr || m_staticRerankHandlers.empty()) {
                    return p_vids.empty();
                }

                const size_t recordBytes = static_cast<size_t>(m_iDataDimension) * sizeof(ValueType);
                std::vector<std::uint64_t> pages;
                pages.reserve(p_vids.size() * 2);
                for (int vid : p_vids) {
                    if (vid < 0 || static_cast<size_t>(vid) >= m_staticRerankCount) continue;
                    const std::uint64_t begin = sizeof(std::int32_t) * 2 +
                                                static_cast<std::uint64_t>(vid) * recordBytes;
                    const std::uint64_t end = begin + recordBytes;
                    const std::uint64_t firstPage = begin >> PageSizeEx;
                    const std::uint64_t lastPage = (end - 1) >> PageSizeEx;
                    for (std::uint64_t page = firstPage; page <= lastPage; ++page) {
                        pages.push_back(page);
                    }
                }

                std::sort(pages.begin(), pages.end());
                pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
                struct PageRange
                {
                    std::uint64_t firstPage;
                    size_t pageCount;
                };
                std::vector<PageRange> ranges;
                ranges.reserve(pages.size());
                std::unordered_map<std::uint64_t, size_t> pageRanges;
                for (size_t begin = 0; begin < pages.size();) {
                    size_t end = begin + 1;
                    while (end < pages.size() && pages[end] == pages[end - 1] + 1) ++end;
                    const size_t rangeIndex = ranges.size();
                    ranges.push_back({ pages[begin], end - begin });
                    for (size_t i = begin; i < end; ++i) pageRanges.emplace(pages[i], rangeIndex);
                    begin = end;
                }

                std::vector<Helper::PageBuffer<std::uint8_t>> pageBuffers(ranges.size());
                std::vector<Helper::AsyncReadRequest> requests(ranges.size());
                for (size_t i = 0; i < ranges.size(); ++i) {
                    const size_t readBytes = ranges[i].pageCount * static_cast<size_t>(PageSize);
                    pageBuffers[i].ReservePageBuffer(readBytes);
                    pageBuffers[i].SetAvailableSize(readBytes);
                    auto& request = requests[i];
                    request.m_buffer = reinterpret_cast<char*>(pageBuffers[i].GetBuffer());
                    request.m_offset = ranges[i].firstPage << PageSizeEx;
                    request.m_readSize = readBytes;
                    request.m_status = p_ioContext;
                    request.m_callback = [](bool) {};
                }
                if (!Helper::BatchReadFileAsync(m_staticRerankHandlers, requests.data(),
                                                static_cast<int>(requests.size()))) {
                    return false;
                }

                std::vector<ValueType> rerankVector(static_cast<size_t>(m_iDataDimension));
                for (int vid : p_vids) {
                    if (vid < 0 || static_cast<size_t>(vid) >= m_staticRerankCount) continue;
                    std::uint8_t* destination = reinterpret_cast<std::uint8_t*>(rerankVector.data());
                    size_t remaining = recordBytes;
                    std::uint64_t cursor = sizeof(std::int32_t) * 2 +
                                           static_cast<std::uint64_t>(vid) * recordBytes;
                    while (remaining > 0) {
                        const std::uint64_t page = cursor >> PageSizeEx;
                        const size_t pageOffset = static_cast<size_t>(cursor & (PageSize - 1));
                        const size_t copyBytes = (std::min)(remaining, static_cast<size_t>(PageSize) - pageOffset);
                        auto found = pageRanges.find(page);
                        if (found == pageRanges.end()) return false;
                        const auto& range = ranges[found->second];
                        const size_t rangeOffset =
                            static_cast<size_t>(page - range.firstPage) * PageSize + pageOffset;
                        std::memcpy(destination, pageBuffers[found->second].GetBuffer() + rangeOffset, copyBytes);
                        destination += copyBytes;
                        cursor += copyBytes;
                        remaining -= copyBytes;
                    }
                    p_queryResults.AddPoint(vid, p_index->ComputeDistance(p_query, rerankVector.data()));
                }

                if (p_probeStats != nullptr) {
                    p_probeStats->m_rerankCandidates += p_vids.size();
                    p_probeStats->m_rerankReadRequests += ranges.size();
                    p_probeStats->m_rerankPhysicalBytes += pages.size() * PageSize;
                }
                return true;
#endif
            }

            ErrorCode SearchIndexPipePQ(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_queryResults,
                                        std::shared_ptr<VectorIndex> p_index, SearchStats* p_stats,
                                        std::set<int>* truth, std::map<int, std::set<int>>* found)
            {
                (void)truth;
                (void)found;
                COMMON::QueryResultSet<ValueType>& queryResults =
                    *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
                if (m_staticPipePQTable == nullptr || m_staticRerankFile == nullptr ||
                    m_opt->m_rerankL < m_opt->m_resultNum) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Static PipePQ search is not initialized or RerankL is smaller than result count.\n");
                    return ErrorCode::Fail;
                }
                p_exWorkSpace->m_postingProbeStats.Reset();

                std::vector<float> query(static_cast<size_t>(m_iDataDimension));
                const ValueType* rawQuery = queryResults.GetTarget();
                for (int d = 0; d < m_iDataDimension; ++d) query[d] = static_cast<float>(rawQuery[d]);
                if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine) {
                    COMMON::Utils::Normalize<float>(query.data(), m_iDataDimension, COMMON::Utils::GetBase<float>());
                }
                std::vector<float> lut(static_cast<size_t>(m_staticPipePQCodeBytes) * 256);
                m_staticPipePQTable->PopulateDistances(query.data(), lut.data(), m_opt->m_distCalcMethod);

                std::priority_queue<std::pair<float, int>> survivors;
                int listElements = 0;
                int diskPages = 0;
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
                auto scanPosting = [&](ListInfo* listInfo, char* buffer) {
                    const std::uint8_t* records = reinterpret_cast<const std::uint8_t*>(buffer + listInfo->pageOffset);
                    for (int i = 0; i < listInfo->listEleCount; ++i) {
                        const std::uint8_t* record = records + static_cast<size_t>(i) * m_vectorInfoSize;
                        int vid = -1;
                        std::memcpy(&vid, record, sizeof(vid));
                        if (vid < 0 || p_exWorkSpace->m_deduper.CheckAndSet(vid)) {
                            --listElements;
                            continue;
                        }
                        const std::uint8_t* code = record + sizeof(int);
                        float distance = 0.0f;
                        for (int m = 0; m < m_staticPipePQCodeBytes; ++m) {
                            distance += lut[static_cast<size_t>(m) * 256 + code[m]];
                        }
                        if (static_cast<int>(survivors.size()) < m_opt->m_rerankL) {
                            survivors.emplace(distance, vid);
                        }
                        else if (distance < survivors.top().first) {
                            survivors.pop();
                            survivors.emplace(distance, vid);
                        }
                    }
                };

                for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                    const SizeType postingId = p_exWorkSpace->m_postingIDs[pi];
                    if (postingId < 0 || postingId >= m_totalListCount) continue;
                    ListInfo* listInfo = &m_listInfos[postingId];
                    const int fileid = m_oneContext ? 0 : postingId / m_listPerFile;
                    diskPages += listInfo->listPageCount;
                    listElements += listInfo->listEleCount;
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = static_cast<size_t>(listInfo->listPageCount) << PageSizeEx;
                    request.m_status = (fileid << 16) | (request.m_status & 0xffff);
                    request.m_payload = listInfo;
                    request.m_success = false;
                    request.m_callback = [&scanPosting, &request, listInfo](bool success) {
                        if (success) scanPosting(listInfo, request.m_buffer);
                    };
                }
                if (!Helper::BatchReadFileAsync(m_indexFiles, p_exWorkSpace->m_diskRequests.data(),
                                                static_cast<int>(postingListCount))) {
                    return ErrorCode::DiskIOFail;
                }

                std::vector<int> rerankVIDs;
                rerankVIDs.reserve(survivors.size());
                while (!survivors.empty()) {
                    rerankVIDs.push_back(survivors.top().second);
                    survivors.pop();
                }
                const int rerankIOContext = p_exWorkSpace->m_diskRequests.empty()
                    ? 0
                    : (p_exWorkSpace->m_diskRequests.front().m_status & 0xffff);
                if (!RerankStaticPipePQ(rerankVIDs, rawQuery, p_index, queryResults,
                                        &p_exWorkSpace->m_postingProbeStats, rerankIOContext)) {
                    return ErrorCode::DiskIOFail;
                }

                p_exWorkSpace->m_postingProbeStats.m_readPostings += postingListCount;
                p_exWorkSpace->m_postingProbeStats.m_scannedVectors +=
                    static_cast<std::uint64_t>((std::max)(listElements, 0));
                p_exWorkSpace->m_postingProbeStats.m_adcScannedVectors +=
                    static_cast<std::uint64_t>((std::max)(listElements, 0));
                p_exWorkSpace->m_postingProbeStats.m_adcSurvivors += rerankVIDs.size();
                p_exWorkSpace->m_postingProbeStats.m_postingPageReads += diskPages;
                p_exWorkSpace->m_postingProbeStats.m_postingPhysicalBytes +=
                    static_cast<std::uint64_t>(diskPages) * PageSize;
                if (p_stats != nullptr) {
                    const auto& probeStats = p_exWorkSpace->m_postingProbeStats;
                    const std::uint64_t rerankPages =
                        probeStats.m_rerankPhysicalBytes / static_cast<std::uint64_t>(PageSize);
                    p_stats->m_totalListElementsCount = listElements;
                    p_stats->m_diskIOCount = static_cast<int>(
                        postingListCount + probeStats.m_rerankReadRequests);
                    p_stats->m_diskAccessCount = static_cast<int>(diskPages + rerankPages);
                }
                queryResults.SetScanned(listElements);
                return ErrorCode::Success;
            }

            int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, std::vector<ListInfo>& p_listInfos)
            {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                    throw std::runtime_error("Failed open file in LoadingHeadInfo");
                }
                m_pCompressor = std::make_unique<Compressor>(); // no need compress level to decompress

                auto readInt = [&ptr](int& value) {
                    return ptr->ReadBinary(sizeof(value), reinterpret_cast<char*>(&value)) == sizeof(value);
                };
                int firstHeaderValue = 0;
                int m_listCount = 0;
                int m_totalDocumentCount = 0;
                int m_listPageOffset = 0;
                if (!readInt(firstHeaderValue)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }

                const bool quantizedHeader =
                    static_cast<std::uint32_t>(firstHeaderValue) == kStaticPipePQMagic;
                const bool tailHeader =
                    static_cast<std::uint32_t>(firstHeaderValue) == kStaticTailMagic;
                const bool metadataHeader =
                    static_cast<std::uint32_t>(firstHeaderValue) == kStaticMetadataMagic;
                if (metadataHeader) {
                    int version = 0;
                    int recordBytes = 0;
                    int numTagsPerVec = 0;
                    int tailPageBudget = 0;
                    if (!readInt(version) || !readInt(m_listCount) || !readInt(m_totalDocumentCount) ||
                        !readInt(m_iDataDimension) || !readInt(recordBytes) ||
                        !readInt(numTagsPerVec) || !readInt(tailPageBudget) ||
                        !readInt(m_listPageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Failed to read static metadata header!\n");
                        throw std::runtime_error("Failed read static metadata header");
                    }
                    const int metadataBytes =
                        static_cast<int>(sizeof(int) +
                                         static_cast<size_t>(numTagsPerVec) * sizeof(uint32_t));
                    const int expectedRecordBytes =
                        m_iDataDimension * sizeof(ValueType) + metadataBytes;
                    if (version != kStaticMetadataVersion || m_staticPipePQ || numTagsPerVec <= 0 ||
                        recordBytes != expectedRecordBytes || tailPageBudget < -1) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Static metadata header mismatch: version=%d record=%d dim=%d tags=%d tailPages=%d\n",
                            version, recordBytes, m_iDataDimension, numTagsPerVec, tailPageBudget);
                        throw std::runtime_error("Static metadata header mismatch");
                    }
                    m_vectorInfoSize = recordBytes;
                    m_staticHasMetadata = true;
                    m_staticNumTagsPerVec = numTagsPerVec;
                    m_staticACLTagCols =
                        m_opt != nullptr && m_opt->m_staticACLTagCols > 0
                        ? m_opt->m_staticACLTagCols
                        : m_staticNumTagsPerVec;
                    if (m_staticACLTagCols > m_staticNumTagsPerVec) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "StaticACLTagCols=%d exceeds STM1 tag count=%d.\n",
                                     m_staticACLTagCols, m_staticNumTagsPerVec);
                        throw std::runtime_error("Static ACL tag-column count mismatch");
                    }
                    m_staticMetadataBytes = metadataBytes;
                    m_staticHasUnfilterTail = false;
                    m_staticTailPageBudget = tailPageBudget;
                }
                else if (tailHeader) {
                    m_staticHasMetadata = false;
                    m_staticNumTagsPerVec = 0;
                    m_staticACLTagCols = 0;
                    m_staticMetadataBytes = sizeof(int);
                    int version = 0;
                    int recordBytes = 0;
                    int tailPageBudget = 0;
                    if (!readInt(version) || !readInt(m_listCount) || !readInt(m_totalDocumentCount) ||
                        !readInt(m_iDataDimension) || !readInt(recordBytes) || !readInt(tailPageBudget) ||
                        !readInt(m_listPageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Failed to read static tail header!\n");
                        throw std::runtime_error("Failed read static tail header");
                    }
                    const int rawRecordBytes = m_iDataDimension * sizeof(ValueType) + sizeof(int);
                    if (version != kStaticTailVersion || m_staticPipePQ || recordBytes != rawRecordBytes ||
                        tailPageBudget < -1) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Static tail header mismatch: version=%d record=%d dim=%d tailPages=%d\n",
                                     version, recordBytes, m_iDataDimension, tailPageBudget);
                        throw std::runtime_error("Static tail header mismatch");
                    }
                    m_vectorInfoSize = recordBytes;
                    m_staticHasUnfilterTail = false;
                    m_staticTailPageBudget = tailPageBudget;
                }
                else if (quantizedHeader) {
                    m_staticHasUnfilterTail = false;
                    m_staticTailPageBudget = 0;
                    m_staticHasMetadata = false;
                    m_staticNumTagsPerVec = 0;
                    m_staticACLTagCols = 0;
                    m_staticMetadataBytes = sizeof(int);
                    int version = 0;
                    int recordBytes = 0;
                    int codeBytes = 0;
                    if (!readInt(version) || !readInt(m_listCount) || !readInt(m_totalDocumentCount) ||
                        !readInt(m_iDataDimension) || !readInt(recordBytes) || !readInt(codeBytes) ||
                        !readInt(m_listPageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read static PipePQ header!\n");
                        throw std::runtime_error("Failed read static PipePQ header");
                    }
                    if (version != kStaticPipePQVersion || !m_staticPipePQ ||
                        codeBytes != m_staticPipePQCodeBytes ||
                        recordBytes != static_cast<int>(sizeof(int) + codeBytes) ||
                        m_iDataDimension != m_staticPipePQDimension) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Static PipePQ header mismatch: version=%d code=%d record=%d dim=%d\n",
                                     version, codeBytes, recordBytes, m_iDataDimension);
                        throw std::runtime_error("Static PipePQ header mismatch");
                    }
                    m_vectorInfoSize = recordBytes;
                }
                else {
                    m_staticHasUnfilterTail = false;
                    m_staticTailPageBudget = 0;
                    m_staticHasMetadata = false;
                    m_staticNumTagsPerVec = 0;
                    m_staticACLTagCols = 0;
                    m_staticMetadataBytes = sizeof(int);
                    if (m_staticPipePQ) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Static PipePQ was requested but %s has a legacy raw-vector header.\n",
                                     p_file.c_str());
                        throw std::runtime_error("Static PipePQ header missing");
                    }
                    m_listCount = firstHeaderValue;
                    if (!readInt(m_totalDocumentCount) || !readInt(m_iDataDimension) ||
                        !readInt(m_listPageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (m_vectorInfoSize == 0) {
                        m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
                    }
                    else if (m_vectorInfoSize != m_iDataDimension * sizeof(ValueType) + sizeof(int)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Failed to read head info file! DataDimension and ValueType are not match!\n");
                        throw std::runtime_error("DataDimension and ValueType don't match in LoadingHeadInfo");
                    }
                }

                size_t totalListCount = p_listInfos.size();
                p_listInfos.resize(totalListCount + m_listCount);

                size_t totalListElementCount = 0;

                std::map<int, int> pageCountDist;

                size_t biglistCount = 0;
                size_t biglistElementCount = 0;
                int pageNum;
                for (int i = 0; i < m_listCount; ++i)
                {
                    ListInfo* listInfo = &(p_listInfos[totalListCount + i]);

                    if (m_enableDataCompression)
                    {
                        if (ptr->ReadBinary(sizeof(listInfo->listTotalBytes), reinterpret_cast<char*>(&(listInfo->listTotalBytes))) != sizeof(listInfo->listTotalBytes)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                            throw std::runtime_error("Failed read file in LoadingHeadInfo");
                        }
                    }
                    if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&(pageNum))) != sizeof(pageNum)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->pageOffset), reinterpret_cast<char*>(&(listInfo->pageOffset))) != sizeof(listInfo->pageOffset)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->listEleCount), reinterpret_cast<char*>(&(listInfo->listEleCount))) != sizeof(listInfo->listEleCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->listPageCount), reinterpret_cast<char*>(&(listInfo->listPageCount))) != sizeof(listInfo->listPageCount)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    m_staticMaxListPageCount = (std::max)(
                        m_staticMaxListPageCount, static_cast<int>(listInfo->listPageCount));
                    listInfo->pureEleCount = listInfo->listEleCount;
                    if (tailHeader || metadataHeader) {
                        if (ptr->ReadBinary(sizeof(listInfo->pureEleCount),
                                            reinterpret_cast<char*>(&(listInfo->pureEleCount))) !=
                            sizeof(listInfo->pureEleCount)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Failed to read static tail pure count!\n");
                            throw std::runtime_error("Failed read static tail pure count");
                        }
                        if (listInfo->pureEleCount < 0 || listInfo->pureEleCount > listInfo->listEleCount) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Invalid static tail pure count: pure=%d total=%d\n",
                                         listInfo->pureEleCount, listInfo->listEleCount);
                            throw std::runtime_error("Invalid static tail pure count");
                        }
                        if (listInfo->pureEleCount < listInfo->listEleCount) {
                            m_staticHasUnfilterTail = true;
                        }
                        if (m_staticTailPageBudget > 0 && p_postingPageLimit > 0 &&
                            listInfo->listPageCount > p_postingPageLimit) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Static tail posting needs %u pages but SearchPostingPageLimit is %d; "
                                "refusing to truncate the tail.\n",
                                listInfo->listPageCount, p_postingPageLimit);
                            throw std::runtime_error("Static tail search page limit too small");
                        }
                    }
                    listInfo->listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
                    if (!m_enableDataCompression)
                    {
                        listInfo->listTotalBytes = listInfo->listEleCount * m_vectorInfoSize;
                        if (m_staticTailPageBudget >= 0) {
                            listInfo->listEleCount = min(
                                listInfo->listEleCount,
                                (min(static_cast<int>(listInfo->listPageCount), p_postingPageLimit) <<
                                 PageSizeEx) /
                                    m_vectorInfoSize);
                            if (m_staticHasUnfilterTail &&
                                listInfo->listEleCount < listInfo->pureEleCount) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                             "Static tail load truncated pure prefix: pure=%d loaded=%d\n",
                                             listInfo->pureEleCount, listInfo->listEleCount);
                                throw std::runtime_error("Static tail pure prefix truncated");
                            }
                            listInfo->listPageCount = static_cast<std::uint16_t>(
                                ceil((m_vectorInfoSize * listInfo->listEleCount + listInfo->pageOffset) *
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
                    size_t dictBufferSize;
                    if (ptr->ReadBinary(sizeof(size_t), reinterpret_cast<char*>(&dictBufferSize)) != sizeof(dictBufferSize)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    char* dictBuffer = new char[dictBufferSize];
                    if (ptr->ReadBinary(dictBufferSize, dictBuffer) != dictBufferSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    try {
                        m_pCompressor->SetDictBuffer(std::string(dictBuffer, dictBufferSize));
                    }
                    catch (std::runtime_error& err) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head info file: %s \n", err.what());
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    delete[] dictBuffer;
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "Finish reading header info, list count %d, total doc count %d, dimension %d, list page offset %d.\n",
                    m_listCount,
                    m_totalDocumentCount,
                    m_iDataDimension,
                    m_listPageOffset);

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
                offsetVectorID = (m_vectorInfoSize - sizeof(int)) * eleCount + sizeof(int) * i;
                offsetVector = (m_vectorInfoSize - sizeof(int)) * i;
            }

            inline void ParsePostingList(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                offsetVectorID = m_vectorInfoSize * i;
                offsetVector = offsetVectorID + m_staticMetadataBytes;
            }

            inline void ParseDeltaEncoding(std::shared_ptr<VectorIndex>& p_index, ListInfo* p_info, ValueType* vector)
            {
                ValueType* headVector = (ValueType*)p_index->GetSample((SizeType)(p_info - m_listInfos.data()));
                COMMON::SIMDUtils::ComputeSum(vector, headVector, m_iDataDimension);
            }

            inline void ParseEncoding(std::shared_ptr<VectorIndex>& p_index, ListInfo* p_info, ValueType* vector) { }

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
                std::shared_ptr<VectorSet> p_fullVectors,
                size_t p_postingListOffset,
                const std::vector<int>* p_pureCounts,
                int p_tailPageBudget,
                const std::vector<int>& p_orderedPageStartAttrs,
                const std::vector<std::uint32_t>& p_orderedPageStartBases)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                auto t1 = std::chrono::high_resolution_clock::now();
                const size_t sidecarSlash = p_outputFile.find_last_of(FolderSep);
                const std::string sidecarPath =
                    (sidecarSlash == std::string::npos ? std::string() :
                                                        p_outputFile.substr(0, sidecarSlash + 1)) +
                    "ordered_page_starts.bin";
                if (p_orderedPageStartAttrs.empty() && fileexists(sidecarPath.c_str())) {
                    if (std::remove(sidecarPath.c_str()) != 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Failed to remove stale ordered page-start directory: %s\n",
                                     sidecarPath.c_str());
                        throw std::runtime_error("Failed to remove stale ordered page-start directory");
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Removed stale ordered page-start directory: %s\n",
                                 sidecarPath.c_str());
                }

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
                const bool staticTailHeader = p_pureCounts != nullptr;
                const bool staticMetadataHeader = m_staticHasMetadata;
                if (staticTailHeader &&
                    (m_staticPipePQ || p_pureCounts->size() != p_postingListSizes.size() ||
                     p_tailPageBudget < -1)) {
                    throw std::runtime_error("Invalid static tail output configuration");
                }
                if (staticMetadataHeader && !staticTailHeader) {
                    throw std::runtime_error("Static metadata output requires pure-count metadata");
                }
                // The legacy format begins with four integers. Static PipePQ and
                // static tail/metadata snapshots use versioned headers so record
                // metadata is never inferred from the legacy vector-only layout.
                std::uint64_t listOffset =
                    staticMetadataHeader ? sizeof(int) * kStaticMetadataHeaderInts
                                        : (staticTailHeader ? sizeof(int) * kStaticTailHeaderInts
                                                            : (m_staticPipePQ
                                                                   ? sizeof(int) * kStaticPipePQHeaderInts
                                                                   : sizeof(int) * 4));
                // meta size of the posting lists
                listOffset += (sizeof(int) + sizeof(std::uint16_t) + sizeof(int) + sizeof(std::uint16_t)) * p_postingListSizes.size();
                if (staticTailHeader) {
                    listOffset += sizeof(int) * p_postingListSizes.size();
                }
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

                auto writeHeaderInt = [&ptr](int value) {
                    if (ptr->WriteBinary(sizeof(value), reinterpret_cast<char*>(&value)) != sizeof(value)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex header");
                    }
                };
                if (staticMetadataHeader) {
                    writeHeaderInt(static_cast<int>(kStaticMetadataMagic));
                    writeHeaderInt(kStaticMetadataVersion);
                    writeHeaderInt(static_cast<int>(p_postingListSizes.size()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Count()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Dimension()));
                    writeHeaderInt(static_cast<int>(p_spacePerVector));
                    writeHeaderInt(m_staticNumTagsPerVec);
                    writeHeaderInt(p_tailPageBudget);
                    writeHeaderInt(static_cast<int>(listOffset / PageSize));
                }
                else if (staticTailHeader) {
                    writeHeaderInt(static_cast<int>(kStaticTailMagic));
                    writeHeaderInt(kStaticTailVersion);
                    writeHeaderInt(static_cast<int>(p_postingListSizes.size()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Count()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Dimension()));
                    writeHeaderInt(static_cast<int>(p_spacePerVector));
                    writeHeaderInt(p_tailPageBudget);
                    writeHeaderInt(static_cast<int>(listOffset / PageSize));
                }
                else if (m_staticPipePQ) {
                    writeHeaderInt(static_cast<int>(kStaticPipePQMagic));
                    writeHeaderInt(kStaticPipePQVersion);
                    writeHeaderInt(static_cast<int>(p_postingListSizes.size()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Count()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Dimension()));
                    writeHeaderInt(static_cast<int>(p_spacePerVector));
                    writeHeaderInt(m_staticPipePQCodeBytes);
                    writeHeaderInt(static_cast<int>(listOffset / PageSize));
                }
                else {
                    writeHeaderInt(static_cast<int>(p_postingListSizes.size()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Count()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Dimension()));
                    writeHeaderInt(static_cast<int>(listOffset / PageSize));
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
                    if (staticTailHeader) {
                        const int pureCount = p_pureCounts->at(i);
                        if (pureCount < 0 || pureCount > listEleCount ||
                            ptr->WriteBinary(sizeof(pureCount), reinterpret_cast<const char*>(&pureCount)) !=
                                sizeof(pureCount)) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Failed to write static tail pure count!\n");
                            throw std::runtime_error("Failed to write static tail pure count");
                        }
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
                std::vector<std::vector<std::int32_t>> orderedPageStarts;
                if (!p_orderedPageStartAttrs.empty()) {
                    orderedPageStarts.resize(p_postingListSizes.size());
                }
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
                    std::string postingListFullData = GetPostingListFullData(
                        postingListId,
                        p_postingListSizes[id],
                        p_postingSelections,
                        p_fullVectors,
                        p_enableDeltaEncoding,
                        p_enablePostingListRearrange,
                        headVector,
                        p_orderedPageStartAttrs.empty() ? nullptr : &p_orderedPageStartAttrs,
                        p_pureCounts != nullptr ? p_pureCounts->at(id) : p_postingListSizes[id]);
                    size_t postingListFullSize = p_postingListSizes[id] * p_spacePerVector;
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

                    if (!p_orderedPageStartAttrs.empty()) {
                        const std::uint16_t listPageCount = static_cast<std::uint16_t>(
                            (postingListFullSize + PageSize - 1) / PageSize);
                        if (!BuildOrderedPageStartsForPosting(
                            postingListFullData,
                            p_pureCounts != nullptr ? p_pureCounts->at(id) : p_postingListSizes[id],
                            p_postPageOffset[id],
                            listPageCount,
                            p_orderedPageStartAttrs,
                            p_orderedPageStartBases,
                            orderedPageStarts[static_cast<size_t>(id)])) {
                            throw std::runtime_error(
                                "OrderedPageStartAttrs are not monotonic after ACL tuple sorting");
                        }
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

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Padded Size: %llu, final total size: %llu.\n", paddedSize, listOffset);

                if (!p_orderedPageStartAttrs.empty()) {
                    if (!SaveOrderedPageStarts(
                            sidecarPath,
                            orderedPageStarts,
                            p_orderedPageStartAttrs,
                            p_orderedPageStartBases)) {
                        throw std::runtime_error("Failed to save ordered page-start directory");
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Saved ordered page-start directory: %s (%zu attrs).\n",
                                 sidecarPath.c_str(),
                                 p_orderedPageStartAttrs.size());
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Output done...\n");
                auto t2 = std::chrono::high_resolution_clock::now();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to write results:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000);
            }

            ErrorCode GetWritePosting(ExtraWorkSpace* p_exWorkSpace, SizeType pid, std::string& posting, bool write = false) override {
                if (write) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Unsupport write\n");
                    return ErrorCode::Undefined;
                }
                ListInfo* listInfo = &(m_listInfos[pid]);
                size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                size_t realBytes = listInfo->listEleCount * m_vectorInfoSize;
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

        private:
            bool m_available = false;

            std::shared_ptr<Helper::Concurrent::ConcurrentQueue<int>> m_freeWorkSpaceIds;
            std::atomic<int> m_workspaceCount = 0;

            std::string m_extraFullGraphFile;

            std::vector<ListInfo> m_listInfos;
            bool m_oneContext;
            Options* m_opt;

            std::vector<std::shared_ptr<Helper::DiskIO>> m_indexFiles;
            std::unique_ptr<Compressor> m_pCompressor;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableDataCompression;
            bool m_enableDictTraining;
            
            void (ExtraStaticSearcher<ValueType>::*m_parsePosting)(uint64_t&, uint64_t&, int, int);
            void (ExtraStaticSearcher<ValueType>::*m_parseEncoding)(std::shared_ptr<VectorIndex>&, ListInfo*, ValueType*);

            int m_vectorInfoSize = 0;
            int m_iDataDimension = 0;

            int m_totalListCount = 0;

            int m_listPerFile = 0;

            bool m_staticPipePQ = false;
            int m_staticPipePQCodeBytes = 0;
            int m_staticPipePQDimension = 0;
            bool m_staticHasUnfilterTail = false;
            int m_staticTailPageBudget = 0;
            int m_staticMaxListPageCount = 0;
            bool m_staticHasMetadata = false;
            int m_staticNumTagsPerVec = 0;
            int m_staticACLTagCols = 0;
            int m_staticMetadataBytes = sizeof(int);
            std::vector<int> m_orderedPageStartAttrs;
            std::vector<std::uint32_t> m_orderedPageStartBases;
            std::vector<std::uint64_t> m_orderedPageStartOffsets;
            std::vector<std::int32_t> m_orderedPageStartBits;
            std::vector<uint32_t> m_staticBuildTags;
            int m_staticBuildNumTagsPerVec = 0;
            std::vector<std::vector<SizeType>> m_staticNodeVectorAssignments;
            std::vector<std::vector<SizeType>> m_staticPrimaryNodeVectorAssignments;
            std::unordered_map<SizeType, int> m_staticHeadVectorOwners;
            const std::unordered_map<SizeType, int>* m_staticHeadVectorOwnersView = nullptr;
            std::vector<std::shared_ptr<VectorIndex>> m_staticHeadBundleIndexes;
            const std::vector<std::vector<SizeType>>* m_staticHeadBundleLocalToGlobalHIDs = nullptr;
            const std::vector<std::vector<SizeType>>* m_staticHeadBundleNodeHeadVectorIDs = nullptr;
            StaticCrossGraphSearch m_staticCrossGraphSearch;
            std::vector<int> m_staticBuildVectorOwners;
            std::vector<int> m_staticBuildHeadOwners;
            std::unique_ptr<PipePQTable> m_staticPipePQTable;
            const std::uint8_t* m_staticPipePQCodes = nullptr;
            size_t m_staticPipePQN = 0;
#ifndef _MSC_VER
            int m_staticPipePQCodeFd = -1;
            void* m_staticPipePQCodeMap = nullptr;
#endif
            size_t m_staticPipePQCodeMapBytes = 0;
            std::shared_ptr<Helper::DiskIO> m_staticRerankFile;
            std::vector<std::shared_ptr<Helper::DiskIO>> m_staticRerankHandlers;
            size_t m_staticRerankCount = 0;
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_EXTRASTATICSEARCHER_H_
