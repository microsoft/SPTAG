// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "inc/SSDServing/VectorSearch/IExtraSearcher.h"

#include <map>
#include <cmath>
#include <memory>
#include <vector>
#include <future>

#ifdef _MSC_VER
#define ASYNC_READ 1
#endif

namespace SPTAG {
    
    extern std::function<std::shared_ptr<Helper::DiskPriorityIO>(void)> f_createAsyncIO;
    
    namespace SSDServing {
        namespace VectorSearch {

            template <typename ValueType>
            class ExtraFullGraphSearcher : public IExtraSearcher<ValueType>
            {
            public:
                ExtraFullGraphSearcher(const std::string& p_extraFullGraphFile, int p_postingPageLimit, int p_ioThreads)
                {
                    m_extraFullGraphFile = p_extraFullGraphFile;
                    std::string curFile = p_extraFullGraphFile;
                    do {
                        auto curIndexFile = f_createAsyncIO();
                        if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in, (1 << 20), 2, 2, p_ioThreads)) {
                            LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                            exit(-1);
                        }

                        m_indexContexts.emplace_back(curIndexFile);
                        m_totalListCount += LoadingHeadInfo(curFile, p_postingPageLimit, m_indexContexts.back());

                        curFile = p_extraFullGraphFile + "_" + std::to_string(m_indexContexts.size());
                    } while (fileexists(curFile.c_str()));
                    m_listPerFile = static_cast<int>((m_totalListCount + m_indexContexts.size() - 1) / m_indexContexts.size());
                }

                virtual ~ExtraFullGraphSearcher()
                {
                }

                virtual size_t GetMaxListSize() const { return (static_cast<size_t>(m_maxListPageCount) << c_pageSizeEx); }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    std::set<int>& truth, std::map<int, std::set<int>>& found)
                {
                    const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                    p_exWorkSpace->m_deduper.clear();

                    std::atomic<int> unprocessed(0);

                    bool oneContext = (m_indexContexts.size() == 1);
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

                        IndexContext* indexContext;
                        ListInfo* listInfo;
                        if (oneContext) {
                            indexContext = &(m_indexContexts[0]);
                            listInfo = &(indexContext->m_listInfos[curPostingID]);
                        }
                        else {
                            indexContext = &(m_indexContexts[curPostingID / m_listPerFile]);
                            listInfo = &(indexContext->m_listInfos[curPostingID % m_listPerFile]);
                        }

                        if (listInfo->listEleCount == 0)
                        {
                            continue;
                        }

                        size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << c_pageSizeEx);
                        char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ
                        ++unprocessed;
                        auto& request = p_exWorkSpace->m_diskRequests[pi];
                        request.m_offset = listInfo->listOffset;
                        request.m_readSize = totalBytes;
                        request.m_buffer = buffer;
                        request.m_callback = [&p_exWorkSpace, &request](bool success)
                        {
                            request.m_success = success;
                            p_exWorkSpace->m_processIocp.Push(&request);
                        };
                        request.m_success = false;
                        request.m_pListInfo = (void*)listInfo;

                        if (!((indexContext->m_indexFile)->ReadFileAsync(request)))
                        {
                            LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                            p_exWorkSpace->m_processIocp.Push(&request);
                        }
#else
                        auto numRead = (indexContext->m_indexFile)->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                        if (numRead != totalBytes) {
                            LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                            exit(-1);
                        }

                        for (int i = 0; i < listInfo->listEleCount; ++i)
                        {
                            char* vectorInfo = buffer + listInfo->pageOffset + i * m_vectorInfoSize;
                            int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                            vectorInfo += sizeof(int);

                            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue;
                            if (truth.count(vectorID)) found[curPostingID].insert(vectorID);

                            auto distance2leaf = p_index->ComputeDistance(p_queryResults.GetQuantizedTarget(), vectorInfo);
                            p_queryResults.AddPoint(vectorID, distance2leaf);
                        }
#endif
                    }

#ifdef ASYNC_READ
                    while (unprocessed > 0)
                    {
                        DiskListRequest* request;
                        if (!(p_exWorkSpace->m_processIocp.Pop(request))) break;

                        --unprocessed;

                        if (request->m_success)
                        {
                            ListInfo* listInfo = (ListInfo*)(request->m_pListInfo);
                            char* buffer = request->m_buffer;

                            for (int i = 0; i < listInfo->listEleCount; ++i)
                            {
                                char* vectorInfo = buffer + listInfo->pageOffset + i * m_vectorInfoSize;
                                int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                                vectorInfo += sizeof(int);

                                if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue;
                                int curPostingID = p_exWorkSpace->m_postingIDs[request - p_exWorkSpace->m_diskRequests.data()];
                                if (truth.count(vectorID)) found[curPostingID].insert(vectorID);

                                auto distance2leaf = p_index->ComputeDistance(p_queryResults.GetQuantizedTarget(), vectorInfo);
                                p_queryResults.AddPoint(vectorID, distance2leaf);
                            }
                        }
                    }
#endif
                }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    SearchStats& p_stats)
                {
                    const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                    p_exWorkSpace->m_deduper.clear();

                    std::atomic<int> unprocessed(0);
                    std::atomic<int> curCheck(0);
                    std::atomic<int> listElements(0);
                    std::atomic<int> diskIO(0);
                    std::atomic<int> diskRead(0);

                    bool oneContext = (m_indexContexts.size() == 1);
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

                        IndexContext* indexContext;
                        ListInfo* listInfo;
                        if (oneContext) {
                            indexContext = &(m_indexContexts[0]);
                            listInfo = &(indexContext->m_listInfos[curPostingID]);
                        }
                        else {
                            indexContext = &(m_indexContexts[curPostingID / m_listPerFile]);
                            listInfo = &(indexContext->m_listInfos[curPostingID % m_listPerFile]);
                        }

                        if (listInfo->listEleCount == 0)
                        {
                            continue;
                        }

                        diskRead += listInfo->listPageCount;
                        diskIO += 1;

                        size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << c_pageSizeEx);
                        char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ
                        ++unprocessed;
                        auto& request = p_exWorkSpace->m_diskRequests[pi];
                        request.m_offset = listInfo->listOffset;
                        request.m_readSize = totalBytes;
                        request.m_buffer = buffer;
                        request.m_callback = [&p_exWorkSpace, &request](bool success)
                        {
                            request.m_success = success;
                            p_exWorkSpace->m_processIocp.Push(&request);
                        };
                        request.m_success = false;
                        request.m_pListInfo = (void*)listInfo;

                        if (!((indexContext->m_indexFile)->ReadFileAsync(request)))
                        {
                            LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                            p_exWorkSpace->m_processIocp.Push(&request);
                        }
#else
                        auto numRead = (indexContext->m_indexFile)->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                        if (numRead != totalBytes) {
                            LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                            exit(-1);
                        }

                        for (int i = 0; i < listInfo->listEleCount; ++i)
                        {
                            char* vectorInfo = buffer + listInfo->pageOffset + i * m_vectorInfoSize;
                            int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                            vectorInfo += sizeof(int);

                            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue;

                            auto distance2leaf = p_index->ComputeDistance(p_queryResults.GetQuantizedTarget(), vectorInfo);
                            p_queryResults.AddPoint(vectorID, distance2leaf);
                            curCheck += 1;
                        }
                        listElements += listInfo->listEleCount;
#endif
                    }

#ifdef ASYNC_READ
                    while (unprocessed > 0)
                    {
                        DiskListRequest* request;
                        if (!(p_exWorkSpace->m_processIocp.Pop(request))) break;

                        --unprocessed;

                        if (request->m_success)
                        {
                            ListInfo* listInfo = (ListInfo*)(request->m_pListInfo);
                            char* buffer = request->m_buffer;

                            for (int i = 0; i < listInfo->listEleCount; ++i)
                            {
                                char* vectorInfo = buffer + listInfo->pageOffset + i * m_vectorInfoSize;
                                int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                                vectorInfo += sizeof(int);

                                if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue;

                                auto distance2leaf = p_index->ComputeDistance(p_queryResults.GetQuantizedTarget(), vectorInfo);
                                p_queryResults.AddPoint(vectorID, distance2leaf);
                                curCheck += 1;
                            }
                            listElements += listInfo->listEleCount;
                        }
                    }
#endif
                    p_stats.m_exCheck = curCheck;
                    p_stats.m_totalListElementsCount = listElements;
                    p_stats.m_diskIOCount = diskIO;
                    p_stats.m_diskAccessCount = diskRead;

                    p_queryResults.SortResult();
                }

            private:
                struct ListInfo
                {
                    int listEleCount = 0;

                    std::uint16_t listPageCount = 0;

                    std::uint64_t listOffset = 0;
                    
                    std::uint16_t pageOffset = 0;
                };

                struct IndexContext {
                    std::vector<ListInfo> m_listInfos;

                    std::shared_ptr<SPTAG::Helper::DiskPriorityIO> m_indexFile;

                    IndexContext(std::shared_ptr<SPTAG::Helper::DiskPriorityIO> indexFile) : m_indexFile(indexFile) {}
                };

            private:
                int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, IndexContext& p_indexContext)
                {
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                        exit(1);
                    }

                    int m_listCount;
                    int m_totalDocumentCount;
                    int m_iDataDimension;
                    int m_listPageOffset;
                    auto& m_listInfos = p_indexContext.m_listInfos;

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

                        m_listInfos[i].listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << c_pageSizeEx);
                        m_listInfos[i].listEleCount = min(m_listInfos[i].listEleCount, (min(static_cast<int>(m_listInfos[i].listPageCount), p_postingPageLimit) << c_pageSizeEx) / m_vectorInfoSize);
                        m_listInfos[i].listPageCount = static_cast<std::uint16_t>(ceil((m_vectorInfoSize * m_listInfos[i].listEleCount + m_listInfos[i].pageOffset) * 1.0 / (1 << c_pageSizeEx)));
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

                        if (pageCount > m_maxListPageCount) m_maxListPageCount = pageCount;
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


            private:

                std::string m_extraFullGraphFile;

                const static int c_pageSizeEx = 12;

                std::vector<IndexContext> m_indexContexts;

                int m_vectorInfoSize = 0;

                int m_totalListCount = 0;

                int m_listPerFile = 0;

                int m_maxListPageCount = 0;
            };
        }
    }
}
