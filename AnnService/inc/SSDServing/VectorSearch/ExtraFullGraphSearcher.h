#pragma once
#include "inc/SSDServing/VectorSearch/IExtraSearcher.h"
#include "inc/SSDServing/VectorSearch/SearchStats.h"
#include "inc/SSDServing/VectorSearch/DiskListCommonUtils.h"
#include "inc/SSDServing/VectorSearch/PrioritizedDiskFileReader.h"

#include <map>
#include <memory>
#include <vector>
#include <future>

#include <fileapi.h>



namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
            void ErrorExit();

            template <typename ValueType>
            class ExtraFullGraphSearcher : public IExtraSearcher<ValueType>
            {
            public:
                ExtraFullGraphSearcher(const std::string& p_extraFullGraphFile)
                {
                    m_extraFullGraphFile = p_extraFullGraphFile;
                    LoadingHeadInfo(p_extraFullGraphFile);
                    m_indexFile = std::make_shared<PrioritizedDiskFileReader>(p_extraFullGraphFile.c_str());
                }

                virtual ~ExtraFullGraphSearcher()
                {
                }

                virtual void InitWorkSpace(ExtraWorkSpace* p_space, int p_resNumHint)
                {
                    p_space->m_deduper.Clear();

                    if (p_space->m_pageBuffers.size() < p_resNumHint)
                    {
                        p_space->m_pageBuffers.resize(p_resNumHint);
                    }

                    if (p_space->m_diskRequests.size() < p_resNumHint)
                    {
                        p_space->m_diskRequests.resize(p_resNumHint);
                    }
                }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    SearchStats& p_stats)
                {
                    const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                    InitWorkSpace(p_exWorkSpace, postingListCount);

                    std::atomic<int> unprocessed = 0;
                    std::atomic_int32_t diskRead = 0;
                    int curCheck = 0;

                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto& request = p_exWorkSpace->m_diskRequests[pi];
                        request.m_requestID = pi;
                        request.m_success = false;

                        if (p_exWorkSpace->m_postingIDs[pi] >= m_listCount)
                        {
                            continue;
                        }

                        const auto& listInfo = m_listInfos[p_exWorkSpace->m_postingIDs[pi]];
                        if (listInfo.listEleCount == 0)
                        {
                            continue;
                        }

                        ++unprocessed;
                        diskRead += listInfo.listPageCount;

                        size_t totalBytes = static_cast<size_t>(listInfo.listPageCount)* c_pageSize;
                        auto& buffer = p_exWorkSpace->m_pageBuffers[pi];
                        buffer.ReservePageBuffer(totalBytes);

                        request.m_buffer = buffer.GetBuffer();
                        request.m_offset = (m_listPageOffset + listInfo.pageNum) * c_pageSize;
                        request.m_readSize = totalBytes;
                        
                        request.m_callback = [&p_exWorkSpace, &request]()
                        {
                            request.m_success = true;
                            ::PostQueuedCompletionStatus(p_exWorkSpace->m_processIocp.GetHandle(),
                                0,
                                NULL,
                                reinterpret_cast<LPOVERLAPPED>(&request));
                        };

                        if (!m_indexFile->ReadFileAsync(request))
                        {
                            ::PostQueuedCompletionStatus(p_exWorkSpace->m_processIocp.GetHandle(),
                                0,
                                NULL,
                                reinterpret_cast<LPOVERLAPPED>(&request));
                        }
                    }

                    // NEW SECTION
                    DWORD cBytes;
                    ULONG_PTR key;
                    OVERLAPPED* ol;
                    HANDLE iocp = p_exWorkSpace->m_processIocp.GetHandle();

                    while (unprocessed > 0)
                    {
                        BOOL ret = ::GetQueuedCompletionStatus(iocp,
                            &cBytes,
                            &key,
                            &ol,
                            INFINITE);

                        if (FALSE == ret || nullptr == ol)
                        {
                            break;
                        }

                        auto request = reinterpret_cast<DiskListRequest*>(ol);
                        uint32_t pi = request->m_requestID;

                        --unprocessed;

                        if (request->m_success)
                        {
                            const auto& vi = p_exWorkSpace->m_postingIDs[pi];
                            const auto& listInfo = m_listInfos[vi];
                            auto& buffer = p_exWorkSpace->m_pageBuffers[pi];

                            for (int i = 0; i < listInfo.listEleCount; ++i)
                            {
                                std::uint8_t* vectorInfo = buffer.GetBuffer() + listInfo.pageOffset + i * m_vectorInfoSize;
                                int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                                vectorInfo += sizeof(int);

                                if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID))
                                {
                                    continue;
                                }

                                auto distance2leaf = p_index->ComputeDistance(p_queryResults.GetTarget(),
                                    vectorInfo);

                                p_queryResults.AddPoint(vectorID, distance2leaf);

                            }
                        }
                    }
                    // OLD SECTION

                    p_stats.m_exCheck = curCheck;
                    p_stats.m_diskAccessCount = diskRead;

                    p_queryResults.SortResult();
                }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index)
                {
                    const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                    InitWorkSpace(p_exWorkSpace, postingListCount);

                    std::atomic<int> unprocessed = 0;
                    std::atomic_int32_t diskRead = 0;
                    int curCheck = 0;

                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto& request = p_exWorkSpace->m_diskRequests[pi];
                        request.m_requestID = pi;
                        request.m_success = false;

                        if (p_exWorkSpace->m_postingIDs[pi] >= m_listCount)
                        {
                            continue;
                        }

                        const auto& listInfo = m_listInfos[p_exWorkSpace->m_postingIDs[pi]];
                        if (listInfo.listEleCount == 0)
                        {
                            continue;
                        }

                        ++unprocessed;
                        diskRead += listInfo.listPageCount;

                        size_t totalBytes = static_cast<size_t>(listInfo.listPageCount)* c_pageSize;
                        auto& buffer = p_exWorkSpace->m_pageBuffers[pi];
                        buffer.ReservePageBuffer(totalBytes);

                        request.m_buffer = buffer.GetBuffer();
                        request.m_offset = (m_listPageOffset + listInfo.pageNum) * c_pageSize;
                        request.m_readSize = totalBytes;
                        
                        request.m_callback = [&p_exWorkSpace, &request]()
                        {
                            request.m_success = true;
                            ::PostQueuedCompletionStatus(p_exWorkSpace->m_processIocp.GetHandle(),
                                0,
                                NULL,
                                reinterpret_cast<LPOVERLAPPED>(&request));
                        };

                        if (!m_indexFile->ReadFileAsync(request))
                        {
                            ::PostQueuedCompletionStatus(p_exWorkSpace->m_processIocp.GetHandle(),
                                0,
                                NULL,
                                reinterpret_cast<LPOVERLAPPED>(&request));
                        }
                    }

                    // NEW SECTION
                    DWORD cBytes;
                    ULONG_PTR key;
                    OVERLAPPED* ol;
                    HANDLE iocp = p_exWorkSpace->m_processIocp.GetHandle();

                    while (unprocessed > 0)
                    {
                        BOOL ret = ::GetQueuedCompletionStatus(iocp,
                            &cBytes,
                            &key,
                            &ol,
                            INFINITE);

                        if (FALSE == ret || nullptr == ol)
                        {
                            break;
                        }

                        auto request = reinterpret_cast<DiskListRequest*>(ol);
                        uint32_t pi = request->m_requestID;

                        --unprocessed;

                        if (request->m_success)
                        {
                            const auto& vi = p_exWorkSpace->m_postingIDs[pi];
                            const auto& listInfo = m_listInfos[vi];
                            auto& buffer = p_exWorkSpace->m_pageBuffers[pi];

                            for (int i = 0; i < listInfo.listEleCount; ++i)
                            {
                                std::uint8_t* vectorInfo = buffer.GetBuffer() + listInfo.pageOffset + i * m_vectorInfoSize;
                                int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                                vectorInfo += sizeof(int);

                                if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID))
                                {
                                    continue;
                                }

                                auto distance2leaf = p_index->ComputeDistance(p_queryResults.GetTarget(),
                                    vectorInfo);

                                p_queryResults.AddPoint(vectorID, distance2leaf);

                            }
                        }
                    }
                    // OLD SECTION

                    p_queryResults.SortResult();
                }

            private:
                struct ListInfo
                {
                    int pageNum = 0;

                    std::uint16_t pageOffset = 0;

                    int listEleCount = 0;

                    std::uint16_t listPageCount = 0;
                };

                void ReadList(HANDLE& p_fileHandle, PageBuffer<std::uint8_t>& p_buffer, const ListInfo& p_listInfo)
                {
                    size_t totalBytes = static_cast<size_t>(p_listInfo.listPageCount)* c_pageSize;

                    p_buffer.ReservePageBuffer(totalBytes);

                    DWORD bytesRead = 0;

                    LARGE_INTEGER li;
                    li.QuadPart = (m_listPageOffset + p_listInfo.pageNum) * c_pageSize;

                    if (!::SetFilePointerEx(p_fileHandle, li, NULL, FILE_BEGIN))
                    {
                        ErrorExit();
                    }

                    if (!::ReadFile(p_fileHandle,
                        p_buffer.GetBuffer(),
                        static_cast<DWORD>(totalBytes),
                        &bytesRead,
                        NULL))
                    {
                        ErrorExit();
                    }
                }

            private:
                void LoadingHeadInfo(const std::string& p_file)
                {
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                        exit(1);
                    }

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

                    m_listInfos.reset(new ListInfo[m_listCount]);

                    size_t totalListElementCount = 0;

                    std::map<int, int> pageCountDist;

                    int biglistCount = 0;
                    int biglistElementCount = 0;
                    for (int i = 0; i < m_listCount; ++i)
                    {
                        if (ptr->ReadBinary(sizeof(m_listInfos[i].pageNum), reinterpret_cast<char*>(&(m_listInfos[i].pageNum))) != sizeof(m_listInfos[i].pageNum)) {
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
                        "Big page (>4K): list count %d, total element count %d.\n",
                        biglistCount,
                        biglistElementCount);

                    LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n", totalListElementCount);

                    for (auto& ele : pageCountDist)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Page Count Dist: %d %d\n", ele.first, ele.second);
                    }

                    m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
                }


            private:

                std::string m_extraFullGraphFile;

                const static std::uint64_t c_pageSize = 4096;

                std::shared_ptr<IDiskFileReader> m_indexFile;

                int m_iDataDimension;

                int m_listCount;

                int m_totalDocumentCount;

                int m_listPageOffset;

                size_t m_vectorInfoSize;

                std::unique_ptr<ListInfo[]> m_listInfos;
            };
        }
    }
}
