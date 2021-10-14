#pragma once
#include "inc/SSDServing/VectorSearch/IExtraSearcherLinux.h"
#include "inc/SSDServing/VectorSearch/SearchStats.h"

#include <fstream>
#include <map>
#include <memory>
#include <vector>
#include <future>

#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {

            void ErrorExit(const char* p_str, int p_errno)
            {
                errno = p_errno;
                perror(p_str);
                exit(-1);
            }

            template <typename ValueType>
            class ExtraFullGraphSearcher : public IExtraSearcher<ValueType>
            {
            public:
                ExtraFullGraphSearcher(const std::string& p_extraFullGraphFile)
                {
                    m_extraFullGraphFile = p_extraFullGraphFile;
                    LoadingHeadInfo(p_extraFullGraphFile);
                    m_fd = open(p_extraFullGraphFile.c_str(), O_RDONLY);
                    if (m_fd == -1) {
                        // function between perror may change errno.
                        int errsv = errno;
                        char input[4096];
                        snprintf(input, 4096, "File %s can't be opened", p_extraFullGraphFile.c_str());
                        ErrorExit(input, errsv);
                    }
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
                }

                virtual void Setup(Options& p_config)
                {
                }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    SPTAG::COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    SearchStats& p_stats)
                {
                    const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                    InitWorkSpace(p_exWorkSpace, postingListCount);

                    std::atomic<std::int32_t> diskRead(0);
                    int curCheck = 0;

                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {

                        if (p_exWorkSpace->m_postingIDs[pi] >= m_listCount)
                        {
                            continue;
                        }

                        const auto& listInfo = m_listInfos[p_exWorkSpace->m_postingIDs[pi]];
                        if (listInfo.listEleCount == 0)
                        {
                            continue;
                        }

                        diskRead += listInfo.listPageCount;

                        size_t totalBytes = static_cast<size_t>(listInfo.listPageCount) * c_pageSize;
                        auto& buffer = p_exWorkSpace->m_pageBuffers[pi];
                        buffer.ReservePageBuffer(totalBytes);
                        void* bufferVoidPtr = reinterpret_cast<void*>(buffer.GetBuffer());
                        off_t offset = (m_listPageOffset + listInfo.pageNum) * c_pageSize;

                        ssize_t numRead = pread(m_fd, bufferVoidPtr, totalBytes, offset);
                        if (numRead == -1) {
                            // function between perror may change errno.
                            int errsv = errno;
                            char input[4096];
                            snprintf(input, 4096, "Read error: offset, %ld, bytes, %ld", offset, totalBytes);
                            ErrorExit(input, errsv);
                        }
                        if (numRead != totalBytes) {
                            fprintf(stderr, "File %s read bytes, expected: %ld, acutal: %ld.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                            exit(-1);
                        }

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

                    p_stats.m_exCheck = curCheck;
                    p_stats.m_diskAccessCount = diskRead;

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

            private:
                void LoadingHeadInfo(const std::string& p_file)
                {
                    std::ifstream input(p_file, std::ios::binary);
                    if (!input.is_open())
                    {
                        fprintf(stderr, "Failed to open file: %s\n", p_file.c_str());
                        exit(1);
                    }

                    input.read(reinterpret_cast<char*>(&m_listCount), sizeof(m_listCount));
                    input.read(reinterpret_cast<char*>(&m_totalDocumentCount), sizeof(m_totalDocumentCount));
                    input.read(reinterpret_cast<char*>(&m_iDataDimension), sizeof(m_iDataDimension));
                    input.read(reinterpret_cast<char*>(&m_listPageOffset), sizeof(m_listPageOffset));

                    m_listInfos.reset(new ListInfo[m_listCount]);

                    size_t totalListElementCount = 0;

                    std::map<int, int> pageCountDist;

                    int biglistCount = 0;
                    int biglistElementCount = 0;
                    for (int i = 0; i < m_listCount; ++i)
                    {
                        input.read(reinterpret_cast<char*>(&(m_listInfos[i].pageNum)), sizeof(m_listInfos[i].pageNum));
                        input.read(reinterpret_cast<char*>(&(m_listInfos[i].pageOffset)), sizeof(m_listInfos[i].pageOffset));
                        input.read(reinterpret_cast<char*>(&(m_listInfos[i].listEleCount)), sizeof(m_listInfos[i].listEleCount));
                        input.read(reinterpret_cast<char*>(&(m_listInfos[i].listPageCount)), sizeof(m_listInfos[i].listPageCount));

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

                    input.close();

                    fprintf(stderr,
                        "Finish reading header info, list count %d, total doc count %d, dimension %d, list page offset %d.\n",
                        m_listCount,
                        m_totalDocumentCount,
                        m_iDataDimension,
                        m_listPageOffset);


                    fprintf(stderr,
                        "Big page (>4K): list count %d, total element count %d.\n",
                        biglistCount,
                        biglistElementCount);

                    fprintf(stderr, "Total Element Count: %lu\n", totalListElementCount);

                    for (auto& ele : pageCountDist)
                    {
                        fprintf(stderr, "Page Count Dist: %d %d\n", ele.first, ele.second);
                    }

                    m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
                }


            private:

                std::string m_extraFullGraphFile;

                const static std::uint64_t c_pageSize = 4096;

                int m_iDataDimension;

                int m_listCount;

                int m_totalDocumentCount;

                int m_listPageOffset;

                size_t m_vectorInfoSize;

                std::unique_ptr<ListInfo[]> m_listInfos;

                int m_fd;
            };
        }
    }
}
