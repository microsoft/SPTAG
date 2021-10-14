#pragma once
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <float.h>

#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"
#include "inc/SSDServing/VectorSearch/Options.h"
#include "inc/SSDServing/VectorSearch/SearchDefault.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
            namespace Local
            {
                const std::uint16_t c_pageSize = 4096;

                struct Edge
                {
                    Edge() : headID(INT_MAX), fullID(INT_MAX), distance(FLT_MAX), order(0)
                    {
                    }

                    int headID;
                    int fullID;
                    float distance;
					char order;
                };

                struct EdgeCompare
                {
                    bool operator()(const Edge& a, int b) const
                    {
                        return a.headID < b;
                    };

                    bool operator()(int a, const Edge& b) const
                    {
                        return a < b.headID;
                    };

                    bool operator()(const Edge& a, const Edge& b) const
                    {
                        if (a.headID == b.headID)
                        {
                            if (a.distance == b.distance)
                            {
                                return a.fullID < b.fullID;
                            }

                            return a.distance < b.distance;
                        }

                        return a.headID < b.headID;
                    };
                } g_edgeComparer;

                void LoadHeadVectorIDSet(const std::string& p_filename, std::unordered_set<int>& p_set)
                {
                    if (!p_filename.empty())
                    {
                        auto ptr = SPTAG::f_createIO();
                        if (ptr == nullptr || !ptr->Initialize(p_filename.c_str(), std::ios::binary | std::ios::in)) {
                            LOG(Helper::LogLevel::LL_Error, "failed open VectorIDTranslate: %s\n", p_filename.c_str());
                            exit(1);
                        }

                        long long vid;
                        while (ptr->ReadBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) == sizeof(vid))
                        {
                            p_set.insert(static_cast<int>(vid));
                        }
                        LOG(Helper::LogLevel::LL_Info, "Loaded %u Vector IDs\n", static_cast<uint32_t>(p_set.size()));
                    }
                    else
                    {
                        LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                        exit(1);
                    }
                }

                void SelectPostingOffset(size_t p_spacePerVector,
                    const std::vector<std::atomic_int>& p_postingListSizes,
                    std::unique_ptr<int[]>& p_postPageNum,
                    std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                    std::vector<int>& p_postingOrderInIndex)
                {
                    p_postPageNum.reset(new int[p_postingListSizes.size()]);
                    p_postPageOffset.reset(new std::uint16_t[p_postingListSizes.size()]);

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
                    p_postingOrderInIndex.reserve(p_postingListSizes.size());

                    PageModWithID listInfo;
                    for (size_t i = 0; i < p_postingListSizes.size(); ++i)
                    {
                        if (p_postingListSizes[i] == 0)
                        {
                            continue;
                        }

                        listInfo.id = static_cast<int>(i);
                        listInfo.rest = static_cast<std::uint16_t>((p_spacePerVector * p_postingListSizes[i]) % c_pageSize);

                        listRestSize.insert(listInfo);
                    }

                    listInfo.id = -1;

                    int currPageNum = 0;
                    std::uint16_t currOffset = 0;

                    while (!listRestSize.empty())
                    {
                        listInfo.rest = c_pageSize - currOffset;
                        auto iter = listRestSize.lower_bound(listInfo);
                        if (iter == listRestSize.end())
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
                            if (currOffset > c_pageSize)
                            {
                                LOG(Helper::LogLevel::LL_Error, "Crossing extra pages\n");
                                exit(1);
                            }

                            if (currOffset == c_pageSize)
                            {
                                ++currPageNum;
                                currOffset = 0;
                            }

                            currPageNum += static_cast<int>((p_spacePerVector * p_postingListSizes[iter->id]) / c_pageSize);

                            listRestSize.erase(iter);
                        }
                    }

                    LOG(Helper::LogLevel::LL_Info, "TotalPageNumbers: %d, IndexSize: %llu\n", currPageNum, static_cast<uint64_t>(currPageNum)* c_pageSize + currOffset);
                }


                void OutputSSDIndexFile(const std::string& p_outputFile,
                    size_t p_spacePerVector,
                    const std::vector<std::atomic_int>& p_postingListSizes,
                    const std::vector<Edge>& p_postingSelections,
                    const std::unique_ptr<int[]>& p_postPageNum,
                    const std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                    const std::vector<int>& p_postingOrderInIndex,
                    std::shared_ptr<VectorSet> p_fullVectors)
                {
                    LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed open file %s\n", p_outputFile.c_str());
                        exit(1);
                    }

                    std::uint64_t listOffset = sizeof(int) * 4;
                    listOffset += (sizeof(int) + sizeof(std::uint16_t) + sizeof(int) + sizeof(std::uint16_t)) * p_postingListSizes.size();

                    std::unique_ptr<char[]> paddingVals(new char[c_pageSize]);
                    memset(paddingVals.get(), 0, sizeof(char) * c_pageSize);

                    std::uint64_t paddingSize = c_pageSize - (listOffset % c_pageSize);
                    if (paddingSize == c_pageSize)
                    {
                        paddingSize = 0;
                    }
                    else
                    {
                        listOffset += paddingSize;
                    }

                    // Number of lists.
                    int i32Val = static_cast<int>(p_postingListSizes.size());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }

                    // Number of all documents.
                    i32Val = static_cast<int>(p_fullVectors->Count());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }

                    // Bytes of each vector.
                    i32Val = static_cast<int>(p_fullVectors->Dimension());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }

                    // Page offset of list content section.
                    i32Val = static_cast<int>(listOffset / c_pageSize);
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }

                    for (int i = 0; i < p_postingListSizes.size(); ++i)
                    {
                        int pageNum = 0;
                        std::uint16_t pageOffset = 0;
                        int listEleCount = 0;
                        std::uint16_t listPageCount = 0;

                        if (p_postingListSizes[i] > 0)
                        {
                            pageNum = p_postPageNum[i];
                            pageOffset = static_cast<std::uint16_t>(p_postPageOffset[i]);
                            listEleCount = static_cast<int>(p_postingListSizes[i]);
                            listPageCount = static_cast<std::uint16_t>((p_spacePerVector * p_postingListSizes[i]) / c_pageSize);
                            if (0 != ((p_spacePerVector * p_postingListSizes[i]) % c_pageSize))
                            {
                                ++listPageCount;
                            }
                        }
                        if (ptr->WriteBinary(sizeof(pageNum), reinterpret_cast<char*>(&pageNum)) != sizeof(pageNum)) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                        if (ptr->WriteBinary(sizeof(pageOffset), reinterpret_cast<char*>(&pageOffset)) != sizeof(pageOffset)) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                        if (ptr->WriteBinary(sizeof(listEleCount), reinterpret_cast<char*>(&listEleCount)) != sizeof(listEleCount)) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                        if (ptr->WriteBinary(sizeof(listPageCount), reinterpret_cast<char*>(&listPageCount)) != sizeof(listPageCount)) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                    }

                    if (paddingSize > 0)
                    {
                        if (ptr->WriteBinary(paddingSize, reinterpret_cast<char*>(paddingVals.get())) != paddingSize) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                    }

                    if (static_cast<uint64_t>(ptr->TellP()) != listOffset)
                    {
                        LOG(Helper::LogLevel::LL_Info, "List offset not match!\n");
                        exit(1);
                    }

                    LOG(Helper::LogLevel::LL_Info, "SubIndex Size: %llu bytes, %llu MBytes\n", listOffset, listOffset >> 20);

                    listOffset = 0;

                    std::uint64_t paddedSize = 0;
                    for (auto id : p_postingOrderInIndex)
                    {
                        std::uint64_t targetOffset = static_cast<uint64_t>(p_postPageNum[id])* c_pageSize + p_postPageOffset[id];
                        if (targetOffset < listOffset)
                        {
                            LOG(Helper::LogLevel::LL_Info, "List offset not match, targetOffset < listOffset!\n");
                            exit(1);
                        }

                        if (targetOffset > listOffset)
                        {
                            if (targetOffset - listOffset > c_pageSize)
                            {
                                LOG(Helper::LogLevel::LL_Error, "Padding size greater than page size!\n");
                                exit(1);
                            }

                            if (ptr->WriteBinary(targetOffset - listOffset, reinterpret_cast<char*>(paddingVals.get())) != targetOffset - listOffset) {
                                LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                                exit(1);
                            }

                            paddedSize += targetOffset - listOffset;

                            listOffset = targetOffset;
                        }


                        std::size_t selectIdx = std::lower_bound(p_postingSelections.begin(), p_postingSelections.end(), id, g_edgeComparer) - p_postingSelections.begin();
                        for (int j = 0; j < p_postingListSizes[id]; ++j)
                        {
                            if (p_postingSelections[selectIdx].headID != id)
                            {
                                LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                                exit(1);
                            }

                            i32Val = p_postingSelections[selectIdx++].fullID;
                            if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                                LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                                exit(1);
                            }
                            if (ptr->WriteBinary(p_fullVectors->PerVectorDataSize(), reinterpret_cast<char*>(p_fullVectors->GetVector(i32Val))) != p_fullVectors->PerVectorDataSize()) {
                                LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                                exit(1);
                            }
                            listOffset += p_spacePerVector;
                        }
                    }

                    paddingSize = c_pageSize - (listOffset % c_pageSize);
                    if (paddingSize == c_pageSize)
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
                        if (ptr->WriteBinary(paddingSize, reinterpret_cast<char*>(paddingVals.get())) != paddingSize) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                    }

                    LOG(Helper::LogLevel::LL_Info, "Padded Size: %llu, final total size: %llu.\n", paddedSize, listOffset);

                    LOG(Helper::LogLevel::LL_Info, "Output done...\n");
                }
            }

            template<typename ValueType>
            void BuildSsdIndex(Options& p_opts)
            {
                using namespace Local;

                TimeUtils::StopW sw;

                std::string outputFile = COMMON_OPTS.m_ssdIndex;

                if (outputFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Error, "Output file can't be empty!\n");
                    exit(1);
                }

                int numThreads = p_opts.m_iNumberOfThreads;
                int candidateNum = p_opts.m_internalResultNum;

                std::unordered_set<int> headVectorIDS;
                LoadHeadVectorIDSet(COMMON_OPTS.m_headIDFile, headVectorIDS);

                SearchDefault<ValueType> searcher;
                LOG(Helper::LogLevel::LL_Info, "Start setup index...\n");
                searcher.Setup(p_opts);

                LOG(Helper::LogLevel::LL_Info, "Setup index finish, start setup hint...\n");
                searcher.SetHint(numThreads, candidateNum, false, p_opts);

                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(COMMON_OPTS.m_valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_vectorType, COMMON_OPTS.m_vectorDelimiter));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(COMMON_OPTS.m_vectorPath))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                    exit(1);
                }
                auto fullVectors = vectorReader->GetVectorSet();
                if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine) fullVectors->Normalize(p_opts.m_iNumberOfThreads);

                LOG(Helper::LogLevel::LL_Info, "Full vector loaded.\n");

                std::vector<Edge> selections(static_cast<size_t>(fullVectors->Count())* p_opts.m_replicaCount);

                std::vector<int> replicaCount(fullVectors->Count(), 0);
                std::vector<std::atomic_int> postingListSize(searcher.HeadIndex()->GetNumSamples());
                for (auto& pls : postingListSize) pls = 0;

                LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");

                std::vector<std::thread> threads;
                threads.reserve(numThreads);

                std::atomic_int nextFullID(0);
                std::atomic_size_t rngFailedCountTotal(0);

                for (int tid = 0; tid < numThreads; ++tid)
                {
                    threads.emplace_back([&, tid]()
                        {
                            COMMON::QueryResultSet<ValueType> resultSet(NULL, candidateNum);
                            SearchStats searchStats;

                            size_t rngFailedCount = 0;

                            while (true)
                            {
                                int fullID = nextFullID.fetch_add(1);
                                if (fullID >= fullVectors->Count())
                                {
                                    break;
                                }

                                if (headVectorIDS.count(fullID) > 0)
                                {
                                    continue;
                                }

                                ValueType* buffer = reinterpret_cast<ValueType*>(fullVectors->GetVector(fullID));
                                resultSet.SetTarget(buffer);
                                resultSet.Reset();

                                searcher.Search(resultSet, searchStats);

                                size_t selectionOffset = static_cast<size_t>(fullID)* p_opts.m_replicaCount;

                                BasicResult* queryResults = resultSet.GetResults();
                                for (int i = 0; i < candidateNum && replicaCount[fullID] < p_opts.m_replicaCount; ++i)
                                {
                                    if (queryResults[i].VID == -1)
                                    {
                                        break;
                                    }

                                    // RNG Check.
                                    bool rngAccpeted = true;
                                    for (int j = 0; j < replicaCount[fullID]; ++j)
                                    {
                                        // VQANNSearch::QueryResultSet<ValueType> resultSet(NULL, candidateNum);

                                        float nnDist = searcher.HeadIndex()->ComputeDistance(
                                            searcher.HeadIndex()->GetSample(queryResults[i].VID),
                                            searcher.HeadIndex()->GetSample(selections[selectionOffset + j].headID));

                                        // LOG(Helper::LogLevel::LL_Info,  "NNDist: %f Original: %f\n", nnDist, queryResults[i].Score);
                                        if (nnDist <= queryResults[i].Dist)
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

                                    ++postingListSize[queryResults[i].VID];

                                    selections[selectionOffset + replicaCount[fullID]].headID = queryResults[i].VID;
                                    selections[selectionOffset + replicaCount[fullID]].fullID = fullID;
                                    selections[selectionOffset + replicaCount[fullID]].distance = queryResults[i].Dist;
									selections[selectionOffset + replicaCount[fullID]].order = (char)replicaCount[fullID];
                                    ++replicaCount[fullID];
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

                std::sort(selections.begin(), selections.end(), g_edgeComparer);

                int postingSizeLimit = INT_MAX;
                if (p_opts.m_postingPageLimit > 0)
                {
                    postingSizeLimit = static_cast<int>(p_opts.m_postingPageLimit * c_pageSize / (fullVectors->PerVectorDataSize() + sizeof(int)));
                }

                LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);

                {
                    std::vector<int> replicaCountDist(p_opts.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0)
                        {
                            continue;
                        }

                        ++replicaCountDist[replicaCount[i]];
                    }

                    LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

                for (int i = 0; i < postingListSize.size(); ++i)
                {
                    if (postingListSize[i] <= postingSizeLimit)
                    {
                        continue;
                    }

                    std::size_t selectIdx = std::lower_bound(selections.begin(), selections.end(), i, g_edgeComparer) - selections.begin();
					/*
					int deletenum = postingListSize[i] - postingSizeLimit;
					for (char remove = p_opts.m_replicaCount - 1; deletenum > 0 && remove > 0; remove--)
					{
						for (int dropID = postingListSize[i] - 1; deletenum > 0 && dropID >= 0; --dropID)
						{
							if (selections[selectIdx + dropID].order == remove) {
								selections[selectIdx + dropID].order = -1;
								--replicaCount[selections[selectIdx + dropID].fullID];
								deletenum--;
							}
						}
					}

					for (int iid = 0; iid < postingSizeLimit + deletenum; iid++) {
						if (selections[selectIdx + iid].order < 0) {
							for (int ij = iid + 1; ij < postingListSize[i]; ij++) {
								if (selections[selectIdx + ij].order >= 0) {
									std::swap(selections[selectIdx + iid], selections[selectIdx + ij]);
									break;
								}
							}
						}
					}
					*/
					
                    for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                    {
                        int fullID = selections[selectIdx + dropID].fullID;
                        --replicaCount[fullID];
                    }
					
                    postingListSize[i] = postingSizeLimit;
                }

                if (p_opts.m_outputEmptyReplicaID)
                {
                    std::vector<int> replicaCountDist(p_opts.m_replicaCount + 1, 0);
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                        exit(1);
                    }
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0)
                        {
                            continue;
                        }

                        ++replicaCountDist[replicaCount[i]];

                        if (replicaCount[i] < 2)
                        {
                            long long vid = i;
                            if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                                LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
                                exit(1);
                            }
                        }
                    }

                    LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

                // VectorSize + VectorIDSize
                size_t vectorInfoSize = sizeof(ValueType) * fullVectors->Dimension() + sizeof(int);

                std::unique_ptr<int[]> postPageNum;
                std::unique_ptr<std::uint16_t[]> postPageOffset;
                std::vector<int> postingOrderInIndex;
                SelectPostingOffset(vectorInfoSize, postingListSize, postPageNum, postPageOffset, postingOrderInIndex);

                OutputSSDIndexFile(outputFile,
                    vectorInfoSize,
                    postingListSize,
                    selections,
                    postPageNum,
                    postPageOffset,
                    postingOrderInIndex,
                    fullVectors);

                double elapsedMinutes = sw.getElapsedMin();
                LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);
            }
        }
    }
}