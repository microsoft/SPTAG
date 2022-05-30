// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASEARCHER_H_
#define _SPTAG_SPANN_EXTRASEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "../Common/TruthSet.h"
#include "Compressor.h"

#include <map>
#include <cmath>
#include <climits>
#include <future>
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

            void SaveBatch()
            {
                auto f_out = f_createIO();
                if (f_out == nullptr || !f_out->Initialize(m_tmpfile.c_str(), std::ios::out | std::ios::binary | (fileexists(m_tmpfile.c_str()) ? std::ios::in : 0))) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot open %s to save selection for batching!\n", m_tmpfile.c_str());
                    exit(1);
                }
                if (f_out->WriteBinary(sizeof(Edge) * (m_end - m_start), (const char*)m_selections.data(), sizeof(Edge) * m_start) != sizeof(Edge) * (m_end - m_start)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot write to %s!\n", m_tmpfile.c_str());
                    exit(1);
                }
                std::vector<Edge> batch_selection;
                m_selections.swap(batch_selection);
                m_start = m_end = 0;
            }

            void LoadBatch(size_t start, size_t end)
            {
                auto f_in = f_createIO();
                if (f_in == nullptr || !f_in->Initialize(m_tmpfile.c_str(), std::ios::in | std::ios::binary)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot open %s to load selection batch!\n", m_tmpfile.c_str());
                    exit(1);
                }

                size_t readsize = end - start;
                m_selections.resize(readsize);
                if (f_in->ReadBinary(readsize * sizeof(Edge), (char*)m_selections.data(), start * sizeof(Edge)) != readsize * sizeof(Edge)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot read from %s! start:%zu size:%zu\n", m_tmpfile.c_str(), start, readsize);
                    exit(1);
                }
                m_start = start;
                m_end = end;
            }

            size_t lower_bound(SizeType node)
            {
                auto ptr = std::lower_bound(m_selections.begin(), m_selections.end(), node, g_edgeComparer);
                return m_start + (ptr - m_selections.begin());
            }

            Edge& operator[](size_t offset)
            {
                if (offset < m_start || offset >= m_end) {
                    LOG(Helper::LogLevel::LL_Error, "Error read offset in selections:%zu\n", offset);
                }
                return m_selections[offset - m_start];
            }
        };

#define ProcessPosting(p_postingListFullData, vectorInfoSize, m_enablePostingListRearrange) \
        for (int i = 0; i < listInfo->listEleCount; i++) { \
            uint64_t offsetVectorID = m_enablePostingListRearrange ? (vectorInfoSize - sizeof(int)) * listInfo->listEleCount + sizeof(int) * i : vectorInfoSize * i; \
            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID));\
            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue; \
            uint64_t offsetVector = m_enablePostingListRearrange ? (vectorInfoSize - sizeof(int)) * i : vectorInfoSize * i + sizeof(int); \
            auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
        } \

        template <typename ValueType>
        class ExtraFullGraphSearcher : public IExtraSearcher
        {
        public:
            ExtraFullGraphSearcher()
            {
            }

            virtual ~ExtraFullGraphSearcher()
            {
            }

            virtual bool LoadIndex(Options& p_opt) {
                m_extraFullGraphFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                std::string curFile = m_extraFullGraphFile;
                do {
                    auto curIndexFile = f_createAsyncIO();
                    if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in, 
#ifndef _MSC_VER
#ifdef BATCH_READ
                        p_opt.m_searchInternalResultNum, 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                        p_opt.m_searchInternalResultNum* p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
#else
                        (p_opt.m_searchPostingPageLimit + 1) * PageSize, 2, 2, p_opt.m_ioThreads
#endif
                    )) {
                        LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", curFile.c_str());
                        return false;
                    }

                    m_indexFiles.emplace_back(curIndexFile);
                    m_listInfos.emplace_back(0);
                    m_totalListCount += LoadingHeadInfo(curFile, p_opt.m_searchPostingPageLimit, m_listInfos.back());

                    curFile = m_extraFullGraphFile + "_" + std::to_string(m_indexFiles.size());
                } while (fileexists(curFile.c_str()));
                m_listPerFile = static_cast<int>((m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());

#ifndef _MSC_VER
                Helper::AIOTimeout.tv_nsec = p_opt.m_iotimeout * 1000;
#endif
                return true;
            }

            virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats,
                std::set<int>* truth, std::map<int, std::set<int>>* found)
            {
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
 
                int diskRead = 0;
                int diskIO = 0;
                int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                bool oneContext = (m_indexFiles.size() == 1);
                
                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    int fileid = 0;
                    ListInfo* listInfo;
                    if (oneContext) {
                        listInfo = &(m_listInfos[0][curPostingID]);
                    }
                    else {
                        fileid = curPostingID / m_listPerFile;
                        listInfo = &(m_listInfos[fileid][curPostingID % m_listPerFile]);
                    }

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    diskRead += listInfo->listPageCount;
                    diskIO += 1;
                    listElements += listInfo->listEleCount;

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_buffer = buffer;
                    request.m_status = (fileid << 16) | p_exWorkSpace->m_spaceID;
                    request.m_payload = (void*)listInfo; 
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    auto vectorInfoSize = m_vectorInfoSize;

                    request.m_callback = [&p_exWorkSpace, &queryResults, &p_index, vectorInfoSize, curPostingID, m_enableDeltaEncoding = m_enableDeltaEncoding, m_enablePostingListRearrange = m_enablePostingListRearrange, m_enableDictTraining = m_enableDictTraining, m_enableDataCompression = m_enableDataCompression, &m_pCompressor = m_pCompressor](Helper::AsyncReadRequest *request)
                    {
                        char* buffer = request->m_buffer;
                        ListInfo* listInfo = (ListInfo*)(request->m_payload);

                        // decompress posting list
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        std::string postingListFullData("");
                        if (m_enableDataCompression)
                        {
                            if (listInfo->listEleCount != 0)
                            {
                                postingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, m_enableDictTraining);
                            }
                            if (postingListFullData.size() != listInfo->listEleCount * vectorInfoSize)
                            {
                                LOG(Helper::LogLevel::LL_Info, "postingListFullData size not match! %zu, %d, \n", postingListFullData.size(), listInfo->listEleCount * vectorInfoSize);
                                exit(1);
                            }
                            p_postingListFullData = &postingListFullData[0];
                        }

                        // delta encoding
                        if (m_enableDeltaEncoding)
                        {
                            ValueType *headVector = (ValueType *)p_index->GetSample(curPostingID);
                            for (int i = 0; i < listInfo->listEleCount; i++)
                            {
                                ValueType *leaf = m_enablePostingListRearrange ? reinterpret_cast<ValueType *>(p_postingListFullData + (vectorInfoSize - sizeof(int)) * i) : reinterpret_cast<ValueType *>(p_postingListFullData + vectorInfoSize * i + sizeof(int));
                                for (auto i = 0; i < p_index->GetFeatureDim(); i++)
                                {
                                    leaf[i] += headVector[i];
                                }
                            }
                        }

                        ProcessPosting(const_cast<char *>(p_postingListFullData), vectorInfoSize, m_enablePostingListRearrange);
                    };
#else // async read
                    request.m_callback = [&p_exWorkSpace](Helper::AsyncReadRequest* request)
                    {
                        p_exWorkSpace->m_processIocp.push(request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
                        unprocessed--;
                    }
#endif
#else // sync read
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        exit(-1);
                    }
                    ProcessPosting(m_vectorInfoSize)
#endif
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    ProcessPosting(m_vectorInfoSize)
                }
#endif
#endif
                if (truth) { // TODO: check truth
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

                        ListInfo* listInfo = &(m_listInfos[curPostingID / m_listPerFile][curPostingID % m_listPerFile]);
                        char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

                        char* p_postingListFullData;
                        if (m_enableDataCompression)
                        {
                            std::string postingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, m_enableDictTraining);
                            p_postingListFullData = const_cast<char*>(postingListFullData.c_str());
                        }
                        else
                        {
                            p_postingListFullData = buffer + listInfo->pageOffset;
                        }

                        for (size_t i = 0; i < listInfo->listEleCount; ++i) {
                            char* vectorInfo = p_postingListFullData + i * m_vectorInfoSize;
                            int vectorID = *(reinterpret_cast<int*>(vectorInfo));

                            LOG(Helper::LogLevel::LL_Info, "vectorID: %d\n", vectorID);

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
            }

            std::string GetPostingListFullData(
                int postingListId,
                size_t p_postingListSize,
                Selection &p_selections,
                std::shared_ptr<VectorSet> p_fullVectors,
                bool m_enableDeltaEncoding = false,
                bool m_enablePostingListRearrange = false,
                const ValueType *headVector = nullptr)
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
                        LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH! node:%d offset:%zu\n", postingListId, selectIdx);
                        exit(1);
                    }
                    std::string vectorID("");
                    std::string vector("");

                    int vid = p_selections[selectIdx++].tonode;
                    vectorID.append(reinterpret_cast<char *>(&vid), sizeof(int));

                    ValueType *p_vector = reinterpret_cast<ValueType *>(p_fullVectors->GetVector(vid));
                    if (m_enableDeltaEncoding)
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

                    if (m_enablePostingListRearrange)
                    {
                        vectorIDs += vectorID;
                        vectors += vector;
                    }
                    else
                    {
                        postingListFullData += (vectorID + vector);
                    }
                }
                if (m_enablePostingListRearrange)
                {
                    return vectors + vectorIDs;
                }
                return postingListFullData;
            }

            bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt) {
                std::string outputFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                if (outputFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Error, "Output file can't be empty!\n");
                    return false;
                }

                int numThreads = p_opt.m_iSSDNumberOfThreads;
                int candidateNum = p_opt.m_internalResultNum;
                // get headVectorIDs
                std::unordered_set<SizeType> headVectorIDS;
                if (p_opt.m_headIDFile.empty()) {
                    LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                    return false;
                }

                {
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize((p_opt.m_indexDirectory + FolderSep +  p_opt.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                        LOG(Helper::LogLevel::LL_Error, "failed open VectorIDTranslate: %s\n", p_opt.m_headIDFile.c_str());
                        return false;
                    }

                    std::uint64_t vid;
                    while (ptr->ReadBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) == sizeof(vid))
                    {
                        headVectorIDS.insert(static_cast<SizeType>(vid));
                    }
                    LOG(Helper::LogLevel::LL_Info, "Loaded %u Vector IDs\n", static_cast<uint32_t>(headVectorIDS.size()));
                }

                SizeType fullCount = 0;
                size_t vectorInfoSize = 0;
                {
                    auto fullVectors = p_reader->GetVectorSet();
                    fullCount = fullVectors->Count();
                    vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);
                }

                Selection selections(static_cast<size_t>(fullCount) * p_opt.m_replicaCount, p_opt.m_tmpdir);
                LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
                std::vector<std::atomic_int> replicaCount(fullCount);
                std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
                for (auto& pls : postingListSize) pls = 0;
                std::unordered_set<SizeType> emptySet;
                SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (p_opt.m_batches > 1) selections.SaveBatch();
                {
                    LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                    SizeType sampleSize = p_opt.m_samples;
                    std::vector<SizeType> samples(sampleSize, 0);
                    for (int i = 0; i < p_opt.m_batches; i++) {
                        SizeType start = i * batchSize;
                        SizeType end = min(start + batchSize, fullCount);
                        auto fullVectors = p_reader->GetVectorSet(start, end);
                        if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                        if (p_opt.m_batches > 1) {
                            selections.LoadBatch(static_cast<size_t>(start) * p_opt.m_replicaCount, static_cast<size_t>(end) * p_opt.m_replicaCount);
                            emptySet.clear();
                            for (auto vid : headVectorIDS) {
                                if (vid >= start && vid < end) emptySet.insert(vid - start);
                            }
                        }
                        else {
                            emptySet = headVectorIDS;
                        }

                        int sampleNum = 0;
                        for (int j = start; j < end && sampleNum < sampleSize; j++)
                        {
                            if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                        }

                        float acc = 0;
#pragma omp parallel for schedule(dynamic)
                        for (int j = 0; j < sampleNum; j++)
                        {
                            COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                        }
                        acc = acc / sampleNum;
                        LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%d,%d) loaded with %d vectors (%zu) HeadIndex acc @%d:%f.\n", i, start, end, fullVectors->Count(), selections.m_selections.size(), candidateNum, acc);

                        p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);
                        LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                    ++postingListSize[selections[vecOffset + resNum].node];
                                    selections[vecOffset + resNum].tonode = j;
                                    ++replicaCount[j];
                                }
                            }
                        }

                        if (p_opt.m_batches > 1) selections.SaveBatch();
                    }
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

                if (p_opt.m_batches > 1) selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount);

                // Sort results either in CPU or GPU
                VectorIndex::SortSelections(&selections.m_selections);

                auto t3 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

                int postingSizeLimit = INT_MAX;
                if (p_opt.m_postingPageLimit > 0)
                {
                    postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
                }

                LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);

                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0) continue;
                        ++replicaCountDist[replicaCount[i]];
                    }

                    LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < postingListSize.size(); ++i)
                {
                    if (postingListSize[i] <= postingSizeLimit) continue;

                    std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, Selection::g_edgeComparer) - selections.m_selections.begin();

                    for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                    {
                        int tonode = selections.m_selections[selectIdx + dropID].tonode;
                        --replicaCount[tonode];
                    }
                    postingListSize[i] = postingSizeLimit;
                }

                if (p_opt.m_outputEmptyReplicaID)
                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                        return false;
                    }
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0) continue;

                        ++replicaCountDist[replicaCount[i]];

                        if (replicaCount[i] < 2)
                        {
                            long long vid = i;
                            if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                                LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
                                return false;
                            }
                        }
                    }

                    LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                    }
                }

                auto t4 = std::chrono::high_resolution_clock::now();
                LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

                // number of posting lists per file
                size_t postingFileSize = (postingListSize.size() + p_opt.m_ssdIndexFileNum - 1) / p_opt.m_ssdIndexFileNum;
                std::vector<size_t> selectionsBatchOffset(p_opt.m_ssdIndexFileNum + 1, 0);
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    selectionsBatchOffset[i + 1] = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), (SizeType)curPostingListEnd, Selection::g_edgeComparer) - selections.m_selections.begin();
                }

                if (p_opt.m_ssdIndexFileNum > 1) selections.SaveBatch();

                auto fullVectors = p_reader->GetVectorSet();
                if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);
                
                // get compressed size of each posting list
                LOG(Helper::LogLevel::LL_Info, "EnableDeltaEncoding: %s\n", p_opt.m_enableDeltaEncoding ? "true" : "false");
                LOG(Helper::LogLevel::LL_Info, "EnableDataCompression: %s, ZstdCompressLevel: %d\n", p_opt.m_enableDataCompression ? "true" : "false", p_opt.m_zstdCompressLevel);
                std::vector<size_t> postingListBytes(headVectorIDS.size());
                m_pCompressor = std::make_unique<Compressor>(p_opt.m_zstdCompressLevel);
                if (p_opt.m_enableDataCompression)
                {
                    LOG(Helper::LogLevel::LL_Info, "Getting compressed size of each posting list...\n");

                    LOG(Helper::LogLevel::LL_Info, "Training dictionary...\n");
                    unsigned nbSamples = 0;
                    std::string samplesBuffer("");
                    std::vector<size_t> samplesSizes;
                    for (int i = 0; i < postingListSize.size(); i++) {
                        // do not compress if no data
                        if (postingListSize[i] == 0) {
                            postingListBytes[i] = 0;
                            continue;
                        }
                        ValueType* headVector = nullptr;
                        if (p_opt.m_enableDeltaEncoding)
                        {
                            headVector = (ValueType*)p_headIndex->GetSample(i);
                        }
                        std::string postingListFullData = GetPostingListFullData(
                            i, postingListSize[i], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);

                        samplesBuffer += postingListFullData;
                        samplesSizes.push_back(postingListFullData.size());
                        if (samplesBuffer.size() > 102400) break;
                    }
                    LOG(Helper::LogLevel::LL_Info, "Using the first %zu postingLists to train dictionary... \n", samplesSizes.size());
                    m_pCompressor->TrainDict(samplesBuffer, &samplesSizes[0], samplesSizes.size());
                    m_pCompressor->CreateCDict();
                    LOG(Helper::LogLevel::LL_Info, "Dictionary trained.\n");

                    // TODO: omp parallel
                    for (int i = 0; i < postingListSize.size(); i++) {
                        // do not compress if no data
                        if (postingListSize[i] == 0) {
                            postingListBytes[i] = 0;
                            continue;
                        }
                        ValueType* headVector = nullptr;
                        if (p_opt.m_enableDeltaEncoding)
                        {
                            headVector = (ValueType*)p_headIndex->GetSample(i);
                        }
                        std::string postingListFullData = GetPostingListFullData(
                            i, postingListSize[i], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);
                        size_t sizeToCompress = postingListSize[i] * vectorInfoSize;
                        if (sizeToCompress != postingListFullData.size()) {
                            LOG(Helper::LogLevel::LL_Error, "Size to compress NOT MATCH! PostingListFullData size: %zu sizeToCompress: %zu \n", postingListFullData.size(), sizeToCompress);
                        }
                        postingListBytes[i] = m_pCompressor->GetCompressedSize(postingListFullData, true);
                        if (i % 10000 == 0 || postingListBytes[i] > static_cast<uint64_t>(p_opt.m_postingPageLimit) * PageSize) {
                            LOG(Helper::LogLevel::LL_Info, "Posting list %d/%d, compressed size: %d, compression ratio: %.4f\n", i, postingListSize.size(), postingListBytes[i], postingListBytes[i] / float(sizeToCompress));
                        }
                    }
                    LOG(Helper::LogLevel::LL_Info, "Getted compressed size for all the %d posting lists.\n", postingListBytes.size());
                    LOG(Helper::LogLevel::LL_Info, "Mean compressed size: %.4f \n", std::accumulate(postingListBytes.begin(), postingListBytes.end(), 0.0) / postingListBytes.size());
                    LOG(Helper::LogLevel::LL_Info, "Mean compression ratio: %.4f \n", std::accumulate(postingListBytes.begin(), postingListBytes.end(), 0.0) / (std::accumulate(postingListSize.begin(), postingListSize.end(), 0.0) * vectorInfoSize));
                }
                else
                {
                    for (int i = 0; i < postingListSize.size(); i++) 
                    {
                        postingListBytes[i] = postingListSize[i] * vectorInfoSize;
                    }
                }
                
                // iterate over files
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    // postingFileSize: number of posting lists in the file
                    // postingListSize: number of vectors in the posting list, type vector<int>
                    size_t curPostingListOffSet = i * postingFileSize;
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    std::vector<int> curPostingListSizes(
                        postingListSize.begin() + curPostingListOffSet,
                        postingListSize.begin() + curPostingListEnd);
                    std::vector<size_t> curPostingListBytes;
                    curPostingListBytes.assign(
                        postingListBytes.begin() + curPostingListOffSet,
                        postingListBytes.begin() + curPostingListEnd);

                    std::unique_ptr<int[]> postPageNum;
                    std::unique_ptr<std::uint16_t[]> postPageOffset;
                    std::vector<int> postingOrderInIndex;
                    SelectPostingOffset(curPostingListBytes, postPageNum, postPageOffset, postingOrderInIndex);

                    // LoadBatch: select vectors for each posting list
                    if (p_opt.m_ssdIndexFileNum > 1) selections.LoadBatch(selectionsBatchOffset[i], selectionsBatchOffset[i + 1]);
                    // write one file
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
                        curPostingListOffSet);
                }

                auto t5 = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
                LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
             
                return true;
            }

            virtual bool CheckValidPosting(SizeType postingID)
            {
                bool oneContext = (m_indexFiles.size() == 1);
                int fileid = 0;
                ListInfo* listInfo;
                if (oneContext) {
                    listInfo = &(m_listInfos[0][postingID]);
                }
                else {
                    fileid = postingID / m_listPerFile;
                    listInfo = &(m_listInfos[fileid][postingID % m_listPerFile]);
                }

                return listInfo->listEleCount != 0;
            }

        private:
            struct ListInfo
            {
                std::size_t listTotalBytes = 0;
                
                int listEleCount = 0;

                std::uint16_t listPageCount = 0;

                std::uint64_t listOffset = 0;

                std::uint16_t pageOffset = 0;
            };

            int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, std::vector<ListInfo>& m_listInfos)
            {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                    exit(1);
                }
                m_pCompressor = std::make_unique<Compressor>(); // no need compress level to decompress

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
                    if (ptr->ReadBinary(sizeof(m_listInfos[i].listTotalBytes), reinterpret_cast<char*>(&(m_listInfos[i].listTotalBytes))) != sizeof(m_listInfos[i].listTotalBytes)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        exit(1);
                    }
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
                    // m_listInfos[i].listEleCount = min(m_listInfos[i].listEleCount, (min(static_cast<int>(m_listInfos[i].listPageCount), p_postingPageLimit) << PageSizeEx) / m_vectorInfoSize);
                    // m_listInfos[i].listPageCount = static_cast<std::uint16_t>(ceil((m_vectorInfoSize * m_listInfos[i].listEleCount + m_listInfos[i].pageOffset) * 1.0 / (1 << PageSizeEx)));
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

                if (ptr->ReadBinary(sizeof(m_enableDeltaEncoding), reinterpret_cast<char *>(&m_enableDeltaEncoding)) != sizeof(m_enableDeltaEncoding))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }
                if (ptr->ReadBinary(sizeof(m_enablePostingListRearrange), reinterpret_cast<char *>(&m_enablePostingListRearrange)) != sizeof(m_enablePostingListRearrange))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }
                if (ptr->ReadBinary(sizeof(m_enableDataCompression), reinterpret_cast<char *>(&m_enableDataCompression)) != sizeof(m_enableDataCompression))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }
                if (ptr->ReadBinary(sizeof(m_enableDictTraining), reinterpret_cast<char *>(&m_enableDictTraining)) != sizeof(m_enableDictTraining))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }

                if (m_enableDataCompression && m_enableDictTraining)
                {
                    size_t dictBufferSize;

                    if (ptr->ReadBinary(sizeof(size_t), reinterpret_cast<char*>(&dictBufferSize)) != sizeof(dictBufferSize)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        exit(1);
                    }
                    char* dictBuffer = new char[dictBufferSize];
                    if (ptr->ReadBinary(dictBufferSize, dictBuffer) != dictBufferSize) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                        exit(1);
                    }
                    m_pCompressor->SetDictBuffer(std::string(dictBuffer, dictBufferSize));
                    delete[] dictBuffer;
                    m_pCompressor->CreateDDict();
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
                            LOG(Helper::LogLevel::LL_Error, "Crossing extra pages\n");
                            exit(1);
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

                LOG(Helper::LogLevel::LL_Info, "TotalPageNumbers: %d, IndexSize: %llu\n", currPageNum, static_cast<uint64_t>(currPageNum) * PageSize + currOffset);
            }

            void OutputSSDIndexFile(const std::string& p_outputFile,
                bool m_enableDeltaEncoding,
                bool m_enablePostingListRearrange,
                bool m_enableDataCompression,
                bool m_enableDictTraining,
                size_t p_spacePerVector,
                const std::vector<int>& p_postingListSizes,
                const std::vector<size_t>& p_postingListBytes,
                std::shared_ptr<VectorIndex> p_headIndex,
                Selection& p_postingSelections,
                const std::unique_ptr<int[]>& p_postPageNum,
                const std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                const std::vector<int>& p_postingOrderInIndex,
                std::shared_ptr<VectorSet> p_fullVectors,
                size_t p_postingListOffset)
            {
                LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                auto t1 = std::chrono::high_resolution_clock::now();

                auto ptr = SPTAG::f_createIO();
                int retry = 3;
                // open file 
                while (retry > 0 && (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed open file %s, retrying...\n", p_outputFile.c_str());
                    retry--;
                }

                if (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed open file %s\n", p_outputFile.c_str());
                    exit(1);
                }
                // meta size of global info
                std::uint64_t listOffset = sizeof(int) * 4;
                // meta size of the posting lists
                listOffset += (sizeof(size_t) + sizeof(int) + sizeof(std::uint16_t) + sizeof(int) + sizeof(std::uint16_t)) * p_postingListSizes.size();

                listOffset += sizeof(bool) * 4;
                // compression dict
                if (m_enableDataCompression && m_enableDictTraining)
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

                // Number of posting lists
                int i32Val = static_cast<int>(p_postingListSizes.size());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }

                // Number of vectors
                i32Val = static_cast<int>(p_fullVectors->Count());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }

                // Vector dimension
                i32Val = static_cast<int>(p_fullVectors->Dimension());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }

                // Page offset of list content section
                i32Val = static_cast<int>(listOffset / PageSize);
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
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
                    // Total bytes of the posting list
                    if (ptr->WriteBinary(sizeof(postingListByte), reinterpret_cast<char*>(&postingListByte)) != sizeof(postingListByte)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                    // Page number of the posting list
                    if (ptr->WriteBinary(sizeof(pageNum), reinterpret_cast<char*>(&pageNum)) != sizeof(pageNum)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                    // Page offset
                    if (ptr->WriteBinary(sizeof(pageOffset), reinterpret_cast<char*>(&pageOffset)) != sizeof(pageOffset)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                    // Number of vectors in the posting list
                    if (ptr->WriteBinary(sizeof(listEleCount), reinterpret_cast<char*>(&listEleCount)) != sizeof(listEleCount)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                    // Page count of the posting list
                    if (ptr->WriteBinary(sizeof(listPageCount), reinterpret_cast<char*>(&listPageCount)) != sizeof(listPageCount)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                }
                // m_enableDeltaEncoding,
                if (ptr->WriteBinary(sizeof(m_enableDeltaEncoding), reinterpret_cast<char *>(&m_enableDeltaEncoding)) != sizeof(m_enableDeltaEncoding))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }
                if (ptr->WriteBinary(sizeof(m_enablePostingListRearrange), reinterpret_cast<char *>(&m_enablePostingListRearrange)) != sizeof(m_enablePostingListRearrange))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }
                if (ptr->WriteBinary(sizeof(m_enableDataCompression), reinterpret_cast<char *>(&m_enableDataCompression)) != sizeof(m_enableDataCompression))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }
                if (ptr->WriteBinary(sizeof(m_enableDictTraining), reinterpret_cast<char *>(&m_enableDictTraining)) != sizeof(m_enableDictTraining))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                    exit(1);
                }
                // compression dict
                if (m_enableDataCompression && m_enableDictTraining)
                {
                    std::string dictBuffer = m_pCompressor->GetDictBuffer();
                    // dict size
                    size_t dictBufferSize = dictBuffer.size();
                    if (ptr->WriteBinary(sizeof(size_t), reinterpret_cast<char *>(&dictBufferSize)) != sizeof(dictBufferSize))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                    // dict
                    if (ptr->WriteBinary(dictBuffer.size(), const_cast<char *>(dictBuffer.data())) != dictBuffer.size())
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                }

                // Write padding vals
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
                // iterate over all the posting lists
                for (auto id : p_postingOrderInIndex)
                {
                    std::uint64_t targetOffset = static_cast<uint64_t>(p_postPageNum[id]) * PageSize + p_postPageOffset[id];
                    if (targetOffset < listOffset)
                    {
                        LOG(Helper::LogLevel::LL_Info, "List offset not match, targetOffset < listOffset!\n");
                        exit(1);
                    }
                    // write padding vals before the posting list
                    if (targetOffset > listOffset)
                    {
                        if (targetOffset - listOffset > PageSize)
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

                    if (p_postingListSizes[id] == 0)
                    {
                        continue;
                    }
                    int postingListId = id + (int)p_postingListOffset;
                    // get posting list full content and write it at once
                    ValueType *headVector = nullptr;
                    if (m_enableDeltaEncoding)
                    {
                        headVector = (ValueType *)p_headIndex->GetSample(postingListId);
                    }
                    std::string postingListFullData = GetPostingListFullData(
                        postingListId, p_postingListSizes[id], p_postingSelections, p_fullVectors, m_enableDeltaEncoding, m_enablePostingListRearrange, headVector);
                    size_t postingListFullSize = p_postingListSizes[id] * p_spacePerVector;
                    if (postingListFullSize != postingListFullData.size())
                    {
                        LOG(Helper::LogLevel::LL_Error, "posting list full data size NOT MATCH! postingListFullData.size(): %zu postingListFullSize: %zu \n", postingListFullData.size(), postingListFullSize);
                        exit(1);
                    }
                    if (m_enableDataCompression)
                    {
                        std::string compressedData = m_pCompressor->Compress(postingListFullData, m_enableDictTraining);
                        size_t compressedSize = compressedData.size();
                        if (compressedSize != p_postingListBytes[id])
                        {
                            LOG(Helper::LogLevel::LL_Error, "Compressed size NOT MATCH! compressed size:%zu, pre-calculated compressed size:%zu\n", compressedSize, p_postingListBytes[id]);
                            exit(1);
                        }
                        if (ptr->WriteBinary(compressedSize, compressedData.data()) != compressedSize)
                        {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
                        }
                        listOffset += compressedSize;
                    }
                    else
                    {
                        if (ptr->WriteBinary(postingListFullSize, postingListFullData.data()) != postingListFullSize)
                        {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                            exit(1);
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
                        LOG(Helper::LogLevel::LL_Error, "Failed to write SSDIndex File!");
                        exit(1);
                    }
                }

                LOG(Helper::LogLevel::LL_Info, "Padded Size: %llu, final total size: %llu.\n", paddedSize, listOffset);

                LOG(Helper::LogLevel::LL_Info, "Output done...\n");
                auto t2 = std::chrono::high_resolution_clock::now();
                LOG(Helper::LogLevel::LL_Info, "Time to write results:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000);
            }

        private:
            
            std::string m_extraFullGraphFile;

            std::vector<std::vector<ListInfo>> m_listInfos;

            std::vector<std::shared_ptr<Helper::DiskIO>> m_indexFiles;
            std::unique_ptr<Compressor> m_pCompressor;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableDataCompression;
            bool m_enableDictTraining;

            int m_vectorInfoSize = 0;

            int m_totalListCount = 0;

            int m_listPerFile = 0;
        };
    } // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_EXTRASEARCHER_H_
