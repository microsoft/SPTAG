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
                } g_edgeComparer;

                struct Selection {
                    std::string m_tmpfile;
                    size_t m_totalsize;
                    size_t m_start;
                    size_t m_end;
                    std::vector<Edge> m_selections;
                    

                    Selection(size_t totalsize, std::string tmpdir) : m_tmpfile(tmpdir + FolderSep + "selection_tmp"), m_totalsize(totalsize), m_start(0), m_end(totalsize) { remove(m_tmpfile.c_str()); m_selections.resize(totalsize); }

                    void SaveBatch()
                    {
                        auto f_out = f_createIO();
                        if (f_out == nullptr || !f_out->Initialize(m_tmpfile.c_str(), std::ios::out | std::ios::binary | (fileexists(m_tmpfile.c_str())? std::ios::in : 0))) {
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
                    const std::vector<int>& p_postingListSizes,
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
                    const std::vector<int>& p_postingListSizes,
                    Selection& p_postingSelections,
                    const std::unique_ptr<int[]>& p_postPageNum,
                    const std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                    const std::vector<int>& p_postingOrderInIndex,
                    std::shared_ptr<VectorSet> p_fullVectors,
                    size_t p_postingListOffset)
                {
                    LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                    auto ptr = SPTAG::f_createIO();
                    int retry = 3;
                    while (retry > 0 && (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed open file %s\n", p_outputFile.c_str());
                        retry--;
                    }

                    if (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)) {
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

                        std::size_t selectIdx = p_postingSelections.lower_bound(id + (int)p_postingListOffset);
                        for (int j = 0; j < p_postingListSizes[id]; ++j)
                        {
                            if (p_postingSelections[selectIdx].node != id + (int)p_postingListOffset)
                            {
                                LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH! node:%d offset:%zu\n", id + (int)p_postingListOffset, selectIdx);
                                exit(1);
                            }

                            i32Val = p_postingSelections[selectIdx++].tonode;
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

                SPTAG::VectorValueType valueType = SPTAG::COMMON::DistanceUtils::Quantizer ? SPTAG::VectorValueType::UInt8 : COMMON_OPTS.m_valueType;
                std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_vectorType, COMMON_OPTS.m_vectorDelimiter));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(COMMON_OPTS.m_vectorPath))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                    exit(1);
                }

                SizeType fullCount = 0;
                size_t vectorInfoSize = 0;
                {
                    auto fullVectors = vectorReader->GetVectorSet();
                    fullCount = fullVectors->Count();
                    vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);
                }

                Selection selections(static_cast<size_t>(fullCount)* p_opts.m_replicaCount, p_opts.m_tmpdir);
                LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
                std::vector<std::atomic_int> replicaCount(fullCount);
                std::vector<std::atomic_int> postingListSize(headVectorIDS.size());
                for (auto& pls : postingListSize) pls = 0;
                std::unordered_set<int> emptySet;
                SizeType batchSize = (fullCount + p_opts.m_batches - 1) / p_opts.m_batches;

                if (p_opts.m_batches > 1) selections.SaveBatch();
                {
                    SearchDefault<ValueType> searcher;
                    LOG(Helper::LogLevel::LL_Info, "Start setup index...\n");
                    ByteArray myByteArray;
                    searcher.Setup(p_opts, myByteArray);

                    LOG(Helper::LogLevel::LL_Info, "Setup index finish, start setup hint...\n");
                    searcher.SetHint(numThreads, candidateNum, false, p_opts);

                    TimeUtils::StopW rngw;
                    LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                    SizeType sampleSize = p_opts.m_samples;
                    SizeType sampleK = candidateNum;
                    float sampleE = 1e-6f;
                    std::vector<SizeType> samples(sampleSize, 0);
                    std::vector<SizeType> recalls(sampleSize, 0);
                    for (int i = 0; i < p_opts.m_batches; i++) {
                        SizeType start = i * batchSize;
                        SizeType end = min(start + batchSize, fullCount);
                        auto fullVectors = vectorReader->GetVectorSet(start, end);
                        if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine) fullVectors->Normalize(p_opts.m_iNumberOfThreads);

                        if (p_opts.m_batches > 1) {
                            selections.LoadBatch(static_cast<size_t>(start)* p_opts.m_replicaCount, static_cast<size_t>(end)* p_opts.m_replicaCount);
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

#pragma omp parallel for schedule(dynamic)
                        for (int j = 0; j < sampleNum; j++)
                        {
                            COMMON::QueryResultSet<void> sampleANN(nullptr, sampleK);
                            COMMON::QueryResultSet<void> sampleTruth(nullptr, sampleK);
                            void* reconstructVector = nullptr;
                            if (SPTAG::COMMON::DistanceUtils::Quantizer)
                            {
                                reconstructVector = _mm_malloc(SPTAG::COMMON::DistanceUtils::Quantizer->ReconstructSize(), ALIGN);
                                SPTAG::COMMON::DistanceUtils::Quantizer->ReconstructVector((const uint8_t*) fullVectors->GetVector(samples[j]), reconstructVector);
                                sampleANN.SetTarget(reconstructVector);
                                sampleTruth.SetTarget(reconstructVector);
                            }
                            else 
                            {
                                sampleANN.SetTarget(fullVectors->GetVector(samples[j]));
                                sampleTruth.SetTarget(fullVectors->GetVector(samples[j]));
                            }

                            searcher.HeadIndex()->SearchIndex(sampleANN);
                            for (SizeType y = 0; y < searcher.HeadIndex()->GetNumSamples(); y++)
                            {
                                float dist = searcher.HeadIndex()->ComputeDistance(sampleTruth.GetQuantizedTarget(), searcher.HeadIndex()->GetSample(y));
                                sampleTruth.AddPoint(y, dist);
                            }
                            sampleTruth.SortResult();

                            recalls[j] = 0;
                            std::vector<bool> visited(sampleK, false);
                            for (SizeType y = 0; y < sampleK; y++) 
                            {
                                for (SizeType z = 0; z < sampleK; z++) 
                                {
                                    if (visited[z]) continue;

                                    if (fabs(sampleANN.GetResult(z)->Dist - sampleTruth.GetResult(y)->Dist) < sampleE) 
                                    {
                                        recalls[j]++;
                                        visited[z] = true;
                                        break;
                                    }
                               }
                            }
                            if (reconstructVector)
                            {
                                _mm_free(reconstructVector);
                            }
                        }
                        float acc = 0;
                        for (int j = 0; j < sampleNum; j++) acc += float(recalls[j]);
                        acc = acc / sampleNum / sampleK;

                        LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%d,%d) loaded with %d vectors (%zu) HeadIndex acc @%d:%f.\n", i, start, end, fullVectors->Count(), selections.m_selections.size(), sampleK, acc);
                        searcher.HeadIndex()->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opts.m_replicaCount, numThreads, p_opts.m_gpuSSDNumTrees, p_opts.m_gpuSSDLeafSize, p_opts.m_rngFactor, p_opts.m_numGPUs);

                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)p_opts.m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < p_opts.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                    ++postingListSize[selections[vecOffset + resNum].node];
                                    selections[vecOffset + resNum].tonode = j;
                                    //selections[vecOffset + resNum].order = (char)resNum;
                                    ++replicaCount[j];
                                }
                            }
                        }

                        if (p_opts.m_batches > 1) selections.SaveBatch();
                    }

                    double rngElapsedMinutes = rngw.getElapsedMin();
                    LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", rngElapsedMinutes);
                }

                if (p_opts.m_batches > 1) selections.LoadBatch(0, static_cast<size_t>(fullCount)* p_opts.m_replicaCount);
                std::sort(selections.m_selections.begin(), selections.m_selections.end(), g_edgeComparer);
                
                int postingSizeLimit = INT_MAX;
                if (p_opts.m_postingPageLimit > 0)
                {
                    postingSizeLimit = static_cast<int>(p_opts.m_postingPageLimit * c_pageSize / vectorInfoSize);
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
                
#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < postingListSize.size(); ++i)
                {
                    if (postingListSize[i] <= postingSizeLimit)
                    {
                        continue;
                    }

                    std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, g_edgeComparer) - selections.m_selections.begin();
                    /*
                    int deletenum = postingListSize[i] - postingSizeLimit;
                    for (char remove = p_opts.m_replicaCount - 1; deletenum > 0 && remove > 0; remove--)
                    {
                        for (int dropID = postingListSize[i] - 1; deletenum > 0 && dropID >= 0; --dropID)
                        {
                            if (selections.m_selections[selectIdx + dropID].order == remove) {
                                selections.m_selections[selectIdx + dropID].order = -1;
                                --replicaCount[selections.m_selections[selectIdx + dropID].tonode];
                                deletenum--;
                            }
                        }
                    }

                    for (int iid = 0; iid < postingSizeLimit + deletenum; iid++) {
                        if (selections.m_selections[selectIdx + iid].order < 0) {
                            for (int ij = iid + 1; ij < postingListSize[i]; ij++) {
                                if (selections.m_selections[selectIdx + ij].order >= 0) {
                                    std::swap(selections.m_selections[selectIdx + iid], selections.m_selections[selectIdx + ij]);
                                    break;
                                }
                            }
                        }
                    }
                    */
                    
                    for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                    {
                        int tonode = selections.m_selections[selectIdx + dropID].tonode;
                        --replicaCount[tonode];
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

                size_t postingFileSize = (postingListSize.size() + COMMON_OPTS.m_ssdIndexFileNum - 1) / COMMON_OPTS.m_ssdIndexFileNum;

                std::vector<size_t> selectionsBatchOffset(COMMON_OPTS.m_ssdIndexFileNum + 1, 0);
                for (int i = 0; i < COMMON_OPTS.m_ssdIndexFileNum; i++) {
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    selectionsBatchOffset[i + 1] = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), (SizeType)curPostingListEnd, g_edgeComparer) - selections.m_selections.begin();
                }

                if (COMMON_OPTS.m_ssdIndexFileNum > 1) selections.SaveBatch();

                auto fullVectors = vectorReader->GetVectorSet();
                if (COMMON_OPTS.m_distCalcMethod == DistCalcMethod::Cosine && !SPTAG::COMMON::DistanceUtils::Quantizer) fullVectors->Normalize(p_opts.m_iNumberOfThreads);
     
                for (int i = 0; i < COMMON_OPTS.m_ssdIndexFileNum; i++) {
                    size_t curPostingListOffSet = i * postingFileSize;
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    std::vector<int> curPostingListSizes(
                        postingListSize.begin() + curPostingListOffSet,
                        postingListSize.begin() + curPostingListEnd);

                    std::unique_ptr<int[]> postPageNum;
                    std::unique_ptr<std::uint16_t[]> postPageOffset;
                    std::vector<int> postingOrderInIndex;
                    SelectPostingOffset(vectorInfoSize, curPostingListSizes, postPageNum, postPageOffset, postingOrderInIndex);
                    
                    if (COMMON_OPTS.m_ssdIndexFileNum > 1) selections.LoadBatch(selectionsBatchOffset[i], selectionsBatchOffset[i + 1]);

                    OutputSSDIndexFile((i == 0)? outputFile : outputFile + "_" + std::to_string(i),
                        vectorInfoSize,
                        curPostingListSizes,
                        selections,
                        postPageNum,
                        postPageOffset,
                        postingOrderInIndex,
                        fullVectors,
                        curPostingListOffSet);
                }

                double elapsedMinutes = sw.getElapsedMin();
                LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);
            }
        }
    }
}
