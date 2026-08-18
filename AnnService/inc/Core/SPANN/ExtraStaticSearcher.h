// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASTATICSEARCHER_H_
#define _SPTAG_SPANN_EXTRASTATICSEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/SPANN/HybridCandidateSelector.h"
#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Core/SPANN/HybridRoutingStats.h"
#include "inc/Core/SPANN/LimitedTagSupport.h"
#include "Compressor.h"
#include "PipePQ.h"

#include <atomic>
#include <map>
#include <algorithm>
#include <chrono>
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

            Selection(size_t totalsize,
                      std::string tmpdir,
                      const std::string& tmpfile = "selection_tmp")
                : m_tmpfile(tmpdir + FolderSep + tmpfile),
                  m_totalsize(totalsize),
                  m_start(0),
                  m_end(totalsize)
            {
                remove(m_tmpfile.c_str());
                m_selections.resize(totalsize);
            }

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
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", this->GetListOrdinal(listInfo), err.what());\
                return;\
            }\
            if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", this->GetListOrdinal(listInfo), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
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
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PostingList %d decompressed size not match! %zu, %d, \n", this->GetListOrdinal(listInfo), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
                }\
             }\
            catch (std::runtime_error& err) {\
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Decompress postingList %d  failed! %s, \n", this->GetListOrdinal(listInfo), err.what());\
            }\
        }\
}\

#define ProcessPosting() \
        { \
        bool postingMatched = false; \
        bool postingContributedUnique = false; \
        for (int staticScanRange = 0; staticScanRange < 2; ++staticScanRange) { \
        int staticScanBegin = 0; \
        int staticScanEnd = 0; \
        if (!this->GetStaticScanRange( \
                p_exWorkSpace, staticPostingSlot, listInfo, \
                staticScanRange, staticScanBegin, staticScanEnd)) continue; \
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
        } \
        if (postingMatched) ++p_exWorkSpace->m_postingProbeStats.m_matchedPostings; \
        if (postingContributedUnique) ++p_exWorkSpace->m_postingProbeStats.m_uniqueMatchedPostings; \
        } \

#define ProcessPostingOffset() \
        while (this->NormalizeStaticScanOffset( \
                p_exWorkSpace, p_exWorkSpace->m_pi, listInfo, p_exWorkSpace->m_offset)) { \
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
        if (!this->NormalizeStaticScanOffset( \
                p_exWorkSpace, p_exWorkSpace->m_pi, listInfo, p_exWorkSpace->m_offset)) { \
            p_exWorkSpace->m_pi++; \
            if (p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) { \
                SizeType nextPostingID = p_exWorkSpace->m_postingIDs[p_exWorkSpace->m_pi]; \
                ListInfo* nextListInfo = this->GetPostingListInfo(p_exWorkSpace, nextPostingID); \
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

            virtual bool HasHybridPurePostings() const override
            {
                return m_hasHybridPurePostings;
            }

            virtual int GetPostingCount() const override
            {
                return m_totalListCount;
            }

            virtual double GetPostingAvgRecords(bool p_useHybrid = false) const override
            {
                if (p_useHybrid) return m_hasHybridPurePostings ? m_hybridAvgRecordsPerList : -1.0;
                return m_avgRecordsPerList;
            }

            virtual double GetPostingAvgPages(bool p_useHybrid = false) const override
            {
                if (p_useHybrid) return m_hasHybridPurePostings ? m_hybridAvgPagesPerList : -1.0;
                return m_avgPagesPerList;
            }

            virtual double GetPostingAvgBytes(bool p_useHybrid = false) const override
            {
                if (p_useHybrid) return m_hasHybridPurePostings ? m_hybridAvgBytesPerList : -1.0;
                return m_avgBytesPerList;
            }

            virtual int GetPostingBufferBytes(
                bool p_useHybrid = false) const override
            {
                const int postingPages = p_useHybrid
                    ? m_hybridMaxListPageCount
                    : m_staticMaxListPageCount;
                return (std::max)(
                    1, postingPages + 1) << PageSizeEx;
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

            void SetHybridGenerationFingerprint(
                std::uint64_t p_generationFingerprint) override
            {
                m_hybridGenerationFingerprint =
                    p_generationFingerprint;
            }

            void SetLimitedTagGenerationFingerprint(
                std::uint64_t p_generationFingerprint) override
            {
                m_hybridGenerationFingerprint =
                    p_generationFingerprint;
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

            int StaticWorkspaceBufferBytes() const
            {
                const int configuredBuildPages = m_opt == nullptr ? 0 : m_opt->m_postingPageLimit;
                const int postingPages = (std::max)(
                    (std::max)(m_staticMaxListPageCount, m_hybridMaxListPageCount),
                    configuredBuildPages);
                return (std::max)(1, postingPages + 1) << PageSizeEx;
            }

            bool ComputeHybridRouteLayout(
                const Selection& p_selections,
                const std::vector<int>& p_postingSizes,
                const std::vector<SizeType>& p_headVectorIDs,
                const std::vector<int>& p_categoricalColumns,
                SizeType p_fullCount,
                HybridRouteLayout& p_layout,
                const std::vector<int>*
                    p_physicalPostingSizes = nullptr)
            {
                p_layout = HybridRouteLayout();
                if (p_categoricalColumns.size() > 16 ||
                    p_postingSizes.size() !=
                        p_headVectorIDs.size() ||
                    (p_physicalPostingSizes != nullptr &&
                     p_physicalPostingSizes->size() !=
                         p_postingSizes.size()) ||
                    p_fullCount <= 0) {
                    return false;
                }
                const size_t maskCount =
                    static_cast<size_t>(1) <<
                    p_categoricalColumns.size();
                std::vector<double> baseSelectivity(
                    maskCount, 1.0);
                const auto mix64 = [](std::uint64_t value) {
                    value += 0x9e3779b97f4a7c15ULL;
                    value =
                        (value ^ (value >> 30)) *
                        0xbf58476d1ce4e5b9ULL;
                    value =
                        (value ^ (value >> 27)) *
                        0x94d049bb133111ebULL;
                    return value ^ (value >> 31);
                };
                const std::uint64_t sampleCount =
                    (std::min<std::uint64_t>)(
                        1ULL << 20,
                        (std::max<std::uint64_t>)(
                            4096,
                            static_cast<std::uint64_t>(
                                p_fullCount) * 32ULL));
                std::vector<std::uint64_t> pairEqual(
                    maskCount, 0);
                for (std::uint64_t sample = 0;
                     sample < sampleCount; ++sample) {
                    const SizeType left = static_cast<SizeType>(
                        mix64(sample) %
                        static_cast<std::uint64_t>(
                            p_fullCount));
                    const SizeType right = static_cast<SizeType>(
                        mix64(sample ^
                              0xd1b54a32d192ed03ULL) %
                        static_cast<std::uint64_t>(
                            p_fullCount));
                    const auto* leftAttributes =
                        m_staticBuildTags.data() +
                        static_cast<size_t>(left) *
                            static_cast<size_t>(
                                m_staticNumTagsPerVec);
                    const auto* rightAttributes =
                        m_staticBuildTags.data() +
                        static_cast<size_t>(right) *
                            static_cast<size_t>(
                                m_staticNumTagsPerVec);
                    size_t equalMask = 0;
                    for (size_t bit = 0;
                         bit < p_categoricalColumns.size();
                         ++bit) {
                        const int column =
                            p_categoricalColumns[bit];
                        if (leftAttributes[column] ==
                            rightAttributes[column]) {
                            equalMask |=
                                static_cast<size_t>(1)
                                << bit;
                        }
                    }
                    ++pairEqual[equalMask];
                }
                std::vector<std::uint64_t> pairMatched =
                    pairEqual;
                for (size_t bit = 0;
                     bit < p_categoricalColumns.size();
                     ++bit) {
                    const size_t bitValue =
                        static_cast<size_t>(1) << bit;
                    for (size_t mask = 0;
                         mask < maskCount; ++mask) {
                        if ((mask & bitValue) == 0) {
                            pairMatched[mask] +=
                                pairMatched[mask | bitValue];
                        }
                    }
                }
                for (size_t mask = 1;
                     mask < maskCount; ++mask) {
                    baseSelectivity[mask] =
                        (std::max)(
                            1.0 / static_cast<double>(
                                sampleCount),
                            static_cast<double>(
                                pairMatched[mask]) /
                                static_cast<double>(
                                    sampleCount));
                }

                std::vector<std::uint64_t> exactEqual(
                    maskCount, 0);
                std::uint64_t totalRecords = 0;
                const bool countUniqueExactly =
                    p_fullCount <= 10000000;
                std::unordered_set<SizeType>
                    retainedVectors;
                constexpr size_t kHLLBits = 16;
                constexpr size_t kHLLRegisters =
                    static_cast<size_t>(1) << kHLLBits;
                std::vector<std::uint8_t> hllRegisters;
                if (countUniqueExactly) {
                    retainedVectors.reserve(
                        static_cast<size_t>(p_fullCount));
                } else {
                    hllRegisters.assign(kHLLRegisters, 0);
                }
                for (SizeType head = 0;
                     head <
                         static_cast<SizeType>(
                             p_postingSizes.size());
                     ++head) {
                    const int postingEnd =
                        p_postingSizes[
                            static_cast<size_t>(head)];
                    if (postingEnd < 0) {
                        return false;
                    }
                    if (postingEnd == 0) {
                        continue;
                    }
                    const SizeType headVectorID =
                        p_headVectorIDs[
                            static_cast<size_t>(head)];
                    if (headVectorID < 0 ||
                        headVectorID >= p_fullCount) {
                        return false;
                    }
                    const auto begin = std::lower_bound(
                        p_selections.m_selections.begin(),
                        p_selections.m_selections.end(),
                        head, Selection::g_edgeComparer);
                    if (begin ==
                            p_selections.m_selections.end() ||
                        begin->node != head ||
                        static_cast<size_t>(
                            p_selections.m_selections.end() -
                            begin) <
                            static_cast<size_t>(
                                postingEnd)) {
                        return false;
                    }
                    const auto* headAttributes =
                        m_staticBuildTags.data() +
                        static_cast<size_t>(
                            headVectorID) *
                            static_cast<size_t>(
                                m_staticNumTagsPerVec);
                    for (int record = 0;
                         record < postingEnd; ++record) {
                        const SizeType vectorID =
                            begin[record].tonode;
                        if (vectorID < 0 ||
                            vectorID >= p_fullCount) {
                            return false;
                        }
                        if (countUniqueExactly) {
                            retainedVectors.insert(vectorID);
                        } else {
                            const std::uint64_t hash =
                                mix64(static_cast<
                                      std::uint64_t>(
                                    vectorID));
                            const size_t bucket =
                                static_cast<size_t>(
                                    hash &
                                    (kHLLRegisters - 1));
                            std::uint64_t suffix =
                                hash >> kHLLBits;
                            std::uint8_t rank = 1;
                            while ((suffix & 1ULL) == 0 &&
                                   rank <=
                                       64 - kHLLBits) {
                                ++rank;
                                suffix >>= 1;
                            }
                            hllRegisters[bucket] =
                                (std::max)(
                                    hllRegisters[bucket],
                                    rank);
                        }
                        const auto* attributes =
                            m_staticBuildTags.data() +
                            static_cast<size_t>(
                                vectorID) *
                                static_cast<size_t>(
                                    m_staticNumTagsPerVec);
                        size_t equalMask = 0;
                        for (size_t bit = 0;
                             bit <
                                 p_categoricalColumns.size();
                             ++bit) {
                            const int column =
                                p_categoricalColumns[bit];
                            if (attributes[column] ==
                                headAttributes[column]) {
                                equalMask |=
                                    static_cast<size_t>(1)
                                    << bit;
                            }
                        }
                        ++exactEqual[equalMask];
                        ++totalRecords;
                    }
                }
                if (totalRecords == 0) return false;
                std::vector<std::uint64_t> matched =
                    exactEqual;
                for (size_t bit = 0;
                     bit < p_categoricalColumns.size();
                     ++bit) {
                    const size_t bitValue =
                        static_cast<size_t>(1) << bit;
                    for (size_t mask = 0;
                         mask < maskCount; ++mask) {
                        if ((mask & bitValue) == 0) {
                            matched[mask] +=
                                matched[mask | bitValue];
                        }
                    }
                }

                p_layout.m_enrichmentByMask.assign(
                    maskCount, 1.0);
                for (size_t mask = 1;
                     mask < maskCount; ++mask) {
                    const double matchRate =
                        static_cast<double>(
                            matched[mask]) /
                        static_cast<double>(
                            totalRecords);
                    p_layout.m_enrichmentByMask[mask] =
                        (std::max)(
                            1e-9,
                            matchRate /
                                (std::max)(
                                    1e-12,
                                    baseSelectivity[mask]));
                }
                p_layout.m_layout.m_averageRecords =
                    static_cast<double>(totalRecords) /
                    static_cast<double>(
                        p_postingSizes.size());
                p_layout.m_layout.m_averageBytes =
                    p_layout.m_layout.m_averageRecords *
                    static_cast<double>(m_vectorInfoSize);
                double uniqueVectors = static_cast<double>(
                    retainedVectors.size());
                if (!countUniqueExactly) {
                    double inverseSum = 0.0;
                    size_t zeroRegisters = 0;
                    for (const std::uint8_t rank :
                         hllRegisters) {
                        inverseSum +=
                            std::ldexp(1.0, -rank);
                        if (rank == 0) ++zeroRegisters;
                    }
                    const double registers =
                        static_cast<double>(
                            kHLLRegisters);
                    const double alpha =
                        0.7213 /
                        (1.0 + 1.079 / registers);
                    uniqueVectors =
                        alpha * registers * registers /
                        inverseSum;
                    if (zeroRegisters > 0 &&
                        uniqueVectors <=
                            2.5 * registers) {
                        uniqueVectors =
                            registers *
                            std::log(
                                registers /
                                static_cast<double>(
                                    zeroRegisters));
                    }
                    uniqueVectors = std::clamp(
                        uniqueVectors, 1.0,
                        static_cast<double>(
                            p_fullCount));
                }
                p_layout.m_layout.m_uniqueRatio =
                    (std::min)(
                        1.0,
                        uniqueVectors /
                            static_cast<double>(
                                totalRecords));

                std::vector<size_t> postingBytes(
                    p_postingSizes.size());
                for (size_t head = 0;
                     head < p_postingSizes.size(); ++head) {
                    const int physicalCount =
                        p_physicalPostingSizes ==
                                nullptr
                        ? p_postingSizes[head]
                        : (*p_physicalPostingSizes)[
                              head];
                    if (physicalCount <
                        p_postingSizes[head]) {
                        return false;
                    }
                    postingBytes[head] =
                        static_cast<size_t>(
                            (std::max)(0, physicalCount)) *
                        static_cast<size_t>(
                            m_vectorInfoSize);
                }
                std::unique_ptr<int[]> pageCount;
                std::unique_ptr<std::uint16_t[]>
                    pageOffset;
                std::vector<int> postingOrder;
                SelectPostingOffset(
                    postingBytes, pageCount, pageOffset,
                    postingOrder);
                double totalPages = 0.0;
                for (size_t head = 0;
                     head < p_postingSizes.size(); ++head) {
                    const size_t physicalPages =
                        postingBytes[head] == 0
                        ? 0
                        : (static_cast<size_t>(
                               pageOffset[head]) +
                           postingBytes[head] +
                           PageSize - 1) /
                              PageSize;
                    std::vector<std::uint8_t>
                        selectedPages(physicalPages, 0);
                    for (int record = 0;
                         record < p_postingSizes[head];
                         ++record) {
                        const size_t beginByte =
                            static_cast<size_t>(
                                pageOffset[head]) +
                            static_cast<size_t>(
                                record) *
                                static_cast<size_t>(
                                    m_vectorInfoSize);
                        const size_t endByte =
                            beginByte +
                            static_cast<size_t>(
                                m_vectorInfoSize) -
                            1;
                        for (size_t page =
                                 beginByte / PageSize;
                             page <=
                             endByte / PageSize;
                             ++page) {
                            selectedPages[page] = 1;
                        }
                    }
                    totalPages += static_cast<double>(
                        std::count(
                            selectedPages.begin(),
                            selectedPages.end(),
                            static_cast<std::uint8_t>(
                                1)));
                }
                p_layout.m_layout.m_averagePages =
                    totalPages /
                    static_cast<double>(
                        p_postingSizes.size());
                return true;
            }

            bool BuildHybridPureSelections(
                const std::vector<std::vector<SizeType>>& p_nodeVectors,
                Selection& p_selections,
                std::vector<std::atomic_int>& p_postingListSize,
                std::shared_ptr<VectorSet> p_fullVectors,
                SizeType p_fullCount,
                SizeType p_globalHeadCount,
                int p_postingSizeLimit,
                const Options& p_opt,
                std::vector<SizeType>& p_globalHeadVectorIDs,
                std::vector<int>& p_categoricalColumns)
            {
                if (p_opt.m_batches != 1 ||
                    p_nodeVectors.size() != 1 ||
                    p_fullVectors == nullptr ||
                    m_staticHeadBundleLocalToGlobalHIDs == nullptr ||
                    m_staticHeadBundleNodeHeadVectorIDs == nullptr ||
                    m_staticHeadBundleIndexes.size() != p_nodeVectors.size() ||
                    m_staticHeadBundleLocalToGlobalHIDs->size() !=
                        p_nodeVectors.size() ||
                    m_staticHeadBundleNodeHeadVectorIDs->size() !=
                        p_nodeVectors.size()) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid static placement requires one build batch and "
                        "one global head/posting node.\n");
                    return false;
                }
                if (!m_staticHasMetadata ||
                    m_staticNumTagsPerVec <= 0 ||
                    m_staticBuildTags.size() !=
                        static_cast<size_t>(p_fullCount) *
                            static_cast<size_t>(
                                m_staticNumTagsPerVec)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid static posting construction requires a complete "
                        "STM1 attribute table.\n");
                    return false;
                }

                HybridDistanceConfig distance;
                std::string error;
                if (!HybridDistanceConfig::Parse(
                        p_opt.m_hybridCategoricalCols,
                        p_opt.m_hybridCategoricalWeights,
                        p_opt.m_hybridNumericCols,
                        p_opt.m_hybridNumericWeights,
                        m_staticNumTagsPerVec,
                        p_opt.m_hybridVectorWeight,
                        distance,
                        error)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Invalid hybrid posting distance: %s\n",
                        error.c_str());
                    return false;
                }

                std::vector<std::vector<std::uint32_t>>
                    nodeHeadAttributes(p_nodeVectors.size());
                std::vector<std::unique_ptr<
                    HybridCandidateSelector<ValueType>>>
                    selectors;
                selectors.reserve(p_nodeVectors.size());
                std::vector<uint8_t> seenGlobalHeads(
                    static_cast<size_t>(p_globalHeadCount), 0);
                p_globalHeadVectorIDs.assign(
                    static_cast<size_t>(p_globalHeadCount), -1);
                for (size_t node = 0;
                     node < p_nodeVectors.size(); ++node) {
                    const auto& index =
                        m_staticHeadBundleIndexes[node];
                    const auto& localToGlobal =
                        (*m_staticHeadBundleLocalToGlobalHIDs)[node];
                    const auto& headVectorIDs =
                        (*m_staticHeadBundleNodeHeadVectorIDs)[node];
                    if (index == nullptr ||
                        index->m_pQuantizer != nullptr ||
                        index->GetNumSamples() !=
                            static_cast<SizeType>(
                                localToGlobal.size()) ||
                        localToGlobal.size() !=
                            headVectorIDs.size() ||
                        !std::is_sorted(
                            headVectorIDs.begin(),
                            headVectorIDs.end())) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Hybrid posting node %zu has inconsistent bundle "
                            "head metadata.\n",
                            node);
                        return false;
                    }
                    auto& attributes = nodeHeadAttributes[node];
                    attributes.resize(
                        headVectorIDs.size() *
                        static_cast<size_t>(
                            m_staticNumTagsPerVec));
                    for (size_t local = 0;
                         local < headVectorIDs.size(); ++local) {
                        const SizeType vectorID =
                            headVectorIDs[local];
                        const SizeType globalHead =
                            localToGlobal[local];
                        if (vectorID < 0 ||
                            vectorID >= p_fullCount ||
                            globalHead < 0 ||
                            globalHead >= p_globalHeadCount ||
                            seenGlobalHeads[
                                static_cast<size_t>(
                                    globalHead)] != 0) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Hybrid posting node %zu contains an invalid "
                                "or duplicate head mapping.\n",
                                node);
                            return false;
                        }
                        seenGlobalHeads[
                            static_cast<size_t>(
                                globalHead)] = 1;
                        p_globalHeadVectorIDs[
                            static_cast<size_t>(
                                globalHead)] = vectorID;
                        std::copy_n(
                            m_staticBuildTags.data() +
                                static_cast<size_t>(vectorID) *
                                    static_cast<size_t>(
                                        m_staticNumTagsPerVec),
                            m_staticNumTagsPerVec,
                            attributes.data() +
                                local *
                                    static_cast<size_t>(
                                        m_staticNumTagsPerVec));
                    }
                    selectors.emplace_back(
                        new HybridCandidateSelector<ValueType>(
                            index.get(),
                            attributes.data(),
                            static_cast<SizeType>(
                                headVectorIDs.size()),
                            m_staticNumTagsPerVec,
                            &distance));
                    if (!selectors.back()->Build(error)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Cannot build hybrid posting candidates for node "
                            "%zu: %s\n",
                            node, error.c_str());
                        return false;
                    }
                }
                if (std::find(
                        seenGlobalHeads.begin(),
                        seenGlobalHeads.end(),
                        static_cast<uint8_t>(0)) !=
                    seenGlobalHeads.end()) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid posting bundle mappings do not cover all "
                        "global heads.\n");
                    return false;
                }
                const std::uint64_t generation =
                    m_hybridGenerationFingerprint;
                if (generation == 0) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid posting build requires a non-zero "
                        "build generation.\n");
                    return false;
                }

                p_categoricalColumns.clear();
                p_categoricalColumns.reserve(
                    distance.m_categorical.size());
                for (const auto& column :
                     distance.m_categorical) {
                    p_categoricalColumns.push_back(
                        column.m_column);
                }

                std::vector<size_t> nodeOffsets(
                    p_nodeVectors.size() + 1, 0);
                for (size_t node = 0;
                     node < p_nodeVectors.size(); ++node) {
                    if (p_nodeVectors[node].size() >
                        (std::numeric_limits<size_t>::max)() -
                            nodeOffsets[node]) {
                        return false;
                    }
                    nodeOffsets[node + 1] =
                        nodeOffsets[node] +
                        p_nodeVectors[node].size();
                }
                const size_t assignmentCountTotal =
                    nodeOffsets.back();
                if (assignmentCountTotal !=
                    static_cast<size_t>(p_fullCount)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid posting placement covers %zu/%d vectors.\n",
                        assignmentCountTotal, p_fullCount);
                    return false;
                }
                if (p_opt.m_replicaCount <= 0 ||
                    assignmentCountTotal >
                        (std::numeric_limits<size_t>::max)() /
                            static_cast<size_t>(
                                p_opt.m_replicaCount)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid posting selection size overflows.\n");
                    return false;
                }

                Selection hybridSelections(
                    assignmentCountTotal *
                        static_cast<size_t>(
                            p_opt.m_replicaCount),
                    p_opt.m_tmpdir,
                    "hybrid_selection_tmp");
                std::vector<std::atomic_int> postingListSize(
                    static_cast<size_t>(p_globalHeadCount));
                for (auto& size : postingListSize) size = 0;
                std::atomic_size_t sent(0);
                std::atomic_bool failed(false);
                std::atomic<std::uint64_t> checkedLeaves(0);
                std::mutex errorLock;
                std::string firstError;
                std::vector<std::thread> threads;
                threads.reserve(
                    p_opt.m_iSSDNumberOfThreads);
                for (int thread = 0;
                     thread < p_opt.m_iSSDNumberOfThreads;
                     ++thread) {
                    threads.emplace_back([&]() {
                        std::vector<HybridScoredCandidate>
                            selected;
                        while (!failed.load()) {
                            const size_t assignment =
                                sent.fetch_add(1);
                            if (assignment >=
                                assignmentCountTotal) {
                                return;
                            }
                            const auto nodePosition =
                                std::upper_bound(
                                    nodeOffsets.begin() + 1,
                                    nodeOffsets.end(),
                                    assignment);
                            const int node =
                                static_cast<int>(
                                    nodePosition -
                                    nodeOffsets.begin() - 1);
                            const SizeType vectorID =
                                p_nodeVectors[
                                    static_cast<size_t>(
                                        node)]
                                    [assignment -
                                     nodeOffsets[
                                         static_cast<size_t>(
                                             node)]];
                            const auto& headVectorIDs =
                                (*m_staticHeadBundleNodeHeadVectorIDs)
                                    [static_cast<size_t>(node)];
                            const auto& localToGlobal =
                                (*m_staticHeadBundleLocalToGlobalHIDs)
                                    [static_cast<size_t>(node)];
                            const auto head =
                                std::lower_bound(
                                    headVectorIDs.begin(),
                                    headVectorIDs.end(),
                                    vectorID);
                            const SizeType self =
                                head != headVectorIDs.end() &&
                                    *head == vectorID
                                ? static_cast<SizeType>(
                                      head -
                                      headVectorIDs.begin())
                                : -1;
                            const size_t selectionOffset =
                                assignment *
                                static_cast<size_t>(
                                    p_opt.m_replicaCount);
                            if (self >= 0) {
                                if (!p_opt.m_excludehead) {
                                    Edge& edge =
                                        hybridSelections
                                            .m_selections[
                                                selectionOffset];
                                    edge.node =
                                        localToGlobal[
                                            static_cast<size_t>(
                                                self)];
                                    edge.tonode = vectorID;
                                    edge.distance = 0.0f;
                                    ++postingListSize[
                                        static_cast<size_t>(
                                            edge.node)];
                                }
                                continue;
                            }

                            std::string selectError;
                            std::uint64_t localChecked = 0;
                            if (!selectors[
                                     static_cast<size_t>(
                                         node)]
                                     ->Select(
                                         static_cast<
                                             const ValueType*>(
                                             p_fullVectors
                                                 ->GetVector(
                                                     vectorID)),
                                         m_staticBuildTags
                                                 .data() +
                                             static_cast<size_t>(
                                                 vectorID) *
                                                 static_cast<
                                                     size_t>(
                                                     m_staticNumTagsPerVec),
                                         vectorID,
                                         -1,
                                         p_opt
                                             .m_hybridCandidateCount,
                                         p_opt.m_replicaCount,
                                         selected,
                                         &localChecked,
                                         selectError)) {
                                failed.store(true);
                                std::lock_guard<std::mutex>
                                    guard(errorLock);
                                if (firstError.empty()) {
                                    firstError =
                                        selectError;
                                }
                                return;
                            }
                            checkedLeaves.fetch_add(
                                localChecked);
                            for (size_t result = 0;
                                 result < selected.size();
                                 ++result) {
                                Edge& edge =
                                    hybridSelections
                                        .m_selections[
                                            selectionOffset +
                                            result];
                                edge.node =
                                    localToGlobal[
                                        static_cast<size_t>(
                                            selected[result]
                                                .m_head)];
                                edge.tonode = vectorID;
                                edge.distance =
                                    selected[result]
                                        .m_distance;
                                ++postingListSize[
                                    static_cast<size_t>(
                                        edge.node)];
                            }
                        }
                    });
                }
                for (auto& thread : threads) thread.join();
                if (failed.load()) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid posting candidate search failed: %s\n",
                        firstError.c_str());
                    return false;
                }

                VectorIndex::SortSelections(
                    &hybridSelections.m_selections);
                if (p_postingSizeLimit < INT_MAX) {
                    for (SizeType head = 0;
                         head < p_globalHeadCount; ++head) {
                        if (postingListSize[
                                static_cast<size_t>(head)] >
                            p_postingSizeLimit) {
                            postingListSize[
                                static_cast<size_t>(head)] =
                                p_postingSizeLimit;
                        }
                    }
                }
                std::vector<int> postingSizes(
                    static_cast<size_t>(p_globalHeadCount));
                std::uint64_t assignmentCount = 0;
                for (SizeType head = 0;
                     head < p_globalHeadCount; ++head) {
                    postingSizes[
                        static_cast<size_t>(head)] =
                        postingListSize[
                            static_cast<size_t>(head)]
                            .load();
                    assignmentCount += static_cast<
                        std::uint64_t>(
                        postingSizes[
                            static_cast<size_t>(head)]);
                }
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Hybrid static placement selected %llu assignments "
                    "(checkedLeaves=%llu); replacing the primary pure "
                    "prefix.\n",
                    static_cast<unsigned long long>(
                        assignmentCount),
                    static_cast<unsigned long long>(
                        checkedLeaves.load()));
                p_selections.m_selections.swap(
                    hybridSelections.m_selections);
                p_selections.m_start = 0;
                p_selections.m_end =
                    p_selections.m_selections.size();
                p_selections.m_totalsize =
                    p_selections.m_end;
                p_postingListSize.swap(
                    postingListSize);
                return true;
            }

            bool BuildLimitedTagPureSelections(
                Selection& p_selections,
                std::vector<std::atomic_int>& p_postingListSize,
                const std::unordered_map<SizeType, SizeType>& p_headVectorIDs,
                std::shared_ptr<VectorSet> p_fullVectors,
                std::shared_ptr<VectorIndex> p_headIndex,
                SizeType p_fullCount,
                int p_postingSizeLimit,
                const Options& p_opt,
                LimitedTagSupport& p_support)
            {
                struct Vote
                {
                    SizeType m_head = MaxSize;
                    std::uint32_t m_tag =
                        LimitedTagSupport::EmptyTag;
                    float m_distance = MaxDist;
                };
                struct SupportChoice
                {
                    std::uint32_t m_tag =
                        LimitedTagSupport::EmptyTag;
                    float m_distance = MaxDist;
                    bool m_isOwnTag = false;
                };

                const SizeType headCount =
                    p_headIndex == nullptr
                        ? 0
                        : p_headIndex->GetNumSamples();
                (void)p_postingSizeLimit;
                if (p_fullVectors == nullptr ||
                    p_headIndex == nullptr ||
                    p_headIndex->m_pQuantizer != nullptr ||
                    headCount <= 0 ||
                    p_fullCount != p_fullVectors->Count() ||
                    !m_staticHasMetadata ||
                    m_staticNumTagsPerVec != 1 ||
                    m_staticBuildTags.size() !=
                        static_cast<size_t>(p_fullCount) ||
                    p_opt.m_batches != 1 ||
                    p_opt.m_limitedTagSlotsPerHead != 4 ||
                    p_opt.m_limitedTagVoteHeadCount <= 0 ||
                    p_opt.m_limitedTagMinHeadCount <= 0 ||
                    p_opt.m_replicaCount <= 0 ||
                    p_opt.m_internalResultNum <
                        p_opt.m_replicaCount ||
                    m_hybridGenerationFingerprint == 0) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag placement requires one-column raw metadata, "
                        "unquantized BKT heads, one batch, exactly four tag slots "
                        "(self plus three external tags), and valid support/RNG parameters.\n");
                    return false;
                }

                const int voteCount =
                    (std::min)(
                        p_opt.m_limitedTagVoteHeadCount,
                        static_cast<int>(headCount));
                if (headCount <
                    p_opt.m_limitedTagMinHeadCount) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag minimum coverage %d exceeds head count %d.\n",
                        p_opt.m_limitedTagMinHeadCount,
                        static_cast<int>(headCount));
                    return false;
                }
                std::vector<std::uint32_t> headOwnTags(
                    static_cast<size_t>(headCount),
                    LimitedTagSupport::EmptyTag);
                for (const auto& head :
                     p_headVectorIDs) {
                    if (head.first < 0 ||
                        head.first >= p_fullCount ||
                        head.second < 0 ||
                        head.second >= headCount ||
                        headOwnTags[
                            static_cast<size_t>(
                                head.second)] !=
                            LimitedTagSupport::EmptyTag) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Limited-tag placement cannot map every head to one source vector.\n");
                        return false;
                    }
                    const std::uint32_t ownTag =
                        m_staticBuildTags[
                            static_cast<size_t>(
                                head.first)];
                    if (ownTag ==
                        LimitedTagSupport::EmptyTag) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Limited-tag head %d has the reserved empty tag.\n",
                            head.second);
                        return false;
                    }
                    headOwnTags[
                        static_cast<size_t>(
                            head.second)] = ownTag;
                }
                if (std::any_of(
                        headOwnTags.begin(),
                        headOwnTags.end(),
                        [](std::uint32_t p_tag) {
                            return p_tag ==
                                LimitedTagSupport::EmptyTag;
                        })) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag placement is missing one or more head source tags.\n");
                    return false;
                }
                const size_t voteCapacity =
                    static_cast<size_t>(p_fullCount) *
                    static_cast<size_t>(voteCount);
                std::vector<Vote> votes(voteCapacity);
                std::atomic<SizeType> nextVector(0);
                std::atomic_bool failed(false);
                std::atomic<std::uint64_t> voteChecked(0);
                const int threadCount =
                    (std::max)(1, p_opt.m_iSSDNumberOfThreads);
                std::vector<std::thread> threads;
                threads.reserve(static_cast<size_t>(threadCount));
                for (int thread = 0; thread < threadCount; ++thread) {
                    threads.emplace_back([&]() {
                        while (!failed.load()) {
                            const SizeType vectorID =
                                nextVector.fetch_add(1);
                            if (vectorID >= p_fullCount) return;
                            if (p_headVectorIDs.count(vectorID) != 0) {
                                continue;
                            }
                            COMMON::QueryResultSet<ValueType> results(
                                static_cast<const ValueType*>(
                                    p_fullVectors->GetVector(vectorID)),
                                voteCount);
                            if (p_headIndex->SearchIndex(results) !=
                                ErrorCode::Success) {
                                failed.store(true);
                                return;
                            }
                            voteChecked.fetch_add(
                                static_cast<std::uint64_t>(
                                    (std::max)(
                                        0, results.GetScanned())),
                                std::memory_order_relaxed);
                            const size_t offset =
                                static_cast<size_t>(vectorID) *
                                static_cast<size_t>(voteCount);
                            for (int rank = 0;
                                 rank < voteCount; ++rank) {
                                const BasicResult* result =
                                    results.GetResult(rank);
                                if (result == nullptr ||
                                    result->VID < 0) {
                                    break;
                                }
                                votes[offset +
                                      static_cast<size_t>(rank)] = {
                                    result->VID,
                                    m_staticBuildTags[
                                        static_cast<size_t>(
                                            vectorID)],
                                    result->Dist};
                            }
                        }
                    });
                }
                for (auto& thread : threads) thread.join();
                if (failed.load()) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag nearest-head vote search failed.\n");
                    return false;
                }

                votes.erase(
                    std::remove_if(
                        votes.begin(), votes.end(),
                        [headCount](const Vote& p_vote) {
                            return p_vote.m_head < 0 ||
                                p_vote.m_head >= headCount ||
                                p_vote.m_tag ==
                                    LimitedTagSupport::EmptyTag;
                        }),
                    votes.end());
                std::sort(
                    votes.begin(), votes.end(),
                    [](const Vote& p_left,
                       const Vote& p_right) {
                        if (p_left.m_head != p_right.m_head)
                            return p_left.m_head < p_right.m_head;
                        if (p_left.m_tag != p_right.m_tag)
                            return p_left.m_tag < p_right.m_tag;
                        if (p_left.m_distance !=
                            p_right.m_distance)
                            return p_left.m_distance <
                                p_right.m_distance;
                        return false;
                    });

                std::vector<Vote> nearestHeadTagVotes;
                nearestHeadTagVotes.reserve(votes.size());
                for (size_t vote = 0; vote < votes.size();) {
                    nearestHeadTagVotes.push_back(votes[vote]);
                    const SizeType head = votes[vote].m_head;
                    const std::uint32_t tag = votes[vote].m_tag;
                    do {
                        ++vote;
                    } while (
                        vote < votes.size() &&
                        votes[vote].m_head == head &&
                        votes[vote].m_tag == tag);
                }
                votes.clear();
                votes.shrink_to_fit();

                std::vector<std::vector<SupportChoice>> headSupport(
                    static_cast<size_t>(headCount));
                std::unordered_map<std::uint32_t, int> coverage;
                std::unordered_map<
                    std::uint32_t,
                    std::vector<std::pair<float, SizeType>>>
                    tagCandidates;
                for (SizeType head = 0;
                    head < headCount; ++head) {
                    const std::uint32_t ownTag =
                       headOwnTags[
                           static_cast<size_t>(
                               head)];
                    headSupport[
                       static_cast<size_t>(head)]
                       .push_back({
                           ownTag, 0.0f, true});
                    ++coverage[ownTag];
                }
                size_t cursor = 0;
                for (SizeType head = 0;
                     head < headCount; ++head) {
                    const size_t begin = cursor;
                    while (cursor <
                               nearestHeadTagVotes.size() &&
                           nearestHeadTagVotes[cursor].m_head ==
                               head) {
                        tagCandidates[
                            nearestHeadTagVotes[cursor].m_tag]
                            .emplace_back(
                                nearestHeadTagVotes[cursor]
                                    .m_distance,
                                head);
                        ++cursor;
                    }
                    std::vector<SupportChoice> candidates;
                    candidates.reserve(cursor - begin);
                    for (size_t vote = begin;
                         vote < cursor; ++vote) {
                        if (nearestHeadTagVotes[vote]
                                .m_tag ==
                            headOwnTags[
                                static_cast<size_t>(
                                    head)]) {
                            continue;
                        }
                        candidates.push_back({
                            nearestHeadTagVotes[vote].m_tag,
                            nearestHeadTagVotes[vote]
                                .m_distance,
                            false});
                    }
                    std::sort(
                        candidates.begin(), candidates.end(),
                        [](const SupportChoice& p_left,
                           const SupportChoice& p_right) {
                            if (p_left.m_distance !=
                                p_right.m_distance)
                                return p_left.m_distance <
                                    p_right.m_distance;
                            return p_left.m_tag <
                                p_right.m_tag;
                        });
                    const size_t keep =
                        (std::min)(
                            candidates.size(),
                            static_cast<size_t>(
                                (std::max)(
                                    0,
                                    p_opt
                                            .m_limitedTagSlotsPerHead -
                                        1)));
                    headSupport[static_cast<size_t>(head)]
                        .insert(
                            headSupport[
                                static_cast<size_t>(
                                    head)]
                                .end(),
                            candidates.begin(),
                            candidates.begin() + keep);
                    for (size_t slot = 0;
                         slot < keep; ++slot) {
                        ++coverage[
                            candidates[slot].m_tag];
                    }
                }

                std::vector<std::uint32_t> observedTags;
                {
                    std::unordered_set<std::uint32_t> unique;
                    for (std::uint32_t tag :
                         m_staticBuildTags) {
                        if (tag ==
                            LimitedTagSupport::EmptyTag) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Limited-tag input contains the reserved empty tag.\n");
                            return false;
                        }
                        if (unique.insert(tag).second)
                            observedTags.push_back(tag);
                    }
                }
                std::sort(
                    observedTags.begin(),
                    observedTags.end());
                const size_t supportTarget =
                    static_cast<size_t>(
                        p_opt.m_limitedTagSlotsPerHead);
                if (observedTags.size() < supportTarget) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag placement needs at least %zu distinct tags "
                        "to fill every head (got %zu).\n",
                        supportTarget, observedTags.size());
                    return false;
                }
                std::uint64_t centroidFallbackSupports = 0;
                if (observedTags.size() > 1) {
                    const int fallbackResultCount =
                        (std::min)(
                            64,
                            static_cast<int>(
                                headCount));
                    for (SizeType head = 0;
                         head < headCount; ++head) {
                        auto& choices =
                            headSupport[
                                static_cast<size_t>(
                                    head)];
                        if (choices.size() >= supportTarget)
                            continue;
                        const std::uint32_t ownTag =
                            headOwnTags[
                                static_cast<size_t>(
                                    head)];
                        const auto alreadySupports =
                            [&choices](std::uint32_t p_tag) {
                                return std::any_of(
                                    choices.begin(),
                                    choices.end(),
                                    [p_tag](
                                        const SupportChoice&
                                            p_choice) {
                                        return p_choice.m_tag ==
                                            p_tag;
                                    });
                            };
                        COMMON::QueryResultSet<ValueType>
                            neighbors(
                                static_cast<const ValueType*>(
                                    p_headIndex
                                        ->GetSample(head)),
                                fallbackResultCount);
                        if (p_headIndex->SearchIndex(
                                neighbors) ==
                            ErrorCode::Success) {
                            for (int rank = 0;
                                 rank <
                                     neighbors
                                         .GetResultNum();
                                 ++rank) {
                                const BasicResult* result =
                                    neighbors
                                        .GetResult(rank);
                                if (result == nullptr ||
                                    result->VID < 0 ||
                                    result->VID >=
                                        headCount ||
                                    result->VID == head) {
                                    continue;
                                }
                                const std::uint32_t tag =
                                    headOwnTags[
                                        static_cast<size_t>(
                                            result->VID)];
                                if (tag == ownTag ||
                                    alreadySupports(tag))
                                    continue;
                                choices.push_back({
                                    tag, result->Dist,
                                    false});
                                ++coverage[tag];
                                ++centroidFallbackSupports;
                                if (choices.size() >=
                                    supportTarget) {
                                    break;
                                }
                            }
                        }
                        if (choices.size() <
                            supportTarget) {
                            std::unordered_map<
                                std::uint32_t, float>
                                closestByTag;
                            for (SizeType candidate = 0;
                                 candidate < headCount;
                                 ++candidate) {
                                const std::uint32_t tag =
                                    headOwnTags[
                                        static_cast<size_t>(
                                            candidate)];
                                if (candidate == head ||
                                    tag == ownTag) {
                                    continue;
                                }
                                if (alreadySupports(tag))
                                    continue;
                                const float distance =
                                    p_headIndex
                                       ->ComputeDistance(
                                            p_headIndex
                                                ->GetSample(
                                                    head),
                                            p_headIndex
                                                ->GetSample(
                                                    candidate));
                                const auto found =
                                    closestByTag.find(tag);
                                if (found ==
                                        closestByTag.end() ||
                                    distance <
                                        found->second) {
                                    closestByTag[tag] =
                                        distance;
                                }
                            }
                            std::vector<SupportChoice>
                                fallbackChoices;
                            fallbackChoices.reserve(
                                closestByTag.size());
                            for (const auto& candidate :
                                 closestByTag) {
                                fallbackChoices.push_back({
                                    candidate.first,
                                    candidate.second,
                                    false});
                            }
                            std::sort(
                                fallbackChoices.begin(),
                                fallbackChoices.end(),
                                [](const SupportChoice&
                                       p_left,
                                   const SupportChoice&
                                       p_right) {
                                    if (p_left.m_distance !=
                                        p_right.m_distance) {
                                        return p_left
                                                  .m_distance <
                                            p_right
                                               .m_distance;
                                    }
                                    return p_left.m_tag <
                                        p_right.m_tag;
                                });
                            for (const auto& candidate :
                                 fallbackChoices) {
                                if (choices.size() >=
                                    supportTarget) {
                                    break;
                                }
                                choices.push_back(
                                    candidate);
                                ++coverage[
                                    candidate.m_tag];
                                ++centroidFallbackSupports;
                            }
                        }
                        if (choices.size() !=
                            supportTarget) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Limited-tag head %d has only %zu/%zu "
                                "distinct support tags.\n",
                                head, choices.size(),
                                supportTarget);
                            return false;
                        }
                    }
                }
                for (auto& entry : tagCandidates) {
                    std::sort(
                        entry.second.begin(),
                        entry.second.end(),
                        [](const std::pair<float, SizeType>& p_left,
                           const std::pair<float, SizeType>& p_right) {
                            if (p_left.first != p_right.first)
                                return p_left.first <
                                    p_right.first;
                            return p_left.second <
                                p_right.second;
                        });
                }

                const auto supports =
                    [&headSupport](
                        SizeType p_head,
                        std::uint32_t p_tag) {
                        const auto& choices =
                            headSupport[
                                static_cast<size_t>(
                                    p_head)];
                        return std::any_of(
                            choices.begin(), choices.end(),
                            [p_tag](
                                const SupportChoice& p_choice) {
                                return p_choice.m_tag ==
                                    p_tag;
                            });
                    };
                for (std::uint32_t tag : observedTags) {
                    auto candidate =
                        tagCandidates.find(tag);
                    if (candidate == tagCandidates.end()) {
                        if (coverage[tag] >=
                            p_opt.m_limitedTagMinHeadCount) {
                            continue;
                        }
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Tag %u has no non-head nearest-head votes.\n",
                            tag);
                        return false;
                    }
                    for (const auto& headCandidate :
                         candidate->second) {
                        if (coverage[tag] >=
                            p_opt.m_limitedTagMinHeadCount)
                            break;
                        const SizeType head =
                            headCandidate.second;
                        auto& choices =
                            headSupport[
                                static_cast<size_t>(head)];
                        if (supports(head, tag)) continue;
                        if (choices.size() <
                            static_cast<size_t>(
                                p_opt
                                    .m_limitedTagSlotsPerHead)) {
                            choices.push_back({
                                tag, headCandidate.first,
                                false});
                            ++coverage[tag];
                            continue;
                        }

                        size_t replacement =
                            choices.size();
                        for (size_t slot = 0;
                             slot < choices.size();
                             ++slot) {
                            if (choices[slot].m_isOwnTag)
                                continue;
                            if (coverage[
                                    choices[slot].m_tag] <=
                                p_opt
                                    .m_limitedTagMinHeadCount)
                                continue;
                            if (replacement ==
                                    choices.size() ||
                                choices[slot].m_distance >
                                    choices[replacement]
                                        .m_distance ||
                                (choices[slot].m_distance ==
                                     choices[replacement]
                                         .m_distance &&
                                 choices[slot].m_tag >
                                     choices[replacement]
                                         .m_tag)) {
                                replacement = slot;
                            }
                        }
                        if (replacement ==
                            choices.size()) {
                            continue;
                        }
                        --coverage[
                            choices[replacement].m_tag];
                        choices[replacement] = {
                            tag, headCandidate.first,
                            false};
                        ++coverage[tag];
                    }
                    if (coverage[tag] <
                        p_opt.m_limitedTagMinHeadCount) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Cannot give tag %u the required %d supported heads "
                            "(got %d).\n",
                            tag,
                            p_opt.m_limitedTagMinHeadCount,
                            coverage[tag]);
                        return false;
                    }
                }

                if (!p_support.Initialize(
                        headCount,
                        p_opt.m_limitedTagSlotsPerHead,
                        p_opt.m_limitedTagVoteHeadCount,
                        p_opt.m_limitedTagMinHeadCount,
                        m_hybridGenerationFingerprint)) {
                    return false;
                }
                for (SizeType head = 0;
                     head < headCount; ++head) {
                    std::vector<std::uint32_t> tags;
                    for (const auto& choice :
                         headSupport[
                             static_cast<size_t>(head)]) {
                        tags.push_back(choice.m_tag);
                    }
                    if (!p_support.SetHeadTags(head, tags)) {
                        return false;
                    }
                }
                std::string supportError;
                if (!p_support.Finalize(&supportError)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Invalid limited-tag support table: %s\n",
                        supportError.c_str());
                    return false;
                }
                const auto supportHeadsByTag = [&]() {
                    std::unordered_map<
                        std::uint32_t,
                        std::vector<SizeType>>
                        headsByTag;
                    for (SizeType head = 0;
                         head < headCount; ++head) {
                        for (const auto& choice :
                             headSupport[
                                 static_cast<size_t>(head)]) {
                            headsByTag[choice.m_tag]
                                .push_back(head);
                        }
                    }
                    return headsByTag;
                }();

                const size_t selectionCapacity =
                    static_cast<size_t>(p_fullCount) *
                    static_cast<size_t>(
                        p_opt.m_replicaCount);
                Selection limitedSelections(
                    selectionCapacity,
                    p_opt.m_tmpdir,
                    "limited_tag_selection_tmp");
                std::vector<std::atomic_int> postingListSize(
                    static_cast<size_t>(headCount));
                std::vector<std::uint8_t> desiredReplicas(
                    static_cast<size_t>(p_fullCount), 0);
                for (auto& size : postingListSize) size = 0;
                nextVector.store(0);
                failed.store(false);
                std::atomic<SizeType> failedVector(MaxSize);
                std::atomic<std::uint32_t> failedTag(
                    LimitedTagSupport::EmptyTag);
                std::atomic<int> failedScanned(0);
                std::atomic<int> failedStatus(
                    static_cast<int>(ErrorCode::Success));
                std::atomic<std::uint64_t> exactFallbacks(0);
                std::atomic<std::uint64_t>
                    exactFallbackDistanceChecks(0);
                std::atomic<std::uint64_t> placementChecked(0);
                threads.clear();
                for (int thread = 0; thread < threadCount; ++thread) {
                    threads.emplace_back([&]() {
                        while (!failed.load()) {
                            const SizeType vectorID =
                                nextVector.fetch_add(1);
                            if (vectorID >= p_fullCount) return;
                            if (p_headVectorIDs.count(vectorID) != 0) {
                                continue;
                            }
                            const std::uint32_t tag =
                                m_staticBuildTags[
                                    static_cast<size_t>(
                                        vectorID)];
                            COMMON::QueryResultSet<ValueType> results(
                                static_cast<const ValueType*>(
                                    p_fullVectors
                                        ->GetVector(vectorID)),
                                (std::max)(
                                    1,
                                    p_opt
                                        .m_internalResultNum));
                            const ErrorCode status =
                                p_headIndex
                                    ->SearchIndexWithResultFilter(
                                        results,
                                        [&p_support, tag](
                                            SizeType p_head) {
                                            return p_support
                                                .Supports(
                                                    p_head,
                                                    tag);
                                        });
                            if (status != ErrorCode::Success) {
                                SizeType expected = MaxSize;
                                if (failedVector.compare_exchange_strong(
                                        expected, vectorID)) {
                                    failedTag.store(tag);
                                    failedScanned.store(
                                        results.GetScanned());
                                    failedStatus.store(
                                        static_cast<int>(status));
                                }
                                failed.store(true);
                                return;
                            }
                            placementChecked.fetch_add(
                                static_cast<std::uint64_t>(
                                    (std::max)(
                                        0, results.GetScanned())),
                                std::memory_order_relaxed);

                            const size_t offset =
                                static_cast<size_t>(vectorID) *
                                static_cast<size_t>(
                                    p_opt.m_replicaCount);
                            int selected = 0;
                            for (int rank = 0;
                                 rank <
                                     results.GetResultNum() &&
                                 selected <
                                     p_opt.m_replicaCount;
                                 ++rank) {
                                const BasicResult* result =
                                    results.GetResult(rank);
                                if (result == nullptr ||
                                    result->VID < 0)
                                    break;
                                bool accepted = true;
                                for (int prior = 0;
                                     prior < selected;
                                     ++prior) {
                                    const SizeType priorHead =
                                        limitedSelections
                                            .m_selections[
                                                offset +
                                                static_cast<size_t>(
                                                    prior)]
                                            .node;
                                    const float headDistance =
                                        p_headIndex
                                            ->ComputeDistance(
                                                p_headIndex
                                                    ->GetSample(
                                                        result
                                                            ->VID),
                                                p_headIndex
                                                    ->GetSample(
                                                        priorHead));
                                    if (p_opt.m_rngFactor *
                                            headDistance <=
                                        result->Dist) {
                                        accepted = false;
                                        break;
                                    }
                                }
                                if (!accepted) continue;
                                Edge& edge =
                                    limitedSelections
                                        .m_selections[
                                            offset +
                                            static_cast<size_t>(
                                                selected)];
                                edge.node = result->VID;
                                edge.tonode = vectorID;
                                edge.distance = result->Dist;
                                ++postingListSize[
                                    static_cast<size_t>(
                                        edge.node)];
                                ++selected;
                            }
                            if (selected == 0) {
                                const auto supported =
                                    supportHeadsByTag.find(tag);
                                SizeType bestHead = MaxSize;
                                float bestDistance = MaxDist;
                                if (supported !=
                                    supportHeadsByTag.end()) {
                                    for (SizeType head :
                                         supported->second) {
                                        const float distance =
                                            p_headIndex
                                                ->ComputeDistance(
                                                    p_fullVectors
                                                        ->GetVector(
                                                            vectorID),
                                                    p_headIndex
                                                        ->GetSample(
                                                            head));
                                        if (distance <
                                                bestDistance ||
                                            (distance ==
                                                 bestDistance &&
                                             head < bestHead)) {
                                            bestHead = head;
                                            bestDistance =
                                                distance;
                                        }
                                    }
                                    exactFallbackDistanceChecks
                                        .fetch_add(
                                            supported->second
                                                .size(),
                                            std::memory_order_relaxed);
                                }
                                if (bestHead != MaxSize) {
                                    Edge& edge =
                                        limitedSelections
                                            .m_selections[
                                                offset];
                                    edge.node = bestHead;
                                    edge.tonode = vectorID;
                                    edge.distance =
                                        bestDistance;
                                    ++postingListSize[
                                        static_cast<size_t>(
                                            bestHead)];
                                    selected = 1;
                                    exactFallbacks.fetch_add(
                                        1,
                                        std::memory_order_relaxed);
                                } else {
                                    SizeType expected = MaxSize;
                                    if (failedVector
                                            .compare_exchange_strong(
                                                expected,
                                                vectorID)) {
                                        failedTag.store(tag);
                                        failedScanned.store(
                                            results
                                                .GetScanned());
                                    }
                                    failed.store(true);
                                    return;
                                }
                            }
                            desiredReplicas[
                                static_cast<size_t>(
                                    vectorID)] =
                                static_cast<std::uint8_t>(
                                    selected);
                        }
                    });
                }
                for (auto& thread : threads) thread.join();
                if (failed.load()) {
                    const std::uint32_t tag =
                        failedTag.load();
                    int supportedHeads = 0;
                    if (tag !=
                        LimitedTagSupport::EmptyTag) {
                        for (SizeType head = 0;
                             head < headCount; ++head) {
                            if (p_support.Supports(
                                    head, tag)) {
                                ++supportedHeads;
                            }
                        }
                    }
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag filtered BKT/RNG placement failed or "
                        "produced an empty non-head assignment "
                        "(vector=%d tag=%u supportedHeads=%d scanned=%d "
                        "status=%d).\n",
                        failedVector.load(), tag,
                        supportedHeads,
                        failedScanned.load(),
                        failedStatus.load());
                    return false;
                }

                VectorIndex::SortSelections(
                    &limitedSelections.m_selections);
                std::vector<int> retainedPerVector(
                    static_cast<size_t>(p_fullCount), 0);
                size_t read = 0;
                std::uint64_t assignmentCount = 0;
                for (SizeType head = 0;
                     head < headCount; ++head) {
                    const size_t begin = read;
                    while (read <
                               limitedSelections
                                   .m_selections.size() &&
                           limitedSelections
                                   .m_selections[read]
                                   .node == head) {
                        ++read;
                    }
                    const int available =
                        static_cast<int>(read - begin);
                    const int kept = available;
                    postingListSize[
                        static_cast<size_t>(head)] = kept;
                    assignmentCount +=
                        static_cast<std::uint64_t>(kept);
                    std::unordered_set<std::uint32_t>
                        headTags;
                    for (int record = 0;
                         record < kept; ++record) {
                        const Edge& edge =
                            limitedSelections
                                .m_selections[
                                    begin +
                                    static_cast<size_t>(
                                        record)];
                        const std::uint32_t tag =
                            m_staticBuildTags[
                                static_cast<size_t>(
                                    edge.tonode)];
                        if (!p_support.Supports(head, tag)) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Limited-tag posting %d contains unsupported tag %u.\n",
                                head, tag);
                            return false;
                        }
                        headTags.insert(tag);
                        ++retainedPerVector[
                            static_cast<size_t>(
                                edge.tonode)];
                    }
                    if (headTags.size() >
                        static_cast<size_t>(
                            p_opt
                                .m_limitedTagSlotsPerHead)) {
                        return false;
                    }
                }
                for (SizeType vectorID = 0;
                     vectorID < p_fullCount; ++vectorID) {
                    if (p_headVectorIDs.count(vectorID) != 0)
                        continue;
                    if (retainedPerVector[
                            static_cast<size_t>(vectorID)] !=
                        desiredReplicas[
                            static_cast<size_t>(vectorID)]) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Limited-tag serialization retained %d/%u RNG replicas for vector %d.\n",
                            retainedPerVector[
                                static_cast<size_t>(
                                    vectorID)],
                            static_cast<unsigned>(
                                desiredReplicas[
                                    static_cast<size_t>(
                                        vectorID)]),
                            vectorID);
                        return false;
                    }
                }

                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Limited-tag support selected %zu tags over %d heads "
                    "(self+top-%d external, slots=%d coverage>=%d); "
                    "retained all %llu RNG assignments "
                    "(voteChecks=%llu placementChecks=%llu "
                    "exactFallbacks=%llu/%llu "
                    "headNeighborFallbackSupports=%llu).\n",
                    observedTags.size(),
                    static_cast<int>(headCount),
                    p_opt.m_limitedTagSlotsPerHead - 1,
                    p_opt.m_limitedTagSlotsPerHead,
                    p_opt.m_limitedTagMinHeadCount,
                    static_cast<unsigned long long>(
                        assignmentCount),
                    static_cast<unsigned long long>(
                        voteChecked.load()),
                    static_cast<unsigned long long>(
                        placementChecked.load()),
                    static_cast<unsigned long long>(
                        exactFallbacks.load()),
                    static_cast<unsigned long long>(
                        exactFallbackDistanceChecks.load()),
                    static_cast<unsigned long long>(
                        centroidFallbackSupports));
                p_selections.m_selections.swap(
                    limitedSelections.m_selections);
                p_selections.m_start = 0;
                p_selections.m_end =
                    p_selections.m_selections.size();
                p_selections.m_totalsize =
                    p_selections.m_end;
                p_postingListSize.swap(postingListSize);
                return true;
            }

            void InitWorkSpace(ExtraWorkSpace* p_exWorkSpace, bool clear = false) override
            {
                if (clear) {
                    p_exWorkSpace->Clear(m_opt->m_searchInternalResultNum, StaticWorkspaceBufferBytes(), false, m_opt->m_enableDataCompression);
                }
                else {
                    p_exWorkSpace->Initialize(m_opt->m_maxCheck, m_opt->m_hashExp, m_opt->m_searchInternalResultNum, StaticWorkspaceBufferBytes(), false, m_opt->m_enableDataCompression);
                    int wid = 0;
                    if (m_freeWorkSpaceIds == nullptr || !m_freeWorkSpaceIds->try_pop(wid))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FreeWorkSpaceIds is not initalized or the workspace number is not enough! Please increase iothread number.\n");
                        wid = m_workspaceCount.fetch_add(1);
                    }
                    p_exWorkSpace->SetAsyncContextID(wid);
                    p_exWorkSpace->m_callback = [m_freeWorkSpaceIds = m_freeWorkSpaceIds, wid] () {
                        if (m_freeWorkSpaceIds) m_freeWorkSpaceIds->push(wid);
                    };
                }
            }

            virtual bool LoadIndex(Options& p_opt, COMMON::VersionLabel& p_versionMap, COMMON::Dataset<std::uint64_t>& p_vectorTranslateMap,  std::shared_ptr<VectorIndex> m_index) {
                m_hybridGenerationFingerprint = 0;
                const bool constrainedPurePostings =
                    p_opt.m_enableHybridDistance ||
                    p_opt.m_enableLimitedTagPosting;
                const std::string& generationText =
                    p_opt.m_enableHybridDistance
                        ? p_opt.m_hybridGenerationFingerprint
                        : p_opt.m_limitedTagGenerationFingerprint;
                if (constrainedPurePostings &&
                    (!Helper::Convert::ConvertStringTo<std::uint64_t>(
                         generationText.c_str(),
                         m_hybridGenerationFingerprint) ||
                     m_hybridGenerationFingerprint == 0)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Constrained pure postings require a non-zero "
                        "persisted build generation.\n");
                    return false;
                }
                m_extraFullGraphFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                m_opt = &p_opt;
                if (!ConfigureStaticPipePQ(p_opt, 0, false)) {
                    return false;
                }
                m_available = false;
                m_listInfos.clear();
                m_indexFiles.clear();
                m_totalListCount = 0;
                m_listPerFile = 0;
                m_oneContext = true;
                m_hybridMaxListPageCount = 0;
                m_hasHybridPurePostings = false;
                m_avgRecordsPerList = -1.0;
                m_avgPagesPerList = -1.0;
                m_avgBytesPerList = -1.0;
                m_hybridAvgRecordsPerList = -1.0;
                m_hybridAvgPagesPerList = -1.0;
                m_hybridAvgBytesPerList = -1.0;
                m_enableDeltaEncoding = p_opt.m_enableDeltaEncoding;
                m_enablePostingListRearrange = p_opt.m_enablePostingListRearrange;
                m_enableDataCompression = p_opt.m_enableDataCompression;
                m_enableDictTraining = p_opt.m_enableDictTraining;
                m_staticHasMetadata = false;
                m_staticNumTagsPerVec = 0;
                m_staticACLTagCols = 0;
                m_staticMetadataBytes = sizeof(int);
                m_staticMaxListPageCount = 0;
                m_staticAttributeOrdered =
                    fileexists((p_opt.m_indexDirectory + FolderSep +
                                "ordered_page_starts.bin").c_str());
                std::string curFile = m_extraFullGraphFile;
                const size_t configuredTagBytes =
                    (!m_staticPipePQ && p_opt.m_numTagsPerVec > 0)
                        ? static_cast<size_t>(p_opt.m_numTagsPerVec) * sizeof(uint32_t)
                        : 0;
                if (configuredTagBytes > 0 &&
                    (m_enableDeltaEncoding || m_enablePostingListRearrange ||
                     m_enableDataCompression)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static metadata snapshots require raw postings without delta encoding, "
                        "rearrangement, or compression.\n");
                    return false;
                }
                if (constrainedPurePostings &&
                    (m_staticPipePQ || m_enableDeltaEncoding ||
                     m_enablePostingListRearrange || m_enableDataCompression)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Constrained pure+tail layout requires raw STM1 postings without "
                        "PipePQ, delta encoding, rearrangement, or compression.\n");
                    return false;
                }
                do {
                    int loadedListCount = 0;
                    try {
                        loadedListCount = LoadingHeadInfo(curFile, m_listInfos);
                    }
                    catch (std::exception& e)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error occurs when loading HeadInfo:%s\n", e.what());
                        return false;
                    }

                    std::shared_ptr<Helper::DiskIO> curIndexFile;
                    if (!OpenStaticIndexFile(curFile, p_opt, curIndexFile)) {
                        return false;
                    }

                    m_indexFiles.emplace_back(curIndexFile);
                    m_totalListCount += loadedListCount;

                    curFile = m_extraFullGraphFile + "_" + std::to_string(m_indexFiles.size());
                } while (fileexists(curFile.c_str()));
                if (constrainedPurePostings &&
                    (!m_staticHasMetadata ||
                     m_indexFiles.size() != 1 ||
                     p_opt.m_tailReplicaCount <= 0 ||
                     p_opt.m_enableOrderedPageStart ||
                     m_staticAttributeOrdered ||
                     p_opt.m_unfilterPureDistanceScanPercent !=
                         100)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Constrained pure mode requires one generation-bound STM1 v2 file with an "
                        "existing vector-distance tail, no ordered-page directory, "
                        "and UnfilterPureDistanceScanPercent=100.\n");
                    return false;
                }
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Static adaptive posting reads: maximum physical posting size is %d pages.\n",
                    m_staticMaxListPageCount);
                m_oneContext = (m_indexFiles.size() == 1);
                const std::uint64_t primaryGeneration =
                    m_staticLoadedGenerationFingerprint;
                if (!constrainedPurePostings &&
                    primaryGeneration != 0) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Generation-bound STM1 requires its constrained pure mode enabled.\n");
                    return false;
                }
                if (constrainedPurePostings &&
                    primaryGeneration !=
                        m_hybridGenerationFingerprint) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Primary static posting generation mismatch: "
                        "config=%llu posting=%llu\n",
                        static_cast<unsigned long long>(
                            m_hybridGenerationFingerprint),
                        static_cast<unsigned long long>(
                            primaryGeneration));
                    return false;
                }

                if (m_staticHasMetadata && p_opt.m_numTagsPerVec > 0 &&
                    p_opt.m_numTagsPerVec != m_staticNumTagsPerVec) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Static metadata tag-column mismatch: index=%d runtime=%d\n",
                        m_staticNumTagsPerVec, p_opt.m_numTagsPerVec);
                    return false;
                }
                if (constrainedPurePostings) {
                    if (!m_staticHasMetadata || m_staticNumTagsPerVec <= 0) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Constrained pure prefixes require STM1 primary postings.\n");
                        return false;
                    }
                    std::uint64_t pureRecords = 0;
                    std::uint64_t pureBytes = 0;
                    std::uint64_t purePages = 0;
                    for (const auto& list :
                         m_listInfos) {
                        if (list.pureEleCount < 0 ||
                            list.pureEleCount >
                                list.listEleCount) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Invalid constrained pure prefix: pure=%d total=%d.\n",
                                list.pureEleCount,
                                list.listEleCount);
                            return false;
                        }
                        const size_t listBytes =
                            static_cast<size_t>(
                                list.pureEleCount) *
                            static_cast<size_t>(
                                m_vectorInfoSize);
                        const int listPages =
                            list.pureEleCount == 0
                            ? 0
                            : static_cast<int>(
                                  (static_cast<size_t>(
                                       list.pageOffset) +
                                   listBytes +
                                   PageSize - 1) >>
                                  PageSizeEx);
                        pureRecords +=
                            static_cast<std::uint64_t>(
                                list.pureEleCount);
                        pureBytes +=
                            static_cast<std::uint64_t>(
                                listBytes);
                        purePages +=
                            static_cast<std::uint64_t>(
                                listPages);
                        m_hybridMaxListPageCount =
                            (std::max)(
                                m_hybridMaxListPageCount,
                                listPages);
                    }
                    const double listCount =
                        static_cast<double>(
                            m_listInfos.size());
                    m_hybridAvgRecordsPerList =
                        listCount == 0.0
                        ? -1.0
                        : static_cast<double>(
                              pureRecords) /
                              listCount;
                    m_hybridAvgPagesPerList =
                        listCount == 0.0
                        ? -1.0
                        : static_cast<double>(
                              purePages) /
                              listCount;
                    m_hybridAvgBytesPerList =
                        listCount == 0.0
                        ? -1.0
                        : static_cast<double>(
                              pureBytes) /
                              listCount;
                    m_hasHybridPurePostings = true;
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Loaded one primary constrained pure+tail posting: pure "
                        "avg records %.2f, avg pages %.2f, avg bytes %.2f, "
                        "max pages %d.\n",
                        m_hybridAvgRecordsPerList,
                        m_hybridAvgPagesPerList,
                        m_hybridAvgBytesPerList,
                        m_hybridMaxListPageCount);
                }
                m_avgRecordsPerList =
                    ComputeAverageRecords(m_listInfos);
                m_avgPagesPerList =
                    ComputeAveragePages(m_listInfos);
                m_avgBytesPerList =
                    ComputeAverageBytes(m_listInfos);

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
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
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
                                ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, p_postingID);
                                return listInfo == nullptr ||
                                    StaticScanLimit(p_exWorkSpace, listInfo) == 0;
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
                    ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, curPostingID);
                    if (listInfo == nullptr) return ErrorCode::Key_OverFlow;
                    int fileid = GetPostingFileId(p_exWorkSpace, curPostingID);

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = GetPostingIndexFile(p_exWorkSpace, fileid);
#endif

                    auto& readRange = p_exWorkSpace->m_postingReadRanges[pi];
                    readRange = BuildStaticPostingReadRange(
                        p_exWorkSpace, curPostingID, listInfo);
                    const int readPageCount = readRange.m_readPageCount;
                    diskRead += readPageCount;
                    diskIO += 1;
                    const int scanCount = readRange.ScanCount();
                    listElements += scanCount;
                    scannedListElements += scanCount;

                    const size_t totalBytes =
                        static_cast<size_t>(readPageCount) << PageSizeEx;

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
                        ListInfo* listInfo = static_cast<ListInfo*>(request.m_payload);
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
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", GetPostingFileBase(p_exWorkSpace).c_str(), totalBytes, numRead);
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
                int retry = 0;
                bool success = false;
                while (retry < 2 && !success)
                {
                    success = BatchReadFileAsync(
                        GetPostingIndexFiles(p_exWorkSpace),
                        p_exWorkSpace->m_diskRequests.data(),
                        postingListCount);
                    retry++;
                }
                if (!success) {
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

                        ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, curPostingID);
                        if (listInfo == nullptr) return ErrorCode::Key_OverFlow;
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
                        for (int scanRange = 0; scanRange < 2; ++scanRange) {
                            int scanBegin = 0;
                            int scanEnd = 0;
                            if (!GetStaticScanRange(
                                    p_exWorkSpace,
                                    staticPostingSlot,
                                    listInfo,
                                    scanRange,
                                    scanBegin,
                                    scanEnd)) {
                                continue;
                            }
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
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
                if (RejectUnsupportedStaticFilter(p_exWorkSpace)) return ErrorCode::Fail;
                if (HasStaticMetadataFilter(p_exWorkSpace)) {
                    auto& postingIDs = p_exWorkSpace->m_postingIDs;
                    postingIDs.erase(
                        std::remove_if(
                            postingIDs.begin(), postingIDs.end(),
                            [this, p_exWorkSpace](SizeType p_postingID) {
                                ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, p_postingID);
                                return listInfo == nullptr ||
                                    StaticScanLimit(p_exWorkSpace, listInfo) == 0;
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
                    ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, curPostingID);
                    if (listInfo == nullptr) return ErrorCode::Key_OverFlow;
                    int fileid = GetPostingFileId(p_exWorkSpace, curPostingID);

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = GetPostingIndexFile(p_exWorkSpace, fileid);
#endif

                    auto& readRange = p_exWorkSpace->m_postingReadRanges[pi];
                    readRange = BuildStaticPostingReadRange(
                        p_exWorkSpace, curPostingID, listInfo);
                    const int readPageCount = readRange.m_readPageCount;
                    diskRead += readPageCount;
                    diskIO += 1;
                    listElements += readRange.ScanCount();

                    const size_t totalBytes =
                        static_cast<size_t>(readPageCount) << PageSizeEx;

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
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", GetPostingFileBase(p_exWorkSpace).c_str(), totalBytes, numRead);
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
                    success = BatchReadFileAsync(
                        GetPostingIndexFiles(p_exWorkSpace),
                        p_exWorkSpace->m_diskRequests.data(),
                        postingListCount);
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
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
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
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
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
                            GetPostingListInfo(
                                p_exWorkSpace,
                                p_exWorkSpace->m_postingIDs.front()));
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

                const auto tailPhaseStart = std::chrono::steady_clock::now();
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
                int firstHeadOwner = -1;
                bool haveMultipleHeadOwners = false;
                for (int owner :
                     m_staticBuildHeadOwners) {
                    if (owner < 0) continue;
                    if (firstHeadOwner < 0) {
                        firstHeadOwner = owner;
                    } else if (owner !=
                               firstHeadOwner) {
                        haveMultipleHeadOwners = true;
                        break;
                    }
                }
                const bool haveCrossBundleOwners =
                    m_staticBuildVectorOwners.size() == static_cast<size_t>(p_fullCount) &&
                    m_staticBuildHeadOwners.size() == static_cast<size_t>(headCount) &&
                    haveMultipleHeadOwners;
                const bool useSingleGlobalBundleTail =
                    !haveMultipleHeadOwners &&
                    m_staticHeadBundleLocalToGlobalHIDs != nullptr &&
                    m_staticHeadBundleIndexes.size() == 1 &&
                    m_staticHeadBundleLocalToGlobalHIDs->size() == 1 &&
                    m_staticHeadBundleIndexes[0] != nullptr &&
                    (*m_staticHeadBundleLocalToGlobalHIDs)[0].size() ==
                        static_cast<size_t>(
                            m_staticHeadBundleIndexes[0]
                                ->GetNumSamples());
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
                        : (useGlobalRNGTail
                            ? "global-RNG-cross-bundle"
                            : (useSingleGlobalBundleTail
                                ? "single-global-node-vector-RNG"
                                : "nearest-head")));
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
                    for (size_t i = pureBegin;                     i < pureEnd && pure[i].node == p_head; ++i) {
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
                        const int vectorOwner = haveCrossBundleOwners
                            ? m_staticBuildVectorOwners[static_cast<size_t>(vectorID)]
                            : -1;
                        if (useSingleGlobalBundleTail) {
                            const auto& globalIndex =
                                m_staticHeadBundleIndexes[0];
                            const auto& localToGlobal =
                                (*m_staticHeadBundleLocalToGlobalHIDs)[0];
                            nearbyHeads.SetTarget(
                                vector,
                                globalIndex->m_pQuantizer);
                            nearbyHeads.Reset();
                            if (globalIndex->SearchIndex(
                                    nearbyHeads) !=
                                ErrorCode::Success) {
                                continue;
                            }
                            BasicResult* results =
                                nearbyHeads.GetResults();
                            for (int rank = 0;
                                 rank < replicaCount;
                                 ++rank) {
                                const SizeType localHead =
                                    results[rank].VID;
                                if (localHead < 0 ||
                                    static_cast<size_t>(
                                        localHead) >=
                                        localToGlobal.size()) {
                                    continue;
                                }
                                offerTailCandidate(
                                    vectorID,
                                    localToGlobal[
                                        static_cast<size_t>(
                                            localHead)],
                                    results[rank].Dist);
                            }
                            continue;
                        }
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

                const auto tailCandidateStart = std::chrono::steady_clock::now();
                std::vector<std::thread> workers;
                workers.reserve(threadCount);
                for (int t = 0; t < threadCount; ++t) workers.emplace_back(collectTailCandidates);
                for (auto& worker : workers) worker.join();
                const auto tailCandidateEnd = std::chrono::steady_clock::now();
                if (tailSearchFailed.load(std::memory_order_acquire)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Single-seed cross-graph tail candidate search failed.\n");
                    return false;
                }

                auto logTailTiming = [&](const char* p_cap) {
                    const auto tailPhaseEnd = std::chrono::steady_clock::now();
                    const double prepareSeconds = std::chrono::duration<double>(
                        tailCandidateStart - tailPhaseStart).count();
                    const double candidateSeconds = std::chrono::duration<double>(
                        tailCandidateEnd - tailCandidateStart).count();
                    const double mergeSeconds = std::chrono::duration<double>(
                        tailPhaseEnd - tailCandidateEnd).count();
                    const double totalSeconds = std::chrono::duration<double>(
                        tailPhaseEnd - tailPhaseStart).count();
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Static Phase 4 timing: prepare=%.2fs candidate=%.2fs merge=%.2fs total=%.2fs cap=%s\n",
                        prepareSeconds, candidateSeconds, mergeSeconds, totalSeconds, p_cap);
                };

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
                        mergedSelections.end(),
                        pure.begin() + pureBegin,
                        pure.begin() + pureRead);

                    auto& candidates = tailCandidatesByHead[static_cast<size_t>(head)];
                    for (const Edge& candidate : candidates) {
                        mergedSelections.push_back(candidate);
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
                    logTailTiming("unbounded");
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
                logTailTiming(
                    unboundedTail ? "unbounded" :
                        ("purePages+" + std::to_string(extraTailPages)).c_str());
                return true;
            }

            bool MergeConstrainedPureWithOriginalPosting(
                Selection& p_constrainedSelections,
                std::vector<std::atomic_int>& p_postingListSize,
                const std::vector<int>& p_constrainedPureSizes,
                const Selection& p_originalPosting,
                const std::vector<int>& p_originalPostingSizes)
            {
                const size_t headCount = p_postingListSize.size();
                if (p_constrainedPureSizes.size() != headCount ||
                    p_originalPostingSizes.size() != headCount) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Constrained/original posting cardinality mismatch.\n");
                    return false;
                }

                size_t maximumRecordCount = 0;
                for (size_t head = 0; head < headCount; ++head) {
                    if (p_constrainedPureSizes[head] < 0 ||
                        p_originalPostingSizes[head] < 0 ||
                        maximumRecordCount >
                            (std::numeric_limits<size_t>::max)() -
                                static_cast<size_t>(p_constrainedPureSizes[head]) -
                                static_cast<size_t>(p_originalPostingSizes[head])) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Constrained/original posting size is invalid.\n");
                        return false;
                    }
                    maximumRecordCount +=
                        static_cast<size_t>(p_constrainedPureSizes[head]) +
                        static_cast<size_t>(p_originalPostingSizes[head]);
                }

                std::vector<Edge> merged;
                merged.reserve(maximumRecordCount);
                size_t hybridRead = 0;
                size_t originalRead = 0;
                size_t hybridRecordCount = 0;
                size_t originalRecordCount = 0;
                size_t overlapRecordCount = 0;
                size_t suffixRecordCount = 0;
                auto vectorDistanceLess =
                    [](const Edge& p_left, const Edge& p_right) {
                        if (p_left.distance != p_right.distance) {
                            return p_left.distance < p_right.distance;
                        }
                        return p_left.tonode < p_right.tonode;
                    };

                for (SizeType head = 0;
                     head < static_cast<SizeType>(headCount);
                     ++head) {
                    const size_t hybridBegin = hybridRead;
                    while (hybridRead <
                               p_constrainedSelections.m_selections.size() &&
                           p_constrainedSelections
                                   .m_selections[hybridRead]
                                   .node == head) {
                        ++hybridRead;
                    }
                    const int hybridCount =
                        p_constrainedPureSizes[static_cast<size_t>(head)];
                    if (hybridRead - hybridBegin <
                        static_cast<size_t>(hybridCount)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Constrained pure posting %d contains %zu/%d records.\n",
                            head, hybridRead - hybridBegin, hybridCount);
                        return false;
                    }

                    std::unordered_set<SizeType> seen;
                    seen.reserve(
                        static_cast<size_t>(hybridCount) +
                        static_cast<size_t>(
                            p_originalPostingSizes[
                                static_cast<size_t>(head)]));
                    for (int record = 0;
                         record < hybridCount;
                         ++record) {
                        const Edge& edge =
                            p_constrainedSelections.m_selections[
                                hybridBegin +
                                static_cast<size_t>(record)];
                        if (edge.tonode < 0 ||
                            !seen.insert(edge.tonode).second) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Constrained pure posting %d contains an invalid or duplicate VID.\n",
                                head);
                            return false;
                        }
                        merged.push_back(edge);
                    }
                    hybridRecordCount +=
                        static_cast<size_t>(hybridCount);

                    const size_t originalBegin = originalRead;
                    while (originalRead <
                               p_originalPosting.m_selections.size() &&
                           p_originalPosting
                                   .m_selections[originalRead]
                                   .node == head) {
                        ++originalRead;
                    }
                    const int originalCount =
                        p_originalPostingSizes[
                            static_cast<size_t>(head)];
                    if (originalRead - originalBegin !=
                        static_cast<size_t>(originalCount)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Original posting %d contains %zu/%d records.\n",
                            head, originalRead - originalBegin,
                            originalCount);
                        return false;
                    }
                    originalRecordCount +=
                        static_cast<size_t>(originalCount);

                    std::vector<Edge> suffix;
                    suffix.reserve(
                        static_cast<size_t>(originalCount));
                    for (int record = 0;
                         record < originalCount;
                         ++record) {
                        const Edge& edge =
                            p_originalPosting.m_selections[
                                originalBegin +
                                static_cast<size_t>(record)];
                        if (edge.tonode < 0) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Original posting %d contains an invalid VID.\n",
                                head);
                            return false;
                        }
                        if (seen.insert(edge.tonode).second) {
                            suffix.push_back(edge);
                        } else {
                            ++overlapRecordCount;
                        }
                    }
                    std::sort(
                        suffix.begin(), suffix.end(),
                        vectorDistanceLess);
                    if (suffix.size() >
                        static_cast<size_t>(
                            (std::numeric_limits<int>::max)() -
                            hybridCount)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Merged posting %d exceeds the supported record count.\n",
                            head);
                        return false;
                    }
                    suffixRecordCount += suffix.size();
                    merged.insert(
                        merged.end(),
                        suffix.begin(), suffix.end());
                    p_postingListSize[
                        static_cast<size_t>(head)] =
                        hybridCount +
                        static_cast<int>(suffix.size());
                }

                if (originalRead !=
                    p_originalPosting.m_selections.size()) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Original posting contains records outside the head range.\n");
                    return false;
                }
                for (; hybridRead <
                       p_constrainedSelections.m_selections.size();
                     ++hybridRead) {
                    if (p_constrainedSelections
                            .m_selections[hybridRead]
                            .node != MaxSize) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Constrained posting contains records outside the head range.\n");
                        return false;
                    }
                }

                p_constrainedSelections.m_selections.swap(merged);
                p_constrainedSelections.m_start = 0;
                p_constrainedSelections.m_end =
                    p_constrainedSelections.m_selections.size();
                p_constrainedSelections.m_totalsize =
                    p_constrainedSelections.m_end;
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Constrained single posting merged H=%zu O=%zu overlap=%zu "
                    "suffix=%zu final=%zu as H|(O-H).\n",
                    hybridRecordCount, originalRecordCount,
                    overlapRecordCount, suffixRecordCount,
                    p_constrainedSelections.m_selections.size());
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
                int& p_replicaCount,
                std::atomic<std::uint64_t>* p_checkedLeaves = nullptr)
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
                if (p_checkedLeaves != nullptr) {
                    p_checkedLeaves->fetch_add(
                        static_cast<std::uint64_t>((std::max)(0, queryResults.GetScanned())),
                        std::memory_order_relaxed);
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
                m_staticAttributeOrdered = p_opt.m_enableOrderedPageStart;
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
                if (p_opt.m_enableHybridDistance &&
                    (!useBundleLocalNodeAwareBuild ||
                     plannedNodeVectors.size() != 1 ||
                     p_opt.m_ssdIndexFileNum != 1 ||
                     p_opt.m_batches != 1 ||
                     !m_staticHasMetadata ||
                     m_staticPipePQ ||
                     p_opt.m_enableDeltaEncoding ||
                     p_opt.m_enablePostingListRearrange ||
                     p_opt.m_enableDataCompression)) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Hybrid static postings require one raw STM1 file, "
                        "one build batch, and one global node-aware placement.\n");
                    return false;
                }
                if (p_opt.m_enableLimitedTagPosting &&
                    (useNodeAwareBuild ||
                     p_opt.m_ssdIndexFileNum != 1 ||
                     p_opt.m_batches != 1 ||
                     !m_staticHasMetadata ||
                     m_staticNumTagsPerVec != 1 ||
                     m_staticPipePQ ||
                     p_opt.m_enableDeltaEncoding ||
                     p_opt.m_enablePostingListRearrange ||
                     p_opt.m_enableDataCompression ||
                     p_opt.m_enableOrderedPageStart ||
                     p_opt.m_limitedTagSupportFile.empty())) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Limited-tag static postings require one raw single-column "
                        "STM1 file, one global BKT placement, one build batch, and "
                        "a support sidecar path.\n");
                    return false;
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
                    std::atomic<std::uint64_t> pureCheckedLeaves(0);
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
                                            localCount,
                                            &pureCheckedLeaves)) {
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
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Static pure placement: assignments=%zu checkedLeaves=%llu avgLeaves=%.2f\n",
                        assignments.size(),
                        static_cast<unsigned long long>(pureCheckedLeaves.load()),
                        assignments.empty()
                            ? 0.0
                            : static_cast<double>(pureCheckedLeaves.load()) /
                                  static_cast<double>(assignments.size()));
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
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Time to sort selections:%.2lf sec.\n",
                    std::chrono::duration<double>(t3 - t2).count());

                int postingSizeLimit = INT_MAX;
                if (p_opt.m_postingPageLimit > 0)
                {
                    p_opt.m_postingPageLimit = max(p_opt.m_postingPageLimit, static_cast<int>((p_opt.m_postingVectorLimit * vectorInfoSize + PageSize - 1) / PageSize));
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
                SPTAGLIB_LOG(
                    SPTAG::Helper::LogLevel::LL_Info,
                    "Time to perform posting cut:%.2lf sec.\n",
                    std::chrono::duration<double>(t4 - t3).count());
                SPTAGLIB_LOG(
                    SPTAG::Helper::LogLevel::LL_Info,
                    "Static Phase 3 (pure assignment/cut) time: %.2lf sec.\n",
                    std::chrono::duration<double>(t4 - t1).count());

                auto fullVectors = p_reader->GetVectorSet();
                if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine &&
                    !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) {
                    fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);
                }
                std::vector<SizeType>
                    hybridHeadVectorIDs;
                std::vector<int>
                    hybridCategoricalColumns;
                std::vector<int>
                    hybridPureSizes;
                std::vector<int> pureCountPerHead;
                LimitedTagSupport limitedTagSupport;
                bool unfilterTailBuilt = false;
                if (p_opt.m_enableHybridDistance ||
                    p_opt.m_enableLimitedTagPosting) {
                    if (p_opt.m_tailReplicaCount <= 0 ||
                        p_opt.m_enableOrderedPageStart ||
                        p_opt.m_ssdIndexFileNum != 1 ||
                        p_opt.m_unfilterPureDistanceScanPercent !=
                            100) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Constrained single-posting build requires a positive "
                            "TailReplicaCount, EnableOrderedPageStart=false, SSDIndexFileNum=1, "
                            "and UnfilterPureDistanceScanPercent=100.\n");
                        return false;
                    }
                    std::vector<int> originalPureCountPerHead;
                    if (!AppendUnfilterTail(
                            selections,
                            postingListSize,
                            headVectorIDS,
                            fullVectors,
                            p_headIndex,
                            fullCount,
                            p_opt,
                            originalPureCountPerHead)) {
                        return false;
                    }
                    std::vector<int> originalPostingSizes(
                        postingListSize.size());
                    for (size_t head = 0;
                         head < postingListSize.size();
                         ++head) {
                        originalPostingSizes[head] =
                            postingListSize[head].load();
                        postingListSize[head] = 0;
                    }
                    Selection originalPosting(
                        0, p_opt.m_tmpdir,
                        "original_constrained_posting_tmp");
                    originalPosting.m_selections.swap(
                        selections.m_selections);
                    originalPosting.m_start = 0;
                    originalPosting.m_end =
                        originalPosting.m_selections.size();
                    originalPosting.m_totalsize =
                        originalPosting.m_end;
                    selections.m_start = 0;
                    selections.m_end = 0;
                    selections.m_totalsize = 0;

                    if (p_opt.m_enableHybridDistance) {
                        if (!BuildHybridPureSelections(
                                plannedNodeVectors,
                                selections,
                                postingListSize,
                                fullVectors,
                                fullCount,
                                p_headIndex->GetNumSamples(),
                                postingSizeLimit,
                                p_opt,
                                hybridHeadVectorIDs,
                                hybridCategoricalColumns)) {
                            return false;
                        }
                    } else {
                        if (!BuildLimitedTagPureSelections(
                                selections,
                                postingListSize,
                                headVectorIDS,
                                fullVectors,
                                p_headIndex,
                                fullCount,
                                postingSizeLimit,
                                p_opt,
                                limitedTagSupport)) {
                            return false;
                        }
                    }
                    hybridPureSizes.resize(
                        postingListSize.size());
                    for (size_t head = 0;
                         head < postingListSize.size();
                         ++head) {
                        hybridPureSizes[head] =
                            postingListSize[head].load();
                    }
                    if (!MergeConstrainedPureWithOriginalPosting(
                            selections,
                            postingListSize,
                            hybridPureSizes,
                            originalPosting,
                            originalPostingSizes)) {
                        return false;
                    }
                    pureCountPerHead =
                        hybridPureSizes;
                    unfilterTailBuilt = true;
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
                if (!unfilterTailBuilt &&
                    !AppendUnfilterTail(
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

                const std::string limitedTagSupportPath =
                    p_opt.m_indexDirectory + FolderSep +
                    p_opt.m_limitedTagSupportFile;
                if (p_opt.m_enableLimitedTagPosting) {
                    std::string supportError;
                    if (!limitedTagSupport.Save(
                            limitedTagSupportPath,
                            &supportError)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Cannot save limited-tag support metadata: %s\n",
                            supportError.c_str());
                        return false;
                    }
                } else if (!p_opt.m_limitedTagSupportFile.empty()) {
                    std::remove(
                        limitedTagSupportPath.c_str());
                }

                if (p_opt.m_enableHybridDistance) {
                    std::vector<int> fullPostingSizes(
                        postingListSize.size());
                    for (size_t head = 0;
                         head < postingListSize.size();
                         ++head) {
                        fullPostingSizes[head] =
                            postingListSize[head].load();
                    }
                    HybridRoutingStats routingStats;
                    routingStats.m_categoricalColumns =
                        hybridCategoricalColumns;
                    routingStats.m_numTagColumns =
                        m_staticNumTagsPerVec;
                    routingStats.m_headAttributes.resize(
                        hybridHeadVectorIDs.size() *
                        static_cast<size_t>(
                            m_staticNumTagsPerVec));
                    for (size_t head = 0;
                         head <
                         hybridHeadVectorIDs.size();
                         ++head) {
                        const SizeType vectorID =
                            hybridHeadVectorIDs[head];
                        if (vectorID < 0 ||
                            vectorID >= fullCount) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "Hybrid head %zu has no valid source vector.\n",
                                head);
                            return false;
                        }
                        std::copy_n(
                            m_staticBuildTags.data() +
                                static_cast<size_t>(
                                    vectorID) *
                                    static_cast<size_t>(
                                        m_staticNumTagsPerVec),
                            m_staticNumTagsPerVec,
                            routingStats
                                    .m_headAttributes
                                    .data() +
                                head *
                                    static_cast<size_t>(
                                        m_staticNumTagsPerVec));
                    }
                    routingStats.m_generationFingerprint =
                        m_hybridGenerationFingerprint;
                    if (!ComputeHybridRouteLayout(
                            selections,
                            hybridPureSizes,
                            hybridHeadVectorIDs,
                            hybridCategoricalColumns,
                            fullCount,
                            routingStats.m_hybrid,
                            &fullPostingSizes) ||
                        !ComputeHybridRouteLayout(
                            selections,
                            fullPostingSizes,
                            hybridHeadVectorIDs,
                            hybridCategoricalColumns,
                            fullCount,
                            routingStats.m_original,
                            &fullPostingSizes)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Cannot compute single-posting hybrid routing "
                            "statistics.\n");
                        return false;
                    }
                    std::fill(
                        routingStats.m_original
                            .m_enrichmentByMask.begin(),
                        routingStats.m_original
                            .m_enrichmentByMask.end(),
                        1.0);
                    std::string error;
                    const std::string statsPath =
                        outputFile + ".hybrid.stats";
                    if (!routingStats.Save(
                            statsPath, error)) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Cannot save hybrid routing statistics: %s\n",
                            error.c_str());
                        return false;
                    }
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Info,
                        "Saved hybrid-pure/full-posting routing statistics "
                        "to %s.\n",
                        statsPath.c_str());
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

            virtual bool CheckValidPosting(
                SizeType postingID,
                const ExtraWorkSpace* p_exWorkSpace) override
            {
                return postingID >= 0 &&
                    static_cast<size_t>(postingID) <
                        m_listInfos.size() &&
                    m_listInfos[static_cast<size_t>(
                        postingID)]
                            .listEleCount != 0;
            }

            virtual ErrorCode CheckPosting(SizeType postingID, std::vector<std::uint8_t> *visited = nullptr,
                                           ExtraWorkSpace *p_exWorkSpace = nullptr) override
            {
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
                const int totalListCount = GetTotalListCount(p_exWorkSpace);
                if (postingID < 0 || postingID >= totalListCount)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: Error postingID %d (should be 0 ~ %d)\n",
                                 postingID, totalListCount);
                    return ErrorCode::Key_OverFlow;
                }
                ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, postingID);
                if (listInfo == nullptr) {
                    return ErrorCode::Key_OverFlow;
                }
                if (listInfo->listEleCount < 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: postingID %d has wrong size:%d\n",
                                 postingID, listInfo->listEleCount);
                    return ErrorCode::Posting_SizeError;
                }
                return ErrorCode::Success;
            }

            virtual ErrorCode GetPostingDebug(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorIndex> p_index, SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs)
            {
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
                VIDs.clear();

                SizeType curPostingID = vid;
                ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, curPostingID);
                if (listInfo == nullptr) return ErrorCode::Key_OverFlow;
                VIDs.resize(listInfo->listEleCount);
                ByteArray vector_array = ByteArray::Alloc(sizeof(ValueType) * listInfo->listEleCount * m_iDataDimension);
                vecs.reset(new BasicVectorSet(vector_array, GetEnumValueType<ValueType>(), m_iDataDimension, listInfo->listEleCount));

                int fileid = GetPostingFileId(p_exWorkSpace, curPostingID);

#ifndef BATCH_READ
                Helper::DiskIO* indexFile = GetPostingIndexFile(p_exWorkSpace, fileid);
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
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", GetPostingFileBase(p_exWorkSpace).c_str(), totalBytes, numRead);
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

            bool DecompressStaticPosting(
                ExtraWorkSpace* p_exWorkSpace,
                char* p_buffer,
                const ListInfo* p_listInfo,
                char*& p_postingData)
            {
                p_postingData =
                    p_buffer +
                    p_listInfo->pageOffset;
                if (!m_enableDataCompression) {
                    return true;
                }
                p_postingData =
                    reinterpret_cast<char*>(
                        p_exWorkSpace
                            ->m_decompressBuffer
                            .GetBuffer());
                try {
                    const size_t decompressed =
                        m_pCompressor->Decompress(
                            p_buffer +
                                p_listInfo
                                    ->pageOffset,
                            p_listInfo
                                ->listTotalBytes,
                            p_postingData,
                            p_listInfo
                                    ->listEleCount *
                                m_vectorInfoSize,
                            m_enableDictTraining);
                    return decompressed ==
                        static_cast<size_t>(
                            p_listInfo
                                ->listEleCount) *
                            static_cast<size_t>(
                                m_vectorInfoSize);
                } catch (const std::runtime_error&
                             error) {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "PostingList %d decompression failed: %s\n",
                        GetListOrdinal(p_listInfo),
                        error.what());
                    return false;
                }
            }

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
                if (m_opt != nullptr) {
                    const int purePercent = m_opt->m_unfilterPureDistanceScanPercent;
                    if (purePercent < 1 || purePercent > 100) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "UnfilterPureDistanceScanPercent must be in [1,100], got %d.\n",
                            purePercent);
                        return true;
                    }
                    if (purePercent < 100) {
                        if (m_staticPipePQ) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "UnfilterPureDistanceScanPercent currently supports raw STATIC postings only.\n");
                            return true;
                        }
                        if (IsAttributeOrdered(p_exWorkSpace)) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "UnfilterPureDistanceScanPercent requires distance-ordered pure postings; "
                                "this index contains ordered_page_starts.bin and is attribute-ordered.\n");
                            return true;
                        }
                        if (m_opt->m_unfilterPurePages ||
                            m_opt->m_unfilterExtraTailPages > 0) {
                            SPTAGLIB_LOG(
                                Helper::LogLevel::LL_Error,
                                "UnfilterPureDistanceScanPercent retains the complete tail and cannot be "
                                "combined with UnfilterPurePages or UnfilterExtraTailPages.\n");
                            return true;
                        }
                    }
                }
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
                if (p_exWorkSpace != nullptr &&
                    p_exWorkSpace->m_scanFullPostingForFilter) {
                    return p_listInfo->listEleCount;
                }
                if (UseHybridPure(p_exWorkSpace)) {
                    return (std::max)(
                        0,
                        (std::min)(
                            p_listInfo->pureEleCount,
                            p_listInfo->listEleCount));
                }
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
                int pageCount = p_listInfo->listPageCount;
                if (scanCount < p_listInfo->listEleCount) {
                    const size_t bytes = static_cast<size_t>(p_listInfo->pageOffset) +
                        static_cast<size_t>(scanCount) * static_cast<size_t>(m_vectorInfoSize);
                    pageCount = static_cast<int>((bytes + PageSize - 1) >> PageSizeEx);
                } else if (m_opt != nullptr && m_staticHasMetadata &&
                           !m_hasHybridPurePostings &&
                           !HasStaticMetadataFilter(p_exWorkSpace) &&
                           (m_opt->m_unfilterPurePages ||
                            m_opt->m_unfilterExtraTailPages > 0)) {
                    const int pureCount = (std::max)(
                        0, (std::min)(p_listInfo->pureEleCount, p_listInfo->listEleCount));
                    int purePageCount = 0;
                    if (pureCount > 0) {
                        const size_t pureBytes = static_cast<size_t>(p_listInfo->pageOffset) +
                            static_cast<size_t>(pureCount) * static_cast<size_t>(m_vectorInfoSize);
                        purePageCount = static_cast<int>(
                            (pureBytes + PageSize - 1) >> PageSizeEx);
                    }
                    const int cappedPageCount = purePageCount +
                        (std::max)(0, m_opt->m_unfilterExtraTailPages);
                    pageCount = (std::min)(pageCount, cappedPageCount);
                }
                return pageCount;
            }

            bool TryOrderedPageStartQuery(const ExtraWorkSpace* p_exWorkSpace,
                                          size_t& p_attrIndex,
                                          std::int32_t& p_queryBit) const
            {
                if (p_exWorkSpace != nullptr &&
                    p_exWorkSpace->m_scanFullPostingForFilter) {
                    return false;
                }
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
                const int baseScanBegin = 0;
                range.m_scanBegin = baseScanBegin;
                range.m_scanEnd = scanLimit;
                const size_t scanBeginBytes =
                    static_cast<size_t>(
                        p_listInfo->pageOffset) +
                    static_cast<size_t>(
                        baseScanBegin) *
                        static_cast<size_t>(
                            m_vectorInfoSize);
                range.m_readStartPage =
                    static_cast<int>(
                        scanBeginBytes >>
                        PageSizeEx);
                range.m_readPageCount =
                    (std::max)(
                        0,
                        StaticReadPageCount(
                            p_exWorkSpace,
                            p_listInfo) -
                            range.m_readStartPage);
                const bool usePureDistancePrefix =
                    m_opt != nullptr &&
                    m_opt->m_unfilterPureDistanceScanPercent < 100 &&
                    !HasStaticMetadataFilter(p_exWorkSpace);
                if (usePureDistancePrefix) {
                    range.SetPureDistancePrefix(
                        p_listInfo->pureEleCount,
                        scanLimit,
                        m_opt->m_unfilterPureDistanceScanPercent);
                    if (range.m_secondScanEnd <= range.m_secondScanBegin) {
                        const size_t bytes = static_cast<size_t>(p_listInfo->pageOffset) +
                            static_cast<size_t>(range.m_scanEnd) *
                                static_cast<size_t>(m_vectorInfoSize);
                        range.m_readPageCount = bytes == 0
                            ? 0
                            : static_cast<int>((bytes + PageSize - 1) >> PageSizeEx);
                    }
                }
                if (range.m_readPageCount <= 0) {
                    range.m_scanEnd =
                        range.m_scanBegin;
                    range.m_secondScanBegin = -1;
                    range.m_secondScanEnd = -1;
                } else {
                    const std::int64_t endBytes =
                        static_cast<std::int64_t>(
                            range.m_readStartPage +
                            range.m_readPageCount) *
                            PageSize -
                        p_listInfo->pageOffset;
                    const int readableRecords = endBytes <= 0
                        ? 0
                        : static_cast<int>(endBytes / m_vectorInfoSize);
                    range.m_scanEnd = (std::min)(range.m_scanEnd, readableRecords);
                    if (range.m_secondScanEnd > range.m_secondScanBegin) {
                        range.m_secondScanBegin =
                            (std::min)(range.m_secondScanBegin, readableRecords);
                        range.m_secondScanEnd =
                            (std::min)(range.m_secondScanEnd, readableRecords);
                        if (range.m_secondScanEnd <= range.m_secondScanBegin) {
                            range.m_secondScanBegin = -1;
                            range.m_secondScanEnd = -1;
                        }
                    }
                }
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
                const int readEnd =
                    (std::min)(pageCount, (std::max)(lowerPage + 1, upperPage + 1));
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
                const int clampedBegin =
                    (std::max)(
                        range.m_scanBegin,
                        (std::min)(
                            scanBegin,
                            scanLimit));
                const int clampedEnd = (std::max)(clampedBegin, (std::min)(scanEnd, scanLimit));
                if (clampedEnd == clampedBegin) return range;

                range.m_scanBegin = clampedBegin;
                range.m_scanEnd = clampedEnd;
                range.m_secondScanBegin = -1;
                range.m_secondScanEnd = -1;
                range.m_readStartPage = readStart;
                range.m_readPageCount = readEnd - readStart;
                return range;
            }

            bool GetStaticScanRange(const ExtraWorkSpace* p_exWorkSpace,
                                    int p_slot,
                                    const ListInfo* p_listInfo,
                                    int p_range,
                                    int& p_begin,
                                    int& p_end) const
            {
                if (p_exWorkSpace != nullptr && p_slot >= 0 &&
                    static_cast<size_t>(p_slot) < p_exWorkSpace->m_postingReadRanges.size()) {
                    return p_exWorkSpace->m_postingReadRanges[static_cast<size_t>(p_slot)]
                        .GetScanRange(p_range, p_begin, p_end);
                }
                if (p_range != 0) return false;
                p_begin = 0;
                p_end = StaticScanLimit(p_exWorkSpace, p_listInfo);
                return p_end > p_begin;
            }

            int StaticScanRecordCount(const ExtraWorkSpace* p_exWorkSpace,
                                      int p_slot,
                                      const ListInfo* p_listInfo) const
            {
                if (p_exWorkSpace != nullptr && p_slot >= 0 &&
                    static_cast<size_t>(p_slot) < p_exWorkSpace->m_postingReadRanges.size()) {
                    return p_exWorkSpace->m_postingReadRanges[static_cast<size_t>(p_slot)]
                        .ScanCount();
                }
                return StaticScanLimit(p_exWorkSpace, p_listInfo);
            }

            bool NormalizeStaticScanOffset(const ExtraWorkSpace* p_exWorkSpace,
                                           int p_slot,
                                           const ListInfo* p_listInfo,
                                           int& p_offset) const
            {
                if (p_exWorkSpace != nullptr && p_slot >= 0 &&
                    static_cast<size_t>(p_slot) < p_exWorkSpace->m_postingReadRanges.size()) {
                    return p_exWorkSpace->m_postingReadRanges[static_cast<size_t>(p_slot)]
                        .NormalizeScanOffset(p_offset);
                }
                const int end = StaticScanLimit(p_exWorkSpace, p_listInfo);
                p_offset = (std::max)(0, p_offset);
                return p_offset < end;
            }

            int StaticScanBegin(const ExtraWorkSpace* p_exWorkSpace,
                                int p_slot,
                                const ListInfo* p_listInfo) const
            {
                if (p_exWorkSpace != nullptr && p_slot >= 0 &&
                    static_cast<size_t>(p_slot) < p_exWorkSpace->m_postingReadRanges.size()) {
                    int begin = 0;
                    if (p_exWorkSpace
                            ->m_postingReadRanges[
                                static_cast<size_t>(
                                    p_slot)]
                            .NormalizeScanOffset(
                                begin)) {
                        return begin;
                    }
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
            static constexpr int kStaticMetadataGenerationVersion = 2;
            static constexpr int kStaticMetadataGenerationHeaderInts = 11;

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
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
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
                    ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, postingId);
                    if (listInfo == nullptr) return ErrorCode::Key_OverFlow;
                    const int fileid = GetPostingFileId(p_exWorkSpace, postingId);
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
                if (!Helper::BatchReadFileAsync(GetPostingIndexFiles(p_exWorkSpace), p_exWorkSpace->m_diskRequests.data(),
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

            int LoadingHeadInfo(const std::string& p_file, std::vector<ListInfo>& p_listInfos)
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
                int metadataVersion = 0;
                m_staticLoadedGenerationFingerprint = 0;
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
                    int recordBytes = 0;
                    int numTagsPerVec = 0;
                    int tailPageBudget = 0;
                    if (!readInt(metadataVersion) || !readInt(m_listCount) || !readInt(m_totalDocumentCount) ||
                        !readInt(m_iDataDimension) || !readInt(recordBytes) ||
                        !readInt(numTagsPerVec) || !readInt(tailPageBudget)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                    "Failed to read static metadata header!\n");
                        throw std::runtime_error("Failed read static metadata header");
                    }
                    m_staticLoadedGenerationFingerprint = 0;
                    if (metadataVersion ==
                            kStaticMetadataGenerationVersion) {
                        int generationLo = 0;
                        int generationHi = 0;
                        if (!readInt(generationLo) ||
                           !readInt(generationHi)) {
                           throw std::runtime_error(
                               "Failed read static metadata generation");
                        }
                        m_staticLoadedGenerationFingerprint =
                           static_cast<std::uint64_t>(
                               static_cast<std::uint32_t>(
                                   generationLo)) |
                           (static_cast<std::uint64_t>(
                                static_cast<std::uint32_t>(
                                    generationHi))
                            << 32);
                        if (m_staticLoadedGenerationFingerprint ==
                            0) {
                            throw std::runtime_error(
                               "Generation-bound STM1 has a zero generation");
                        }
                    }
                    if (!readInt(m_listPageOffset)) {
                        throw std::runtime_error(
                           "Failed read static metadata page offset");
                    }
                    const int metadataBytes =
                        static_cast<int>(sizeof(int) +
                                         static_cast<size_t>(numTagsPerVec) * sizeof(uint32_t));
                    const int expectedRecordBytes =
                        m_iDataDimension * sizeof(ValueType) + metadataBytes;
                    if ((metadataVersion !=
                             kStaticMetadataVersion &&
                         metadataVersion !=
                             kStaticMetadataGenerationVersion) ||
                        m_staticPipePQ || numTagsPerVec <= 0 ||
                        recordBytes != expectedRecordBytes || tailPageBudget < -1) {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "Static metadata header mismatch: version=%d record=%d dim=%d tags=%d tailPages=%d\n",
                            metadataVersion, recordBytes, m_iDataDimension, numTagsPerVec, tailPageBudget);
                        throw std::runtime_error("Static metadata header mismatch");
                    }
                    if (m_opt != nullptr &&
                        (m_opt->m_enableHybridDistance ||
                         m_opt->m_enableLimitedTagPosting) &&
                        metadataVersion !=
                            kStaticMetadataGenerationVersion) {
                        throw std::runtime_error(
                            "Constrained pure mode requires generation-bound STM1 version 2; rebuild the index");
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
                    }
                    listInfo->listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
                    if (!m_enableDataCompression)
                    {
                        listInfo->listTotalBytes = listInfo->listEleCount * m_vectorInfoSize;
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
                p_postPageNum.reset(new int[p_postingListBytes.size()]());
                p_postPageOffset.reset(new std::uint16_t[p_postingListBytes.size()]());

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
                const std::vector<std::uint32_t>& p_orderedPageStartBases,
                bool p_manageOrderedPageStarts = true)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start output...\n");

                auto t1 = std::chrono::high_resolution_clock::now();
                if (!p_manageOrderedPageStarts && !p_orderedPageStartAttrs.empty()) {
                    throw std::runtime_error("Ordered page-start output is disabled for this sidecar");
                }
                std::string sidecarPath;
                if (p_manageOrderedPageStarts) {
                    const size_t sidecarSlash = p_outputFile.find_last_of(FolderSep);
                    sidecarPath =
                        (sidecarSlash == std::string::npos ? std::string() :
                                                            p_outputFile.substr(0, sidecarSlash + 1)) +
                        "ordered_page_starts.bin";
                }
                if (p_manageOrderedPageStarts &&
                    p_orderedPageStartAttrs.empty() &&
                    fileexists(sidecarPath.c_str())) {
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
                const bool staticMetadataGenerationHeader =
                    staticMetadataHeader &&
                    m_hybridGenerationFingerprint != 0;
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
                    staticMetadataHeader
                        ? sizeof(int) *
                              (staticMetadataGenerationHeader
                                   ? kStaticMetadataGenerationHeaderInts
                                   : kStaticMetadataHeaderInts)
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
                    writeHeaderInt(
                        staticMetadataGenerationHeader
                            ? kStaticMetadataGenerationVersion
                            : kStaticMetadataVersion);
                    writeHeaderInt(static_cast<int>(p_postingListSizes.size()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Count()));
                    writeHeaderInt(static_cast<int>(p_fullVectors->Dimension()));
                    writeHeaderInt(static_cast<int>(p_spacePerVector));
                    writeHeaderInt(m_staticNumTagsPerVec);
                    writeHeaderInt(p_tailPageBudget);
                    if (staticMetadataGenerationHeader) {
                        writeHeaderInt(
                            static_cast<int>(
                                static_cast<std::uint32_t>(
                                    m_hybridGenerationFingerprint)));
                        writeHeaderInt(
                            static_cast<int>(
                                static_cast<std::uint32_t>(
                                    m_hybridGenerationFingerprint >> 32)));
                    }
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
                if (p_manageOrderedPageStarts && !p_orderedPageStartAttrs.empty()) {
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

                    if (p_manageOrderedPageStarts && !p_orderedPageStartAttrs.empty()) {
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

                if (p_manageOrderedPageStarts && !p_orderedPageStartAttrs.empty()) {
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
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Info,
                    "Time to write results:%.2lf sec.\n",
                    std::chrono::duration<double>(t2 - t1).count());
            }

            ErrorCode GetWritePosting(ExtraWorkSpace* p_exWorkSpace, SizeType pid, std::string& posting, bool write = false) override {
                if (!ValidateHybridWorkspace(p_exWorkSpace)) return ErrorCode::Fail;
                if (write) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Unsupport write\n");
                    return ErrorCode::Undefined;
                }
                ListInfo* listInfo = GetPostingListInfo(p_exWorkSpace, pid);
                if (listInfo == nullptr) return ErrorCode::Key_OverFlow;
                size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                size_t realBytes = listInfo->listEleCount * m_vectorInfoSize;
                posting.resize(totalBytes);
                int fileid = GetPostingFileId(p_exWorkSpace, pid);
                Helper::DiskIO* indexFile = GetPostingIndexFile(p_exWorkSpace, fileid);
                auto numRead = indexFile->ReadBinary(totalBytes, (char*)posting.data(), listInfo->listOffset);
                if (numRead != totalBytes) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "File %s read bytes, expected: %zu, acutal: %llu.\n", GetPostingFileBase(p_exWorkSpace).c_str(), totalBytes, numRead);
                    return ErrorCode::DiskIOFail;
                }
                char* ptr = (char*)(posting.c_str());
                memcpy(ptr, posting.c_str() + listInfo->pageOffset, realBytes);
                posting.resize(realBytes);
                return ErrorCode::Success;
            }

        private:
            bool UseHybridPure(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return p_exWorkSpace != nullptr &&
                    p_exWorkSpace->m_useHybridPure;
            }

            bool ValidateHybridWorkspace(const ExtraWorkSpace* p_exWorkSpace) const
            {
                if (!UseHybridPure(p_exWorkSpace)) return true;
                if (m_hasHybridPurePostings) return true;
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "The hybrid pure prefix was requested, but the primary "
                    "posting does not contain a valid hybrid layout.\n");
                return false;
            }

            int GetTotalListCount(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return m_totalListCount;
            }

            std::vector<std::shared_ptr<Helper::DiskIO>>& GetPostingIndexFiles(
                const ExtraWorkSpace* p_exWorkSpace)
            {
                return m_indexFiles;
            }

            const std::vector<std::shared_ptr<Helper::DiskIO>>& GetPostingIndexFiles(
                const ExtraWorkSpace* p_exWorkSpace) const
            {
                return m_indexFiles;
            }

            const std::string& GetPostingFileBase(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return m_extraFullGraphFile;
            }

            bool IsAttributeOrdered(const ExtraWorkSpace* p_exWorkSpace) const
            {
                return m_staticAttributeOrdered;
            }

            ListInfo* GetPostingListInfo(const ExtraWorkSpace* p_exWorkSpace, SizeType p_postingID)
            {
                if (p_postingID < 0 ||
                    static_cast<size_t>(p_postingID) >=
                        m_listInfos.size()) {
                    return nullptr;
                }
                return &m_listInfos[
                    static_cast<size_t>(p_postingID)];
            }

            const ListInfo* GetPostingListInfo(const ExtraWorkSpace* p_exWorkSpace, SizeType p_postingID) const
            {
                if (p_postingID < 0 ||
                    static_cast<size_t>(p_postingID) >=
                        m_listInfos.size()) {
                    return nullptr;
                }
                return &m_listInfos[
                    static_cast<size_t>(p_postingID)];
            }

            int GetPostingFileId(const ExtraWorkSpace* p_exWorkSpace, SizeType p_postingID) const
            {
                if (m_oneContext) return 0;
                return m_listPerFile <= 0
                    ? 0
                    : static_cast<int>(
                          p_postingID /
                          m_listPerFile);
            }

            Helper::DiskIO* GetPostingIndexFile(const ExtraWorkSpace* p_exWorkSpace, int p_fileID) const
            {
                const auto& indexFiles = GetPostingIndexFiles(p_exWorkSpace);
                if (p_fileID < 0 || static_cast<size_t>(p_fileID) >= indexFiles.size()) return nullptr;
                return indexFiles[static_cast<size_t>(p_fileID)].get();
            }

            int GetListOrdinal(const ListInfo* p_listInfo) const
            {
                if (p_listInfo == nullptr) return -1;
                auto tryLocate = [p_listInfo](const std::vector<ListInfo>& p_listInfos) {
                    if (p_listInfos.empty()) return -1;
                    const ListInfo* begin = p_listInfos.data();
                    const ListInfo* end = begin + p_listInfos.size();
                    return (p_listInfo >= begin && p_listInfo < end)
                        ? static_cast<int>(p_listInfo - begin)
                        : -1;
                };
                return tryLocate(m_listInfos);
            }

            bool OpenStaticIndexFile(const std::string& p_file,
                                     const Options& p_opt,
                                     std::shared_ptr<Helper::DiskIO>& p_indexFile) const
            {
                p_indexFile = f_createAsyncIO();
#ifndef _MSC_VER
                const int staticOpenMode = O_RDONLY |
                    (p_opt.m_useDirectIO ? O_DIRECT : O_NOATIME);
#endif
                if (p_indexFile == nullptr || !p_indexFile->Initialize(p_file.c_str(),
#ifndef _MSC_VER
#ifdef BATCH_READ
                    staticOpenMode, p_opt.m_searchInternalResultNum, 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                    staticOpenMode, p_opt.m_searchInternalResultNum * p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
#else
                    GENERIC_READ, StaticWorkspaceBufferBytes(), 2, 2, (std::uint16_t)p_opt.m_ioThreads
#endif
                )) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file:%s!\n", p_file.c_str());
                    return false;
                }
                return true;
            }

            double ComputeAverageRecords(
                const std::vector<ListInfo>& p_listInfos) const
            {
                if (p_listInfos.empty()) return -1.0;
                double total = 0.0;
                for (const auto& listInfo : p_listInfos) {
                    total += listInfo.listEleCount;
                }
                return total / static_cast<double>(p_listInfos.size());
            }

            double ComputeAveragePages(
                const std::vector<ListInfo>& p_listInfos) const
            {
                if (p_listInfos.empty()) return -1.0;
                double total = 0.0;
                for (const auto& listInfo : p_listInfos) {
                    total += listInfo.listPageCount;
                }
                return total / static_cast<double>(p_listInfos.size());
            }

            double ComputeAverageBytes(
                const std::vector<ListInfo>& p_listInfos) const
            {
                if (p_listInfos.empty()) return -1.0;
                double total = 0.0;
                for (const auto& listInfo : p_listInfos) {
                    total += static_cast<double>(
                        listInfo.listTotalBytes);
                }
                return total / static_cast<double>(p_listInfos.size());
            }

            bool m_available = false;

            std::shared_ptr<Helper::Concurrent::ConcurrentQueue<int>> m_freeWorkSpaceIds;
            std::atomic<int> m_workspaceCount = 0;

            std::string m_extraFullGraphFile;
            std::uint64_t m_hybridGenerationFingerprint = 0;
            std::uint64_t m_staticLoadedGenerationFingerprint = 0;

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
            bool m_hasHybridPurePostings = false;
            double m_avgRecordsPerList = -1.0;
            double m_avgPagesPerList = -1.0;
            double m_avgBytesPerList = -1.0;
            double m_hybridAvgRecordsPerList = -1.0;
            double m_hybridAvgPagesPerList = -1.0;
            double m_hybridAvgBytesPerList = -1.0;

            bool m_staticPipePQ = false;
            int m_staticPipePQCodeBytes = 0;
            int m_staticPipePQDimension = 0;
            bool m_staticHasUnfilterTail = false;
            int m_staticTailPageBudget = 0;
            int m_staticMaxListPageCount = 0;
            int m_hybridMaxListPageCount = 0;
            bool m_staticHasMetadata = false;
            bool m_staticAttributeOrdered = false;
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
