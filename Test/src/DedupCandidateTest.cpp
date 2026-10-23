// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Test.h"

#include <limits>
#include <set>

namespace
{
    class PostingStore : public SPTAG::Helper::KeyValueIO
    {
    public:
        std::string posting;

        void ShutDown() override {}

        SPTAG::ErrorCode MultiGet(const std::vector<SPTAG::SizeType>& keys,
            std::vector<SPTAG::Helper::PageBuffer<std::uint8_t>>& values,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            for (std::size_t i = 0; i < keys.size(); ++i) {
                values[i].ReservePageBuffer(posting.size());
                memcpy(values[i].GetBuffer(), posting.data(), posting.size());
                values[i].SetAvailableSize(posting.size());
            }
            return SPTAG::ErrorCode::Success;
        }

        SPTAG::ErrorCode Get(SPTAG::SizeType, std::string*, const std::chrono::microseconds&,
            std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            BOOST_ERROR("Unexpected Get");
            return SPTAG::ErrorCode::Undefined;
        }

        SPTAG::ErrorCode MultiGet(const std::vector<SPTAG::SizeType>&, std::vector<std::string>*,
            const std::chrono::microseconds&, std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            BOOST_ERROR("Unexpected string MultiGet");
            return SPTAG::ErrorCode::Undefined;
        }

        SPTAG::ErrorCode Put(SPTAG::SizeType, const std::string&, const std::chrono::microseconds&,
            std::vector<SPTAG::Helper::AsyncReadRequest>*) override
        {
            BOOST_ERROR("Unexpected Put");
            return SPTAG::ErrorCode::Undefined;
        }

        SPTAG::ErrorCode Merge(SPTAG::SizeType, const std::string&, const std::chrono::microseconds&,
            std::vector<SPTAG::Helper::AsyncReadRequest>*, int&) override
        {
            BOOST_ERROR("Unexpected Merge");
            return SPTAG::ErrorCode::Undefined;
        }

        SPTAG::ErrorCode Delete(SPTAG::SizeType) override
        {
            BOOST_ERROR("Unexpected Delete");
            return SPTAG::ErrorCode::Undefined;
        }
    };

    class CollisionDeduper : public SPTAG::COMMON::OptHashPosVector
    {
    public:
        std::uint64_t Bucket(SPTAG::SizeType id) const
        {
            return hash_func(static_cast<std::uint64_t>(id + 1), m_poolSize);
        }

        bool UsesSecondBlock() const { return m_secondHash; }
    };
}

BOOST_AUTO_TEST_SUITE(DedupCandidateTest)

BOOST_AUTO_TEST_CASE(SearchPathsMatchDeleteFirstAcrossVisibilityInterleavings)
{
    using namespace SPTAG;
    for (bool isTiKV : {false, true})
    for (bool iterative : {false, true})
    for (bool asyncMerge : {false, true})
    for (bool useOverride : {false, true})
    {
        SPANN::Options options;
        options.m_dim = 1;
        options.m_storage = isTiKV ? Storage::TIKVIO : Storage::FILEIO;
        options.m_asyncMergeInSearch = asyncMerge;
        options.m_mergeThreshold = -1; // No merge jobs in this search-only fixture.
        options.m_distributedVersionMap = false;
        SPANN::Index<std::uint8_t> head;
        head.SetParameter("Dim", "1", "Base");
        head.SetParameter("DistCalcMethod", "L2", "Base");
        auto store = std::make_shared<PostingStore>();
        SPANN::ExtraDynamicSearcher<std::uint8_t> searcher(options, 0, &head, store);
        SPANN::ExtraWorkSpace workspace;
        workspace.Initialize(64, 2, 2, 4096, false, false);
        workspace.m_postingIDs = {10, 11};
        SPANN::ExtraWorkSpace::IteratorLayerState state;
        state.Reset(64, 2);
        if (useOverride) workspace.m_deduperOverride = state.m_deduper.get();
        // Keep sparse-map Count() above the test IDs for the existing iterative range check.
        for (SizeType id = 3; id < 8; ++id) searcher.ResetIndex(id);
        const std::uint8_t target = 0;
        const std::size_t recordSize = sizeof(SizeType) + 2;
        store->posting.resize(recordSize * 2);

        // Exhaust five-record visibility sequences through the actual search entry points.
        for (int sequence = 0; sequence < 7776; ++sequence)
        {
            workspace.m_deduper.clear();
            state.Reset(64, 2);
            if (useOverride) {
                // The local deduper must be ignored while the override is active.
                for (SizeType id = 0; id < 3; ++id) workspace.m_deduper.CheckAndSet(id);
            }
            std::set<SizeType> referenceSeen;
            int remaining = sequence;
            for (int record = 0; record < 5; ++record)
            {
                const int event = remaining % 6;
                remaining /= 6;
                const SizeType id = event / 2;
                const bool live = (event % 2) != 0;
                if (live) searcher.ResetIndex(id);
                else searcher.DeleteIndex(id);
                const bool newCandidate = (isTiKV || live) && referenceSeen.insert(id).second;
                const bool expected = live && newCandidate;
                const auto value = static_cast<std::uint8_t>(id);
                searcher.Serialize(&store->posting[0], id, 0, &value);
                searcher.Serialize(&store->posting[recordSize], id, 0, &value);
                COMMON::QueryResultSet<std::uint8_t> query(&target, 4);
                if (iterative) {
                    std::vector<BasicResult> results;
                    BOOST_REQUIRE(searcher.SearchIndexIterativeScan(&workspace, query, results, false)
                        == ErrorCode::Success);
                    BOOST_REQUIRE_EQUAL(results.size(), expected ? 1 : 0);
                    if (expected) BOOST_CHECK_EQUAL(results[0].VID, id);
                } else {
                    SPANN::SearchStats stats;
                    BOOST_REQUIRE(searcher.SearchIndex(&workspace, query, &stats, nullptr, nullptr, false)
                        == ErrorCode::Success);
                    int accepted = 0;
                    for (int i = 0; i < query.GetResultNum(); ++i) {
                        if (query.GetResult(i)->VID < 0) continue;
                        ++accepted;
                        BOOST_CHECK_EQUAL(query.GetResult(i)->VID, id);
                    }
                    BOOST_CHECK_EQUAL(accepted, expected ? 1 : 0);
                    BOOST_CHECK_EQUAL(query.GetScanned(), newCandidate ? 1 : 0);
                    BOOST_CHECK_EQUAL(stats.m_totalListElementsCount, newCandidate ? 1 : 0);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(SearchPreservesMaxHeapAfterVersionFiltering)
{
    using namespace SPTAG;
    for (bool isTiKV : {false, true})
    for (int deletedID : {-1, 0, 31, 63, 64})
    {
        SPANN::Options options;
        options.m_dim = 1;
        options.m_storage = isTiKV ? Storage::TIKVIO : Storage::FILEIO;
        options.m_asyncMergeInSearch = false;
        options.m_distributedVersionMap = false;
        SPANN::Index<std::uint8_t> head;
        head.SetParameter("Dim", "1", "Base");
        head.SetParameter("DistCalcMethod", "L2", "Base");
        auto store = std::make_shared<PostingStore>();
        SPANN::ExtraDynamicSearcher<std::uint8_t> searcher(options, 0, &head, store);
        constexpr std::size_t recordSize = sizeof(SizeType) + 2;
        store->posting.resize(64 * recordSize);
        // Keep sparse-map Count() above all posting IDs even after deleting every posting.
        for (SizeType id = 0; id < 128; ++id) searcher.ResetIndex(id);
        for (SizeType id = 0; id < 64; ++id) {
            const auto value = static_cast<std::uint8_t>(id + 1);
            searcher.Serialize(store->posting.data() + id * recordSize, id, 0xff, &value);
            if (id == deletedID || deletedID == 64) searcher.DeleteIndex(id);
        }

        SPANN::ExtraWorkSpace workspace;
        workspace.Initialize(64, 2, 2, 4096, false, false);
        workspace.m_postingIDs = {100};
        const std::uint8_t target = 0;
        COMMON::QueryResultSet<std::uint8_t> query(&target, 64);
        SPANN::SearchStats stats;
        BOOST_REQUIRE(searcher.SearchIndex(&workspace, query, &stats, nullptr, nullptr, true)
            == ErrorCode::Success);
        BOOST_REQUIRE(std::is_heap(query.GetResults(), query.GetResults() + 64, COMMON::Compare));
        BOOST_REQUIRE(query.AddPoint(1000, 0.25f));
        query.SortResult();

        std::vector<SizeType> expected{1000};
        for (SizeType id = 0; id < 64 && expected.size() < 64; ++id) {
            if (id != deletedID && deletedID != 64) expected.push_back(id);
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            BOOST_TEST_CONTEXT("TiKV=" << isTiKV << ", deletedID=" << deletedID << ", result=" << i) {
                BOOST_CHECK_EQUAL(query.GetResult(i)->VID, expected[i]);
            }
        }
        for (std::size_t i = expected.size(); i < 64; ++i) {
            BOOST_CHECK_EQUAL(query.GetResult(i)->VID, -1);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
