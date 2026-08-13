// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Core/SPANN/HybridHeadGraph.h"
#include "inc/Core/SPANN/HybridRoutingStats.h"
#include "inc/Test.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace SPTAG;
using namespace SPTAG::SPANN;

BOOST_AUTO_TEST_SUITE(HybridDistanceTest)

BOOST_AUTO_TEST_CASE(ParsesAndComputesConfiguredDistance)
{
    HybridDistanceConfig config;
    std::string error;
    BOOST_REQUIRE(HybridDistanceConfig::Parse(
        "0", "3", "2", "2", 3, 0.5f, config, error));

    const std::uint32_t left[] = {10, 20, 100};
    const std::uint32_t right[] = {11, 20, 104};
    BOOST_CHECK_CLOSE(
        config.PairDistance(10.0f, left, right, 3),
        16.0f, 0.001f);
    BOOST_CHECK_CLOSE(
        config.PairDistance(-10.0f, left, right, 3),
        6.0f, 0.001f);

    const float unnormalizedQuery[] = {30.0f, 40.0f};
    const auto cosineTransform =
        HybridQueryDistanceTransform::ForCosine(
            unnormalizedQuery, 2);
    BOOST_CHECK_SMALL(
        cosineTransform.Apply(-49.0f), 0.001f);
    BOOST_CHECK_CLOSE(
        cosineTransform.Apply(1.0f), 1.0f, 0.001f);

    HybridDistanceConfig invalid;
    BOOST_CHECK(!HybridDistanceConfig::Parse(
        "0", "1", "0", "1", 3, 1.0f, invalid, error));
    BOOST_CHECK(!error.empty());
}

BOOST_AUTO_TEST_CASE(UsesOnlyConstrainedQueryAttributes)
{
    HybridDistanceConfig config;
    std::string error;
    BOOST_REQUIRE(HybridDistanceConfig::Parse(
        "0", "3", "2", "2", 3, 1.0f, config, error));

    Cache::DNFPredicate dnf;
    Cache::DNFClause clause;
    clause.lits.push_back({0, 10, Cache::DNF_EQ, 0});
    clause.lits.push_back({2, 100, Cache::DNF_GE, 1});
    clause.lits.push_back({2, 200, Cache::DNF_LE, 1});
    dnf.clauses.push_back(clause);

    const std::uint32_t outside[] = {11, 99, 250};
    BOOST_CHECK_CLOSE(
        config.PredicateDistance(outside, 3, &dnf, {}),
        103.0, 0.001);

    const std::uint32_t inside[] = {10, 9999, 150};
    BOOST_CHECK_SMALL(
        config.PredicateDistance(inside, 3, &dnf, {}), 1e-9);

    const std::vector<std::pair<int, std::uint32_t>> flat = {{0, 10}};
    BOOST_CHECK_SMALL(
        config.PredicateDistance(inside, 3, nullptr, flat), 1e-9);
    BOOST_CHECK_CLOSE(
        config.PredicateDistance(outside, 3, nullptr, flat),
        3.0, 0.001);

    HybridDistanceConfig flatOrConfig;
    BOOST_REQUIRE(HybridDistanceConfig::Parse(
        "0,1", "3,7", "", "", 3, 1.0f,
        flatOrConfig, error));
    const std::vector<std::pair<int, std::uint32_t>>
        crossColumnOr = {{0, 10}, {1, 20}};
    const std::uint32_t flatValues[] =
        {11, 20, 250};
    BOOST_CHECK_SMALL(
        flatOrConfig.PredicateDistance(
            flatValues, 3, nullptr,
            crossColumnOr),
        1e-9);
    const std::vector<std::pair<int, std::uint32_t>>
        noMatchOr = {{0, 10}, {1, 21}};
    BOOST_CHECK_CLOSE(
        flatOrConfig.PredicateDistance(
            flatValues, 3, nullptr, noMatchOr),
        3.0, 0.001);
    const std::vector<std::pair<int, std::uint32_t>>
        unweightedAlternative = {{0, 10}, {2, 250}};
    BOOST_CHECK_SMALL(
        flatOrConfig.PredicateDistance(
            flatValues, 3, nullptr,
            unweightedAlternative),
        1e-9);
}

BOOST_AUTO_TEST_CASE(EstimatesCostFromYieldAndPhysicalLayout)
{
    HybridPostingLayoutStats layout;
    layout.m_averageRecords = 100.0;
    layout.m_averagePages = 2.0;
    layout.m_averageBytes = 8192.0;
    layout.m_uniqueRatio = 0.5;
    layout.m_enrichment = 4.0;
    layout.m_headFixedCostUS = 5.0;
    layout.m_headPerPostingCostUS = 0.5;

    HybridRouteCostConfig cost;
    cost.m_resultSafety = 2.0;
    cost.m_ioFixedUS = 8.0;
    cost.m_pageUS = 4.0;
    cost.m_vectorUS = 0.04;
    cost.m_bytesPerUS = 4096.0;

    const auto estimate = EstimateHybridRouteCost(
        10, 5, 100, 0.01, layout, cost);
    BOOST_CHECK_CLOSE(
        estimate.m_expectedMatchesPerPosting, 2.0, 0.001);
    BOOST_CHECK_EQUAL(estimate.m_postings, 10);
    BOOST_CHECK_CLOSE(estimate.m_costUS, 230.0, 0.001);

    const auto capped = EstimateHybridRouteCost(
        10, 5, 2048, 1e-300, layout, cost);
    BOOST_CHECK_EQUAL(capped.m_postings, 2048);
}

BOOST_AUTO_TEST_CASE(FingerprintIsHeadOrderIndependentAndConfigSensitive)
{
    HybridDistanceConfig config;
    std::string error;
    BOOST_REQUIRE(HybridDistanceConfig::Parse(
        "0", "3", "2", "2", 3, 0.5f, config, error));
    const std::uint32_t first[] = {10, 20, 100};
    const std::uint32_t second[] = {11, 20, 104};

    HybridGenerationFingerprint left(config, 3, 4, 16);
    left.AddHead(7, first);
    left.AddHead(9, second);
    HybridGenerationFingerprint right(config, 3, 4, 16);
    right.AddHead(9, second);
    right.AddHead(7, first);
    BOOST_CHECK_EQUAL(left.Value(), right.Value());

    HybridGenerationFingerprint changed(config, 3, 5, 16);
    changed.AddHead(7, first);
    changed.AddHead(9, second);
    BOOST_CHECK_NE(left.Value(), changed.Value());
}

BOOST_AUTO_TEST_CASE(RejectsKDTHeadBundles)
{
    auto index = VectorIndex::CreateInstance(
        IndexAlgoType::SPANN,
        VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(
        index->SetParameter(
            "IndexAlgoType", "KDT",
            "Base") == ErrorCode::Success);
    BOOST_REQUIRE(
        index->SetParameter(
            "EnableHybridDistance", "true",
            "BuildSSDIndex") ==
        ErrorCode::Success);

    const float vectors[] = {
        0.0f, 0.0f,
        1.0f, 1.0f};
    BOOST_CHECK(
        index->BuildIndex(
            vectors, 2, 2, false,
            false) ==
        ErrorCode::FailedParseValue);
}

BOOST_AUTO_TEST_CASE(PersistsStrictHybridGraphTopology)
{
    const std::string path = "hybrid_head_graph_test.bin";
    std::filesystem::remove(path);

    HybridHeadGraph graph;
    graph.m_numTagColumns = 2;
    graph.m_degree = 2;
    graph.m_generationFingerprint = 1234;
    graph.m_nodes.resize(1);
    graph.m_nodes[0].m_nodeID = 0;
    graph.m_nodes[0].m_headCount = 2;
    graph.m_nodes[0].m_attributes = {10, 20, 10, 21};
    graph.m_nodes[0].m_neighbors = {1, -1, 0, -1};

    std::string error;
    BOOST_REQUIRE(graph.Save(path, error));
    HybridHeadGraph loaded;
    BOOST_REQUIRE(loaded.Load(path, {2}, 2, 2, error));
    BOOST_CHECK_EQUAL(loaded.TotalHeads(), 2);
    BOOST_CHECK_EQUAL(loaded.TotalEdges(), 2);
    BOOST_CHECK_EQUAL(loaded.m_nodes[0].m_attributes[3], 21);

    HybridHeadGraph rejected;
    BOOST_CHECK(!rejected.Load(path, {3}, 2, 2, error));
    BOOST_CHECK(!error.empty());
    std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(PersistsRoutingStatisticsAndConservativeDNFMask)
{
    const std::string path = "hybrid_route_stats_test.bin";
    std::filesystem::remove(path);

    HybridRoutingStats stats;
    stats.m_categoricalColumns = {0, 2};
    stats.m_generationFingerprint = 1234;
    stats.m_original.m_layout = {
        100.0, 5.0, 20480.0, 0.5};
    stats.m_hybrid.m_layout = {
        80.0, 4.0, 16384.0, 0.75};
    stats.m_original.m_enrichmentByMask =
        {1.0, 1.2, 1.1, 1.3};
    stats.m_hybrid.m_enrichmentByMask =
        {1.0, 2.0, 3.0, 6.0};

    std::string error;
    BOOST_REQUIRE(stats.Save(path, error));
    HybridRoutingStats loaded;
    BOOST_REQUIRE(loaded.Load(path, error));
    BOOST_CHECK_CLOSE(
        loaded.Enrichment(true, 3), 6.0, 0.001);

    Cache::DNFPredicate dnf;
    Cache::DNFClause first;
    first.lits.push_back(
        {0, 10, Cache::DNF_EQ, 0});
    first.lits.push_back(
        {2, 20, Cache::DNF_EQ, 0});
    Cache::DNFClause second;
    second.lits.push_back(
        {0, 11, Cache::DNF_EQ, 0});
    dnf.clauses = {first, second};
    BOOST_CHECK_EQUAL(
        loaded.ConfiguredMask(&dnf, {}), 1);
    BOOST_CHECK_EQUAL(
        loaded.ConfiguredMask(
            nullptr, {{0, 10}, {0, 11}}),
        1);
    BOOST_CHECK_EQUAL(
        loaded.ConfiguredMask(
            nullptr, {{0, 10}, {2, 20}}),
        0);
    BOOST_CHECK_EQUAL(
        loaded.ConfiguredMask(
            nullptr, {{0, 10}, {1, 20}}),
        0);

    {
        std::fstream corrupt(
            path,
            std::ios::in | std::ios::out |
                std::ios::binary);
        BOOST_REQUIRE(corrupt.good());
        const std::uint32_t badMagic = 0;
        corrupt.write(
            reinterpret_cast<const char*>(&badMagic),
            sizeof(badMagic));
    }
    BOOST_CHECK(!loaded.Load(path, error));
    BOOST_CHECK(!error.empty());
    std::filesystem::remove(path);
}

BOOST_AUTO_TEST_SUITE_END()
