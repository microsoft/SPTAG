// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Core/SPANN/HybridHeadGraph.h"
#include "inc/Core/SPANN/HybridRoutingStats.h"
#include "inc/Test.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
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

BOOST_AUTO_TEST_CASE(RejectsOutOfRangeDNFColumns)
{
    Cache::DNFPredicate dnf;
    Cache::DNFClause clause;
    clause.lits.push_back(
        {std::numeric_limits<std::uint32_t>::max(),
         10, Cache::DNF_EQ, 0});
    dnf.clauses.push_back(clause);

    const std::uint32_t tags[] = {10};
    BOOST_CHECK(!dnf.Matches(tags, 1));
}

BOOST_AUTO_TEST_CASE(EstimatesNprobeIndependentRouteDeformation)
{
    std::array<float, 8> vectorComponents = {
        8.0f, 4.0f, 2.0f, 7.0f,
        1.0f, 6.0f, 3.0f, 5.0f};
    const std::array<double, 8>
        attributeDistances = {
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0};
    const auto estimate =
        EstimateHybridRouteDeformation(
            vectorComponents.data(),
            attributeDistances.data(),
            vectorComponents.size());
    BOOST_REQUIRE(estimate.m_valid);
    BOOST_CHECK_EQUAL(estimate.m_samples, 8);
    BOOST_CHECK_CLOSE(
        estimate.m_attributeRMS, 2.0, 0.001);
    BOOST_CHECK_CLOSE(
        estimate.m_nearVectorSpan, 2.0, 0.001);
    BOOST_CHECK_CLOSE(
        estimate.m_deformation, 1.0, 0.001);

    BOOST_CHECK(ShouldUseHybridRoute(
        0.05, estimate, 0.1, 1.0));
    BOOST_CHECK(!ShouldUseHybridRoute(
        0.2, estimate, 0.1, 1.0));
    BOOST_CHECK(!ShouldUseHybridRoute(
        0.05, estimate, 0.1, 1.01));

    std::array<float, 2> invalidVectors = {
        1.0f, 2.0f};
    const std::array<double, 2>
        invalidAttributes = {
            0.0,
            (std::numeric_limits<double>::infinity)()};
    BOOST_CHECK(
        !EstimateHybridRouteDeformation(
             invalidVectors.data(),
             invalidAttributes.data(),
             invalidVectors.size())
             .m_valid);
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
    left.AddEdgeBody(1234);
    HybridGenerationFingerprint right(config, 3, 4, 16);
    right.AddHead(9, second);
    right.AddHead(7, first);
    right.AddEdgeBody(1234);
    BOOST_CHECK_EQUAL(left.Value(), right.Value());

    HybridGenerationFingerprint changed(config, 3, 5, 16);
    changed.AddHead(7, first);
    changed.AddHead(9, second);
    changed.AddEdgeBody(1234);
    BOOST_CHECK_NE(left.Value(), changed.Value());

    HybridGenerationFingerprint changedBody(
        config, 3, 4, 16);
    changedBody.AddHead(7, first);
    changedBody.AddHead(9, second);
    changedBody.AddEdgeBody(1235);
    BOOST_CHECK_NE(left.Value(), changedBody.Value());
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

BOOST_AUTO_TEST_CASE(RejectsIncludedHeadPostings)
{
    auto index = VectorIndex::CreateInstance(
        IndexAlgoType::SPANN,
        VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(
        index->SetParameter(
            "IndexAlgoType", "BKT",
            "Base") == ErrorCode::Success);
    BOOST_REQUIRE(
        index->SetParameter(
            "EnableHybridDistance", "true",
            "BuildSSDIndex") ==
        ErrorCode::Success);
    BOOST_REQUIRE(
        index->SetParameter(
            "ExcludeHead", "false",
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

BOOST_AUTO_TEST_CASE(PersistsHybridEdgesInCrossEdgeFormat)
{
    const std::string path =
        "hybrid_head_cross_edges_test.bin";
    std::filesystem::remove(path);

    HybridHeadGraph graph;
    graph.m_numTagColumns = 2;
    graph.m_degree = 2;
    graph.m_generationFingerprint = 1234;
    graph.m_contentFingerprint = 5678;
    graph.m_nodes.resize(1);
    graph.m_nodes[0].m_nodeID = 0;
    graph.m_nodes[0].m_headCount = 2;
    graph.m_nodes[0].m_attributes = {10, 20, 10, 21};
    graph.m_nodes[0].m_neighbors = {1, -1, 0, -1};
    Helper::HeadCrossEdgesBodyFingerprint bodyFingerprint;
    bodyFingerprint.AddRecord(100, 1);
    bodyFingerprint.AddEntry({200, 0.0f});
    bodyFingerprint.AddRecord(200, 1);
    bodyFingerprint.AddEntry({100, 0.0f});
    graph.m_edgeBodyFingerprint =
        bodyFingerprint.Value();

    std::string error;
    BOOST_REQUIRE(
        graph.SaveCrossEdges(
            path, {{100, 200}}, 8, error));
    std::ifstream input(path, std::ios::binary);
    BOOST_REQUIRE(input.good());
    Helper::HeadCrossEdgesHeader header{};
    input.read(
        reinterpret_cast<char*>(&header),
        sizeof(header));
    BOOST_REQUIRE(input.good());
    BOOST_CHECK_EQUAL(
        header.magic,
        Helper::kHeadCrossEdgesMagic);
    BOOST_CHECK_EQUAL(
        header.version,
        Helper::kHybridHeadCrossEdgesVersion);
    BOOST_CHECK_EQUAL(header.totalHeads, 2);
    BOOST_CHECK_EQUAL(header.maxEdgesPerHead, 2);
    BOOST_CHECK_EQUAL(
        header.reserved,
        Helper::kHybridHeadCrossEdgesMarker);
    Helper::HybridHeadCrossEdgesExtension extension{};
    input.read(
        reinterpret_cast<char*>(&extension),
        sizeof(extension));
    BOOST_REQUIRE(input.good());
    BOOST_CHECK_EQUAL(
        extension.generationFingerprint, 1234);
    BOOST_CHECK_EQUAL(
        extension.contentFingerprint, 5678);
    const std::array<std::int32_t, 2>
        expectedSources = {100, 200};
    const std::array<std::int32_t, 2>
        expectedTargets = {200, 100};
    for (size_t record = 0;
         record < expectedSources.size();
         ++record) {
        std::int32_t source = -1;
        std::int32_t edgeCount = 0;
        Helper::HeadCrossEdgeEntry edge{};
        input.read(
            reinterpret_cast<char*>(&source),
            sizeof(source));
        input.read(
            reinterpret_cast<char*>(&edgeCount),
            sizeof(edgeCount));
        input.read(
            reinterpret_cast<char*>(&edge),
            sizeof(edge));
        BOOST_REQUIRE(input.good());
        BOOST_CHECK_EQUAL(
            source, expectedSources[record]);
        BOOST_CHECK_EQUAL(edgeCount, 1);
        BOOST_CHECK_EQUAL(
            edge.neighborGlobalVID,
            expectedTargets[record]);
    }
    std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(PersistsRoutingStatisticsAndConservativeDNFMask)
{
    const std::string path = "hybrid_route_stats_test.bin";
    std::filesystem::remove(path);

    HybridRoutingStats stats;
    stats.m_categoricalColumns = {0, 2};
    stats.m_numTagColumns = 3;
    stats.m_headAttributes = {
        10, 11, 12,
        20, 21, 22};
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
    BOOST_REQUIRE(
        loaded.Load(path, 3, 2, 1234, error));
    BOOST_CHECK_CLOSE(
        loaded.Enrichment(true, 3), 6.0, 0.001);
    BOOST_CHECK_CLOSE(
        loaded.Enrichment(false, 3), 1.0, 0.001);
    BOOST_CHECK_EQUAL(loaded.HeadCount(), 2);
    BOOST_REQUIRE(
        loaded.HeadAttributes(1) != nullptr);
    BOOST_CHECK_EQUAL(
        loaded.HeadAttributes(1)[2], 22);

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

    BOOST_CHECK(
        !loaded.Load(path, 3, 3, 1234, error));
    BOOST_CHECK(
        !loaded.Load(path, 3, 2, 4321, error));
    {
        std::fstream corrupt(
            path,
            std::ios::in | std::ios::out |
                std::ios::binary);
        BOOST_REQUIRE(corrupt.good());
        const std::int32_t excessiveHeads =
            (std::numeric_limits<std::int32_t>::max)();
        corrupt.seekp(
            offsetof(
                HybridRoutingStatsHeader,
                m_headCount));
        corrupt.write(
            reinterpret_cast<const char*>(
                &excessiveHeads),
            sizeof(excessiveHeads));
    }
    BOOST_CHECK(
        !loaded.Load(path, 3, 2, 1234, error));
    {
        std::fstream corrupt(
            path,
            std::ios::in | std::ios::out |
                std::ios::binary);
        BOOST_REQUIRE(corrupt.good());
        const std::int32_t headCount = 2;
        corrupt.seekp(
            offsetof(
                HybridRoutingStatsHeader,
                m_headCount));
        corrupt.write(
            reinterpret_cast<const char*>(
                &headCount),
            sizeof(headCount));
        corrupt.seekp(0);
        const std::uint32_t badMagic = 0;
        corrupt.write(
            reinterpret_cast<const char*>(&badMagic),
            sizeof(badMagic));
    }
    BOOST_CHECK(
        !loaded.Load(path, 3, 2, 1234, error));
    BOOST_CHECK(!error.empty());
    std::filesystem::remove(path);
}

BOOST_AUTO_TEST_SUITE_END()
