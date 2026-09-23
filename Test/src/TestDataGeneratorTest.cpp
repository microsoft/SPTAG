// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/TestDataGenerator.h"

BOOST_AUTO_TEST_SUITE(TestDataGeneratorTest)

BOOST_AUTO_TEST_CASE(RecallUsesQueryAndBatchStrides)
{
    constexpr int queries = 3;
    constexpr int k = 3;
    constexpr int batches = 2;
    constexpr int rows = (batches + 1) * queries;
    auto bytes = SPTAG::ByteArray::Alloc(
        rows * k * (sizeof(SPTAG::SizeType) + sizeof(float)));
    auto* ids = reinterpret_cast<SPTAG::SizeType*>(bytes.Data());
    auto* distances = reinterpret_cast<float*>(bytes.Data() + rows * k * sizeof(SPTAG::SizeType));
    for (int i = 0; i < rows * k; ++i)
    {
        ids[i] = 100 + i;
        distances[i] = 1000.0f + i;
    }
    std::shared_ptr<SPTAG::VectorSet> truth = std::make_shared<SPTAG::BasicVectorSet>(
        bytes, SPTAG::VectorValueType::Float, k,
        rows * (sizeof(SPTAG::SizeType) + sizeof(float)) / sizeof(float));

    for (int batch = 0; batch <= batches; ++batch)
    {
        std::vector<SPTAG::QueryResult> results;
        for (int query = 0; query < queries; ++query)
        {
            results.emplace_back(nullptr, k, false);
            for (int neighbor = 0; neighbor < k; ++neighbor)
                results.back().SetResult(neighbor, ids[(batch * queries + query) * k + neighbor], 1.0f);
        }
        BOOST_CHECK_EQUAL(TestUtils::TestDataGenerator<std::uint8_t>::EvaluateRecall(
            results, truth, k, k, batch, batches), 1.0f);

        for (auto& result : results)
            result.SetResult(0, -1, 1.0f);
        BOOST_CHECK_CLOSE(TestUtils::TestDataGenerator<std::uint8_t>::EvaluateRecall(
            results, truth, k, k, batch, batches), 2.0f / 3.0f, 0.001f);
    }
}

BOOST_AUTO_TEST_SUITE_END()
