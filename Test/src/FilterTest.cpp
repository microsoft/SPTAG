// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/BKT/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Test.h"

#include <chrono>
#include <unordered_set>

template <typename T>
void BuildWithMetaMapping(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet> &vec,
                          std::shared_ptr<SPTAG::MetadataSet> &meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex =
        SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "16");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta, true));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T> void SearchWithFilter(const std::string folder, T *vec, SPTAG::SizeType n, int k)
{
    std::cout << "start search with filter" << std::endl;
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(folder, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);
    std::string value = "2";
    std::function<bool(const SPTAG::ByteArray &)> filterFunction = [value](const SPTAG::ByteArray &meta) -> bool {
        std::string metaValue((char *)meta.Data(), meta.Length());
        std::cout << metaValue << std::endl;
        return metaValue != value;
    };
    for (SPTAG::SizeType i = 0; i < n; i++)
    {
        SPTAG::QueryResult res(vec, k, true);
        vecIndex->SearchIndexWithFilter(res, filterFunction);
        std::unordered_set<std::string> resmeta;
        for (int j = 0; j < k; j++)
        {
            resmeta.insert(std::string((char *)res.GetMetadata(j).Data(), res.GetMetadata(j).Length()));
            std::cout << res.GetResult(j)->Dist << "@(" << res.GetResult(j)->VID << ","
                      << std::string((char *)res.GetMetadata(j).Data(), res.GetMetadata(j).Length()) << ") ";
        }
        std::cout << std::endl;
        for (int j = 0; j < k; j++)
        {
            BOOST_CHECK(resmeta.find("2") == resmeta.end());
        }
        vec += vecIndex->GetFeatureDim();
    }
    vecIndex.reset();
}

template <typename T> void FTest(SPTAG::IndexAlgoType algo, std::string distCalcMethod)
{
    SPTAG::SizeType n = 2000, q = 3;
    SPTAG::DimensionType m = 10;
    int k = 3;
    std::vector<T> vec;
    for (SPTAG::SizeType i = 0; i < n; i++)
    {
        for (SPTAG::DimensionType j = 0; j < m; j++)
        {
            vec.push_back((T)i);
        }
    }

    std::vector<T> query;
    for (SPTAG::SizeType i = 0; i < q; i++)
    {
        for (SPTAG::DimensionType j = 0; j < m; j++)
        {
            query.push_back((T)i * 2);
        }
    }

    std::vector<char> meta;
    std::vector<std::uint64_t> metaoffset;
    for (SPTAG::SizeType i = 0; i < n; i++)
    {
        metaoffset.push_back((std::uint64_t)meta.size());
        std::string a = std::to_string(i);
        for (size_t j = 0; j < a.length(); j++)
            meta.push_back(a[j]);
    }
    metaoffset.push_back((std::uint64_t)meta.size());

    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
        SPTAG::ByteArray((std::uint8_t *)vec.data(), sizeof(T) * n * m, false), SPTAG::GetEnumValueType<T>(), m, n));

    std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
        SPTAG::ByteArray((std::uint8_t *)meta.data(), meta.size() * sizeof(char), false),
        SPTAG::ByteArray((std::uint8_t *)metaoffset.data(), metaoffset.size() * sizeof(std::uint64_t), false), n));

    BuildWithMetaMapping<T>(algo, distCalcMethod, vecset, metaset, "testindices");

    SearchWithFilter<T>("testindices", query.data(), q, k);
}

BOOST_AUTO_TEST_SUITE(FilterTest)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    FTest<float>(SPTAG::IndexAlgoType::BKT, "L2");
}

BOOST_AUTO_TEST_CASE(BKTAlwaysTrueResultFilterMatchesNativeSearch)
{
    constexpr SPTAG::SizeType count = 2048;
    constexpr SPTAG::DimensionType dimension = 16;
    constexpr int resultCount = 10;

    std::vector<float> vectors(static_cast<size_t>(count) * dimension);
    std::uint32_t state = 0x12345678U;
    for (float& value : vectors)
    {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>(state & 0xffffU) / 65535.0f;
    }

    auto base = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    auto index = std::dynamic_pointer_cast<SPTAG::BKT::Index<float>>(base);
    BOOST_REQUIRE(index != nullptr);
    BOOST_REQUIRE(index->SetParameter("DistCalcMethod", "L2") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("NumberOfThreads", "1") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("TPTNumber", "1") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("NeighborhoodSize", "16") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("RefineIterations", "0") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("CEF", "128") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("MaxCheckForRefineGraph", "256") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("MaxCheck", "512") == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(index->BuildIndex(vectors.data(), count, dimension) == SPTAG::ErrorCode::Success);

    const float* query = vectors.data() + static_cast<size_t>(123) * dimension;
    SPTAG::COMMON::QueryResultSet<float> nativeResults(query, resultCount);
    SPTAG::COMMON::QueryResultSet<float> filteredResults(query, resultCount);
    BOOST_REQUIRE(index->SearchIndex(nativeResults) == SPTAG::ErrorCode::Success);
    BOOST_REQUIRE(
        index->SearchIndexWithResultFilter(
            filteredResults, [](SPTAG::SizeType) { return true; }) ==
        SPTAG::ErrorCode::Success);

    BOOST_CHECK_EQUAL(filteredResults.GetScanned(), nativeResults.GetScanned());
    for (int i = 0; i < resultCount; ++i)
    {
        BOOST_CHECK_EQUAL(filteredResults.GetResult(i)->VID, nativeResults.GetResult(i)->VID);
        BOOST_CHECK_EQUAL(filteredResults.GetResult(i)->Dist, nativeResults.GetResult(i)->Dist);
    }
}

BOOST_AUTO_TEST_SUITE_END()
