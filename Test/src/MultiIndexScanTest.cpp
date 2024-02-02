// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/MultiIndexScan.h"
#include "inc/Core/Common/CommonUtils.h"

#include <chrono>

#include <numeric>

static std::string indexName(unsigned int id){
    return "testindices" + std::to_string(id);
}
template <typename T>
void BuildIndex(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "16");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

float rankFunc(std::vector<float> in){
    return std::accumulate(in.begin(), in.end(), 0.0f);
}

template <typename T>
void MultiIndexSearch(unsigned int n, std::vector<std::vector<T>> &queries, int k, bool userTimer, int termCond)
{

    std::vector<std::shared_ptr<SPTAG::VectorIndex>> vecIndices;
    std::vector<void*> p_targets;
    for ( unsigned int i = 0; i < n; i++ ) {
        std::shared_ptr<SPTAG::VectorIndex> vecIndex;
        BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(indexName(i).c_str(), vecIndex));
        BOOST_CHECK(nullptr != vecIndex);
        vecIndices.push_back(vecIndex);
        p_targets.push_back(queries[i].data());
    }

    SPTAG::MultiIndexScan scan(vecIndices, p_targets, k, &rankFunc, false, 10, 10);
    SPTAG::BasicResult result;

    std::cout << "Start Scanning!!! " << std::endl;

    for (int i = 0; i < 100; i++) {
        bool hasResult = scan.Next(result);
        if (!hasResult) break;
        std::cout << "hasResult: " << hasResult << std::endl;
        std::cout << "result: " << result.VID << std::endl;  
    }
    scan.Close();
}

template <typename T>
void GenerateVectorDataSet(SPTAG::IndexAlgoType algo, std::string distCalcMethod, unsigned int id, std::vector<T> &query){
    SPTAG::SizeType n = 2000, q = 3;
    SPTAG::DimensionType m = 10;

    std::vector<T> vec;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        for (SPTAG::DimensionType j = 0; j < m; j++) {
            vec.push_back((T)i+id*n);
        }
    }

    for (SPTAG::SizeType i = 0; i < q; i++) {
        for (SPTAG::DimensionType j = 0; j < m; j++) {
            query.push_back((T)i * 2 +id*n);
        }
    }

    std::vector<char> meta;
    std::vector<std::uint64_t> metaoffset;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        metaoffset.push_back((std::uint64_t)meta.size());
        std::string a = std::to_string(i+id*n);
        for (size_t j = 0; j < a.length(); j++)
            meta.push_back(a[j]);
    }
    metaoffset.push_back((std::uint64_t)meta.size());

    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
        SPTAG::ByteArray((std::uint8_t*)vec.data(), sizeof(T) * n * m, false),
        SPTAG::GetEnumValueType<T>(), m, n));

    std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
        SPTAG::ByteArray((std::uint8_t*)meta.data(), meta.size() * sizeof(char), false),
        SPTAG::ByteArray((std::uint8_t*)metaoffset.data(), metaoffset.size() * sizeof(std::uint64_t), false),
        n));

    BuildIndex<T>(algo, distCalcMethod, vecset, metaset, indexName(id).c_str());
}

template <typename T>
void TestMultiIndexScanN(SPTAG::IndexAlgoType algo, std::string distCalcMethod, unsigned int n)
{
    int k = 3;
    std::vector<std::vector<T>> queries(n, std::vector<T>());

    for (unsigned int i = 0; i < n; i++ ){
        GenerateVectorDataSet<T>(algo, distCalcMethod, i, queries[i]);
    }

    MultiIndexSearch<T>(n, queries, k, false, 10);
    
}

BOOST_AUTO_TEST_SUITE(MultiIndexScanTest)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    TestMultiIndexScanN<float>(SPTAG::IndexAlgoType::BKT, "L2", 2);
}

BOOST_AUTO_TEST_SUITE_END()
