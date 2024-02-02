// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/ResultIterator.h"
#include "inc/Core/Common/CommonUtils.h"

#include <unordered_set>
#include <chrono>

template <typename T>
void BuildIndex(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "16");
    vecIndex->SetParameter("MaxCheck", "5");
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T>
void SearchIterativeBatch(const std::string folder, T* vec, SPTAG::SizeType n, std::string* truthmeta)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;

    BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(folder, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);

    std::shared_ptr<ResultIterator> resultIterator = vecIndex->GetIterator(vec);
    //std::cout << "relaxedMono:" << resultIterator->GetRelaxedMono() << std::endl;
    int batch = 5;
    int ri = 0;
    for (int i = 0; i < 2; i++) {
        auto results = resultIterator->Next(batch);
        int resultCount = results->GetResultNum();
        if (resultCount <= 0) break;
        BOOST_CHECK(resultCount == batch);
        for (int j = 0; j < resultCount; j++) {
            
            BOOST_CHECK(std::string((char*)((results->GetMetadata(j)).Data()), (results->GetMetadata(j)).Length()) == truthmeta[ri]);
            BOOST_CHECK(results->GetResult(j)->RelaxedMono == true);
            std::cout << "Result[" << ri << "] VID:" << results->GetResult(j)->VID << " Dist:" << results->GetResult(j)->Dist << " RelaxedMono:"
                << results->GetResult(j)->RelaxedMono << std::endl;
            ri++;
        }
    }
    resultIterator->Close();
}

template <typename T>
void TestIterativeScan(SPTAG::IndexAlgoType algo, std::string distCalcMethod)
{
    SPTAG::SizeType n = 2000, q = 1;
    SPTAG::DimensionType m = 10;
    std::vector<T> vec;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        for (SPTAG::DimensionType j = 0; j < m; j++) {
            vec.push_back((T)i);
        }
    }

    std::vector<T> query;
    for (SPTAG::SizeType i = 0; i < q; i++) {
        for (SPTAG::DimensionType j = 0; j < m; j++) {
            query.push_back((T)i * 2);
        }
    }

    std::vector<char> meta;
    std::vector<std::uint64_t> metaoffset;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        metaoffset.push_back((std::uint64_t)meta.size());
        std::string a = std::to_string(i);
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

    BuildIndex<T>(algo, distCalcMethod, vecset, metaset, "testindices");
    std::string truthmeta1[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
    SearchIterativeBatch<T>("testindices", query.data(), q, truthmeta1);
}

BOOST_AUTO_TEST_SUITE(IterativeScanTest)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    TestIterativeScan<float>(SPTAG::IndexAlgoType::BKT, "L2");
}

BOOST_AUTO_TEST_SUITE_END()
