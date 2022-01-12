// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"

#include <thread>
#include <unordered_set>
#include <ctime>

template <typename T>
void ConcurrentAddSearchSave(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "16");
    vecIndex->SetParameter("DataBlockSize", "1024");
    vecIndex->SetParameter("DataCapacity", "1048576");

    bool stop = false;

    auto AddThread = [&stop, &vecIndex, &vec, &meta]() {
        int i = 0;
        while (!stop)
        {
            SPTAG::ByteArray metaarr = meta->GetMetadata(i);
            std::uint64_t offset[2] = { 0, metaarr.Length() };
            std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(metaarr, SPTAG::ByteArray((std::uint8_t*)offset, 2 * sizeof(std::uint64_t), false), 1));
            SPTAG::ErrorCode ret = vecIndex->AddIndex(vec->GetVector(i), 1, vec->Dimension(), metaset, true);
            if (SPTAG::ErrorCode::Success != ret) std::cerr << "Error AddIndex(" << (int)(ret) << ") for vector " << i << std::endl;
            i = (i + 1) % vec->Count();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "Stop AddThread..." << std::endl;
    };

    auto DeleteThread = [&stop, &vecIndex, &vec, &meta]() {
        int i = 0;
        while (!stop)
        {
            SPTAG::ByteArray metaarr = meta->GetMetadata(i);
            vecIndex->DeleteIndex(metaarr);
            i = (i + 1) % vec->Count();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "Stop DeleteThread..." << std::endl;
    };

    auto SearchThread = [&stop, &vecIndex, &vec]() {
        while (!stop) {
            SPTAG::QueryResult res(vec->GetVector(SPTAG::COMMON::Utils::rand(vec->Count())), 5, true);
            vecIndex->SearchIndex(res);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "Stop SearchThread..." << std::endl;
    };

    auto SaveThread = [&stop, &vecIndex, &out]() {
        while (!stop)
        {
            vecIndex->SaveIndex(out);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::cout << "Stop SaveThread..." << std::endl;
    };

    std::vector<std::thread> threads;
    threads.emplace_back(std::thread(AddThread));
    threads.emplace_back(std::thread(DeleteThread));
    threads.emplace_back(std::thread(SearchThread));
    threads.emplace_back(std::thread(SaveThread));

    std::this_thread::sleep_for(std::chrono::seconds(30));

    stop = true;
    for (auto& thread : threads) { thread.join(); }
    std::cout << "Main Thread quit!" << std::endl;
}

template <typename T>
void CTest(SPTAG::IndexAlgoType algo, std::string distCalcMethod)
{
    SPTAG::SizeType n = 2000, q = 3;
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

    ConcurrentAddSearchSave<T>(algo, distCalcMethod, vecset, metaset, "testindices");
}

BOOST_AUTO_TEST_SUITE(ConcurrentTest)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    CTest<float>(SPTAG::IndexAlgoType::BKT, "L2");
}

BOOST_AUTO_TEST_CASE(KDTTest)
{
    CTest<float>(SPTAG::IndexAlgoType::KDT, "L2");
}

BOOST_AUTO_TEST_SUITE_END()
