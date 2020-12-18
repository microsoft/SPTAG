// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/DistanceUtils.h"
#include <thread>
#include <unordered_set>
#include <ctime>

using namespace SPTAG;

template <typename T>
void Add(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, const std::string out)
{
    std::shared_ptr<VectorIndex> vecIndex = VectorIndex::CreateInstance(algo, GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "5");
    vecIndex->SetParameter("AddCEF", "200");
    vecIndex->SetParameter("CEF", "1500");
    vecIndex->SetParameter("MaxCheck", "4096");
    vecIndex->SetParameter("MaxCheckForRefineGraph", "3200");
    vecIndex->SetParameter("DataBlockSize", "200000");
    vecIndex->SetParameter("DataCapacity", "200000");

    omp_set_num_threads(5);

    auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (SizeType i = 0; i < vec->Count(); i++) {
        ByteArray metaarr = meta->GetMetadata(i);
        ErrorCode ret = vecIndex->AddOne(vec->GetVector(i), vec->Dimension(), metaarr, true);
        if (ErrorCode::Success != ret) std::cerr << "Error AddIndex(" << (int)(ret) << ") for vector " << i << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Add time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(vec->Count())) << "us" << std::endl;

    Sleep(10000);

    BOOST_CHECK(ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T>
void Build(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "5");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T>
void Search(const std::string folder, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth)
{
    std::shared_ptr<VectorIndex> vecIndex;
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex(folder, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);

    std::vector<QueryResult> res(queryset->Count(), QueryResult(nullptr, k, true));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        res[i].SetTarget(queryset->GetVector(i));
        vecIndex->SearchIndex(res[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(queryset->Count())) << "us" << std::endl;

    float recall = 0;
    bool deleted = false;
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        for (int j = 0; j < truth->Dimension(); j++) {
            for (int l = 0; l < k; l++) {
                std::string truthstr = std::to_string(nn[j]);
                ByteArray truthmeta = ByteArray((std::uint8_t*)(truthstr.c_str()), truthstr.length(), false);
                float truthdist = vecIndex->ComputeDistance(queryset->GetVector(i), vecIndex->GetSample(truthmeta, deleted));
                if (fabs(truthdist - res[i].GetResult(l)->Dist) < 1e-6 * truthdist) {
                    recall += 1.0;
                    break;
                }
            }
        }
    }
    LOG(Helper::LogLevel::LL_Info, "Recall %d@%d: %f\n", k, truth->Dimension(), recall / queryset->Count() / truth->Dimension());
}

template <typename T>
void GenerateData(std::shared_ptr<VectorSet>& vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, std::string distCalcMethod, int k)
{
    SizeType n = 200000, q = 1000;
    DimensionType m = 64;
    if (fileexists("test_vector.bin") && fileexists("test_meta.bin") && fileexists("test_metaidx.bin") && fileexists("test_query.bin")) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<T>(), m, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("test_vector.bin,test_meta.bin,test_metaidx.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet();
        metaset = vectorReader->GetMetadataSet();

        if (ErrorCode::Success != vectorReader->LoadFile("test_query.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
            exit(1);
        }
        queryset = vectorReader->GetVectorSet();
    }
    else {
        ByteArray vec = ByteArray::Alloc(sizeof(T) * n * m);
        for (SizeType i = 0; i < n; i++) {
            for (DimensionType j = 0; j < m; j++) {
                ((T*)vec.Data())[i * m + j] = (T)COMMON::Utils::rand(127, -127);
            }
        }
        vecset.reset(new BasicVectorSet(vec, GetEnumValueType<T>(), m, n));
        vecset->Save("test_vector.bin");

        ByteArray meta = ByteArray::Alloc(n * 6);
        ByteArray metaoffset = ByteArray::Alloc((n + 1) * sizeof(std::uint64_t));
        std::uint64_t offset = 0;
        for (SizeType i = 0; i < n; i++) {
            ((std::uint64_t*)metaoffset.Data())[i] = offset;
            std::string a = std::to_string(i);
            memcpy(meta.Data() + offset, a.c_str(), a.length());
            offset += a.length();
        }
        ((std::uint64_t*)metaoffset.Data())[n] = offset;
        metaset.reset(new MemMetadataSet(meta, metaoffset, n, 1024 * 1024, 1024 * 1024, 10));
        metaset->SaveMetadata("test_meta.bin", "test_metaidx.bin");

        ByteArray query = ByteArray::Alloc(sizeof(T) * q * m);
        for (SizeType i = 0; i < q; i++) {
            for (DimensionType j = 0; j < m; j++) {
                ((T*)query.Data())[i * m + j] = (T)COMMON::Utils::rand(127, -127);
            }
        }
        queryset.reset(new BasicVectorSet(query, GetEnumValueType<T>(), m, q));
        queryset->Save("test_query.bin");
    }

    if (fileexists(("test_truth." + distCalcMethod).c_str())) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), k, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("test_truth." + distCalcMethod))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read truth file.\n");
            exit(1);
        }
        truth = vectorReader->GetVectorSet();
    }
    else {
        omp_set_num_threads(5);
        
        DistCalcMethod distMethod;
        Helper::Convert::ConvertStringTo(distCalcMethod.c_str(), distMethod);
        if (distMethod == DistCalcMethod::Cosine) {
            COMMON::Utils::BatchNormalize((T*)(vecset->GetData()), vecset->Count(), vecset->Dimension(), COMMON::Utils::GetBase<T>(), 5);
        }
       
        ByteArray tru = ByteArray::Alloc(sizeof(float) * queryset->Count() * k);

#pragma omp parallel for
        for (SizeType i = 0; i < queryset->Count(); ++i)
        {
            SizeType* neighbors = ((SizeType*)tru.Data()) + i * k;
            
            COMMON::QueryResultSet<T> res((const T*)queryset->GetVector(i), k);
            for (SizeType j = 0; j < vecset->Count(); j++)
            {
                float dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), reinterpret_cast<T*>(vecset->GetVector(j)), queryset->Dimension(), distMethod);
                res.AddPoint(j, dist);
            }
            res.SortResult();
            for (int j = 0; j < k; j++) neighbors[j] = res.GetResult(j)->VID;
        }
        truth.reset(new BasicVectorSet(tru, GetEnumValueType<float>(), k, queryset->Count()));
        truth->Save("test_truth." + distCalcMethod);
    }
}

template <typename T>
void PTest(IndexAlgoType algo, std::string distCalcMethod)
{
    std::shared_ptr<VectorSet> vecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    GenerateData<T>(vecset, metaset, queryset, truth, distCalcMethod, 5);

    Add<T>(algo, distCalcMethod, vecset, metaset, "testindices");
    Search<T>("testindices", queryset, 5, truth);
}

BOOST_AUTO_TEST_SUITE(PerfTest)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    PTest<std::int8_t>(IndexAlgoType::BKT, "L2");
}

BOOST_AUTO_TEST_SUITE_END()
