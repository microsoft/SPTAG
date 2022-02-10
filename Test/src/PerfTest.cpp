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
void Search(std::shared_ptr<VectorIndex>& vecIndex, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth)
{
    std::vector<QueryResult> res(queryset->Count(), QueryResult(nullptr, k, true));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        res[i].SetTarget(queryset->GetVector(i));
        vecIndex->SearchIndex(res[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(queryset->Count())) << "us" << std::endl;

    float eps = 1e-6f, recall = 0;
    bool deleted;
    int truthDimension = min(k, truth->Dimension());
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        for (int j = 0; j < truthDimension; j++)
        {
            std::string truthstr = std::to_string(nn[j]);
            ByteArray truthmeta = ByteArray((std::uint8_t*)(truthstr.c_str()), truthstr.length(), false);
            float truthdist = vecIndex->ComputeDistance(queryset->GetVector(i), vecIndex->GetSample(truthmeta, deleted));
            for (int l = 0; l < k; l++)
            {
                if (fabs(truthdist - res[i].GetResult(l)->Dist) <= eps * (fabs(truthdist) + eps)) {
                    recall += 1.0;
                    break;
                }
            }
        }
    }
    LOG(Helper::LogLevel::LL_Info, "Recall %d@%d: %f\n", k, truthDimension, recall / queryset->Count() / truthDimension);
}

template <typename T>
void PerfAdd(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out)
{
    std::shared_ptr<VectorIndex> vecIndex = VectorIndex::CreateInstance(algo, GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    if (algo == IndexAlgoType::KDT) vecIndex->SetParameter("KDTNumber", "2");
    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "5");
    vecIndex->SetParameter("AddCEF", "200");
    vecIndex->SetParameter("CEF", "1500");
    vecIndex->SetParameter("MaxCheck", "4096");
    vecIndex->SetParameter("MaxCheckForRefineGraph", "4096");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < vec->Count(); i++) {
        ByteArray metaarr = meta->GetMetadata(i);
        std::uint64_t offset[2] = { 0, metaarr.Length() };
        std::shared_ptr<MetadataSet> metaset(new MemMetadataSet(metaarr, ByteArray((std::uint8_t*)offset, 2 * sizeof(std::uint64_t), false), 1));
        ErrorCode ret = vecIndex->AddIndex(vec->GetVector(i), 1, vec->Dimension(), metaset, true);
        if (ErrorCode::Success != ret) std::cerr << "Error AddIndex(" << (int)(ret) << ") for vector " << i << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Add time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(vec->Count())) << "us" << std::endl;

    Sleep(10000);

    Search<T>(vecIndex, queryset, k, truth);

    BOOST_CHECK(ErrorCode::Success == vecIndex->SaveIndex(out));
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex(out, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);
    Search<T>(vecIndex, queryset, k, truth);
}

template<typename T>
void PerfBuild(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out)
{
    std::shared_ptr<VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    if (algo == IndexAlgoType::KDT) vecIndex->SetParameter("KDTNumber", "2");
    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "5");
    vecIndex->SetParameter("RefineIterations", "3");
    vecIndex->SetParameter("MaxCheck", "4096");
    vecIndex->SetParameter("MaxCheckForRefineGraph", "8192");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta, true));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
    Search<T>(vecIndex, queryset, k, truth);
}

template <typename T>
void GenerateData(std::shared_ptr<VectorSet>& vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, std::string distCalcMethod, int k)
{

    SizeType n = 2000, q = 2000;
    DimensionType m = 128;
    
    if (fileexists("perftest_vector.bin") && fileexists("perftest_meta.bin") && fileexists("perftest_metaidx.bin") && fileexists("perftest_query.bin")) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<T>(), m, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("perftest_vector.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet();

        metaset.reset(new MemMetadataSet("perftest_meta.bin", "perftest_metaidx.bin", vecset->Count() * 2, vecset->Count() * 2, 10));

        if (ErrorCode::Success != vectorReader->LoadFile("perftest_query.bin"))
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
        vecset->Save("perftest_vector.bin");

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
        metaset.reset(new MemMetadataSet(meta, metaoffset, n, n * 2, n * 2, 10));
        metaset->SaveMetadata("perftest_meta.bin", "perftest_metaidx.bin");

        ByteArray query = ByteArray::Alloc(sizeof(T) * q * m);
        for (SizeType i = 0; i < q; i++) {
            for (DimensionType j = 0; j < m; j++) {
                ((T*)query.Data())[i * m + j] = (T)COMMON::Utils::rand(127, -127);
            }
        }
        queryset.reset(new BasicVectorSet(query, GetEnumValueType<T>(), m, q));
        queryset->Save("perftest_query.bin");
    }
    
    if (fileexists(("perftest_truth." + distCalcMethod).c_str())) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), k, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("perftest_truth." + distCalcMethod))
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
            std::cout << "Normalize vecset!" << std::endl;
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
        truth->Save("perftest_truth." + distCalcMethod);
    }
}

template <typename T>
void PTest(IndexAlgoType algo, std::string distCalcMethod)
{
    std::shared_ptr<VectorSet> vecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    GenerateData<T>(vecset, metaset, queryset, truth, distCalcMethod, 10);
    PerfAdd<T>(algo, distCalcMethod, vecset, metaset, queryset, 10, truth, "testindices");
    PerfBuild<T>(algo, distCalcMethod, vecset, metaset, queryset, 10, truth, "testindices");
    std::shared_ptr<VectorIndex> vecIndex;
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex("testindices", vecIndex));
    BOOST_CHECK(nullptr != vecIndex);
    Search<T>(vecIndex, queryset, 10, truth);
}

BOOST_AUTO_TEST_SUITE(PerfTest)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    PTest<std::int8_t>(IndexAlgoType::BKT, "Cosine");
}

BOOST_AUTO_TEST_CASE(KDTTest)
{
    PTest<std::int8_t>(IndexAlgoType::KDT, "Cosine");
}

BOOST_AUTO_TEST_SUITE_END()