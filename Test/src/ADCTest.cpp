// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include <random>
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/Common/PQQuantizer.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/DistanceUtils.h"
#include <thread>
#include <iostream>
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
    /*
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
    */
    truthDimension = k * 2;
    for (SizeType i = 0; i < queryset->Count(); i++) {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        std::cout << "Query " << i + 1 << std::endl;
        for (int l = 0; l < k; l++) {
            std::cout << res[i].GetResult(l)->VID << " ";
        }
        std::cout << std::endl;
        for (int l = 0; l < truthDimension; l++) {
            std::cout << nn[l] << " ";
        }
        std::cout << std::endl;
        for (int l = 0; l < k; l++)
        {
            for (int j = 0; j < truthDimension; j++)
            {
                if (res[i].GetResult(l)->VID == nn[j]) {
                    recall += 1.0;
                    break;
                }
            }
        }
    }
    LOG(Helper::LogLevel::LL_Info, "Recall %d@%d: %f\n", k, truthDimension, recall / queryset->Count() / truthDimension * 2);
}

template <typename T>
void ADCAdd(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out)
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
void ADCBuild(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out)
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
void GeneratePQData_ADC(std::shared_ptr<VectorSet>& vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, std::string distCalcMethod, int k)
{
    std::shared_ptr<VectorSet> real_vecset, real_queryset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.01, 100.0);
    int n = 500, q = 10;
    int m = 128;
    int M = 4;
    int Ks = 256;


    /*
    if (fileexists("test_vector.bin") && fileexists("test_meta.bin") && fileexists("test_metaidx.bin") && fileexists("test_query.bin")) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<T>(), m, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("test_vector.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet();

        metaset.reset(new MemMetadataSet("test_meta.bin", "test_metaidx.bin", vecset->Count() * 2, vecset->Count() * 2, 10));

        if (ErrorCode::Success != vectorReader->LoadFile("test_query.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
            exit(1);
        }
        queryset = vectorReader->GetVectorSet();
    }
    else {
       */
    int QuanDim = m / M;
    ByteArray PQvec = ByteArray::Alloc(sizeof(T) * n * QuanDim);
    ByteArray real_vec = ByteArray::Alloc(sizeof(T) * n * m);
    std::vector< std::vector<std::uint8_t> > baseQ(n, std::vector<std::uint8_t>(QuanDim));

    float* vecs = new float[n * m];
    for (int i = 0; i < n * m; i++) {
        vecs[i] = COMMON::Utils::rand(255, 0);
        ((T*)real_vec.Data())[i] = (T)vecs[i];
    }

    real_vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<T>(), m, n));
    std::cout << "Building codebooks!" << std::endl;
    std::string CODEBOOK_FILE = "test-quantizer-tree.bin";
    float* codebooks = new float[M * Ks * QuanDim];

    float* kmeans = new float[Ks * QuanDim];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < Ks; j++) {
            for (int t = 0; t < QuanDim; t++) {
                kmeans[j * QuanDim + t] = vecs[j * m + i * QuanDim + t];
            }
        }
        int cnt = 100;
        int* belong = new int[n];
        while (cnt--) {
            //calculate cluster
            for (int ii = 0; ii < n; ii++) {
                double min_dis = 1e9;
                int min_id = 0;
                for (int jj = 0; jj < Ks; jj++) {
                    double now_dis = 0;
                    for (int kk = 0; kk < QuanDim; kk++) {
                        now_dis += (1.0 * vecs[ii * m + i * QuanDim + kk] - kmeans[jj * QuanDim + kk]) * (1.0 * vecs[ii * m + i * QuanDim + kk] - kmeans[jj * QuanDim + kk]);
                    }
                    if (now_dis < min_dis) {
                        min_dis = now_dis;
                        min_id = jj;
                    }
                }
                belong[ii] = min_id;
            }
            //recalculate kmeans
            for (int ii = 0; ii < Ks; ii++) {
                int num = 0;
                float* newmeans = new float[QuanDim]();
                for (int jj = 0; jj < n; jj++) {
                    if (belong[jj] == ii) {
                        num++;
                        for (int kk = 0; kk < QuanDim; kk++) {
                            newmeans[kk] += vecs[jj * m + i * QuanDim + kk];
                        }
                    }
                }
                for (int jj = 0; jj < QuanDim; jj++) {
                    newmeans[jj] /= num;
                    kmeans[ii * QuanDim + jj] = newmeans[jj];
                }
                delete[] newmeans;
            }
        }
        //use kmeans to calculate codebook
        for (int j = 0; j < Ks; j++) {
            double min_dis = 1e9;
            int min_id = 0;
            for (int ii = 0; ii < n; ii++) {
                double now_dis = 0;
                for (int t = 0; t < QuanDim; t++) {
                    now_dis += (vecs[ii * m + i * QuanDim + t] - kmeans[j * QuanDim + t]) * (vecs[ii * m + i * QuanDim + t] - kmeans[j * QuanDim + t]);
                }
                if (now_dis < min_dis) {
                    min_dis = now_dis;
                    min_id = ii;
                }
            }
            for (int t = 0; t < QuanDim; t++) {
                codebooks[i * Ks * QuanDim + j * QuanDim + t] = vecs[min_id * m + i * QuanDim + t];
            }
        }
        delete[] belong;
    }
    delete[] kmeans;
    std::cout << "Building Finish!" << std::endl;
    auto baseQuantizer = std::make_shared<SPTAG::COMMON::PQQuantizer>(M, Ks, (m / M), false, codebooks);
    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr != nullptr && ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::out));
    baseQuantizer->SaveQuantizer(ptr);
    ptr->ShutDown();

    BOOST_ASSERT(ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::in));
    SPTAG::COMMON::Quantizer::LoadQuantizer(ptr, SPTAG::QuantizerType::PQQuantizer);
    BOOST_ASSERT(SPTAG::COMMON::DistanceUtils::Quantizer != nullptr);


    for (int i = 0; i < n; i++) {
        auto nvec = &vecs[i * m];
        baseQuantizer->QuantizeVector(nvec, baseQ[i].data());
    }
    for (SizeType i = 0; i < n; i++) {
        for (DimensionType j = 0; j < (m / M); j++) {
            ((T*)PQvec.Data())[i * (m / M) + j] = (T)baseQ[i][j];
        }
    }
    vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<T>(), (m / M), n));
    vecset->Save("test_vector_adc.bin");

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
    metaset->SaveMetadata("test_meta_adc.bin", "test_metaidx_adc.bin");
    /*
    ByteArray PQquery = ByteArray::Alloc(sizeof(T) * q * (m / M));
    ByteArray real_query = ByteArray::Alloc(sizeof(T) * q * m);
    float* querys = new float[q * m];
    std::vector< std::vector<std::uint8_t> > baseQ1(n, std::vector<std::uint8_t>(m / M));
    for (int i = 0; i < q * m; i++) {
        querys[i] = COMMON::Utils::rand(255, 0);
        ((T*)real_query.Data())[i] = (T)querys[i];
    }
    real_queryset.reset(new BasicVectorSet(real_query, GetEnumValueType<T>(), m, q));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < q; i++) {
        auto nquery = &querys[i * m];
        baseQuantizer->QuantizeVector(nquery, baseQ1[i].data());
    }
    for (SizeType i = 0; i < q; i++) {
        for (DimensionType j = 0; j < (m / M); j++) {
            ((T*)PQquery.Data())[i * (m / M) + j] = (T)baseQ1[i][j];
        }
    }
    */
    ByteArray PQquery = ByteArray::Alloc(sizeof(float) * q * M * Ks + sizeof(float) * q * m);
    ByteArray real_query = ByteArray::Alloc(sizeof(T) * q * m);
    float* querys = new float[q * m];
    std::vector< std::vector<std::uint8_t> > baseQ1(n, std::vector<std::uint8_t>(m / M));
    for (int i = 0; i < q * m; i++) {
        querys[i] = COMMON::Utils::rand(255, 0); 
        ((T*)real_query.Data())[i] = (T)querys[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < m; j++) {
            ((float*)PQquery.Data())[i * m + j] = querys[i];
        }
    }
    for (int i = 0; i < q; i++) {
        auto nquery = &querys[i * m];
        for (DimensionType j = 0; j < M; j++) {
            ((float*)PQquery.Data())[i * (m / M) + j] = 0;
            for (DimensionType ii = 0; ii < 256; ii++) {
                for (int l = 0; l < (m / M); l++) {
                    ((float*)PQquery.Data())[q * m + i * (m / M) + j] += (querys[i * m + j * (m / M) + l] - codebooks[j * Ks * (m / M) + ii * (m / M) + l]) * (querys[i * m + j * (m / M) + l] - codebooks[j * Ks * (m / M) + ii * (m / M) + l]);
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Query Quantization time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(q)) << "us" << std::endl;

    queryset.reset(new BasicVectorSet(PQquery, GetEnumValueType<T>(), (m / M), q));
    queryset->Save("test_query_adc.bin");
    //}
/*
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
    else { */
    omp_set_num_threads(5);

    DistCalcMethod distMethod;
    Helper::Convert::ConvertStringTo(distCalcMethod.c_str(), distMethod);
    if (distMethod == DistCalcMethod::Cosine) {
        std::cout << "Normalize vecset!" << std::endl;
        COMMON::Utils::BatchNormalize((T*)(vecset->GetData()), vecset->Count(), vecset->Dimension(), COMMON::Utils::GetBase<T>(), 5);
    }

    ByteArray tru = ByteArray::Alloc(sizeof(float) * queryset->Count() * 2 * k);

#pragma omp parallel for
    for (SizeType i = 0; i < real_queryset->Count(); ++i)
    {
        SizeType* neighbors = ((SizeType*)tru.Data()) + i * 2 * k;

        COMMON::QueryResultSet<T> res((const T*)real_queryset->GetVector(i), 2 * k);
        for (SizeType j = 0; j < real_vecset->Count(); j++)
        {
            float dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), reinterpret_cast<T*>(real_vecset->GetVector(j)), real_queryset->Dimension(), distMethod);
            res.AddPoint(j, dist);
        }
        res.SortResult();
        for (int j = 0; j < 2 * k; j++) neighbors[j] = res.GetResult(j)->VID;
    }
    truth.reset(new BasicVectorSet(tru, GetEnumValueType<float>(), 2 * k, real_queryset->Count()));
    truth->Save("test_truth_adc." + distCalcMethod);
    //}
}

template <typename T>
void ADCPQTest(IndexAlgoType algo, std::string distCalcMethod)
{
    std::shared_ptr<VectorSet> vecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    GeneratePQData_ADC<T>(vecset, metaset, queryset, truth, distCalcMethod, 10);

    ADCAdd<T>(algo, distCalcMethod, vecset, metaset, queryset, 10, truth, "testindices");
    std::cout << "PerfAdd Finish!" << std::endl;

    ADCBuild<T>(algo, distCalcMethod, vecset, metaset, queryset, 10, truth, "testindices");
    std::cout << "PerfBuild Finish!" << std::endl;
    std::shared_ptr<VectorIndex> vecIndex;
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex("testindices", vecIndex));
    BOOST_CHECK(nullptr != vecIndex);
    Search<T>(vecIndex, queryset, 10, truth);
}


BOOST_AUTO_TEST_SUITE(ADCTest)

BOOST_AUTO_TEST_CASE(ADCBKTTest)
{
    ADCPQTest<std::uint8_t>(IndexAlgoType::BKT, "L2");

    SPTAG::COMMON::DistanceUtils::Quantizer = nullptr;
}

BOOST_AUTO_TEST_SUITE_END()
