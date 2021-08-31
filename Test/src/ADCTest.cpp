// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include <random>
#include "inc/Helper/VectorSetReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/PQQuantizer.h"
#include <thread>
#include <iostream>
#include <unordered_set>
#include <ctime>

using namespace SPTAG;

template <typename T>
void ADCSearch(std::shared_ptr<VectorIndex>& vecIndex, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, int size_orignal)
{
    std::vector<QueryResult> res(queryset->Count() - size_orignal, QueryResult(nullptr, k, true));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count() - size_orignal; i++)
    {
        res[i].SetTarget(queryset->GetVector(i + size_orignal));
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
    for (SizeType i = 0; i < queryset->Count() - size_orignal; i++) {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        /*
        std::cout << "Query " << i + 1 << std::endl;
        for (int l = 0; l < k; l++) {
            std::cout << res[i].GetResult(l)->VID << " ";
        }
        std::cout << std::endl;
        for (int l = 0; l < truthDimension; l++) {
            std::cout << nn[l] << " ";
        }
        std::cout << std::endl;
        */
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
void ADCAdd(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out, int size_orignal)
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

    ADCSearch<T>(vecIndex, queryset, k, truth, size_orignal);

    BOOST_CHECK(ErrorCode::Success == vecIndex->SaveIndex(out));
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex(out, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);
    ADCSearch<T>(vecIndex, queryset, k, truth, size_orignal);
}

template<typename T>
void ADCBuild(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out, int size_orignal)
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
    ADCSearch<T>(vecIndex, queryset, k, truth, size_orignal);
}


template <typename T>
void GeneratePQData_ADC(std::shared_ptr<VectorSet>& vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, std::string distCalcMethod, int k, int & size_original)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.01, 100.0);

    const int n = 1500, q = 100;
    const int m = 768;
    const int M = 8;
    const int Ks = 256;


    int QuanDim = m / M;
    ByteArray PQvec = ByteArray::Alloc(sizeof(T) * n * QuanDim);
    ByteArray real_vec = ByteArray::Alloc(sizeof(float) * n * m);
    ByteArray reconstruct_vec = ByteArray::Alloc(sizeof(float) * n * m);
    //ByteArray PQquery = ByteArray::Alloc(sizeof(T) * q * (m / M));
    ByteArray PQquery = ByteArray::Alloc(sizeof(float) * (2 * q * m + q * M * Ks));
    ByteArray real_query = ByteArray::Alloc(sizeof(float) * q * m);
    std::vector< std::vector<std::uint8_t> > baseQ(n, std::vector<std::uint8_t>(QuanDim));
    float* querys = new float[q * m];
    float* vecs = new float[n * m];

    std::shared_ptr<VectorSet> real_vecset, real_queryset;
    if (fileexists("vectors.bin") && fileexists("querys.bin")) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), m, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("vectors.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        real_vecset = vectorReader->GetVectorSet();

        std::cout << real_vecset->Dimension() << " " << real_vecset->Count() << std::endl;
        if (ErrorCode::Success != vectorReader->LoadFile("querys.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
            exit(1);
        }
        real_queryset = vectorReader->GetVectorSet();
    }
    else {

        for (int i = 0; i < n * m; i++) {
            vecs[i] = COMMON::Utils::rand(255, 0);
            ((float*)real_vec.Data())[i] = (float)vecs[i];
        }
        real_vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<T>(), m, n));
        for (int i = 0; i < q * m; i++) {
            querys[i] = COMMON::Utils::rand(255, 0);
            ((T*)real_query.Data())[i] = (T)querys[i];
        }
        real_queryset.reset(new BasicVectorSet(real_query, GetEnumValueType<T>(), m, q));
    }
   
    std::string CODEBOOK_FILE = "test-quantizer-adc.bin";
    float* codebooks = new float[M * Ks * QuanDim];
    if (fileexists("codebooks.bin")) {
    //if (0) {
        std::shared_ptr<VectorSet> temp_codebooks;
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), M * Ks, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("codebooks.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read coodebooks file.\n");
            exit(1);
        }
        temp_codebooks = vectorReader->GetVectorSet();
        std::cout << temp_codebooks->Dimension() << " " << temp_codebooks->Count() << std::endl;
        float* temp = new float[Ks * M];
        for (int i = 0; i < QuanDim; i++) {
            temp = (float*)temp_codebooks->GetVector(i);
            for (int j = 0; j < Ks; j++) {
                for (int l = 0; l < M; l++) {
                    codebooks[l * Ks * QuanDim + j * QuanDim + i] = temp[j * M + l];
                }
            }
            //std::cout << codebooks[i] << std::endl;
        }
    }
    else {
        std::cout << "Building codebooks!" << std::endl;
        float* kmeans = new float[Ks * QuanDim];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < Ks; j++) {
                for (int t = 0; t < QuanDim; t++) {
                    kmeans[j * QuanDim + t] = vecs[j * m + i * QuanDim + t];
                }
            }
            int cnt = 5;
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
    }
    std::cout << "Codebooks Building Finish!" << std::endl;
    auto baseQuantizer = std::make_shared<SPTAG::COMMON::PQQuantizer>(M, Ks, (m / M), true, codebooks);
    //std::cout << baseQuantizer->GetEnableADC() << std::endl;
    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr != nullptr && ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::out));
    baseQuantizer->SaveQuantizer(ptr);
    ptr->ShutDown();

    BOOST_ASSERT(ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::in));
    SPTAG::COMMON::Quantizer::LoadQuantizer(ptr, SPTAG::QuantizerType::PQQuantizer);
    BOOST_ASSERT(SPTAG::COMMON::DistanceUtils::Quantizer != nullptr);
    //std::cout << SPTAG::COMMON::DistanceUtils::Quantizer->GetEnableADC() << std::endl;

    std::cout << "Quantize Vector" << std::endl;
    for (int i = 0; i < n; i++) {
        auto nvec = real_vecset->GetVector(i);
        //std::cout << real_vecset->Dimension() << " " << real_vecset->Count() << std::endl;
        baseQuantizer->QuantizeVector((const float*)nvec, baseQ[i].data());
        //std::cout << i << " " << baseQ[i].size() << std::endl;
    }
    for (SizeType i = 0; i < n; i++) {
        for (DimensionType j = 0; j < (m / M); j++) {
            ((T*)PQvec.Data())[i * (m / M) + j] = (T)baseQ[i][j];
        }
    }
    vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<T>(), (m / M), n));
    vecset->Save("ADCtest_vector.bin");

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
    std::cout << "Quantize Vector" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    querys = (float*)real_queryset->GetData();
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < m; j++) {
            ((float*)PQquery.Data())[i * m + j] = (float)querys[i * m + j];
        }
    }
    size_original = (q * m + M * Ks - 1) / (M * Ks);
    for (int i = 0; i < q; i++) {
        for (DimensionType j = 0; j < M; j++) {
            ((float*)PQquery.Data())[size_original * M * Ks + i * (m / M) + j] = 0;
            for (DimensionType ii = 0; ii < Ks; ii++) {
                for (int l = 0; l < (m / M); l++) {
                    ((float*)PQquery.Data())[size_original * M * Ks + q * m + i * (m / M) + j] += (querys[i * m + j * (m / M) + l] - codebooks[j * Ks * (m / M) + ii * (m / M) + l]) * (querys[i * m + j * (m / M) + l] - codebooks[j * Ks * (m / M) + ii * (m / M) + l]);
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Query Quantization time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(q)) << "us" << std::endl;

    //real_queryset.reset(new BasicVectorSet(querys, GetEnumValueType<float>(), m, q));
    queryset.reset(new BasicVectorSet(PQquery, GetEnumValueType<float>(), M * Ks, q + size_original));
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
    /*
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
    */
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
    //}
}

template <typename T>
void ADCPQTest(IndexAlgoType algo, std::string distCalcMethod)
{
    int size_o;
    std::shared_ptr<VectorSet> vecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    GeneratePQData_ADC<T>(vecset, metaset, queryset, truth, distCalcMethod, 10, size_o);

    ADCAdd<T>(algo, distCalcMethod, vecset, metaset, queryset, 10, truth, "testindices", size_o);
    std::cout << "PerfAdd Finish!" << std::endl;

    ADCBuild<T>(algo, distCalcMethod, vecset, metaset, queryset, 10, truth, "testindices", size_o);
    std::cout << "PerfBuild Finish!" << std::endl;
    std::shared_ptr<VectorIndex> vecIndex;
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex("testindices", vecIndex));
    BOOST_CHECK(nullptr != vecIndex);
    ADCSearch<T>(vecIndex, queryset, 10, truth, size_o);
}


BOOST_AUTO_TEST_SUITE(ADCTest)

BOOST_AUTO_TEST_CASE(ADCBKTTest)
{
    ADCPQTest<std::uint8_t>(IndexAlgoType::BKT, "L2");

    SPTAG::COMMON::DistanceUtils::Quantizer = nullptr;
}

BOOST_AUTO_TEST_SUITE_END()
