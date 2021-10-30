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
    std::vector<SPTAG::COMMON::QueryResultSet<T>> res(queryset->Count(), SPTAG::COMMON::QueryResultSet<T>(nullptr, k * 2));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        res[i].Reset();
        res[i].SetTarget((const T*)queryset->GetVector(i));
        vecIndex->SearchIndex(res[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(queryset->Count())) << "us" << std::endl;

    float eps = 1e-6f, recall = 0;
    bool deleted;

    int truthDimension = min(k, truth->Dimension());
    for (SizeType i = 0; i < queryset->Count(); i++) {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        std::vector<bool> visited(k, false);
        for (int j = 0; j < truthDimension; j++) {
            float truthdist = vecIndex->ComputeDistance(res[i].GetQuantizedTarget(), vecIndex->GetSample(nn[j]));
            for (int l = 0; l < k*2; l++) {
                if (visited[l]) continue;

                //std::cout << res[i].GetResult(l)->Dist << " " << truthdist << std::endl;
                if (res[i].GetResult(l)->VID == nn[j]) {
                    recall += 1.0;
                    visited[l] = true;
                    break;
                }
                else if (fabs(res[i].GetResult(l)->Dist - truthdist) <= eps) {
                    recall += 1.0;
                    visited[l] = true;
                    break;
                }
            }
        }
    }

    LOG(Helper::LogLevel::LL_Info, "Recall %d@%d: %f\n", truthDimension, k*2, recall / queryset->Count() / truthDimension);
}



template<typename T>
std::shared_ptr<VectorIndex> PerfBuild(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out)
{
    std::shared_ptr<VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    if (algo == IndexAlgoType::KDT) vecIndex->SetParameter("KDTNumber", "2");
    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "12");
    vecIndex->SetParameter("RefineIterations", "3");
    vecIndex->SetParameter("MaxCheck", "4096");
    vecIndex->SetParameter("MaxCheckForRefineGraph", "8192");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta, true));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
    //Search<T>(vecIndex, queryset, k, truth);
    return vecIndex;
}


template <typename R>
void LoadReconstructData(std::shared_ptr<VectorSet>& real_vecset, std::shared_ptr<VectorSet>& rec_vecset, std::shared_ptr<VectorSet>& quan_vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, DistCalcMethod distCalcMethod, int k)
{
    int m = 960;
    int M = 480;
    int Ks = 256;
    int QuanDim = m / M;
    std::shared_ptr<VectorSet> rec_queryset;
    std::shared_ptr<VectorSet> loaded_codebooks;
    std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<R>(), m, VectorFileType::DEFAULT));
    auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile("gist_vector.bin"))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
        exit(1);
    }
    real_vecset = vectorReader->GetVectorSet();
    if (ErrorCode::Success != vectorReader->LoadFile("gist_query.bin"))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
        exit(1);
    }
    queryset = vectorReader->GetVectorSet();

    int n = real_vecset->Count();
    int q = queryset->Count();

    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr->Initialize("gist_codebooks.bin", std::ios::binary | std::ios::in));
    SPTAG::COMMON::IQuantizer::LoadIQuantizer(ptr);
    BOOST_ASSERT(SPTAG::COMMON::DistanceUtils::Quantizer != nullptr);

    ByteArray PQvec = ByteArray::Alloc(sizeof(std::uint8_t) * n * M);
    ByteArray rec_vec = ByteArray::Alloc(sizeof(R) * n * m);

    rec_vecset.reset(new BasicVectorSet(rec_vec, GetEnumValueType<R>(), m, n));
    quan_vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<std::uint8_t>(), M, n));

    for (int i = 0; i < real_vecset->Count(); i++) {
        COMMON::DistanceUtils::Quantizer->QuantizeVector(real_vecset->GetVector(i), (uint8_t*)quan_vecset->GetVector(i));
        COMMON::DistanceUtils::Quantizer->ReconstructVector((uint8_t*)quan_vecset->GetVector(i), rec_vecset->GetVector(i));
    }
    quan_vecset->Save("quan_vector.bin");
    rec_vecset->Save("rec_vector.bin");

    ByteArray pq_query = ByteArray::Alloc(sizeof(uint8_t) * q * M);
    ByteArray rec_query = ByteArray::Alloc(sizeof(R) * q * m);

    std::shared_ptr<VectorSet> pq_queryset;
    pq_queryset.reset(new BasicVectorSet(pq_query, GetEnumValueType<std::uint8_t>(), M, q));
    queryset.reset(new BasicVectorSet(rec_query, GetEnumValueType<R>(), m, q));

    for (int i = 0; i < queryset->Count(); i++) {
        COMMON::DistanceUtils::Quantizer->QuantizeVector(queryset->GetVector(i), (uint8_t*)pq_queryset->GetVector(i));
        COMMON::DistanceUtils::Quantizer->ReconstructVector((uint8_t*)pq_queryset->GetVector(i), rec_queryset->GetVector(i));
    }

    ByteArray tru = ByteArray::Alloc(sizeof(SizeType) * queryset->Count() * 2 * k);
    //ByteArray quan_tru = ByteArray::Alloc(sizeof(SizeType) * queryset->Count() * 2 * k);
    auto quan_holder = COMMON::DistanceUtils::Quantizer;
    COMMON::DistanceUtils::Quantizer.reset();

    for (SizeType i = 0; i < queryset->Count(); ++i)
    {
        SizeType* neighbors = ((SizeType*)tru.Data()) + i * 2 * k;

        COMMON::QueryResultSet<R> res((const R*)queryset->GetVector(i), 2 * k);
        res.Reset();

        for (SizeType j = 0; j < real_vecset->Count(); j++)
        {
            float dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), reinterpret_cast<R*>(real_vecset->GetVector(j)), queryset->Dimension(), distCalcMethod);
            res.AddPoint(j, dist);
        }

        res.SortResult();
        for (int j = 0; j < 2 * k; j++) neighbors[j] = res.GetResult(j)->VID;
    }
    COMMON::DistanceUtils::Quantizer = quan_holder;
    truth.reset(new BasicVectorSet(tru, GetEnumValueType<float>(), 2 * k, queryset->Count()));
}

template <typename R>
void GenerateReconstructData(std::shared_ptr<VectorSet>& real_vecset, std::shared_ptr<VectorSet>& rec_vecset, std::shared_ptr<VectorSet>& quan_vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, DistCalcMethod distCalcMethod, int k)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<R> dist(-1000, 1000);
    int n = 10000, q = 2000;
    int m = 256;
    int M = 128;
    int Ks = 256;

    int QuanDim = m / M;
    ByteArray PQvec = ByteArray::Alloc(sizeof(std::uint8_t) * n * M);
    ByteArray real_vec = ByteArray::Alloc(sizeof(R) * n * m);
    ByteArray rec_vec = ByteArray::Alloc(sizeof(R) * n * m);
    std::vector< std::vector<std::uint8_t> > baseQ(n, std::vector<std::uint8_t>(M));

    R* vecs = new R[n * m];
    for (int i = 0; i < n * m; i++) {
        vecs[i] = dist(gen);
        ((R*)real_vec.Data())[i] = (R)vecs[i];
    }

    real_vecset.reset(new BasicVectorSet(real_vec, GetEnumValueType<R>(), m, n));
    rec_vecset.reset(new BasicVectorSet(rec_vec, GetEnumValueType<R>(), m, n));
    quan_vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<std::uint8_t>(), M, n));
    std::cout << "Building codebooks!" << std::endl;
    std::string CODEBOOK_FILE = "test-quantizer-tree.bin";
    R* codebooks = new R[M * Ks * QuanDim];

    R* kmeans = new R[Ks * QuanDim];
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
                R* newmeans = new R[QuanDim]();
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
    auto baseQuantizer = std::make_shared<SPTAG::COMMON::PQQuantizer<R>>(M, Ks, QuanDim, false, codebooks);
    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr != nullptr && ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::out));
    baseQuantizer->SaveQuantizer(ptr);
    ptr->ShutDown();

    BOOST_ASSERT(ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::in));
    SPTAG::COMMON::IQuantizer::LoadIQuantizer(ptr);
    SPTAG::COMMON::DistanceUtils::Quantizer = baseQuantizer;
    BOOST_ASSERT(SPTAG::COMMON::DistanceUtils::Quantizer);

    for (int i = 0; i < n; i++) {
        auto nvec = &vecs[i * m];
        COMMON::DistanceUtils::Quantizer->QuantizeVector(nvec, (uint8_t*)quan_vecset->GetVector(i));
    }
    for (int i = 0; i < n; i++) {
        
        COMMON::DistanceUtils::Quantizer->ReconstructVector((uint8_t*)quan_vecset->GetVector(i), rec_vecset->GetVector(i));
    }
    quan_vecset->Save("quan_vector.bin");
    rec_vecset->Save("rec_vector.bin");
    ByteArray real_query = ByteArray::Alloc(sizeof(R) * q * m);
    ByteArray pq_query = ByteArray::Alloc(sizeof(uint8_t) * q * M);
    ByteArray rec_query = ByteArray::Alloc(sizeof(R) * q * m);
    std::shared_ptr<VectorSet> real_queryset;
    std::shared_ptr<VectorSet> pq_queryset;
    std::shared_ptr<VectorSet> rec_queryset;
    real_queryset.reset(new BasicVectorSet(real_query, GetEnumValueType<R>(), m, q));
    pq_queryset.reset(new BasicVectorSet(pq_query, GetEnumValueType<std::uint8_t>(), M, q));
    rec_queryset.reset(new BasicVectorSet(rec_query, GetEnumValueType<R>(), m, q));
    R* queries = new R[q * m];
    for (int i = 0; i < q * m; i++) {
        queries[i] = dist(gen);
        ((R*)real_query.Data())[i] = (R)queries[i];
    }
    for (int i = 0; i < q; i++) {
        COMMON::DistanceUtils::Quantizer->QuantizeVector(real_queryset->GetVector(i), (uint8_t*)pq_queryset->GetVector(i));
        COMMON::DistanceUtils::Quantizer->ReconstructVector((uint8_t*)pq_queryset->GetVector(i), rec_queryset->GetVector(i));
    }
    

    queryset = real_queryset;
    ByteArray tru = ByteArray::Alloc(sizeof(SizeType) * queryset->Count() * k);
    //ByteArray quan_tru = ByteArray::Alloc(sizeof(SizeType) * queryset->Count() * 2 * k);
    auto quan_holder = COMMON::DistanceUtils::Quantizer;
    COMMON::DistanceUtils::Quantizer.reset();

    for (SizeType i = 0; i < queryset->Count(); ++i)
    {
        SizeType* neighbors = ((SizeType*)tru.Data()) + i * k;

        COMMON::QueryResultSet<R> res((const R*)queryset->GetVector(i), k);

        for (SizeType j = 0; j < real_vecset->Count(); j++)
        {
            float dist = COMMON::DistanceUtils::ComputeDistance<R>(res.GetTarget(), reinterpret_cast<R*>(real_vecset->GetVector(j)), queryset->Dimension(), distCalcMethod);
            res.AddPoint(j, dist);
        }

        res.SortResult();
        for (int j = 0; j < k; j++) neighbors[j] = res.GetResult(j)->VID;
    }
    COMMON::DistanceUtils::Quantizer = quan_holder;
    truth.reset(new BasicVectorSet(tru, GetEnumValueType<float>(), k, queryset->Count()));
//    truth->Save("rec_truth.");

}

template <typename R>
void ReconstructTest(IndexAlgoType algo, DistCalcMethod distMethod)
{
    std::shared_ptr<VectorSet> real_vecset, rec_vecset, quan_vecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    GenerateReconstructData<R>(real_vecset, rec_vecset, quan_vecset, metaset, queryset, truth, distMethod, 10);
    //LoadReconstructData<R>(real_vecset, rec_vecset, quan_vecset, metaset, queryset, truth, distMethod, 10);
    
    auto quantizer = COMMON::DistanceUtils::Quantizer;
    COMMON::DistanceUtils::Quantizer.reset();
    BOOST_ASSERT(!COMMON::DistanceUtils::Quantizer);
    auto real_idx = PerfBuild<R>(algo, Helper::Convert::ConvertToString<DistCalcMethod>(distMethod), real_vecset, metaset, queryset, 10, truth, "real_idx");
    Search<R>(real_idx, queryset, 10, truth);
    auto rec_idx = PerfBuild<R>(algo, Helper::Convert::ConvertToString<DistCalcMethod>(distMethod), rec_vecset, metaset, queryset, 10, truth, "rec_idx");
    Search<R>(rec_idx, queryset, 10, truth);
    COMMON::DistanceUtils::Quantizer = quantizer;
    auto quan_idx = PerfBuild<std::uint8_t>(algo, Helper::Convert::ConvertToString<DistCalcMethod>(distMethod), quan_vecset, metaset, queryset, 10, truth, "quan_idx");

    LOG(Helper::LogLevel::LL_Info, "Test search with SDC");
    Search<R>(quan_idx, queryset, 10, truth);
    
    LOG(Helper::LogLevel::LL_Info, "Test search with ADC");
    SPTAG::COMMON::DistanceUtils::Quantizer->SetEnableADC(true);
    Search<R>(quan_idx, queryset, 10, truth);
}


BOOST_AUTO_TEST_SUITE(ReconstructIndexSimilarityTest)

BOOST_AUTO_TEST_CASE(BKTReconstructTest)
{
    SPTAG::COMMON::DistanceUtils::Quantizer.reset();

    ReconstructTest<float>(IndexAlgoType::BKT, DistCalcMethod::L2);

    SPTAG::COMMON::DistanceUtils::Quantizer.reset();
}

BOOST_AUTO_TEST_CASE(KDTReconstructTest)
{
    SPTAG::COMMON::DistanceUtils::Quantizer.reset();

    ReconstructTest<float>(IndexAlgoType::KDT, DistCalcMethod::L2);

    SPTAG::COMMON::DistanceUtils::Quantizer.reset();
}

BOOST_AUTO_TEST_SUITE_END()
