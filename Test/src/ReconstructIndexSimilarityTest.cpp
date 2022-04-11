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
        res[i].SetTarget((const T*)queryset->GetVector(i), vecIndex->m_pQuantizer);
        vecIndex->SearchIndex(res[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(queryset->Count())) << "us" << std::endl;

    float eps = 1e-6f, recall = 0;

    int truthDimension = min(k, truth->Dimension());
    for (SizeType i = 0; i < queryset->Count(); i++) {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        std::vector<bool> visited(2 * k, false);
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
std::shared_ptr<VectorIndex> PerfBuild(IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<VectorSet>& vec, std::shared_ptr<MetadataSet>& meta, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth, std::string out, std::shared_ptr<COMMON::IQuantizer> quantizer)
{
    std::shared_ptr<VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    vecIndex->SetQuantizer(quantizer);
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
void GenerateReconstructData(std::shared_ptr<VectorSet>& real_vecset, std::shared_ptr<VectorSet>& rec_vecset, std::shared_ptr<VectorSet>& quan_vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& truth, DistCalcMethod distCalcMethod, int k, std::shared_ptr<COMMON::IQuantizer>& quantizer)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<R> dist(-1000, 1000);
    int n = 1000, q = 200;
    int m = 256;
    int M = 128;
    int Ks = 256;
    int QuanDim = m / M;
    std::string CODEBOOK_FILE = "quantest_quantizer.bin";

    if (fileexists("quantest_vector.bin") && fileexists("quantest_query.bin")) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<R>(), m, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("quantest_vector.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        real_vecset = vectorReader->GetVectorSet();

        if (ErrorCode::Success != vectorReader->LoadFile("quantest_query.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
            exit(1);
        }
        queryset = vectorReader->GetVectorSet();
    }
    else {
        ByteArray real_vec = ByteArray::Alloc(sizeof(R) * n * m);
        for (int i = 0; i < n * m; i++) {
            ((R*)real_vec.Data())[i] = (R)(dist(gen));
        }
        real_vecset.reset(new BasicVectorSet(real_vec, GetEnumValueType<R>(), m, n));
        real_vecset->Save("quantest_vector.bin");

        ByteArray real_query = ByteArray::Alloc(sizeof(R) * q * m);
        for (int i = 0; i < q * m; i++) {
            ((R*)real_query.Data())[i] = (R)(dist(gen));
        }
        queryset.reset(new BasicVectorSet(real_query, GetEnumValueType<R>(), m, q));
        queryset->Save("quantest_query.bin");
    }

    if (fileexists(("quantest_truth." + SPTAG::Helper::Convert::ConvertToString(distCalcMethod)).c_str())) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), k, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("quantest_truth." + SPTAG::Helper::Convert::ConvertToString(distCalcMethod)))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read truth file.\n");
            exit(1);
        }
        truth = vectorReader->GetVectorSet();
    }
    else {
        omp_set_num_threads(5);

        ByteArray tru = ByteArray::Alloc(sizeof(SizeType) * queryset->Count() * k);

#pragma omp parallel for
        for (SizeType i = 0; i < queryset->Count(); ++i)
        {
            SizeType* neighbors = ((SizeType*)tru.Data()) + i * k;

            COMMON::QueryResultSet<R> res((const R*)queryset->GetVector(i), k);
            for (SizeType j = 0; j < real_vecset->Count(); j++)
            {
                float dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), reinterpret_cast<R*>(real_vecset->GetVector(j)), queryset->Dimension(), distCalcMethod);
                res.AddPoint(j, dist);
            }
            res.SortResult();
            for (int j = 0; j < k; j++) neighbors[j] = res.GetResult(j)->VID;
        }
        truth.reset(new BasicVectorSet(tru, GetEnumValueType<float>(), k, queryset->Count()));
        truth->Save("quantest_truth." + SPTAG::Helper::Convert::ConvertToString(distCalcMethod));
    }

    if (fileexists(CODEBOOK_FILE.c_str()) && fileexists("quantest_quan_vector.bin") && fileexists("quantest_rec_vector.bin")) {
        auto ptr = SPTAG::f_createIO();
        if (ptr == nullptr || !ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::in)) {
            BOOST_ASSERT("Canot Open CODEBOOK_FILE to read!" == "Error");
        }
        quantizer->LoadIQuantizer(ptr);
        BOOST_ASSERT(quantizer);

        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<R>(), m, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("quantest_rec_vector.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        rec_vecset = vectorReader->GetVectorSet();

        std::shared_ptr<Helper::ReaderOptions> quanOptions(new Helper::ReaderOptions(GetEnumValueType<std::uint8_t>(), M, VectorFileType::DEFAULT));
        vectorReader = Helper::VectorSetReader::CreateInstance(quanOptions);
        if (ErrorCode::Success != vectorReader->LoadFile("quantest_quan_vector.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            exit(1);
        }
        quan_vecset = vectorReader->GetVectorSet();
    }
    else {
        omp_set_num_threads(16);

        std::cout << "Building codebooks!" << std::endl;
        R* vecs = (R*)(real_vecset->GetData());

        std::unique_ptr<R[]> codebooks = std::make_unique<R[]>(M * Ks * QuanDim);
        std::unique_ptr<int[]> belong(new int[n]);
        for (int i = 0; i < M; i++) {
            R* kmeans = codebooks.get() + i * Ks * QuanDim;
            for (int j = 0; j < Ks; j++) {
                std::memcpy(kmeans + j * QuanDim, vecs + j * m + i * QuanDim, sizeof(R) * QuanDim);
            }
            int cnt = 100;
            while (cnt--) {
                //calculate cluster
#pragma omp parallel for
                for (int ii = 0; ii < n; ii++) {
                    double min_dis = 1e9;
                    int min_id = 0;
                    for (int jj = 0; jj < Ks; jj++) {
                        double now_dis = COMMON::DistanceUtils::ComputeDistance(vecs + ii * m + i * QuanDim, kmeans + jj * QuanDim, QuanDim, DistCalcMethod::L2);
                        if (now_dis < min_dis) {
                            min_dis = now_dis;
                            min_id = jj;
                        }
                    }
                    belong[ii] = min_id;
                }
                //recalculate kmeans
                std::memset(kmeans, 0, sizeof(R) * Ks * QuanDim);
#pragma omp parallel for
                for (int ii = 0; ii < Ks; ii++) {
                    int num = 0;
                    for (int jj = 0; jj < n; jj++) {
                        if (belong[jj] == ii) {
                            num++;
                            for (int kk = 0; kk < QuanDim; kk++) {
                                kmeans[ii * QuanDim + kk] += vecs[jj * m + i * QuanDim + kk];
                            }
                        }
                    }
                    for (int jj = 0; jj < QuanDim; jj++) {
                        kmeans[ii * QuanDim + jj] /= num;
                    }
                }
            }
        }

        std::cout << "Building Finish!" << std::endl;
        quantizer = std::make_shared<SPTAG::COMMON::PQQuantizer<R>>(M, Ks, QuanDim, false, std::move(codebooks));
        auto ptr = SPTAG::f_createIO();
        if (ptr == nullptr || !ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::out)) {
            BOOST_ASSERT("Canot Open CODEBOOK_FILE to write!" == "Error");
        }
        quantizer->SaveQuantizer(ptr);
        ptr->ShutDown();

        if (!ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::in)) {
            BOOST_ASSERT("Canot Open CODEBOOK_FILE to read!" == "Error");
        }
        quantizer->LoadIQuantizer(ptr);
        BOOST_ASSERT(quantizer);

        rec_vecset.reset(new BasicVectorSet(ByteArray::Alloc(sizeof(R) * n * m), GetEnumValueType<R>(), m, n));
        quan_vecset.reset(new BasicVectorSet(ByteArray::Alloc(sizeof(std::uint8_t) * n * M), GetEnumValueType<std::uint8_t>(), M, n));
        for (int i = 0; i < n; i++) {
            auto nvec = &vecs[i * m];
            quantizer->QuantizeVector(nvec, (uint8_t*)quan_vecset->GetVector(i));
            quantizer->ReconstructVector((uint8_t*)quan_vecset->GetVector(i), rec_vecset->GetVector(i));
        }
        quan_vecset->Save("quantest_quan_vector.bin");
        rec_vecset->Save("quantest_rec_vector.bin");
    }
}

template <typename R>
void ReconstructTest(IndexAlgoType algo, DistCalcMethod distMethod)
{
    std::shared_ptr<VectorSet> real_vecset, rec_vecset, quan_vecset, queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    std::shared_ptr<COMMON::IQuantizer> quantizer;
    GenerateReconstructData<R>(real_vecset, rec_vecset, quan_vecset, metaset, queryset, truth, distMethod, 10, quantizer);
    //LoadReconstructData<R>(real_vecset, rec_vecset, quan_vecset, metaset, queryset, truth, distMethod, 10);
    
    auto real_idx = PerfBuild<R>(algo, Helper::Convert::ConvertToString<DistCalcMethod>(distMethod), real_vecset, metaset, queryset, 10, truth, "real_idx", nullptr);
    Search<R>(real_idx, queryset, 10, truth);
    auto rec_idx = PerfBuild<R>(algo, Helper::Convert::ConvertToString<DistCalcMethod>(distMethod), rec_vecset, metaset, queryset, 10, truth, "rec_idx", nullptr);
    Search<R>(rec_idx, queryset, 10, truth);
    auto quan_idx = PerfBuild<std::uint8_t>(algo, Helper::Convert::ConvertToString<DistCalcMethod>(distMethod), quan_vecset, metaset, queryset, 10, truth, "quan_idx", quantizer);

    LOG(Helper::LogLevel::LL_Info, "Test search with SDC");
    Search<R>(quan_idx, queryset, 10, truth);
    
    LOG(Helper::LogLevel::LL_Info, "Test search with ADC");
    quan_idx->SetQuantizerADC(true);
    Search<R>(quan_idx, queryset, 10, truth);
}


BOOST_AUTO_TEST_SUITE(ReconstructIndexSimilarityTest)

BOOST_AUTO_TEST_CASE(BKTReconstructTest)
{

    ReconstructTest<float>(IndexAlgoType::BKT, DistCalcMethod::L2);

}

BOOST_AUTO_TEST_CASE(KDTReconstructTest)
{

    ReconstructTest<float>(IndexAlgoType::KDT, DistCalcMethod::L2);

}

BOOST_AUTO_TEST_SUITE_END()
