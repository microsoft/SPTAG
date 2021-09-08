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
void ADCSearch(std::shared_ptr<VectorIndex>& vecIndex, std::shared_ptr<VectorSet>& queryset, int k, std::shared_ptr<VectorSet>& truth)
{
    std::vector<QueryResult> res(queryset->Count(), QueryResult(nullptr, k, true));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SizeType i = 0; i < queryset->Count(); i++)
    {
        res[i].SetTarget(queryset->GetVector(i));
        //prefetch L2
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
            if (nn[j] < 0) {
                LOG(Helper::LogLevel::LL_Info, "error nn[%d][%d]:%d", i, j, nn[j]);
                continue;
            }
            //std::string truthstr = std::to_string(nn[j]);
            //ByteArray truthmeta = ByteArray((std::uint8_t*)(truthstr.c_str()), truthstr.length(), false);
            //float truthdist = vecIndex->ComputeDistance(queryset->GetVector(i), vecIndex->GetSample(truthmeta, deleted));
            float truthdist = vecIndex->ComputeDistance(queryset->GetVector(i), vecIndex->GetSample(nn[j]));
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

    for (SizeType i = 0; i < queryset->Count(); i++) {
        SizeType* nn = (SizeType*)(truth->GetVector(i));
        std::vector<bool> visited(k, false);
        for (int j = 0; j < truthDimension; j++) {
            float truthdist = vecIndex->ComputeDistance(queryset->GetVector(i), vecIndex->GetSample(nn[j]));
            for (int l = 0; l < k; l++) {
                if (visited[l]) continue;

                if (res[i].GetResult(l)->VID == nn[j]) {
                    recall += 1.0;
                    visited[l] = true;
                    break;
                }
                //else if (fabs(res[i].GetResult(l)->Dist - truthdist) <= eps) {
                //	recall += 1.0;
                //	visited[l] = true;
                //	break;
                //}
            }
        }
    }

    LOG(Helper::LogLevel::LL_Info, "Recall %d@%d: %f\n", k, truthDimension, recall / queryset->Count() / truthDimension);
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
    vecIndex->SetParameter("MaxCheck", "8192");
    vecIndex->SetParameter("MaxCheckForRefineGraph", "8192");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta, true));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
    ADCSearch<T>(vecIndex, queryset, k, truth);
}


template <typename T>
void GeneratePQData_ADC(std::shared_ptr<VectorSet>& vecset, std::shared_ptr<VectorSet>& real_vecset, std::shared_ptr<MetadataSet>& metaset, std::shared_ptr<VectorSet>& queryset, std::shared_ptr<VectorSet>& real_queryset, std::shared_ptr<VectorSet>& truth, std::string distCalcMethod, int k)
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
    ByteArray reconstruct_vec = ByteArray::Alloc(sizeof(float) * n * m);
    ByteArray PQquery = ByteArray::Alloc(sizeof(float) * q * QuanDim * Ks);

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
    }
    else {
        ByteArray real_vec = ByteArray::Alloc(sizeof(float) * n * m);
        for (int i = 0; i < n * m; i++) {
            ((float*)real_vec.Data())[i] = (float)COMMON::Utils::rand(255, 0);
        }
        real_vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<T>(), m, n));
        ByteArray real_query = ByteArray::Alloc(sizeof(float) * q * m);
        for (int i = 0; i < q * m; i++) {
            ((float*)real_query.Data())[i] = (float)COMMON::Utils::rand(255, 0);
        }
        real_queryset.reset(new BasicVectorSet(real_query, GetEnumValueType<T>(), m, q));

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
    }

    std::string CODEBOOK_FILE = "test-quantizer-adc.bin";
    float* codebooks = new float[M * Ks * QuanDim];
    if (fileexists("codebooks.bin")) {
        std::shared_ptr<VectorSet> temp_codebooks;
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), Ks * M, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("codebooks.bin"))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read coodebooks file.\n");
            exit(1);
        }
        temp_codebooks = vectorReader->GetVectorSet();
        std::memcpy(codebooks, temp_codebooks->GetData(), sizeof(float) * QuanDim * Ks * M);
    }
    else {
        std::cout << "Building codebooks!" << std::endl;
        float* kmeans = new float[Ks * QuanDim];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < Ks; j++) {
                for (int t = 0; t < QuanDim; t++) {
                    kmeans[j * QuanDim + t] = ((float*)(real_vecset->GetData()))[j * m + i * QuanDim + t];
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
                            now_dis += (1.0 * ((float*)(real_vecset->GetData()))[ii * m + i * QuanDim + kk] - kmeans[jj * QuanDim + kk]) * (1.0 * ((float*)(real_vecset->GetData()))[ii * m + i * QuanDim + kk] - kmeans[jj * QuanDim + kk]);
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
                                newmeans[kk] += ((float*)(real_vecset->GetData()))[jj * m + i * QuanDim + kk];
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
                        now_dis += (((float*)(real_vecset->GetData()))[ii * m + i * QuanDim + t] - kmeans[j * QuanDim + t]) * (((float*)(real_vecset->GetData()))[ii * m + i * QuanDim + t] - kmeans[j * QuanDim + t]);
                    }
                    if (now_dis < min_dis) {
                        min_dis = now_dis;
                        min_id = ii;
                    }
                }
                for (int t = 0; t < QuanDim; t++) {
                    codebooks[i * Ks * QuanDim + j * QuanDim + t] = ((float*)(real_vecset->GetData()))[min_id * m + i * QuanDim + t];
                }
            }
            delete[] belong;
        }
        delete[] kmeans;
    }
    std::cout << "Codebooks Building Finish!" << std::endl;
    auto baseQuantizer = std::make_shared<SPTAG::COMMON::PQQuantizer>(QuanDim, Ks, M, true, codebooks);
    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr != nullptr && ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::out));
    baseQuantizer->SaveQuantizer(ptr);
    ptr->ShutDown();

    BOOST_ASSERT(ptr->Initialize(CODEBOOK_FILE.c_str(), std::ios::binary | std::ios::in));
    SPTAG::COMMON::Quantizer::LoadQuantizer(ptr, SPTAG::QuantizerType::PQQuantizer);
    BOOST_ASSERT(SPTAG::COMMON::DistanceUtils::Quantizer != nullptr);

    SPTAG::COMMON::DistanceUtils::Quantizer = nullptr;
    std::cout << "Quantize Vector" << std::endl;
    for (int i = 0; i < n; i++) {
        //std::cout << real_vecset->Dimension() << " " << real_vecset->Count() << std::endl;
        baseQuantizer->QuantizeVector((const float*)(real_vecset->GetVector(i)), ((T*)PQvec.Data()) + i * QuanDim);
        //std::cout << i << " " << baseQ[i].size() << std::endl;
    }
    vecset.reset(new BasicVectorSet(PQvec, GetEnumValueType<T>(), QuanDim, n));
    vecset->Save("ADCtest_vector.bin");


    for (int i = 0; i < n; i++) {
        T* quan = (T*)(vecset->GetVector(i));
        for (int j = 0; j < QuanDim; j++) {
            std::memcpy(((float*)(reconstruct_vec.Data())) + i * m + j * M, codebooks + j * Ks * M + quan[j] * M, sizeof(float) * M);
        }
    }
    vecset.reset(new BasicVectorSet(reconstruct_vec, GetEnumValueType<float>(), m, n));
    vecset->Save("ADCtest_reconstruct.bin");


    std::cout << "Quantize Query Vector" << std::endl;

    int total_size = Ks * QuanDim;
    int block_size = Ks * M;
    auto L2Dist = COMMON::DistanceCalcSelector<float>(DistCalcMethod::L2);
    float* destPtr = (float*)(PQquery.Data());
    float* queryPtr = (float*)(real_queryset->GetData());
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < q; i++) {
        for (int j = 0; j < QuanDim; j++) {
            float* basec = codebooks + j * block_size;
            for (int ii = 0; ii < Ks; ii++) {
                destPtr[ii] = L2Dist(queryPtr, basec + ii * M, M);
            }
            destPtr += Ks;
            queryPtr += M;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Query Quantization time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(q)) << "us" << std::endl;

    //real_queryset.reset(new BasicVectorSet(querys, GetEnumValueType<float>(), m, q));
    queryset.reset(new BasicVectorSet(PQquery, GetEnumValueType<float>(), Ks * QuanDim, q));
    queryset->Save("test_query_adc.bin");
    //}

    /*
    if (fileexists(("test_truth_adc." + distCalcMethod).c_str())) {
        std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(GetEnumValueType<float>(), k, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile("test_truth_adc." + distCalcMethod))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read truth file.\n");
            exit(1);
        }
        truth = vectorReader->GetVectorSet();
    }
    else {
    */
    omp_set_num_threads(5);

    DistCalcMethod distMethod;
    Helper::Convert::ConvertStringTo(distCalcMethod.c_str(), distMethod);
    if (distMethod == DistCalcMethod::Cosine) {
        std::cout << "Normalize vecset!" << std::endl;
        COMMON::Utils::BatchNormalize((T*)(vecset->GetData()), vecset->Count(), vecset->Dimension(), COMMON::Utils::GetBase<T>(), 5);
    }

    ByteArray tru = ByteArray::Alloc(sizeof(float) * queryset->Count() * k);

#pragma omp parallel for
    for (SizeType i = 0; i < real_queryset->Count(); ++i)
    {
        SizeType* neighbors = ((SizeType*)tru.Data()) + i * k;
        COMMON::QueryResultSet<float> res((const float*)real_queryset->GetVector(i), k);
        for (SizeType j = 0; j < real_vecset->Count(); j++)
        {
            float dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), (const float*)(real_vecset->GetVector(j)), real_queryset->Dimension(), distMethod);
            res.AddPoint(j, dist);
        }
        res.SortResult();
        for (int j = 0; j < k; j++) neighbors[j] = res.GetResult(j)->VID;
    }
    truth.reset(new BasicVectorSet(tru, GetEnumValueType<float>(), k, real_queryset->Count()));
    truth->Save("test_truth_adc." + distCalcMethod);
    //}
}

void Replace(const char* filename, std::string oStr, std::string nStr) {
    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr->Initialize(filename, std::ios::in));
    auto ptrout = SPTAG::f_createIO();
    BOOST_ASSERT(ptrout->Initialize((filename + std::string(".tmp")).c_str(), std::ios::out));

    std::uint64_t c_bufferSize = 1 << 16;
    std::unique_ptr<char[]> line(new char[c_bufferSize]);

    while (true)
    {
        if (!ptr->ReadString(c_bufferSize, line)) break;

        std::string tmp(line.get());
        tmp = Helper::StrUtils::ReplaceAll(tmp, oStr, nStr) + "\n";
        ptrout->WriteString(tmp.c_str());
    }
    ptr->ShutDown();
    ptrout->ShutDown();

    std::remove(filename);
    std::rename((filename + std::string(".tmp")).c_str(), filename);
}

template <typename T>
void ADCPQTest(IndexAlgoType algo, std::string distCalcMethod)
{
    std::shared_ptr<VectorSet> real_vecset, vecset, queryset, real_queryset, truth;
    std::shared_ptr<MetadataSet> metaset;
    GeneratePQData_ADC<T>(vecset, real_vecset, metaset, queryset, real_queryset, truth, distCalcMethod, 10);

    SPTAG::COMMON::DistanceUtils::Quantizer = nullptr;
    ADCBuild<float>(algo, distCalcMethod, vecset, metaset, real_queryset, 10, truth, "testindices-adc");

    std::remove("testindices-adc\\vectors.bin");
    std::rename("ADCtest_vector.bin", "testindices-adc\\vectors.bin");
    Replace("testindices-adc\\indexloader.ini", "Float", "UInt8");

    auto ptr = SPTAG::f_createIO();
    BOOST_ASSERT(ptr->Initialize("test-quantizer-adc.bin", std::ios::binary | std::ios::in));
    SPTAG::COMMON::Quantizer::LoadQuantizer(ptr, SPTAG::QuantizerType::PQQuantizer);
    BOOST_ASSERT(SPTAG::COMMON::DistanceUtils::Quantizer != nullptr);

    std::cout << "ADCBuild Finish!" << std::endl;
    std::shared_ptr<VectorIndex> vecIndex;
    BOOST_CHECK(ErrorCode::Success == VectorIndex::LoadIndex("testindices-adc", vecIndex));
    BOOST_CHECK(nullptr != vecIndex);

    ADCSearch<T>(vecIndex, queryset, 10, truth);

}


BOOST_AUTO_TEST_SUITE(ADCTest)

BOOST_AUTO_TEST_CASE(ADCBKTTest)
{
    ADCPQTest<std::uint8_t>(IndexAlgoType::BKT, "L2");

    SPTAG::COMMON::DistanceUtils::Quantizer = nullptr;
}

BOOST_AUTO_TEST_SUITE_END()

