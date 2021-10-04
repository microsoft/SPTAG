// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/VectorIndex.h"
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <set>
#include <ctime>
#include <chrono>

using namespace SPTAG;

class SearcherOptions : public Helper::ReaderOptions
{
public:
    SearcherOptions() : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
    {
        AddRequiredOption(m_queryFile, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_indexFolder, "-x", "--index", "Index folder.");
        AddOptionalOption(m_truthFile, "-r", "--truth", "Truth file.");
        AddOptionalOption(m_resultFile, "-o", "--result", "Output result file.");
        AddOptionalOption(m_maxCheck, "-m", "--maxcheck", "MaxCheck for index.");
        AddOptionalOption(m_withMeta, "-a", "--withmeta", "Output metadata instead of vector id.");
        AddOptionalOption(m_K, "-k", "--KNN", "K nearest neighbors for search.");
        AddOptionalOption(m_truthK, "-tk", "--truthKNN", "truth set number.");
        AddOptionalOption(m_dataFile, "-df", "--data", "original data file.");
        AddOptionalOption(m_dataFileType, "-dft", "--dataFileType", "original data file type. (TXT, or DEFAULT)");
        AddOptionalOption(m_batch, "-b", "--batchsize", "Batch query size.");
        AddOptionalOption(m_genTruth, "-g", "--gentruth", "Generate truth file.");
        AddOptionalOption(m_debugQuery, "-q", "--debugquery", "Debug query number.");
        AddOptionalOption(m_quantizerFile, "-pq", "--quantizer", "Quantizer File");
        AddOptionalOption(m_enableADC, "-adc", "--adc", "Enable ADC Distance computation");
        AddOptionalOption(m_reconstructType, "-rt", "--reconstructtype", "Reconstruction value type for quantized vectors. Default is float.");
    }

    ~SearcherOptions() {}

    std::string m_queryFile;

    std::string m_indexFolder;

    std::string m_dataFile = "";

    std::string m_truthFile = "";

    std::string m_resultFile = "";

    std::string m_maxCheck = "8192";

    VectorFileType m_dataFileType = VectorFileType::DEFAULT;

    int m_withMeta = 0;

    int m_K = 32;

    int m_truthK = -1;

    int m_batch = 10000;

    int m_genTruth = 0;

    int m_debugQuery = -1;

    std::string m_quantizerFile;

    bool m_enableADC = false;

    VectorValueType m_reconstructType;
};

template <typename T>
float CalcRecall(VectorIndex* index, std::vector<QueryResult>& results, const std::vector<std::set<SizeType>>& truth, SizeType NumQuerys, int K, int truthK, std::shared_ptr<SPTAG::VectorSet> querySet, std::shared_ptr<SPTAG::VectorSet> vectorSet, std::ofstream& log, bool debug = false)
{
    float eps = 1e-6f;
    float meanrecall = 0, minrecall = MaxDist, maxrecall = 0, stdrecall = 0;
    std::vector<float> thisrecall(NumQuerys, 0);
    std::unique_ptr<bool[]> visited(new bool[K]);
    for (SizeType i = 0; i < NumQuerys; i++)
    {
        memset(visited.get(), 0, K * sizeof(bool));
        for (SizeType id : truth[i])
        {
            for (int j = 0; j < K; j++)
            {
                if (visited[j] || results[i].GetResult(j)->VID < 0) continue;

                if (results[i].GetResult(j)->VID == id)
                {
                    thisrecall[i] += 1;
                    visited[j] = true;
                    break;
                }
                else if (vectorSet != nullptr) {
                    float dist = index->ComputeDistance(querySet->GetVector(i), vectorSet->GetVector(results[i].GetResult(j)->VID));
                    float truthDist = index->ComputeDistance(querySet->GetVector(i), vectorSet->GetVector(id));
                    if (index->GetDistCalcMethod() == SPTAG::DistCalcMethod::Cosine && fabs(dist - truthDist) < eps) {
                        thisrecall[i] += 1;
                        visited[j] = true;
                        break;
                    }
                    else if (index->GetDistCalcMethod() == SPTAG::DistCalcMethod::L2 && fabs(dist - truthDist) < eps * (dist + eps)) {
                        thisrecall[i] += 1;
                        visited[j] = true;
                        break;
                    }
                }
            }
        }
        thisrecall[i] /= truthK;
        meanrecall += thisrecall[i];
        if (thisrecall[i] < minrecall) minrecall = thisrecall[i];
        if (thisrecall[i] > maxrecall) maxrecall = thisrecall[i];

        if (debug) {
            std::string ll("recall:" + std::to_string(thisrecall[i]) + "\ngroundtruth:");
            std::vector<NodeDistPair> truthvec;
            for (SizeType id : truth[i]) {
                float truthDist = 0.0;
                if (vectorSet != nullptr) {
                    truthDist = index->ComputeDistance(querySet->GetVector(i), vectorSet->GetVector(id));
                }
                truthvec.emplace_back(id, truthDist);
            }
            std::sort(truthvec.begin(), truthvec.end());
            for (int j = 0; j < truthvec.size(); j++)
                ll += std::to_string(truthvec[j].node) + "@" + std::to_string(truthvec[j].distance) + ",";
            LOG(Helper::LogLevel::LL_Info, "%s\n", ll.c_str());
            ll = "ann:";
            for (int j = 0; j < K; j++)
                ll += std::to_string(results[i].GetResult(j)->VID) + "@" + std::to_string(results[i].GetResult(j)->Dist) + ",";
            LOG(Helper::LogLevel::LL_Info, "%s\n", ll.c_str());
        }
    }
    meanrecall /= NumQuerys;
    for (SizeType i = 0; i < NumQuerys; i++)
    {
        stdrecall += (thisrecall[i] - meanrecall) * (thisrecall[i] - meanrecall);
    }
    stdrecall = std::sqrt(stdrecall / NumQuerys);
    log << meanrecall << " " << stdrecall << " " << minrecall << " " << maxrecall << std::endl;
    return meanrecall;
}

void LoadTruth(std::ifstream& fp, std::vector<std::set<SizeType>>& truth, SizeType NumQuerys, int K)
{
    SizeType get;
    std::string line;
    for (SizeType i = 0; i < NumQuerys; ++i)
    {
        truth[i].clear();
        for (int j = 0; j < K; ++j)
        {
            fp >> get;
            truth[i].insert(get);
        }
        std::getline(fp, line);
    }
}

void LoadTruthBin(std::ifstream& fp, std::vector<std::set<SizeType>>& truth, SizeType NumQuerys, int K, int truthDim)
{
    std::unique_ptr<int[]> ptr(new int[truthDim]);
    for (SizeType i = 0; i < NumQuerys; ++i)
    {
        truth[i].clear();
        fp.read((char*)(ptr.get()), truthDim * sizeof(int));
        for (int j = 0; j < K; j++) truth[i].insert(ptr[j]);
    }
}

template <typename T>
int Process(std::shared_ptr<SearcherOptions> options, VectorIndex& index)
{
    std::ofstream log("Recall-result.out", std::ios::app);
    if (!log.is_open())
    {
        LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open logging file!\n");
        exit(-1);
    }

    auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
    if (ErrorCode::Success != vectorReader->LoadFile(options->m_queryFile))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
        exit(1);
    }
    auto queryVectors = vectorReader->GetVectorSet(0, options->m_debugQuery);
    auto queryMetas = vectorReader->GetMetadataSet();

    std::shared_ptr<Helper::ReaderOptions> dataOptions(new Helper::ReaderOptions(queryVectors->GetValueType(), queryVectors->Dimension(), options->m_dataFileType));
    auto dataReader = Helper::VectorSetReader::CreateInstance(dataOptions);
    std::shared_ptr<VectorSet> dataVectors;
    if (options->m_dataFile != "")
    {
        if (ErrorCode::Success != dataReader->LoadFile(options->m_dataFile))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read data file.\n");
            exit(1);
        }
        dataVectors = dataReader->GetVectorSet();
    }

    std::ifstream ftruth;
    int truthDim = 0;
    if (options->m_truthFile != "")
    {
        if (options->m_genTruth) {
            std::ofstream ftruthout;
            if (options->m_truthFile.find("bin") != std::string::npos) {
                ftruthout.open(options->m_truthFile, std::ofstream::binary | std::ofstream::out);
                int count = queryVectors->Count();
                ftruthout.write((char*)&count, sizeof(int));
                ftruthout.write((char*)&(options->m_truthK), sizeof(int));
            }
            else {
                ftruthout.open(options->m_truthFile, std::ofstream::out);
            }

            if (!ftruthout.is_open())
            {
                LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for write!\n", options->m_truthFile.c_str());
            }

            std::vector<COMMON::QueryResultSet<T>> results(options->m_batch, COMMON::QueryResultSet<T>(NULL, options->m_truthK));
            for (int startQuery = 0; startQuery < queryVectors->Count(); startQuery += options->m_batch) {
                int numQuerys = min(options->m_batch, queryVectors->Count() - startQuery);

#pragma omp parallel for
                for (int qid = 0; qid < numQuerys; qid++) {
                    results[qid].Reset();
                    results[qid].SetTarget((T*)queryVectors->GetVector(qid + startQuery));
                    for (SizeType y = 0; y < index.GetNumSamples(); y++)
                    {
                        float dist = index.ComputeDistance(results[qid].GetTarget(), index.GetSample(y));
                        results[qid].AddPoint(y, dist);
                    }
                    results[qid].SortResult();
                }

                for (int qid = 0; qid < numQuerys; qid++) {
                    if (options->m_truthFile.find("bin") != std::string::npos) {
                        for (int rid = 0; rid < options->m_truthK; rid++) {
                            ftruthout.write((char*)&(results[qid].GetResult(rid)->VID), sizeof(int));
                        }
                    }
                    else {
                        for (int rid = 0; rid < options->m_truthK; rid++) {
                            ftruthout << std::to_string(results[qid].GetResult(rid)->VID) << " ";
                        }
                        ftruthout << std::endl;
                    }
                }
            }
            ftruthout.close();
        }

        if (options->m_truthFile.find("bin") != std::string::npos) {
            ftruth.open(options->m_truthFile, std::ifstream::binary | std::ifstream::in);
            if (!ftruth.is_open())
            {
                LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for read!\n", options->m_truthFile.c_str());
            }
            int tq;
            ftruth.read((char*)(&tq), sizeof(int));
            ftruth.read((char*)(&truthDim), sizeof(int));
            LOG(Helper::LogLevel::LL_Info, "Load binary truth(%d, %d)...\n", tq, truthDim);
        }
        else {
            ftruth.open(options->m_truthFile);
            if (!ftruth.is_open())
            {
                LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for read!\n", options->m_truthFile.c_str());
            }
            LOG(Helper::LogLevel::LL_Info, "Load txt truth...\n");
        }
    }

    std::ofstream fp;
    if (options->m_resultFile != "")
    {
        fp.open(options->m_resultFile);
        if (!fp.is_open())
        {
            LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for write!\n", options->m_resultFile.c_str());
        }
    }

    std::vector<std::string> maxCheck = Helper::StrUtils::SplitString(options->m_maxCheck, "#");
    if (options->m_truthK < 0) options->m_truthK = options->m_K;

    std::vector<std::set<SizeType>> truth(options->m_batch);
    std::vector<QueryResult> results(options->m_batch, QueryResult(NULL, options->m_K, options->m_withMeta != 0));
    std::vector<float> latencies(options->m_batch, 0);
    int baseSquare = SPTAG::COMMON::Utils::GetBase<T>() * SPTAG::COMMON::Utils::GetBase<T>();

    LOG(Helper::LogLevel::LL_Info, "[query]\t\t[maxcheck]\t[avg] \t[99%] \t[95%] \t[recall] \t[qps] \t[mem]\n");
    std::vector<float> totalAvg(maxCheck.size(), 0.0), total99(maxCheck.size(), 0.0), total95(maxCheck.size(), 0.0), totalRecall(maxCheck.size(), 0.0), totalLatency(maxCheck.size(), 0.0);
    for (int startQuery = 0; startQuery < queryVectors->Count(); startQuery += options->m_batch)
    {
        int numQuerys = min(options->m_batch, queryVectors->Count() - startQuery);
        for (SizeType i = 0; i < numQuerys; i++) results[i].SetTarget(queryVectors->GetVector(startQuery + i));
        if (ftruth.is_open() && truthDim > 0) LoadTruthBin(ftruth, truth, numQuerys, options->m_truthK, truthDim); else LoadTruth(ftruth, truth, numQuerys, options->m_truthK);


        for (int mc = 0; mc < maxCheck.size(); mc++)
        {
            index.SetParameter("MaxCheck", maxCheck[mc].c_str());

#pragma omp parallel for
            for (SizeType i = 0; i < numQuerys; i++) results[i].Reset();

            auto batchstart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic)
            for (int qid = 0; qid < numQuerys; qid++)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                index.SearchIndex(results[qid]);
                auto t2 = std::chrono::high_resolution_clock::now();
                latencies[qid] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0;
            }
            auto batchend = std::chrono::high_resolution_clock::now();
            float batchLatency = std::chrono::duration_cast<std::chrono::microseconds>(batchend - batchstart).count() / 1000000.0;

            float timeMean = 0, timeMin = MaxDist, timeMax = 0, timeStd = 0;
            for (int qid = 0; qid < numQuerys; qid++)
            {
                timeMean += latencies[qid];
                if (latencies[qid] > timeMax) timeMax = latencies[qid];
                if (latencies[qid] < timeMin) timeMin = latencies[qid];
            }
            timeMean /= numQuerys;
            for (int qid = 0; qid < numQuerys; qid++) timeStd += ((float)latencies[qid] - timeMean) * ((float)latencies[qid] - timeMean);
            timeStd = std::sqrt(timeStd / numQuerys);
            log << timeMean << " " << timeStd << " " << timeMin << " " << timeMax << " ";

            std::sort(latencies.begin(), latencies.begin() + numQuerys);
            float l99 = latencies[SizeType(numQuerys * 0.99)];
            float l95 = latencies[SizeType(numQuerys * 0.95)];

            float recall = 0;
            if (ftruth.is_open())
            {
                recall = CalcRecall<T>(&index, results, truth, numQuerys, options->m_K, options->m_truthK, queryVectors, dataVectors, log, options->m_debugQuery > 0);
            }

#ifndef _MSC_VER
            struct rusage rusage;
            getrusage(RUSAGE_SELF, &rusage);
            unsigned long long peakWSS = rusage.ru_maxrss * 1024 / 1000000000;
#else
            PROCESS_MEMORY_COUNTERS pmc;
            GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
            unsigned long long peakWSS = pmc.PeakWorkingSetSize / 1000000000;
#endif
            LOG(Helper::LogLevel::LL_Info, "%d-%d\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t\t%.2f\t\t%lluGB\n", startQuery, (startQuery + numQuerys), maxCheck[mc].c_str(), timeMean, l99, l95, recall, (numQuerys / batchLatency), peakWSS);
            totalAvg[mc] += timeMean * numQuerys;
            total95[mc] += l95 * numQuerys;
            total99[mc] += l99 * numQuerys;
            totalRecall[mc] += recall * numQuerys;
            totalLatency[mc] += batchLatency;
        }

        if (fp.is_open())
        {
            fp << std::setprecision(3) << std::fixed;
            for (SizeType i = 0; i < numQuerys; i++)
            {
                if (queryMetas != nullptr) {
                    ByteArray qmeta = queryMetas->GetMetadata(startQuery + i);
                    fp.write((const char*)qmeta.Data(), qmeta.Length());
                }
                else {
                    fp << i;
                }
                fp << ":";
                for (int j = 0; j < options->m_K; j++)
                {
                    if (results[i].GetResult(j)->VID < 0) {
                        fp << results[i].GetResult(j)->Dist << "@NULL" << std::endl;
                        continue;
                    }

                    if (!options->m_withMeta) {
                        fp << (results[i].GetResult(j)->Dist / baseSquare) << "@" << results[i].GetResult(j)->VID << std::endl;
                    }
                    else {
                        ByteArray vm = index.GetMetadata(results[i].GetResult(j)->VID);
                        fp << (results[i].GetResult(j)->Dist / baseSquare) << "@";
                        fp.write((const char*)vm.Data(), vm.Length());
                    }
                    fp << "|";
                }
                fp << std::endl;
            }
        }
    }
    for (int mc = 0; mc < maxCheck.size(); mc++)
        LOG(Helper::LogLevel::LL_Info, "%d-%d\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t%.2f\n", 0, queryVectors->Count(), maxCheck[mc].c_str(), (totalAvg[mc] / queryVectors->Count()), (total99[mc] / queryVectors->Count()), (total95[mc] / queryVectors->Count()), (totalRecall[mc] / queryVectors->Count()), (queryVectors->Count() / totalLatency[mc]));

    LOG(Helper::LogLevel::LL_Info, "Output results finish!\n");

    ftruth.close();
    fp.close();
    log.close();
    return 0;
}

int main(int argc, char** argv)
{
    std::shared_ptr<SearcherOptions> options(new SearcherOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }

    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    auto ret = SPTAG::VectorIndex::LoadIndex(options->m_indexFolder, vecIndex);
    if (SPTAG::ErrorCode::Success != ret || nullptr == vecIndex)
    {
        LOG(Helper::LogLevel::LL_Error, "Cannot open index configure file!");
        return -1;
    }

    Helper::IniReader iniReader;
    for (int i = 1; i < argc; i++)
    {
        std::string param(argv[i]);
        size_t idx = param.find("=");
        if (idx == std::string::npos) continue;

        std::string paramName = param.substr(0, idx);
        std::string paramVal = param.substr(idx + 1);
        std::string sectionName;
        idx = paramName.find(".");
        if (idx != std::string::npos) {
            sectionName = paramName.substr(0, idx);
            paramName = paramName.substr(idx + 1);
        }
        iniReader.SetParameter(sectionName, paramName, paramVal);
        LOG(Helper::LogLevel::LL_Info, "Set [%s]%s = %s\n", sectionName.c_str(), paramName.c_str(), paramVal.c_str());
    }

    if (!iniReader.DoesParameterExist("Index", "NumberOfThreads"))
        iniReader.SetParameter("Index", "NumberOfThreads", std::to_string(options->m_threadNum));

    for (const auto& iter : iniReader.GetParameters("Index"))
    {
        vecIndex->SetParameter(iter.first.c_str(), iter.second.c_str());
    }

    vecIndex->UpdateIndex();

    if (!options->m_quantizerFile.empty())
    {
        auto ptr = SPTAG::f_createIO();
        if (!ptr->Initialize(options->m_quantizerFile.c_str(), std::ios::binary | std::ios::in))
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read quantizer file.\n");
            exit(1);
        }
        auto code = SPTAG::COMMON::Quantizer::LoadQuantizer(ptr, QuantizerType::PQQuantizer, options->m_reconstructType);
        if (code != ErrorCode::Success)
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to load quantizer.\n");
            exit(1);
        }
        COMMON::DistanceUtils::Quantizer->SetEnableADC(options->m_enableADC);
    }

    switch (vecIndex->GetVectorValueType())
    {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        Process<Type>(options, *(vecIndex.get())); \
        break; \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

    default: break;
    }
    return 0;
}