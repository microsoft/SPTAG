// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/PQQuantizer.h"
#include "inc/Core/VectorIndex.h"
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <set>

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
        AddOptionalOption(m_batch, "-b", "--batchsize", "Batch query size.");
    }

    ~SearcherOptions() {}

    std::string m_queryFile;

    std::string m_indexFolder;

    std::string m_truthFile = "";

    std::string m_resultFile = "";

    std::string m_maxCheck = "8192";

    int m_withMeta = 0;

    int m_K = 32;

    int m_batch = 10000;
};

template <typename T>
float CalcRecall(std::vector<QueryResult>& results, const std::vector<std::set<SizeType>>& truth, SizeType NumQuerys, int K, std::ofstream& log)
{
    float meanrecall = 0, minrecall = MaxDist, maxrecall = 0, stdrecall = 0;
    std::vector<float> thisrecall(NumQuerys, 0);
    for (SizeType i = 0; i < NumQuerys; i++)
    {
        for (SizeType id : truth[i])
        {
            for (int j = 0; j < K; j++)
            {
                if (results[i].GetResult(j)->VID == id)
                {
                    thisrecall[i] += 1;
                    break;
                }
            }
        }
        thisrecall[i] /= K;
        meanrecall += thisrecall[i];
        if (thisrecall[i] < minrecall) minrecall = thisrecall[i];
        if (thisrecall[i] > maxrecall) maxrecall = thisrecall[i];
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
    auto queryVectors = vectorReader->GetVectorSet();
    auto queryMetas = vectorReader->GetMetadataSet();

    std::ifstream ftruth;
    if (options->m_truthFile != "")
    {
        ftruth.open(options->m_truthFile);
        if (!ftruth.is_open())
        {
            LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot open %s for read!\n", options->m_truthFile.c_str());
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

    std::vector<std::set<SizeType>> truth(options->m_batch);
    std::vector<QueryResult> results(options->m_batch, QueryResult(NULL, options->m_K, options->m_withMeta != 0));
    std::vector<clock_t> latencies(options->m_batch + 1, 0);
    int baseSquare = (GetEnumValueType<T>() == VectorValueType::UInt8 && SPTAG::COMMON::DistanceUtils::PQQuantizer != nullptr) ? 1 : SPTAG::COMMON::Utils::GetBase<T>() * SPTAG::COMMON::Utils::GetBase<T>();
    LOG(Helper::LogLevel::LL_Info, "[query]\t\t[maxcheck]\t[avg] \t[99%] \t[95%] \t[recall] \t[mem]\n");
    std::vector<float> totalAvg(maxCheck.size(), 0.0), total99(maxCheck.size(), 0.0), total95(maxCheck.size(), 0.0), totalRecall(maxCheck.size(), 0.0);
    for (int startQuery = 0; startQuery < queryVectors->Count(); startQuery += options->m_batch)
    {
        int numQuerys = min(options->m_batch, queryVectors->Count() - startQuery);
        for (SizeType i = 0; i < numQuerys; i++) {
            void* vec = queryVectors->GetVector(startQuery + i);
            if (SPTAG::COMMON::DistanceUtils::PQQuantizer != nullptr) {
                SPTAG::COMMON::Utils::Normalize<float>((float*) vec, options->m_dimension, 1);
                vec = (void*)SPTAG::COMMON::DistanceUtils::PQQuantizer->QuantizeVector((const float*) vec);
            }
            results[i].SetTarget(vec);
        }
        if (ftruth.is_open()) LoadTruth(ftruth, truth, numQuerys, options->m_K);

        SizeType subSize = (numQuerys - 1) / omp_get_num_threads() + 1;
        for (int mc = 0; mc < maxCheck.size(); mc++)
        {
            index.SetParameter("MaxCheck", maxCheck[mc].c_str());
            for (SizeType i = 0; i < numQuerys; i++) results[i].Reset();

#pragma omp parallel for
            for (int tid = 0; tid < omp_get_num_threads(); tid++)
            {
                SizeType start = tid * subSize;
                SizeType end = min((tid + 1) * subSize, numQuerys);
                for (SizeType i = start; i < end; i++)
                {
                    latencies[i] = clock();
                    index.SearchIndex(results[i]);
                }
            }
            latencies[numQuerys] = clock();

            float timeMean = 0, timeMin = MaxDist, timeMax = 0, timeStd = 0;
            for (SizeType i = 0; i < numQuerys; i++)
            {
                if (latencies[i + 1] >= latencies[i])
                    latencies[i] = latencies[i + 1] - latencies[i];
                else
                    latencies[i] = latencies[numQuerys] - latencies[i];
                timeMean += latencies[i];
                if (latencies[i] > timeMax) timeMax = (float)latencies[i];
                if (latencies[i] < timeMin) timeMin = (float)latencies[i];
            }
            timeMean /= numQuerys;
            for (SizeType i = 0; i < numQuerys; i++) timeStd += ((float)latencies[i] - timeMean) * ((float)latencies[i] - timeMean);
            timeStd = std::sqrt(timeStd / numQuerys);
            log << timeMean << " " << timeStd << " " << timeMin << " " << timeMax << " ";

            std::sort(latencies.begin(), latencies.begin() + numQuerys);
            float l99 = float(latencies[SizeType(numQuerys * 0.99)]) / CLOCKS_PER_SEC;
            float l95 = float(latencies[SizeType(numQuerys * 0.95)]) / CLOCKS_PER_SEC;

            float recall = 0;
            if (ftruth.is_open())
            {
                recall = CalcRecall<T>(results, truth, numQuerys, options->m_K, log);
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
            LOG(Helper::LogLevel::LL_Info, "%d-%d\t%s\t%.4f\t%.4f\t%.4f\t%.2f\t\t%lluGB\n", startQuery, (startQuery + numQuerys), maxCheck[mc].c_str(), (timeMean / CLOCKS_PER_SEC), l99, l95, recall, peakWSS);
            totalAvg[mc] += timeMean / CLOCKS_PER_SEC * numQuerys;
            total95[mc] += l95 * numQuerys;
            total99[mc] += l99 * numQuerys;
            totalRecall[mc] += recall * numQuerys;
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
        LOG(Helper::LogLevel::LL_Info, "%d-%d\t%s\t%.4f\t%.4f\t%.4f\t%.2f\n", 0, queryVectors->Count(), maxCheck[mc].c_str(), (totalAvg[mc] / queryVectors->Count()), (total99[mc] / queryVectors->Count()), (total95[mc] / queryVectors->Count()), (totalRecall[mc] / queryVectors->Count()));

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