// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/ArgumentsParser.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Core/Common.h"
#include "inc/Core/MetadataSet.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SearchQuery.h"
#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/Common/DataUtils.h"
#include <iomanip>
#include <set>

using namespace SPTAG;

class SearcherOptions : public Helper::ArgumentsParser
{
public:
	SearcherOptions()
	{
		AddRequiredOption(m_queryFile, "-i", "--input", "Input raw data.");
		AddRequiredOption(m_indexFolder, "-x", "--index", "Index folder.");
		AddOptionalOption(m_truthFile, "-r", "--truth", "Truth file.");
		AddOptionalOption(m_resultFile, "-o", "--result", "Output result file.");
		AddOptionalOption(m_maxCheck, "-m", "--maxcheck", "MaxCheck for index.");
		AddOptionalOption(m_withMeta, "-a", "--withmeta", "Output metadata instead of vector id.");
		AddOptionalOption(m_K, "-k", "--KNN", "K nearest neighbors for search.");
		AddOptionalOption(m_batch, "-b", "--batchsize", "Batch query size.");
		AddOptionalOption(m_threadNum, "-t", "--thread", "Thread Number.");
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

	std::uint32_t m_threadNum = 0;
};

template <typename T>
struct QueryReader
{
public:
	std::vector<std::vector<T>> Query;
	std::vector<std::string> QStrings;
	int base = 1;

	QueryReader(std::string inputFile, int batchSize, int featureDim, DistCalcMethod distMethod) {
		Query.resize(batchSize, std::vector<T>(featureDim, 0));
		if (distMethod == DistCalcMethod::Cosine) base = COMMON::Utils::GetBase<T>();

		if (inputFile.find("BIN:") == 0) {
			m_isBinary = true;
			m_inStream.open(inputFile.substr(4), std::ifstream::binary);
			if (!m_inStream.is_open())
			{
				std::cerr << "ERROR: Cannot Load Query file " << inputFile << "!" << std::endl;
			}
			SizeType numQuerys;
			m_inStream.read((char*)&numQuerys, sizeof(SizeType));
			m_inStream.read((char*)&m_featureDim, sizeof(SizeType));
			if (m_featureDim != featureDim) {
				std::cerr << "ERROR: Feature dimension is not match between query file and index!" << std::endl;
			}
		}
		else {
			m_isBinary = false;
			m_inStream.open(inputFile);
			if (!m_inStream.is_open())
			{
				std::cerr << "ERROR: Cannot Load Query file " << inputFile << "!" << std::endl;
			}
			m_featureDim = featureDim;
		}
		m_distMethod = distMethod;
	}

	~QueryReader() {
		for (int i = 0; i < Query.size(); i++) Query[i].clear();
		Query.clear();
		QStrings.clear();
		m_inStream.close();
	}

	int ReadBatch() {
		int readQuerys = (int)Query.size();
		if (m_isBinary) {
			QStrings.resize(readQuerys, "");
			for (int i = 0; i < readQuerys; i++) {
				m_inStream.read((char*)Query[i].data(), sizeof(T)*m_featureDim);
				if (m_inStream.eof()) readQuerys = i;
			}
		}
		else {
			QStrings.clear();
			COMMON::Utils::PrepareQuerys(m_inStream, QStrings, Query, readQuerys, m_featureDim, m_distMethod, base);
		}
		return readQuerys;
	}
private:
	bool m_isBinary;
	std::ifstream m_inStream;
	int m_featureDim;
	DistCalcMethod m_distMethod;
};

template <typename T>
float CalcRecall(std::vector<QueryResult> &results, const std::vector<std::set<SizeType>> &truth, SizeType NumQuerys, int K, std::ofstream& log)
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
	QueryReader<T> reader(options->m_queryFile, options->m_batch, index.GetFeatureDim(), index.GetDistCalcMethod());
   
	std::ifstream ftruth;
	if (options->m_truthFile != "")
	{
		ftruth.open(options->m_truthFile);
		if (!ftruth.is_open())
		{
			std::cerr << "ERROR: Cannot open " << options->m_truthFile << " for read!" << std::endl;
		}
	}

    std::ofstream fp;
    if (options->m_resultFile != "")
    {
        fp.open(options->m_resultFile);
        if (!fp.is_open())
        {
            std::cerr << "ERROR: Cannot open " << options->m_resultFile << " for write!" << std::endl;
        }
    }

    std::ofstream log(index.GetIndexName() + "_" + std::to_string(options->m_K) + ".txt");
    if (!log.is_open())
    {
        std::cerr << "ERROR: Cannot open logging file!" << std::endl;
        return -1;
    }

    std::vector<std::string> maxCheck = Helper::StrUtils::SplitString(options->m_maxCheck, "#");

    std::vector<std::set<SizeType>> truth(options->m_batch);
    std::vector<QueryResult> results(options->m_batch, QueryResult(NULL, options->m_K, options->m_withMeta != 0));
    clock_t * latencies = new clock_t[options->m_batch + 1];
	
	std::cout << "[query]\t\t[maxcheck]\t[avg]      \t[99%] \t[95%] \t[recall] \t[mem]" << std::endl;
	int numQuerys;
	int totalQuerys = 0;
	std::vector<float> totalAvg(maxCheck.size(), 0.0), total99(maxCheck.size(), 0.0), total95(maxCheck.size(), 0.0), totalRecall(maxCheck.size(), 0.0);
    while ((numQuerys = reader.ReadBatch()) != 0)
    {
        for (SizeType i = 0; i < numQuerys; i++) results[i].SetTarget(reader.Query[i].data());
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

            std::sort(latencies, latencies + numQuerys, [](clock_t x, clock_t y)
            {
                return x < y;
            });
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
            std::cout << totalQuerys << "-" << (totalQuerys + numQuerys) << "\t" << maxCheck[mc] << "\t" << std::fixed << std::setprecision(6) << (timeMean / CLOCKS_PER_SEC) << "\t" << std::setprecision(4) << l99 << "\t" << l95 << "\t" << recall << "\t\t" << peakWSS << "GB" << std::endl;
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
                fp << reader.QStrings[i] << ":";
                for (int j = 0; j < options->m_K; j++)
                {
					if (results[i].GetResult(j)->VID < 0) {
						fp << results[i].GetResult(j)->Dist << "@NULL" << std::endl;
						continue;
					}

                    if (!options->m_withMeta) {
                        fp << (results[i].GetResult(j)->Dist / (reader.base * reader.base)) << "@" << results[i].GetResult(j)->VID << std::endl;
                    }
                    else {
                        ByteArray vm = index.GetMetadata(results[i].GetResult(j)->VID);
                        fp << (results[i].GetResult(j)->Dist / (reader.base * reader.base)) << "@";
                        fp.write((const char*)vm.Data(), vm.Length());
                    }
                    fp << "|";
                }
                fp << std::endl;
            }
        }
		totalQuerys += numQuerys;
    }
	for (int mc = 0; mc < maxCheck.size(); mc++)
		std::cout << 0 << "-" << totalQuerys << "\t" << maxCheck[mc] << "\t" << std::fixed << std::setprecision(6) << (totalAvg[mc]/totalQuerys) << "\t" << std::setprecision(4) << (total99[mc]/totalQuerys) << "\t" << (total95[mc]/totalQuerys) << "\t" << (totalRecall[mc]/totalQuerys) << std::endl;

    std::cout << "Output results finish!" << std::endl;

	ftruth.close();
    fp.close();
    log.close();

	for (int i = 0; i < truth.size(); i++) truth[i].clear();
	truth.clear();
	results.clear();
    delete[] latencies;
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
        std::cerr << "Cannot open index configure file!" << std::endl;
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
        std::cout << "Set [" << sectionName << "]" << paramName << " = " << paramVal << std::endl;
    }

	if (options->m_threadNum != 0)
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
