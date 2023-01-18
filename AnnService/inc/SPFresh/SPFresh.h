// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/VectorSetReader.h"
#include <future>

#include <iomanip>
#include <iostream>
#include <fstream>

using namespace SPTAG;

namespace SPTAG {
	namespace SSDServing {
        namespace SPFresh {

            typedef std::chrono::steady_clock SteadClock;

            double getMsInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                return (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1.0) / 1000.0;
            }

            double getSecInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                return (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1.0) / 1000.0;
            }

            double getMinInterval(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
                return (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() * 1.0) / 60.0;
            }

            /// Clock class
            class StopWSPFresh {
            private:
                std::chrono::steady_clock::time_point time_begin;
            public:
                StopWSPFresh() {
                    time_begin = std::chrono::steady_clock::now();
                }

                double getElapsedMs() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getMsInterval(time_begin, time_end);
                }

                double getElapsedSec() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getSecInterval(time_begin, time_end);
                }
                    
                double getElapsedMin() {
                    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
                    return getMinInterval(time_begin, time_end);
                }

                void reset() {
                    time_begin = std::chrono::steady_clock::now();
                }
            };

            template <typename ValueType>
            void OutputResult(const std::string& p_output, std::vector<QueryResult>& p_results, int p_resultNum)
            {
                if (!p_output.empty())
                {
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_output.c_str(), std::ios::binary | std::ios::out)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed create file: %s\n", p_output.c_str());
                        exit(1);
                    }
                    int32_t i32Val = static_cast<int32_t>(p_results.size());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        exit(1);
                    }
                    i32Val = p_resultNum;
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        exit(1);
                    }

                    float fVal = 0;
                    for (size_t i = 0; i < p_results.size(); ++i)
                    {
                        for (int j = 0; j < p_resultNum; ++j)
                        {
                            i32Val = p_results[i].GetResult(j)->VID;
                            if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                exit(1);
                            }

                            fVal = p_results[i].GetResult(j)->Dist;
                            if (ptr->WriteBinary(sizeof(fVal), reinterpret_cast<char*>(&fVal)) != sizeof(fVal)) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                exit(1);
                            }
                        }
                    }
                }
            }

            void ShowMemoryStatus(std::shared_ptr<SPTAG::VectorSet> vectorSet, double second)
            {
                int tSize = 0, resident = 0, share = 0;
                std::ifstream buffer("/proc/self/statm");
                buffer >> tSize >> resident >> share;
                buffer.close();
#ifndef _MSC_VER
                long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
#else
                SYSTEM_INFO sysInfo;
                GetSystemInfo(&sysInfo);
                long page_size_kb = sysInfo.dwPageSize / 1024;
#endif
                long rss = resident * page_size_kb;
                long vector_size;
                if (vectorSet != nullptr)
                    vector_size = vectorSet->PerVectorDataSize() * (vectorSet->Count() / 1024);
                else 
                    vector_size = 0;
                long vector_size_mb = vector_size / 1024;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Current time: %.0lf. RSS : %ld MB, Vector Set Size : %ld MB, True Size: %ld MB\n", second, rss / 1024, vector_size_mb, rss / 1024 - vector_size_mb);
            }

            template<typename T, typename V>
            void PrintPercentiles(const std::vector<V>& p_values, std::function<T(const V&)> p_get, const char* p_format, bool reverse=false)
            {
                double sum = 0;
                std::vector<T> collects;
                collects.reserve(p_values.size());
                for (const auto& v : p_values)
                {
                    T tmp = p_get(v);
                    sum += tmp;
                    collects.push_back(tmp);
                }
                if (reverse) {
                    std::sort(collects.begin(), collects.end(), std::greater<T>());
                }
                else {
                    std::sort(collects.begin(), collects.end());
                }
                if (reverse) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg\t50tiles\t90tiles\t95tiles\t99tiles\t99.9tiles\tMin\n");
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg\t50tiles\t90tiles\t95tiles\t99tiles\t99.9tiles\tMax\n");
                }

                std::string formatStr("%.3lf");
                for (int i = 1; i < 7; ++i)
                {
                    formatStr += '\t';
                    formatStr += p_format;
                }

                formatStr += '\n';

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    formatStr.c_str(),
                    sum / collects.size(),
                    collects[static_cast<size_t>(collects.size() * 0.50)],
                    collects[static_cast<size_t>(collects.size() * 0.90)],
                    collects[static_cast<size_t>(collects.size() * 0.95)],
                    collects[static_cast<size_t>(collects.size() * 0.99)],
                    collects[static_cast<size_t>(collects.size() * 0.999)],
                    collects[static_cast<size_t>(collects.size() - 1)]);
            }

            template <typename T>
            static float CalculateRecallSPFresh(VectorIndex* index, std::vector<QueryResult>& results, const std::vector<std::set<SizeType>>& truth, int K, int truthK, std::shared_ptr<SPTAG::VectorSet> querySet, std::shared_ptr<SPTAG::VectorSet> vectorSet, SizeType NumQuerys, std::ofstream* log = nullptr, bool debug = false)
            {
                float meanrecall = 0, minrecall = MaxDist, maxrecall = 0, stdrecall = 0;
                std::vector<float> thisrecall(NumQuerys, 0);
                std::unique_ptr<bool[]> visited(new bool[K]);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start Calculating Recall\n");
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
                            } else if (vectorSet != nullptr) {
                                float dist = results[i].GetResult(j)->Dist;
                                float truthDist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(id), vectorSet->Dimension(), index->GetDistCalcMethod());
                                if (index->GetDistCalcMethod() == SPTAG::DistCalcMethod::Cosine && fabs(dist - truthDist) < Epsilon) {
                                    thisrecall[i] += 1;
                                    visited[j] = true;
                                    break;
                                }
                                else if (index->GetDistCalcMethod() == SPTAG::DistCalcMethod::L2 && fabs(dist - truthDist) < Epsilon * (dist + Epsilon)) {
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
                                truthDist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(id), querySet->Dimension(), index->GetDistCalcMethod());
                            }
                            truthvec.emplace_back(id, truthDist);
                        }
                        std::sort(truthvec.begin(), truthvec.end());
                        for (int j = 0; j < truthvec.size(); j++)
                            ll += std::to_string(truthvec[j].node) + "@" + std::to_string(truthvec[j].distance) + ",";
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%s\n", ll.c_str());
                        ll = "ann:";
                        for (int j = 0; j < K; j++)
                            ll += std::to_string(results[i].GetResult(j)->VID) + "@" + std::to_string(results[i].GetResult(j)->Dist) + ",";
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%s\n", ll.c_str());
                    }
                }
                meanrecall /= NumQuerys;
                for (SizeType i = 0; i < NumQuerys; i++)
                {
                    stdrecall += (thisrecall[i] - meanrecall) * (thisrecall[i] - meanrecall);
                }
                stdrecall = std::sqrt(stdrecall / NumQuerys);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "stdrecall: %.6lf, maxrecall: %.2lf, minrecall: %.2lf\n", stdrecall, maxrecall, minrecall);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nRecall Distribution:\n");
                PrintPercentiles<float, float>(thisrecall,
                    [](const float recall) -> float
                    {
                        return recall;
                    },
                    "%.3lf", true);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recall%d@%d: %f\n", K, truthK, meanrecall);
                
                if (log) (*log) << meanrecall << " " << stdrecall << " " << minrecall << " " << maxrecall << std::endl;
                return meanrecall;
            }

            template <typename ValueType>
            double SearchSequential(SPANN::Index<ValueType>* p_index,
                int p_numThreads,
                std::vector<QueryResult>& p_results,
                std::vector<SPANN::SearchStats>& p_stats,
                int p_maxQueryCount, int p_internalResultNum)
            {
                int numQueries = min(static_cast<int>(p_results.size()), p_maxQueryCount);

                std::atomic_size_t queriesSent(0);

                std::vector<std::thread> threads;

                StopWSPFresh sw;

                auto func = [&]()
                {
                    p_index->Initialize();
                    StopWSPFresh threadws;
                    size_t index = 0;
                    while (true)
                    {
                        index = queriesSent.fetch_add(1);
                        if (index < numQueries)
                        {
                            double startTime = threadws.getElapsedMs();
                            p_index->GetMemoryIndex()->SearchIndex(p_results[index]);
                            double endTime = threadws.getElapsedMs();

                            p_stats[index].m_totalLatency = endTime - startTime;

                            p_index->SearchDiskIndex(p_results[index], &(p_stats[index]));
                            double exEndTime = threadws.getElapsedMs();

                            p_stats[index].m_exLatency = exEndTime - endTime;
                            p_stats[index].m_totalLatency = p_stats[index].m_totalSearchLatency = exEndTime - startTime;
                        }
                        else
                        {
                            p_index->ExitBlockController();
                            return;
                        }
                    }
                };
                for (int i = 0; i < p_numThreads; i++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }

                auto sendingCost = sw.getElapsedSec();

                return numQueries / sendingCost;
            }

            template <typename ValueType>
            void PrintStats(std::vector<SPANN::SearchStats>& stats)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nEx Elements Count:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_totalListElementsCount;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nHead Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_totalSearchLatency - ss.m_exLatency;
                    },
                    "%.3lf");
                
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nSetup Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_exSetUpLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nComp Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_compLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nSPDK Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_diskReadLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nEx Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_exLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_totalSearchLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Disk Page Access Distribution(KB):\n");
                PrintPercentiles<int, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> int
                    {
                        return ss.m_diskAccessCount;
                    },
                    "%4d");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Disk IO Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> int
                    {
                        return ss.m_diskIOCount;
                    },
                    "%4d");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");
            }

            void ResetStats(std::vector<SPANN::SearchStats>& totalStats) 
            {
                for (int i = 0; i < totalStats.size(); i++)
                {
                    totalStats[i].m_totalListElementsCount = 0;
                    totalStats[i].m_exLatency = 0;
                    totalStats[i].m_totalSearchLatency = 0;
                    totalStats[i].m_diskAccessCount = 0;
                    totalStats[i].m_diskIOCount = 0;
                    totalStats[i].m_compLatency = 0;
                    totalStats[i].m_diskReadLatency = 0;
                    totalStats[i].m_exSetUpLatency = 0;
                }
            }

            void AddStats(std::vector<SPANN::SearchStats>& totalStats, std::vector<SPANN::SearchStats>& addedStats)
            {
                for (int i = 0; i < totalStats.size(); i++)
                {
                    totalStats[i].m_totalListElementsCount += addedStats[i].m_totalListElementsCount;
                    totalStats[i].m_exLatency += addedStats[i].m_exLatency;
                    totalStats[i].m_totalSearchLatency += addedStats[i].m_totalSearchLatency;
                    totalStats[i].m_diskAccessCount += addedStats[i].m_diskAccessCount;
                    totalStats[i].m_diskIOCount += addedStats[i].m_diskIOCount;
                    totalStats[i].m_compLatency += addedStats[i].m_compLatency;
                    totalStats[i].m_diskReadLatency += addedStats[i].m_diskReadLatency;
                    totalStats[i].m_exSetUpLatency += addedStats[i].m_exSetUpLatency;
                }
            }

            void AvgStats(std::vector<SPANN::SearchStats>& totalStats, int avgStatsNum)
            {
                for (int i = 0; i < totalStats.size(); i++)
                {
                    totalStats[i].m_totalListElementsCount /= avgStatsNum;
                    totalStats[i].m_exLatency /= avgStatsNum;
                    totalStats[i].m_totalSearchLatency /= avgStatsNum;
                    totalStats[i].m_diskAccessCount /= avgStatsNum;
                    totalStats[i].m_diskIOCount /= avgStatsNum;
                    totalStats[i].m_compLatency /= avgStatsNum;
                    totalStats[i].m_diskReadLatency /= avgStatsNum;
                    totalStats[i].m_exSetUpLatency /= avgStatsNum;
                }
            }

            std::string convertFloatToString(const float value, const int precision = 0)
            {
                std::stringstream stream{};
                stream<<std::fixed<<std::setprecision(precision)<<value;
                return stream.str();
            }

            std::string GetTruthFileName(std::string& truthFilePrefix, int vectorCount)
            {
                std::string fileName(truthFilePrefix);
                fileName += "-";
                if (vectorCount < 1000)
                {
                    fileName += std::to_string(vectorCount);
                } 
                else if (vectorCount < 1000000)
                {
                    fileName += std::to_string(vectorCount/1000);
                    fileName += "k";
                }
                else if (vectorCount < 1000000000)
                {
                    if (vectorCount % 1000000 == 0) {
                        fileName += std::to_string(vectorCount/1000000);
                        fileName += "M";
                    } 
                    else 
                    {
                        float vectorCountM = ((float)vectorCount)/1000000;
                        fileName += convertFloatToString(vectorCountM, 2);
                        fileName += "M";
                    }
                }
                else
                {
                    fileName += std::to_string(vectorCount/1000000000);
                    fileName += "B";
                }
                return fileName;
            }

            std::shared_ptr<VectorSet>  LoadVectorSet(SPANN::Options& p_opts, int numThreads)
            {
                std::shared_ptr<VectorSet> vectorSet;
                if (p_opts.m_loadAllVectors) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading VectorSet...\n");
                    if (!p_opts.m_fullVectorPath.empty() && fileexists(p_opts.m_fullVectorPath.c_str())) {
                        std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                        auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                        if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_fullVectorPath))
                        {
                            vectorSet = vectorReader->GetVectorSet();
                            if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(numThreads);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nLoad VectorSet(%d,%d).\n", vectorSet->Count(), vectorSet->Dimension());
                        }
                    }
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Reduce memory usage\n");
                }
                return vectorSet;
            }

            std::shared_ptr<VectorSet>  LoadUpdateVectors(SPANN::Options& p_opts, std::vector<SizeType>& insertSet, SizeType updateSize)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load Update Vectors\n");
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_opts.m_fullVectorPath.c_str(), std::ios::binary | std::ios::in)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s.\n", p_opts.m_fullVectorPath.c_str());
                    throw std::runtime_error("Failed read file");
                }

                SizeType row;
                DimensionType col;
                if (ptr->ReadBinary(sizeof(SizeType), (char*)&row) != sizeof(SizeType)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
                    throw std::runtime_error("Failed read file");
                }
                if (ptr->ReadBinary(sizeof(DimensionType), (char*)&col) != sizeof(DimensionType)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
                    throw std::runtime_error("Failed read file");
                }

                std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(p_opts.m_valueType)) * updateSize * col;
                ByteArray vectorSet;
                if (totalRecordVectorBytes > 0) {
                    vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
                    char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());
                    std::uint64_t readSize = ((std::uint64_t)GetValueTypeSize(p_opts.m_valueType)) * col;
                    for (int i = 0; i < updateSize; i++) {
                        std::uint64_t offset = ((std::uint64_t)GetValueTypeSize(p_opts.m_valueType)) * insertSet[i] * col + sizeof(SizeType) + sizeof(DimensionType);
                        if (ptr->ReadBinary(readSize, vecBuf + i*readSize, offset) != readSize) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read VectorSet!\n");
                            throw std::runtime_error("Failed read file");
                        }
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load Vector(%d,%d)\n",updateSize, col);
                return std::make_shared<BasicVectorSet>(vectorSet,
                                                        p_opts.m_valueType,
                                                        col,
                                                        updateSize);

            }

            std::shared_ptr<VectorSet> LoadQuerySet(SPANN::Options& p_opts)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading QuerySet...\n");
                std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_queryType, p_opts.m_queryDelimiter));
                auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_queryPath))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                    exit(1);
                }
                return queryReader->GetVectorSet();
            }

            void LoadTruth(SPANN::Options& p_opts, std::vector<std::set<SizeType>>& truth, int numQueries, std::string truthfilename, int truthK)
            {
                auto ptr = f_createIO();
                if (p_opts.m_update) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...: %s\n", truthfilename.c_str());
                    
                    if (ptr == nullptr || !ptr->Initialize(truthfilename.c_str(), std::ios::in | std::ios::binary)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthfilename.c_str());
                        exit(1);
                    }
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...: %s\n", p_opts.m_truthPath.c_str());
                    
                    if (ptr == nullptr || !ptr->Initialize(p_opts.m_truthPath.c_str(), std::ios::in | std::ios::binary)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthfilename.c_str());
                        exit(1);
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "K: %d, TruthResultNum: %d\n", p_opts.m_resultNum, p_opts.m_truthResultNum);    
                COMMON::TruthSet::LoadTruth(ptr, truth, numQueries, p_opts.m_truthResultNum, p_opts.m_resultNum, p_opts.m_truthType);
                char tmp[4];
                if (ptr->ReadBinary(4, tmp) == 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Truth number is larger than query number(%d)!\n", numQueries);
                }
            }

            template <typename ValueType>
            void StableSearch(SPANN::Index<ValueType>* p_index,
                int numThreads,
                std::shared_ptr<SPTAG::VectorSet> querySet,
                std::shared_ptr<SPTAG::VectorSet> vectorSet,
                int avgStatsNum,
                int queryCountLimit,
                int internalResultNum,
                std::string& truthFileName,
                SPANN::Options& p_opts,
                double second = 0,
                bool showStatus = true)
            {
                if (avgStatsNum == 0) return;
                int numQueries = querySet->Count();

                std::vector<QueryResult> results(numQueries, QueryResult(NULL, internalResultNum, false));

                if (showStatus) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching: numThread: %d, numQueries: %d, searchTimes: %d.\n", numThreads, numQueries, avgStatsNum);
                std::vector<SPANN::SearchStats> stats(numQueries);
                std::vector<SPANN::SearchStats> TotalStats(numQueries);
                ResetStats(TotalStats);
                double totalQPS = 0;
                for (int i = 0; i < avgStatsNum; i++)
                {
                    for (int j = 0; j < numQueries; ++j)
                    {
                        results[j].SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(j)));
                        results[j].Reset();
                    }
                    totalQPS += SearchSequential(p_index, numThreads, results, stats, queryCountLimit, internalResultNum);
                    //PrintStats<ValueType>(stats);
                    AddStats(TotalStats, stats);
                }
                if (showStatus) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current time: %.0lf, Searching Times: %d, AvgQPS: %.2lf.\n", second, avgStatsNum, totalQPS/avgStatsNum);

                AvgStats(TotalStats, avgStatsNum);

                if (showStatus) PrintStats<ValueType>(TotalStats);

                if (p_opts.m_calTruth)
                {
                    if (p_opts.m_searchResult.empty()) {
                        std::vector<std::set<SizeType>> truth;
                        int K = p_opts.m_resultNum;
                        int truthK = p_opts.m_resultNum;
                        // float MRR, recall;
                        LoadTruth(p_opts, truth, numQueries, truthFileName, truthK);
                        CalculateRecallSPFresh<ValueType>((p_index->GetMemoryIndex()).get(), results, truth, K, truthK, querySet, vectorSet, numQueries);
                        // recall = COMMON::TruthSet::CalculateRecall<ValueType>((p_index->GetMemoryIndex()).get(), results, truth, K, truthK, querySet, vectorSet, numQueries, nullptr, false, &MRR);
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
                    } else {
                        OutputResult<ValueType>(p_opts.m_searchResult + std::to_string(second), results, p_opts.m_resultNum);
                    }
                }
            }

            void LoadUpdateMapping(std::string fileName, std::vector<SizeType>& reverseIndices)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading %s\n", fileName.c_str());

                int vectorNum;

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(fileName.c_str(), std::ios::in | std::ios::binary)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open trace file: %s\n", fileName.c_str());
                    exit(1);
                }
                
                if (ptr->ReadBinary(4, (char *)&vectorNum) != 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "vector Size Error!\n");
                }

                reverseIndices.clear();
                reverseIndices.resize(vectorNum);

                if (ptr->ReadBinary(vectorNum * 4, (char*)reverseIndices.data()) != vectorNum * 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "update mapping Error!\n");
                    exit(1);
                }
            }

            void LoadUpdateTrace(std::string fileName, SizeType& updateSize, std::vector<SizeType>& insertSet, std::vector<SizeType>& deleteSet)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading %s\n", fileName.c_str());

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(fileName.c_str(), std::ios::in | std::ios::binary)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open trace file: %s\n", fileName.c_str());
                    exit(1);
                }

                int tempSize;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading Size\n");
                
                if (ptr->ReadBinary(4, (char *)&tempSize) != 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Update Size Error!\n");
                }

                updateSize = tempSize;

                deleteSet.clear();
                deleteSet.resize(updateSize);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading deleteSet\n");

                if (ptr->ReadBinary(updateSize * 4, (char*)deleteSet.data()) != updateSize * 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Delete Set Error!\n");
                    exit(1);
                }

                insertSet.clear();
                insertSet.resize(updateSize);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading insertSet\n");

                if (ptr->ReadBinary(updateSize * 4, (char*)insertSet.data()) != updateSize * 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Insert Set Error!\n");
                    exit(1);
                }
            }

            void LoadUpdateTraceStressTest(std::string fileName, SizeType& updateSize, std::vector<SizeType>& insertSet)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading %s\n", fileName.c_str());

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(fileName.c_str(), std::ios::in | std::ios::binary)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open trace file: %s\n", fileName.c_str());
                    exit(1);
                }

                int tempSize;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading Size\n");
                
                if (ptr->ReadBinary(4, (char *)&tempSize) != 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Update Size Error!\n");
                }

                updateSize = tempSize;

                insertSet.clear();
                insertSet.resize(updateSize);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading insertSet\n");

                if (ptr->ReadBinary(updateSize * 4, (char*)insertSet.data()) != updateSize * 4) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Insert Set Error!\n");
                    exit(1);
                }
            }

            template <typename ValueType>
            void InsertVectorsBySet(SPANN::Index<ValueType>* p_index, 
                int insertThreads, 
                std::shared_ptr<SPTAG::VectorSet> vectorSet, 
                std::vector<SizeType>& insertSet,
                std::vector<SizeType>& mapping,
                int updateSize,
                SPANN::Options& p_opts)
            {
                StopWSPFresh sw;
                std::vector<std::thread> threads;
                std::vector<double> latency_vector(updateSize);

                std::atomic_size_t vectorsSent(0);

                auto func = [&]()
                {
                    p_index->Initialize();
                    size_t index = 0;
                    while (true)
                    {
                        index = vectorsSent.fetch_add(1);
                        if (index < updateSize)
                        {
                            if ((index & ((1 << 14) - 1)) == 0 && p_opts.m_showUpdateProgress)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Insert: Sent %.2lf%%...\n", index * 100.0 / updateSize);
                            }
                            // std::vector<char> meta;
                            // std::vector<std::uint64_t> metaoffset;
                            // std::string a = std::to_string(insertSet[index]);
                            // metaoffset.push_back((std::uint64_t)meta.size());
                            // for (size_t j = 0; j < a.length(); j++)
                            //     meta.push_back(a[j]);
                            // metaoffset.push_back((std::uint64_t)meta.size());
                            // std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
                            //     SPTAG::ByteArray((std::uint8_t*)meta.data(), meta.size() * sizeof(char), false),
                            //     SPTAG::ByteArray((std::uint8_t*)metaoffset.data(), metaoffset.size() * sizeof(std::uint64_t), false),
                            //     1));
                            if (p_opts.m_stressTest) p_index->DeleteIndex(mapping[insertSet[index]]);
                            auto insertBegin = std::chrono::high_resolution_clock::now();
                            if (p_opts.m_loadAllVectors)
                                p_index->AddIndexSPFresh(vectorSet->GetVector(insertSet[index]), 1, p_opts.m_dim, &mapping[insertSet[index]]);
                            else
                                p_index->AddIndexSPFresh(vectorSet->GetVector(index), 1, p_opts.m_dim, &mapping[insertSet[index]]);
                            auto insertEnd = std::chrono::high_resolution_clock::now();
                            latency_vector[index] = std::chrono::duration_cast<std::chrono::microseconds>(insertEnd - insertBegin).count();
                        }
                        else
                        {
                            p_index->ExitBlockController();
                            return;
                        }
                    }
                };
                for (int j = 0; j < insertThreads; j++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }

                double sendingCost = sw.getElapsedSec();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Insert: Finish sending in %.3lf seconds, sending throughput is %.2lf , insertion count %u.\n",
                sendingCost,
                updateSize / sendingCost,
                static_cast<uint32_t>(updateSize));

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Insert: During Update\n");

                while(!p_index->AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                double syncingCost = sw.getElapsedSec();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Insert: Finish syncing in %.3lf seconds, actuall throughput is %.2lf, insertion count %u.\n",
                syncingCost,
                updateSize / syncingCost,
                static_cast<uint32_t>(updateSize));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Insert Latency Distribution:\n");
                PrintPercentiles<double, double>(latency_vector,
                    [](const double& ss) -> double
                    {
                        return ss;
                    },
                    "%.3lf");
            }

            template <typename ValueType>
            void DeleteVectorsBySet(SPANN::Index<ValueType>* p_index, 
                int deleteThreads, 
                std::shared_ptr<SPTAG::VectorSet> vectorSet,
                std::vector<SizeType>& deleteSet,
                std::vector<SizeType>& mapping,
                int updateSize,
                SPANN::Options& p_opts,
                int batch)
            {
                int avgQPS = p_opts.m_deleteQPS / deleteThreads;
                if (p_opts.m_deleteQPS == -1) avgQPS = -1;
                std::vector<double> latency_vector(updateSize);
                std::vector<std::thread> threads;
                StopWSPFresh sw;
                std::atomic_size_t vectorsSent(0);
                auto func = [&]()
                {
                    int deleteCount = 0;
                    while (true)
                    {
                        deleteCount++;
                        size_t index = 0;
                        index = vectorsSent.fetch_add(1);
                        if (index < updateSize)
                        {
                            if ((index & ((1 << 14) - 1)) == 0 && p_opts.m_showUpdateProgress)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Delete: Sent %.2lf%%...\n", index * 100.0 / updateSize);
                            }
                            auto deleteBegin = std::chrono::high_resolution_clock::now();
                            p_index->DeleteIndex(mapping[deleteSet[index]]);
                            // p_index->DeleteIndex(vectorSet->GetVector(deleteSet[index]), deleteSet[index]);
                            // std::vector<char> meta;
                            // std::string a = std::to_string(deleteSet[index]);
                            // for (size_t j = 0; j < a.length(); j++)
                            //     meta.push_back(a[j]);
                            // ByteArray metarr = SPTAG::ByteArray((std::uint8_t*)meta.data(), meta.size() * sizeof(char), false);

                            // if (p_index->VectorIndex::DeleteIndex(metarr) == ErrorCode::VectorNotFound) {
                            //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"VID meta no found: %d\n", deleteSet[index]);
                            //     exit(1);
                            // }

                            auto deleteEnd = std::chrono::high_resolution_clock::now();
                            latency_vector[index] = std::chrono::duration_cast<std::chrono::microseconds>(deleteEnd - deleteBegin).count();
                            if (avgQPS != -1 && deleteCount == avgQPS) {
                                std::this_thread::sleep_for(std::chrono::seconds(1));
                                deleteCount = 0;
                            }
                        }
                        else
                        {
                            return;
                        }
                    }
                };
                for (int j = 0; j < deleteThreads; j++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }
                double sendingCost = sw.getElapsedSec();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Delete: Finish sending in %.3lf seconds, sending throughput is %.2lf , deletion count %u.\n",
                sendingCost,
                updateSize / sendingCost,
                static_cast<uint32_t>(updateSize));
            }
            
            template <typename ValueType>
            void SteadyStateSPFresh(SPANN::Index<ValueType>* p_index)
            {
                SPANN::Options& p_opts = *(p_index->GetOptions());
                int days = p_opts.m_days;
                if (days == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Need to input update days\n");
                    exit(1);
                }
                StopWSPFresh sw;

                int numThreads = p_opts.m_searchThreadNum;
                int internalResultNum = p_opts.m_searchInternalResultNum;
                int searchTimes = p_opts.m_searchTimes;

                auto vectorSet = LoadVectorSet(p_opts, numThreads);

                auto querySet = LoadQuerySet(p_opts);

                int curCount = p_index->GetNumSamples();

                bool calTruthOrigin = p_opts.m_calTruth;

                p_index->ForceCompaction();

                p_index->GetDBStat();

                if (!p_opts.m_onlySearchFinalBatch) {
                    if (p_opts.m_maxInternalResultNum != -1) 
                    {
                        for (int iterInternalResultNum = p_opts.m_minInternalResultNum; iterInternalResultNum <= p_opts.m_maxInternalResultNum; iterInternalResultNum += p_opts.m_stepInternalResultNum) 
                        {
                            StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, iterInternalResultNum, p_opts.m_truthPath, p_opts, sw.getElapsedSec());
                        }
                    }
                    else 
                    {
                        StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, internalResultNum, p_opts.m_truthPath, p_opts, sw.getElapsedSec());
                    }
                }
                // exit(1);

                ShowMemoryStatus(vectorSet, sw.getElapsedSec());
                p_index->GetDBStat();

                int insertThreads = p_opts.m_insertThreadNum;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Updating: numThread: %d, total days: %d.\n", insertThreads, days);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start updating...\n");

                int updateSize;
                std::vector<SizeType> insertSet;
                std::vector<SizeType> deleteSet;
                std::vector<SizeType> mapping;
                if (p_opts.m_endVectorNum == -1) p_opts.m_endVectorNum = curCount;
                mapping.resize(p_opts.m_endVectorNum);
                for (int i = 0; i < p_opts.m_endVectorNum; i++) {
                    mapping[i] = i;
                }
                
                for (int i = 0; i < days; i++)
                {   

                    std::string traceFileName = p_opts.m_updateFilePrefix + std::to_string(i);
                    if (!p_opts.m_stressTest) LoadUpdateTrace(traceFileName, updateSize, insertSet, deleteSet);
                    else LoadUpdateTraceStressTest(traceFileName, updateSize, insertSet);
                    if (!p_opts.m_loadAllVectors) {
                        vectorSet = LoadUpdateVectors(p_opts, insertSet, updateSize);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Updating day: %d: numThread: %d, updateSize: %d,total days: %d.\n", i, insertThreads, updateSize, days);

                    int sampleSize = updateSize / p_opts.m_sampling;

                    int nextSamplePoint = sampleSize + updateSize * i;

                    bool showStatus = false;

                    std::future<void> delete_future;
                    if (!p_opts.m_stressTest) {
                        delete_future =
                            std::async(std::launch::async, DeleteVectorsBySet<ValueType>, p_index,
                                    1, vectorSet, std::ref(deleteSet), std::ref(mapping), updateSize, std::ref(p_opts), i);
                    }

                    std::future_status delete_status;

                    std::future<void> insert_future =
                        std::async(std::launch::async, InsertVectorsBySet<ValueType>, p_index,
                                insertThreads, vectorSet, std::ref(insertSet), std::ref(mapping), updateSize, std::ref(p_opts));

                    std::future_status insert_status;

                    std::string tempFileName;
                    p_opts.m_calTruth = false;
                    do {
                        insert_status = insert_future.wait_for(std::chrono::seconds(2));
                        if (!p_opts.m_stressTest) delete_status = delete_future.wait_for(std::chrono::seconds(2));
                        else delete_status = std::future_status::ready;
                        if (insert_status == std::future_status::timeout || delete_status == std::future_status::timeout) {
                            if (p_index->GetNumDeleted() >= nextSamplePoint) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Samppling Size: %d\n", nextSamplePoint);
                                showStatus = true;
                                nextSamplePoint += sampleSize;
                                ShowMemoryStatus(vectorSet, sw.getElapsedSec());
                                p_index->GetIndexStat(-1, false, false);
                            } else {
                                showStatus = false;
                            }
                            p_index->GetDBStat();
                            if(p_opts.m_searchDuringUpdate) StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, internalResultNum, tempFileName, p_opts, sw.getElapsedSec(), showStatus);
                            p_index->GetDBStat();
                        }
                    } while (insert_status != std::future_status::ready || delete_status != std::future_status::ready);

                    curCount += updateSize;

                    p_index->GetIndexStat(updateSize, true, true);
                    p_index->GetDBStat();

                    ShowMemoryStatus(vectorSet, sw.getElapsedSec());

                    std::string truthFileName;

                    if (!p_opts.m_stressTest) truthFileName = p_opts.m_truthFilePrefix + std::to_string(i);
                    else truthFileName = p_opts.m_truthPath;

                    p_index->Checkpoint();

                    p_opts.m_calTruth = calTruthOrigin;
                    if (p_opts.m_onlySearchFinalBatch && days - 1 != i) continue;
                    p_index->StopMerge();
                    if (p_opts.m_maxInternalResultNum != -1) 
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Latency & Recall Tradeoff\n");
                        for (int iterInternalResultNum = p_opts.m_minInternalResultNum; iterInternalResultNum <= p_opts.m_maxInternalResultNum; iterInternalResultNum += p_opts.m_stepInternalResultNum) 
                        {
                            StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, iterInternalResultNum, truthFileName, p_opts, sw.getElapsedSec());
                        }
                    }
                    else 
                    {
                        StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, internalResultNum, truthFileName, p_opts, sw.getElapsedSec());
                    }
                    p_index->OpenMerge();
                }
            }

            template <typename ValueType>
            void InsertVectors(SPANN::Index<ValueType>* p_index, 
                int insertThreads, 
                std::shared_ptr<SPTAG::VectorSet> vectorSet, 
                int curCount, 
                int step,
                SPANN::Options& p_opts)
            {
                StopWSPFresh sw;
                std::vector<std::thread> threads;

                std::atomic_size_t vectorsSent(0);
                std::vector<double> latency_vector(step);

                auto func = [&]()
                {
                    p_index->Initialize();
                    size_t index = 0;
                    while (true)
                    {
                        index = vectorsSent.fetch_add(1);
                        if (index < step)
                        {
                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / step);
                            }
                            auto insertBegin = std::chrono::high_resolution_clock::now();
                            p_index->AddIndex(vectorSet->GetVector((SizeType)(index + curCount)), 1, p_opts.m_dim, nullptr);
                            auto insertEnd = std::chrono::high_resolution_clock::now();
                            latency_vector[index] = std::chrono::duration_cast<std::chrono::microseconds>(insertEnd - insertBegin).count();
                        }
                        else
                        {
                            p_index->ExitBlockController();
                            return;
                        }
                    }
                };
                for (int j = 0; j < insertThreads; j++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }

                double sendingCost = sw.getElapsedSec();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Finish sending in %.3lf seconds, sending throughput is %.2lf , insertion count %u.\n",
                sendingCost,
                step/ sendingCost,
                static_cast<uint32_t>(step));

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"During Update\n");

                while(!p_index->AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                double syncingCost = sw.getElapsedSec();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Finish syncing in %.3lf seconds, actuall throughput is %.2lf, insertion count %u.\n",
                syncingCost,
                step / syncingCost,
                static_cast<uint32_t>(step));
                PrintPercentiles<double, double>(latency_vector,
                [](const double& ss) -> double
                {
                    return ss;
                },
                "%.3lf");
            }

            template <typename ValueType>
            void UpdateSPFresh(SPANN::Index<ValueType>* p_index)
            {
                SPANN::Options& p_opts = *(p_index->GetOptions());
                int step = p_opts.m_step;
                if (step == 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Incremental Test Error, Need to set step.\n");
                    exit(1);
                }
                StopWSPFresh sw;

                int numThreads = p_opts.m_searchThreadNum;
                int internalResultNum = p_opts.m_searchInternalResultNum;
                int searchTimes = p_opts.m_searchTimes;

                auto vectorSet = LoadVectorSet(p_opts, numThreads);

                auto querySet = LoadQuerySet(p_opts);

                int curCount = p_index->GetNumSamples();
                int insertCount = vectorSet->Count() - curCount;

                bool calTruthOrigin = p_opts.m_calTruth;

                if (p_opts.m_endVectorNum != -1)
                {
                    insertCount = p_opts.m_endVectorNum - curCount;
                }

                p_index->ForceCompaction();

                p_index->GetDBStat();

                if (!p_opts.m_onlySearchFinalBatch) {
                    if (p_opts.m_maxInternalResultNum != -1) 
                    {
                        for (int iterInternalResultNum = p_opts.m_minInternalResultNum; iterInternalResultNum <= p_opts.m_maxInternalResultNum; iterInternalResultNum += p_opts.m_stepInternalResultNum) 
                        {
                            StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, iterInternalResultNum, p_opts.m_truthPath, p_opts, sw.getElapsedSec());
                        }
                    }
                    else 
                    {
                        StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, internalResultNum, p_opts.m_truthPath, p_opts, sw.getElapsedSec());
                    }
                }

                ShowMemoryStatus(vectorSet, sw.getElapsedSec());
                p_index->GetDBStat();

                int batch;
                if (step == 0) {
                    batch = 0;
                } else {
                    batch = insertCount / step;
                }
                
                int finishedInsert = 0;
                int insertThreads = p_opts.m_insertThreadNum;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Updating: numThread: %d, step: %d, insertCount: %d, totalBatch: %d.\n", insertThreads, step, insertCount, batch);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start updating...\n");
                for (int i = 0; i < 1; i++)
                {   
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Updating Batch %d: numThread: %d, step: %d.\n", i, insertThreads, step);

                    std::future<void> insert_future =
                        std::async(std::launch::async, InsertVectors<ValueType>, p_index,
                                insertThreads, vectorSet, curCount, step, std::ref(p_opts));

                    std::future_status insert_status;

                    std::string tempFileName;
                    p_opts.m_calTruth = false;
                    do {
                        insert_status = insert_future.wait_for(std::chrono::seconds(3));
                        if (insert_status == std::future_status::timeout) {
                            ShowMemoryStatus(vectorSet, sw.getElapsedSec());
                            p_index->GetIndexStat(-1, false, false);
                            p_index->GetDBStat();
                            if(p_opts.m_searchDuringUpdate) StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, internalResultNum, tempFileName, p_opts, sw.getElapsedSec());
                        }
                    }while (insert_status != std::future_status::ready);

                    curCount += step;
                    finishedInsert += step;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total Vector num %d \n", curCount);

                    p_index->GetIndexStat(finishedInsert, true, true);

                    ShowMemoryStatus(vectorSet, sw.getElapsedSec());

                    std::string truthFileName = p_opts.m_truthFilePrefix + std::to_string(i);

                    p_opts.m_calTruth = calTruthOrigin;
                    if (p_opts.m_onlySearchFinalBatch && batch - 1 != i) continue;
                    // p_index->ForceGC();
                    // p_index->ForceCompaction();
                    p_index->StopMerge();
                    if (p_opts.m_maxInternalResultNum != -1) 
                    {
                        for (int iterInternalResultNum = p_opts.m_minInternalResultNum; iterInternalResultNum <= p_opts.m_maxInternalResultNum; iterInternalResultNum += p_opts.m_stepInternalResultNum) 
                        {
                            StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, iterInternalResultNum, truthFileName, p_opts, sw.getElapsedSec());
                        }
                    }
                    else 
                    {
                        StableSearch(p_index, numThreads, querySet, vectorSet, searchTimes, p_opts.m_queryCountLimit, internalResultNum, truthFileName, p_opts, sw.getElapsedSec());
                    }
                    p_index->OpenMerge();
                }
            }

            int UpdateTest(const char* storePath) {

                std::shared_ptr<VectorIndex> index;

                if (index->LoadIndex(storePath, index) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to load index.\n");
                    return 1;
                }

                SPANN::Options* opts = nullptr;

            #define DefineVectorValueType(Name, Type) \
                if (index->GetVectorValueType() == VectorValueType::Name) { \
                    opts = ((SPANN::Index<Type>*)index.get())->GetOptions(); \
                } \

            #include "inc/Core/DefinitionList.h"
            #undef DefineVectorValueType

            #define DefineVectorValueType(Name, Type) \
                if (opts->m_valueType == VectorValueType::Name) { \
                    if (opts->m_steadyState) SteadyStateSPFresh((SPANN::Index<Type>*)(index.get())); \
                    else UpdateSPFresh((SPANN::Index<Type>*)(index.get())); \
                } \

            #include "inc/Core/DefinitionList.h"
            #undef DefineVectorValueType

                return 0;
            }
        }
    }
}