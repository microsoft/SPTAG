#pragma once

#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"
#include "inc/SSDServing/VectorSearch/Options.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Helper/Concurrent.h"
#include "inc/SSDServing/VectorSearch/SearchStats.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inttypes.h"

namespace SPTAG {
	namespace SSDServing {
		namespace VectorSearch {
            template <typename T>
            float CalcRecall(std::vector<COMMON::QueryResultSet<T>>& results, const std::vector<std::set<int>>& truth, int K)
            {
                float recall = 0;
                for (int i = 0; i < results.size(); i++)
                {
                    for (int id : truth[i])
                    {
                        for (int j = 0; j < K; j++)
                        {
                            if (results[i].GetResult(j)->VID == id)
                            {
                                recall++;
                                break;
                            }
                        }
                    }
                }
                return static_cast<float>(recall)/static_cast<float>(results.size() * K);
            }

            void LoadTruthTXT(std::string truthPath, std::vector<std::set<int>>& truth, int K, SizeType p_iTruthNumber)
            {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(truthPath.c_str(), std::ios::in)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthPath.c_str());
                    exit(1);
                }
                std::size_t lineBufferSize = 20;
                std::unique_ptr<char[]> currentLine(new char[lineBufferSize]);
                truth.clear();
                truth.resize(p_iTruthNumber);
                for (int i = 0; i < p_iTruthNumber; ++i)
                {
                    truth[i].clear();
                    for (int j = 0; j < K; ++j)
                    {
                        if (ptr->ReadString(lineBufferSize, currentLine, ' ') == 0) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to read truth file!\n");
                            exit(1);
                        }
                        truth[i].insert(std::atoi(currentLine.get()));
                    }
                    if (ptr->ReadString(lineBufferSize, currentLine, '\n') == 0) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to read truth file!\n");
                        exit(1);
                    }
                }
            }

            void LoadTruthXVEC(std::string truthPath, std::vector<std::set<int>>& truth, int K, SizeType p_iTruthNumber)
            {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(truthPath.c_str(), std::ios::in | std::ios::binary)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthPath.c_str());
                    exit(1);
                }

                DimensionType dim = K;
                std::vector<int> temp_vec(K);
                truth.clear();
                truth.resize(p_iTruthNumber);
                for (size_t i = 0; i < p_iTruthNumber; i++) {
                    if (ptr->ReadBinary(4, (char*)&dim) != 4 || dim < K) {
                        LOG(Helper::LogLevel::LL_Error, "Error: Xvec file %s has No.%" PRId64 " vector whose dims are fewer than expected. Expected: %" PRId32 ", Fact: %" PRId32 "\n", truthPath.c_str(), i, K, dim);
                        exit(1);
                    }
                    if (dim > K) temp_vec.resize(dim);
                    if (ptr->ReadBinary(dim * 4, (char*)temp_vec.data()) != dim * 4) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to read truth file!\n");
                        exit(1);
                    }
                    truth[i].insert(temp_vec.begin(), temp_vec.begin() + K);
                }
            }

            void LoadTruthDefault(std::string truthPath, std::vector<std::set<int>>& truth, int K, SizeType p_iTruthNumber) {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(truthPath.c_str(), std::ios::in | std::ios::binary)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthPath.c_str());
                    exit(1);
                }

                int row, column;
                if (ptr->ReadBinary(4, (char*)&row) != 4 || ptr->ReadBinary(4, (char*)&column) != 4) {
                    LOG(Helper::LogLevel::LL_Error, "Fail to read truth file!\n");
                    exit(1);
                }
                truth.clear();
                truth.resize(row);
                std::vector<int> vec(column);
                for (size_t i = 0; i < row; i++)
                {
                    if (ptr->ReadBinary(4 * column, (char*)vec.data()) != 4 * column) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to read truth file!\n");
                        exit(1);
                    }
                    truth[i].insert(vec.begin(), vec.begin() + K);
                }
            }

            void LoadTruth(std::string truthPath, std::vector<std::set<int>>& truth, int NumQuerys, int K)
            {
                if (COMMON_OPTS.m_truthType == TruthFileType::TXT)
                {
                    LoadTruthTXT(truthPath, truth, K, NumQuerys);
                } 
                else if (COMMON_OPTS.m_truthType == TruthFileType::XVEC)
                {
                    LoadTruthXVEC(truthPath, truth, K, NumQuerys);
                }
                else if (COMMON_OPTS.m_truthType == TruthFileType::DEFAULT) {
                    LoadTruthDefault(truthPath, truth, K, NumQuerys);
                }
                else
                {
                    LOG(Helper::LogLevel::LL_Error, "TruthFileType Unsupported.\n");
                    exit(1);
                }
            }

            template <typename ValueType>
            void OutputResult(const std::string& p_output, std::vector<COMMON::QueryResultSet<ValueType>>& p_results, int p_resultNum)
            {
                if (!p_output.empty())
                {
                    auto ptr = SPTAG::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_output.c_str(), std::ios::binary | std::ios::out)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed create file: %s\n", p_output.c_str());
                        exit(1);
                    }
                    int32_t i32Val = static_cast<int32_t>(p_results.size());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        exit(1);
                    }
                    i32Val = p_resultNum;
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        exit(1);
                    }

                    float fVal = 0;
                    for (size_t i = 0; i < p_results.size(); ++i)
                    {
                        for (int j = 0; j < p_resultNum; ++j)
                        {
                            i32Val = p_results[i].GetResult(j)->VID;
                            if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                                LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                exit(1);
                            }

                            fVal = p_results[i].GetResult(j)->Dist;
                            if (ptr->WriteBinary(sizeof(fVal), reinterpret_cast<char*>(&fVal)) != sizeof(fVal)) {
                                LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                exit(1);
                            }
                        }
                    }
                }
            }

            template<typename T, typename V>
            void PrintPercentiles(const std::vector<V>& p_values, std::function<T(const V&)> p_get, const char* p_format)
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

                std::sort(collects.begin(), collects.end());

                LOG(Helper::LogLevel::LL_Info, "Avg\t50tiles\t90tiles\t95tiles\t99tiles\t99.9tiles\tMax\n");

                std::string formatStr("%.3lf");
                for (int i = 1; i < 7; ++i)
                {
                    formatStr += '\t';
                    formatStr += p_format;
                }

                formatStr += '\n';

                LOG(Helper::LogLevel::LL_Info,
                    formatStr.c_str(),
                    sum / collects.size(),
                    collects[static_cast<size_t>(collects.size() * 0.50)],
                    collects[static_cast<size_t>(collects.size() * 0.90)],
                    collects[static_cast<size_t>(collects.size() * 0.95)],
                    collects[static_cast<size_t>(collects.size() * 0.99)],
                    collects[static_cast<size_t>(collects.size() * 0.999)],
                    collects[static_cast<size_t>(collects.size() - 1)]);
            }


            template <typename ValueType>
            void SearchSequential(SearchDefault<ValueType>& p_searcher,
                int p_numThreads,
                std::vector<COMMON::QueryResultSet<ValueType>>& p_results,
                std::vector<SearchStats>& p_stats,
                int p_maxQueryCount)
            {
                int numQueries = min(static_cast<int>(p_results.size()), p_maxQueryCount);

                TimeUtils::StopW sw;

                LOG(Helper::LogLevel::LL_Info, "Searching: numThread: %d, numQueries: %d.\n", p_numThreads, numQueries);
#pragma omp parallel for num_threads(p_numThreads)
                for (int next = 0; next < numQueries; ++next)
                {
                    if ((next & ((1 << 14) - 1)) == 0)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Processed %.2lf%%...\n", next * 100.0 / numQueries);
                    }

                    p_searcher.Search(p_results[next], p_stats[next]);
                }

                double sendingCost = sw.getElapsedSec();

                LOG(Helper::LogLevel::LL_Info,
                    "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                    sendingCost,
                    numQueries / sendingCost,
                    static_cast<uint32_t>(numQueries));
            }


            template <typename ValueType>
            void SearchAsync(SearchDefault<ValueType>& p_searcher,
                uint32_t p_qps,
                std::vector<COMMON::QueryResultSet<ValueType>>& p_results,
                std::vector<SearchStats>& p_stats,
                int p_maxQueryCount)
            {
                size_t numQueries = std::min<size_t>(p_results.size(), p_maxQueryCount);

                LOG(Helper::LogLevel::LL_Info, "Using Async sending with QPS setting %u\n", p_qps);

                Helper::Concurrent::WaitSignal waitFinish;
                waitFinish.Reset(static_cast<uint32_t>(numQueries));

                auto callback = [&waitFinish]()
                {
                    waitFinish.FinishOne();
                };

                std::atomic_size_t queriesSent(0);

                TimeUtils::SteadClock::time_point startTime = TimeUtils::SteadClock::now();

                auto func = [&]()
                {
                    size_t index = 0;

                    while (true)
                    {
                        TimeUtils::SteadClock::time_point currentTime = TimeUtils::SteadClock::now();

                        double timeElapsedSec = TimeUtils::getMsInterval(startTime, currentTime);

                        size_t targetQueries = std::min<size_t>(static_cast<size_t>(p_qps * timeElapsedSec), numQueries);

                        while (targetQueries > index)
                        {
                            index = queriesSent.fetch_add(1);

                            if (index < numQueries)
                            {
                                p_searcher.SearchAsync(p_results[index], p_stats[index], callback);

                                if ((index & ((1 << 14) - 1)) == 0)
                                {
                                    LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / numQueries);
                                }
                            }
                            else
                            {
                                return;
                            }
                        }
                    }
                };

                std::thread thread1(func);
                std::thread thread2(func);
                std::thread thread3(func);

                thread1.join();
                thread2.join();
                thread3.join();

                TimeUtils::SteadClock::time_point finishSending = TimeUtils::SteadClock::now();
                double sendingCost = TimeUtils::getSecInterval(startTime, finishSending);

                LOG(Helper::LogLevel::LL_Info,
                    "Finish sending in %.3lf seconds, QPS setting is %u, actuallQPS is %.2lf, query count %u.\n",
                    sendingCost,
                    p_qps,
                    numQueries / sendingCost,
                    static_cast<uint32_t>(numQueries));

                waitFinish.Wait();

                TimeUtils::SteadClock::time_point finishSearch = TimeUtils::SteadClock::now();
                double searchCost = TimeUtils::getSecInterval(startTime, finishSearch);

                LOG(Helper::LogLevel::LL_Info,
                    "Finish searching in %.3lf seconds.\n",
                    searchCost);
            }

            template <typename ValueType>
            void Search(Options& p_opts)
            {
                std::string outputFile = p_opts.m_searchResult;
                std::string truthFile = COMMON_OPTS.m_truthPath;
                std::string warmupFile = COMMON_OPTS.m_warmupPath;

                if (!p_opts.m_logFile.empty())
                {
                    SPTAG::g_pLogger.reset(new Helper::FileLogger(Helper::LogLevel::LL_Info, p_opts.m_logFile.c_str()));
                }
                int numThreads = p_opts.m_iNumberOfThreads;
                int asyncCallQPS = p_opts.m_qpsLimit;

                int internalResultNum = std::max<int>(p_opts.m_internalResultNum, p_opts.m_resultNum);
                int K = std::min<int>(p_opts.m_resultNum, internalResultNum);

                SearchDefault<ValueType> searcher;
                LOG(Helper::LogLevel::LL_Info, "Start setup index...\n");
                searcher.Setup(p_opts);

                LOG(Helper::LogLevel::LL_Info, "Setup index finish, start setup hint...\n");
                searcher.SetHint(numThreads, internalResultNum, asyncCallQPS > 0, p_opts);

                if (!warmupFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Info, "Start loading warmup query set...\n");
                    std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(COMMON_OPTS.m_valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_warmupType, COMMON_OPTS.m_warmupDelimiter));
                    auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                    if (ErrorCode::Success != queryReader->LoadFile(COMMON_OPTS.m_warmupPath))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                        exit(1);
                    }
                    auto warmupQuerySet = queryReader->GetVectorSet();
                    int warmupNumQueries = warmupQuerySet->Count();

                    std::vector<COMMON::QueryResultSet<ValueType>> warmupResults(warmupNumQueries, COMMON::QueryResultSet<ValueType>(NULL, internalResultNum));
                    std::vector<SearchStats> warmpUpStats(warmupNumQueries);
                    for (int i = 0; i < warmupNumQueries; ++i)
                    {
                        warmupResults[i].SetTarget(reinterpret_cast<ValueType*>(warmupQuerySet->GetVector(i)));
                        warmupResults[i].Reset();
                    }

                    LOG(Helper::LogLevel::LL_Info, "Start warmup...\n");
                    if (asyncCallQPS == 0)
                    {
                        SearchSequential(searcher, numThreads, warmupResults, warmpUpStats, p_opts.m_queryCountLimit);
                    }
                    else
                    {
                        SearchAsync(searcher, asyncCallQPS, warmupResults, warmpUpStats, p_opts.m_queryCountLimit);
                    }

                    LOG(Helper::LogLevel::LL_Info, "\nFinish warmup...\n");
                }

                LOG(Helper::LogLevel::LL_Info, "Start loading QuerySet...\n");
                std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(COMMON_OPTS.m_valueType, COMMON_OPTS.m_dim, COMMON_OPTS.m_queryType, COMMON_OPTS.m_queryDelimiter));
                auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                if (ErrorCode::Success != queryReader->LoadFile(COMMON_OPTS.m_queryPath))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                    exit(1);
                }
                auto querySet = queryReader->GetVectorSet();
                int numQueries = querySet->Count();

                std::vector<std::set<int>> truth;
                if (!truthFile.empty())
                {

                    LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...\n");
                    LoadTruth(truthFile, truth, numQueries, K);
                }

                std::vector<COMMON::QueryResultSet<ValueType>> results(numQueries, COMMON::QueryResultSet<ValueType>(NULL, internalResultNum));
                std::vector<SearchStats> stats(numQueries);
                for (int i = 0; i < numQueries; ++i)
                {
                    results[i].SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(i)));
                    results[i].Reset();
                }


                LOG(Helper::LogLevel::LL_Info, "Start ANN Search...\n");

                if (asyncCallQPS == 0)
                {
                    SearchSequential(searcher, numThreads, results, stats, p_opts.m_queryCountLimit);
                }
                else
                {
                    SearchAsync(searcher, asyncCallQPS, results, stats, p_opts.m_queryCountLimit);
                }

                LOG(Helper::LogLevel::LL_Info, "\nFinish ANN Search...\n");

                float recall = 0;

                if (!truthFile.empty())
                {
                    recall = CalcRecall(results, truth, K);
                    LOG(Helper::LogLevel::LL_Info, "Recall: %f\n", recall);
                }

                long long exCheckSum = 0;
                int exCheckMax = 0;
                long long exListSum = 0;
                std::for_each(stats.begin(), stats.end(), [&](const SearchStats& p_stat)
                    {
                        exCheckSum += p_stat.m_exCheck;
                        exCheckMax = std::max<int>(exCheckMax, p_stat.m_exCheck);
                        exListSum += p_stat.m_totalListElementsCount;
                    });

                LOG(Helper::LogLevel::LL_Info,
                    "Max Ex Dist Check: %d, Average Ex Dist Check: %.2lf, Average Ex Elements Count: %.2lf.\n",
                    exCheckMax,
                    static_cast<double>(exCheckSum) / numQueries,
                    static_cast<double>(exListSum) / numQueries);

                LOG(Helper::LogLevel::LL_Info, "\nSleep Latency Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_sleepLatency;
                    },
                    "%.3lf");


                LOG(Helper::LogLevel::LL_Info, "\nIn Queue Latency Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_queueLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nEx Latency Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_exLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Search Latency Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_totalSearchLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Latency Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_totalLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Disk Acess Distribution:\n");
                PrintPercentiles<int, SearchStats>(stats,
                    [](const SearchStats& ss) -> int
                    {
                        return ss.m_diskAccessCount;
                    },
                    "%4d");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Async Latency 0 Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_asyncLatency0;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Async Latency 1 Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_asyncLatency1;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Async Latency 2 Distribution:\n");
                PrintPercentiles<double, SearchStats>(stats,
                    [](const SearchStats& ss) -> double
                    {
                        return ss.m_asyncLatency2;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\n");

                if (!outputFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Info, "Start output to %s\n", outputFile.c_str());
                    OutputResult(outputFile, results, K);
                }

                LOG(Helper::LogLevel::LL_Info,
                    "Recall: %f, MaxExCheck: %d, AverageExCheck: %.2lf, AverageExElements: %.2lf\n",
                    recall,
                    exCheckMax,
                    static_cast<double>(exCheckSum) / numQueries,
                    static_cast<double>(exListSum) / numQueries);

                LOG(Helper::LogLevel::LL_Info, "\n");
            }
		}
	}
}