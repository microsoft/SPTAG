// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/SSDServing/VectorSearch/IExtraSearcher.h"
#include "inc/SSDServing/VectorSearch/ExtraFullGraphSearcher.h"

#include "inc/SSDServing/IndexBuildManager/Utils.h"
#include "inc/SSDServing/IndexBuildManager/CommonDefines.h"
#include "inc/SSDServing/VectorSearch/TimeUtils.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Core/VectorIndex.h"

#include <boost/lockfree/stack.hpp>

#include <atomic>

namespace SPTAG {
	namespace SSDServing {
		namespace VectorSearch {
			// LARGE_INTEGER g_systemPerfFreq;

			template <typename ValueType>
			class SearchDefault
			{
			public:
				SearchDefault()
					:m_workspaces(128)
				{
					m_tids = 0;
					//QueryPerformanceFrequency(&g_systemPerfFreq);
				}

				~SearchDefault()
				{
					ExtraWorkSpace* context;
					while (m_workspaces.pop(context))
					{
						delete context;
					}
				}

				void ReadVectorIDsFile(ByteArray& p_myArray) {
					p_myArray = p_myArray.Alloc(sizeof(long long) * m_index->GetNumSamples());
					if (COMMON_OPTS.m_headIDFile.empty()) {
						LOG(Helper::LogLevel::LL_Error, "Config error: VectorTranlateMap Empty for Searching SSD vectors.\n");
						exit(1);
					}
					auto ptr = f_createIO();
					if (ptr == nullptr || !ptr->Initialize(COMMON_OPTS.m_headIDFile.c_str(), std::ios::binary | std::ios::in)) {
						LOG(Helper::LogLevel::LL_Error, "Failed open %s\n", COMMON_OPTS.m_headIDFile.c_str());
						exit(1);
					}
					if (ptr->ReadBinary(sizeof(long long) * m_index->GetNumSamples(), reinterpret_cast<char*>(p_myArray.Data())) != sizeof(long long) * m_index->GetNumSamples()) {
						LOG(Helper::LogLevel::LL_Error, "Failed to read vectorTanslateMap!\n");
						exit(1);
					}
				}

				void LoadHeadIndex(Options& p_opts) {
					LOG(Helper::LogLevel::LL_Info, "Start loading head index. \n");

					if (VectorIndex::LoadIndex(COMMON_OPTS.m_headIndexFolder, m_index) != ErrorCode::Success) {
						LOG(Helper::LogLevel::LL_Error, "ERROR: Cannot Load index files!\n");
						exit(1);
					}

					m_index->SetParameter("NumberOfThreads", std::to_string(p_opts.m_iNumberOfThreads));
					m_index->SetParameter("MaxCheck", std::to_string(p_opts.m_maxCheck));
					m_index->SetParameter("HashTableExponent", std::to_string(p_opts.m_hashExp));

					if (!p_opts.m_headConfig.empty())
					{
						Helper::IniReader iniReader;
						if (iniReader.LoadIniFile(p_opts.m_headConfig) != ErrorCode::Success) {
							LOG(Helper::LogLevel::LL_Error, "ERROR of loading head index config: %s\n", p_opts.m_headConfig.c_str());
							exit(1);
						}

						for (const auto& iter : iniReader.GetParameters("Index"))
						{
							m_index->SetParameter(iter.first.c_str(), iter.second.c_str());
						}
					}
					m_index->UpdateIndex();
					LOG(Helper::LogLevel::LL_Info, "End loading head index. \n");
				}

				void LoadVectorIdsSSDIndex(long long* vectorTranslateMap, std::string extraFullGraphFile, int postingPageLimit, int ioThreads)
				{
					if (extraFullGraphFile.empty()) {
						LOG(Helper::LogLevel::LL_Error, "Config error: SsdIndex empty for Searching SSD vectors.\n");
						exit(1);
					}

					m_vectorTranslateMap = vectorTranslateMap;

					LOG(Helper::LogLevel::LL_Info, "Using FullGraph without cache.\n");

					m_extraSearcher.reset(new ExtraFullGraphSearcher<ValueType>(extraFullGraphFile, postingPageLimit, ioThreads));
				}

				void CheckHeadIndexType() {
					SPTAG::VectorValueType v1 = m_index->GetVectorValueType(), v2 = GetEnumValueType<ValueType>();
					if (v1 != v2) {
						LOG(Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
							SPTAG::Helper::Convert::ConvertToString(v1).c_str(),
							SPTAG::Helper::Convert::ConvertToString(v2).c_str()
						);
						if (!SPTAG::COMMON::DistanceUtils::Quantizer)
						{
							exit(1);
						}
					}
				}

				template <typename DataType>
				DataType GetParameter(const std::string& p_config, const std::string& p_param, DataType& p_defaultVal) const {
					size_t begin = p_config.find(p_param);
					if (begin == std::string::npos) return p_defaultVal;

					size_t valuebegin = p_config.find("=", begin), end = p_config.find("\n", begin);
					if (valuebegin == std::string::npos || end == std::string::npos) return p_defaultVal;

					std::string value = p_config.substr(valuebegin + 1, end - valuebegin - 1);
					return Helper::Convert::ConvertStringTo<DataType>(value.c_str(), p_defaultVal);
				}

				void LoadIndex4ANNIndexTestTool(const std::string& p_config,
					const std::vector<ByteArray>& p_indexBlobs,
					long long* vectorTranslateMap,
					std::string& extraFullGraphFile)
				{
					if (VectorIndex::LoadIndex(p_config, p_indexBlobs, m_index) != SPTAG::ErrorCode::Success) {
						LOG(Helper::LogLevel::LL_Error, "LoadIndex error in LoadIndex4ANNIndexTestTool.\n");
						exit(1);
					}
					CheckHeadIndexType();
					LoadVectorIdsSSDIndex(vectorTranslateMap, extraFullGraphFile, 
						GetParameter(p_config, "SearchPostingPageLimit", (std::numeric_limits<int>::max)()),
						GetParameter(p_config, "IOThreadsPerHandler", 4));
				}

				void Setup(Options& p_config, ByteArray& p_myArray)
				{
					m_maxDistRatio = p_config.m_maxDistRatio;
					LoadHeadIndex(p_config);
					CheckHeadIndexType();
					if (!p_config.m_buildSsdIndex)
					{
						ReadVectorIDsFile(p_myArray);
						LoadVectorIdsSSDIndex(reinterpret_cast<long long*>(p_myArray.Data()), COMMON_OPTS.m_ssdIndex, p_config.m_searchPostingPageLimit, p_config.m_ioThreads);
						int internalResultNum = std::max<int>(p_config.m_internalResultNum, p_config.m_resultNum);
						for (int i = 0; i < p_config.m_iNumberOfThreads; i++) {
							ExtraWorkSpace* ws = new ExtraWorkSpace();
							ws->m_postingIDs.reserve(internalResultNum);
							ws->m_deduper.Init(p_config.m_maxCheck, p_config.m_hashExp);
							ws->m_pageBuffers.resize(internalResultNum);
							for (int pi = 0; pi < internalResultNum; pi++) {
								ws->m_pageBuffers[pi].ReservePageBuffer(m_extraSearcher->GetMaxListSize());
							}
							ws->m_diskRequests.resize(internalResultNum);
							m_workspaces.push(ws);
						}
					}
				}

				void Search(COMMON::QueryResultSet<ValueType>& p_queryResults, SearchStats& p_stats)
				{
					//LARGE_INTEGER qpcStartTime;
					//LARGE_INTEGER qpcEndTime;
					//QueryPerformanceCounter(&qpcStartTime);
					TimeUtils::StopW sw;
					double StartingTime, EndingTime, ExEndingTime;

					StartingTime = sw.getElapsedMs();

					m_index->SearchIndex(p_queryResults);

					EndingTime = sw.getElapsedMs();

					ExtraWorkSpace* auto_ws = nullptr;
					if (nullptr != m_extraSearcher)
					{
						auto_ws = GetWs(p_queryResults.GetResultNum());
						auto_ws->m_postingIDs.clear();

						float limitDist = p_queryResults.GetResult(0)->Dist * m_maxDistRatio;
						for (int i = 0; i < p_queryResults.GetResultNum(); ++i)
						{
							auto res = p_queryResults.GetResult(i);
							if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist)) break;
							auto_ws->m_postingIDs.emplace_back(res->VID);
						}

						for (int i = 0; i < p_queryResults.GetResultNum(); ++i)
						{
							auto res = p_queryResults.GetResult(i);
							if (res->VID == -1) break;
							res->VID = static_cast<int>(m_vectorTranslateMap[res->VID]);
						}

						p_queryResults.Reverse();
						m_extraSearcher->Search(auto_ws, p_queryResults, m_index, p_stats);
						RetWs(auto_ws);
					}

					ExEndingTime = sw.getElapsedMs();
					//QueryPerformanceCounter(&qpcEndTime);
					//unsigned __int32 latency = static_cast<unsigned __int32>(static_cast<double>(qpcEndTime.QuadPart - qpcStartTime.QuadPart) * 1000000 / g_systemPerfFreq.QuadPart);

					p_stats.m_exLatency = ExEndingTime - EndingTime;
					p_stats.m_totalSearchLatency = ExEndingTime - StartingTime;
					p_stats.m_totalLatency = p_stats.m_totalSearchLatency;
					//p_stats.m_totalLatency = latency * 1.0;
				}

				void Search4ANNIndexTestTool(COMMON::QueryResultSet<ValueType>& p_queryResults)
				{
					m_index->SearchIndex(p_queryResults);

					ExtraWorkSpace* auto_ws = nullptr;
					auto_ws = GetWs(p_queryResults.GetResultNum());
					auto_ws->m_postingIDs.clear();

					for (int i = 0; i < p_queryResults.GetResultNum(); ++i)
					{
						auto res = p_queryResults.GetResult(i);
						if (res->VID != -1)
						{
							auto_ws->m_postingIDs.emplace_back(res->VID);
							res->VID = static_cast<int>(m_vectorTranslateMap[res->VID]);
						}
					}

					p_queryResults.Reverse();

					//TODO may be removed
					SearchStats stats;
					m_extraSearcher->Search(auto_ws, p_queryResults, m_index, stats);
					RetWs(auto_ws);
				}

				class SearchAsyncJob : public SPTAG::Helper::ThreadPool::Job
				{
				private:
					SearchDefault* m_processor;
					COMMON::QueryResultSet<ValueType>& m_queryResults;
					SearchStats& m_stats;
					std::function<void()> m_callback;
				public:
					SearchAsyncJob(SearchDefault* p_processor,
						COMMON::QueryResultSet<ValueType>& p_queryResults, SearchStats& p_stats, std::function<void()> p_callback)
						: m_processor(p_processor),
						m_queryResults(p_queryResults), m_stats(p_stats), m_callback(p_callback) {}

					~SearchAsyncJob() {}

					void exec(IAbortOperation* p_abort) {
						m_processor->ProcessAsyncSearch(m_queryResults, m_stats, std::move(m_callback));
					}
				};

				void SearchAsync(COMMON::QueryResultSet<ValueType>& p_queryResults, SearchStats& p_stats, std::function<void()> p_callback)
				{
					p_stats.m_searchRequestTime = std::chrono::steady_clock::now();

					SearchAsyncJob* curJob = new SearchAsyncJob(this, p_queryResults, p_stats, p_callback);

					m_threadPool->add(curJob);
				}


				void SetHint(int p_threadNum, int p_resultNum, bool p_asyncCall, const Options& p_opts)
				{
					LOG(Helper::LogLevel::LL_Info, "ThreadNum: %d, ResultNum: %d, AsyncCall: %d\n", p_threadNum, p_resultNum, p_asyncCall ? 1 : 0);

					if (p_asyncCall)
					{
						m_threadPool.reset(new Helper::ThreadPool());
						m_threadPool->init(p_threadNum);
					}
				}

				std::shared_ptr<VectorIndex> HeadIndex() {
					return m_index;
				}

				SPTAG::DimensionType GetDimension() {
					return HeadIndex()->GetFeatureDim();
				}

				ExtraWorkSpace* GetWs(int internalResultNum) {
					ExtraWorkSpace* ws = nullptr;
					if (!m_workspaces.pop(ws)) {
						ws = new ExtraWorkSpace();
						ws->m_postingIDs.reserve(internalResultNum);
						ws->m_deduper.Init(atoi(m_index->GetParameter("MaxCheck").c_str()),
							atoi(m_index->GetParameter("HashTableExponent").c_str()));
						ws->m_pageBuffers.resize(internalResultNum);
						for (int pi = 0; pi < internalResultNum; pi++) {
							ws->m_pageBuffers[pi].ReservePageBuffer(m_extraSearcher->GetMaxListSize());
						}
						ws->m_diskRequests.resize(internalResultNum);
					}
					return ws;
				}

				void RetWs(ExtraWorkSpace* ws) {
					if (ws != nullptr)
					{
						m_workspaces.push(ws);
					}
				}

			protected:
				void ProcessAsyncSearch(COMMON::QueryResultSet<ValueType>& p_queryResults, SearchStats& p_stats, std::function<void()> p_callback)
				{
					static thread_local int tid = m_tids.fetch_add(1);

					std::chrono::steady_clock::time_point startPoint = std::chrono::steady_clock::now();
					p_stats.m_queueLatency = TimeUtils::getMsInterval(p_stats.m_searchRequestTime, startPoint);

					p_stats.m_threadID = tid;

					static thread_local std::chrono::steady_clock::time_point m_lastQuit = startPoint;
					p_stats.m_sleepLatency = TimeUtils::getMsInterval(m_lastQuit, startPoint);

					Search(p_queryResults, p_stats);

					p_stats.m_totalLatency = TimeUtils::getMsInterval(p_stats.m_searchRequestTime, std::chrono::steady_clock::now());

					p_callback();

					m_lastQuit = std::chrono::steady_clock::now();
				}

				std::shared_ptr<VectorIndex> m_index;

				long long* m_vectorTranslateMap = nullptr;

				std::unique_ptr<IExtraSearcher<ValueType>> m_extraSearcher;

				std::unique_ptr<Helper::ThreadPool> m_threadPool;

				std::atomic<std::int32_t> m_tids;

				boost::lockfree::stack<ExtraWorkSpace*> m_workspaces;

				float m_maxDistRatio;
			};
		}
	}
}