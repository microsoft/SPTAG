// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_TRUTHSET_H_
#define _SPTAG_COMMON_TRUTHSET_H_

#include "inc/Core/VectorIndex.h"
#include "QueryResultSet.h"

namespace SPTAG
{
    namespace COMMON
    {
        class TruthSet {
        public:
            static void LoadTruthTXT(std::shared_ptr<SPTAG::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, int K, int& originalK, SizeType& p_iTruthNumber)
            {
                std::size_t lineBufferSize = 20;
                std::unique_ptr<char[]> currentLine(new char[lineBufferSize]);
                truth.clear();
                truth.resize(p_iTruthNumber);
                for (int i = 0; i < p_iTruthNumber; ++i)
                {
                    truth[i].clear();
                    if (ptr->ReadString(lineBufferSize, currentLine, '\n') == 0) {
                        LOG(Helper::LogLevel::LL_Error, "Truth number(%d) and query number(%d) are not match!\n", i, p_iTruthNumber);
                        exit(1);
                    }
                    char* tmp = strtok(currentLine.get(), " ");
                    for (int j = 0; j < K; ++j)
                    {
                        if (tmp == nullptr) {
                            LOG(Helper::LogLevel::LL_Error, "Truth number(%d, %d) and query number(%d) are not match!\n", i, j, p_iTruthNumber);
                            exit(1);
                        }
                        int vid = std::atoi(tmp);
                        if (vid >= 0) truth[i].insert(vid);
                        tmp = strtok(nullptr, " ");
                    }
                }
            }

            static void LoadTruthXVEC(std::shared_ptr<SPTAG::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, int K, int& originalK, SizeType& p_iTruthNumber)
            {
                truth.clear();
                truth.resize(p_iTruthNumber);
                std::vector<int> vec(K);
                for (int i = 0; i < p_iTruthNumber; i++) {
                    if (ptr->ReadBinary(4, (char*)&originalK) != 4 || originalK < K) {
                        LOG(Helper::LogLevel::LL_Error, "Error: Xvec file has No.%d vector whose dims are fewer than expected. Expected: %d, Fact: %d\n", i, K, originalK);
                        exit(1);
                    }
                    if (originalK > K) vec.resize(originalK);
                    if (ptr->ReadBinary(originalK * 4, (char*)vec.data()) != originalK * 4) {
                        LOG(Helper::LogLevel::LL_Error, "Truth number(%d) and query number(%d) are not match!\n", i, p_iTruthNumber);
                        exit(1);
                    }
                    truth[i].insert(vec.begin(), vec.begin() + K);
                }
            }

            static void LoadTruthDefault(std::shared_ptr<SPTAG::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, int K, int& originalK, SizeType& p_iTruthNumber) {
                if (ptr->TellP() == 0) {
                    int row;
                    if (ptr->ReadBinary(4, (char*)&row) != 4 || ptr->ReadBinary(4, (char*)&originalK) != 4) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to read truth file!\n");
                        exit(1);
                    }
                }
                truth.clear();
                truth.resize(p_iTruthNumber);
                std::vector<int> vec(originalK);
                for (int i = 0; i < p_iTruthNumber; i++)
                {
                    if (ptr->ReadBinary(4 * originalK, (char*)vec.data()) != 4 * originalK) {
                        LOG(Helper::LogLevel::LL_Error, "Truth number(%d) and query number(%d) are not match!\n", i, p_iTruthNumber);
                        exit(1);
                    }
                    truth[i].insert(vec.begin(), vec.begin() + K);
                }
            }

            static void LoadTruth(std::shared_ptr<SPTAG::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, SizeType& NumQuerys, int& originalK, int K, TruthFileType type)
            {
                if (type == TruthFileType::TXT)
                {
                    LoadTruthTXT(ptr, truth, K, originalK, NumQuerys);
                }
                else if (type == TruthFileType::XVEC)
                {
                    LoadTruthXVEC(ptr, truth, K, originalK, NumQuerys);
                }
                else if (type == TruthFileType::DEFAULT) {
                    LoadTruthDefault(ptr, truth, K, originalK, NumQuerys);
                }
                else
                {
                    LOG(Helper::LogLevel::LL_Error, "TruthFileType Unsupported.\n");
                    exit(1);
                }
            }

            static void writeTruthFile(const std::string truthFile, SizeType queryNumber, const int K, std::vector<std::vector<SPTAG::SizeType>>& truthset, std::vector<std::vector<float>>& distset, SPTAG::TruthFileType TFT) {
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::out | std::ios::binary)) {
                    LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", truthFile.c_str());
                    exit(1);
                }

                if (TFT == SPTAG::TruthFileType::TXT)
                {
                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        for (int k = 0; k < K; k++)
                        {
                            if (ptr->WriteString((std::to_string(truthset[i][k]) + " ").c_str()) == 0) {
                                LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
                                exit(1);
                            }
                        }
                        if (ptr->WriteString("\n") == 0) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
                            exit(1);
                        }
                    }
                }
                else if (TFT == SPTAG::TruthFileType::XVEC)
                {
                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        if (ptr->WriteBinary(sizeof(K), (char*)&K) != sizeof(K) || ptr->WriteBinary(K * 4, (char*)(truthset[i].data())) != K * 4) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
                            exit(1);
                        }
                    }
                }
                else if (TFT == SPTAG::TruthFileType::DEFAULT) {
                    ptr->WriteBinary(4, (char*)&queryNumber);
                    ptr->WriteBinary(4, (char*)&K);

                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        if (ptr->WriteBinary(K * 4, (char*)(truthset[i].data())) != K * 4) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
                            exit(1);
                        }
                    }
                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        if (ptr->WriteBinary(K * 4, (char*)(distset[i].data())) != K * 4) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
                            exit(1);
                        }
                    }
                }
                else {
                    LOG(Helper::LogLevel::LL_Error, "Found unsupported file type for generating truth.");
                    exit(-1);
                }
            }

            template<typename T>
            static void GenerateTruth(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
                const SPTAG::DistCalcMethod distMethod, const int K, const SPTAG::TruthFileType p_truthFileType, const std::shared_ptr<IQuantizer>& quantizer) {
                if (querySet->Dimension() != vectorSet->Dimension() && !quantizer)
                {
                    LOG(Helper::LogLevel::LL_Error, "query and vector have different dimensions.");
                    exit(1);
                }

                LOG(Helper::LogLevel::LL_Info, "Begin to generate truth for query(%d,%d) and doc(%d,%d)...\n", querySet->Count(), querySet->Dimension(), vectorSet->Count(), vectorSet->Dimension());
                std::vector< std::vector<SPTAG::SizeType> > truthset(querySet->Count(), std::vector<SPTAG::SizeType>(K, 0));
                std::vector< std::vector<float> > distset(querySet->Count(), std::vector<float>(K, 0));
                auto fComputeDistance = quantizer ? quantizer->DistanceCalcSelector<T>(distMethod) : COMMON::DistanceCalcSelector<T>(distMethod);
#pragma omp parallel for
                for (int i = 0; i < querySet->Count(); ++i)
                {
                    SPTAG::COMMON::QueryResultSet<T> query((const T*)(querySet->GetVector(i)), K);
                    query.SetTarget((const T*)(querySet->GetVector(i)), quantizer);
                    for (SPTAG::SizeType j = 0; j < vectorSet->Count(); j++)
                    {
                        float dist = fComputeDistance(query.GetQuantizedTarget(), reinterpret_cast<T*>(vectorSet->GetVector(j)), vectorSet->Dimension());
                        query.AddPoint(j, dist);
                    }
                    query.SortResult();

                    for (int k = 0; k < K; k++)
                    {
                        truthset[i][k] = (query.GetResult(k))->VID;
                        distset[i][k] = (query.GetResult(k))->Dist;
                    }

                }
                LOG(Helper::LogLevel::LL_Info, "Start to write truth file...\n");
                writeTruthFile(truthFile, querySet->Count(), K, truthset, distset, p_truthFileType);

                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize((truthFile + ".dist.bin").c_str(), std::ios::out | std::ios::binary)) {
                    LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", (truthFile + ".dist.bin").c_str());
                    exit(1);
                }

                int int32_queryNumber = (int)querySet->Count();
                ptr->WriteBinary(4, (char*)&int32_queryNumber);
                ptr->WriteBinary(4, (char*)&K);

                for (size_t i = 0; i < int32_queryNumber; i++)
                {
                    for (int k = 0; k < K; k++) {
                        if (ptr->WriteBinary(4, (char*)(&(truthset[i][k]))) != 4) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                            exit(1);
                        }
                        if (ptr->WriteBinary(4, (char*)(&(distset[i][k]))) != 4) {
                            LOG(Helper::LogLevel::LL_Error, "Fail to write the truth dist file!\n");
                            exit(1);
                        }
                    }
                }
            }

            template <typename T>
            static float CalculateRecall(VectorIndex* index, std::vector<QueryResult>& results, const std::vector<std::set<SizeType>>& truth, int K, int truthK, std::shared_ptr<SPTAG::VectorSet> querySet, std::shared_ptr<SPTAG::VectorSet> vectorSet, SizeType NumQuerys, std::ofstream* log = nullptr, bool debug = false, float* MRR = nullptr)
            {
                float meanrecall = 0, minrecall = MaxDist, maxrecall = 0, stdrecall = 0, meanmrr = 0;
                std::vector<float> thisrecall(NumQuerys, 0);
                std::unique_ptr<bool[]> visited(new bool[K]);
                for (SizeType i = 0; i < NumQuerys; i++)
                {
                    int minpos = K;
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
                                if (j < minpos) minpos = j;
                                break;
                            }
                            else if (vectorSet != nullptr) {
                                float dist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(results[i].GetResult(j)->VID), vectorSet->Dimension(), index->GetDistCalcMethod());
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
                    thisrecall[i] /= truth[i].size();
                    meanrecall += thisrecall[i];
                    if (thisrecall[i] < minrecall) minrecall = thisrecall[i];
                    if (thisrecall[i] > maxrecall) maxrecall = thisrecall[i];
                    if (minpos < K) meanmrr += 1.0f / (minpos + 1);

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
                if (log) (*log) << meanrecall << " " << stdrecall << " " << minrecall << " " << maxrecall << std::endl;
                if (MRR) *MRR = meanmrr / NumQuerys;
                return meanrecall;
            }

            template <typename T>
            static float CalculateRecall(VectorIndex* index, T* query, int K) {
                COMMON::QueryResultSet<void> sampleANN(query, K);
                COMMON::QueryResultSet<void> sampleTruth(query, K);
                void* reconstructVector = nullptr;
                if (index->m_pQuantizer)
                {
                    reconstructVector = ALIGN_ALLOC(index->m_pQuantizer->ReconstructSize());
                    index->m_pQuantizer->ReconstructVector((const uint8_t*)query, reconstructVector);
                    sampleANN.SetTarget(reconstructVector, index->m_pQuantizer);
                    sampleTruth.SetTarget(reconstructVector, index->m_pQuantizer);
                }

                index->SearchIndex(sampleANN);
                for (SizeType y = 0; y < index->GetNumSamples(); y++)
                {
                    float dist = index->ComputeDistance(sampleTruth.GetQuantizedTarget(), index->GetSample(y));
                    sampleTruth.AddPoint(y, dist);
                }
                sampleTruth.SortResult();

                float recalls = 0;
                std::vector<bool> visited(K, false);
                for (SizeType y = 0; y < K; y++)
                {
                    for (SizeType z = 0; z < K; z++)
                    {
                        if (visited[z]) continue;

                        if (fabs(sampleANN.GetResult(z)->Dist - sampleTruth.GetResult(y)->Dist) < Epsilon)
                        {
                            recalls += 1;
                            visited[z] = true;
                            break;
                        }
                    }
                }
                if (reconstructVector)
                {
                    ALIGN_FREE(reconstructVector);
                }

                return recalls / K;
            }
        };
    }
}

#endif // _SPTAG_COMMON_TRUTHSET_H_
