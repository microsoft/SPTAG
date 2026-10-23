// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include <absl/synchronization/mutex.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace SPTAG;

template <typename T>
void Read(std::ifstream& stream, T* data, std::size_t count)
{
    if (!stream.read(reinterpret_cast<char*>(data), sizeof(T) * count)) {
        throw std::runtime_error("Incomplete input file");
    }
}

int main(int argc, char** argv)
{
    if (argc != 7) {
        std::cerr << "Usage: query_probe INDEX QUERIES TRUTH PHASE ROUNDS OUTPUT\n";
        return 2;
    }
    try {
        absl::SetMutexDeadlockDetectionMode(absl::OnDeadlockCycle::kIgnore);
        static_assert(sizeof(SizeType) == 4);
        constexpr int queries = 200, k = 5, threads = 16, dimension = 128;
        const int phase = std::stoi(argv[4]);
        const int rounds = std::stoi(argv[5]);
        if (phase < 0 || phase > 10 || rounds < 1) throw std::runtime_error("Invalid phase or rounds");
        std::ifstream queryFile(argv[2], std::ios::binary);
        SizeType queryCount, queryDimension;
        Read(queryFile, &queryCount, 1);
        Read(queryFile, &queryDimension, 1);
        if (queryCount < queries || queryDimension != dimension) throw std::runtime_error("Wrong query header");
        std::vector<std::uint8_t> data(queries * dimension);
        Read(queryFile, data.data(), data.size());
        std::ifstream truthFile(argv[3], std::ios::binary);
        SizeType truthCount, truthDimension;
        Read(truthFile, &truthCount, 1);
        Read(truthFile, &truthDimension, 1);
        if (truthCount != 4400 || truthDimension != k) throw std::runtime_error("Wrong truth header");
        std::vector<SizeType> truthIDs(11 * queries * k);
        std::vector<float> truthDistances(truthIDs.size());
        Read(truthFile, truthIDs.data(), truthIDs.size());
        Read(truthFile, truthDistances.data(), truthDistances.size());
        std::shared_ptr<VectorIndex> index;
        if (VectorIndex::LoadIndex(argv[1], index) != ErrorCode::Success || !index) {
            throw std::runtime_error("LoadIndex failed");
        }
        const int expectedCount = 990000 + phase * 1000;
        if (index->GetNumSamples() != expectedCount) throw std::runtime_error("Wrong active vector count");
        auto spann = std::dynamic_pointer_cast<SPANN::Index<std::uint8_t>>(index);
        if (!spann) throw std::runtime_error("Expected UInt8 SPANN index");
        std::ofstream output(argv[6]);
        if (!output) throw std::runtime_error("Cannot open output");
        output << std::fixed << std::setprecision(6)
               << "{\"phase\":" << phase << ",\"activeVectors\":" << expectedCount
               << ",\"queries\":200,\"topK\":5,\"searchThreads\":16,\"rounds\":[";
        for (int round = 0; round < rounds; ++round) {
            std::vector<QueryResult> results;
            results.reserve(queries);
            for (int q = 0; q < queries; ++q) results.emplace_back(data.data() + q * dimension, k, false);
            std::vector<double> latency(queries);
            std::vector<ErrorCode> status(queries, ErrorCode::Undefined);
            std::atomic<int> next{0};
            std::vector<std::thread> workers;
            const auto begin = std::chrono::steady_clock::now();
            for (int t = 0; t < threads; ++t) {
                workers.emplace_back([&] {
                    int q;
                    while ((q = next.fetch_add(1)) < queries) {
                        SPANN::SearchStats stats;
                        const auto start = std::chrono::steady_clock::now();
                        status[q] = spann->SearchIndex(results[q], &stats);
                        latency[q] = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
                    }
                });
            }
            for (auto& worker : workers) worker.join();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            int hits = 0, idHits = 0;
            double mean = 0;
            for (int q = 0; q < queries; ++q) {
                if (status[q] != ErrorCode::Success) throw std::runtime_error("SearchIndex failed");
                mean += latency[q];
                std::set<SizeType> seen;
                for (int j = 0; j < k; ++j) {
                    const auto* result = results[q].GetResult(j);
                    if (result->VID < 0 || result->VID >= expectedCount || !seen.insert(result->VID).second) {
                        throw std::runtime_error("Invalid or duplicate search result");
                    }
                    if (j && results[q].GetResult(j - 1)->Dist > result->Dist) {
                        throw std::runtime_error("Search results not sorted");
                    }
                }
                for (int j = 0; j < k; ++j) {
                    const int offset = (phase * queries + q) * k + j;
                    bool matched = false, idMatched = false;
                    for (int l = 0; l < k; ++l) {
                        const auto* result = results[q].GetResult(l);
                        idMatched |= result->VID == truthIDs[offset];
                        matched |= result->VID == truthIDs[offset]
                            || std::fabs(truthDistances[offset] - result->Dist)
                                <= 1e-4f * (std::fabs(truthDistances[offset]) + 1e-4f);
                    }
                    hits += matched;
                    idHits += idMatched;
                }
            }
            std::sort(latency.begin(), latency.end());
            if (round) output << ',';
            output << "{\"recallAt5\":" << hits / 1000.0 << ",\"idRecallAt5\":" << idHits / 1000.0
                   << ",\"meanMs\":" << mean / queries << ",\"p95Ms\":" << latency[190]
                   << ",\"p99Ms\":" << latency[198] << ",\"qps\":" << queries / seconds << ",\"results\":[";
            for (int q = 0; q < queries; ++q) {
                if (q) output << ',';
                output << '[';
                for (int j = 0; j < k; ++j) {
                    if (j) output << ',';
                    const auto* result = results[q].GetResult(j);
                    output << "{\"id\":" << result->VID << ",\"distance\":" << result->Dist << '}';
                }
                output << ']';
            }
            output << "]}";
            output.flush();
            std::cout << "Round " << round + 1 << " recall@5=" << hits / 1000.0
                      << " ID recall@5=" << idHits / 1000.0 << " meanMs=" << mean / queries << '\n';
        }
        output << "]}\n";
        if (!output) throw std::runtime_error("Failed writing output");
    } catch (const std::exception& error) {
        std::cerr << "Query probe failed: " << error.what() << '\n';
        return 1;
    }
}
