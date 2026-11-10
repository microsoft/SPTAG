#include "inc/Core/Common/DistanceUtils.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int dimension = 128;
constexpr int k = 5;
using Neighbor = std::pair<float, std::int32_t>;
using Heap = std::array<Neighbor, k>;

struct Vectors {
    std::int32_t count;
    std::vector<std::uint8_t> data;
};

Vectors load(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    std::int32_t count, dim;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    stream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (count <= 0 || dim != dimension ||
        std::filesystem::file_size(path) != 8 + std::uint64_t(count) * dimension) {
        throw std::runtime_error("Invalid UInt8/128D input: " + path);
    }
    Vectors result{count, std::vector<std::uint8_t>(std::size_t(count) * dimension)};
    stream.read(reinterpret_cast<char*>(result.data.data()), result.data.size());
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 8) {
            throw std::runtime_error("Usage: exact_truth BASE QUERY INITIAL INSERT_BATCH BATCHES THREADS OUTPUT");
        }
        const auto vectors = load(argv[1]);
        const auto queries = load(argv[2]);
        const int initial = std::stoi(argv[3]);
        const int insertBatch = std::stoi(argv[4]);
        const int batches = std::stoi(argv[5]);
        const int threads = std::stoi(argv[6]);
        const std::string output = argv[7];
        if (initial < k || insertBatch <= 0 || batches < 1 || threads < 1 ||
            std::int64_t(initial) + std::int64_t(insertBatch) * batches != vectors.count ||
            std::filesystem::exists(output) || std::filesystem::exists(output + ".partial")) {
            throw std::runtime_error("Invalid phase sizes/threads, or output already exists");
        }
        SPTAG::COMMON::DistanceCalcReturn<std::uint8_t> distance =
            &SPTAG::COMMON::DistanceUtils::ComputeL2Distance<std::uint8_t>;
        if (__builtin_cpu_supports("avx2")) {
            distance = &SPTAG::COMMON::DistanceUtils::ComputeL2Distance_AVX;
        }
        const std::size_t perPhase = std::size_t(queries.count) * k;
        const std::size_t total = std::size_t(batches + 1) * perPhase;
        std::vector<std::int32_t> ids(total);
        std::vector<float> distances(total);
        std::vector<Heap> heaps(queries.count);
        for (auto& heap : heaps) {
            heap.fill({std::numeric_limits<float>::max(), -1});
        }
        const auto started = std::chrono::steady_clock::now();
        int begin = 0;
        for (int phase = 0; phase <= batches; ++phase) {
            const int end = initial + phase * insertBatch;
            std::atomic<int> next{0};
            std::atomic<int> completed{0};
            std::vector<std::thread> workers;
            std::cout << "Phase " << phase << ": exact scan of new IDs [" << begin << ", " << end
                      << "), queries=" << queries.count << ", threads=" << threads << std::endl;
            for (int t = 0; t < threads; ++t) {
                workers.emplace_back([&] {
                    for (int q = next.fetch_add(1); q < queries.count; q = next.fetch_add(1)) {
                        auto& heap = heaps[q];
                        const auto* target = queries.data.data() + std::size_t(q) * dimension;
                        for (int id = begin; id < end; ++id) {
                            const Neighbor candidate{
                                distance(target, vectors.data.data() + std::size_t(id) * dimension, dimension), id};
                            if (candidate < heap.front()) {
                                std::pop_heap(heap.begin(), heap.end());
                                heap.back() = candidate;
                                std::push_heap(heap.begin(), heap.end());
                            }
                        }
                        auto sorted = heap;
                        std::sort(sorted.begin(), sorted.end());
                        for (int rank = 0; rank < k; ++rank) {
                            const auto offset = std::size_t(phase) * perPhase + std::size_t(q) * k + rank;
                            ids[offset] = sorted[rank].second;
                            distances[offset] = sorted[rank].first;
                        }
                        const int count = completed.fetch_add(1) + 1;
                        if (phase == 0 && count % 1000 == 0) {
                            std::cout << "Initial exact scan: " << count << "/" << queries.count << std::endl;
                        }
                    }
                });
            }
            for (auto& worker : workers) worker.join();
            begin = end;
            std::cout << "Phase " << phase << " completed, elapsed="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                      << " seconds" << std::endl;
        }
        if (2 * total / k > std::size_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("Truth header count exceeds int32");
        }
        const std::int32_t rows = 2 * total / k;
        const std::int32_t columns = k;
        std::ofstream stream(output + ".partial", std::ios::binary);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        stream.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
        stream.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(ids[0]));
        stream.write(reinterpret_cast<const char*>(distances.data()), distances.size() * sizeof(distances[0]));
        stream.close();
        std::filesystem::rename(output + ".partial", output);
        std::cout << "Saved exact phase-specific truth: " << output << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
