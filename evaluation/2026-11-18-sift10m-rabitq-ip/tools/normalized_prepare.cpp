#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
constexpr int dim = 128, k = 5;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void fresh(const std::filesystem::path& path) {
    require(!std::filesystem::exists(path), "Refusing to overwrite an existing artifact");
}
std::ifstream input(const std::filesystem::path& path, int& count, int width) {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    int dimension;
    in.read(reinterpret_cast<char*>(&count), 4);
    in.read(reinterpret_cast<char*>(&dimension), 4);
    require(count > 0 && dimension == dim &&
            std::filesystem::file_size(path) == 8 + std::uint64_t(count) * dim * width,
            "Invalid vector file shape/length");
    return in;
}
void normalize(const std::filesystem::path& source, int count,
               const std::filesystem::path& output) {
    int available;
    auto in = input(source, available, 1);
    require(count > 0 && count <= available, "Invalid normalization prefix");
    fresh(output); fresh(output.string() + ".json"); fresh(output.string() + ".zeros.csv");
    std::ofstream out(output, std::ios::binary), zeros(output.string() + ".zeros.csv");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(reinterpret_cast<char*>(&count), 4);
    out.write(reinterpret_cast<const char*>(&dim), 4);
    zeros << "source_id\n";
    double minInput = std::numeric_limits<double>::infinity(), maxInput = 0;
    double minStored = std::numeric_limits<double>::infinity(), maxStored = 0, maxError = 0;
    std::uint64_t zeroCount = 0;
    for (int id = 0; id < count; ++id) {
        std::array<std::uint8_t, dim> raw;
        std::array<float, dim> values{};
        in.read(reinterpret_cast<char*>(raw.data()), raw.size());
        double squared = 0;
        for (auto v : raw) squared += double(v) * v;
        if (squared == 0) { ++zeroCount; zeros << id << '\n'; }
        else {
            const double norm = std::sqrt(squared);
            minInput = std::min(minInput, norm); maxInput = std::max(maxInput, norm);
            double storedSquared = 0;
            for (int d = 0; d < dim; ++d) {
                values[d] = static_cast<float>(double(raw[d]) / norm);
                require(std::isfinite(values[d]), "Nonfinite normalized coordinate");
                storedSquared += double(values[d]) * values[d];
            }
            const double storedNorm = std::sqrt(storedSquared);
            minStored = std::min(minStored, storedNorm); maxStored = std::max(maxStored, storedNorm);
            maxError = std::max(maxError, std::abs(storedNorm - 1));
            require(std::abs(storedNorm - 1) <= 1e-6, "Stored Float32 unit norm outside tolerance");
        }
        out.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
    }
    out.close(); zeros.close();
    if (zeroCount == std::uint64_t(count)) minInput = minStored = 0;
    std::ofstream report(output.string() + ".json");
    report << std::setprecision(17) << "{\"rows\":" << count << ",\"dimension\":128,"
           << "\"dtype\":\"little-endian Float32\",\"source_ids\":\"unchanged prefix order\","
           << "\"zero_rows\":" << zeroCount << ",\"nonfinite_input_coordinates\":0,"
           << "\"nonfinite_output_coordinates\":0,\"norm_tolerance\":1e-6,"
           << "\"nonzero_input_norm_min\":" << minInput << ",\"nonzero_input_norm_max\":" << maxInput
           << ",\"nonzero_stored_norm_min\":" << minStored << ",\"nonzero_stored_norm_max\":" << maxStored
           << ",\"max_stored_unit_norm_error\":" << maxError
           << ",\"recipe\":\"Sum UInt8 coordinate squares in FP64; sqrt in FP64; divide in FP64; "
              "round once to Float32. Keep zero rows as positive zeros; no row removal.\"}\n";
}
struct Vectors { int count; std::vector<float> data; };
Vectors load(const std::filesystem::path& path) {
    int count;
    auto in = input(path, count, 4);
    Vectors result{count, std::vector<float>(std::size_t(count) * dim)};
    in.read(reinterpret_cast<char*>(result.data.data()), result.data.size() * sizeof(float));
    for (auto value : result.data) require(std::isfinite(value), "Nonfinite Float32 input");
    return result;
}
void truth(const Vectors& base, const Vectors& queries, int threads,
           const std::filesystem::path& output, bool squaredL2 = false) {
    require(base.count >= k && threads > 0, "Invalid truth workload");
    fresh(output);
    std::vector<std::int32_t> ids(std::size_t(queries.count) * k);
    std::vector<float> distances(ids.size());
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) workers.emplace_back([&] {
        using Neighbor = std::pair<double, std::int32_t>;
        for (int q = next.fetch_add(1); q < queries.count; q = next.fetch_add(1)) {
            std::array<Neighbor, k> heap;
            heap.fill({std::numeric_limits<double>::infinity(), std::numeric_limits<int>::max()});
            const float* query = queries.data.data() + std::size_t(q) * dim;
            for (int id = 0; id < base.count; ++id) {
                const float* vector = base.data.data() + std::size_t(id) * dim;
                double distance;
                if (squaredL2) {
                    distance = 0;
                    for (int d = 0; d < dim; ++d) {
                        const double delta = double(query[d]) - vector[d];
                        distance += delta * delta;
                    }
                } else {
                    double dot = 0;
                    for (int d = 0; d < dim; ++d) dot += double(query[d]) * vector[d];
                    distance = 1.0 - dot;
                }
                Neighbor item{distance, id};
                if (item < heap.front()) {
                    std::pop_heap(heap.begin(), heap.end()); heap.back() = item;
                    std::push_heap(heap.begin(), heap.end());
                }
            }
            std::sort(heap.begin(), heap.end());
            for (int rank = 0; rank < k; ++rank) {
                ids[std::size_t(q) * k + rank] = heap[rank].second;
                distances[std::size_t(q) * k + rank] = static_cast<float>(heap[rank].first);
            }
        }
    });
    for (auto& worker : workers) worker.join();
    std::ofstream out(output, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    const std::int32_t rows = queries.count * 2;
    out.write(reinterpret_cast<const char*>(&rows), 4);
    out.write(reinterpret_cast<const char*>(&k), 4);
    out.write(reinterpret_cast<const char*>(ids.data()), ids.size() * 4);
    out.write(reinterpret_cast<const char*>(distances.data()), distances.size() * 4);
    out.close();
    std::cout << "Exact exhaustive " << (squaredL2 ? "squared L2" : "IP")
              << " truth from stored Float32; fixed-order FP64 accumulation, "
                 "source-ID tie break; base=" << base.count << " queries=" << queries.count << '\n';
}
}
int main(int argc, char** argv) {
    try {
        require(std::endian::native == std::endian::little && sizeof(float) == 4,
                "Little-endian IEEE Float32 host required");
        require(argc >= 2, "Commands: normalize RAW COUNT OUTPUT; truth BASE QUERY THREADS OUTPUT [InnerProduct|L2]");
        const std::string command(argv[1]);
        if (command == "normalize" && argc == 5) normalize(argv[2], std::stoi(argv[3]), argv[4]);
        else if (command == "truth" && (argc == 6 || argc == 7)) {
            const std::string metric = argc == 7 ? argv[6] : "InnerProduct";
            require(metric == "InnerProduct" || metric == "L2", "Unsupported truth metric");
            truth(load(argv[2]), load(argv[3]), std::stoi(argv[4]), argv[5], metric == "L2");
        }
        else throw std::runtime_error("Invalid command/arguments");
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
