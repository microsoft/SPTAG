// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/PostingBackend.h"
#include "inc/Core/SPANN/TenantIndexManager.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sys/resource.h>
#include <thread>

using namespace SPTAG;
using SPANN::SharedRocksDB;
namespace fs = std::filesystem;
namespace
{
void Require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}
void Check(ErrorCode code)
{
    Require(code == ErrorCode::Success, "SPTAG status " + std::to_string(int(code)));
}
struct Reader
{
    const std::string &data;
    size_t position = 0;
    uint64_t Number()
    {
        uint64_t value;
        Require(position <= data.size() && data.size() - position >= 8 &&
                    SharedRocksDB::DecodeNumber(data.substr(position, 8), value), "Invalid binary number");
        position += 8;
        return value;
    }
    std::string String()
    {
        auto size = Number();
        Require(size <= data.size() - position, "Invalid binary string");
        auto value = data.substr(position, size);
        position += size;
        return value;
    }
    void End() { Require(position == data.size(), "Trailing binary data"); }
};
struct Tenant
{
    std::string name;
    uint64_t id;
};
std::vector<Tenant> Catalog(const std::shared_ptr<SharedRocksDB> &state)
{
    std::string data;
    Check(state->Get("checkpoint_meta", "catalog", data));
    Reader reader{data};
    Require(reader.Number() == 1, "Unsupported catalog");
    const auto count = reader.Number();
    Require(count > 0 && count <= 128, "Expected between 1 and 128 tenants");
    std::vector<Tenant> result;
    for (uint64_t i = 0; i < count; ++i)
    {
        Tenant tenant{reader.String(), reader.Number()};
        Require(reader.Number() == uint64_t(VectorValueType::UInt8) && reader.Number() == 128 &&
                    reader.Number() == uint64_t(DistCalcMethod::L2), "Expected UInt8/128/L2");
        reader.Number();
        result.push_back(std::move(tenant));
    }
    reader.End();
    return result;
}
SPANN::PostingBackendOptions Options(const std::string &backend, const std::string &pd, const std::string &prefix)
{
    SPANN::PostingBackendOptions result;
    if (backend == "rocksdb")
        result.storage = Storage::ROCKSDBIO;
    else if (backend == "fileio")
        result.storage = Storage::FILEIO;
    else if (backend == "tikv")
        result.storage = Storage::TIKVIO;
    else
        throw std::runtime_error("Unknown backend");
    result.tikvPDAddresses = pd;
    result.tikvKeyPrefix = prefix;
    return result;
}
std::string Configuration(const SPANN::PostingBackendOptions &options)
{
    auto result = SharedRocksDB::EncodeNumber(1) + SharedRocksDB::EncodeNumber(uint64_t(options.storage));
    if (options.storage == Storage::TIKVIO)
        for (const auto &value : {options.tikvPDAddresses, options.tikvKeyPrefix})
            result += SharedRocksDB::EncodeNumber(value.size()) + value;
    return result;
}
std::shared_ptr<SharedRocksDB> OpenState(const fs::path &path)
{
    std::shared_ptr<SharedRocksDB> state;
    Check(SharedRocksDB::Open(path.string(), {"postings", "head_snapshots", "checkpoint_meta"}, state,
                              false, std::shared_ptr<rocksdb::Cache>(rocksdb::NewLRUCache(3ULL << 30))));
    return state;
}
std::shared_ptr<SPANN::PostingBackend> OpenOwner(const SPANN::PostingBackendOptions &options,
                                              const fs::path &work, const fs::path &checkpoint,
                                              const std::string &identity, uint64_t workID)
{
    std::shared_ptr<SPANN::PostingBackend> owner;
    if (options.storage == Storage::FILEIO)
        Check(SPANN::OpenFilePostingBackend(work.string(), checkpoint.string(), owner));
    else
        Check(SPANN::OpenTiKVPostingBackend(options, identity, workID, checkpoint.string(), owner));
    return owner;
}
uint64_t Hash(const std::string &value)
{
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : value)
        hash = (hash ^ byte) * 1099511628211ULL;
    return hash;
}
std::ofstream Output(const fs::path &path)
{
    std::ofstream output(path);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << std::setprecision(17);
    return output;
}
void PrepareClone(const fs::path &root, const SPANN::PostingBackendOptions &options)
{
    std::shared_ptr<SharedRocksDB> recovery;
    Check(SharedRocksDB::Open((root / "recovery").string(),
                             {"recovery_log", "catalog", "recovery_meta"}, recovery, false,
                             std::shared_ptr<rocksdb::Cache>(rocksdb::NewLRUCache(64ULL << 20))));
    std::string current;
    uint64_t checkpoint;
    Check(recovery->Get("recovery_meta", "current", current));
    Require(SharedRocksDB::DecodeNumber(current, checkpoint), "Invalid checkpoint");
    auto state = OpenState(root / ("checkpoint-" + std::to_string(checkpoint)));
    Check(state->Put("checkpoint_meta", "posting_backend", Configuration(options)));
    Check(recovery->Put("recovery_meta", "posting_backend", Configuration(options)));
    std::random_device random;
    std::string identity;
    for (int i = 0; i < 4; ++i)
        identity += SharedRocksDB::EncodeNumber(random());
    Check(recovery->Put("recovery_meta", "node_identity", identity));
    Output(root / "namespace-identity").write(identity.data(), identity.size());
    Check(state->FlushWAL());
    Check(recovery->FlushWAL());
}
void Convert(const fs::path &source, const fs::path &destination, const SPANN::PostingBackendOptions &options)
{
    Require(!fs::exists(destination), "Conversion destination must be new");
    fs::create_directories(destination);
    // Never open a source DB: RocksDB open may write logs, even for an immutable fixture.
    fs::copy(source / "recovery", destination / "recovery", fs::copy_options::recursive);
    std::shared_ptr<SharedRocksDB> recovery;
    Check(SharedRocksDB::Open((destination / "recovery").string(),
                              {"recovery_log", "catalog", "recovery_meta"}, recovery, false,
                              std::shared_ptr<rocksdb::Cache>(rocksdb::NewLRUCache(64 << 20))));
    Check(recovery->Scan("recovery_log", "", [](const std::string &, const std::string &) {
        throw std::runtime_error("Frozen fixture has uncheckpointed recovery records");
        return ErrorCode::Fail;
    }));
    std::string current;
    uint64_t checkpoint;
    Check(recovery->Get("recovery_meta", "current", current));
    Require(SharedRocksDB::DecodeNumber(current, checkpoint), "Invalid published pointer");
    auto statePath = destination / ("checkpoint-" + std::to_string(checkpoint));
    fs::copy(source / statePath.filename(), statePath, fs::copy_options::recursive);
    auto state = OpenState(statePath);
    auto tenants = Catalog(state);
    std::random_device random;
    std::string identity;
    for (int i = 0; i < 4; ++i)
        identity += SharedRocksDB::EncodeNumber(random());
    Check(recovery->Put("recovery_meta", "node_identity", identity));
    auto owner = options.storage == Storage::ROCKSDBIO ? nullptr :
        OpenOwner(options, destination / "import-postings", "", identity, 0);
    if (options.storage == Storage::TIKVIO)
        Output(destination / "namespace-identity").write(identity.data(), identity.size());
    auto report = Output(destination / "postings.csv");
    report << "tenant,index,key,bytes,fnv1a64\n";
    for (const auto &tenant : tenants)
    {
        auto prefix = SharedRocksDB::Namespace(tenant.name, tenant.id);
        auto view = owner ? owner->View(tenant.name, tenant.id) : state->PostingView(tenant.name, tenant.id);
        Check(state->Scan("postings", prefix, [&](const std::string &key, const std::string &value) {
            uint64_t number;
            Require(SharedRocksDB::DecodeNumber(key.substr(prefix.size()), number) &&
                        number <= uint64_t(MaxSize), "Invalid posting key");
            if (owner)
            {
                Check(view->Put(SizeType(number), value, std::chrono::microseconds::zero(), nullptr));
                std::string readback;
                Check(view->Get(SizeType(number), &readback, std::chrono::microseconds::zero(), nullptr));
                Require(readback == value, "Imported posting differs");
            }
            report << tenant.name << "," << tenant.id << "," << number << "," << value.size()
                   << "," << Hash(value) << "\n";
            return ErrorCode::Success;
        }));
    }
    report.close();
    if (owner)
    {
        Check(owner->Checkpoint((statePath / "posting-store").string()));
        // The new immutable image, not obsolete source postings, supplies all reads.
        Check(state->DeleteRange("postings", std::string(1, '\0'), std::string(1, '\xff')));
    }
    Check(state->Put("checkpoint_meta", "posting_backend", Configuration(options)));
    Check(recovery->Put("recovery_meta", "posting_backend", Configuration(options)));
    Check(recovery->Put("recovery_meta", "format", "SPTAG-Tenant-3"));
    Check(state->FlushWAL());
    Check(recovery->FlushWAL());
}
void Materialize(const std::shared_ptr<SharedRocksDB> &state, const Tenant &tenant, const fs::path &directory)
{
    auto prefix = SharedRocksDB::Namespace(tenant.name, tenant.id);
    std::string manifest;
    Check(state->Get("head_snapshots", prefix + "manifest", manifest));
    Reader reader{manifest};
    const auto count = reader.Number();
    Require(count <= 1000, "Invalid manifest");
    for (uint64_t i = 0; i < count; ++i)
    {
        const auto file = reader.String();
        const auto size = reader.Number();
        fs::path relative(file);
        Require(!relative.empty() && !relative.is_absolute(), "Unsafe snapshot path");
        for (const auto &component : relative)
            Require(component != ".." && component != ".", "Unsafe snapshot component");
        auto path = directory / relative;
        fs::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        for (uint64_t offset = 0, chunk = 0; offset < size; ++chunk)
        {
            std::string data;
            auto key = prefix + SharedRocksDB::EncodeNumber(file.size()) + file +
                       SharedRocksDB::EncodeNumber(chunk);
            Check(state->Get("head_snapshots", key, data));
            Require(data.size() == std::min(uint64_t(1 << 20), size - offset), "Invalid snapshot chunk");
            output.write(data.data(), data.size());
            offset += data.size();
        }
    }
    reader.End();
}
struct Engine
{
    std::unique_ptr<SPANN::TenantIndexManager> manager;
    std::shared_ptr<SharedRocksDB> state;
    std::shared_ptr<SPANN::PostingBackend> owner;
    std::vector<Tenant> tenants;
    std::vector<std::shared_ptr<VectorIndex>> indexes;
    void Open(const std::string &mode, const fs::path &root, const fs::path &output,
              const SPANN::PostingBackendOptions &options)
    {
        if (mode == "managed")
        {
            SPANN::TenantIndexManager::Limits limits;
            limits.maxTenants = limits.maxResidentTenants = 64;
            limits.posting = options;
            Check(SPANN::TenantIndexManager::Open(root.string(), limits, manager));
            for (const auto &name : manager->List())
                tenants.push_back({name, 0});
        }
        else
        {
            Require(mode == "direct", "Expected direct or managed");
            // Direct adapter control deliberately bypasses manager search/lifecycle.
            fs::path checkpoint;
            for (const auto &entry : fs::directory_iterator(root))
                if (entry.path().filename().string().rfind("checkpoint-", 0) == 0)
                {
                    Require(checkpoint.empty(), "Direct input needs one published checkpoint");
                    checkpoint = entry.path();
                }
            Require(!checkpoint.empty(), "Missing checkpoint");
            fs::copy(checkpoint, output / "direct-state", fs::copy_options::recursive);
            state = OpenState(output / "direct-state");
            tenants = Catalog(state);
            if (options.storage != Storage::ROCKSDBIO)
            {
                std::string identity;
                std::random_device random;
                for (int i = 0; i < 4; ++i)
                    identity += SharedRocksDB::EncodeNumber(random());
                owner = OpenOwner(options, output / "direct-postings", checkpoint / "posting-store", identity, 1);
                if (options.storage == Storage::TIKVIO)
                    Output(output / "namespace-identity").write(identity.data(), identity.size());
            }
            for (const auto &tenant : tenants)
            {
                auto directory = output / ("index-" + std::to_string(tenant.id));
                Materialize(state, tenant, directory);
                std::shared_ptr<VectorIndex> index;
                Check(VectorIndex::LoadIndex(directory.string(), index,
                    owner ? owner->View(tenant.name, tenant.id) : state->PostingView(tenant.name, tenant.id)));
                Require(dynamic_cast<SPANN::Index<uint8_t> *>(index.get()) != nullptr,
                        "Expected UInt8 SPANN snapshot");
                indexes.push_back(std::move(index));
            }
        }
        Require(tenants.size() == 1 || tenants.size() == 32, "Unexpected tenant count");
    }
};
struct IO
{
    uint64_t read = 0, write = 0, rchar = 0, wchar = 0;
    static IO Read()
    {
        IO result;
        std::ifstream input("/proc/self/io");
        std::string key;
        uint64_t value;
        while (input >> key >> value)
        {
            if (key == "read_bytes:") result.read = value;
            if (key == "write_bytes:") result.write = value;
            if (key == "rchar:") result.rchar = value;
            if (key == "wchar:") result.wchar = value;
        }
        Require(input.eof(), "Cannot read process I/O");
        return result;
    }
};
void Benchmark(const std::string &mode, const fs::path &root, const fs::path &queries,
               const fs::path &output, const SPANN::PostingBackendOptions &options, int expectedCount = 200)
{
    Require(!fs::exists(output), "Benchmark output must be new");
    fs::create_directories(output);
    std::ifstream input(queries, std::ios::binary);
    int32_t count, dimension;
    input.read(reinterpret_cast<char *>(&count), 4);
    input.read(reinterpret_cast<char *>(&dimension), 4);
    Require(input && count == expectedCount && dimension == 128 &&
                fs::file_size(queries) == uint64_t(8 + expectedCount * 128),
            "Expected exact " + std::to_string(expectedCount) + "-query UInt8/128 input");
    auto data = ByteArray::Alloc(count * 128);
    input.read(reinterpret_cast<char *>(data.Data()), data.Length());
    Require(bool(input), "Truncated queries");
    Engine engine;
    engine.Open(mode, root, output, options);
    if (expectedCount == 10000)
    {
        Output(output / "restore-ready") << "All restore/load completed; waiting for external stability gate\n";
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(30);
        while (!fs::exists(output / "measurement-go"))
        {
            Require(std::chrono::steady_clock::now() < deadline, "Stability gate timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    auto batches = Output(output / "batches.csv");
    auto raw = Output(output / "queries.csv");
    auto answers = Output(output / "answers.csv");
    batches << "pass,phase,round,queries,threads,elapsed_us,elapsed_s_float,qps_float,mean_ms,p50_ms,p95_ms,p99_ms,process_read_bytes,process_write_bytes,rchar,wchar,cpu_seconds,maxrss_kib\n";
    raw << "pass,query,tenant,status,ms\n";
    answers << "pass,query,tenant,rank,id,distance_bits\n";
    for (int pass = 0; pass < 4; ++pass)
    {
        std::vector<QueryResult> results(count);
        std::vector<std::vector<BasicResult>> managedResults(count);
        std::vector<ErrorCode> statuses(count, ErrorCode::Fail);
        std::vector<float> latencies(count);
        std::vector<std::exception_ptr> failures(16);
        for (int q = 0; q < count; ++q)
            results[q] = QueryResult(data.Data() + q * 128, 5, false);
        std::atomic_size_t sent(0);
        std::vector<std::thread> threads;
        threads.reserve(16);
        if (expectedCount == 10000)
            Output(output / ("pass-" + std::to_string(pass) + "-start")) << "ready\n";
        const auto ioBefore = IO::Read();
        rusage before{}, after{};
        Require(getrusage(RUSAGE_SELF, &before) == 0, "getrusage failed");
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();
        for (int worker = 0; worker < 16; ++worker)
            threads.emplace_back([&, worker] {
                try
                {
                    size_t q;
                    while ((q = sent.fetch_add(1)) < size_t(count))
                    {
                        const auto tenant = q % engine.tenants.size();
                        const auto t1 = Clock::now();
                        if (engine.manager)
                            statuses[q] = engine.manager->Search(engine.tenants[tenant].name,
                                ByteArray(data.Data() + q * 128, 128, false), 5, managedResults[q],
                                VectorValueType::UInt8, 128);
                        else
                            statuses[q] = dynamic_cast<SPANN::Index<uint8_t> *>(engine.indexes[tenant].get())
                                              ->SearchIndex(results[q]);
                        const auto t2 = Clock::now();
                        latencies[q] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
                    }
                }
                catch (...) { failures[worker] = std::current_exception(); }
            });
        for (auto &thread : threads)
            thread.join();
        const auto end = Clock::now();
        Require(getrusage(RUSAGE_SELF, &after) == 0, "getrusage failed");
        const auto ioAfter = IO::Read();
        for (const auto &failure : failures)
            if (failure) std::rethrow_exception(failure);
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        const float seconds = elapsed / 1000000.0f;
        Require(elapsed > 0, "Invalid batch duration");
        float mean = 0;
        for (int q = 0; q < count; ++q)
        {
            Check(statuses[q]);
            Require(std::isfinite(latencies[q]) && latencies[q] >= 0, "Invalid latency");
            const auto &tenant = engine.tenants[q % engine.tenants.size()].name;
            raw << pass << "," << q << "," << tenant << "," << int(statuses[q]) << "," << latencies[q] << "\n";
            mean += latencies[q];
            Require(!engine.manager || managedResults[q].size() == 5, "Invalid result count");
            for (int rank = 0; rank < 5; ++rank)
            {
                const auto &result = engine.manager ? managedResults[q][rank] : *results[q].GetResult(rank);
                Require(result.VID >= 0 && std::isfinite(result.Dist), "Invalid result");
                uint32_t bits;
                std::memcpy(&bits, &result.Dist, sizeof(bits));
                answers << pass << "," << q << "," << tenant << "," << rank << "," << result.VID << "," << bits << "\n";
            }
        }
        std::sort(latencies.begin(), latencies.end());
        auto cpu = [](const timeval &time) { return time.tv_sec + time.tv_usec * 1e-6; };
        batches << pass << "," << (pass % 2 ? "measured" : "primer") << "," << pass / 2 + 1
                << "," << count << ",16," << elapsed << "," << seconds << "," << count / seconds << ","
                << mean / count << "," << latencies[count / 2] << "," << latencies[count * 95 / 100]
                << "," << latencies[count * 99 / 100]
                << "," << ioAfter.read - ioBefore.read << "," << ioAfter.write - ioBefore.write
                << "," << ioAfter.rchar - ioBefore.rchar << "," << ioAfter.wchar - ioBefore.wchar
                << "," << cpu(after.ru_utime) + cpu(after.ru_stime) - cpu(before.ru_utime) - cpu(before.ru_stime)
                << "," << after.ru_maxrss << "\n";
        batches.flush(); raw.flush(); answers.flush();
        if (expectedCount == 10000)
            Output(output / ("pass-" + std::to_string(pass) + "-done")) << "validated\n";
        std::cout << "pass=" << pass << " qps=" << count / seconds << std::endl;
    }
}

void Matrix(int argc, char **argv)
{
    Require(argc == 14, "matrix ROOT QUERIES OUTPUT BACKEND PD PREFIX TRACE WORKLOAD BUDGET OPS SEED THREADS");
    const fs::path root(argv[2]), output(argv[4]);
    const auto options = Options(argv[5], argv[6], argv[7]);
    const std::string trace(argv[8]), workload(argv[9]);
    const auto budget = std::stoull(argv[10]), operations = std::stoull(argv[11]);
    const auto seed = std::stoull(argv[12]);
    const int workers = std::stoi(argv[13]);
    Require((trace == "random" || trace == "hot" || trace == "cyclic") &&
                (workload == "search" || workload == "mixed") &&
                budget > 0 && operations >= 128 && operations <= 1000000 &&
                workers >= 1 && workers <= 64 && !fs::exists(output),
            "Invalid matrix configuration or existing output");
    fs::create_directories(output);
    std::ifstream input(argv[3], std::ios::binary);
    int32_t count = 0, dimension = 0;
    input.read(reinterpret_cast<char *>(&count), 4);
    input.read(reinterpret_cast<char *>(&dimension), 4);
    Require(input && count > 0 && dimension == 128 &&
                fs::file_size(argv[3]) == uint64_t(8) + uint64_t(count) * 128,
            "Invalid UInt8/128 query input");
    auto data = ByteArray::Alloc(uint64_t(count) * 128);
    input.read(reinterpret_cast<char *>(data.Data()), data.Length());
    Require(bool(input), "Truncated queries");
    SPANN::TenantIndexManager::Limits limits;
    limits.maxTenants = 128;
    limits.maxResidentTenants = 0;
    limits.maxResidentBytes = budget;
    limits.cacheBytes = 3ULL << 30;
    limits.autoCheckpoint = false;
    limits.posting = options;
    std::unique_ptr<SPANN::TenantIndexManager> manager;
    Check(SPANN::TenantIndexManager::Open(root.string(), limits, manager));
    const auto names = manager->List();
    Require(!names.empty() && names.size() <= 128, "Invalid tenant count");
    auto catalog = Output(output / "tenants.csv");
    catalog << "tenant,name\n";
    for (size_t tenant = 0; tenant < names.size(); ++tenant)
        catalog << tenant << "," << names[tenant] << "\n";
    auto cache = Output(output / "cache.csv");
    cache << "phase,resident_tenants,resident_bytes,reserved_bytes,peak_charged_bytes,"
             "hits,misses,loads,evictions,capacity_waits,capacity_timeouts,oversized_requests\n";
    auto stats = [&](int phase) {
        SPANN::TenantIndexManager::NodeStats node;
        std::vector<SPANN::TenantIndexManager::TenantStats> tenants;
        manager->GetStats(node, tenants);
        uint64_t hits = 0, misses = 0, loads = 0, evictions = 0;
        for (const auto &tenant : tenants)
        {
            hits += tenant.cacheHits;
            misses += tenant.cacheMisses;
            loads += tenant.loads;
            evictions += tenant.evictions;
        }
        cache << phase << "," << node.residentTenants << "," << node.residentBytes << ","
              << node.reservedBytes << "," << node.peakChargedBytes << "," << hits << ","
              << misses << "," << loads << "," << evictions << "," << node.capacityWaits << ","
              << node.capacityTimeouts << "," << node.oversizedRequests << "\n";
        cache.flush();
        Require(!node.failed && node.peakChargedBytes <= budget, "Node failure or accounting limit exceeded");
    };
    for (const auto &name : names)
    {
        std::vector<BasicResult> results;
        Check(manager->Search(name, ByteArray(data.Data(), 128, false), 5, results,
                              VectorValueType::UInt8, 128));
    }
    stats(-1);
    struct Operation
    {
        size_t tenant = 0, query = 0;
        bool write = false;
        ErrorCode status = ErrorCode::Fail;
        SizeType id = -1;
        double ms = 0;
        std::array<uint8_t, 128> vector{};
        std::vector<BasicResult> results;
    };
    auto raw = Output(output / "operations.csv");
    auto answers = Output(output / "answers.csv");
    auto batches = Output(output / "batches.csv");
    raw << "phase,operation,tenant,query,write,status,id,ms\n";
    answers << "phase,operation,tenant,query,rank,id,distance_bits\n";
    batches << "phase,operations,reads,ack_inserts,capacity_rejections,errors,seconds,ops_per_s,"
               "p50_ms,p95_ms,p99_ms,cpu_seconds,read_bytes,write_bytes,maxrss_kib\n";
    std::vector<std::pair<size_t, SizeType>> acknowledged;
    bool valid = true;
    for (int phase = 0; phase < 2; ++phase)
    {
        const auto size = phase ? operations : std::min<uint64_t>(256, operations);
        std::vector<Operation> work(size);
        std::mt19937_64 random(seed + phase);
        const size_t hot = std::max<size_t>(1, names.size() / 5);
        for (size_t index = 0; index < size; ++index)
        {
            auto &operation = work[index];
            operation.tenant = trace == "cyclic" ? index % names.size() : random() % names.size();
            if (trace == "hot" && names.size() > hot)
                operation.tenant = random() % 10 < 8 ? random() % hot :
                    hot + random() % (names.size() - hot);
            operation.query = random() % count;
            operation.write = phase && workload == "mixed" && random() % 10 == 0;
            if (operation.write)
            {
                std::memcpy(operation.vector.data(), data.Data() + operation.query * 128, 128);
                operation.vector[0] = uint8_t(index);
                operation.vector[1] = uint8_t(index >> 8);
            }
        }
        std::atomic<size_t> next{0};
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> failures(workers);
        const auto ioBefore = IO::Read();
        rusage before{}, after{};
        Require(getrusage(RUSAGE_SELF, &before) == 0, "getrusage failed");
        const auto start = std::chrono::steady_clock::now();
        for (int worker = 0; worker < workers; ++worker)
            threads.emplace_back([&, worker] {
                try
                {
                    size_t index;
                    while ((index = next.fetch_add(1)) < work.size())
                    {
                        auto &operation = work[index];
                        const auto begin = std::chrono::steady_clock::now();
                        if (operation.write)
                            operation.status = manager->Insert(names[operation.tenant],
                                ByteArray(operation.vector.data(), 128, false), 1, operation.id,
                                VectorValueType::UInt8, 128);
                        else
                            operation.status = manager->Search(names[operation.tenant],
                                ByteArray(data.Data() + operation.query * 128, 128, false), 5,
                                operation.results, VectorValueType::UInt8, 128);
                        operation.ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin).count();
                    }
                }
                catch (...) { failures[worker] = std::current_exception(); }
            });
        for (auto &thread : threads)
            thread.join();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Require(getrusage(RUSAGE_SELF, &after) == 0, "getrusage failed");
        const auto ioAfter = IO::Read();
        uint64_t reads = 0, inserts = 0, rejections = 0, errors = 0;
        std::vector<double> latencies;
        for (size_t index = 0; index < work.size(); ++index)
        {
            const auto &operation = work[index];
            raw << phase << "," << index << "," << operation.tenant << "," << operation.query << ","
                << operation.write << "," << int(operation.status) << "," << operation.id << ","
                << operation.ms << "\n";
            latencies.push_back(operation.ms);
            if (operation.status == ErrorCode::Success)
            {
                if (operation.write)
                {
                    ++inserts;
                    acknowledged.emplace_back(operation.tenant, operation.id);
                }
                else
                {
                    ++reads;
                    Require(operation.results.size() == 5, "Invalid result count");
                    float previous = -1;
                    for (size_t rank = 0; rank < operation.results.size(); ++rank)
                    {
                        const auto &result = operation.results[rank];
                        Require(result.VID >= 0 && std::isfinite(result.Dist) && result.Dist >= previous,
                                "Invalid or unsorted result");
                        previous = result.Dist;
                        uint32_t bits;
                        std::memcpy(&bits, &result.Dist, sizeof(bits));
                        answers << phase << "," << index << "," << operation.tenant << ","
                                << operation.query << "," << rank << "," << result.VID << "," << bits << "\n";
                    }
                }
            }
            else if (operation.status == ErrorCode::CapacityExceeded ||
                     operation.status == ErrorCode::CapacityTimeout)
            {
                ++rejections;
                Require(operation.id == -1, "Rejected operation consumed a VID");
            }
            else
                ++errors;
        }
        std::sort(latencies.begin(), latencies.end());
        auto cpu = [](const timeval &time) { return time.tv_sec + time.tv_usec * 1e-6; };
        batches << phase << "," << work.size() << "," << reads << "," << inserts << "," << rejections
                << "," << errors << "," << seconds << "," << work.size() / seconds << ","
                << latencies[latencies.size() / 2] << "," << latencies[latencies.size() * 95 / 100]
                << "," << latencies[latencies.size() * 99 / 100] << ","
                << cpu(after.ru_utime) + cpu(after.ru_stime) - cpu(before.ru_utime) - cpu(before.ru_stime)
                << "," << ioAfter.read - ioBefore.read << "," << ioAfter.write - ioBefore.write
                << "," << after.ru_maxrss << "\n";
        batches.flush(); raw.flush(); answers.flush();
        for (const auto &failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        stats(phase);
        valid = valid && errors == 0;
    }
    for (size_t tenant = 0; tenant < names.size(); ++tenant)
    {
        const auto query = tenant % size_t(count);
        std::vector<BasicResult> results;
        const auto status = manager->Search(names[tenant],
            ByteArray(data.Data() + query * 128, 128, false), 5, results, VectorValueType::UInt8, 128);
        raw << "2," << tenant << "," << tenant << "," << query << ",0," << int(status) << ",-1,0\n";
        Check(status);
        Require(results.size() == 5, "Invalid final accuracy result");
        for (size_t rank = 0; rank < results.size(); ++rank)
        {
            uint32_t bits;
            std::memcpy(&bits, &results[rank].Dist, sizeof(bits));
            answers << "2," << tenant << "," << tenant << "," << query << "," << rank << ","
                    << results[rank].VID << "," << bits << "\n";
        }
    }
    raw.flush(); answers.flush();
    stats(2);
    Check(manager->Close());
    manager.reset();
    // This verifies acknowledged VID membership after graceful reopen, not crash durability.
    if (!acknowledged.empty())
    {
        Check(SPANN::TenantIndexManager::Open(root.string(), limits, manager));
        auto verified = Output(output / "ack-membership.csv");
        verified << "tenant,id,status\n";
        for (const auto &[tenant, id] : acknowledged)
        {
            const auto status = manager->DeleteVector(names[tenant], id);
            verified << tenant << "," << id << "," << int(status) << "\n";
            Check(status);
        }
        Check(manager->Close());
        manager.reset();
    }
    Require(valid, "Unexpected API errors; raw failures retained");
    Output(output / "completed") << "Normal teardown and acknowledged VID membership verified\n";
}
}
int main(int argc, char **argv)
{
    try
    {
        if (argc == 7 && std::string(argv[1]) == "convert")
            Convert(argv[2], argv[3], Options(argv[4], argv[5], argv[6]));
        else if (argc == 6 && std::string(argv[1]) == "prepare")
            PrepareClone(argv[2], Options(argv[3], argv[4], argv[5]));
        else if (argc == 9 && std::string(argv[1]) == "run")
            Benchmark(argv[2], argv[3], argv[4], argv[5], Options(argv[6], argv[7], argv[8]));
        else if (argc == 9 && std::string(argv[1]) == "run10k")
            Benchmark(argv[2], argv[3], argv[4], argv[5], Options(argv[6], argv[7], argv[8]), 10000);
        else if (argc == 14 && std::string(argv[1]) == "matrix")
            Matrix(argc, argv);
        else
            throw std::runtime_error("postingbackendperf convert SOURCE NEW_ROOT BACKEND PD PREFIX\n"
                                     "postingbackendperf prepare OWNED_CLONE BACKEND PD PREFIX\n"
                                     "postingbackendperf run|run10k direct|managed ROOT QUERIES NEW_OUTPUT BACKEND PD PREFIX\n"
                                     "run10k waits for NEW_OUTPUT/measurement-go after writing restore-ready\n"
                                     "postingbackendperf matrix ROOT QUERIES OUTPUT BACKEND PD PREFIX "
                                     "random|hot|cyclic search|mixed BUDGET_BYTES OPS SEED THREADS");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
