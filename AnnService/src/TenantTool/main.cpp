// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/TenantIndexManager.h"
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace SPTAG;

namespace
{
bool ReadVectors(const char *path, VectorValueType type, ByteArray &data, SizeType &count, DimensionType &dimension)
{
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    input.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
    SPANN::TenantIndexManager::Limits limits;
    if (!input || count <= 0 || count > limits.maxVectorsPerTenant || dimension <= 0 ||
        dimension > limits.maxDimension || GetValueTypeSize(type) == 0)
        return false;
    const uint64_t bytes = static_cast<uint64_t>(count) * dimension * GetValueTypeSize(type);
    input.seekg(0, std::ios::end);
    if (input.tellg() != static_cast<std::streamoff>(bytes + 8))
        return false;
    input.seekg(8);
    data = ByteArray::Alloc(bytes);
    input.read(reinterpret_cast<char *>(data.Data()), bytes);
    return static_cast<bool>(input);
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Trusted offline administration (stop server first):\n"
                  << "tenanttool ROOT create TENANT TYPE METRIC VECTORS.bin [BKT|KDT]\n"
                  << "tenanttool ROOT insert TENANT VECTORS.bin\n"
                  << "tenanttool ROOT search TENANT QUERY.bin K\n"
                  << "tenanttool ROOT delete-vector TENANT VID\n"
                  << "tenanttool ROOT delete-tenant TENANT\n"
                  << "tenanttool ROOT list|stats|checkpoint\n"
                  << "Backend environment: SPTAG_POSTING_STORAGE=ROCKSDBIO|FILEIO|TIKVIO,\n"
                  << "SPTAG_TIKV_PD=host:port, SPTAG_TIKV_PREFIX=owned-prefix\n"
                  << "Head-cache environment: SPTAG_RESIDENT_BYTES, SPTAG_RESIDENT_TENANTS,\n"
                  << "SPTAG_RESIDENT_WAIT_MILLISECONDS (unsigned integers)\n"
                  << "Checkpoint environment: SPTAG_AUTO_CHECKPOINT=0|1,\n"
                  << "SPTAG_CHECKPOINT_LOG_BYTES, SPTAG_CHECKPOINT_PRESSURE_BYTES,\n"
                  << "SPTAG_CHECKPOINT_MIN_INTERVAL_SECONDS (unsigned integers)\n"
                  << "Binary input: int32 count, int32 dimension, packed vectors. Query count must be 1.\n";
        return 2;
    }
    const std::string command = argv[2];
    bool valid = (command == "create" && (argc == 7 || argc == 8)) || (command == "insert" && argc == 5) ||
                 (command == "search" && argc == 6) || (command == "delete-vector" && argc == 5) ||
                 (command == "delete-tenant" && argc == 4) ||
                 ((command == "list" || command == "stats" || command == "checkpoint") && argc == 3);
    if (!valid)
    {
        std::cerr << "Invalid command or arguments\n";
        return 2;
    }
    std::unique_ptr<SPANN::TenantIndexManager> node;
    SPANN::TenantIndexManager::Limits limits;
    const auto readLimit = [](const char *name, auto &value) {
        const char *text = std::getenv(name);
        if (!text)
            return true;
        const char *end = text + std::char_traits<char>::length(text);
        const auto parsed = std::from_chars(text, end, value);
        return parsed.ec == std::errc() && parsed.ptr == end;
    };
    uint32_t automatic = limits.autoCheckpoint;
    if (!readLimit("SPTAG_AUTO_CHECKPOINT", automatic) || automatic > 1 ||
        !readLimit("SPTAG_CHECKPOINT_LOG_BYTES", limits.checkpointLogBytes) ||
        !readLimit("SPTAG_CHECKPOINT_PRESSURE_BYTES", limits.checkpointPressureBytes) ||
        !readLimit("SPTAG_CHECKPOINT_MIN_INTERVAL_SECONDS", limits.checkpointMinIntervalSeconds))
    {
        std::cerr << "Invalid checkpoint environment\n";
        return 2;
    }
    limits.autoCheckpoint = automatic != 0;
    if (!readLimit("SPTAG_RESIDENT_BYTES", limits.maxResidentBytes) ||
        !readLimit("SPTAG_RESIDENT_TENANTS", limits.maxResidentTenants) ||
        !readLimit("SPTAG_RESIDENT_WAIT_MILLISECONDS", limits.residentWaitMilliseconds))
    {
        std::cerr << "Invalid head-cache environment\n";
        return 2;
    }
    if (const char *storage = std::getenv("SPTAG_POSTING_STORAGE"))
        if (!Helper::Convert::ConvertStringTo<Storage>(storage, limits.posting.storage))
        {
            std::cerr << "Invalid SPTAG_POSTING_STORAGE\n";
            return 2;
        }
    if (const char *pd = std::getenv("SPTAG_TIKV_PD"))
        limits.posting.tikvPDAddresses = pd;
    if (const char *prefix = std::getenv("SPTAG_TIKV_PREFIX"))
        limits.posting.tikvKeyPrefix = prefix;
    auto status = SPANN::TenantIndexManager::Open(argv[1], limits, node);
    if (status != ErrorCode::Success)
        return 1;
    ByteArray vectors;
    SizeType count = 0;
    DimensionType dimension = 0;
    VectorValueType type = VectorValueType::Undefined;
    if (command == "create")
    {
        DistCalcMethod metric;
        IndexAlgoType headIndexAlgo = IndexAlgoType::Undefined;
        if (!Helper::Convert::ConvertStringTo<VectorValueType>(argv[4], type) ||
            !Helper::Convert::ConvertStringTo<DistCalcMethod>(argv[5], metric) ||
            (argc == 8 && (!Helper::Convert::ConvertStringTo<IndexAlgoType>(argv[7], headIndexAlgo) ||
                           (headIndexAlgo != IndexAlgoType::BKT && headIndexAlgo != IndexAlgoType::KDT))) ||
            !ReadVectors(argv[6], type, vectors, count, dimension))
        {
            std::cerr << "Invalid type, metric, head algorithm or vector file\n";
            return 2;
        }
        status = node->Create(argv[3], type, dimension, metric, vectors, count, headIndexAlgo);
    }
    else if (command == "insert" || command == "search")
    {
        DimensionType expected;
        status = node->Describe(argv[3], type, expected);
        if (status != ErrorCode::Success)
            return 1;
        if (!ReadVectors(argv[4], type, vectors, count, dimension) || dimension != expected)
        {
            std::cerr << "Invalid vector file or dimension\n";
            return 2;
        }
        if (command == "insert")
        {
            SizeType first;
            status = node->Insert(argv[3], vectors, count, first);
            if (status == ErrorCode::Success)
                std::cout << "first_id=" << first << "\n";
        }
        else
        {
            SizeType k;
            if (count != 1 || !Helper::Convert::ConvertStringTo<SizeType>(argv[5], k))
            {
                std::cerr << "Search requires one query and integer K\n";
                return 2;
            }
            std::vector<BasicResult> results;
            status = node->Search(argv[3], vectors, k, results, type);
            if (status == ErrorCode::Success)
                for (const auto &result : results)
                    std::cout << result.VID << " " << result.Dist << "\n";
        }
    }
    else if (command == "delete-vector")
    {
        SizeType id;
        if (!Helper::Convert::ConvertStringTo<SizeType>(argv[4], id))
        {
            std::cerr << "Invalid vector ID\n";
            return 2;
        }
        status = node->DeleteVector(argv[3], id);
    }
    else if (command == "delete-tenant")
        status = node->DeleteTenant(argv[3]);
    else if (command == "checkpoint")
        status = node->Checkpoint();
    else if (command == "stats")
    {
        SPANN::TenantIndexManager::NodeStats state;
        std::vector<SPANN::TenantIndexManager::TenantStats> tenants;
        node->GetStats(state, tenants);
        std::cout << "resident_tenants=" << state.residentTenants
                  << " max_resident_tenants=" << state.maxResidentTenants
                  << " resident_allocated_bytes=" << state.residentBytes << " reserved_bytes=" << state.reservedBytes
                  << " loading_bytes=" << state.loadingBytes << " retiring_bytes=" << state.retiringBytes
                  << " peak_charged_bytes=" << state.peakChargedBytes
                  << " max_resident_bytes=" << state.maxResidentBytes << " capacity_waits=" << state.capacityWaits
                  << " capacity_timeouts=" << state.capacityTimeouts
                  << " oversized_requests=" << state.oversizedRequests << " block_cache_bytes=" << state.blockCacheBytes
                  << " block_cache_capacity_bytes=" << state.blockCacheCapacityBytes
                  << " local_rocksdb_memtable_bytes=" << state.localRocksDBMemtableBytes
                  << " recovery_log_bytes=" << state.recoveryLogBytes << " auto_checkpoint=" << state.autoCheckpoint
                  << " checkpoint_in_progress=" << state.checkpointInProgress
                  << " auto_checkpoints=" << state.autoCheckpoints << "\n";
        const char *names[] = {"Unloaded", "Loading", "Resident", "Evicting"};
        for (const auto &tenant : tenants)
            std::cout << tenant.name << " index_id=" << tenant.indexID
                      << " state=" << names[static_cast<int>(tenant.state)]
                      << " snapshot_bytes=" << tenant.snapshotBytes << " allocated_bytes=" << tenant.allocatedBytes
                      << " reserved_bytes=" << tenant.reservedBytes << " dirty=" << tenant.dirty << "\n";
    }
    else
        for (const auto &tenant : node->List())
            std::cout << tenant << "\n";

    if (status != ErrorCode::Success)
    {
        std::cerr << "Operation failed: " << static_cast<int>(status) << "\n";
        return 1;
    }
    return node->Close() == ErrorCode::Success ? 0 : 1;
}
