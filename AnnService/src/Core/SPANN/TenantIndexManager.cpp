// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/TenantIndexManager.h"
#include "inc/Core/SPANN/UpdateContext.h"
#ifdef MULTITENANT
#include <fstream>
#include <map>
#include <random>
#include <set>

namespace SPTAG::SPANN
{
namespace
{
const std::vector<std::string> StateFamilies{"postings", "head_snapshots", "checkpoint_meta"};
const std::vector<std::string> RecoveryFamilies{"recovery_log", "catalog", "recovery_meta"};
constexpr size_t ChunkSize = 1 << 20;

ErrorCode Failure(const char *message, ErrorCode status = ErrorCode::Fail)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Tenant manager: %s\n", message);
    return status;
}

ErrorCode MakeDirectory(const std::string &path)
{
    std::error_code error;
    std::filesystem::create_directories(path, error);
    return error ? Failure(error.message().c_str(), ErrorCode::DiskIOFail) : ErrorCode::Success;
}

void Number(std::string &output, std::uint64_t number)
{
    output += SharedRocksDB::EncodeNumber(number);
}
void String(std::string &output, const std::string &value)
{
    Number(output, value.size());
    output += value;
}

class Reader
{
  public:
    explicit Reader(const std::string &data) : m_data(data)
    {
    }
    bool Number(std::uint64_t &value)
    {
        if (m_data.size() - m_pos < 8)
            return false;
        bool result = SharedRocksDB::DecodeNumber(m_data.substr(m_pos, 8), value);
        m_pos += 8;
        return result;
    }
    bool String(std::string &value)
    {
        std::uint64_t size;
        if (!Number(size) || size > m_data.size() - m_pos)
            return false;
        value.assign(m_data, m_pos, static_cast<size_t>(size));
        m_pos += static_cast<size_t>(size);
        return true;
    }
    bool End() const
    {
        return m_pos == m_data.size();
    }

  private:
    const std::string &m_data;
    size_t m_pos = 0;
};

std::string BackendConfiguration(const PostingBackendOptions &options)
{
    std::string result;
    Number(result, 1);
    Number(result, static_cast<uint64_t>(options.storage));
    if (options.storage == Storage::TIKVIO)
    {
        String(result, options.tikvPDAddresses);
        String(result, options.tikvKeyPrefix);
    }
    return result;
}

class RocksPostingBackend final : public PostingBackend
{
  public:
    explicit RocksPostingBackend(std::shared_ptr<SharedRocksDB> state) : m_state(std::move(state))
    {
    }
    std::shared_ptr<Helper::KeyValueIO> View(const std::string &tenant, uint64_t index) override
    {
        return m_state->PostingView(tenant, index);
    }
    ErrorCode Checkpoint(const std::string &) override
    {
        return ErrorCode::Success;
    }
    ErrorCode DeleteIndex(const std::string &tenant, uint64_t index) override
    {
        auto prefix = SharedRocksDB::Namespace(tenant, index);
        return m_state->DeleteRange("postings", prefix, SharedRocksDB::PrefixEnd(prefix));
    }

  private:
    std::shared_ptr<SharedRocksDB> m_state;
};

std::string FileKey(const std::string &prefix, const std::string &file, std::uint64_t chunk)
{
    std::string key = prefix;
    String(key, file);
    Number(key, chunk);
    return key;
}

bool SafeRelativePath(const std::string &file)
{
    std::filesystem::path path(file);
    if (file.empty() || file.find('\0') != std::string::npos || path.is_absolute())
        return false;
    for (const auto &component : path)
        if (component == ".." || component == ".")
            return false;
    return true;
}

ErrorCode DecodeSnapshotManifest(const std::string &manifest, std::map<std::string, uint64_t> &files)
{
    Reader reader(manifest);
    uint64_t count;
    if (!reader.Number(count) || count > 1000)
        return Failure("Invalid snapshot manifest");
    for (uint64_t i = 0; i < count; ++i)
    {
        std::string file;
        uint64_t size;
        if (!reader.String(file) || !SafeRelativePath(file) || !reader.Number(size) ||
            !files.emplace(file, size).second)
            return Failure("Invalid snapshot component");
    }
    return reader.End() ? ErrorCode::Success : Failure("Trailing snapshot manifest data");
}
} // namespace

TenantIndexManager::TenantIndexManager(std::string root, Limits limits)
    : m_root(std::move(root)), m_limits(limits), m_cache(rocksdb::NewLRUCache(limits.cacheBytes))
{
}

TenantIndexManager::~TenantIndexManager()
{
    m_closed.store(true);
    StopCheckpointWorker();
    m_directory.store(nullptr);
    m_readers.Synchronize();
    m_directoryOwner.reset();
    // Destruction does not publish a checkpoint: the last acknowledged log must remain replayable.
    for (auto &entry : m_tenants)
        if (entry.second->index)
            entry.second->index->DrainUpdates();
}

ErrorCode TenantIndexManager::Open(const std::string &root, const Limits &limits,
                                   std::unique_ptr<TenantIndexManager> &result)
{
    result.reset();
#ifndef TIKV
    if (limits.posting.storage == Storage::TIKVIO)
        return Failure("TiKV postings require a build with both ROCKSDB and TIKV", ErrorCode::FailedParseValue);
#endif
    if (root.empty() || !limits.maxTenants || limits.maxVectorsPerTenant < 64 || limits.maxDimension <= 0 ||
        limits.maxResults <= 0 || !limits.cacheBytes || !limits.maxResidentTenants ||
        (limits.autoCheckpoint &&
         (!limits.checkpointLogBytes || limits.checkpointPressureBytes < limits.checkpointLogBytes)) ||
        (limits.posting.storage != Storage::ROCKSDBIO && limits.posting.storage != Storage::FILEIO &&
         limits.posting.storage != Storage::TIKVIO) ||
        (limits.posting.storage == Storage::TIKVIO &&
         (limits.posting.tikvPDAddresses.empty() || limits.posting.tikvKeyPrefix.empty())))
        return Failure("Invalid node limits", ErrorCode::FailedParseValue);
    auto manager = std::unique_ptr<TenantIndexManager>(new TenantIndexManager(root, limits));
    auto status = manager->Initialize();
    if (status != ErrorCode::Success)
        return status;
    manager->PublishDirectory();
    if (limits.autoCheckpoint)
    {
        try
        {
            manager->m_checkpointThread = std::thread([owner = manager.get()] { owner->CheckpointWorker(); });
        }
        catch (...)
        {
            return Failure("Cannot start automatic checkpoint worker");
        }
    }
    result = std::move(manager);
    return ErrorCode::Success;
}

ErrorCode TenantIndexManager::AllocateID(std::uint64_t &id)
{
    std::lock_guard<std::mutex> allocation(m_idMutex);
    std::string value;
    auto status = m_recovery->Get("recovery_meta", "next_id", value);
    id = 1;
    if (status == ErrorCode::Success)
    {
        if (!SharedRocksDB::DecodeNumber(value, id) || id == 0 || id == UINT64_MAX)
            return Failure("Invalid or exhausted node ID counter");
    }
    else if (status != ErrorCode::Key_NotFound)
        return status;
    return m_recovery->Put("recovery_meta", "next_id", SharedRocksDB::EncodeNumber(id + 1));
}

ErrorCode TenantIndexManager::Initialize()
{
    auto status = MakeDirectory(m_root);
    if (status != ErrorCode::Success)
        return status;
    std::error_code error;
    const bool existing = std::filesystem::exists(m_root + "/recovery", error);
    if (error)
        return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
    if (!existing)
    {
        bool empty = std::filesystem::is_empty(m_root, error);
        if (error)
            return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
        if (!empty)
            return Failure("A new tenant node requires an empty, dedicated root directory");
    }
    // Recovery DB's LOCK excludes a second node owner before any state is restored.
    status = SharedRocksDB::Open(m_root + "/recovery", RecoveryFamilies, m_recovery, !existing, m_cache);
    if (status != ErrorCode::Success)
        return status;
    std::string format;
    status = m_recovery->Get("recovery_meta", "format", format);
    if (status == ErrorCode::Key_NotFound)
    {
        status = m_recovery->Put("recovery_meta", "format", "SPTAG-Tenant-3");
    }
    else if (status == ErrorCode::Success && format != "SPTAG-Tenant-2" && format != "SPTAG-Tenant-3")
        return Failure("Unsupported node format", ErrorCode::FailedParseValue);
    if (format == "SPTAG-Tenant-2" && m_limits.posting.storage != Storage::ROCKSDBIO)
        return Failure("Legacy node format supports only RocksDB postings", ErrorCode::FailedParseValue);
    if (status != ErrorCode::Success)
        return status;
    const auto backend = BackendConfiguration(m_limits.posting);
    std::string storedBackend;
    status = m_recovery->Get("recovery_meta", "posting_backend", storedBackend);
    if (status == ErrorCode::Key_NotFound)
    {
        std::string current;
        const auto currentStatus = m_recovery->Get("recovery_meta", "current", current);
        if (currentStatus != ErrorCode::Success && currentStatus != ErrorCode::Key_NotFound)
            return currentStatus;
        if (currentStatus == ErrorCode::Success && m_limits.posting.storage != Storage::ROCKSDBIO)
            return Failure("Legacy RocksDB node requires explicit offline backend migration");
        status = m_recovery->Put("recovery_meta", "posting_backend", backend);
    }
    else if (status == ErrorCode::Success && storedBackend != backend)
        return Failure("Posting backend configuration does not match the node", ErrorCode::FailedParseValue);
    if (status != ErrorCode::Success)
        return status;
    status = m_recovery->Get("recovery_meta", "node_identity", m_nodeIdentity);
    if (status == ErrorCode::Key_NotFound)
    {
        std::random_device random;
        for (int i = 0; i < 4; ++i)
            m_nodeIdentity += SharedRocksDB::EncodeNumber(random());
        status = m_recovery->Put("recovery_meta", "node_identity", m_nodeIdentity);
    }
    if (status != ErrorCode::Success || m_nodeIdentity.size() != 32)
        return status == ErrorCode::Success ? Failure("Invalid node identity") : status;
    std::uint64_t workID;
    if ((status = AllocateID(workID)) != ErrorCode::Success)
        return status;
    m_workID = workID;
    m_work = m_root + "/work-" + std::to_string(workID);
    bool collision = std::filesystem::exists(m_work, error);
    if (error)
        return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
    if (collision)
        return Failure("Work directory already exists");
    if ((status = m_recovery->Put("recovery_meta", "artifact/" + SharedRocksDB::EncodeNumber(workID), "work")) !=
        ErrorCode::Success)
        return status;
    if ((status = MakeDirectory(m_work)) != ErrorCode::Success)
        return status;

    std::string current;
    status = m_recovery->Get("recovery_meta", "current", current);
    const bool fresh = status == ErrorCode::Key_NotFound;
    if (!fresh && status != ErrorCode::Success)
        return status;
    if (!fresh)
    {
        std::uint64_t checkpoint;
        if (!SharedRocksDB::DecodeNumber(current, checkpoint))
            return Failure("Invalid checkpoint pointer");
        std::error_code error;
        std::filesystem::copy(m_root + "/checkpoint-" + std::to_string(checkpoint), m_work + "/state",
                              std::filesystem::copy_options::recursive, error);
        if (error)
            return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
    }
    status = SharedRocksDB::Open(m_work + "/state", StateFamilies, m_state, fresh, m_cache);
    if (status != ErrorCode::Success)
        return status;
    if (!fresh)
    {
        std::string checkpointBackend;
        status = m_state->Get("checkpoint_meta", "posting_backend", checkpointBackend);
        if (status == ErrorCode::Key_NotFound && m_limits.posting.storage == Storage::ROCKSDBIO)
            status = ErrorCode::Success; // Legacy checkpoints contain only the RocksDB posting CF.
        else if (status == ErrorCode::Success && checkpointBackend != backend)
            return Failure("Posting checkpoint does not match its published backend");
        if (status != ErrorCode::Success)
            return status;
    }
    if (m_limits.posting.storage == Storage::ROCKSDBIO)
        m_postings = std::make_shared<RocksPostingBackend>(m_state);
    else if (m_limits.posting.storage == Storage::FILEIO)
        status = OpenFilePostingBackend(m_work + "/postings", fresh ? "" : m_work + "/state/posting-store", m_postings);
    else
        status = OpenTiKVPostingBackend(m_limits.posting, m_nodeIdentity, workID,
                                        fresh ? "" : m_work + "/state/posting-store", m_postings);
    if (status != ErrorCode::Success)
        return status;
    if (fresh)
        return CheckpointLocked();

    std::string catalog;
    if ((status = m_state->Get("checkpoint_meta", "catalog", catalog)) != ErrorCode::Success)
        return status;
    std::string publishedCatalog;
    if ((status = m_recovery->Get("catalog", "tenants", publishedCatalog)) != ErrorCode::Success)
        return status;
    if (publishedCatalog != catalog)
        return Failure("Published catalog does not match checkpoint");
    if ((status = RestoreCatalog(catalog)) != ErrorCode::Success)
        return status;
    for (auto &entry : m_tenants)
    {
        if ((status = Load(*entry.second)) != ErrorCode::Success)
            return status;
        if ((status = Replay(*entry.second)) != ErrorCode::Success)
            return status;
        if ((status = Save(*entry.second)) != ErrorCode::Success)
            return status;
        entry.second->index.reset();
    }
    return CollectGarbage();
}

ErrorCode TenantIndexManager::Ready() const
{
    return (m_closed || m_failed.load()) ? Failure("Node closed or failed; reopen to recover") : ErrorCode::Success;
}

void TenantIndexManager::PublishDirectory()
{
    auto directory = std::make_unique<TenantDirectory>(m_tenants);
    m_directory.store(directory.get());
    m_readers.Synchronize();
    m_directoryOwner = std::move(directory);
}

int64_t TenantIndexManager::LastAccess(const Tenant &tenant)
{
    return (std::max)(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tenant.lastAccess.time_since_epoch()).count(),
        tenant.lastReadAccess.load(std::memory_order_relaxed));
}

TenantIndexManager::ReadLease::~ReadLease()
{
    if (tenant)
    {
        auto now =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        auto previous = tenant->lastReadAccess.load(std::memory_order_relaxed);
        while (previous < now &&
               !tenant->lastReadAccess.compare_exchange_weak(previous, now, std::memory_order_relaxed))
        {
        }
    }
}

ErrorCode TenantIndexManager::Poison(ErrorCode status)
{
    if (status != ErrorCode::Success)
    {
        {
            std::lock_guard<std::mutex> checkpoint(m_checkpointMutex);
            m_failed.store(true);
        }
        m_checkpointChanged.notify_all();
        m_cacheChanged.notify_all();
        Failure("State mutation failed; node fenced until recovery", status);
    }
    return status;
}

std::string TenantIndexManager::Directory(const Tenant &tenant) const
{
    return m_work + "/index-" + std::to_string(tenant.id);
}

bool TenantIndexManager::ValidVectors(const Tenant &tenant, const ByteArray &vectors, SizeType count) const
{
    if (count <= 0 || count > m_limits.maxVectorsPerTenant || !vectors.Data())
        return false;
    auto bytes = static_cast<std::uint64_t>(count) * tenant.dimension * GetValueTypeSize(tenant.type);
    if (bytes != vectors.Length())
        return false;
    if (tenant.type == VectorValueType::Float)
    {
        for (size_t i = 0; i < bytes / sizeof(float); ++i)
        {
            float value;
            memcpy(&value, vectors.Data() + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value))
                return false;
        }
    }
    return true;
}

std::string TenantIndexManager::SerializeCatalog() const
{
    std::string result;
    Number(result, 1);
    Number(result, m_tenants.size());
    for (const auto &entry : m_tenants)
    {
        const auto &tenant = *entry.second;
        String(result, tenant.name);
        Number(result, tenant.id);
        Number(result, static_cast<std::uint64_t>(tenant.type));
        Number(result, tenant.dimension);
        Number(result, static_cast<std::uint64_t>(tenant.metric));
        Number(result, tenant.sequence);
    }
    return result;
}

ErrorCode TenantIndexManager::RestoreCatalog(const std::string &data)
{
    Reader reader(data);
    std::uint64_t version, count;
    if (!reader.Number(version) || version != 1 || !reader.Number(count) || count > m_limits.maxTenants)
        return Failure("Invalid catalog header or tenant limit");
    std::set<std::uint64_t> ids;
    for (std::uint64_t i = 0; i < count; ++i)
    {
        auto tenant = std::make_shared<Tenant>();
        std::uint64_t type, dimension, metric;
        if (!reader.String(tenant->name) || tenant->name.empty() || tenant->name.size() > 1024 ||
            !reader.Number(tenant->id) || !tenant->id || !ids.insert(tenant->id).second || !reader.Number(type) ||
            !reader.Number(dimension) || !reader.Number(metric) || !reader.Number(tenant->sequence) || dimension == 0 ||
            dimension > static_cast<uint64_t>(m_limits.maxDimension))
            return Failure("Invalid tenant catalog record");
        tenant->type = static_cast<VectorValueType>(type);
        tenant->dimension = static_cast<DimensionType>(dimension);
        tenant->metric = static_cast<DistCalcMethod>(metric);
        tenant->savedSequence = tenant->sequence;
        tenant->hasSnapshot = true;
        if (GetValueTypeSize(tenant->type) == 0 ||
            (tenant->metric != DistCalcMethod::L2 && tenant->metric != DistCalcMethod::Cosine) ||
            !m_tenants.emplace(tenant->name, tenant).second)
            return Failure("Unsupported or duplicate tenant catalog record");
    }
    return reader.End() ? ErrorCode::Success : Failure("Trailing catalog bytes");
}

ErrorCode TenantIndexManager::Create(const std::string &name, VectorValueType type, DimensionType dimension,
                                     DistCalcMethod metric, const ByteArray &vectors, SizeType count,
                                     IndexAlgoType headIndexAlgo, int postingLayers,
                                     const Helper::IniReader *configuration)
{
    std::lock_guard<std::mutex> management(m_management);
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    std::shared_lock<std::shared_mutex> updates(m_updates);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    if (name.empty() || name.size() > 1024 || m_tenants.count(name) || m_tenants.size() >= m_limits.maxTenants ||
        dimension <= 0 || dimension > m_limits.maxDimension || GetValueTypeSize(type) == 0 ||
        (metric != DistCalcMethod::L2 && metric != DistCalcMethod::Cosine) || count < 64 || postingLayers < 0 ||
        (headIndexAlgo != IndexAlgoType::Undefined && headIndexAlgo != IndexAlgoType::BKT &&
         headIndexAlgo != IndexAlgoType::KDT))
        return Failure("Invalid create request, duplicate tenant, or node capacity", ErrorCode::FailedParseValue);
    auto tenant = std::make_shared<Tenant>();
    tenant->name = name;
    tenant->type = type;
    tenant->dimension = dimension;
    tenant->metric = metric;
    if (!ValidVectors(*tenant, vectors, count))
        return Failure("Invalid seed vectors", ErrorCode::FailedParseValue);
    tenant->index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, type);
    if (!tenant->index)
        return Failure("Unsupported vector type", ErrorCode::FailedParseValue);
    if (configuration)
        for (const auto *section : {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"})
            for (const auto &parameter : configuration->GetParameters(section))
                if ((status = tenant->index->SetParameter(parameter.first.c_str(), parameter.second.c_str(),
                                                          section)) != ErrorCode::Success)
                    return status;
    if (postingLayers && tenant->index->SetParameter("Layers", std::to_string(postingLayers).c_str(),
                                                     "BuildSSDIndex") != ErrorCode::Success)
        return ErrorCode::FailedParseValue;
    if (headIndexAlgo != IndexAlgoType::Undefined &&
        tenant->index->SetParameter("IndexAlgoType", Helper::Convert::ConvertToString(headIndexAlgo).c_str(), "Base") !=
            ErrorCode::Success)
        return ErrorCode::FailedParseValue;
    SizeType maxID = 0;
    int layers = 0;
    IndexAlgoType configuredHead = IndexAlgoType::Undefined;
    if (!Helper::Convert::ConvertStringTo<SizeType>(tenant->index->GetParameter("MaxID", "BuildSSDIndex").c_str(),
                                                    maxID) ||
        !Helper::Convert::ConvertStringTo<int>(tenant->index->GetParameter("Layers", "BuildSSDIndex").c_str(),
                                               layers) ||
        !Helper::Convert::ConvertStringTo<IndexAlgoType>(tenant->index->GetParameter("IndexAlgoType", "Base").c_str(),
                                                         configuredHead) ||
        maxID <= 0 || count > maxID || layers <= 0 || layers > MaxSize / maxID ||
        (configuredHead != IndexAlgoType::BKT && configuredHead != IndexAlgoType::KDT))
        return Failure("Invalid head algorithm or posting layer ID ranges", ErrorCode::FailedParseValue);
    if (configuration)
    {
        for (const auto *key : {"HeadVectorIDs", "DeletedIDs", "HeadVectors", "HeadIndexFolder", "SSDIndex"})
            if (!SafeRelativePath(tenant->index->GetParameter(key, "Base")))
                return Failure("Index files must stay within the tenant directory", ErrorCode::FailedParseValue);
        for (const auto *key : {"FullDeletedIDFile", "KVFile", "SsdMappingFile", "SsdInfoFile", "ChecksumFile"})
            if (!SafeRelativePath(tenant->index->GetParameter(key, "BuildSSDIndex")))
                return Failure("Posting files must stay within the tenant directory", ErrorCode::FailedParseValue);
        auto head = VectorIndex::CreateInstance(configuredHead, type);
        for (const auto &parameter : configuration->GetParameters("BuildHead"))
            if (!Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), "isExecute"))
                head->SetParameter(parameter.first.c_str(), parameter.second.c_str());
        const auto files = head->GetIndexFiles();
        for (const auto &file : *files)
            if (!SafeRelativePath(file))
                return Failure("Head files must stay within the tenant directory", ErrorCode::FailedParseValue);
    }
    if ((status = AllocateID(tenant->id)) != ErrorCode::Success)
        return Poison(status);
    if ((status = MakeDirectory(Directory(*tenant))) != ErrorCode::Success)
        return Poison(status);
    const auto set = [&](const char *section, const char *key, const std::string &value) {
        return tenant->index->SetParameter(key, value.c_str(), section);
    };
    const std::vector<std::tuple<std::string, std::string, std::string>> parameters{
        {"Base", "ValueType", Helper::Convert::ConvertToString(type)},
        {"Base", "Dim", std::to_string(dimension)},
        {"Base", "DistCalcMethod", Helper::Convert::ConvertToString(metric)},
        {"Base", "IndexDirectory", Directory(*tenant)},
        {"Base", "DataCapacity", std::to_string(m_limits.maxVectorsPerTenant)},
        {"SelectHead", "isExecute", "true"},
        {"BuildHead", "isExecute", "true"},
        {"BuildHead", "DataCapacity", std::to_string(m_limits.maxVectorsPerTenant)},
        {"BuildHead", "DeletePercentageForRefine", "2"},
        {"BuildSSDIndex", "isExecute", "true"},
        {"BuildSSDIndex", "BuildSsdIndex", "true"},
        {"BuildSSDIndex", "TmpDir", Directory(*tenant)}};
    for (const auto &parameter : parameters)
    {
        status = set(std::get<0>(parameter).c_str(), std::get<1>(parameter).c_str(), std::get<2>(parameter));
        if (status != ErrorCode::Success)
            return Poison(status);
    }
    status = tenant->index->SetPostingStore(m_postings->View(name, tenant->id));
    if (status != ErrorCode::Success)
        return Poison(status);
    auto seed = ByteArray::Alloc(vectors.Length());
    memcpy(seed.Data(), vectors.Data(), vectors.Length());
    status = tenant->index->BuildIndex(seed.Data(), count, dimension);
    if (status != ErrorCode::Success)
        return Poison(status);
    if ((status = Save(*tenant)) != ErrorCode::Success)
        return Poison(status);
    tenant->index.reset();
    updates.unlock();
    barrier.unlock();
    std::unique_lock<std::shared_mutex> publish(m_barrier);
    if ((status = Ready()) != ErrorCode::Success)
        return status;
    m_tenants.emplace(name, tenant);
    status = Poison(CheckpointLocked());
    if (status == ErrorCode::Success)
        PublishDirectory();
    return status;
}

ErrorCode TenantIndexManager::Save(Tenant &tenant)
{
    if (!tenant.index)
        return ErrorCode::Success; // Unload already saved this exact, now immutable state.
    auto status = tenant.index->DrainUpdates();
    if (status != ErrorCode::Success)
        return status;
    if (tenant.hasSnapshot && tenant.sequence == tenant.savedSequence)
        return ErrorCode::Success;
    if ((status = tenant.index->SaveIndex(Directory(tenant), nullptr, true)) != ErrorCode::Success)
        return status;
    std::set<std::string> files{"indexloader.ini"};
    int layers = 0;
    if (!Helper::Convert::ConvertStringTo<int>(tenant.index->GetParameter("Layers", "BuildSSDIndex").c_str(), layers) ||
        layers < 1)
        return Failure("Invalid posting layer count in snapshot");
    const auto versionFile = tenant.index->GetParameter("DeletedIDs", "Base");
    for (int layer = 0; layer < layers; ++layer)
        files.insert(versionFile + "_" + std::to_string(layer));
    auto indexFiles = tenant.index->GetIndexFiles();
    for (const auto &file : *indexFiles)
        files.insert(file);
    const auto prefix = SharedRocksDB::Namespace(tenant.name, tenant.id);
    std::map<std::string, uint64_t> previousSizes;
    if (tenant.hasSnapshot)
    {
        std::string previousManifest;
        if ((status = m_state->Get("head_snapshots", prefix + "manifest", previousManifest)) != ErrorCode::Success ||
            (status = DecodeSnapshotManifest(previousManifest, previousSizes)) != ErrorCode::Success)
            return status;
    }
    const auto chunks = [](uint64_t bytes) { return bytes / ChunkSize + (bytes % ChunkSize != 0); };
    std::string manifest;
    std::string previous;
    uint64_t snapshotBytes = 0;
    Number(manifest, files.size());
    std::vector<char> buffer(ChunkSize);
    for (const auto &file : files)
    {
        if (!SafeRelativePath(file))
            return Failure("Unsafe snapshot file path");
        std::ifstream input(Directory(tenant) + "/" + file, std::ios::binary);
        if (!input)
            return Failure("Cannot open snapshot component", ErrorCode::FailedOpenFile);
        auto old = previousSizes.find(file);
        const auto oldChunks = old == previousSizes.end() ? 0 : chunks(old->second);
        std::uint64_t size = 0, chunk = 0;
        while (input)
        {
            input.read(buffer.data(), buffer.size());
            const auto bytes = input.gcount();
            if (bytes > 0)
            {
                const auto key = FileKey(prefix, file, chunk);
                // Snapshot comparison must not fill the cache with blocks unrelated to posting searches.
                status =
                    chunk < oldChunks ? m_state->Get("head_snapshots", key, previous, false) : ErrorCode::Key_NotFound;
                if (status != ErrorCode::Success && status != ErrorCode::Key_NotFound)
                    return status;
                if (status == ErrorCode::Key_NotFound || previous.size() != static_cast<size_t>(bytes) ||
                    memcmp(previous.data(), buffer.data(), static_cast<size_t>(bytes)) != 0)
                {
                    if ((status = m_state->Put("head_snapshots", key,
                                               std::string(buffer.data(), static_cast<size_t>(bytes)))) !=
                        ErrorCode::Success)
                        return status;
                }
                ++chunk;
                size += bytes;
            }
        }
        if (!input.eof())
            return Failure("Snapshot component read failed", ErrorCode::DiskIOFail);
        if (chunk < oldChunks &&
            (status = m_state->DeleteRange("head_snapshots", FileKey(prefix, file, chunk),
                                           FileKey(prefix, file, oldChunks))) != ErrorCode::Success)
            return status;
        previousSizes.erase(file);
        String(manifest, file);
        Number(manifest, size);
        snapshotBytes += size;
    }
    for (const auto &file : previousSizes)
        if (file.second &&
            (status = m_state->DeleteRange("head_snapshots", FileKey(prefix, file.first, 0),
                                           FileKey(prefix, file.first, chunks(file.second)))) != ErrorCode::Success)
            return status;
    status = m_state->Put("head_snapshots", prefix + "manifest", manifest);
    if (status == ErrorCode::Success)
    {
        std::lock_guard<std::mutex> cache(m_cacheMutex);
        tenant.savedSequence = tenant.sequence;
        tenant.hasSnapshot = true;
        tenant.filesReady = true;
        if (tenant.stats.state != Residency::Unloaded)
            m_residentBytes = m_residentBytes - tenant.stats.snapshotBytes + snapshotBytes;
        tenant.stats.snapshotBytes = snapshotBytes;
    }
    return status;
}

TenantIndexManager::Lease::~Lease()
{
    if (owner)
        owner->Release(*tenant);
}

void TenantIndexManager::Release(Tenant &tenant)
{
    std::lock_guard<std::mutex> cache(m_cacheMutex);
    --tenant.stats.activeRequests;
    tenant.lastAccess = std::chrono::steady_clock::now();
    if (!tenant.stats.activeRequests)
        m_cacheChanged.notify_all();
}

ErrorCode TenantIndexManager::Evict(Tenant &tenant, std::unique_lock<std::mutex> &cacheLock)
{
    tenant.stats.state = Residency::Evicting;
    tenant.readIndex.store(nullptr);
    const auto retired = m_readers.Advance();
    auto start = std::chrono::steady_clock::now();
    cacheLock.unlock();
    ErrorCode status;
    try
    {
        m_readers.Wait(retired);
        status = Save(tenant);
        if (status == ErrorCode::Success)
            tenant.index.reset();
    }
    catch (...)
    {
        status = ErrorCode::Fail;
    }
    cacheLock.lock();
    if (status == ErrorCode::Success)
    {
        --m_residentCount;
        m_residentBytes -= tenant.stats.snapshotBytes;
        tenant.stats.state = Residency::Unloaded;
        ++tenant.stats.evictions;
        tenant.stats.lastEvictionMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    else
    {
        tenant.stats.state = Residency::Resident;
        tenant.readIndex.store(tenant.index.get());
        Poison(status);
    }
    m_cacheChanged.notify_all();
    return status;
}

ErrorCode TenantIndexManager::Acquire(Tenant &tenant, Lease &lease, bool updatesHeld)
{
    const auto pin = [&] {
        ++tenant.stats.activeRequests;
        tenant.lastAccess = std::chrono::steady_clock::now();
        lease.owner = this;
        lease.tenant = &tenant;
    };
    {
        std::lock_guard<std::mutex> cache(m_cacheMutex);
        auto status = Ready();
        if (status != ErrorCode::Success)
            return status;
        if (tenant.stats.state == Residency::Resident)
        {
            ++tenant.stats.cacheHits;
            pin();
            return ErrorCode::Success;
        }
        ++tenant.stats.cacheMisses;
    }
    std::shared_lock<std::shared_mutex> updates(m_updates, std::defer_lock);
    if (!updatesHeld)
        updates.lock();
    std::unique_lock<std::mutex> cache(m_cacheMutex);
    for (;;)
    {
        auto status = Ready();
        if (status != ErrorCode::Success)
            return status;
        if (tenant.stats.state == Residency::Resident)
        {
            pin();
            return ErrorCode::Success;
        }
        if (tenant.stats.state != Residency::Unloaded)
        {
            m_cacheChanged.wait(cache);
            continue;
        }
        const auto charge = tenant.stats.snapshotBytes;
        const bool bytesFit =
            !m_limits.maxResidentBytes || m_residentCount == 0 ||
            (charge <= m_limits.maxResidentBytes && m_residentBytes <= m_limits.maxResidentBytes - charge);
        if (m_residentCount < m_limits.maxResidentTenants && bytesFit)
        {
            tenant.stats.state = Residency::Loading;
            ++m_residentCount;
            m_residentBytes += charge;
            auto start = std::chrono::steady_clock::now();
            cache.unlock();
            try
            {
                status = Load(tenant);
            }
            catch (...)
            {
                status = ErrorCode::Fail;
            }
            cache.lock();
            if (status != ErrorCode::Success)
            {
                tenant.index.reset();
                --m_residentCount;
                m_residentBytes -= tenant.stats.snapshotBytes;
                tenant.stats.state = Residency::Unloaded;
                m_cacheChanged.notify_all();
                return Poison(status);
            }
            tenant.stats.state = Residency::Resident;
            tenant.readIndex.store(tenant.index.get());
            ++tenant.stats.loads;
            tenant.stats.lastLoadMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            pin();
            m_cacheChanged.notify_all();
            return ErrorCode::Success;
        }
        Tenant *victim = nullptr;
        for (auto &entry : m_tenants)
        {
            auto &candidate = *entry.second;
            if (candidate.stats.state == Residency::Resident && !candidate.stats.activeRequests &&
                (!victim || LastAccess(candidate) < LastAccess(*victim)) && !m_readers.Active(&candidate))
                victim = &candidate;
        }
        if (victim)
        {
            if ((status = Evict(*victim, cache)) != ErrorCode::Success)
                return status;
        }
        else
            // Read-side epoch exits deliberately do not acquire this mutex or notify.
            m_cacheChanged.wait_for(cache, std::chrono::microseconds(50));
    }
}

ErrorCode TenantIndexManager::Load(Tenant &tenant)
{
    if (tenant.index)
        return ErrorCode::Success;
    uint64_t snapshotBytes = tenant.stats.snapshotBytes;
    if (!tenant.filesReady)
    {
        auto status = MakeDirectory(Directory(tenant));
        if (status != ErrorCode::Success)
            return status;
        auto prefix = SharedRocksDB::Namespace(tenant.name, tenant.id);
        std::string manifest;
        if ((status = m_state->Get("head_snapshots", prefix + "manifest", manifest)) != ErrorCode::Success)
            return status;
        std::map<std::string, uint64_t> files;
        if ((status = DecodeSnapshotManifest(manifest, files)) != ErrorCode::Success)
            return status;
        snapshotBytes = 0;
        for (const auto &component : files)
        {
            const auto &file = component.first;
            const auto size = component.second;
            snapshotBytes += size;
            auto path = std::filesystem::path(Directory(tenant)) / file;
            if ((status = MakeDirectory(path.parent_path().string())) != ErrorCode::Success)
                return status;
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output)
                return Failure("Cannot restore snapshot component", ErrorCode::FailedCreateFile);
            for (std::uint64_t offset = 0, chunk = 0; offset < size; ++chunk)
            {
                std::string data;
                if ((status = m_state->Get("head_snapshots", FileKey(prefix, file, chunk), data)) != ErrorCode::Success)
                    return status;
                auto expected = (std::min)(static_cast<std::uint64_t>(ChunkSize), size - offset);
                if (data.size() != expected)
                    return Failure("Truncated snapshot component");
                output.write(data.data(), data.size());
                offset += data.size();
            }
            output.close();
            if (!output)
                return Failure("Snapshot write failed", ErrorCode::DiskIOFail);
        }
    }
    auto status = VectorIndex::LoadIndex(Directory(tenant), tenant.index, m_postings->View(tenant.name, tenant.id));
    if (status != ErrorCode::Success)
    {
        tenant.index.reset();
        return status;
    }
    if (tenant.index->GetFeatureDim() != tenant.dimension || tenant.index->GetVectorValueType() != tenant.type)
        return Failure("Snapshot configuration does not match catalog");
    {
        std::lock_guard<std::mutex> cache(m_cacheMutex);
        if (tenant.stats.state != Residency::Unloaded)
            m_residentBytes = m_residentBytes - tenant.stats.snapshotBytes + snapshotBytes;
        tenant.stats.snapshotBytes = snapshotBytes;
        tenant.filesReady = true;
    }
    return ErrorCode::Success;
}

ErrorCode TenantIndexManager::CheckpointLocked()
try
{
    const auto hook = [&](const char *stage) {
        if (m_limits.checkpointHook)
            m_limits.checkpointHook(stage);
    };
    hook("before_snapshots");
    for (auto &entry : m_tenants)
    {
        auto status = Save(*entry.second);
        if (status != ErrorCode::Success)
            return status;
    }
    const auto catalog = SerializeCatalog();
    auto status = m_state->Put("checkpoint_meta", "catalog", catalog);
    if (status != ErrorCode::Success)
        return status;
    status = m_state->Put("checkpoint_meta", "posting_backend", BackendConfiguration(m_limits.posting));
    if (status != ErrorCode::Success)
        return status;
    std::uint64_t id;
    if ((status = AllocateID(id)) != ErrorCode::Success)
        return status;
    const auto path = m_root + "/checkpoint-" + std::to_string(id);
    std::error_code error;
    bool collision = std::filesystem::exists(path, error);
    if (error)
        return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
    if (collision)
        return Failure("Checkpoint directory already exists");
    if ((status = m_recovery->Put("recovery_meta", "artifact/" + SharedRocksDB::EncodeNumber(id), "checkpoint")) !=
        ErrorCode::Success)
        return status;
    if ((status = m_state->Checkpoint(path)) != ErrorCode::Success)
        return status;
    hook("after_state_snapshot");
    if ((status = m_postings->Checkpoint(path + "/posting-store")) != ErrorCode::Success)
        return status;
    hook("after_posting_snapshot");
    status = m_recovery->Write(
        {{"recovery_meta", "current", SharedRocksDB::EncodeNumber(id)}, {"catalog", "tenants", catalog}});
    if (status != ErrorCode::Success)
        return status;
    if ((status = m_recovery->FlushWAL()) != ErrorCode::Success)
        return status;
    hook("after_publish");
    for (const auto &entry : m_tenants)
    {
        auto prefix = SharedRocksDB::Namespace(entry.first, entry.second->id);
        status = m_recovery->DeleteRange("recovery_log", prefix,
                                         prefix + SharedRocksDB::EncodeNumber(entry.second->sequence + 1));
        if (status != ErrorCode::Success)
            return status;
    }
    hook("after_log_cleanup");
    status = CollectGarbage();
    if (status == ErrorCode::Success)
    {
        {
            std::lock_guard<std::mutex> checkpoint(m_checkpointMutex);
            m_recoveryLogBytes.store(0);
            m_lastCheckpoint = std::chrono::steady_clock::now();
        }
        m_checkpointChanged.notify_all();
    }
    return status;
}
catch (...)
{
    return Failure("Checkpoint preparation or publication threw an exception");
}

ErrorCode TenantIndexManager::CollectGarbage()
{
    std::set<std::string> active;
    for (const auto &entry : m_tenants)
        active.insert(SharedRocksDB::Namespace(entry.first, entry.second->id));
    std::string removed;
    auto status = m_recovery->Scan("recovery_log", "", [&](const std::string &key, const std::string &) {
        std::uint64_t length;
        if (key.size() < 25 || key[0] != '\1' || !SharedRocksDB::DecodeNumber(key.substr(1, 8), length) ||
            length == 0 || length > 1024 || key.size() != length + 25)
            return Failure("Invalid recovery log namespace");
        auto prefix = key.substr(0, key.size() - 8);
        if (active.count(prefix) || prefix == removed)
            return ErrorCode::Success;
        removed = prefix;
        return m_recovery->DeleteRange("recovery_log", prefix, SharedRocksDB::PrefixEnd(prefix));
    });
    if (status != ErrorCode::Success)
        return status;
    std::string current;
    status = m_recovery->Get("recovery_meta", "current", current);
    if (status != ErrorCode::Success)
        return status;
    std::uint64_t checkpoint;
    if (!SharedRocksDB::DecodeNumber(current, checkpoint))
        return Failure("Invalid checkpoint during cleanup");
    // Only delete explicitly registered artifacts; never enumerate/remove arbitrary root contents.
    return m_recovery->Scan("recovery_meta", "artifact/", [&](const std::string &key, const std::string &kind) {
        std::uint64_t id;
        if (!SharedRocksDB::DecodeNumber(key.substr(9), id) || (kind != "work" && kind != "checkpoint"))
            return Failure("Invalid artifact registry");
        if ((kind == "work" && id == m_workID) || (kind == "checkpoint" && id == checkpoint))
            return ErrorCode::Success;
        if (kind == "work" && (status = m_postings->RemoveWork(id)) != ErrorCode::Success)
            return status;
        auto path = std::filesystem::path(m_root) / (kind + "-" + std::to_string(id));
        std::error_code error;
        if (std::filesystem::is_symlink(path, error))
            return Failure("Refusing to remove symlink artifact");
        if (error && error != std::errc::no_such_file_or_directory)
            return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
        error.clear();
        std::filesystem::remove_all(path, error);
        if (error)
            return Failure(error.message().c_str(), ErrorCode::DiskIOFail);
        return m_recovery->Delete("recovery_meta", key);
    });
}

ErrorCode TenantIndexManager::Apply(Tenant &tenant, const std::string &record)
{
    Reader reader(record);
    std::uint64_t version, operation, id, count;
    std::string bytes;
    if (!reader.Number(version) || version != 1 || !reader.Number(operation) || !reader.Number(id) || id > MaxSize)
        return Failure("Invalid recovery record");
    ErrorCode status;
    if (operation == 1)
    {
        if (!reader.Number(count) || count == 0 || count > static_cast<uint64_t>(m_limits.maxVectorsPerTenant) ||
            !reader.String(bytes) || !reader.End() || tenant.index->GetNumSamples() != static_cast<SizeType>(id) ||
            id + count > static_cast<uint64_t>(m_limits.maxVectorsPerTenant))
            return Failure("Invalid insertion recovery record or VID sequence");
        ByteArray data = ByteArray::Alloc(bytes.size());
        memcpy(data.Data(), bytes.data(), bytes.size());
        if (!ValidVectors(tenant, data, static_cast<SizeType>(count)))
            return Failure("Invalid recovery vector size");
        SizeType first;
        status = tenant.index->ReserveVectorIDs(static_cast<SizeType>(count), m_limits.maxVectorsPerTenant, first);
        if (status != ErrorCode::Success || first != static_cast<SizeType>(id))
            return status == ErrorCode::Success ? Failure("Recovery VID reservation mismatch") : status;
        status = tenant.index->InsertAt(data.Data(), static_cast<SizeType>(count), tenant.dimension, first);
    }
    else if (operation == 2 && reader.End())
    {
        status = tenant.index->DeleteIndex(static_cast<SizeType>(id));
    }
    else
        return Failure("Unsupported recovery operation");
    if (status != ErrorCode::Success)
        return status;
    return tenant.index->DrainUpdates();
}

ErrorCode TenantIndexManager::Record(Tenant &tenant, const std::string &record)
{
    if (tenant.sequence == UINT64_MAX - 1)
        return Poison(Failure("Operation sequence exhausted"));
    const auto key =
        SharedRocksDB::Namespace(tenant.name, tenant.id) + SharedRocksDB::EncodeNumber(tenant.sequence + 1);
    auto status = m_recovery->Put("recovery_log", key, record);
    if (status != ErrorCode::Success)
        return Poison(status);
    ++tenant.sequence;
    {
        std::lock_guard<std::mutex> checkpoint(m_checkpointMutex);
        m_recoveryLogBytes.fetch_add(key.size() + record.size());
    }
    m_checkpointChanged.notify_all();
    return ErrorCode::Success;
}

ErrorCode TenantIndexManager::Replay(Tenant &tenant)
{
    const auto prefix = SharedRocksDB::Namespace(tenant.name, tenant.id);
    return m_recovery->Scan("recovery_log", prefix, [&](const std::string &key, const std::string &record) {
        m_recoveryLogBytes.fetch_add(key.size() + record.size());
        std::uint64_t sequence;
        if (!SharedRocksDB::DecodeNumber(key.substr(prefix.size()), sequence))
            return Failure("Invalid recovery sequence key");
        if (sequence <= tenant.sequence)
            return ErrorCode::Success;
        if (sequence != tenant.sequence + 1)
            return Failure("Gap in recovery sequence");
        auto status = Apply(tenant, record);
        if (status == ErrorCode::Success)
            tenant.sequence = sequence;
        return status;
    });
}

ErrorCode TenantIndexManager::Insert(const std::string &name, const ByteArray &vectors, SizeType count,
                                     SizeType &firstID, VectorValueType expectedType, DimensionType expectedDimension)
{
    auto admission = WaitForCheckpoint();
    if (admission != ErrorCode::Success)
        return admission;
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    std::shared_lock<std::shared_mutex> updates(m_updates);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    auto it = m_tenants.find(name);
    if (it == m_tenants.end())
        return Failure("Tenant not found", ErrorCode::Key_NotFound);
    auto &tenant = *it->second;
    if (!ValidVectors(tenant, vectors, count) ||
        (expectedType != VectorValueType::Undefined && expectedType != tenant.type) ||
        (expectedDimension != 0 && expectedDimension != tenant.dimension))
        return Failure("Invalid insertion data", ErrorCode::FailedParseValue);
    Lease lease;
    if ((status = Acquire(tenant, lease, true)) != ErrorCode::Success)
        return status;
    SizeType begin = -1;
    auto completion = std::make_shared<UpdateContext>();
    try
    {
        std::string record;
        record.reserve(vectors.Length() + 40);
        Number(record, 1);
        Number(record, 1);
        const auto vidOffset = record.size();
        Number(record, 0);
        Number(record, count);
        Number(record, vectors.Length());
        record.append(reinterpret_cast<const char *>(vectors.Data()), vectors.Length());
        {
            std::lock_guard<std::mutex> lock(tenant.submitMutex);
            if ((status = Ready()) != ErrorCode::Success)
                return status;
            if (static_cast<int64_t>(tenant.index->GetNumSamples()) + count > m_limits.maxVectorsPerTenant)
                return ErrorCode::MemoryOverFlow;
            try
            {
                status = tenant.index->ReserveVectorIDs(count, m_limits.maxVectorsPerTenant, begin);
                if (status != ErrorCode::Success)
                    return Poison(status);
                tenant.pendingInserts.emplace(begin, count);
                auto encoded = SharedRocksDB::EncodeNumber(begin);
                record.replace(vidOffset, encoded.size(), encoded);
                if ((status = Record(tenant, record)) != ErrorCode::Success)
                    return status;
            }
            catch (...)
            {
                // Fence before another submitter can record a VID after an unlogged reservation.
                return Poison(ErrorCode::Fail);
            }
        }
        UpdateContext::Scope scope(completion);
        status = tenant.index->InsertAt(vectors.Data(), count, tenant.dimension, begin);
    }
    catch (...)
    {
        status = ErrorCode::Fail;
    }
    if (status != ErrorCode::Success)
        Poison(status);
    if (!completion->Wait())
        status = ErrorCode::Fail;
    if (status == ErrorCode::Success)
        status = tenant.index->GetUpdateStatus();
    {
        std::lock_guard<std::mutex> lock(tenant.submitMutex);
        tenant.pendingInserts.erase(begin);
        if (status != ErrorCode::Success)
            return Poison(status);
        if ((status = Ready()) != ErrorCode::Success)
            return status;
        firstID = begin;
    }
    return ErrorCode::Success;
}

ErrorCode TenantIndexManager::DeleteVector(const std::string &name, SizeType id)
{
    auto admission = WaitForCheckpoint();
    if (admission != ErrorCode::Success)
        return admission;
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    std::shared_lock<std::shared_mutex> updates(m_updates);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    auto it = m_tenants.find(name);
    if (it == m_tenants.end())
        return Failure("Tenant not found", ErrorCode::Key_NotFound);
    auto &tenant = *it->second;
    Lease lease;
    if ((status = Acquire(tenant, lease, true)) != ErrorCode::Success)
        return status;
    std::lock_guard<std::mutex> lock(tenant.submitMutex);
    if ((status = Ready()) != ErrorCode::Success)
        return status;
    auto pending = tenant.pendingInserts.upper_bound(id);
    if (pending != tenant.pendingInserts.begin())
    {
        --pending;
        if (static_cast<int64_t>(id) < static_cast<int64_t>(pending->first) + pending->second)
            return ErrorCode::ExternalAbort;
    }
    if (!tenant.index->ContainSample(id))
        return Failure("Vector not found", ErrorCode::VectorNotFound);
    std::string record;
    Number(record, 1);
    Number(record, 2);
    Number(record, id);
    try
    {
        if ((status = Record(tenant, record)) != ErrorCode::Success)
            return status;
        return Poison(tenant.index->DeleteIndex(id));
    }
    catch (...)
    {
        return Poison(ErrorCode::Fail);
    }
}

ErrorCode TenantIndexManager::Search(const std::string &name, const ByteArray &query, SizeType k,
                                     std::vector<BasicResult> &results, VectorValueType expectedType,
                                     DimensionType expectedDimension)
{
    {
        ReadLease lease(m_readers);
        auto status = Ready();
        if (status != ErrorCode::Success)
            return status;
        const auto *directory = m_directory.load();
        if (!directory)
            return Failure("Node closed or failed; reopen to recover");
        auto it = directory->find(name);
        if (it == directory->end())
            return Failure("Tenant not found", ErrorCode::Key_NotFound);
        auto &tenant = *it->second;
        if (!ValidVectors(tenant, query, 1) || k <= 0 || k > m_limits.maxResults ||
            (expectedType != VectorValueType::Undefined && expectedType != tenant.type) ||
            (expectedDimension != 0 && expectedDimension != tenant.dimension))
            return Failure("Invalid query or result limit", ErrorCode::FailedParseValue);
        lease.guard.SetOwner(&tenant);
        if (auto *index = tenant.readIndex.load())
        {
            lease.tenant = &tenant;
            tenant.readHits.fetch_add(1, std::memory_order_relaxed);
            return SearchLoaded(tenant, *index, query, k, results);
        }
    }
    // Never wait for loading, capacity, or a lifecycle barrier inside an epoch.
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    auto it = m_tenants.find(name);
    if (it == m_tenants.end())
        return Failure("Tenant not found", ErrorCode::Key_NotFound);
    auto &tenant = *it->second;
    if (!ValidVectors(tenant, query, 1) || k <= 0 || k > m_limits.maxResults ||
        (expectedType != VectorValueType::Undefined && expectedType != tenant.type) ||
        (expectedDimension != 0 && expectedDimension != tenant.dimension))
        return Failure("Invalid query or result limit", ErrorCode::FailedParseValue);
    Lease lease;
    if ((status = Acquire(tenant, lease)) != ErrorCode::Success)
        return status;
    return SearchLoaded(tenant, *tenant.index, query, k, results);
}

ErrorCode TenantIndexManager::SearchLoaded(const Tenant &tenant, VectorIndex &index, const ByteArray &query, SizeType k,
                                           std::vector<BasicResult> &results)
{
    auto target = ByteArray::Alloc(query.Length());
    memcpy(target.Data(), query.Data(), query.Length());
    if (tenant.metric == DistCalcMethod::Cosine)
        BasicVectorSet(target, tenant.type, tenant.dimension, 1).Normalize(1);
    QueryResult result(target.Data(), k, false);
    auto status = index.SearchIndex(result);
    if (status != ErrorCode::Success)
        return status;
    results.assign(result.GetResults(), result.GetResults() + k);
    return ErrorCode::Success;
}

ErrorCode TenantIndexManager::DeleteTenant(const std::string &name)
{
    std::lock_guard<std::mutex> management(m_management);
    std::unique_lock<std::shared_mutex> barrier(m_barrier);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    auto it = m_tenants.find(name);
    if (it == m_tenants.end())
        return Failure("Tenant not found", ErrorCode::Key_NotFound);
    it->second->readIndex.store(nullptr);
    m_readers.Synchronize();
    if (it->second->index && (status = it->second->index->DrainUpdates()) != ErrorCode::Success)
        return Poison(status);
    const auto prefix = SharedRocksDB::Namespace(name, it->second->id);
    const auto indexID = it->second->id;
    const auto directory = Directory(*it->second);
    {
        std::lock_guard<std::mutex> cache(m_cacheMutex);
        if (it->second->stats.state == Residency::Resident)
        {
            --m_residentCount;
            m_residentBytes -= it->second->stats.snapshotBytes;
        }
    }
    m_tenants.erase(it);
    if ((status = m_postings->DeleteIndex(name, indexID)) != ErrorCode::Success)
        return Poison(status);
    if ((status = m_state->DeleteRange("head_snapshots", prefix, SharedRocksDB::PrefixEnd(prefix))) !=
        ErrorCode::Success)
        return Poison(status);
    if ((status = CheckpointLocked()) != ErrorCode::Success)
        return Poison(status);
    PublishDirectory();
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    return error ? Poison(Failure(error.message().c_str(), ErrorCode::DiskIOFail)) : ErrorCode::Success;
}

ErrorCode TenantIndexManager::Unload(const std::string &name)
{
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    std::shared_lock<std::shared_mutex> updates(m_updates);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    auto it = m_tenants.find(name);
    if (it == m_tenants.end())
        return Failure("Tenant not found", ErrorCode::Key_NotFound);
    auto &tenant = *it->second;
    std::unique_lock<std::mutex> cache(m_cacheMutex);
    while (tenant.stats.state == Residency::Loading || tenant.stats.state == Residency::Evicting)
    {
        m_cacheChanged.wait(cache);
        if ((status = Ready()) != ErrorCode::Success)
            return status;
    }
    if (tenant.stats.state == Residency::Unloaded)
        return ErrorCode::Success;
    tenant.stats.state = Residency::Evicting;
    tenant.readIndex.store(nullptr);
    while (tenant.stats.activeRequests)
        m_cacheChanged.wait(cache);
    if ((status = Ready()) != ErrorCode::Success)
    {
        tenant.stats.state = Residency::Resident;
        tenant.readIndex.store(tenant.index.get());
        m_cacheChanged.notify_all();
        return status;
    }
    return Evict(tenant, cache);
}

ErrorCode TenantIndexManager::Checkpoint()
{
    return CheckpointInternal(false);
}

bool TenantIndexManager::CheckpointDue() const
{
    const auto bytes = m_recoveryLogBytes.load();
    return bytes >= m_limits.checkpointLogBytes && (bytes >= m_limits.checkpointPressureBytes ||
                                                    std::chrono::steady_clock::now() - m_lastCheckpoint >=
                                                        std::chrono::seconds(m_limits.checkpointMinIntervalSeconds));
}

ErrorCode TenantIndexManager::WaitForCheckpoint()
{
    if (!m_limits.autoCheckpoint)
        return Ready();
    // Never wait while holding a lease or either lock needed by checkpoint.
    std::unique_lock<std::mutex> checkpoint(m_checkpointMutex);
    m_checkpointChanged.wait(checkpoint, [&] {
        return m_checkpointStopping || m_closed || m_failed ||
               m_recoveryLogBytes.load() < m_limits.checkpointPressureBytes;
    });
    if (m_checkpointStopping)
        return Failure("Node closing; update admission stopped");
    return Ready();
}

void TenantIndexManager::CheckpointWorker()
try
{
    std::unique_lock<std::mutex> checkpoint(m_checkpointMutex);
    while (!m_checkpointStopping && !m_closed && !m_failed)
    {
        if (CheckpointDue())
        {
            checkpoint.unlock();
            auto status = CheckpointInternal(true);
            checkpoint.lock();
            if (status != ErrorCode::Success)
                break;
        }
        else if (m_recoveryLogBytes.load() >= m_limits.checkpointLogBytes)
        {
            m_checkpointChanged.wait_until(checkpoint, m_lastCheckpoint +
                                                           std::chrono::seconds(m_limits.checkpointMinIntervalSeconds));
        }
        else
        {
            m_checkpointChanged.wait(checkpoint);
        }
    }
}
catch (...)
{
    Poison(Failure("Automatic checkpoint worker failed"));
}

void TenantIndexManager::StopCheckpointWorker()
{
    std::lock_guard<std::mutex> stop(m_checkpointStopMutex);
    {
        std::lock_guard<std::mutex> checkpoint(m_checkpointMutex);
        m_checkpointStopping = true;
    }
    m_checkpointChanged.notify_all();
    if (m_checkpointThread.joinable())
        m_checkpointThread.join();
}

ErrorCode TenantIndexManager::CheckpointInternal(bool automatic)
{
    std::lock_guard<std::mutex> management(m_management);
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    std::unique_lock<std::shared_mutex> updates(m_updates);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    if (automatic)
    {
        std::lock_guard<std::mutex> checkpoint(m_checkpointMutex);
        if (m_checkpointStopping || !CheckpointDue())
            return ErrorCode::Success;
    }
    m_checkpointInProgress.store(true);
    status = Poison(CheckpointLocked());
    m_checkpointInProgress.store(false);
    if (automatic && status == ErrorCode::Success)
        m_autoCheckpoints.fetch_add(1);
    m_checkpointChanged.notify_all();
    return status;
}

ErrorCode TenantIndexManager::Close()
{
    StopCheckpointWorker();
    std::lock_guard<std::mutex> management(m_management);
    std::unique_lock<std::shared_mutex> barrier(m_barrier);
    if (m_closed)
        return ErrorCode::Success;
    auto status = Ready();
    m_closed.store(true);
    m_directory.store(nullptr);
    m_readers.Synchronize();
    if (status == ErrorCode::Success)
        status = Poison(CheckpointLocked());
    m_directoryOwner.reset();
    for (auto &entry : m_tenants)
        if (entry.second->index)
            entry.second->index->DrainUpdates();
    m_tenants.clear();
    m_residentCount = 0;
    m_residentBytes = 0;
    m_postings.reset();
    m_state.reset();
    m_recovery.reset();
    return status;
}

void TenantIndexManager::GetStats(NodeStats &node, std::vector<TenantStats> &tenants) const
{
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    {
        std::lock_guard<std::mutex> cache(m_cacheMutex);
        node.failed = m_failed.load();
        node.closed = m_closed;
        node.residentTenants = m_residentCount;
        node.maxResidentTenants = m_limits.maxResidentTenants;
        node.residentBytes = m_residentBytes;
        node.maxResidentBytes = m_limits.maxResidentBytes;
        node.overByteTarget = m_limits.maxResidentBytes && m_residentBytes > m_limits.maxResidentBytes;
        node.autoCheckpoint = m_limits.autoCheckpoint;
        node.checkpointInProgress = m_checkpointInProgress.load();
        node.recoveryLogBytes = m_recoveryLogBytes.load();
        node.autoCheckpoints = m_autoCheckpoints.load();
    }
    tenants.clear();
    for (const auto &entry : m_tenants)
    {
        const auto &tenant = *entry.second;
        std::lock_guard<std::mutex> submit(entry.second->submitMutex);
        std::lock_guard<std::mutex> cache(m_cacheMutex);
        auto stats = tenant.stats;
        stats.cacheHits += tenant.readHits.load(std::memory_order_relaxed);
        stats.activeRequests += m_readers.Active(&tenant);
        stats.name = tenant.name;
        stats.indexID = tenant.id;
        stats.dirty = !tenant.hasSnapshot || tenant.sequence != tenant.savedSequence;
        tenants.emplace_back(std::move(stats));
    }
}

std::vector<std::string> TenantIndexManager::List() const
{
    std::shared_lock<std::shared_mutex> barrier(m_barrier);
    std::vector<std::string> result;
    for (const auto &entry : m_tenants)
        result.push_back(entry.first);
    return result;
}

ErrorCode TenantIndexManager::Describe(const std::string &name, VectorValueType &type, DimensionType &dimension) const
{
    ReadEpoch::Guard guard(m_readers);
    auto status = Ready();
    if (status != ErrorCode::Success)
        return status;
    const auto *directory = m_directory.load();
    if (!directory)
        return Failure("Node closed or failed; reopen to recover");
    auto it = directory->find(name);
    if (it == directory->end())
        return Failure("Tenant not found", ErrorCode::Key_NotFound);
    type = it->second->type;
    dimension = it->second->dimension;
    return ErrorCode::Success;
}
} // namespace SPTAG::SPANN
#endif
