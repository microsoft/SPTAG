// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef MULTITENANT
#include "inc/Core/SPANN/PostingBackend.h"
#include "inc/Core/Common/LocalVersionMap.h"
#ifdef TIKV
#include "inc/Core/Common/TiKVVersionMap.h"
#endif
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/TenantIndexManager.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Server/ServiceContext.h"
#include "inc/Test.h"
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#ifndef _MSC_VER
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

using namespace SPTAG;
using namespace SPTAG::SPANN;

namespace
{
namespace fs = std::filesystem;
const std::chrono::microseconds IOTimeout(10000000);
constexpr SizeType SeedCount = 64;
constexpr float InsertedBase = 2.345f;

struct TestDirectory
{
    fs::path path = fs::current_path() / boost::filesystem::unique_path("posting-test-%%%%-%%%%-%%%%").string();
    ~TestDirectory()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
    std::string Node() const
    {
        return (path / "node").string();
    }
};

void Success(ErrorCode status)
{
    BOOST_REQUIRE_EQUAL(static_cast<int>(status), static_cast<int>(ErrorCode::Success));
}

bool TiKVEnabled()
{
    const char *pd = std::getenv("SPTAG_TIKV_PD");
    if (!pd || !*pd)
    {
        BOOST_TEST_MESSAGE("SKIPPED real TiKV integration: set SPTAG_TIKV_PD to a reachable PD endpoint.");
        return false;
    }
#ifndef TIKV
    BOOST_FAIL("SPTAG_TIKV_PD is set, but this test binary was built without TIKV support");
    return false;
#else
    return true;
#endif
}

TenantIndexManager::Limits LimitsFor(Storage storage)
{
    TenantIndexManager::Limits limits;
    limits.maxTenants = 4;
    limits.maxVectorsPerTenant = 4096;
    limits.maxDimension = 4;
    limits.maxResults = 16;
    limits.cacheBytes = 8ULL << 20;
    limits.maxResidentTenants = 2;
    limits.posting.storage = storage;
    if (const char *pd = std::getenv("SPTAG_TIKV_PD"))
        limits.posting.tikvPDAddresses = pd;
    return limits;
}

ByteArray Data(SizeType count, float base)
{
    auto bytes = ByteArray::Alloc(static_cast<size_t>(count) * 4 * sizeof(float));
    auto *values = reinterpret_cast<float *>(bytes.Data());
    for (SizeType i = 0; i < count; ++i)
        for (int d = 0; d < 4; ++d)
            values[i * 4 + d] = base + i * 0.1f + d * 0.01f;
    return bytes;
}

std::unique_ptr<TenantIndexManager> Open(const std::string &root, const TenantIndexManager::Limits &limits)
{
    std::unique_ptr<TenantIndexManager> result;
    Success(TenantIndexManager::Open(root, limits, result));
    BOOST_REQUIRE(result);
    return result;
}

void Create(TenantIndexManager &node, const std::string &tenant = "A", float base = 0, int layers = 1,
            bool localVersions = false, int searchCandidates = 64)
{
    Helper::IniReader profile;
    if (localVersions)
        profile.SetParameter("BuildSSDIndex", "DistributedVersionMap", "false");
    profile.SetParameter("BuildSSDIndex", "SearchInternalResultNum", std::to_string(searchCandidates));
    // Keep a one-vector append well below the split threshold: recovery must not
    // depend on a split saving the mutable head/version state as a side effect.
    profile.SetParameter("BuildSSDIndex", "PostingPageLimit", "64");
    profile.SetParameter("BuildSSDIndex", "MaxID", "4096");
    profile.SetParameter("BuildSSDIndex", "NumberOfThreads", "1");
    profile.SetParameter("SelectHead", "NumberOfThreads", "1");
    profile.SetParameter("Base", "DataBlockSize", "4096");
    profile.SetParameter("BuildHead", "DataBlockSize", "4096");
    Success(node.Create(tenant, VectorValueType::Float, 4, DistCalcMethod::L2, Data(SeedCount, base), SeedCount,
                        IndexAlgoType::BKT, layers, &profile));
}

void Nearest(TenantIndexManager &node, const std::string &tenant, float base, SizeType id)
{
    std::vector<BasicResult> results;
    Success(node.Search(tenant, Data(1, base), 1, results));
    BOOST_REQUIRE_EQUAL(results.size(), 1);
    BOOST_CHECK_EQUAL(results.front().VID, id);
    BOOST_CHECK_SMALL(results.front().Dist, 0.00001f);
}

void AppendAndDelete(TenantIndexManager &node)
{
    SizeType id = -1;
    Success(node.Insert("A", Data(1, InsertedBase), 1, id));
    BOOST_REQUIRE_EQUAL(id, SeedCount);
    Success(node.DeleteVector("A", 0));
}

void CheckRecovered(TenantIndexManager &node)
{
    Nearest(node, "A", InsertedBase, SeedCount);
    std::vector<BasicResult> results;
    Success(node.Search("A", Data(1, 0), 8, results));
    for (const auto &result : results)
        BOOST_CHECK_NE(result.VID, 0);
    BOOST_CHECK(node.DeleteVector("A", 0) == ErrorCode::VectorNotFound);
}

void RecoverTwice(const std::string &root, Storage storage)
{
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        auto node = Open(root, LimitsFor(storage));
        CheckRecovered(*node);
        SizeType id = -1;
        Success(node->Insert("A", Data(1, 4.321f + attempt * 0.02f), 1, id));
        BOOST_CHECK_EQUAL(id, SeedCount + 1 + attempt);
        // Intentionally no Close(): the second Open must repeat old replay
        // exactly once and then replay this additional acknowledged insertion.
    }
}

uint64_t Identity(TenantIndexManager &node, const std::string &name)
{
    TenantIndexManager::NodeStats stats;
    std::vector<TenantIndexManager::TenantStats> tenants;
    node.GetStats(stats, tenants);
    auto found = std::find_if(tenants.begin(), tenants.end(), [&](const auto &t) { return t.name == name; });
    BOOST_REQUIRE(found != tenants.end());
    return found->indexID;
}

void ManagerIsolation(Storage storage)
{
    TestDirectory directory;
    const auto limits = LimitsFor(storage);
    auto node = Open(directory.Node(), limits);
    Create(*node, "A", 0, 2);
    Create(*node, "AA", 100, 2);
    const auto oldID = Identity(*node, "A");
    AppendAndDelete(*node);
    SizeType id;
    Success(node->Insert("AA", Data(1, 102.345f), 1, id));
    BOOST_CHECK_EQUAL(id, SeedCount);
    Success(node->Unload("A"));
    Nearest(*node, "AA", 102.345f, SeedCount);
    CheckRecovered(*node);
    Success(node->Checkpoint());
    Success(node->DeleteTenant("A"));
    Create(*node, "A", -100, 2);
    BOOST_CHECK_NE(Identity(*node, "A"), oldID);
    Nearest(*node, "A", -100, 0);
    Nearest(*node, "AA", 102.345f, SeedCount);
    Success(node->Unload("AA"));
    Nearest(*node, "A", -100, 0);
    Success(node->Close());
    node.reset();
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        node = Open(directory.Node(), limits);
        Nearest(*node, "A", -100, 0);
        Nearest(*node, "AA", 102.345f, SeedCount);
        BOOST_CHECK_NE(Identity(*node, "A"), oldID);
        node.reset();
    }
    auto wrong = limits;
    wrong.posting.storage = storage == Storage::FILEIO ? Storage::ROCKSDBIO : Storage::FILEIO;
    BOOST_CHECK(TenantIndexManager::Open(directory.Node(), wrong, node) != ErrorCode::Success);
    BOOST_CHECK(!node);
    if (storage == Storage::TIKVIO)
    {
        wrong = limits;
        wrong.posting.tikvKeyPrefix += "different/";
        BOOST_CHECK(TenantIndexManager::Open(directory.Node(), wrong, node) != ErrorCode::Success);
        BOOST_CHECK(!node);
        wrong = limits;
        wrong.posting.tikvPDAddresses += ",127.0.0.1:1";
        BOOST_CHECK(TenantIndexManager::Open(directory.Node(), wrong, node) != ErrorCode::Success);
        BOOST_CHECK(!node);
    }
    node = Open(directory.Node(), limits);
    Nearest(*node, "AA", 102.345f, SeedCount);
}

void AutomaticCheckpoint(Storage storage)
{
    TestDirectory directory;
    auto limits = LimitsFor(storage);
    limits.checkpointLogBytes = 1;
    limits.checkpointPressureBytes = 1;
    limits.checkpointMinIntervalSeconds = 3600;
    auto node = Open(directory.Node(), limits);
    Create(*node);
    SizeType first;
    Success(node->Insert("A", Data(1, InsertedBase), 1, first));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    TenantIndexManager::NodeStats state;
    std::vector<TenantIndexManager::TenantStats> tenants;
    do
    {
        node->GetStats(state, tenants);
        if (state.autoCheckpoints)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    BOOST_REQUIRE_EQUAL(state.autoCheckpoints, 1);
    BOOST_CHECK_EQUAL(state.recoveryLogBytes, 0);
    Nearest(*node, "A", InsertedBase, first);
    node.reset();
    node = Open(directory.Node(), limits);
    Nearest(*node, "A", InsertedBase, first);
}

void DestructorRecovery(Storage storage)
{
    TestDirectory directory;
    {
        auto node = Open(directory.Node(), LimitsFor(storage));
        Create(*node);
        Success(node->Checkpoint());
        AppendAndDelete(*node);
    }
    RecoverTwice(directory.Node(), storage);
}

void FailedCheckpointFences(Storage storage)
{
    for (const std::string stage : {"before_snapshots", "after_publish"})
    {
        BOOST_TEST_CONTEXT("throwing checkpoint hook " << stage)
        {
            TestDirectory directory;
            bool armed = false;
            auto limits = LimitsFor(storage);
            limits.checkpointHook = [&](const char *at) {
                if (armed && stage == at)
                    throw std::runtime_error("injected checkpoint failure");
            };
            auto node = Open(directory.Node(), limits);
            Create(*node);
            AppendAndDelete(*node);
            armed = true;
            BOOST_CHECK(node->Checkpoint() != ErrorCode::Success);
            TenantIndexManager::NodeStats stats;
            std::vector<TenantIndexManager::TenantStats> tenants;
            node->GetStats(stats, tenants);
            BOOST_CHECK(stats.failed);
            SizeType id;
            std::vector<BasicResult> results;
            BOOST_CHECK(node->Insert("A", Data(1, 1), 1, id) != ErrorCode::Success);
            BOOST_CHECK(node->Search("A", Data(1, 1), 1, results) != ErrorCode::Success);
            BOOST_CHECK(node->Unload("A") != ErrorCode::Success);
            BOOST_CHECK(node->Checkpoint() != ErrorCode::Success);
            node.reset();
            RecoverTwice(directory.Node(), storage);
        }
    }
}

void SparseVersionBatches()
{
    TestDirectory directory;
    fs::create_directories(directory.path);
    COMMON::LocalVersionMap map;
    SizeType low = 0, high = 64, sparse = 4095;
    uint8_t lowVersion = 2, highVersion = 7, sparseVersion = 13;
    map.SetVersion(low, lowVersion);
    map.SetVersion(high, highVersion);
    map.SetVersion(sparse, sparseVersion);
    BOOST_REQUIRE(map.Delete(low));
    const auto check = [](COMMON::IVersionMap &versions) {
        const std::vector<SizeType> ids{64, 0, 4095, -1, 2000, 64};
        const std::vector<uint8_t> expected{7, 0xfe, 13, 0xfe, 0xfe, 7};
        std::vector<uint8_t> actual;
        versions.BatchGetVersions(ids, actual);
        BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.end(), expected.begin(), expected.end());
        versions.BatchGetVersions(ids, actual, COMMON::VersionReadPolicy::BypassCacheNoFill);
        BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.end(), expected.begin(), expected.end());
    };
    check(map);
    const auto image = (directory.path / "versions").string();
    Success(map.Save(image));
    COMMON::LocalVersionMap restored;
    Success(restored.Load(image, 128, 4096));
    check(restored);
}

void TiKVTopKAfterDeletions()
{
    TestDirectory directory;
    const auto limits = LimitsFor(Storage::TIKVIO);
    auto node = Open(directory.Node(), limits);
    Create(*node, "A", 0, 1, false, 8);
    for (SizeType id = 0; id < 16; ++id)
        Success(node->DeleteVector("A", id));
    SizeType appended;
    Success(node->Insert("A", Data(1, 0.012f), 1, appended));
    BOOST_REQUIRE_EQUAL(appended, SeedCount);
    const std::vector<SizeType> expected{SeedCount, 16, 17, 18, 19, 20, 21, 22};
    for (int attempt = 0; attempt < 3; ++attempt)
    {
        std::vector<BasicResult> results;
        Success(node->Search("A", Data(1, 0.012f), 8, results));
        BOOST_REQUIRE_EQUAL(results.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            BOOST_CHECK_EQUAL(results[i].VID, expected[i]);
        if (attempt == 0)
            Success(node->Unload("A"));
        else
        {
            node.reset();
            node = Open(directory.Node(), limits);
        }
    }
}

void ServiceBackendSelection(Storage storage)
{
    TestDirectory directory;
    fs::create_directories(directory.path);
    auto limits = LimitsFor(storage);
    if (storage == Storage::TIKVIO)
        limits.posting.tikvKeyPrefix = "sptag/service-test/" + directory.path.filename().string() + "/";
    const auto config = directory.path / "service.ini";
    const auto writeConfig = [&](const std::string &backend, const std::string &prefix) {
        std::ofstream output(config);
        BOOST_REQUIRE(output.is_open());
        output << "[MultiTenant]\nRoot=" << directory.Node() << "\nStorage=" << backend
               << "\nTiKVPDAddresses=" << limits.posting.tikvPDAddresses << "\nTiKVKeyPrefix=" << prefix
               << "\nMaxVectorsPerTenant=4096\nMaxDimension=4\nMaxResults=16\nCacheBytes=8388608\n"
                  "MaxResidentTenants=2\nMaxTenants=4\nAutoCheckpoint=false\nCheckpointLogBytes=1\n"
                  "CheckpointPressureBytes=1\nCheckpointMinIntervalSeconds=0\n";
        output.close();
        BOOST_REQUIRE(!output.fail());
    };
    const std::string backend = storage == Storage::FILEIO ? "FILEIO" : "TIKVIO";
    writeConfig(backend, limits.posting.tikvKeyPrefix);
    {
        Service::ServiceContext service(config.string());
        BOOST_REQUIRE(service.IsInitialized());
        BOOST_REQUIRE(service.GetTenantManager());
        Create(*service.GetTenantManager());
        AppendAndDelete(*service.GetTenantManager());
        TenantIndexManager::NodeStats state;
        std::vector<TenantIndexManager::TenantStats> tenants;
        service.GetTenantManager()->GetStats(state, tenants);
        BOOST_CHECK(!state.autoCheckpoint);
        BOOST_CHECK(state.recoveryLogBytes > 0);
        Success(service.GetTenantManager()->Close());
    }
    {
        Service::ServiceContext service(config.string());
        BOOST_REQUIRE(service.IsInitialized());
        BOOST_REQUIRE(service.GetTenantManager());
        CheckRecovered(*service.GetTenantManager());
    }
    {
        // A public manager reopen verifies that ServiceContext persisted the
        // requested backend/PD/prefix, rather than silently using defaults.
        auto node = Open(directory.Node(), limits);
        CheckRecovered(*node);
    }
    writeConfig("ROCKSDBIO", limits.posting.tikvKeyPrefix);
    {
        Service::ServiceContext wrongBackend(config.string());
        BOOST_CHECK(!wrongBackend.IsInitialized());
        BOOST_CHECK(!wrongBackend.GetTenantManager());
    }
    if (storage == Storage::TIKVIO)
    {
        writeConfig(backend, limits.posting.tikvKeyPrefix + "different/");
        Service::ServiceContext wrongPrefix(config.string());
        BOOST_CHECK(!wrongPrefix.IsInitialized());
        BOOST_CHECK(!wrongPrefix.GetTenantManager());
    }
}

#ifndef _MSC_VER
constexpr int Crashed = 77;

class MaintenanceLog : public Helper::Logger
{
  public:
    std::mutex mutex;
    size_t loadedHeads = 0, savedHeads = 0;
    void Logging(const char *, Helper::LogLevel, const char *, int, const char *, const char *format, ...) override
    {
        char message[4096];
        va_list args;
        va_start(args, format);
        vsnprintf(message, sizeof(message), format, args);
        va_end(args);
        if (std::getenv("SPTAG_POSTING_TRACE_MAINTENANCE"))
            std::fputs(message, stderr);
        std::lock_guard<std::mutex> lock(mutex);
        size_t count = 0;
        if (!loadedHeads && std::sscanf(message, "Load Vector (%zu,", &count) == 1)
            loadedHeads = count;
        if (std::sscanf(message, "Save Vector (%zu,", &count) == 1)
            savedHeads = std::max(savedHeads, count);
    }
};

ByteArray MaintenanceVectors(SizeType first, SizeType count)
{
    auto bytes = ByteArray::Alloc(static_cast<size_t>(count) * 4 * sizeof(float));
    auto *values = reinterpret_cast<float *>(bytes.Data());
    for (SizeType i = 0; i < count; ++i)
    {
        uint32_t random = 0x9e3779b9U ^ static_cast<uint32_t>(first + i);
        for (int d = 0; d < 4; ++d)
        {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            values[i * 4 + d] = static_cast<float>(random & 0xffffU) / 32768.0f - 1.0f;
        }
    }
    return bytes;
}

constexpr SizeType MaintenanceBeforeDelete = SeedCount + 4 * 256;
constexpr SizeType MaintenanceDeleted = 960;
constexpr SizeType MaintenanceEnd = MaintenanceBeforeDelete + 256;

std::vector<SizeType> MaintenanceLogicalState(TenantIndexManager &node)
{
    std::vector<SizeType> state;
    for (SizeType probe = MaintenanceDeleted; probe < MaintenanceEnd; probe += 17)
    {
        std::vector<BasicResult> results;
        Success(node.Search("A", MaintenanceVectors(probe, 1), 8, results));
        BOOST_REQUIRE_EQUAL(results.size(), 8);
        BOOST_CHECK_EQUAL(results.front().VID, probe);
        std::set<SizeType> unique;
        float distance = -1;
        for (const auto &result : results)
        {
            BOOST_CHECK_GE(result.VID, MaintenanceDeleted);
            BOOST_CHECK_LT(result.VID, MaintenanceEnd);
            BOOST_CHECK_GE(result.Dist, distance);
            distance = result.Dist;
            BOOST_CHECK(unique.insert(result.VID).second);
            state.push_back(result.VID);
        }
    }
    for (SizeType id : {SizeType(0), SizeType(255), MaintenanceDeleted - 1})
        BOOST_CHECK(node.DeleteVector("A", id) == ErrorCode::VectorNotFound);
    return state;
}

void CreateMaintenanceTenant(TenantIndexManager &node)
{
    Helper::IniReader profile;
    profile.SetParameter("Base", "DataBlockSize", "4096");
    profile.SetParameter("BuildHead", "DataBlockSize", "4096");
    profile.SetParameter("SelectHead", "NumberOfThreads", "1");
    profile.SetParameter("SelectHead", "SelectHeadType", "Random");
    profile.SetParameter("SelectHead", "Count", "2");
    profile.SetParameter("BuildSSDIndex", "NumberOfThreads", "1");
    profile.SetParameter("BuildSSDIndex", "PostingPageLimit", "1");
    profile.SetParameter("BuildSSDIndex", "PostingVectorLimit", "1");
    profile.SetParameter("BuildSSDIndex", "BufferLength", "0");
    profile.SetParameter("BuildSSDIndex", "MaxID", "4096");
    profile.SetParameter("BuildSSDIndex", "ReplicaCount", "2");
    profile.SetParameter("BuildSSDIndex", "DisableReassign", "false");
    profile.SetParameter("BuildSSDIndex", "ReassignK", "8");
    Success(node.Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, MaintenanceVectors(0, SeedCount), SeedCount,
                        IndexAlgoType::BKT, 1, &profile));
}

size_t ReassignedWorkingVersions(TenantIndexManager &node, const std::string &root)
{
    const auto indexName = "index-" + std::to_string(Identity(node, "A"));
    fs::path indexDirectory;
    for (const auto &entry : fs::directory_iterator(root))
        if (entry.path().filename().string().find("work-") == 0 &&
            fs::exists(entry.path() / indexName / "indexloader.ini"))
        {
            BOOST_REQUIRE(indexDirectory.empty());
            indexDirectory = entry.path() / indexName;
        }
    BOOST_REQUIRE(!indexDirectory.empty());
    Helper::IniReader config;
    Success(config.LoadIniFile((indexDirectory / "indexloader.ini").string()));
    COMMON::LocalVersionMap versions;
    Success(versions.Load(
        (indexDirectory / (config.GetParameter("Base", "DeletedIDs", std::string("DeletedIDs.bin")) + "_0")).string(),
        4096, 4096));
    std::vector<SizeType> ids;
    Success(versions.GetContainedIDs(ids));
    return std::count_if(ids.begin(), ids.end(), [&](SizeType id) {
        const auto version = versions.GetVersion(id);
        return version != versions.Default() && version != 0xfe;
    });
}

void MaintenanceWorker(TenantIndexManager &node, const std::string &root, bool distributed)
{
    auto previous = GetLoggerHolder().GetLogger();
    auto log = std::make_shared<MaintenanceLog>();
    GetLoggerHolder().SetLogger(log);
    for (SizeType first = SeedCount; first < MaintenanceBeforeDelete; first += 256)
    {
        SizeType id;
        Success(node.Insert("A", MaintenanceVectors(first, 256), 256, id));
        BOOST_REQUIRE_EQUAL(id, first);
    }
    Success(node.Unload("A"));
    // Completion logs are deliberately suppressed after an already-idle
    // boundary. Prove real maintenance from its semantic effects instead:
    // additional heads and incremented VID assignment versions.
    const auto reassigned = distributed ? 0 : ReassignedWorkingVersions(node, root);
    size_t loadedHeads, savedHeads;
    {
        std::lock_guard<std::mutex> lock(log->mutex);
        BOOST_REQUIRE_GT(log->loadedHeads, 0);
        BOOST_REQUIRE_GT(log->savedHeads, log->loadedHeads);
        loadedHeads = log->loadedHeads;
        savedHeads = log->savedHeads;
    }
    for (SizeType id = 0; id < MaintenanceDeleted; ++id)
        Success(node.DeleteVector("A", id));
    SizeType id;
    Success(node.Insert("A", MaintenanceVectors(MaintenanceBeforeDelete, 256), 256, id));
    BOOST_REQUIRE_EQUAL(id, MaintenanceBeforeDelete);
    // Drain maintenance and save only mutable working state, never publish a
    // checkpoint. The crash must still recover from the original seed image.
    Success(node.Unload("A"));
    GetLoggerHolder().SetLogger(previous);
    const auto state = MaintenanceLogicalState(node);
    std::ofstream output(fs::path(root).parent_path() / "logical-state");
    BOOST_REQUIRE(output.is_open());
    output << loadedHeads << ' ' << savedHeads << ' ' << reassigned << '\n';
    for (auto vid : state)
        output << vid << '\n';
    output.close();
    BOOST_REQUIRE(!output.fail());
    if (distributed)
    {
        Success(node.Checkpoint());
        BOOST_FAIL("unpublished archive crash hook did not fire");
    }
}

class CrashDuringRecoverySave : public Helper::SimpleFileIO
{
  public:
    bool Initialize(const char *path, int mode, uint64_t maxIO = 1 << 20, uint32_t reads = 2, uint32_t writes = 2,
                    uint16_t threads = 4, uint64_t maxFile = 300ULL << 30) override
    {
        // Open restores and replays before SaveIndex writes this configuration.
        // Exit before Open returns, with replayed posting mutations in work state.
        if ((mode & std::ios::out) && fs::path(path).filename() == "indexloader.ini")
            _exit(Crashed);
        return SimpleFileIO::Initialize(path, mode, maxIO, reads, writes, threads, maxFile);
    }
};

void RunChild(const std::string &root, Storage storage, const std::string &mode, const std::string &stage = "")
{
    // Construct all argv/env storage before fork. The child only calls
    // async-signal-safe execve/_exit, even when the parent has live TBB threads.
    std::vector<std::string> environment;
    for (char **entry = environ; *entry; ++entry)
        if (std::string(*entry).find("SPTAG_POSTING_WORKER_") != 0)
            environment.emplace_back(*entry);
    environment.push_back("SPTAG_POSTING_WORKER_ROOT=" + root);
    environment.push_back("SPTAG_POSTING_WORKER_STORAGE=" + std::to_string(static_cast<int>(storage)));
    environment.push_back("SPTAG_POSTING_WORKER_MODE=" + mode);
    environment.push_back("SPTAG_POSTING_WORKER_STAGE=" + stage);
    std::vector<char *> env;
    for (auto &entry : environment)
        env.push_back(&entry[0]);
    env.push_back(nullptr);
    char executable[] = "/proc/self/exe";
    char selector[] = "--run_test=PostingBackends/CrashWorker";
    char *args[] = {executable, selector, nullptr};
    const auto child = fork();
    BOOST_REQUIRE(child >= 0);
    if (child == 0)
    {
        execve(executable, args, env.data());
        _exit(127);
    }
    int status = 0;
    pid_t waited = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
    for (;;)
    {
        waited = waitpid(child, &status, WNOHANG);
        if (waited < 0 && errno == EINTR)
            continue;
        if (waited != 0 || std::chrono::steady_clock::now() >= deadline)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (waited == 0)
    {
        kill(child, SIGKILL);
        while (waitpid(child, &status, 0) < 0 && errno == EINTR)
        {
        }
        BOOST_FAIL("posting recovery/crash worker exceeded 90 seconds");
    }
    BOOST_REQUIRE_EQUAL(waited, child);
    if (stage == "after_publish")
    {
        BOOST_REQUIRE(WIFSIGNALED(status));
        BOOST_CHECK_EQUAL(WTERMSIG(status), SIGKILL);
    }
    else
    {
        BOOST_REQUIRE(WIFEXITED(status));
        BOOST_REQUIRE_EQUAL(WEXITSTATUS(status), Crashed);
    }
}

void ProcessCrashRecovery(Storage storage)
{
    TestDirectory directory;
    {
        auto node = Open(directory.Node(), LimitsFor(storage));
        Create(*node, "A", 0, 2);
        Success(node->Close());
    }
    RunChild(directory.Node(), storage, "write");
    RunChild(directory.Node(), storage, "recover");
    RecoverTwice(directory.Node(), storage);
}

void CheckpointCrashes(Storage storage)
{
    for (const std::string stage :
         {"before_snapshots", "after_state_snapshot", "after_posting_snapshot", "after_publish", "after_log_cleanup"})
    {
        BOOST_TEST_CONTEXT("process crash at " << stage)
        {
            TestDirectory directory;
            {
                auto node = Open(directory.Node(), LimitsFor(storage));
                Create(*node, "A", 0, 2);
                Success(node->Close());
            }
            RunChild(directory.Node(), storage, "checkpoint", stage);
            RecoverTwice(directory.Node(), storage);
        }
    }
}

size_t ReassignedArchivedVersions(const std::string &snapshot, uint64_t index);

void MaintenanceCrashRecovery(Storage storage)
{
    TestDirectory directory;
    uint64_t index;
    {
        auto node = Open(directory.Node(), LimitsFor(storage));
        CreateMaintenanceTenant(*node);
        index = Identity(*node, "A");
        Success(node->Close());
    }
    RunChild(directory.Node(), storage, "maintenance");
    std::ifstream input(directory.path / "logical-state");
    BOOST_REQUIRE(input.is_open());
    size_t loadedHeads, savedHeads, reassigned;
    BOOST_REQUIRE(static_cast<bool>(input >> loadedHeads >> savedHeads >> reassigned));
    BOOST_REQUIRE_GT(savedHeads, loadedHeads);
    if (storage == Storage::TIKVIO)
    {
        // The worker crashed after exporting its final state but before publishing it.
        // Inspect that actual image in an independent work namespace before recovery
        // removes it. Replay below must still start at the original seed checkpoint.
        uint64_t latest = 0;
        for (auto &entry : fs::directory_iterator(directory.Node()))
        {
            auto name = entry.path().filename().string();
            if (name.starts_with("checkpoint-"))
                latest = (std::max)(latest, static_cast<uint64_t>(std::stoull(name.substr(11))));
        }
        const auto snapshot = fs::path(directory.Node()) / ("checkpoint-" + std::to_string(latest)) / "posting-store";
        reassigned = ReassignedArchivedVersions(snapshot.string(), index);
    }
    BOOST_REQUIRE_GT(reassigned, 0);
    BOOST_TEST_MESSAGE("Maintenance evidence: heads " << loadedHeads << " -> " << savedHeads
                                                      << ", reassigned live VIDs=" << reassigned);
    std::vector<SizeType> expected;
    SizeType id;
    while (input >> id)
        expected.push_back(id);
    BOOST_REQUIRE(input.eof());
    BOOST_REQUIRE(!expected.empty());
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        auto node = Open(directory.Node(), LimitsFor(storage));
        const auto actual = MaintenanceLogicalState(*node);
        BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.end(), expected.begin(), expected.end());
        SizeType next;
        Success(node->Insert("A", Data(1, 100 + attempt), 1, next));
        BOOST_CHECK_EQUAL(next, MaintenanceEnd + attempt);
    }
}
#endif

struct OwnerFixture
{
    TestDirectory directory;
    Storage storage;
    PostingBackendOptions options;
    std::string identity;
    uint64_t work = 0;
    std::shared_ptr<PostingBackend> owner;

    explicit OwnerFixture(Storage type)
        : storage(type), options(LimitsFor(type).posting), identity(directory.path.filename().string())
    {
        Success(OpenNext("", owner));
    }
    ~OwnerFixture()
    {
        if (storage != Storage::TIKVIO)
            return;
        // Owner API deliberately permits deleting only older work IDs. Use one
        // final owner to remove every populated namespace created by this test.
        std::shared_ptr<PostingBackend> cleanup;
        auto status = OpenNext("", cleanup);
        if (status == ErrorCode::Success)
            for (uint64_t id = 1; id < work; ++id)
                BOOST_CHECK(cleanup->RemoveWork(id) == ErrorCode::Success);
    }
    ErrorCode OpenNext(const std::string &snapshot, std::shared_ptr<PostingBackend> &result)
    {
        ++work;
        if (storage == Storage::FILEIO)
            return OpenFilePostingBackend((directory.path / ("work-" + std::to_string(work))).string(), snapshot,
                                          result);
        return OpenTiKVPostingBackend(options, identity, work, snapshot, result);
    }
    std::string Snapshot(const std::string &name = "snapshot") const
    {
        return (directory.path / name).string();
    }
    fs::path Image(const std::string &snapshot) const
    {
        return fs::path(snapshot) / (storage == Storage::FILEIO ? "postings.image" : "tikv-postings.bin");
    }
};

#ifndef _MSC_VER
size_t ReassignedArchivedVersions(const std::string &snapshot, uint64_t index)
{
#ifdef TIKV
    OwnerFixture fixture(Storage::TIKVIO);
    Success(fixture.OpenNext(snapshot, fixture.owner));
    COMMON::TiKVVersionMap versions(true);
    versions.SetDB(fixture.owner->View("A", index));
    Success(versions.Load("", 4096, 4096));
    BOOST_REQUIRE_EQUAL(versions.Count(), MaintenanceEnd);
    std::vector<SizeType> ids;
    Success(versions.GetContainedIDs(ids));
    std::vector<uint8_t> values;
    versions.BatchGetVersions(ids, values);
    BOOST_REQUIRE(!versions.Failed());
    return std::count_if(values.begin(), values.end(), [](auto v) { return v < 0x80; });
#else
    BOOST_FAIL("TiKV unavailable");
    return 0;
#endif
}
#endif

void Put(const std::shared_ptr<Helper::KeyValueIO> &view, SizeType key, const std::string &value)
{
    BOOST_REQUIRE(view);
    Success(view->Put(key, value, IOTimeout, nullptr));
}

std::string Get(const std::shared_ptr<Helper::KeyValueIO> &view, SizeType key)
{
    BOOST_REQUIRE(view);
    std::string value;
    Success(view->Get(key, &value, IOTimeout, nullptr));
    return value;
}

void Missing(const std::shared_ptr<Helper::KeyValueIO> &view, SizeType key)
{
    BOOST_REQUIRE(view);
    std::string value;
    BOOST_CHECK(view->Get(key, &value, IOTimeout, nullptr) == ErrorCode::Key_NotFound);
}

void OwnerIsolationAndFrozenSnapshot(Storage storage)
{
    OwnerFixture fixture(storage);
    auto first = fixture.owner->View("a", 1);
    auto second = fixture.owner->View("a", 2);
    auto otherTenant = fixture.owner->View(std::string("a\0b", 3), 1);
    const std::string binary("first\0posting", 13);
    Put(first, 7, binary);
    Put(second, 7, "second-index");
    Put(otherTenant, 7, "other-tenant");
    auto disposable = fixture.owner->View("disposable", 1);
    Put(disposable, 7, "discard");
    disposable->ShutDown();
    disposable.reset();
    BOOST_CHECK_EQUAL(Get(first, 7), binary);
    BOOST_CHECK_EQUAL(Get(second, 7), "second-index");
    BOOST_CHECK_EQUAL(Get(otherTenant, 7), "other-tenant");
    Success(fixture.owner->Checkpoint(fixture.Snapshot()));
    BOOST_CHECK(fixture.owner->Checkpoint(fixture.Snapshot()) != ErrorCode::Success);
    Put(first, 7, "overwritten-after-checkpoint");
    Put(first, 8, "new-after-checkpoint");
    Success(fixture.owner->DeleteIndex("a", 2));
    Missing(second, 7);
    Put(fixture.owner->View("a", 3), 7, "recreated-identity");
    BOOST_CHECK_EQUAL(Get(otherTenant, 7), "other-tenant");

    std::shared_ptr<PostingBackend> restored;
    Success(fixture.OpenNext(fixture.Snapshot(), restored));
    BOOST_CHECK_EQUAL(Get(restored->View("a", 1), 7), binary);
    Missing(restored->View("a", 1), 8);
    BOOST_CHECK_EQUAL(Get(restored->View("a", 2), 7), "second-index");
    Missing(restored->View("a", 3), 7);
    BOOST_CHECK_EQUAL(Get(restored->View(std::string("a\0b", 3), 1), 7), "other-tenant");
    // Restoration is another working namespace, never the frozen source.
    Put(restored->View("a", 1), 7, "mutated-restoration");
    Success(fixture.owner->DeleteIndex("a", 1));
    std::shared_ptr<PostingBackend> restoredAgain;
    Success(fixture.OpenNext(fixture.Snapshot(), restoredAgain));
    BOOST_CHECK_EQUAL(Get(restoredAgain->View("a", 1), 7), binary);
}

void Damage(const fs::path &image, bool truncate)
{
    fs::permissions(image, fs::perms::owner_write, fs::perm_options::add);
    if (truncate)
    {
        const auto size = fs::file_size(image);
        BOOST_REQUIRE(size > 1);
        fs::resize_file(image, size - 1);
        return;
    }
    std::fstream file(image, std::ios::in | std::ios::out | std::ios::binary);
    BOOST_REQUIRE(file.is_open());
    const auto offset = static_cast<std::streamoff>(fs::file_size(image) / 2);
    file.seekg(offset);
    char byte;
    file.read(&byte, 1);
    BOOST_REQUIRE(file.good());
    byte ^= 0x40;
    file.seekp(offset);
    file.write(&byte, 1);
    file.close();
    BOOST_REQUIRE(!file.fail());
}

void DamagedOwnerSnapshots(Storage storage)
{
    OwnerFixture fixture(storage);
    Put(fixture.owner->View("A", 1), 5, "valuable original posting");
    Success(fixture.owner->Checkpoint(fixture.Snapshot()));
    for (bool truncate : {false, true})
    {
        const auto bad = fixture.Snapshot(truncate ? "truncated" : "corrupt");
        fs::copy(fixture.Snapshot(), bad, fs::copy_options::recursive);
        Damage(fixture.Image(bad), truncate);
        std::shared_ptr<PostingBackend> rejected;
        BOOST_CHECK(fixture.OpenNext(bad, rejected) != ErrorCode::Success);
        BOOST_CHECK(!rejected);
    }
    std::shared_ptr<PostingBackend> restored;
    Success(fixture.OpenNext(fixture.Snapshot(), restored));
    BOOST_CHECK_EQUAL(Get(restored->View("A", 1), 5), "valuable original posting");
}

void DamagedManagedSnapshots(Storage storage)
{
    for (bool truncate : {false, true})
    {
        TestDirectory directory;
        {
            auto node = Open(directory.Node(), LimitsFor(storage));
            Create(*node);
            Success(node->Close());
        }
        std::string current;
        {
            std::shared_ptr<SharedRocksDB> recovery;
            Success(SharedRocksDB::Open(directory.Node() + "/recovery", {"recovery_log", "catalog", "recovery_meta"},
                                        recovery, false, rocksdb::NewLRUCache(1 << 20)));
            Success(recovery->Get("recovery_meta", "current", current));
        }
        uint64_t checkpoint;
        BOOST_REQUIRE(SharedRocksDB::DecodeNumber(current, checkpoint));
        const auto image = fs::path(directory.Node()) / ("checkpoint-" + std::to_string(checkpoint)) / "posting-store" /
                           (storage == Storage::FILEIO ? "postings.image" : "tikv-postings.bin");
        Damage(image, truncate);
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            std::unique_ptr<TenantIndexManager> rejected;
            BOOST_CHECK(TenantIndexManager::Open(directory.Node(), LimitsFor(storage), rejected) != ErrorCode::Success);
            BOOST_CHECK(!rejected);
        }
    }
}

void ConcurrentMerges(Storage storage)
{
    OwnerFixture fixture(storage);
    auto view = fixture.owner->View("A", 1);
    Put(view, 11, "base:");
    constexpr int Writers = 4;
    constexpr int Appends = 24;
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<std::vector<int>>> writers;
    for (int writer = 0; writer < Writers; ++writer)
        writers.emplace_back(std::async(std::launch::async, [&, writer] {
            std::vector<int> sizes;
            gate.wait();
            for (int i = 0; i < Appends; ++i)
            {
                int size = -1;
                auto status =
                    view->Merge(11, std::string(1, static_cast<char>('a' + writer)), IOTimeout, nullptr, size);
                if (status != ErrorCode::Success)
                    return std::vector<int>{-static_cast<int>(status) - 10000};
                sizes.push_back(size);
            }
            return sizes;
        }));
    start.set_value();
    std::set<int> lengths;
    for (auto &writer : writers)
    {
        const auto sizes = writer.get();
        BOOST_REQUIRE_EQUAL(sizes.size(), Appends);
        lengths.insert(sizes.begin(), sizes.end());
    }
    const auto value = Get(view, 11);
    BOOST_REQUIRE_GE(value.size(), 5);
    BOOST_CHECK_EQUAL(value.substr(0, 5), "base:");
    BOOST_CHECK_EQUAL(value.size(), 5 + Writers * Appends);
    for (int writer = 0; writer < Writers; ++writer)
        BOOST_CHECK_EQUAL(std::count(value.begin() + 5, value.end(), static_cast<char>('a' + writer)), Appends);
    BOOST_CHECK_EQUAL(lengths.size(), Writers * Appends);
    BOOST_REQUIRE(!lengths.empty());
    BOOST_CHECK_EQUAL(*lengths.begin(), 6);
    BOOST_CHECK_EQUAL(*lengths.rbegin(), value.size());

    // Put must replace, not append; subsequent Merge must see the replacement.
    Put(view, 11, "replacement");
    int size;
    Success(view->Merge(11, "+tail", IOTimeout, nullptr, size));
    BOOST_CHECK_EQUAL(Get(view, 11), "replacement+tail");
    BOOST_CHECK_EQUAL(size, 16);
}

void FileConcurrentPutMergeAndCompaction()
{
    ConcurrentMerges(Storage::FILEIO);
    OwnerFixture fixture(Storage::FILEIO);
    auto view = fixture.owner->View("A", 1);
    // For two simultaneous operations the only legal outcomes are Put alone,
    // or Put followed by Merge. Neither may publish a partial posting value.
    for (int round = 0; round < 24; ++round)
    {
        Put(view, 1, "old");
        std::promise<void> start;
        auto gate = start.get_future().share();
        auto put = std::async(std::launch::async, [&] {
            gate.wait();
            return view->Put(1, "new", IOTimeout, nullptr);
        });
        auto merge = std::async(std::launch::async, [&] {
            gate.wait();
            int size;
            return view->Merge(1, "+merge", IOTimeout, nullptr, size);
        });
        start.set_value();
        Success(put.get());
        Success(merge.get());
        const auto result = Get(view, 1);
        BOOST_CHECK(result == "new" || result == "new+merge");
    }
    for (int overwrite = 0; overwrite < 12; ++overwrite)
        Put(view, 2, std::string(256 << 10, static_cast<char>('a' + overwrite)));
    Put(view, 2, "live");
    Put(view, 3, std::string(256 << 10, 'z'));
    Success(view->Delete(3));
    const auto before = Get(view, 1);
    const auto workingBytes = [&] {
        uintmax_t bytes = 0;
        for (const auto &entry : fs::directory_iterator(fixture.directory.path / "work-1"))
            if (entry.is_regular_file())
                bytes += entry.file_size();
        return bytes;
    };
    BOOST_CHECK_GT(workingBytes(), 3ULL << 20);
    Success(fixture.owner->Checkpoint(fixture.Snapshot()));
    BOOST_CHECK_LT(fs::file_size(fixture.Image(fixture.Snapshot())), 64ULL << 10);
    BOOST_CHECK_LT(workingBytes(), 64ULL << 10);
    std::shared_ptr<PostingBackend> restored;
    Success(fixture.OpenNext(fixture.Snapshot(), restored));
    auto restoredView = restored->View("A", 1);
    BOOST_CHECK_EQUAL(Get(restoredView, 1), before);
    BOOST_CHECK_EQUAL(Get(restoredView, 2), "live");
    Missing(restoredView, 3);
    std::vector<std::string> batch;
    Success(restoredView->MultiGet(std::vector<SizeType>{2, 1, 2}, &batch, IOTimeout, nullptr));
    BOOST_REQUIRE_EQUAL(batch.size(), 3);
    BOOST_CHECK_EQUAL(batch[0], "live");
    BOOST_CHECK_EQUAL(batch[1], before);
    BOOST_CHECK_EQUAL(batch[2], "live");
    std::vector<Helper::PageBuffer<uint8_t>> pages(3);
    Success(restoredView->MultiGet(std::vector<SizeType>{2, 1, 2}, pages, IOTimeout, nullptr));
    BOOST_CHECK_EQUAL(pages[0].GetAvailableSize(), 4);
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(pages[0].GetBuffer()), 4), "live");
    BOOST_CHECK_EQUAL(pages[1].GetAvailableSize(), before.size());
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(pages[1].GetBuffer()), before.size()), before);
}

void FilePooledReadsAndFailedCompletion()
{
    OwnerFixture fixture(Storage::FILEIO);
    auto view = fixture.owner->View("A", 1);
    Put(view, 1, std::string(8193, 'a'));
    Put(view, 2, "second");
    const std::vector<SizeType> keys{1, 2, 1};
    for (int round = 0; round < 2; ++round)
    {
        std::promise<void> start;
        auto gate = start.get_future().share();
        std::vector<std::future<ErrorCode>> readers;
        for (int i = 0; i < 70; ++i)
            readers.push_back(std::async(std::launch::async, [&] {
                gate.wait();
                std::vector<std::string> values;
                auto result = view->MultiGet(keys, &values, IOTimeout, nullptr);
                if (result != ErrorCode::Success)
                    return result;
                return values == std::vector<std::string>{std::string(8193, 'a'), "second", std::string(8193, 'a')}
                           ? ErrorCode::Success
                           : ErrorCode::Fail;
            }));
        start.set_value();
        for (auto &reader : readers)
            Success(reader.get());
    }
    // EOF completions must all be drained before this pooled context is reused.
    fs::resize_file(fixture.directory.path / "work-1" / "postings.active", 0);
    std::vector<std::string> values;
    BOOST_CHECK(view->MultiGet(keys, &values, IOTimeout, nullptr) != ErrorCode::Success);
    Put(view, 3, "after-error");
    Success(view->MultiGet(std::vector<SizeType>{3, 3}, &values, IOTimeout, nullptr));
    BOOST_REQUIRE_EQUAL(values.size(), 2);
    BOOST_CHECK_EQUAL(values[0], "after-error");
    BOOST_CHECK_EQUAL(values[1], "after-error");
}

#ifndef _MSC_VER
void OwnerIndexMergeMaintenance(Storage storage)
{
    OwnerFixture fixture(storage);
    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::Float);
    BOOST_REQUIRE(index);
    const auto indexDirectory = (fixture.directory.path / "index").string();
    fs::create_directories(indexDirectory);
    const auto set = [&](const char *section, const char *name, const std::string &value) {
        Success(index->SetParameter(name, value.c_str(), section));
    };
    set("Base", "IndexDirectory", indexDirectory);
    set("Base", "IndexAlgoType", "BKT");
    set("Base", "DistCalcMethod", "L2");
    set("Base", "DataBlockSize", "4096");
    set("Base", "DataCapacity", "4096");
    set("SelectHead", "isExecute", "true");
    set("SelectHead", "SelectHeadType", "Random");
    set("SelectHead", "Count", "2");
    set("SelectHead", "NumberOfThreads", "1");
    set("BuildHead", "isExecute", "true");
    set("BuildHead", "DataBlockSize", "4096");
    set("BuildHead", "DataCapacity", "4096");
    set("BuildHead", "DeletePercentageForRefine", "2");
    set("BuildSSDIndex", "isExecute", "true");
    set("BuildSSDIndex", "BuildSsdIndex", "true");
    set("BuildSSDIndex", "TmpDir", indexDirectory);
    set("BuildSSDIndex", "MaxID", "4096");
    set("BuildSSDIndex", "NumberOfThreads", "1");
    set("BuildSSDIndex", "PostingPageLimit", "1");
    set("BuildSSDIndex", "ReplicaCount", "2");
    set("BuildSSDIndex", "MergeThreshold", "80");
    Success(index->SetPostingStore(fixture.owner->View("merge", 1)));
    // TenantIndexManager intentionally forbids search-triggered mutation.
    // An owner-backed public VectorIndex can explicitly enable it to exercise
    // posting-list consolidation rather than only the storage append operator.
    set("BuildSSDIndex", "AsyncMergeInSearch", "true");
    auto seeds = MaintenanceVectors(0, SeedCount);
    Success(index->BuildIndex(seeds.Data(), SeedCount, 4));
    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(index);
    BOOST_REQUIRE(spann);
    std::vector<SizeType> before, after;
    Success(spann->GetHeadIndexMapping(1, before));
    BOOST_REQUIRE_GT(before.size(), 1);
    for (SizeType id = 0; id < 16; ++id)
        Success(index->DeleteIndex(id));
    auto query = MaintenanceVectors(16, 1);
    std::vector<BasicResult> results(8);
    Success(index->SearchIndex(query.Data(), 1, 8, false, results.data()));
    Success(index->DrainUpdates());
    Success(spann->GetHeadIndexMapping(1, after));
    BOOST_REQUIRE_LT(after.size(), before.size());
    Success(index->SaveIndex(indexDirectory));
    Success(fixture.owner->Checkpoint(fixture.Snapshot()));
    Success(index->DeleteIndex(SizeType(16)));
    spann.reset();
    index.reset();
    std::shared_ptr<PostingBackend> restored;
    Success(fixture.OpenNext(fixture.Snapshot(), restored));
    Success(VectorIndex::LoadIndex(indexDirectory, index, restored->View("merge", 1)));
    for (SizeType id = 16; id < SeedCount; ++id)
    {
        query = MaintenanceVectors(id, 1);
        Success(index->SearchIndex(query.Data(), 1, 8, false, results.data()));
        BOOST_CHECK_EQUAL(results.front().VID, id);
        for (const auto &result : results)
            BOOST_CHECK(result.VID >= 16 && result.VID < SeedCount);
    }
}
#endif

void TiKVFullScan()
{
    OwnerFixture fixture(Storage::TIKVIO);
    auto view = fixture.owner->View("sparse", 1);
    constexpr int Count = 5000;
    constexpr int Workers = 4;
    std::vector<std::future<ErrorCode>> writers;
    for (int worker = 0; worker < Workers; ++worker)
        writers.emplace_back(std::async(std::launch::async, [&, worker] {
            for (int i = worker; i < Count; i += Workers)
            {
                auto status = view->Put(static_cast<SizeType>(i * 7919), "v" + std::to_string(i), IOTimeout, nullptr);
                if (status != ErrorCode::Success)
                    return status;
            }
            return ErrorCode::Success;
        }));
    for (auto &writer : writers)
        Success(writer.get());
    Put(fixture.owner->View("untouched", 1), 0, "neighbor");
    Success(fixture.owner->Checkpoint(fixture.Snapshot()));
    Success(fixture.owner->DeleteIndex("sparse", 1));
    Missing(view, 0);
    Missing(view, (Count - 1) * 7919);
    BOOST_CHECK_EQUAL(Get(fixture.owner->View("untouched", 1), 0), "neighbor");
    std::shared_ptr<PostingBackend> restored;
    Success(fixture.OpenNext(fixture.Snapshot(), restored));
    auto restoredView = restored->View("sparse", 1);
    std::vector<std::future<bool>> readers;
    for (int worker = 0; worker < Workers; ++worker)
        readers.emplace_back(std::async(std::launch::async, [&, worker] {
            for (int i = worker; i < Count; i += Workers)
            {
                std::string value;
                if (restoredView->Get(i * 7919, &value, IOTimeout, nullptr) != ErrorCode::Success ||
                    value != "v" + std::to_string(i))
                    return false;
            }
            return true;
        }));
    for (auto &reader : readers)
        BOOST_CHECK(reader.get());
    BOOST_TEST_MESSAGE("Verified all 5000 sparse TiKV postings after export, delete, and restore. "
                       "Region splitting is not exposed by PostingBackend; provision split Regions "
                       "in the real cluster to exercise cross-Region traversal with this same case.");
}
} // namespace

BOOST_AUTO_TEST_SUITE(PostingBackends)

#ifndef _MSC_VER
BOOST_AUTO_TEST_CASE(CrashWorker)
{
    const char *root = std::getenv("SPTAG_POSTING_WORKER_ROOT");
    if (!root)
        return;
    const char *storage = std::getenv("SPTAG_POSTING_WORKER_STORAGE");
    const char *mode = std::getenv("SPTAG_POSTING_WORKER_MODE");
    const char *stage = std::getenv("SPTAG_POSTING_WORKER_STAGE");
    BOOST_REQUIRE(storage && mode && stage);
    auto limits = LimitsFor(static_cast<Storage>(std::stoi(storage)));
    const std::string operation(mode), boundary(stage);
    if (operation == "recover")
        f_createIO = []() -> std::shared_ptr<Helper::DiskIO> { return std::make_shared<CrashDuringRecoverySave>(); };
    if (operation == "maintenance" && limits.posting.storage == Storage::TIKVIO)
        limits.checkpointHook = [](const char *at) {
            if (std::string_view(at) == "after_posting_snapshot") _exit(Crashed);
        };
    if (operation == "checkpoint")
        limits.checkpointHook = [boundary](const char *at) {
            if (boundary != at)
                return;
            if (boundary == "after_publish")
                kill(getpid(), SIGKILL);
            _exit(Crashed);
        };
    auto node = Open(root, limits);
    if (operation == "recover")
        BOOST_FAIL("recovery returned without reaching the injected SaveIndex crash");
    if (operation == "maintenance")
    {
        MaintenanceWorker(*node, root, limits.posting.storage == Storage::TIKVIO);
        _exit(Crashed);
    }
    AppendAndDelete(*node);
    if (operation == "checkpoint")
    {
        Success(node->Checkpoint());
        BOOST_FAIL("checkpoint did not reach the requested crash boundary");
    }
    BOOST_REQUIRE_EQUAL(operation, "write");
    _exit(Crashed);
}
#endif

BOOST_AUTO_TEST_CASE(FileManagerIsolationAndIdentity)
{
    ManagerIsolation(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileDestructorRecovery)
{
    DestructorRecovery(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileFailedCheckpointFences)
{
    FailedCheckpointFences(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileOwnerIsolationAndFrozenSnapshot)
{
    OwnerIsolationAndFrozenSnapshot(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileDamagedOwnerSnapshots)
{
    DamagedOwnerSnapshots(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileDamagedManagedSnapshots)
{
    DamagedManagedSnapshots(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileConcurrentPutMergeAndCompactSnapshot)
{
    FileConcurrentPutMergeAndCompaction();
}
BOOST_AUTO_TEST_CASE(FilePooledReadsDrainFailures)
{
    FilePooledReadsAndFailedCompletion();
}
BOOST_AUTO_TEST_CASE(FileMergeKeepsCompletePages)
{
    OwnerFixture fixture(Storage::FILEIO);
    auto view = fixture.owner->View("A", 1);
    const auto active = fixture.directory.path / "work-1" / "postings.active";
    std::string expected(8192, 'a');
    Put(view, 1, expected);
    Put(view, 2, "interleave");
    int size = 0;
    for (const std::string suffix : {std::string(4096, 'b'), std::string("c"), std::string(4096, 'd')})
    {
        const auto before = fs::file_size(active);
        const auto tailBytes = expected.size() % 4096 + suffix.size();
        Success(view->Merge(1, suffix, IOTimeout, nullptr, size));
        BOOST_CHECK_EQUAL(fs::file_size(active) - before, ((tailBytes + 4095) / 4096) * 4096);
        expected += suffix;
        BOOST_CHECK_EQUAL(size, expected.size());
        BOOST_CHECK(Get(view, 1) == expected);
    }
    std::vector<std::string> batch;
    Success(view->MultiGet({1, 2, 1}, &batch, IOTimeout, nullptr));
    BOOST_CHECK((batch == std::vector<std::string>{expected, "interleave", expected}));
    Success(fixture.owner->Checkpoint(fixture.Snapshot()));
    Put(view, 1, "new");
    std::shared_ptr<PostingBackend> restored;
    Success(fixture.OpenNext(fixture.Snapshot(), restored));
    BOOST_CHECK(Get(restored->View("A", 1), 1) == expected);
}
BOOST_AUTO_TEST_CASE(SparseLocalVersionMapBatchReads)
{
    SparseVersionBatches();
}
BOOST_AUTO_TEST_CASE(FileServiceContextSelectsBackend)
{
    ServiceBackendSelection(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileAutoCheckpoint)
{
    AutomaticCheckpoint(Storage::FILEIO);
}
#ifndef _MSC_VER
BOOST_AUTO_TEST_CASE(FileProcessCrashAndInterruptedRecovery)
{
    ProcessCrashRecovery(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileCheckpointCrashBoundaries)
{
    CheckpointCrashes(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileMaintenanceCrashReplay)
{
    MaintenanceCrashRecovery(Storage::FILEIO);
}
BOOST_AUTO_TEST_CASE(FileOwnerIndexMergeMaintenance)
{
    OwnerIndexMergeMaintenance(Storage::FILEIO);
}
#endif

BOOST_AUTO_TEST_CASE(TiKVManagerIsolationAndIdentity)
{
    if (TiKVEnabled())
        ManagerIsolation(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVVersionModeSurvivesRestore)
{
    if (!TiKVEnabled())
        return;
    for (bool local : {false, true})
    {
        TestDirectory directory;
        const auto limits = LimitsFor(Storage::TIKVIO);
        auto node = Open(directory.Node(), limits);
        Create(*node, "A", 0, 2, local);
        SizeType refused = -1;
        BOOST_CHECK(node->DeleteVector("A", SeedCount) == ErrorCode::VectorNotFound);
        BOOST_CHECK(node->Insert("A", Data(4096, 0), 4096, refused) == ErrorCode::MemoryOverFlow);
        BOOST_CHECK_EQUAL(refused, -1);
        AppendAndDelete(*node);
        Success(node->Close());
        node.reset();
        node = Open(directory.Node(), limits);
        CheckRecovered(*node);
        bool inspected = false;
        for (auto &work : fs::directory_iterator(directory.Node()))
        {
            if (!work.path().filename().string().starts_with("work-"))
                continue;
            auto path = work.path() / ("index-" + std::to_string(Identity(*node, "A")));
            if (!fs::exists(path / "indexloader.ini"))
                continue;
            Helper::IniReader config;
            Success(config.LoadIniFile((path / "indexloader.ini").string()));
            BOOST_CHECK_EQUAL(config.GetParameter("BuildSSDIndex", "DistributedVersionMap", local), !local);
            const auto file = config.GetParameter("Base", "DeletedIDs", std::string("DeletedIDs.bin"));
            BOOST_CHECK_EQUAL(fs::exists(path / (file + "_0")), local);
            // Neither a missing local file nor an empty TiKV namespace is a
            // valid substitute for this head checkpoint's version state.
            const auto damaged = directory.path / "damaged-head";
            fs::copy(path, damaged, fs::copy_options::recursive);
            if (local)
                BOOST_REQUIRE(fs::remove(damaged / (file + "_0")));
            OwnerFixture empty(Storage::TIKVIO);
            std::shared_ptr<VectorIndex> index;
            BOOST_CHECK(VectorIndex::LoadIndex(damaged.string(), index, empty.owner->View("A", 1)) != ErrorCode::Success);
            inspected = true;
        }
        BOOST_REQUIRE(inspected);
    }
}
#ifdef TIKV
BOOST_AUTO_TEST_CASE(TiKVVersionMetadataFailsClosed)
{
    if (!TiKVEnabled())
        return;
    OwnerFixture fixture(Storage::TIKVIO);
    auto view = fixture.owner->View("A", 1);
    COMMON::TiKVVersionMap versions(true);
    versions.SetDB(view);
    BOOST_REQUIRE(versions.Load("", 64, 64) != ErrorCode::Success);
    Success(versions.Initialize(8, 64, 64));
    SizeType legacyAlias = 0;
    std::memcpy(&legacyAlias, "vc:0", 4);
    Put(view, legacyAlias, "posting-not-count");
    Success(versions.Load("", 64, 64));
    BOOST_CHECK_EQUAL(versions.Count(), 8);
    BOOST_CHECK(versions.Load("", 64, 7) != ErrorCode::Success);
    Success(view->Put("version/vc:0", "bogus", IOTimeout, nullptr));
    BOOST_CHECK(versions.Load("", 64, 64) != ErrorCode::Success);
    BOOST_CHECK_EQUAL(Get(view, legacyAlias), "posting-not-count");
    view->ShutDown();
    BOOST_CHECK(versions.Save("") != ErrorCode::Success);
    BOOST_CHECK(versions.Failed());
}
#endif
BOOST_AUTO_TEST_CASE(TiKVDestructorRecovery)
{
    if (TiKVEnabled())
        DestructorRecovery(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVVersionGrowthRejectsBeforeLogging)
{
    if (!TiKVEnabled())
        return;
    TestDirectory directory;
    auto limits = LimitsFor(Storage::TIKVIO);
    auto node = Open(directory.Node(), limits);
    Create(*node);
    Nearest(*node, "A", 0, 0);
    TenantIndexManager::NodeStats stats;
    std::vector<TenantIndexManager::TenantStats> tenants;
    node->GetStats(stats, tenants);
    limits.maxResidentBytes = stats.peakChargedBytes;
    Success(node->Close());
    node.reset();
    node = Open(directory.Node(), limits);
    Nearest(*node, "A", 0, 0);
    node->GetStats(stats, tenants);
    const auto logBytes = stats.recoveryLogBytes;
    SizeType first = -7;
    BOOST_REQUIRE(node->Insert("A", Data(2048, 1), 2048, first) == ErrorCode::CapacityExceeded);
    node->GetStats(stats, tenants);
    BOOST_CHECK_EQUAL(first, -7);
    BOOST_CHECK_EQUAL(stats.recoveryLogBytes, logBytes);
    BOOST_CHECK_EQUAL(stats.reservedBytes, 0);
    BOOST_CHECK(!stats.failed);
    Success(node->Close());
    node.reset();
    node = Open(directory.Node(), LimitsFor(Storage::TIKVIO));
    Success(node->Insert("A", Data(1, InsertedBase), 1, first));
    BOOST_CHECK_EQUAL(first, SeedCount);
}
BOOST_AUTO_TEST_CASE(TiKVTopKRefillsAfterDeletions)
{
    if (TiKVEnabled())
        TiKVTopKAfterDeletions();
}
BOOST_AUTO_TEST_CASE(TiKVServiceContextSelectsBackend)
{
    if (TiKVEnabled())
        ServiceBackendSelection(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVAutoCheckpoint)
{
    if (TiKVEnabled())
        AutomaticCheckpoint(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVFailedCheckpointFences)
{
    if (TiKVEnabled())
        FailedCheckpointFences(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVOwnerIsolationAndFrozenSnapshot)
{
    if (TiKVEnabled())
        OwnerIsolationAndFrozenSnapshot(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVDamagedOwnerSnapshots)
{
    if (TiKVEnabled())
        DamagedOwnerSnapshots(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVDamagedManagedSnapshots)
{
    if (TiKVEnabled())
        DamagedManagedSnapshots(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVConcurrentMerges)
{
    if (TiKVEnabled())
        ConcurrentMerges(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVExportScanBeyond4096SparseKeys)
{
    if (TiKVEnabled())
        TiKVFullScan();
}
#ifndef _MSC_VER
BOOST_AUTO_TEST_CASE(TiKVProcessCrashAndInterruptedRecovery)
{
    if (TiKVEnabled())
        ProcessCrashRecovery(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVCheckpointCrashBoundaries)
{
    if (TiKVEnabled())
        CheckpointCrashes(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVMaintenanceCrashReplay)
{
    if (TiKVEnabled())
        MaintenanceCrashRecovery(Storage::TIKVIO);
}
BOOST_AUTO_TEST_CASE(TiKVOwnerIndexMergeMaintenance)
{
    if (TiKVEnabled())
        OwnerIndexMergeMaintenance(Storage::TIKVIO);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
#endif
