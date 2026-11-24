// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef MULTITENANT
#include "inc/Core/Common/LocalVersionMap.h"
#include "inc/Core/Common/PQQuantizer.h"
#include "inc/Core/Common/ResidentHashMap.h"
#include "inc/Core/Common/ResidentMetadata.h"
#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/SPANN/ExtraRocksDBController.h"
#include "inc/Core/SPANN/IExtraSearcher.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/TenantIndexManager.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Server/SearchExecutor.h"
#include "inc/Socket/RemoteSearchQuery.h"
#include "inc/Test.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <future>
#include <latch>
#include <map>
#include <set>
#include <zstd.h>
#ifndef _MSC_VER
#include <sys/wait.h>
#endif

using namespace SPTAG;
using namespace SPTAG::SPANN;

namespace
{
struct TemporaryNode
{
    std::string root =
        (boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("sptag-multitenant-%%%%-%%%%-%%%%"))
            .string();
    ~TemporaryNode()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

ByteArray Vectors(SizeType count, DimensionType dimension, float base)
{
    auto result = ByteArray::Alloc(static_cast<size_t>(count) * dimension * sizeof(float));
    auto *data = reinterpret_cast<float *>(result.Data());
    for (SizeType i = 0; i < count; ++i)
        for (DimensionType j = 0; j < dimension; ++j)
            data[i * dimension + j] = base + i * 0.1f + j * 0.01f;
    return result;
}

void RequireSuccess(ErrorCode status)
{
    BOOST_REQUIRE_EQUAL(static_cast<int>(status), static_cast<int>(ErrorCode::Success));
}

std::unique_ptr<TenantIndexManager> OpenNode(const std::string &path)
{
    TenantIndexManager::Limits limits;
    limits.maxTenants = 4;
    limits.maxVectorsPerTenant = 4096;
    limits.maxDimension = 32;
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(path, limits, node));
    return node;
}

std::unique_ptr<TenantIndexManager> OpenNode(const std::string &path, const TenantIndexManager::Limits &limits)
{
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(path, limits, node));
    return node;
}

TenantIndexManager::NodeStats NodeState(TenantIndexManager &node)
{
    TenantIndexManager::NodeStats state;
    std::vector<TenantIndexManager::TenantStats> tenants;
    node.GetStats(state, tenants);
    return state;
}

bool WaitUntil(const std::function<bool()> &ready)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready())
    {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

struct SnapshotProbe
{
    std::atomic<int> saves{0};
    std::atomic<bool> block{false};
    std::atomic<bool> blockLoad{false};
    std::atomic<bool> failLoad{false};
    std::atomic<bool> failSave{false};
    std::promise<void> entered;
    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();
};

class SnapshotFileIO : public Helper::SimpleFileIO
{
  public:
    explicit SnapshotFileIO(std::shared_ptr<SnapshotProbe> probe) : m_probe(std::move(probe))
    {
    }
    bool Initialize(const char *path, int mode, uint64_t maxIO = 1 << 20, uint32_t reads = 2, uint32_t writes = 2,
                    uint16_t threads = 4, uint64_t maxFile = 300ULL << 30) override
    {
        if ((mode & std::ios::in) && std::filesystem::path(path).filename() == "indexloader.ini" &&
            m_probe->blockLoad.exchange(false))
        {
            m_probe->entered.set_value();
            m_probe->released.wait();
        }
        if (std::filesystem::path(path).filename() == "indexloader.ini" &&
            (((mode & std::ios::in) && m_probe->failLoad) || ((mode & std::ios::out) && m_probe->failSave)))
            return false;
        if ((mode & std::ios::out) && std::filesystem::path(path).filename() == "indexloader.ini")
        {
            ++m_probe->saves;
            if (m_probe->block.exchange(false))
            {
                m_probe->entered.set_value();
                m_probe->released.wait();
            }
        }
        return SimpleFileIO::Initialize(path, mode, maxIO, reads, writes, threads, maxFile);
    }

  private:
    std::shared_ptr<SnapshotProbe> m_probe;
};

struct SnapshotFactory
{
    inline static std::shared_ptr<SnapshotProbe> active;
    decltype(f_createIO) original = f_createIO;
    explicit SnapshotFactory(std::shared_ptr<SnapshotProbe> probe)
    {
        active = std::move(probe);
        f_createIO = []() -> std::shared_ptr<Helper::DiskIO> { return std::make_shared<SnapshotFileIO>(active); };
    }
    ~SnapshotFactory()
    {
        f_createIO = original;
        active.reset();
    }
};
struct CountedWorkspace
{
    inline static std::atomic<int> alive{0};
    int value;
    explicit CountedWorkspace(int v) : value(v)
    {
        ++alive;
    }
    ~CountedWorkspace()
    {
        --alive;
    }
};
struct CountedMaintenanceJob : Helper::ThreadPool::Job
{
    std::atomic<int> &completed;
    explicit CountedMaintenanceJob(std::atomic<int> &count) : completed(count)
    {
    }
    void exec(IAbortOperation *) override
    {
        ++completed;
    }
    void exec(void *, IAbortOperation *abort) override
    {
        exec(abort);
    }
};
struct UnsupportedManagedQuantizer : COMMON::PQQuantizer<float>
{
    QuantizerType GetQuantizerType() const override
    {
        return QuantizerType::RaBitQQuantizer;
    }
};
} // namespace

BOOST_AUTO_TEST_SUITE(MultiTenant)

BOOST_AUTO_TEST_CASE(LoadedVersionMembershipFollowsDeletesAndReloads)
{
    TemporaryNode node;
    std::filesystem::create_directories(node.root);
    auto file = node.root + "/versions";
    COMMON::ManagedLocalVersionMap map;
    uint8_t version = 17;
    for (SizeType i = 0; i < 16; ++i)
        map.SetVersion(i, version);
    RequireSuccess(map.Save(file));
    RequireSuccess(map.Load(file, 8, 32));
    BOOST_CHECK(!map.Deleted(7));
    BOOST_CHECK_EQUAL(map.GetVersion(7), version);
    BOOST_REQUIRE(map.Delete(7));
    BOOST_CHECK(map.Deleted(7));
    BOOST_CHECK(!map.Deleted(8));
    RequireSuccess(map.Save(file));
    RequireSuccess(map.Load(file, 8, 32));
    BOOST_CHECK(map.Deleted(7));
    BOOST_CHECK(map.Deleted(-1));
    BOOST_CHECK(!map.Deleted(15));
    map.DeleteAll();
    BOOST_CHECK(map.Deleted(0));
}

BOOST_AUTO_TEST_CASE(ResidentAllocationReservationsAreDisjointAndRecyclable)
{
    COMMON::ResidentMemory memory(4096);
    auto account = memory.CreateAccount();
    COMMON::ResidentMemory::Reservation reservation;
    BOOST_REQUIRE(account->Reserve(3072, std::chrono::milliseconds(0), reservation) ==
                  COMMON::ResidentMemory::Admission::Success);
    COMMON::ResidentMemory::Allocation retained;
    {
        COMMON::ResidentMemory::Scope scope(account, &reservation);
        retained = COMMON::ResidentMemory::Charge(1024);
        {
            auto replacement = COMMON::ResidentMemory::Charge(2048);
            BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 3072);
            BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
            BOOST_CHECK_THROW(COMMON::ResidentMemory::Charge(1), std::bad_alloc);
        }
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 1024);
        BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 2048);
        auto replacement = COMMON::ResidentMemory::Charge(2048);
        BOOST_CHECK_EQUAL(memory.GetStats().peakBytes, 3072);
    }
    reservation.Reset();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 1024);
    BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
    std::thread reclaim([allocation = std::move(retained)]() mutable { allocation.Reset(); });
    reclaim.join();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentAllocationAdmissionRejectsOversizeAndTimesOut)
{
    COMMON::ResidentMemory memory(1024);
    auto first = memory.CreateAccount();
    auto second = memory.CreateAccount();
    auto pinned = first->Allocate(1024);
    COMMON::ResidentMemory::Reservation reservation;
    BOOST_CHECK(second->Reserve(1025, std::chrono::seconds(1), reservation) ==
                COMMON::ResidentMemory::Admission::Oversized);
    BOOST_CHECK(second->Reserve(1, std::chrono::milliseconds(10), reservation) ==
                COMMON::ResidentMemory::Admission::TimedOut);
    BOOST_CHECK_THROW(second->Allocate(1), std::bad_alloc);
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 1024);
    BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
    BOOST_CHECK_EQUAL(memory.GetStats().oversized, 1);
    BOOST_CHECK_EQUAL(memory.GetStats().timeouts, 1);
}

BOOST_AUTO_TEST_CASE(ResidentAllocationFreeWakesConcurrentAccounts)
{
    COMMON::ResidentMemory memory(1024);
    auto first = memory.CreateAccount();
    auto second = memory.CreateAccount();
    auto pinned = first->Allocate(1024);
    auto waiter = std::async(std::launch::async, [&] {
        COMMON::ResidentMemory::Reservation reservation;
        auto status = second->Reserve(1024, std::chrono::seconds(5), reservation);
        if (status != COMMON::ResidentMemory::Admission::Success)
            return false;
        COMMON::ResidentMemory::Scope scope(second, &reservation);
        auto allocation = COMMON::ResidentMemory::Charge(1024);
        return memory.GetStats().allocatedBytes == 1024 && memory.GetStats().reservedBytes == 0;
    });
    BOOST_REQUIRE(WaitUntil([&] { return memory.GetStats().waits != 0; }));
    pinned.Reset();
    BOOST_CHECK(waiter.get());
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
    BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
    BOOST_CHECK_LE(memory.GetStats().peakBytes, 1024);
}

BOOST_AUTO_TEST_CASE(ResidentAllocationReclaimsOutsideLedgerLockAndReservesAtomically)
{
    COMMON::ResidentMemory memory(1024);
    auto first = memory.CreateAccount();
    auto second = memory.CreateAccount();
    auto old = first->Allocate(1024);
    COMMON::ResidentMemory::Scope::Context context;
    context.account = second;
    context.reclaim = [&](uint64_t bytes) {
        old.Reset();
        COMMON::ResidentMemory::Reservation reservation;
        BOOST_REQUIRE(second->Reserve(bytes, std::chrono::milliseconds(0), reservation) ==
                      COMMON::ResidentMemory::Admission::Success);
        return reservation;
    };
    {
        COMMON::ResidentMemory::Scope scope(std::move(context));
        auto allocation = COMMON::ResidentMemory::Charge(768);
        BOOST_CHECK_EQUAL(first->GetStats().allocatedBytes, 0);
        BOOST_CHECK_EQUAL(second->GetStats().allocatedBytes, 768);
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 768);
        BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
        BOOST_CHECK(memory.GetStats().peakBytes <= 1024);
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentHashGrowthStaysWithinPreparedPeakAcrossRehashes)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    {
        COMMON::ResidentHashMap<int, int> map(account);
        const auto bound = map.GrowthBytes(64);
        COMMON::ResidentMemory::Reservation reservation;
        BOOST_REQUIRE(account->Reserve(bound, std::chrono::milliseconds(0), reservation) ==
                      COMMON::ResidentMemory::Admission::Success);
        COMMON::ResidentMemory::Scope scope(account, &reservation);
        for (int i = 0; i < 64; ++i)
            map.Set(i * 8, i);
        BOOST_CHECK(memory.GetStats().peakBytes <= bound);
        std::set<int> visited;
        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            BOOST_CHECK_EQUAL(map.at(iter->first), iter->second);
            visited.insert(iter->second);
        }
        BOOST_CHECK_EQUAL(visited.size(), 64);
        for (int i = 0; i < 64; i += 2)
            BOOST_CHECK(map.erase(i * 8));
        BOOST_CHECK_EQUAL(map.size(), 32);
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
    BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentHashFailedRehashKeepsPublishedEntriesAndCapacity)
{
    COMMON::ResidentMemory memory(4096);
    auto account = memory.CreateAccount();
    COMMON::ResidentHashMap<int, int> map(account);
    for (int i = 0; i < 8; ++i)
        map.Set(i, i + 10);
    const auto before = memory.GetStats().allocatedBytes;
    auto blocker = account->Allocate(4096 - before);
    BOOST_CHECK_THROW(map.Set(8, 18), std::bad_alloc);
    BOOST_CHECK_EQUAL(map.size(), 8);
    BOOST_CHECK_EQUAL(map.bucket_count(), 8);
    for (int i = 0; i < 8; ++i)
        BOOST_CHECK_EQUAL(map.at(i), i + 10);
    blocker.Reset();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, before);
    map.Set(8, 18);
    BOOST_CHECK_EQUAL(map.at(8), 18);
}

BOOST_AUTO_TEST_CASE(ResidentStringReplacementAdmitsBeforePublicationAndReleasesOldBacking)
{
    COMMON::ResidentMemory memory(1024);
    auto account = memory.CreateAccount();
    COMMON::ResidentStringOwners<1> owners(account);
    std::string value;
    owners.Assign(value, std::string(256, 'a'));
    const auto bytes = value.capacity() + 1;
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, bytes);
    auto blocker = memory.CreateAccount()->Allocate(1024 - bytes);
    BOOST_CHECK_NO_THROW(owners.Assign(value, std::string(256, 'a')));
    BOOST_CHECK_THROW(owners.Assign(value, std::string(256, 'b')), std::bad_alloc);
    BOOST_CHECK(value == std::string(256, 'a'));
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, bytes);
    blocker.Reset();
    owners.Assign(value, std::string(256, 'b'));
    BOOST_CHECK(value == std::string(256, 'b'));
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, value.capacity() + 1);
    owners.Assign(value, "");
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
    COMMON::ResidentStringOwners<1> adopted;
    std::string staged(256, 'c');
    adopted.Bind(account);
    adopted.Assign(staged, staged);
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, staged.capacity() + 1);
    adopted.Assign(staged, "");
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentArrayReplacementChargesOverlapAndPreservesFailedTarget)
{
    COMMON::ResidentMemory memory(1024);
    auto account = memory.CreateAccount();
    {
        COMMON::ResidentMemory::Scope scope(account);
        COMMON::ResidentArrayOwner owner;
        std::unique_ptr<uint64_t[]> data;
        owner.Allocate(data, 8);
        data[0] = 42;
        owner.Allocate(data, 16, [&](uint64_t *replacement) {
            BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 24 * sizeof(uint64_t));
            replacement[0] = data[0];
        });
        BOOST_CHECK_EQUAL(data[0], 42);
        BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 16 * sizeof(uint64_t));
        BOOST_CHECK_THROW(owner.Allocate(data, 200), std::bad_alloc);
        BOOST_CHECK_EQUAL(data[0], 42);
        BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 16 * sizeof(uint64_t));
    }
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentQuantizerRefusesUnsupportedManagedOwnership)
{
    TemporaryNode directory;
    std::shared_ptr<SharedRocksDB> engine;
    RequireSuccess(SharedRocksDB::Open(directory.root, {"postings"}, engine, true, rocksdb::NewLRUCache(1 << 20)));
    auto quantizer = std::make_shared<UnsupportedManagedQuantizer>();
    SPANN::Index<float> native;
    native.SetQuantizer(quantizer);
    BOOST_CHECK(native.SetPostingStore(engine->PostingView("native", 1)) == ErrorCode::FailedParseValue);
    SPANN::Index<float> managed;
    RequireSuccess(managed.SetPostingStore(engine->PostingView("managed", 1)));
    BOOST_CHECK_THROW(managed.SetQuantizer(quantizer), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ResidentWorkspaceRegistrationReusesExitedWorkersWithoutGrowingMetadata)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    const auto initial = CountedWorkspace::alive.load();
    {
        COMMON::ResidentMemory::Scope scope(account);
        auto factory = std::allocate_shared<COMMON::InstanceThreadLocalWorkSpaceFactory<CountedWorkspace>>(
            COMMON::ResidentAllocator<COMMON::InstanceThreadLocalWorkSpaceFactory<CountedWorkspace>>());
        const auto bytes = account->GetStats().allocatedBytes;
        std::atomic<int> cached{0};
        for (int i = 0; i < 65; ++i)
        {
            std::thread worker([&] {
                factory->ReturnWorkSpace(std::make_unique<CountedWorkspace>(42));
                auto workspace = factory->GetWorkSpace();
                if (workspace)
                {
                    ++cached;
                    factory->ReturnWorkSpace(std::move(workspace));
                }
            });
            worker.join();
        }
        BOOST_CHECK_EQUAL(cached.load(), 65);
        BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, bytes);
    }
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
    BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), initial);
}

BOOST_AUTO_TEST_CASE(ResidentWorkspaceResourcesAreReclaimedAcrossWorkerWaves)
{
    struct Workspace
    {
        std::shared_ptr<std::atomic<int>> available;
        explicit Workspace(std::shared_ptr<std::atomic<int>> count) : available(std::move(count))
        {
        }
        ~Workspace()
        {
            available->fetch_add(1);
        }
    };
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    auto available = std::make_shared<std::atomic<int>>(16);
    std::atomic<int> failures{0};
    {
        COMMON::ResidentMemory::Scope scope(account);
        COMMON::InstanceThreadLocalWorkSpaceFactory<Workspace> factory;
        const auto bytes = account->GetStats().allocatedBytes;
        for (int wave = 0; wave < 6; ++wave)
        {
            std::latch ready(16);
            std::promise<void> release;
            auto done = release.get_future().share();
            std::vector<std::thread> workers;
            for (int i = 0; i < 16; ++i)
                workers.emplace_back([&] {
                    auto workspace = factory.GetWorkSpace();
                    if (!workspace)
                    {
                        if (available->fetch_sub(1) > 0)
                            workspace = std::make_unique<Workspace>(available);
                        else
                        {
                            available->fetch_add(1);
                            ++failures;
                        }
                    }
                    ready.count_down();
                    done.wait();
                    factory.ReturnWorkSpace(std::move(workspace));
                });
            ready.wait();
            release.set_value();
            for (auto &worker : workers)
                worker.join();
            BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, bytes);
        }
        BOOST_CHECK_EQUAL(failures.load(), 0);
    }
    BOOST_CHECK_EQUAL(available->load(), 16);
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentWorkspaceConcurrentRegistrationIsBoundedAndRetiresWhileWorkersLive)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    const auto initial = CountedWorkspace::alive.load();
    COMMON::ResidentMemory::Scope scope(account);
    auto factory = std::allocate_shared<COMMON::InstanceThreadLocalWorkSpaceFactory<CountedWorkspace>>(
        COMMON::ResidentAllocator<COMMON::InstanceThreadLocalWorkSpaceFactory<CountedWorkspace>>());
    const auto bytes = account->GetStats().allocatedBytes;
    std::latch ready(65);
    std::promise<void> release;
    auto done = release.get_future().share();
    std::atomic<int> cached{0};
    std::vector<std::thread> workers;
    auto *owner = factory.get();
    for (int i = 0; i < 65; ++i)
        workers.emplace_back([&] {
            owner->ReturnWorkSpace(std::make_unique<CountedWorkspace>(42));
            auto workspace = owner->GetWorkSpace();
            if (workspace)
            {
                ++cached;
                owner->ReturnWorkSpace(std::move(workspace));
            }
            ready.count_down();
            done.wait();
        });
    ready.wait();
    BOOST_CHECK_EQUAL(cached.load(), 64);
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, bytes);
    factory.reset();
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
    BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), initial);
    release.set_value();
    for (auto &worker : workers)
        worker.join();
}

BOOST_AUTO_TEST_CASE(ResidentWorkspacePublicationDoesNotRequireJoiningExitedWorkers)
{
    struct ExitSignal
    {
        std::atomic<int> *exits;
        ~ExitSignal()
        {
            exits->fetch_add(1, std::memory_order_relaxed);
        }
    };
    struct Workspace
    {
        std::shared_ptr<std::atomic<int>> available, failures;
        int payload = 0;
        ~Workspace()
        {
            if (payload != 42)
                failures->fetch_add(1, std::memory_order_relaxed);
            available->fetch_add(1, std::memory_order_relaxed);
        }
    };
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    auto available = std::make_shared<std::atomic<int>>(1);
    auto failures = std::make_shared<std::atomic<int>>(0);
    std::atomic<int> exits{0};
    bool timedOut = false;
    {
        COMMON::ResidentMemory::Scope scope(account);
        COMMON::InstanceThreadLocalWorkSpaceFactory<Workspace> factory;
        const auto bytes = account->GetStats().allocatedBytes;
        std::vector<std::thread> retired;
        for (int i = 0; i < 96; ++i)
        {
            retired.emplace_back([&] {
                thread_local ExitSignal signal{&exits};
                auto workspace = factory.GetWorkSpace();
                if (!workspace)
                {
                    if (available->fetch_sub(1, std::memory_order_relaxed) <= 0)
                    {
                        available->fetch_add(1, std::memory_order_relaxed);
                        failures->fetch_add(1, std::memory_order_relaxed);
                        return;
                    }
                    workspace = std::make_unique<Workspace>();
                    workspace->available = available;
                    workspace->failures = failures;
                }
                workspace->payload = 42;
                factory.ReturnWorkSpace(std::move(workspace));
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (exits.load(std::memory_order_relaxed) != i + 1 && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            if (exits.load(std::memory_order_relaxed) != i + 1)
            {
                timedOut = true;
                break;
            }
        }
        for (auto &worker : retired)
            worker.join();
        BOOST_CHECK(!timedOut);
        BOOST_CHECK_EQUAL(failures->load(), 0);
        BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, bytes);
    }
    BOOST_CHECK_EQUAL(available->load(), 1);
    BOOST_CHECK_EQUAL(account->GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentPreparationFollowsBackgroundOwnersPastSubmittingScope)
{
    COMMON::ResidentMemory memory(4096);
    auto account = memory.CreateAccount();
    COMMON::ResidentMemory::Reservation reservation;
    BOOST_REQUIRE(account->Reserve(4096, std::chrono::milliseconds(0), reservation) ==
                  COMMON::ResidentMemory::Admission::Success);
    COMMON::ResidentMemory::Scope::Context background;
    {
        COMMON::ResidentMemory::Scope scope(account, &reservation);
        background = COMMON::ResidentMemory::Scope::Capture();
    }
    reservation.Reset();
    BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 4096);
    auto worker = std::async(std::launch::async, [context = std::move(background)] {
        COMMON::ResidentMemory::Scope scope(context);
        return COMMON::ResidentMemory::Charge(1024);
    });
    auto backing = worker.get();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 1024);
    BOOST_CHECK_EQUAL(memory.GetStats().reservedBytes, 0);
    backing.Reset();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentContainerCapacityKeepsOldAndNewBackingCharged)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    using Storage = std::vector<uint64_t, COMMON::ResidentAllocator<uint64_t>>;
    {
        COMMON::ResidentMemory::Scope scope(account);
        Storage values;
        values.reserve(128);
        const auto firstCharge = memory.GetStats().allocatedBytes;
        BOOST_CHECK_GE(firstCharge, 128 * sizeof(uint64_t));
        values.reserve(256);
        const auto nextCharge = memory.GetStats().allocatedBytes;
        BOOST_CHECK_GE(nextCharge, 256 * sizeof(uint64_t));
        BOOST_CHECK_EQUAL(memory.GetStats().peakBytes, firstCharge + nextCharge);
        values.clear();
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, nextCharge);
        std::thread reclaim([values = std::move(values)] {});
        reclaim.join();
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentLazyLockBackingKeepsCapturedOwnership)
{
    COMMON::ResidentMemory memory(16 << 20);
    auto account = memory.CreateAccount();
    std::unique_ptr<COMMON::FineGrainedRWLock> locks;
    std::unique_ptr<COMMON::FineGrainedLock> mutexes;
    {
        COMMON::ResidentMemory::Scope scope(account);
        locks = std::make_unique<COMMON::FineGrainedRWLock>();
        mutexes = std::make_unique<COMMON::FineGrainedLock>();
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
    BOOST_CHECK_GT(locks->ResidentGrowthBytes(1), 0);
    BOOST_CHECK_GT(mutexes->ResidentGrowthBytes(), 0);
    const auto &readOnly = *locks;
    auto *first = &readOnly[17];
    BOOST_CHECK_EQUAL(reinterpret_cast<std::uintptr_t>(first) % alignof(std::shared_timed_mutex), 0);
    const auto withLazyLock = memory.GetStats().allocatedBytes;
    BOOST_CHECK_GE(withLazyLock, 4096 * 64);
    BOOST_CHECK(first == &readOnly[17]);
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, withLazyLock);
    BOOST_CHECK(&readOnly[18] != first);
    BOOST_CHECK_GT(memory.GetStats().allocatedBytes, withLazyLock);
    const auto beforeMutexes = memory.GetStats().allocatedBytes;
    (void)(*mutexes)[5];
    BOOST_CHECK_GT(memory.GetStats().allocatedBytes, beforeMutexes);
    BOOST_CHECK_EQUAL(mutexes->ResidentGrowthBytes(), 0);
    std::thread reclaim([locks = std::move(locks), mutexes = std::move(mutexes)] {});
    reclaim.join();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentTreeRebuildTracksRetainedAndReplacementBacking)
{
    COMMON::ResidentMemory memory(16 << 20);
    auto account = memory.CreateAccount();
    {
        COMMON::ResidentMemory::Scope scope(account);
        COMMON::Dataset<float> data;
        auto vectors = Vectors(64, 4, 0);
        RequireSuccess(data.Initialize(64, 4, 64, 64, vectors.Data(), false));
        COMMON::BKTree tree;
        tree.m_iTreeNumber = 1;
        tree.BuildTrees(data, DistCalcMethod::L2, 1);
        const auto first = memory.GetStats().allocatedBytes;
        tree.Rebuild(data, DistCalcMethod::L2, nullptr);
        BOOST_CHECK_GT(memory.GetStats().peakBytes, first);
        BOOST_CHECK_GT(memory.GetStats().allocatedBytes, data.AllocatedBackingBytes());
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentDatasetChargesBlocksAndPreservesSharedBacking)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    auto vectors = std::make_unique<COMMON::Dataset<float>>();
    auto graph = std::make_unique<COMMON::Dataset<SizeType>>();
    uint64_t sharedCharge = 0;
    {
        COMMON::ResidentMemory::Scope scope(account);
        RequireSuccess(vectors->Initialize(2, 2, 4, 16, nullptr, true, nullptr, 0, 32));
        RequireSuccess(graph->InitializeSharedView(*vectors, 2, 8));
        (*vectors)[0][0] = 3.5f;
        (*graph)[0][0] = 7;
        sharedCharge = memory.GetStats().allocatedBytes;
        BOOST_CHECK_GE(sharedCharge, 64);
        BOOST_CHECK_EQUAL(vectors->AllocatedBackingBytes(), 64);
        BOOST_CHECK_EQUAL(graph->AllocatedBackingBytes(), 64);
        RequireSuccess(vectors->AddBatch(1));
        RequireSuccess(graph->AddBatch(1));
        (*vectors)[2][0] = 8.5f;
        (*graph)[2][0] = 11;
        BOOST_CHECK_EQUAL(vectors->AllocatedBackingBytes(), 64 + 4 * 32);
        BOOST_CHECK_EQUAL(graph->AllocatedBackingBytes(), 64 + 4 * 32);
        BOOST_CHECK_GT(memory.GetStats().allocatedBytes, sharedCharge);
        sharedCharge = memory.GetStats().allocatedBytes;
    }
    vectors.reset();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, sharedCharge);
    BOOST_CHECK_EQUAL((*graph)[0][0], 7);
    BOOST_CHECK_EQUAL((*graph)[2][0], 11);
    graph.reset();
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentDatasetFailedGrowthAndReplacementReleaseOnlyNewMemory)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    {
        COMMON::ResidentMemory::Scope scope(account);
        COMMON::Dataset<float> data;
        RequireSuccess(data.Initialize(2, 2, 4, 16));
        data[0][0] = 9;
        const auto before = memory.GetStats().allocatedBytes;
        auto pinned = account->Allocate(memory.GetStats().capacityBytes - before);
        BOOST_CHECK(data.AddBatch(1) == ErrorCode::MemoryOverFlow);
        BOOST_CHECK(data.Initialize(4, 2, 4, 16) == ErrorCode::MemoryOverFlow);
        BOOST_CHECK_EQUAL(data.R(), 2);
        BOOST_CHECK_EQUAL(data[0][0], 9);
        pinned.Reset();
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, before);
        RequireSuccess(data.AddBatch(1));
        BOOST_CHECK_EQUAL(data.R(), 3);
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentDatasetPreparesAdmittedRowsWithoutPublishingThem)
{
    COMMON::ResidentMemory memory(1 << 20);
    auto account = memory.CreateAccount();
    {
        COMMON::ResidentMemory::Scope scope(account);
        COMMON::Dataset<float> data;
        RequireSuccess(data.Initialize(2, 2, 4, 1024));
        RequireSuccess(data.ReserveRows(9));
        const auto prepared = memory.GetStats().allocatedBytes;
        BOOST_CHECK_EQUAL(data.R(), 2);
        BOOST_CHECK_EQUAL(data.AllocatedBackingBytes(), (2 + 8) * 2 * sizeof(float));
        RequireSuccess(data.AddBatch(7));
        BOOST_CHECK_EQUAL(data.R(), 9);
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, prepared);
        RequireSuccess(data.ReserveRows(9));
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, prepared);
        BOOST_CHECK(data.ReserveRows(1025) == ErrorCode::MemoryOverFlow);
        BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, prepared);
    }
    BOOST_CHECK_EQUAL(memory.GetStats().allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentTopHeadPromotionIsUniqueAcrossConcurrentSubmitters)
{
    TemporaryNode directory;
    boost::filesystem::create_directories(directory.root + "/index");
    std::shared_ptr<SharedRocksDB> engine;
    RequireSuccess(SharedRocksDB::Open(directory.root + "/db", {"postings"}, engine, true, nullptr));
    SPANN::Index<float> index;
    RequireSuccess(index.SetParameter("IndexDirectory", (directory.root + "/index").c_str(), "Base"));
    RequireSuccess(index.SetParameter("ValueType", "Float", "Base"));
    RequireSuccess(index.SetParameter("Dim", "4", "Base"));
    RequireSuccess(index.SetParameter("DistCalcMethod", "L2", "Base"));
    RequireSuccess(index.SetParameter("IndexAlgoType", "BKT", "Base"));
    RequireSuccess(index.SetParameter("DataCapacity", "256", "Base"));
    RequireSuccess(index.SetParameter("DataBlockSize", "16", "Base"));
    RequireSuccess(index.SetParameter("isExecute", "true", "SelectHead"));
    RequireSuccess(index.SetParameter("isExecute", "true", "BuildHead"));
    RequireSuccess(index.SetParameter("DataCapacity", "256", "BuildHead"));
    RequireSuccess(index.SetParameter("DataBlockSize", "16", "BuildHead"));
    RequireSuccess(index.SetParameter("isExecute", "true", "BuildSSDIndex"));
    RequireSuccess(index.SetParameter("Layers", "1", "BuildSSDIndex"));
    RequireSuccess(index.SetPostingStore(engine->PostingView("tenant", 0)));
    auto seed = Vectors(128, 4, 0);
    RequireSuccess(index.BuildIndex(seed.Data(), 128, 4));
    auto promoted = Vectors(1, 4, 500);
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<ErrorCode>> submitters;
    for (int i = 0; i < 8; ++i)
        submitters.emplace_back(std::async(std::launch::async, [&] {
            gate.wait();
            return index.AddHeadIndex(promoted.Data(), 128, 0xff, 4, 1, nullptr);
        }));
    start.set_value();
    for (auto &submitter : submitters)
        RequireSuccess(submitter.get());
    std::vector<SizeType> heads;
    RequireSuccess(index.GetHeadIndexMapping(1, heads));
    BOOST_CHECK_EQUAL(std::count(heads.begin(), heads.end(), 128), 1);
}

BOOST_AUTO_TEST_CASE(MaintenancePoolsStopAfterStartupAndDrain, *boost::unit_test::timeout(60))
{
    std::atomic<int> completed{0};
    for (int round = 0; round < 512; ++round)
    {
        {
            Helper::ThreadPool idle;
            idle.init(2);
        }
        Helper::ThreadPool pool;
        pool.init(2);
        pool.add(new CountedMaintenanceJob(completed));
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!pool.allClear() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        BOOST_REQUIRE(pool.allClear());
    }
    BOOST_CHECK_EQUAL(completed.load(), 512);
}

BOOST_AUTO_TEST_CASE(InstanceWorkspacesAreIsolatedAndReclaimed)
{
    using Factory = COMMON::InstanceThreadLocalWorkSpaceFactory<CountedWorkspace>;
    {
        Factory first, second;
        first.ReturnWorkSpace(std::make_unique<CountedWorkspace>(1));
        second.ReturnWorkSpace(std::make_unique<CountedWorkspace>(2));
        auto outer = first.GetWorkSpace();
        BOOST_REQUIRE(outer);
        BOOST_CHECK_EQUAL(outer->value, 1);
        BOOST_CHECK(!first.GetWorkSpace());
        first.ReturnWorkSpace(std::make_unique<CountedWorkspace>(3));
        first.ReturnWorkSpace(std::move(outer));
        BOOST_CHECK_EQUAL(first.GetWorkSpace()->value, 1);
        BOOST_CHECK_EQUAL(second.GetWorkSpace()->value, 2);
        std::thread worker([&] { first.ReturnWorkSpace(std::make_unique<CountedWorkspace>(4)); });
        worker.join();
        BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), 0);
    }
    auto factory = std::make_unique<Factory>();
    std::promise<void> cached, finish;
    auto done = finish.get_future();
    std::thread worker([&] {
        factory->ReturnWorkSpace(std::make_unique<CountedWorkspace>(5));
        cached.set_value();
        done.wait();
    });
    cached.get_future().wait();
    BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), 1);
    factory.reset();
    BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), 0);
    finish.set_value();
    worker.join();
}

BOOST_AUTO_TEST_CASE(WorkerHeadScratchReusesStorageAcrossConfigurations)
{
    std::thread worker([] {
        int small = 256, large = 300, exponent = 2;
        COMMON::WorkerWorkSpaceFactory first(small, small, exponent), second(large, large, exponent);
        auto workspace = first.GetWorkSpace();
        auto *identity = workspace.get();
        workspace->Reset(small, 10);
        BOOST_CHECK(!workspace->CheckAndSet(42));
        BOOST_CHECK(workspace->CheckAndSet(42));
        auto capture = std::make_shared<int>(1);
        std::weak_ptr<int> weak = capture;
        workspace->m_filterFunc = [capture](const ByteArray &) { return true; };
        capture.reset();
        first.ReturnWorkSpace(std::move(workspace));
        BOOST_CHECK(weak.expired());

        workspace = second.GetWorkSpace();
        BOOST_CHECK_EQUAL(workspace.get(), identity);
        workspace->Reset(large, 10);
        BOOST_CHECK(!workspace->CheckAndSet(42));
        for (int i = 0; i < large * 10 + 5; ++i)
            workspace->m_SPTQueue.insert(NodeDistPair(i, float(i)));
        BOOST_CHECK_EQUAL(workspace->m_SPTQueue.size(), large * 10);
        auto nested = second.GetWorkSpace();
        BOOST_CHECK_NE(nested.get(), workspace.get());
        nested->Reset(large, 10);
        BOOST_CHECK(!nested->CheckAndSet(42));
        BOOST_CHECK(workspace->CheckAndSet(42));
        nested.reset();
        second.ReturnWorkSpace(std::move(workspace));

        workspace = first.GetWorkSpace();
        BOOST_CHECK_EQUAL(workspace.get(), identity);
        workspace->Reset(small, 10);
        for (int i = 0; i < small * 10 + 5; ++i)
            workspace->m_SPTQueue.insert(NodeDistPair(i, float(i)));
        BOOST_CHECK_EQUAL(workspace->m_SPTQueue.size(), small * 10);
        first.ReturnWorkSpace(std::move(workspace));
    });
    worker.join();
}

BOOST_AUTO_TEST_CASE(WorkerDeduperReusePreservesInstanceIOAndIteratorState)
{
    std::thread worker([] {
        COMMON::InstanceThreadLocalWorkSpaceFactory<ExtraWorkSpace> factory;
        auto workspace = std::make_unique<ExtraWorkSpace>();
        workspace->UseWorkerScratch(128, 2);
        workspace->Initialize(128, 2, 1, 4096, true, false);
        auto *identity = &workspace->Deduper();
        auto *buffer = workspace->m_pageBuffers[0].GetBuffer();
        workspace->m_diskRequests[0].m_status = 17;
        int callbacks = 0;
        workspace->m_callback = [&] { ++callbacks; };
        BOOST_CHECK(!workspace->Deduper().CheckAndSet(42));
        factory.ReturnWorkSpace(std::move(workspace));
        BOOST_CHECK_EQUAL(callbacks, 0);

        ExtraWorkSpace other;
        other.UseWorkerScratch(256, 1);
        BOOST_CHECK_EQUAL(&other.Deduper(), identity);
        BOOST_CHECK_EQUAL(other.Deduper().HashTableExponent(), 1);
        BOOST_CHECK(!other.Deduper().CheckAndSet(42));

        workspace = factory.GetWorkSpace();
        workspace->UseWorkerScratch(128, 2);
        BOOST_CHECK_NE(&workspace->Deduper(), &other.Deduper());
        BOOST_CHECK_EQUAL(workspace->m_pageBuffers[0].GetBuffer(), buffer);
        BOOST_CHECK_EQUAL(workspace->m_diskRequests[0].m_status, 17);
        BOOST_CHECK_EQUAL(callbacks, 0);
        workspace->Deduper().clear();
        BOOST_CHECK(!workspace->Deduper().CheckAndSet(42));

        ExtraWorkSpace::IteratorLayerState layer;
        layer.Reset(128, 2);
        workspace->m_deduperOverride = layer.m_deduper.get();
        BOOST_CHECK(!workspace->Deduper().CheckAndSet(42));
        BOOST_CHECK(workspace->Deduper().CheckAndSet(42));
        other.Deduper().clear();
        BOOST_CHECK(workspace->Deduper().CheckAndSet(42));
        workspace->m_deduperOverride = nullptr;
        BOOST_CHECK(workspace->Deduper().CheckAndSet(42));
        other.ReleaseScratch();
        workspace.reset();
        BOOST_CHECK_EQUAL(callbacks, 1);

        ExtraWorkSpace larger, smaller;
        larger.UseWorkerScratch(64, 2);
        identity = &larger.Deduper();
        const auto capacity = larger.Deduper().StorageBytes();
        for (int id = 0; id < 32; ++id)
            BOOST_CHECK(!larger.Deduper().CheckAndSet(id));
        larger.ReleaseScratch();
        smaller.UseWorkerScratch(16, 2);
        BOOST_CHECK_EQUAL(&smaller.Deduper(), identity);
        BOOST_CHECK_EQUAL(smaller.Deduper().StorageBytes(), capacity);
        BOOST_CHECK(!smaller.Deduper().CheckAndSet(7));
        smaller.ReleaseScratch();
        larger.UseWorkerScratch(64, 2);
        BOOST_CHECK_EQUAL(&larger.Deduper(), identity);
        for (int id = 0; id < 32; ++id)
            BOOST_CHECK(!larger.Deduper().CheckAndSet(id));
        larger.Deduper().DoubleSize();
        const auto expanded = larger.Deduper().StorageBytes();
        larger.UseWorkerScratch(64, 2);
        BOOST_CHECK_EQUAL(larger.Deduper().HashTableExponent(), 2);
        BOOST_CHECK_EQUAL(larger.Deduper().StorageBytes(), expanded);
        BOOST_CHECK(!larger.Deduper().CheckAndSet(7));
    });
    worker.join();
}

BOOST_AUTO_TEST_CASE(WorkerScratchPoolsAreThreadLocalAndBounded)
{
    std::promise<void> cached, release;
    auto done = release.get_future();
    COMMON::WorkSpace *identity = nullptr;
    int budget = 256, exponent = 2;
    std::thread first([&] {
        COMMON::WorkerWorkSpaceFactory factory(budget, budget, exponent);
        auto workspace = factory.GetWorkSpace();
        identity = workspace.get();
        factory.ReturnWorkSpace(std::move(workspace));
        cached.set_value();
        done.wait();
        workspace = factory.GetWorkSpace();
        BOOST_CHECK_EQUAL(workspace.get(), identity);
    });
    cached.get_future().wait();
    std::thread second([&] {
        COMMON::WorkerWorkSpaceFactory factory(budget, budget, exponent);
        auto workspace = factory.GetWorkSpace();
        BOOST_CHECK_NE(workspace.get(), identity);
        factory.ReturnWorkSpace(std::move(workspace));

        using Pool = COMMON::WorkerSizeClassPool<CountedWorkspace>;
        Pool::Return(std::make_unique<CountedWorkspace>(1), 4096);
        auto held = Pool::Rent(4096);
        BOOST_REQUIRE(held);
        BOOST_CHECK(!Pool::Rent(4096));
        Pool::Return(std::make_unique<CountedWorkspace>(2), 4096);
        Pool::Return(std::make_unique<CountedWorkspace>(3), 4096);
        BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), 2);
        Pool::Return(std::make_unique<CountedWorkspace>(4), std::size_t(1) << 30);
        BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), 2);
        BOOST_CHECK(!Pool::Rent(std::size_t(1) << 30));
    });
    second.join();
    release.set_value();
    first.join();
    BOOST_CHECK_EQUAL(CountedWorkspace::alive.load(), 0);
}

BOOST_AUTO_TEST_CASE(WorkerBuffersRebindAcrossConfigurationsWithoutMovingIOOwnership)
{
    std::thread worker([] {
        COMMON::InstanceThreadLocalWorkSpaceFactory<ExtraWorkSpace> factory;
        auto first = std::make_unique<ExtraWorkSpace>();
        first->EnableWorkerBuffers();
        first->Initialize(128, 2, 1, 8192, false, true);
        auto *page = first->m_pageBuffers[0].GetBuffer();
        auto *decompression = first->m_decompressBuffer.GetBuffer();
        BOOST_CHECK_EQUAL(reinterpret_cast<std::uintptr_t>(page) % PageSize, 0);
        BOOST_CHECK_EQUAL(reinterpret_cast<std::uintptr_t>(decompression) % PageSize, 0);
        const std::string firstData(5000, 'a');
        std::string compressed(ZSTD_compressBound(firstData.size()), '\0');
        auto compressedSize =
            ZSTD_compress(compressed.data(), compressed.size(), firstData.data(), firstData.size(), 1);
        BOOST_REQUIRE(!ZSTD_isError(compressedSize));
        compressed.resize(compressedSize);
        BOOST_CHECK_EQUAL(ZSTD_decompress(decompression, 8192, compressed.data(), compressed.size()), firstData.size());
        BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(decompression), firstData.size()), firstData);
        first->m_pageBuffers[0].SetAvailableSize(123);
        first->m_decompressBuffer.SetAvailableSize(456);
        first->m_diskRequests[0].m_status = 17;
        int callbacks = 0;
        first->m_callback = [&] { ++callbacks; };
        auto capture = std::make_shared<int>(1);
        std::weak_ptr<int> weak = capture;
        first->m_diskRequests[0].m_callback = [capture](bool) {};
        capture.reset();
        first->m_processIocp.push(&first->m_diskRequests[0]);
        factory.ReturnWorkSpace(std::move(first));
        BOOST_CHECK_EQUAL(callbacks, 0);
        BOOST_CHECK(weak.expired());

        ExtraWorkSpace second;
        second.EnableWorkerBuffers();
        second.Initialize(128, 2, 1, 8192, true, true);
        BOOST_CHECK_EQUAL(second.m_pageBuffers[0].GetBuffer(), page);
        BOOST_CHECK_EQUAL(second.m_decompressBuffer.GetBuffer(), decompression);
        BOOST_CHECK_EQUAL(second.m_pageBuffers[0].GetAvailableSize(), 0);
        BOOST_CHECK_EQUAL(second.m_decompressBuffer.GetAvailableSize(), 0);
        const std::string secondData(6000, 'b');
        compressed.resize(ZSTD_compressBound(secondData.size()));
        compressedSize = ZSTD_compress(compressed.data(), compressed.size(), secondData.data(), secondData.size(), 1);
        BOOST_REQUIRE(!ZSTD_isError(compressedSize));
        compressed.resize(compressedSize);
        BOOST_CHECK_EQUAL(ZSTD_decompress(second.m_decompressBuffer.GetBuffer(),
                                          second.m_decompressBuffer.GetPageSize(), compressed.data(),
                                          compressed.size()),
                          secondData.size());
        BOOST_CHECK_EQUAL(
            std::string(reinterpret_cast<char *>(second.m_decompressBuffer.GetBuffer()), secondData.size()),
            secondData);
        BOOST_REQUIRE_EQUAL(second.m_diskRequests.size(), 2);
        for (auto &request : second.m_diskRequests)
            request.m_status = 23;
        second.Clear(2, 8192, true, true);
        BOOST_REQUIRE_EQUAL(second.m_diskRequests.size(), 4);
        for (size_t i = 0; i < second.m_diskRequests.size(); ++i)
        {
            auto &request = second.m_diskRequests[i];
            BOOST_CHECK_EQUAL(request.m_status, 23);
            BOOST_CHECK_EQUAL(request.m_extension, &second.m_processIocp);
            BOOST_CHECK_EQUAL(request.m_buffer,
                              reinterpret_cast<char *>(second.m_pageBuffers[i / 2].GetBuffer()) + (i % 2) * PageSize);
#ifndef _MSC_VER
            BOOST_CHECK_EQUAL(request.myiocb.aio_buf, reinterpret_cast<std::uintptr_t>(request.m_buffer));
            BOOST_CHECK_EQUAL(request.myiocb.aio_data, reinterpret_cast<std::uintptr_t>(&request));
#endif
        }
        second.ReleaseScratch();
        BOOST_CHECK(second.m_pageBuffers.empty());
        BOOST_CHECK_EQUAL(second.m_decompressBuffer.GetPageSize(), 0);
        for (auto &request : second.m_diskRequests)
            BOOST_CHECK(request.m_buffer == nullptr);

        first = factory.GetWorkSpace();
        BOOST_REQUIRE(first);
        Helper::AsyncReadRequest *completed = nullptr;
        BOOST_CHECK(!first->m_processIocp.try_pop(completed));
        first->Clear(2, 8192, false, false);
        BOOST_CHECK(!first->m_enableDataCompression);
        BOOST_CHECK_EQUAL(first->m_pageBuffers[0].GetBuffer(), page);
        BOOST_REQUIRE_EQUAL(first->m_diskRequests.size(), 2);
        for (size_t i = 0; i < first->m_diskRequests.size(); ++i)
        {
            BOOST_CHECK_EQUAL(first->m_diskRequests[i].m_status, 17);
            BOOST_CHECK_EQUAL(first->m_diskRequests[i].m_extension, &first->m_processIocp);
            BOOST_CHECK_EQUAL(first->m_diskRequests[i].m_buffer,
                              reinterpret_cast<char *>(first->m_pageBuffers[i].GetBuffer()));
        }
        first.reset();
        BOOST_CHECK_EQUAL(callbacks, 1);
    });
    worker.join();
}

BOOST_AUTO_TEST_CASE(WorkerBufferLeasesProtectNestedQueriesAndOwningAliases)
{
    std::thread worker([] {
        ExtraWorkSpace held, nested, reused;
        for (auto *workspace : {&held, &nested, &reused})
            workspace->EnableWorkerBuffers();
        held.Initialize(128, 2, 1, 4096, false, true);
        auto *heldPage = held.m_pageBuffers[0].GetBuffer();
        heldPage[0] = 17;
        held.m_loadedPostingNum = 1;
        held.m_offset = 7;
        nested.Initialize(128, 2, 1, 4096, false, true);
        auto *nestedPage = nested.m_pageBuffers[0].GetBuffer();
        BOOST_CHECK_NE(nestedPage, heldPage);
        BOOST_CHECK_NE(nested.m_decompressBuffer.GetBuffer(), held.m_decompressBuffer.GetBuffer());
        nested.ReleaseScratch();
        reused.Initialize(128, 2, 1, 4096, false, true);
        BOOST_CHECK_EQUAL(reused.m_pageBuffers[0].GetBuffer(), nestedPage);
        BOOST_CHECK_EQUAL(heldPage[0], 17);
        BOOST_CHECK_EQUAL(held.m_loadedPostingNum, 1);
        BOOST_CHECK_EQUAL(held.m_offset, 7);
        reused.ReleaseScratch();

        // A shared owning view must not become another tenant's mutable scratch.
        auto pageAlias = held.m_pageBuffers[0];
        auto decompressionAlias = held.m_decompressBuffer;
        auto *heldDecompression = decompressionAlias.GetBuffer();
        heldDecompression[0] = 29;
        held.ReleaseScratch();
        reused.Clear(1, 4096, false, true);
        BOOST_CHECK_NE(reused.m_pageBuffers[0].GetBuffer(), heldPage);
        BOOST_CHECK_NE(reused.m_decompressBuffer.GetBuffer(), heldDecompression);
        reused.m_pageBuffers[0].GetBuffer()[0] = 43;
        reused.m_decompressBuffer.GetBuffer()[0] = 47;
        BOOST_CHECK_EQUAL(pageAlias.GetBuffer()[0], 17);
        BOOST_CHECK_EQUAL(decompressionAlias.GetBuffer()[0], 29);
        reused.ReleaseScratch();

        // RocksDB can grow a posting beyond the configured page budget.
        ExtraWorkSpace grown, differentlySized;
        grown.EnableWorkerBuffers();
        differentlySized.EnableWorkerBuffers();
        grown.Initialize(128, 2, 1, 4096, false, false);
        grown.m_pageBuffers[0].ReservePageBuffer(12288);
        auto *grownPage = grown.m_pageBuffers[0].GetBuffer();
        grown.ReleaseScratch();
        differentlySized.Initialize(128, 2, 1, 12288, true, false);
        BOOST_CHECK_EQUAL(differentlySized.m_pageBuffers[0].GetBuffer(), grownPage);
        BOOST_REQUIRE_EQUAL(differentlySized.m_diskRequests.size(), 3);
        BOOST_CHECK_EQUAL(differentlySized.m_diskRequests[2].m_buffer,
                          reinterpret_cast<char *>(grownPage) + 2 * PageSize);
        differentlySized.ReleaseScratch();
    });
    worker.join();
}

BOOST_AUTO_TEST_CASE(WorkerBuffersRequireBackendCompletionGuarantee)
{
    std::thread worker([] {
        TemporaryNode legacyTemporary;
        SPANN::Index<float> legacy;
        RequireSuccess(legacy.SetPostingStore(std::make_shared<RocksDBIO>(legacyTemporary.root.c_str(), false)));
        ExtraWorkSpace pending, other;
        legacy.InitWorkSpace(&pending, true);
        auto *page = pending.m_pageBuffers[0].GetBuffer();
        pending.m_diskRequests[0].m_callback = [page](bool) { page[0] = 31; };
        pending.ReleaseScratch();
        BOOST_REQUIRE(!pending.m_pageBuffers.empty());
        BOOST_CHECK_EQUAL(pending.m_pageBuffers[0].GetBuffer(), page);
        other.EnableWorkerBuffers();
        other.Initialize(128, 2, 1, 4096, false, false);
        BOOST_CHECK_NE(other.m_pageBuffers[0].GetBuffer(), page);
        pending.m_diskRequests[0].m_callback(true);
        BOOST_CHECK_EQUAL(page[0], 31);

        TemporaryNode temporary;
        std::shared_ptr<SharedRocksDB> engine;
        RequireSuccess(SharedRocksDB::Open(temporary.root, {"postings"}, engine, true, nullptr));
        auto store = engine->PostingView("tenant", 0);
        SPANN::Index<float> managed;
        RequireSuccess(managed.SetPostingStore(store));
        ExtraWorkSpace synchronous;
        managed.InitWorkSpace(&synchronous, true);
        RequireSuccess(store->Put(1, std::string(50000, 'x'), std::chrono::microseconds(0), nullptr));
        RequireSuccess(
            store->MultiGet({1}, synchronous.m_pageBuffers, std::chrono::microseconds(0), &synchronous.m_diskRequests));
        BOOST_CHECK_EQUAL(synchronous.m_pageBuffers[0].GetAvailableSize(), 50000);
        synchronous.ReleaseScratch();
        BOOST_CHECK(synchronous.m_pageBuffers.empty());
        managed.InitWorkSpace(&synchronous, true);
        BOOST_CHECK_EQUAL(synchronous.m_pageBuffers[0].GetAvailableSize(), 0);
        BOOST_CHECK(store->MultiGet({-1}, synchronous.m_pageBuffers, std::chrono::microseconds(0),
                                    &synchronous.m_diskRequests) != ErrorCode::Success);
        synchronous.ReleaseScratch();
        BOOST_CHECK(synchronous.m_pageBuffers.empty());
    });
    worker.join();
}

BOOST_AUTO_TEST_CASE(EpochWaitsForOldReadersButNotNewOnes)
{
    ReadEpoch epoch;
    int first, second;
    std::promise<void> entered, leave;
    auto gate = leave.get_future();
    auto reader = std::async(std::launch::async, [&] {
        ReadEpoch::Guard outer(epoch);
        outer.SetOwner(&first);
        {
            ReadEpoch::Guard inner(epoch);
            inner.SetOwner(&second);
            BOOST_CHECK_EQUAL(epoch.Active(&second), 1);
        }
        BOOST_CHECK_EQUAL(epoch.Active(&first), 1);
        entered.set_value();
        gate.wait();
    });
    entered.get_future().wait();
    auto retired = epoch.Advance();
    std::promise<void> waiting;
    auto reclaim = std::async(std::launch::async, [&] {
        waiting.set_value();
        epoch.Wait(retired);
    });
    waiting.get_future().wait();
    BOOST_CHECK(reclaim.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    {
        ReadEpoch::Guard later(epoch);
        later.SetOwner(&second);
        leave.set_value();
        reader.get();
        BOOST_CHECK(reclaim.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    }
    reclaim.get();
    BOOST_CHECK_EQUAL(epoch.Active(&first) + epoch.Active(&second), 0);
}

BOOST_AUTO_TEST_CASE(TenantWireBoundsAndRoundTrip)
{
    Socket::RemoteTenantRequest request;
    request.operation = Socket::TenantOperation::Create;
    request.tenant = "tenant / with delimiters";
    request.valueType = VectorValueType::Float;
    request.metric = DistCalcMethod::L2;
    request.dimension = 4;
    request.count = 64;
    request.vectors = Vectors(64, 4, 0);
    ByteArray body;
    BOOST_REQUIRE(request.Write(body));
    Socket::RemoteTenantRequest decoded;
    BOOST_REQUIRE(decoded.Read(body.Data(), body.Length()));
    BOOST_CHECK_EQUAL(decoded.tenant, request.tenant);
    BOOST_CHECK_EQUAL(decoded.count, 64);
    BOOST_CHECK(decoded.headIndexAlgo == IndexAlgoType::Undefined);
    BOOST_CHECK_EQUAL(memcmp(decoded.vectors.Data(), request.vectors.Data(), request.vectors.Length()), 0);
    for (size_t size = 0; size < body.Length(); ++size)
        BOOST_CHECK(!decoded.Read(body.Data(), size));
    BOOST_CHECK(!decoded.Read(nullptr, body.Length()));
    auto invalid = ByteArray::Alloc(body.Length() + 1);
    memcpy(invalid.Data(), body.Data(), body.Length());
    BOOST_CHECK(!decoded.Read(invalid.Data(), invalid.Length()));
    invalid[0] = 3;
    BOOST_CHECK(!decoded.Read(invalid.Data(), body.Length()));
    memcpy(invalid.Data(), body.Data(), body.Length());
    invalid[2] = 255;
    BOOST_CHECK(!decoded.Read(invalid.Data(), body.Length()));
    memcpy(invalid.Data(), body.Data(), body.Length());
    std::uint32_t oversized = UINT32_MAX;
    memcpy(invalid.Data() + 25, &oversized, sizeof(oversized));
    BOOST_CHECK(!decoded.Read(invalid.Data(), body.Length()));

    auto legacyBody = body;
    for (auto head : {IndexAlgoType::BKT, IndexAlgoType::KDT})
    {
        request.headIndexAlgo = head;
        BOOST_REQUIRE(request.Write(body));
        BOOST_REQUIRE(decoded.Read(body.Data(), body.Length()));
        BOOST_CHECK(decoded.headIndexAlgo == head);
        BOOST_CHECK_EQUAL(decoded.tenant, request.tenant);
        BOOST_CHECK_EQUAL(memcmp(decoded.vectors.Data(), request.vectors.Data(), request.vectors.Length()), 0);
        BOOST_CHECK(!decoded.Read(body.Data(), 29));
        BOOST_CHECK(!decoded.Read(body.Data(), body.Length() - 1));
        body[29] = static_cast<std::uint8_t>(IndexAlgoType::SPANN);
        BOOST_CHECK(!decoded.Read(body.Data(), body.Length()));
        body[29] = static_cast<std::uint8_t>(IndexAlgoType::Undefined);
        BOOST_CHECK(!decoded.Read(body.Data(), body.Length()));
        body[29] = static_cast<std::uint8_t>(head);
        body[2] = static_cast<std::uint8_t>(Socket::TenantOperation::Insert);
        BOOST_CHECK(!decoded.Read(body.Data(), body.Length()));
    }
    BOOST_REQUIRE(decoded.Read(legacyBody.Data(), legacyBody.Length()));
    BOOST_CHECK(decoded.headIndexAlgo == IndexAlgoType::Undefined);
    request.headIndexAlgo = IndexAlgoType::SPANN;
    BOOST_CHECK(!request.Write(body));
    request.headIndexAlgo = IndexAlgoType::BKT;
    request.operation = Socket::TenantOperation::Insert;
    request.metric = DistCalcMethod::Undefined;
    BOOST_CHECK(!request.Write(body));

    Socket::RemoteTenantResult result;
    result.status = ErrorCode::Success;
    result.count = 1;
    result.results.resize(1);
    result.results[0].VID = 42;
    result.results[0].Dist = 0.25f;
    BOOST_REQUIRE(result.Write(body));
    Socket::RemoteTenantResult reply;
    BOOST_REQUIRE(reply.Read(body.Data(), body.Length()));
    RequireSuccess(reply.status);
    BOOST_CHECK_EQUAL(reply.results[0].VID, 42);
    BOOST_CHECK_EQUAL(reply.results[0].Dist, 0.25f);
    for (size_t size = 0; size < body.Length(); ++size)
        BOOST_CHECK(!reply.Read(body.Data(), size));
    memcpy(body.Data() + 12, &oversized, sizeof(oversized));
    BOOST_CHECK(!reply.Read(body.Data(), body.Length()));
    result.status = ErrorCode::Key_NotFound;
    result.results.clear();
    result.count = 0;
    BOOST_REQUIRE(result.Write(body));
    BOOST_REQUIRE(reply.Read(body.Data(), body.Length()));
    BOOST_CHECK_EQUAL(std::string(reply.StatusName()), "Key_NotFound");
    BOOST_CHECK(reply.results.empty());
}

BOOST_AUTO_TEST_CASE(DedicatedRootAndExclusiveOwner)
{
    TemporaryNode nonempty;
    std::filesystem::create_directories(nonempty.root);
    {
        std::ofstream sentinel(nonempty.root + "/keep.txt");
        sentinel << "existing data";
    }
    std::unique_ptr<TenantIndexManager> node;
    BOOST_CHECK(TenantIndexManager::Open(nonempty.root, {}, node) != ErrorCode::Success);
    BOOST_CHECK(std::filesystem::exists(nonempty.root + "/keep.txt"));
    BOOST_CHECK(!std::filesystem::exists(nonempty.root + "/recovery"));
    TemporaryNode directory;
    node = OpenNode(directory.root);
    std::unique_ptr<TenantIndexManager> duplicate;
    BOOST_CHECK(TenantIndexManager::Open(directory.root, {}, duplicate) != ErrorCode::Success);
    BOOST_CHECK(!duplicate);
    RequireSuccess(node->Checkpoint());
    RequireSuccess(node->Close());
    node.reset();
    {
        std::shared_ptr<SharedRocksDB> recovery;
        RequireSuccess(SharedRocksDB::Open(directory.root + "/recovery", {"recovery_log", "catalog", "recovery_meta"},
                                           recovery, false, rocksdb::NewLRUCache(1 << 20)));
        RequireSuccess(recovery->Put("recovery_meta", "format", "SPTAG-Tenant-1"));
    }
    BOOST_CHECK(TenantIndexManager::Open(directory.root, {}, node) == ErrorCode::FailedParseValue);
    BOOST_CHECK(!node);
}

BOOST_AUTO_TEST_CASE(NamespacesAndViewOwnership)
{
    TemporaryNode directory;
    std::filesystem::create_directories(directory.root);
    std::shared_ptr<SharedRocksDB> engine;
    RequireSuccess(
        SharedRocksDB::Open(directory.root + "/state", {"postings"}, engine, true, rocksdb::NewLRUCache(1 << 20)));
    auto a = engine->PostingView("a/b", 1);
    auto b = engine->PostingView("a", 1);
    RequireSuccess(a->Put(1, "alpha", MaxTimeout, nullptr));
    RequireSuccess(b->Put(1, "beta", MaxTimeout, nullptr));
    int mergedSize = 0;
    RequireSuccess(a->Merge(1, "-tail", MaxTimeout, nullptr, mergedSize));
    BOOST_CHECK_EQUAL(mergedSize, 10);
    std::string value;
    RequireSuccess(a->Get(1, &value, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(value, "alpha-tail");
    RequireSuccess(a->DeleteRange(0, 2));
    BOOST_CHECK(a->Get(1, &value, MaxTimeout, nullptr) == ErrorCode::Key_NotFound);
    a->ShutDown();
    RequireSuccess(b->Get(1, &value, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(value, "beta");
    RequireSuccess(b->MergeWithCAS(2, "first", MaxTimeout, nullptr, mergedSize));
    BOOST_CHECK_EQUAL(mergedSize, 5);
    std::vector<Helper::PageBuffer<uint8_t>> buffers(2);
    RequireSuccess(b->MultiGet({1, 2}, buffers, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(buffers[1].GetAvailableSize(), 5);
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(buffers[1].GetBuffer()), 5), "first");
    RequireSuccess(b->Put(3, std::string(20000, 'x'), MaxTimeout, nullptr));
    RequireSuccess(b->Put(4, "", MaxTimeout, nullptr));
    RequireSuccess(b->MultiGet({3, 4}, buffers, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(buffers[0].GetAvailableSize(), 20000);
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(buffers[0].GetBuffer()), 20000), std::string(20000, 'x'));
    BOOST_CHECK_EQUAL(buffers[1].GetAvailableSize(), 0);
    RequireSuccess(b->MultiGet({3, 99}, buffers, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(buffers[0].GetAvailableSize(), 20000);
    BOOST_CHECK_EQUAL(buffers[1].GetAvailableSize(), 0);
    BOOST_CHECK(b->MultiGet({-1}, buffers, MaxTimeout, nullptr) != ErrorCode::Success);
    std::vector<std::string> missing;
    BOOST_CHECK(b->MultiGet({1, 999}, &missing, MaxTimeout, nullptr) == ErrorCode::Key_NotFound);
    RequireSuccess(engine->Checkpoint(directory.root + "/snapshot"));
    BOOST_CHECK(engine->Checkpoint(directory.root + "/snapshot") != ErrorCode::Success);
    bool hasBlob = false;
    for (const auto &file : std::filesystem::directory_iterator(directory.root + "/snapshot"))
        hasBlob |= file.path().extension() == ".blob";
    BOOST_REQUIRE(hasBlob);
    std::shared_ptr<SharedRocksDB> restored;
    RequireSuccess(SharedRocksDB::Open(directory.root + "/snapshot", {"postings"}, restored, false,
                                       rocksdb::NewLRUCache(1 << 20)));
    auto restoredView = restored->PostingView("a", 1);
    RequireSuccess(restoredView->MultiGet({1, 3}, buffers, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(buffers[0].GetBuffer()), buffers[0].GetAvailableSize()),
                      "beta");
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(buffers[1].GetBuffer()), buffers[1].GetAvailableSize()),
                      std::string(20000, 'x'));
    RocksDBIO privateStore((directory.root + "/private").c_str(), false, false, false);
    RequireSuccess(privateStore.MergeWithCAS(7, "alpha", MaxTimeout, nullptr, mergedSize));
    RequireSuccess(privateStore.Merge(7, "-tail", MaxTimeout, nullptr, mergedSize));
    BOOST_CHECK_EQUAL(mergedSize, 10);
    std::vector<Helper::PageBuffer<uint8_t>> privateBuffers(2);
    RequireSuccess(privateStore.MultiGet({7, 8}, privateBuffers, MaxTimeout, nullptr));
    BOOST_CHECK_EQUAL(privateBuffers[0].GetAvailableSize(), 10);
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(privateBuffers[0].GetBuffer()), 10), "alpha-tail");
    BOOST_CHECK_EQUAL(privateBuffers[1].GetAvailableSize(), 0);
}

BOOST_AUTO_TEST_CASE(OrderedLogKeysAndBoundedScan)
{
    TemporaryNode directory;
    std::filesystem::create_directories(directory.root);
    std::shared_ptr<SharedRocksDB> engine;
    RequireSuccess(SharedRocksDB::Open(directory.root + "/recovery", {"recovery_log"}, engine, true,
                                       rocksdb::NewLRUCache(1 << 20)));
    const auto prefix = SharedRocksDB::Namespace("tenant", 1);
    for (auto number : {65536ULL, 256ULL, 1ULL, 255ULL})
        RequireSuccess(engine->Put("recovery_log", prefix + SharedRocksDB::EncodeNumber(number), "record"));
    RequireSuccess(
        engine->Put("recovery_log", SharedRocksDB::Namespace("tenant", 2) + SharedRocksDB::EncodeNumber(1), "other"));
    std::vector<uint64_t> seen;
    RequireSuccess(engine->Scan("recovery_log", prefix, [&](const std::string &key, const std::string &) {
        uint64_t number;
        BOOST_REQUIRE(SharedRocksDB::DecodeNumber(key.substr(prefix.size()), number));
        seen.push_back(number);
        return ErrorCode::Success;
    }));
    const std::vector<uint64_t> expected{1, 255, 256, 65536};
    BOOST_CHECK_EQUAL_COLLECTIONS(seen.begin(), seen.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(IndependentIndexesCheckpointAndReload)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    auto a = Vectors(128, 4, 0);
    auto b = Vectors(128, 7, 1000);
    Helper::IniReader profile;
    profile.SetParameter("Base", "IndexAlgoType", "BKT");
    profile.SetParameter("Base", "ValueType", "UInt8");
    profile.SetParameter("Base", "IndexDirectory", directory.root + "/unused");
    profile.SetParameter("BuildSSDIndex", "TmpDir", directory.root + "/unused");
    profile.SetParameter("BuildSSDIndex", "Layers", "2");
    profile.SetParameter("BuildSSDIndex", "MaxCheck", "512");
    profile.SetParameter("BuildSSDIndex", "Storage", "FILEIO");
    profile.SetParameter("BuildSSDIndex", "ShareDB", "false");
    profile.SetParameter("BuildSSDIndex", "EnableWAL", "true");
    profile.SetParameter("BuildSSDIndex", "Recovery", "true");
    profile.SetParameter("BuildSSDIndex", "AsyncMergeInSearch", "true");
    profile.SetParameter("BuildSSDIndex", "AsyncAppendQueueSize", "32");
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128));
    RequireSuccess(node->Create("B", VectorValueType::Float, 7, DistCalcMethod::L2, b, 128, IndexAlgoType::Undefined, 0,
                                &profile));
    RequireSuccess(
        node->Create("C", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128, IndexAlgoType::KDT, 1, &profile));
    profile.SetParameter("BuildSSDIndex", "Layers", "0");
    BOOST_CHECK(node->Create("invalid-profile", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128,
                             IndexAlgoType::Undefined, 0, &profile) == ErrorCode::FailedParseValue);
    profile.SetParameter("BuildSSDIndex", "Layers", "2");
    profile.SetParameter("Base", "HeadVectors", "../outside.bin");
    BOOST_CHECK(node->Create("invalid-path", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128,
                             IndexAlgoType::Undefined, 0, &profile) == ErrorCode::FailedParseValue);
    profile.SetParameter("Base", "HeadVectors", "SPTAGHeadVectors.bin");
    profile.SetParameter("BuildHead", "VectorFilePath", "../outside.bin");
    BOOST_CHECK(node->Create("invalid-head-path", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128,
                             IndexAlgoType::Undefined, 0, &profile) == ErrorCode::FailedParseValue);
    BOOST_CHECK(!std::filesystem::exists(directory.root + "/unused"));
    BOOST_CHECK(node->Create("invalid", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128, IndexAlgoType::SPANN) ==
                ErrorCode::FailedParseValue);
    BOOST_CHECK(node->Create("invalid-layers", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128,
                             IndexAlgoType::BKT, -1) == ErrorCode::FailedParseValue);
    BOOST_CHECK_EQUAL(node->List().size(), 3);
    const auto defaultHead = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::Float)
                                 ->GetParameter("IndexAlgoType", "Base");
    const auto checkHeads = [&] {
        const std::map<std::string, std::string> expected{{"A", defaultHead}, {"B", "BKT"}, {"C", "KDT"}};
        TenantIndexManager::NodeStats state;
        std::vector<TenantIndexManager::TenantStats> tenants;
        node->GetStats(state, tenants);
        size_t checked = 0;
        for (const auto &entry : std::filesystem::directory_iterator(directory.root))
        {
            if (entry.path().filename().string().find("work-") != 0)
                continue;
            for (const auto &tenant : tenants)
            {
                Helper::IniReader config;
                RequireSuccess(config.LoadIniFile(
                    (entry.path() / ("index-" + std::to_string(tenant.indexID)) / "indexloader.ini").string()));
                BOOST_CHECK_EQUAL(config.GetParameter("Base", "IndexAlgoType", std::string()),
                                  expected.at(tenant.name));
                BOOST_CHECK_EQUAL(config.GetParameter("BuildSSDIndex", "Layers", 0), tenant.name == "B" ? 2 : 1);
                if (tenant.name != "A")
                {
                    BOOST_CHECK_EQUAL(config.GetParameter("BuildSSDIndex", "MaxCheck", 0), 512);
                    BOOST_CHECK_EQUAL(config.GetParameter("Base", "ValueType", std::string()), "Float");
                    BOOST_CHECK_EQUAL(config.GetParameter("Base", "Dim", 0), tenant.name == "B" ? 7 : 4);
                    BOOST_CHECK_EQUAL(config.GetParameter("BuildSSDIndex", "Storage", std::string()), "ROCKSDBIO");
                    BOOST_CHECK(config.GetParameter("BuildSSDIndex", "ShareDB", false));
                    BOOST_CHECK(!config.GetParameter("BuildSSDIndex", "EnableWAL", true));
                    BOOST_CHECK(!config.GetParameter("BuildSSDIndex", "Recovery", true));
                    BOOST_CHECK(!config.GetParameter("BuildSSDIndex", "AsyncMergeInSearch", true));
                    BOOST_CHECK_EQUAL(config.GetParameter("BuildSSDIndex", "AsyncAppendQueueSize", -1), 0);
                }
                ++checked;
            }
        }
        BOOST_CHECK_EQUAL(checked, 3);
    };
    checkHeads();
    BOOST_CHECK(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, a, 128) != ErrorCode::Success);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    RequireSuccess(node->Search("B", Vectors(1, 7, 1000), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    RequireSuccess(node->Search("C", Vectors(1, 4, 0), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    BOOST_CHECK(node->Search("A", Vectors(1, 7, 0), 3, results) != ErrorCode::Success);
    BOOST_CHECK(node->Search("missing", Vectors(1, 4, 0), 3, results) == ErrorCode::Key_NotFound);
    SizeType id = -1;
    BOOST_CHECK(node->Insert("A", Vectors(1, 4, 500), 1, id, VectorValueType::UInt8, 16) != ErrorCode::Success);
    BOOST_CHECK(node->Insert("A", Vectors(1, 4, 500), 1, id, VectorValueType::Float, 5) != ErrorCode::Success);
    BOOST_CHECK(node->Search("A", Vectors(1, 4, 0), 3, results, VectorValueType::UInt8, 16) != ErrorCode::Success);
    BOOST_CHECK(node->Search("A", Vectors(1, 4, 0), 3, results, VectorValueType::Float, 5) != ErrorCode::Success);
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    BOOST_CHECK_EQUAL(id, 128);
    RequireSuccess(node->DeleteVector("A", 0));
    RequireSuccess(node->Insert("B", Vectors(1, 7, 1500), 1, id));
    BOOST_CHECK_EQUAL(id, 128);
    RequireSuccess(node->Unload("A"));
    RequireSuccess(node->Search("A", Vectors(1, 4, 500), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 128);
    RequireSuccess(node->Search("B", Vectors(1, 7, 1000), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    RequireSuccess(node->Search("B", Vectors(1, 7, 1500), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 128);
    RequireSuccess(node->Insert("A", Vectors(1, 4, 600), 1, id));
    BOOST_CHECK_EQUAL(id, 129);
    RequireSuccess(node->Checkpoint());
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root);
    RequireSuccess(node->Search("A", Vectors(1, 4, 500), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 128);
    RequireSuccess(node->Search("B", Vectors(1, 7, 1000), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    RequireSuccess(node->Search("C", Vectors(1, 4, 0), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    checkHeads();
    RequireSuccess(node->Insert("A", Vectors(1, 4, 700), 1, id));
    BOOST_CHECK_EQUAL(id, 130);
    RequireSuccess(node->DeleteVector("B", 0));
    RequireSuccess(node->Insert("B", Vectors(1, 7, 1600), 1, id));
    BOOST_CHECK_EQUAL(id, 129);
    RequireSuccess(node->Unload("B"));
    RequireSuccess(node->Search("B", Vectors(1, 7, 1600), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 129);
    RequireSuccess(node->DeleteTenant("A"));
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, -500), 128));
    RequireSuccess(node->Search("A", Vectors(1, 4, -500), 3, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    RequireSuccess(node->Close());
}

BOOST_AUTO_TEST_CASE(ConcurrentTenantsAndBoundedArtifacts)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    for (const auto *name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128,
                                    name[0] == 'A' ? IndexAlgoType::KDT : IndexAlgoType::BKT, name[0] == 'A' ? 1 : 2));
    auto update = [&](const char *name, float base) {
        SizeType first;
        auto status = node->Insert(name, Vectors(1024, 4, base), 1024, first);
        if (status != ErrorCode::Success)
            return status;
        std::vector<BasicResult> results;
        status = node->Search(name, Vectors(1, 4, base), 1, results);
        if (status != ErrorCode::Success)
            return status;
        return results.front().VID == 128 ? ErrorCode::Success : ErrorCode::Fail;
    };
    auto a = std::async(std::launch::async, update, "A", 500);
    auto b = std::async(std::launch::async, update, "B", -500);
    RequireSuccess(a.get());
    RequireSuccess(b.get());
    RequireSuccess(node->Checkpoint());
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 500), 1, results));
    BOOST_CHECK_EQUAL(results[0].VID, 128);
    RequireSuccess(node->Search("B", Vectors(1, 4, -500), 1, results));
    BOOST_CHECK_EQUAL(results[0].VID, 128);
    size_t work = 0, checkpoint = 0;
    for (const auto &entry : std::filesystem::directory_iterator(directory.root))
    {
        const auto name = entry.path().filename().string();
        if (name.find("work-") == 0)
            ++work;
        if (name.find("checkpoint-") == 0)
            ++checkpoint;
    }
    BOOST_CHECK_EQUAL(work, 1);
    BOOST_CHECK_EQUAL(checkpoint, 1);
}

BOOST_AUTO_TEST_CASE(SameTenantConcurrentReadersWritersAndReload)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    RequireSuccess(node->Unload("A"));
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<std::pair<ErrorCode, SizeType>>> writers;
    for (int worker = 0; worker < 4; ++worker)
        writers.emplace_back(std::async(std::launch::async, [&, worker] {
            gate.wait();
            SizeType first = -1;
            auto status = node->Insert("A", Vectors(256, 4, 1000 + worker * 100), 256, first);
            return std::make_pair(status, first);
        }));
    std::vector<std::future<ErrorCode>> readers;
    for (int worker = 0; worker < 3; ++worker)
        readers.emplace_back(std::async(std::launch::async, [&] {
            gate.wait();
            for (int i = 0; i < 200; ++i)
            {
                std::vector<BasicResult> results;
                auto status = node->Search("A", Vectors(1, 4, 0), 3, results);
                if (status != ErrorCode::Success)
                    return status;
                if (results[0].VID != 0)
                    return ErrorCode::Fail;
            }
            return ErrorCode::Success;
        }));
    start.set_value();
    std::vector<SizeType> firstIDs;
    for (auto &writer : writers)
    {
        auto result = writer.get();
        RequireSuccess(result.first);
        firstIDs.push_back(result.second);
    }
    for (auto &reader : readers)
        RequireSuccess(reader.get());
    auto sorted = firstIDs;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 4; ++i)
        BOOST_CHECK_EQUAL(sorted[i], 128 + i * 256);
    std::promise<void> deleteStart;
    auto deleteGate = deleteStart.get_future().share();
    auto remove = [&] {
        deleteGate.wait();
        return node->DeleteVector("A", 0);
    };
    auto a = std::async(std::launch::async, remove);
    auto b = std::async(std::launch::async, remove);
    deleteStart.set_value();
    auto sa = a.get(), sb = b.get();
    BOOST_CHECK((sa == ErrorCode::Success && sb == ErrorCode::VectorNotFound) ||
                (sb == ErrorCode::Success && sa == ErrorCode::VectorNotFound));
    // Reopen without publishing: replay must retain all independently allocated IDs.
    node.reset();
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        node = OpenNode(directory.root);
        for (int worker = 0; worker < 4; ++worker)
        {
            std::vector<BasicResult> results;
            RequireSuccess(node->Search("A", Vectors(1, 4, 1000 + worker * 100), 1, results));
            BOOST_CHECK_EQUAL(results[0].VID, firstIDs[worker]);
        }
        BOOST_CHECK(node->DeleteVector("A", 0) == ErrorCode::VectorNotFound);
        node.reset();
    }
}

BOOST_AUTO_TEST_CASE(CheckpointReusesCleanSnapshotsAndAllowsQueries)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    auto probe = std::make_shared<SnapshotProbe>();
    SnapshotFactory factory(probe);
    RequireSuccess(node->Checkpoint());
    BOOST_CHECK_EQUAL(probe->saves.load(), 0);
    SizeType first;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, first));
    probe->block = true;
    auto entered = probe->entered.get_future();
    auto checkpoint = std::async(std::launch::async, [&] { return node->Checkpoint(); });
    bool saving = entered.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    auto query = std::async(std::launch::async, [&] {
        std::vector<BasicResult> results;
        return node->Search("A", Vectors(1, 4, 500), 1, results);
    });
    bool queryFinished = query.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    auto writer = std::async(std::launch::async, [&] {
        SizeType id;
        return node->Insert("A", Vectors(1, 4, 600), 1, id);
    });
    bool writerBlocked = writer.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    probe->release.set_value();
    RequireSuccess(checkpoint.get());
    RequireSuccess(query.get());
    RequireSuccess(writer.get());
    BOOST_CHECK(saving);
    BOOST_CHECK(queryFinished);
    BOOST_CHECK(writerBlocked);
    BOOST_CHECK_EQUAL(probe->saves.load(), 1);
    RequireSuccess(node->Checkpoint());
    BOOST_CHECK_EQUAL(probe->saves.load(), 2);
    RequireSuccess(node->Unload("A"));
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 600), 1, results));
    RequireSuccess(node->Checkpoint());
    BOOST_CHECK_EQUAL(probe->saves.load(), 2);
    node.reset();
    node = OpenNode(directory.root);
    probe->saves = 0;
    RequireSuccess(node->Checkpoint());
    BOOST_CHECK_EQUAL(probe->saves.load(), 0);
}

BOOST_AUTO_TEST_CASE(AutoCheckpointNodeWideBytesCooldownAndIdle, *boost::unit_test::timeout(60))
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.checkpointLogBytes = 160;
    limits.checkpointPressureBytes = 1 << 20;
    limits.checkpointMinIntervalSeconds = 1;
    std::atomic<bool> armed{false};
    std::mutex timingMutex;
    std::vector<std::chrono::steady_clock::time_point> publications;
    limits.checkpointHook = [&](const char *stage) {
        if (armed && std::string(stage) == "after_publish")
        {
            std::lock_guard<std::mutex> lock(timingMutex);
            publications.push_back(std::chrono::steady_clock::now());
        }
    };
    auto node = OpenNode(directory.root, limits);
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    RequireSuccess(node->Checkpoint());
    const auto started = std::chrono::steady_clock::now();
    armed = true;
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    const auto firstBytes = NodeState(*node).recoveryLogBytes;
    BOOST_REQUIRE(firstBytes > 0 && firstBytes < limits.checkpointLogBytes);
    RequireSuccess(node->Insert("B", Vectors(1, 4, 600), 1, id));
    BOOST_REQUIRE(WaitUntil([&] { return NodeState(*node).autoCheckpoints == 1; }));
    BOOST_CHECK_EQUAL(NodeState(*node).recoveryLogBytes, 0);
    RequireSuccess(node->Insert("A", Vectors(2, 4, 700), 2, id));
    BOOST_CHECK(NodeState(*node).recoveryLogBytes < limits.checkpointLogBytes);
    RequireSuccess(node->Insert("B", Vectors(1, 4, 800), 1, id));
    BOOST_REQUIRE(WaitUntil([&] { return NodeState(*node).autoCheckpoints == 2; }));
    {
        std::lock_guard<std::mutex> lock(timingMutex);
        BOOST_REQUIRE_EQUAL(publications.size(), 2);
        BOOST_CHECK(publications[0] - started >= std::chrono::milliseconds(950));
        BOOST_CHECK(publications[1] - publications[0] >= std::chrono::seconds(1));
    }
    RequireSuccess(node->Insert("A", Vectors(1, 4, 900), 1, id));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    BOOST_CHECK_EQUAL(NodeState(*node).autoCheckpoints, 2);
    BOOST_CHECK_EQUAL(NodeState(*node).recoveryLogBytes, firstBytes);
}

BOOST_AUTO_TEST_CASE(AutoCheckpointPressureBypassesCooldownAndBlocksNewWrites, *boost::unit_test::timeout(60))
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.checkpointLogBytes = 1;
    limits.checkpointPressureBytes = 1;
    limits.checkpointMinIntervalSeconds = 3600;
    std::atomic<bool> armed{false};
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    limits.checkpointHook = [&](const char *stage) {
        if (std::string(stage) == "before_snapshots" && armed.exchange(false))
        {
            entered.set_value();
            released.wait();
        }
    };
    auto node = OpenNode(directory.root, limits);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    armed = true;
    auto entering = entered.get_future();
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    const bool checkpointEntered = entering.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    auto writer = std::async(std::launch::async, [&] {
        SizeType next;
        return node->Insert("A", Vectors(1, 4, 600), 1, next);
    });
    const bool writerBlocked = writer.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    std::vector<BasicResult> results;
    const auto searched = node->Search("A", Vectors(1, 4, 500), 1, results);
    release.set_value();
    RequireSuccess(writer.get());
    BOOST_CHECK(checkpointEntered);
    BOOST_CHECK(writerBlocked);
    RequireSuccess(searched);
    BOOST_REQUIRE_EQUAL(results.size(), 1);
    BOOST_CHECK_EQUAL(results[0].VID, id);
    BOOST_CHECK(WaitUntil([&] { return NodeState(*node).autoCheckpoints >= 1; }));
}

BOOST_AUTO_TEST_CASE(AutoCheckpointFailureWakesWritersAndKeepsReplayLog, *boost::unit_test::timeout(60))
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.checkpointLogBytes = 1;
    limits.checkpointPressureBytes = 1;
    std::atomic<bool> armed{false};
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    limits.checkpointHook = [&](const char *stage) {
        if (std::string(stage) == "before_snapshots" && armed.exchange(false))
        {
            entered.set_value();
            released.wait();
            throw std::runtime_error("automatic checkpoint failure");
        }
    };
    auto node = OpenNode(directory.root, limits);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    armed = true;
    auto entering = entered.get_future();
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    const bool checkpointEntered = entering.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    auto writer = std::async(std::launch::async, [&] {
        SizeType next;
        return node->Insert("A", Vectors(1, 4, 600), 1, next);
    });
    const bool writerBlocked = writer.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    release.set_value();
    BOOST_CHECK(writer.get() != ErrorCode::Success);
    BOOST_CHECK(checkpointEntered);
    BOOST_CHECK(writerBlocked);
    BOOST_CHECK(NodeState(*node).failed);
    BOOST_CHECK(NodeState(*node).recoveryLogBytes > 0);
    node.reset();
    limits.autoCheckpoint = false;
    limits.checkpointHook = {};
    node = OpenNode(directory.root, limits);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 500), 1, results));
    BOOST_REQUIRE_EQUAL(results.size(), 1);
    BOOST_CHECK_EQUAL(results[0].VID, id);
    BOOST_CHECK(NodeState(*node).recoveryLogBytes > 0);
    RequireSuccess(node->Checkpoint());
    BOOST_CHECK_EQUAL(NodeState(*node).recoveryLogBytes, 0);
}

BOOST_AUTO_TEST_CASE(AutoCheckpointCanBeDisabledAndRestoresByteAccounting)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.autoCheckpoint = false;
    limits.checkpointLogBytes = 1;
    limits.checkpointPressureBytes = 1;
    auto node = OpenNode(directory.root, limits);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    const auto bytes = NodeState(*node).recoveryLogBytes;
    BOOST_CHECK(bytes > 0);
    BOOST_CHECK_EQUAL(NodeState(*node).autoCheckpoints, 0);
    node.reset();
    node = OpenNode(directory.root, limits);
    BOOST_CHECK_EQUAL(NodeState(*node).recoveryLogBytes, bytes);
    RequireSuccess(node->Checkpoint());
    BOOST_CHECK_EQUAL(NodeState(*node).recoveryLogBytes, 0);
}

BOOST_AUTO_TEST_CASE(AutoCheckpointCloseInterruptsCooldownAndPreservesUpdates, *boost::unit_test::timeout(60))
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.checkpointLogBytes = 1;
    limits.checkpointPressureBytes = 1 << 20;
    limits.checkpointMinIntervalSeconds = 3600;
    auto node = OpenNode(directory.root, limits);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    auto closing = std::async(std::launch::async, [&] { return node->Close(); });
    BOOST_CHECK(closing.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    RequireSuccess(closing.get());
    node.reset();
    node = OpenNode(directory.root, limits);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 500), 1, results));
    BOOST_REQUIRE_EQUAL(results.size(), 1);
    BOOST_CHECK_EQUAL(results[0].VID, id);
}

BOOST_AUTO_TEST_CASE(AutoCheckpointRejectsInvalidThresholds)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    std::unique_ptr<TenantIndexManager> node;
    limits.checkpointLogBytes = 0;
    BOOST_CHECK(TenantIndexManager::Open(directory.root, limits, node) == ErrorCode::FailedParseValue);
    limits.checkpointLogBytes = 100;
    limits.checkpointPressureBytes = 99;
    BOOST_CHECK(TenantIndexManager::Open(directory.root, limits, node) == ErrorCode::FailedParseValue);
    limits.autoCheckpoint = false;
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
}

BOOST_AUTO_TEST_CASE(DirtyCheckpointReusesChunksAndRemovesObsoleteOnes)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    TenantIndexManager::NodeStats stats;
    std::vector<TenantIndexManager::TenantStats> tenants;
    node->GetStats(stats, tenants);
    const auto prefix = SharedRocksDB::Namespace("A", tenants[0].indexID);
    const auto checkpointPath = [&] {
        for (const auto &entry : std::filesystem::directory_iterator(directory.root))
            if (entry.path().filename().string().find("checkpoint-") == 0)
                return entry.path().string();
        BOOST_FAIL("No published checkpoint");
        return std::string();
    };
    const auto snapshot = [&] {
        std::vector<rocksdb::ColumnFamilyDescriptor> families;
        for (auto name : {"default", "postings", "head_snapshots", "checkpoint_meta"})
            families.emplace_back(name, rocksdb::ColumnFamilyOptions());
        std::vector<rocksdb::ColumnFamilyHandle *> handles;
        rocksdb::DB *raw = nullptr;
        BOOST_REQUIRE(
            rocksdb::DB::OpenForReadOnly(rocksdb::DBOptions(), checkpointPath(), families, &handles, &raw).ok());
        std::unique_ptr<rocksdb::DB> db(raw);
        std::map<std::string, std::string> values;
        {
            std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(rocksdb::ReadOptions(), handles[2]));
            for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix); it->Next())
                values.emplace(it->key().ToString(), it->value().ToString());
            BOOST_REQUIRE(it->status().ok());
        }
        auto sequence = db->GetLatestSequenceNumber();
        for (auto *handle : handles)
            delete handle;
        return std::make_pair(sequence, std::move(values));
    };
    auto before = snapshot();
    RequireSuccess(node->DeleteVector("A", 0));
    RequireSuccess(node->Checkpoint());
    auto after = snapshot();
    BOOST_REQUIRE_GT(before.second.size(), 3);
    BOOST_CHECK_LT(after.first - before.first, before.second.size());
    BOOST_CHECK(before.second != after.second);
    node.reset();

    const auto fileKey = [&](const std::string &file, uint64_t chunk) {
        return prefix + SharedRocksDB::EncodeNumber(file.size()) + file + SharedRocksDB::EncodeNumber(chunk);
    };
    {
        std::shared_ptr<SharedRocksDB> engine;
        RequireSuccess(SharedRocksDB::Open(checkpointPath(), {"postings", "head_snapshots", "checkpoint_meta"}, engine,
                                           false, rocksdb::NewLRUCache(1 << 20)));
        auto manifest = after.second.at(prefix + "manifest");
        uint64_t count;
        BOOST_REQUIRE(SharedRocksDB::DecodeNumber(manifest.substr(0, 8), count));
        size_t position = 8;
        bool enlarged = false;
        for (uint64_t i = 0; i < count; ++i)
        {
            uint64_t length;
            BOOST_REQUIRE(SharedRocksDB::DecodeNumber(manifest.substr(position, 8), length));
            position += 8;
            auto file = manifest.substr(position, length);
            position += length;
            if (file == "indexloader.ini")
            {
                auto data = after.second.at(fileKey(file, 0));
                // Comments are valid padding that disappears on serialization.
                // A version-map binary with trailing garbage is not a valid snapshot.
                data += "\n;";
                while (data.size() + 2 < (1 << 20))
                    data += ";\n";
                data.resize(1 << 20, ';');
                RequireSuccess(engine->Put("head_snapshots", fileKey(file, 0), data));
                RequireSuccess(engine->Put("head_snapshots", fileKey(file, 1), ";\n;\n"));
                manifest.replace(position, 8, SharedRocksDB::EncodeNumber((1 << 20) + 4));
                enlarged = true;
            }
            position += 8;
        }
        BOOST_REQUIRE(enlarged);
        manifest.replace(0, 8, SharedRocksDB::EncodeNumber(count + 2));
        for (const std::string file : {"obsolete.bin", "obsolete-empty.bin"})
        {
            const std::string data = file == "obsolete.bin" ? "old" : "";
            manifest += SharedRocksDB::EncodeNumber(file.size()) + file + SharedRocksDB::EncodeNumber(data.size());
            if (!data.empty())
                RequireSuccess(engine->Put("head_snapshots", fileKey(file, 0), data));
        }
        RequireSuccess(engine->Put("head_snapshots", prefix + "manifest", manifest));
    }
    node = OpenNode(directory.root);
    RequireSuccess(node->DeleteVector("A", 1));
    RequireSuccess(node->Checkpoint());
    after = snapshot();
    BOOST_CHECK_EQUAL(after.second.count(fileKey("indexloader.ini", 1)), 0);
    BOOST_CHECK_EQUAL(after.second.count(fileKey("obsolete.bin", 0)), 0);
    BOOST_CHECK_LT(after.second.at(fileKey("indexloader.ini", 0)).size(), 1 << 20);
    BOOST_CHECK(after.second.at(prefix + "manifest").find("obsolete") == std::string::npos);
    node.reset();
    node = OpenNode(directory.root);
    BOOST_CHECK(node->DeleteVector("A", 0) == ErrorCode::VectorNotFound);
    BOOST_CHECK(node->DeleteVector("A", 1) == ErrorCode::VectorNotFound);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 0.2f), 1, results));
    BOOST_REQUIRE(!results.empty());
    BOOST_CHECK_EQUAL(results[0].VID, 2);
}

BOOST_AUTO_TEST_CASE(ResidentCacheLRUDirtyEvictionAndRecovery)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.maxResidentTenants = 2;
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    for (auto name : {"A", "B", "C"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    std::vector<std::filesystem::path> configs;
    auto oldTime = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
    for (const auto &entry : std::filesystem::recursive_directory_iterator(directory.root))
        if (entry.path().filename() == "indexloader.ini")
        {
            configs.push_back(entry.path());
            std::filesystem::last_write_time(entry.path(), oldTime);
        }
    BOOST_REQUIRE(!configs.empty());
    TenantIndexManager::NodeStats state;
    std::vector<TenantIndexManager::TenantStats> tenants;
    auto refresh = [&] { node->GetStats(state, tenants); };
    auto search = [&](const char *name, float value = 0) {
        std::vector<BasicResult> results;
        RequireSuccess(node->Search(name, Vectors(1, 4, value), 1, results));
        return results[0].VID;
    };
    refresh();
    BOOST_CHECK_EQUAL(state.residentTenants, 0);
    search("A");
    search("B");
    search("A");
    search("C");
    for (const auto &config : configs)
        BOOST_CHECK(std::filesystem::last_write_time(config) == oldTime);
    refresh();
    BOOST_CHECK_EQUAL(state.residentTenants, 2);
    BOOST_CHECK(tenants[1].state == TenantIndexManager::Residency::Unloaded);
    BOOST_CHECK_EQUAL(tenants[1].evictions, 1);
    SizeType first;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, first));
    refresh();
    BOOST_CHECK(tenants[0].dirty);
    search("B");
    search("C");
    refresh();
    BOOST_CHECK(tenants[0].state == TenantIndexManager::Residency::Unloaded);
    BOOST_CHECK(!tenants[0].dirty);
    BOOST_CHECK_EQUAL(search("A", 500), first);
    node.reset();
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    refresh();
    BOOST_CHECK_EQUAL(state.residentTenants, 0);
    BOOST_CHECK_EQUAL(search("A", 500), first);
    RequireSuccess(node->DeleteTenant("A"));
    refresh();
    BOOST_CHECK_EQUAL(state.residentTenants, 0);
    RequireSuccess(node->Close());
    refresh();
    BOOST_CHECK(state.closed);
    BOOST_CHECK(tenants.empty());
}

BOOST_AUTO_TEST_CASE(ResidentByteCapacityAndConcurrentMisses)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    std::vector<BasicResult> found;
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 1, found));
    const auto one = NodeState(*node).residentBytes;
    RequireSuccess(node->Search("B", Vectors(1, 4, 0), 1, found));
    const auto both = NodeState(*node).residentBytes;
    limits.maxResidentBytes = both - (std::min)(one, both - one) / 2;
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root, limits);
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<ErrorCode>> readers;
    for (int i = 0; i < 8; ++i)
        readers.emplace_back(std::async(std::launch::async, [&] {
            gate.wait();
            std::vector<BasicResult> results;
            return node->Search("A", Vectors(1, 4, 0), 1, results);
        }));
    start.set_value();
    for (auto &reader : readers)
        RequireSuccess(reader.get());
    TenantIndexManager::NodeStats state;
    std::vector<TenantIndexManager::TenantStats> tenants;
    node->GetStats(state, tenants);
    BOOST_CHECK_EQUAL(tenants[0].loads, 1);
    BOOST_CHECK_EQUAL(state.residentTenants, 1);
    BOOST_CHECK(!state.overByteTarget);
    BOOST_CHECK(state.residentBytes > 0);
    BOOST_CHECK(state.residentBytes + state.reservedBytes <= limits.maxResidentBytes);
    readers.clear();
    for (int worker = 0; worker < 4; ++worker)
        readers.emplace_back(std::async(std::launch::async, [&, worker] {
            for (int i = 0; i < 10; ++i)
            {
                std::vector<BasicResult> results;
                auto status = node->Search((i + worker) % 2 ? "A" : "B", Vectors(1, 4, 0), 1, results);
                if (status != ErrorCode::Success)
                    return status;
                if (results[0].VID != 0)
                    return ErrorCode::Fail;
            }
            return ErrorCode::Success;
        }));
    for (auto &reader : readers)
        RequireSuccess(reader.get());
    node->GetStats(state, tenants);
    BOOST_CHECK_EQUAL(state.residentTenants, 1);
    BOOST_CHECK_EQUAL(tenants[0].activeRequests + tenants[1].activeRequests, 0);
    BOOST_CHECK(!state.failed);
    RequireSuccess(node->Unload("A"));
    RequireSuccess(node->Unload("B"));
    node->GetStats(state, tenants);
    BOOST_CHECK_EQUAL(state.residentBytes, 0);
    BOOST_CHECK_EQUAL(state.reservedBytes, 0);
    for (const auto &tenant : tenants)
        BOOST_CHECK_EQUAL(tenant.allocatedBytes, 0);
}

BOOST_AUTO_TEST_CASE(ResidentOversizeIsRejectedWithoutFencingOrLeaking)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.maxResidentBytes = 1;
    auto node = OpenNode(directory.root, limits);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        std::vector<BasicResult> results;
        BOOST_CHECK(node->Search("A", Vectors(1, 4, 0), 1, results) == ErrorCode::CapacityExceeded);
        const auto state = NodeState(*node);
        BOOST_CHECK(!state.failed);
        BOOST_CHECK_EQUAL(state.residentTenants, 0);
        BOOST_CHECK_EQUAL(state.residentBytes, 0);
        BOOST_CHECK_EQUAL(state.reservedBytes, 0);
    }
    RequireSuccess(node->Close());
}

BOOST_AUTO_TEST_CASE(ResidentByteCapacityUsesDifferentSizedTenantAllocations)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    auto node = OpenNode(directory.root, limits);
    for (auto name : {"A", "C"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    RequireSuccess(node->Create("B", VectorValueType::Float, 32, DistCalcMethod::L2, Vectors(2048, 32, 0), 2048));
    auto search = [&](const char *name) {
        std::vector<BasicResult> results;
        RequireSuccess(node->Search(name, Vectors(1, name[0] == 'B' ? 32 : 4, 0), 1, results));
        BOOST_REQUIRE(!results.empty());
        BOOST_CHECK_EQUAL(results[0].VID, 0);
    };
    for (auto name : {"A", "B", "C"})
        search(name);
    TenantIndexManager::NodeStats state;
    std::vector<TenantIndexManager::TenantStats> tenants;
    node->GetStats(state, tenants);
    const auto smallPair = tenants[0].allocatedBytes + tenants[2].allocatedBytes;
    const auto large = tenants[1].allocatedBytes;
    BOOST_REQUIRE(large > tenants[0].allocatedBytes);
    const auto pairWithLarge = large + tenants[2].allocatedBytes;
    BOOST_REQUIRE_LT(pairWithLarge, state.residentBytes);
    limits.maxResidentBytes = (state.residentBytes + (std::max)(smallPair, pairWithLarge)) / 2;
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root, limits);
    search("A");
    search("C");
    BOOST_CHECK_EQUAL(NodeState(*node).residentTenants, 2);
    search("B");
    node->GetStats(state, tenants);
    BOOST_CHECK(tenants[0].state == TenantIndexManager::Residency::Unloaded);
    BOOST_CHECK_EQUAL(tenants[0].allocatedBytes, 0);
    BOOST_CHECK(tenants[1].state == TenantIndexManager::Residency::Resident);
    BOOST_CHECK(tenants[2].state == TenantIndexManager::Residency::Resident);
    BOOST_CHECK_EQUAL(state.residentBytes, tenants[1].allocatedBytes + tenants[2].allocatedBytes);
    BOOST_CHECK(state.peakChargedBytes <= limits.maxResidentBytes);
    const auto remaining = tenants[2].allocatedBytes;
    RequireSuccess(node->Unload("B"));
    BOOST_CHECK_EQUAL(NodeState(*node).residentBytes, remaining);
}

BOOST_AUTO_TEST_CASE(ResidentGrowthBelowRebuildThresholdSurvivesReopen)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.autoCheckpoint = false;
    auto node = OpenNode(directory.root, limits);
    std::vector<BasicResult> results;
    Helper::IniReader config;
    config.SetParameter("BuildHead", "AddCountForRebuild", "514");
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(512, 4, 0), 512,
                                IndexAlgoType::BKT, 1, &config));
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 1, results));
    // Full sample, graph, deletion and local-to-global blocks, lazy write-lock tables, plus metadata.
    const uint64_t blockBytes = (1ULL << 20) * (4 * sizeof(float) + 33 * sizeof(SizeType) + 1);
    const uint64_t lazyWriteLocks =
        COMMON::FineGrainedLock().ResidentGrowthBytes() + COMMON::FineGrainedRWLock().ResidentGrowthBytes(0);
    limits.maxResidentBytes = NodeState(*node).residentBytes + blockBytes + lazyWriteLocks + (300ULL << 10);
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root, limits);
    auto inserted = Vectors(1, 4, 500);
    SizeType first = -1;
    // At equality we still reserve the full rebuild overlap.
    BOOST_REQUIRE(node->Insert("A", Vectors(2, 4, 500), 2, first) == ErrorCode::CapacityExceeded);
    BOOST_CHECK_EQUAL(first, -1);
    BOOST_CHECK_EQUAL(NodeState(*node).recoveryLogBytes, 0);
    RequireSuccess(node->Insert("A", inserted, 1, first));
    BOOST_REQUIRE_EQUAL(first, 512);
    auto state = NodeState(*node);
    BOOST_CHECK_GT(state.recoveryLogBytes, 0);
    BOOST_CHECK_LE(state.peakChargedBytes, limits.maxResidentBytes);
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root, limits);
    RequireSuccess(node->Search("A", inserted, 1, results));
    BOOST_REQUIRE_EQUAL(results.size(), 1);
    BOOST_CHECK_EQUAL(results[0].VID, first);
}

BOOST_AUTO_TEST_CASE(ResidentGrowthRejectionPrecedesVIDAndDurableLog)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    auto node = OpenNode(directory.root, limits);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 1, results));
    limits.maxResidentBytes = NodeState(*node).peakChargedBytes * 2;
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root, limits);
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 1, results));
    const auto before = NodeState(*node);
    SizeType first = -7;
    BOOST_CHECK(node->Insert("A", Vectors(1, 4, 500), 1, first) == ErrorCode::CapacityExceeded);
    const auto after = NodeState(*node);
    BOOST_CHECK_EQUAL(first, -7);
    BOOST_CHECK_EQUAL(after.recoveryLogBytes, before.recoveryLogBytes);
    BOOST_CHECK_EQUAL(after.reservedBytes, 0);
    BOOST_CHECK_LE(after.peakChargedBytes, limits.maxResidentBytes);
    BOOST_CHECK(!after.failed);
    RequireSuccess(node->Close());
    node.reset();
    limits.maxResidentBytes = 32ULL << 30;
    node = OpenNode(directory.root, limits);
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, first));
    BOOST_CHECK_EQUAL(first, 128);
}

BOOST_AUTO_TEST_CASE(ResidentLoadingCapacityWaitIsBoundedAndRetryable)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.residentWaitMilliseconds = 50;
    auto node = OpenNode(directory.root, limits);
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    auto probe = std::make_shared<SnapshotProbe>();
    SnapshotFactory factory(probe);
    probe->blockLoad = true;
    auto entered = probe->entered.get_future();
    auto first = std::async(std::launch::async, [&] {
        std::vector<BasicResult> results;
        return node->Search("A", Vectors(1, 4, 0), 1, results);
    });
    const bool loading = entered.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    auto second = std::async(std::launch::async, [&] {
        std::vector<BasicResult> results;
        return node->Search("B", Vectors(1, 4, 0), 1, results);
    });
    const bool bounded = second.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    probe->release.set_value();
    RequireSuccess(first.get());
    const auto status = second.get();
    BOOST_CHECK(loading);
    BOOST_CHECK(bounded);
    BOOST_CHECK(status == ErrorCode::CapacityTimeout);
    BOOST_CHECK(!NodeState(*node).failed);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("B", Vectors(1, 4, 0), 1, results));
}

BOOST_AUTO_TEST_CASE(UnloadAndPrivateCreateDoNotBlockOtherTenant)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("B", Vectors(1, 4, 0), 1, results));
    for (bool creating : {false, true})
    {
        auto probe = std::make_shared<SnapshotProbe>();
        SnapshotFactory factory(probe);
        probe->block = true;
        auto entered = probe->entered.get_future();
        auto slow = std::async(std::launch::async, [&] {
            return creating ? node->Create("C", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128)
                            : node->Unload("A");
        });
        bool saving = entered.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
        auto other = std::async(std::launch::async, [&] {
            std::vector<BasicResult> found;
            auto status = node->Search("B", Vectors(1, 4, 0), 1, found);
            if (status != ErrorCode::Success)
                return status;
            SizeType first;
            return node->Insert("B", Vectors(1, 4, 700), 1, first);
        });
        bool concurrent = other.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        probe->release.set_value();
        RequireSuccess(slow.get());
        RequireSuccess(other.get());
        BOOST_CHECK(saving);
        BOOST_CHECK(concurrent);
    }
}

BOOST_AUTO_TEST_CASE(ResidentSearchContinuesDuringDirectoryMutation)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    auto probe = std::make_shared<SnapshotProbe>();
    SnapshotFactory factory(probe);
    probe->block = true;
    auto entered = probe->entered.get_future();
    auto mutation = std::async(std::launch::async, [&] { return node->DeleteTenant("B"); });
    bool saving = entered.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    auto query = std::async(std::launch::async, [&] {
        std::vector<BasicResult> results;
        VectorValueType type;
        DimensionType dimension;
        auto status = node->Describe("A", type, dimension);
        if (status != ErrorCode::Success || type != VectorValueType::Float || dimension != 4)
            return false;
        status = node->Search("A", Vectors(1, 4, 500), 1, results);
        return status == ErrorCode::Success && results[0].VID == id;
    });
    bool concurrent = query.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    probe->release.set_value();
    RequireSuccess(mutation.get());
    BOOST_CHECK(query.get());
    BOOST_CHECK(saving);
    BOOST_CHECK(concurrent);
    std::vector<BasicResult> results;
    BOOST_CHECK(node->Search("B", Vectors(1, 4, 0), 1, results) == ErrorCode::Key_NotFound);
    RequireSuccess(node->Close());
    BOOST_CHECK(node->Search("A", Vectors(1, 4, 0), 1, results) != ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(CachePressureWithConcurrentWritesAndCheckpoints)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.maxResidentTenants = 1;
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<ErrorCode>> workers;
    std::vector<SizeType> ids[4];
    for (int worker = 0; worker < 4; ++worker)
        workers.emplace_back(std::async(std::launch::async, [&, worker] {
            gate.wait();
            for (int i = 0; i < 8; ++i)
            {
                SizeType first;
                auto status = node->Insert(worker % 2 ? "A" : "B", Vectors(1, 4, 500 + worker * 100 + i), 1, first);
                if (status != ErrorCode::Success)
                    return status;
                ids[worker].push_back(first);
            }
            return ErrorCode::Success;
        }));
    auto checkpoint = std::async(std::launch::async, [&] {
        gate.wait();
        for (int i = 0; i < 3; ++i)
        {
            auto status = node->Checkpoint();
            if (status != ErrorCode::Success)
                return status;
        }
        return ErrorCode::Success;
    });
    start.set_value();
    for (auto &worker : workers)
        RequireSuccess(worker.get());
    RequireSuccess(checkpoint.get());
    for (int tenant = 0; tenant < 2; ++tenant)
    {
        auto combined = ids[tenant];
        combined.insert(combined.end(), ids[tenant + 2].begin(), ids[tenant + 2].end());
        std::sort(combined.begin(), combined.end());
        for (int i = 0; i < 16; ++i)
            BOOST_CHECK_EQUAL(combined[i], 128 + i);
    }
    node.reset();
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    for (auto name : {"A", "B"})
    {
        SizeType first;
        RequireSuccess(node->Insert(name, Vectors(1, 4, 900), 1, first));
        BOOST_CHECK_EQUAL(first, 144);
    }
}

BOOST_AUTO_TEST_CASE(CacheLoadFailureReleasesReservationAndWakesWaiters)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    {
        auto probe = std::make_shared<SnapshotProbe>();
        SnapshotFactory factory(probe);
        probe->failLoad = true;
        std::promise<void> start;
        auto gate = start.get_future().share();
        std::vector<std::future<ErrorCode>> readers;
        for (int i = 0; i < 4; ++i)
            readers.emplace_back(std::async(std::launch::async, [&] {
                gate.wait();
                std::vector<BasicResult> results;
                return node->Search("A", Vectors(1, 4, 0), 1, results);
            }));
        start.set_value();
        for (auto &reader : readers)
            BOOST_CHECK(reader.get() != ErrorCode::Success);
        TenantIndexManager::NodeStats state;
        std::vector<TenantIndexManager::TenantStats> tenants;
        node->GetStats(state, tenants);
        BOOST_CHECK(state.failed);
        BOOST_CHECK_EQUAL(state.residentTenants, 0);
        BOOST_CHECK_EQUAL(state.residentBytes, 0);
        BOOST_CHECK_EQUAL(tenants[0].activeRequests, 0);
    }
    node.reset();
    node = OpenNode(directory.root);
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 1, results));
}

BOOST_AUTO_TEST_CASE(FailedDirtyEvictionRetainsRecoveryLog)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.maxResidentTenants = 1;
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    for (auto name : {"A", "B"})
        RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    SizeType first;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, first));
    std::vector<BasicResult> results;
    {
        auto probe = std::make_shared<SnapshotProbe>();
        SnapshotFactory factory(probe);
        probe->failSave = true;
        BOOST_CHECK(node->Search("B", Vectors(1, 4, 0), 1, results) != ErrorCode::Success);
        TenantIndexManager::NodeStats state;
        std::vector<TenantIndexManager::TenantStats> tenants;
        node->GetStats(state, tenants);
        BOOST_CHECK(state.failed);
        BOOST_CHECK_EQUAL(state.residentTenants, 1);
        BOOST_CHECK(tenants[0].dirty);
    }
    node.reset();
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    RequireSuccess(node->Search("A", Vectors(1, 4, 500), 1, results));
    BOOST_CHECK_EQUAL(results[0].VID, first);
}

BOOST_AUTO_TEST_CASE(FullyDeletedSnapshotAndAdmissionLimits)
{
    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.maxTenants = 1;
    limits.maxVectorsPerTenant = 64;
    limits.maxDimension = 4;
    limits.maxResults = 5;
    std::unique_ptr<TenantIndexManager> node;
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    auto seed = Vectors(64, 4, 0);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, seed, 64));
    BOOST_CHECK(node->Create("B", VectorValueType::Float, 4, DistCalcMethod::L2, seed, 64) != ErrorCode::Success);
    SizeType first;
    BOOST_CHECK(node->Insert("A", Vectors(1, 4, 0), 1, first) != ErrorCode::Success);
    std::vector<BasicResult> results;
    BOOST_CHECK(node->Search("A", Vectors(1, 4, 0), 6, results) != ErrorCode::Success);
    auto invalid = Vectors(1, 4, 0);
    reinterpret_cast<float *>(invalid.Data())[0] = std::numeric_limits<float>::quiet_NaN();
    BOOST_CHECK(node->Search("A", invalid, 1, results) != ErrorCode::Success);
    for (SizeType i = 0; i < 64; ++i)
    {
        RequireSuccess(node->DeleteVector("A", i));
        RequireSuccess(
            node->Search("A", ByteArray(seed.Data() + i * 4 * sizeof(float), 4 * sizeof(float), false), 3, results));
        for (const auto &result : results)
            BOOST_CHECK(result.VID == -1 || result.VID > i);
    }
    RequireSuccess(node->Close());
    node.reset();
    RequireSuccess(TenantIndexManager::Open(directory.root, limits, node));
    RequireSuccess(node->Search("A", Vectors(1, 4, 0), 3, results));
    for (const auto &result : results)
        BOOST_CHECK_EQUAL(result.VID, -1);
}

BOOST_AUTO_TEST_CASE(CosineInputOwnershipAndIntegerTenant)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    auto seed = Vectors(128, 4, 1);
    std::string before(reinterpret_cast<char *>(seed.Data()), seed.Length());
    RequireSuccess(node->Create("cos", VectorValueType::Float, 4, DistCalcMethod::Cosine, seed, 128));
    BOOST_CHECK_EQUAL(std::string(reinterpret_cast<char *>(seed.Data()), seed.Length()), before);
    auto integers = ByteArray::Alloc(128 * 5);
    for (size_t i = 0; i < integers.Length(); ++i)
        integers[i] = static_cast<uint8_t>((i * 13 + i / 5) % 251);
    RequireSuccess(node->Create("int", VectorValueType::UInt8, 5, DistCalcMethod::L2, integers, 128));
    std::vector<BasicResult> results;
    RequireSuccess(node->Search("int", ByteArray(integers.Data(), 5, false), 1, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
    RequireSuccess(node->Close());
    node.reset();
    node = OpenNode(directory.root);
    RequireSuccess(node->Search("cos", ByteArray(seed.Data(), 4 * sizeof(float), false), 1, results));
    BOOST_CHECK_SMALL(results[0].Dist, 0.0001f);
    RequireSuccess(node->Search("int", ByteArray(integers.Data(), 5, false), 1, results));
    BOOST_CHECK_EQUAL(results[0].VID, 0);
}

BOOST_AUTO_TEST_CASE(StrictQueryOptionsAreTenantOnly)
{
    auto settings = std::make_shared<Service::ServiceSettings>();
    for (const auto *query : {"$datatype:invalid 0|0", "$extractmetadata:invalid 0|0", "$resultnum:invalid 0|0"})
    {
        Service::SearchExecutionContext legacy(settings), tenant(settings);
        RequireSuccess(legacy.ParseQuery(query));
        RequireSuccess(tenant.ParseQuery(query));
        RequireSuccess(legacy.ExtractOption());
        BOOST_CHECK(tenant.ExtractOption(true) == ErrorCode::FailedParseValue);
    }
}

BOOST_AUTO_TEST_CASE(TrustedServerRoutesExactlyOneTenant)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    RequireSuccess(node->Create("B", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 1000), 128));
    RequireSuccess(node->Close());
    node.reset();
    const auto config = directory.root + "/server.ini";
    {
        std::ofstream output(config);
        output << "[MultiTenant]\nRoot=" << directory.root << "\n";
    }
    auto service = std::make_shared<Service::ServiceContext>(config);
    BOOST_REQUIRE(service->IsInitialized());
    auto execute = [&](const std::string &query) {
        std::shared_ptr<Service::SearchExecutionContext> result;
        Service::SearchExecutor request(query, service, [&](auto response) { result = std::move(response); });
        request.Execute();
        return result;
    };
    auto response = execute("$tenantid:A $resultnum:1 0|0.01|0.02|0.03");
    BOOST_REQUIRE(response);
    BOOST_REQUIRE_EQUAL(response->GetResults().size(), 1);
    BOOST_CHECK_EQUAL(response->GetResults()[0].m_indexName, "A");
    BOOST_CHECK_EQUAL(response->GetResults()[0].m_results.GetResult(0)->VID, 0);
    for (const auto *bad : {"0|0.01|0.02|0.03", "$tenantid:missing 0|0|0|0", "$tenantid:A $indexname:B 0|0|0|0",
                            "$tenantid:A 0|0", "$tenantid:A $tenantid:B 0|0|0|0", "$tenantid:A $resultnum:1001 0|0|0|0",
                            "$tenantid:A $resultnum:invalid 0|0|0|0"})
        BOOST_CHECK(!execute(bad));
}

BOOST_AUTO_TEST_CASE(PublishedWatermarkSkipsOldLogsAndCorruptionFailsClosed)
{
    TemporaryNode directory;
    auto node = OpenNode(directory.root);
    RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
    SizeType first;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, first));
    node.reset();
    const auto openRecovery = [&] {
        std::shared_ptr<SharedRocksDB> engine;
        RequireSuccess(SharedRocksDB::Open(directory.root + "/recovery", {"recovery_log", "catalog", "recovery_meta"},
                                           engine, false, rocksdb::NewLRUCache(1 << 20)));
        return engine;
    };
    std::string logKey, logRecord;
    {
        auto recovery = openRecovery();
        RequireSuccess(recovery->Scan("recovery_log", "", [&](const auto &key, const auto &value) {
            logKey = key;
            logRecord = value;
            return ErrorCode::Success;
        }));
        BOOST_REQUIRE(!logKey.empty());
    }
    node = OpenNode(directory.root);
    RequireSuccess(node->Checkpoint());
    node.reset();
    {
        auto recovery = openRecovery();
        // Simulate a crash after publishing a checkpoint, before covered-log deletion.
        RequireSuccess(recovery->Put("recovery_log", logKey, logRecord));
    }
    node = OpenNode(directory.root);
    RequireSuccess(node->Insert("A", Vectors(1, 4, 600), 1, first));
    BOOST_CHECK_EQUAL(first, 129);
    node.reset();
    {
        auto recovery = openRecovery();
        auto prefix = logKey.substr(0, logKey.size() - 8);
        RequireSuccess(recovery->Put("recovery_log", prefix + SharedRocksDB::EncodeNumber(2), "truncated"));
    }
    BOOST_CHECK(TenantIndexManager::Open(directory.root, {}, node) != ErrorCode::Success);
    BOOST_CHECK(!node);
}

#ifndef _MSC_VER
BOOST_AUTO_TEST_CASE(ConcurrentCrashWriter)
{
    const char *root = std::getenv("SPTAG_TEST_CONCURRENT_CRASH_ROOT");
    if (!root)
        return;
    auto node = OpenNode(root);
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::vector<std::future<ErrorCode>> writers;
    for (int i = 0; i < 4; ++i)
        writers.emplace_back(std::async(std::launch::async, [&, i] {
            gate.wait();
            SizeType id;
            return node->Insert("A", Vectors(32, 4, 500 + i * 100), 32, id);
        }));
    start.set_value();
    for (auto &writer : writers)
        RequireSuccess(writer.get());
    RequireSuccess(node->DeleteVector("A", 0));
    _exit(0);
}

BOOST_AUTO_TEST_CASE(ConcurrentProcessCrashAndRepeatedReplay)
{
    TemporaryNode directory;
    {
        auto node = OpenNode(directory.root);
        RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128));
        RequireSuccess(node->Close());
    }
    auto child = fork();
    BOOST_REQUIRE(child >= 0);
    if (child == 0)
    {
        setenv("SPTAG_TEST_CONCURRENT_CRASH_ROOT", directory.root.c_str(), 1);
        execl("/proc/self/exe", "/proc/self/exe", "--run_test=MultiTenant/ConcurrentCrashWriter", nullptr);
        _exit(127);
    }
    int status;
    BOOST_REQUIRE_EQUAL(waitpid(child, &status, 0), child);
    BOOST_REQUIRE(WIFEXITED(status));
    BOOST_REQUIRE_EQUAL(WEXITSTATUS(status), 0);
    std::vector<SizeType> firstIDs;
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        auto node = OpenNode(directory.root);
        std::set<SizeType> unique;
        for (int i = 0; i < 4; ++i)
        {
            std::vector<BasicResult> results;
            RequireSuccess(node->Search("A", Vectors(1, 4, 500 + i * 100), 1, results));
            auto id = results[0].VID;
            BOOST_CHECK(id >= 128 && id < 256);
            unique.insert(id);
            if (attempt == 0)
                firstIDs.push_back(id);
            else
                BOOST_CHECK_EQUAL(id, firstIDs[i]);
        }
        BOOST_CHECK_EQUAL(unique.size(), 4);
        BOOST_CHECK(node->DeleteVector("A", 0) == ErrorCode::VectorNotFound);
    }
}

BOOST_AUTO_TEST_CASE(CrashWriter)
{
    const char *root = std::getenv("SPTAG_TEST_CRASH_ROOT");
    if (!root)
        return;
    auto node = OpenNode(root);
    SizeType id;
    RequireSuccess(node->Insert("A", Vectors(1, 4, 500), 1, id));
    BOOST_REQUIRE_EQUAL(id, 128);
    RequireSuccess(node->DeleteVector("A", 0));
    _exit(0);
}

BOOST_AUTO_TEST_CASE(ProcessCrashAndRepeatedReplay)
{
    TemporaryNode directory;
    {
        auto node = OpenNode(directory.root);
        RequireSuccess(node->Create("A", VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128,
                                    IndexAlgoType::BKT, 2));
        RequireSuccess(node->Close());
    }
    auto child = fork();
    BOOST_REQUIRE(child >= 0);
    if (child == 0)
    {
        setenv("SPTAG_TEST_CRASH_ROOT", directory.root.c_str(), 1);
        execl("/proc/self/exe", "/proc/self/exe", "--run_test=MultiTenant/CrashWriter", nullptr);
        _exit(127);
    }
    int status;
    BOOST_REQUIRE_EQUAL(waitpid(child, &status, 0), child);
    BOOST_REQUIRE(WIFEXITED(status));
    BOOST_REQUIRE_EQUAL(WEXITSTATUS(status), 0);
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        auto node = OpenNode(directory.root);
        std::vector<BasicResult> results;
        RequireSuccess(node->Search("A", Vectors(1, 4, 500), 3, results));
        BOOST_CHECK_EQUAL(results[0].VID, 128);
        RequireSuccess(node->Search("A", Vectors(1, 4, 0), 3, results));
        for (const auto &result : results)
            BOOST_CHECK_NE(result.VID, 0);
        // No Close(): reopening must replay the original log against a fresh checkpoint copy.
    }
}

BOOST_AUTO_TEST_CASE(ReadOnlyLoadDefersThreadsAndLocksAndCleanReopenSkipsLoad)
{
    auto countThreads = [] {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line))
        {
            if (line.rfind("Threads:", 0) == 0)
            {
                int threads = 0;
                std::istringstream(line.substr(8)) >> threads;
                return threads;
            }
        }
        return -1;
    };

    TemporaryNode directory;
    TenantIndexManager::Limits limits;
    limits.maxResidentTenants = 16;
    {
        auto node = OpenNode(directory.root, limits);
        for (auto name : {"T0", "T1", "T2", "T3"})
            RequireSuccess(node->Create(name, VectorValueType::Float, 4, DistCalcMethod::L2, Vectors(128, 4, 0), 128,
                                        IndexAlgoType::BKT, 2));
        RequireSuccess(node->Checkpoint());
        RequireSuccess(node->Close());
    }

    auto node = OpenNode(directory.root, limits);
    TenantIndexManager::NodeStats nodeStats;
    std::vector<TenantIndexManager::TenantStats> tenantStats;
    node->GetStats(nodeStats, tenantStats);
    BOOST_CHECK_EQUAL(nodeStats.residentTenants, 0);
    BOOST_CHECK_EQUAL(nodeStats.residentBytes, 0);
    for (const auto &ts : tenantStats)
    {
        BOOST_CHECK_GT(ts.snapshotBytes, 0);
        BOOST_CHECK_EQUAL(ts.loads, 0);
    }

    std::vector<BasicResult> results;
    RequireSuccess(node->Search("T0", Vectors(1, 4, 0), 1, results));
    const int threadsAfterFirst = countThreads();
    BOOST_REQUIRE_GT(threadsAfterFirst, 0);
    for (auto name : {"T1", "T2", "T3"})
        RequireSuccess(node->Search(name, Vectors(1, 4, 0), 1, results));
    BOOST_CHECK_EQUAL(countThreads(), threadsAfterFirst);

    node->GetStats(nodeStats, tenantStats);
    const uint64_t readOnlyBytes = tenantStats[0].allocatedBytes;
    BOOST_CHECK_LT(readOnlyBytes, 256ULL << 10);

    SizeType firstID = -1;
    RequireSuccess(node->Insert("T0", Vectors(1, 4, 42), 1, firstID));
    BOOST_CHECK_EQUAL(firstID, 128);
    node->GetStats(nodeStats, tenantStats);
    BOOST_CHECK_GT(tenantStats[0].allocatedBytes, readOnlyBytes);
    RequireSuccess(node->Close());
}
#endif

BOOST_AUTO_TEST_SUITE_END()
#endif
