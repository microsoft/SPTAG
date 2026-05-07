// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Core/Common/IVersionMap.h"
#include "inc/Core/Common/LocalVersionMap.h"

#ifdef TIKV
#include "inc/Core/SPANN/ExtraTiKVController.h"
#include "inc/Core/Common/TiKVVersionMap.h"
#endif

#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <set>

using namespace SPTAG;
using namespace SPTAG::COMMON;

// Helper: create a LocalVersionMap
static std::unique_ptr<LocalVersionMap> MakeLocalVersionMap()
{
    return std::make_unique<LocalVersionMap>();
}

#ifdef TIKV
// Helper: create a TiKVVersionMap connected to the live TiKV cluster.
// Requires env TIKV_PD_ADDRESSES (e.g. "127.0.0.1:23791,127.0.0.1:23792,127.0.0.1:23793").
// Uses a unique key prefix per test to avoid collision.
static std::unique_ptr<TiKVVersionMap> MakeTiKVVersionMap(const std::string& testName, int chunkSize = 64, int cacheMaxChunks = 100)
{
    const char* pdAddr = std::getenv("TIKV_PD_ADDRESSES");
    if (!pdAddr || std::string(pdAddr).empty()) {
        BOOST_TEST_MESSAGE("TIKV_PD_ADDRESSES not set, skipping TiKV test");
        return nullptr;
    }

    // Unique prefix per test invocation
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string prefix = "vmtest_" + testName + "_" + std::to_string(now) + "_";

    auto db = std::make_shared<SPTAG::SPANN::TiKVIO>(std::string(pdAddr), prefix);
    auto vm = std::make_unique<TiKVVersionMap>();
    vm->SetDB(db);
    vm->SetLayer(0);
    vm->SetChunkSize(chunkSize);
    vm->SetCacheMaxChunks(cacheMaxChunks);
    return vm;
}
#endif

// ==============================================================
// LocalVersionMap Tests
// ==============================================================

BOOST_AUTO_TEST_SUITE(VersionMapTest)

BOOST_AUTO_TEST_CASE(Local_InitializeAndCount)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(100, 1, 100);
    BOOST_CHECK_EQUAL(vm->Count(), 100);
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 0);
}

BOOST_AUTO_TEST_CASE(Local_DeleteAndDeleted)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(50, 1, 50);

    BOOST_CHECK(!vm->Deleted(0));
    BOOST_CHECK(vm->Delete(0));
    BOOST_CHECK(vm->Deleted(0));
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 1);

    // Double delete returns false
    BOOST_CHECK(!vm->Delete(0));
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 1);
}

BOOST_AUTO_TEST_CASE(Local_GetSetVersion)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(100, 1, 100);

    // VersionLabel initializes to 0xff (not 0, not deleted=0xfe)
    BOOST_CHECK_EQUAL(vm->GetVersion(0), 0xff);

    vm->SetVersion(5, 42);
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 42);
}

BOOST_AUTO_TEST_CASE(Local_IncVersion)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(100, 1, 100);

    // VersionLabel initial value is 0xff, so first Inc gives (0xff+1)&0x7f = 0
    uint8_t newVer = 0;
    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 0);

    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 1);

    // IncVersion on deleted VID should fail
    vm->Delete(10);
    BOOST_CHECK(!vm->IncVersion(10, &newVer));
}

BOOST_AUTO_TEST_CASE(Local_IncVersionWraparound)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(10, 1, 10);

    // Set version to 0x7e (126), one below max
    vm->SetVersion(0, 0x7e);
    uint8_t newVer = 0;
    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 0x7f); // 127

    // Next increment should wrap to 0
    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 0x00);
}

BOOST_AUTO_TEST_CASE(Local_AddBatch)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(100, 1, 200);

    BOOST_CHECK_EQUAL(vm->Count(), 100);
    vm->AddBatch(50);
    BOOST_CHECK_EQUAL(vm->Count(), 150);
}

BOOST_AUTO_TEST_CASE(Local_DeleteAll)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(20, 1, 20);

    vm->DeleteAll();
    for (int i = 0; i < 20; i++) {
        BOOST_CHECK(vm->Deleted(i));
    }
}

BOOST_AUTO_TEST_CASE(Local_BatchGetVersions)
{
    auto vm = MakeLocalVersionMap();
    vm->Initialize(100, 1, 100);

    vm->SetVersion(5, 10);
    vm->SetVersion(10, 20);
    vm->Delete(15);

    std::vector<SizeType> vids = {5, 10, 15, 0, 99};
    std::vector<uint8_t> versions;
    vm->BatchGetVersions(vids, versions);

    BOOST_CHECK_EQUAL(versions[0], 10);
    BOOST_CHECK_EQUAL(versions[1], 20);
    BOOST_CHECK_EQUAL(versions[2], 0xfe); // deleted
    BOOST_CHECK_EQUAL(versions[3], 0xff); // VersionLabel defaults to 0xff
    BOOST_CHECK_EQUAL(versions[4], 0xff); // VersionLabel defaults to 0xff

    // Out-of-range VIDs → 0xfe
    std::vector<SizeType> oob = {-1, 100, 200};
    std::vector<uint8_t> oobVersions;
    vm->BatchGetVersions(oob, oobVersions);
    for (auto v : oobVersions) {
        BOOST_CHECK_EQUAL(v, 0xfe);
    }
}

// ==============================================================
// TiKV VersionMap Tests (skipped if TIKV not available)
// ==============================================================

#ifdef TIKV

BOOST_AUTO_TEST_CASE(TiKV_InitializeAlive)
{
    auto vm = MakeTiKVVersionMap("InitAlive");
    if (!vm) return;

    vm->Initialize(100, 1, 100);
    BOOST_CHECK_EQUAL(vm->Count(), 100);
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 0);

    // All VIDs should be alive (version 0)
    for (int i = 0; i < 100; i++) {
        BOOST_CHECK(!vm->Deleted(i));
        BOOST_CHECK_EQUAL(vm->GetVersion(i), 0);
    }
}

BOOST_AUTO_TEST_CASE(TiKV_ChunkBoundary)
{
    // chunkSize=64, so VID 63 is the last in chunk 0, VID 64 is first in chunk 1
    auto vm = MakeTiKVVersionMap("ChunkBound", 64);
    if (!vm) return;

    vm->Initialize(200, 1, 200);

    // Set versions at boundaries
    vm->SetVersion(63, 10);
    vm->SetVersion(64, 20);
    vm->SetVersion(127, 30);
    vm->SetVersion(128, 40);

    BOOST_CHECK_EQUAL(vm->GetVersion(63), 10);
    BOOST_CHECK_EQUAL(vm->GetVersion(64), 20);
    BOOST_CHECK_EQUAL(vm->GetVersion(127), 30);
    BOOST_CHECK_EQUAL(vm->GetVersion(128), 40);

    // Non-boundary VIDs should still be 0
    BOOST_CHECK_EQUAL(vm->GetVersion(0), 0);
    BOOST_CHECK_EQUAL(vm->GetVersion(65), 0);
}

BOOST_AUTO_TEST_CASE(TiKV_DeleteAndDeleted)
{
    auto vm = MakeTiKVVersionMap("DeleteCheck");
    if (!vm) return;

    vm->Initialize(50, 1, 50);

    BOOST_CHECK(!vm->Deleted(0));
    BOOST_CHECK(vm->Delete(0));
    BOOST_CHECK(vm->Deleted(0));
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 1);

    // Double delete returns false
    BOOST_CHECK(!vm->Delete(0));
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 1);
}

BOOST_AUTO_TEST_CASE(TiKV_DeleteMultiple)
{
    auto vm = MakeTiKVVersionMap("DeleteMulti");
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    for (int i = 0; i < 100; i += 2) {
        BOOST_CHECK(vm->Delete(i));
    }
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 50);

    for (int i = 0; i < 100; i++) {
        if (i % 2 == 0) {
            BOOST_CHECK(vm->Deleted(i));
        } else {
            BOOST_CHECK(!vm->Deleted(i));
        }
    }
}

BOOST_AUTO_TEST_CASE(TiKV_GetSetVersion)
{
    auto vm = MakeTiKVVersionMap("GetSet");
    if (!vm) return;

    vm->Initialize(100, 1, 100);
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 0);

    vm->SetVersion(5, 42);
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 42);

    // Setting to 0xfe should count as delete
    SizeType delBefore = vm->GetDeleteCount();
    vm->SetVersion(10, 0xfe);
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), delBefore + 1);
    BOOST_CHECK(vm->Deleted(10));

    // SetVersion from deleted → alive should decrement delete count
    vm->SetVersion(10, 0x00);
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), delBefore);
    BOOST_CHECK(!vm->Deleted(10));
}

BOOST_AUTO_TEST_CASE(TiKV_IncVersionBasic)
{
    auto vm = MakeTiKVVersionMap("IncBasic");
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    uint8_t newVer = 0;
    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 1);

    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 2);

    // Inc deleted VID should fail
    vm->Delete(10);
    BOOST_CHECK(!vm->IncVersion(10, &newVer));
}

BOOST_AUTO_TEST_CASE(TiKV_IncVersionWithExpectedOld)
{
    auto vm = MakeTiKVVersionMap("IncCAS");
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    // VID 0 is at version 0. IncVersion(0, ptr, 0) should set it to 1.
    uint8_t newVer = 0;
    BOOST_CHECK(vm->IncVersion(0, &newVer, 0));
    BOOST_CHECK_EQUAL(newVer, 1);

    // Now it's at 1. IncVersion(0, ptr, 0) again →
    // current(1) == (expectedOld(0)+1)&0x7f == 1 → idempotent success
    BOOST_CHECK(vm->IncVersion(0, &newVer, 0));
    BOOST_CHECK_EQUAL(newVer, 1);

    // IncVersion(0, ptr, 1) should succeed → 2
    BOOST_CHECK(vm->IncVersion(0, &newVer, 1));
    BOOST_CHECK_EQUAL(newVer, 2);

    // IncVersion(0, ptr, 0) now: current=2, expected_target=(0+1)&0x7f=1 != 2, expected=0 != 2 → fail
    BOOST_CHECK(!vm->IncVersion(0, &newVer, 0));
}

BOOST_AUTO_TEST_CASE(TiKV_IncVersionIdempotent)
{
    auto vm = MakeTiKVVersionMap("IncIdem");
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    // VID 0 at version 0 → IncVersion(0, ptr, 0) → version 1
    uint8_t newVer = 0;
    BOOST_CHECK(vm->IncVersion(0, &newVer, 0));
    BOOST_CHECK_EQUAL(newVer, 1);

    // Now VID 0 is already at version 1.
    // IncVersion(0, ptr, 0) again with expectedOld=0 should detect
    // that current == (0+1)&0x7f == 1 → idempotent success
    BOOST_CHECK(vm->IncVersion(0, &newVer, 0));
    BOOST_CHECK_EQUAL(newVer, 1);
}

BOOST_AUTO_TEST_CASE(TiKV_IncVersionWraparound)
{
    auto vm = MakeTiKVVersionMap("IncWrap");
    if (!vm) return;

    vm->Initialize(10, 1, 10);

    vm->SetVersion(0, 0x7e);
    uint8_t newVer = 0;
    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 0x7f);

    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 0x00);
}

BOOST_AUTO_TEST_CASE(TiKV_AddBatch)
{
    auto vm = MakeTiKVVersionMap("AddBatch");
    if (!vm) return;

    vm->Initialize(100, 1, 100);
    BOOST_CHECK_EQUAL(vm->Count(), 100);

    vm->AddBatch(50);
    BOOST_CHECK_EQUAL(vm->Count(), 150);

    // Old VIDs should still be alive
    BOOST_CHECK(!vm->Deleted(0));
    BOOST_CHECK(!vm->Deleted(99));

    // VIDs in newly created chunks (beyond old last chunk) should be deleted
    // With chunkSize=64: old chunks are 0,1. NewLastChunk = ChunkId(149)=2.
    // VIDs 128-149 are in chunk 2, which is newly created as 0xfe.
    BOOST_CHECK(vm->Deleted(128));
    BOOST_CHECK(vm->Deleted(149));

    // VIDs 100-127 are in chunk 1 (existing), initialized as 0x00 by Initialize
    // They remain 0x00 after AddBatch (not overwritten)
    BOOST_CHECK(!vm->Deleted(100));
    BOOST_CHECK(!vm->Deleted(127));
}

BOOST_AUTO_TEST_CASE(TiKV_DeleteAll)
{
    auto vm = MakeTiKVVersionMap("DelAll");
    if (!vm) return;

    vm->Initialize(50, 1, 50);
    vm->DeleteAll();

    for (int i = 0; i < 50; i++) {
        BOOST_CHECK(vm->Deleted(i));
    }
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), 50);
}

BOOST_AUTO_TEST_CASE(TiKV_BatchGetVersions)
{
    auto vm = MakeTiKVVersionMap("BatchGet");
    if (!vm) return;

    vm->Initialize(200, 1, 200);
    vm->SetVersion(5, 10);
    vm->SetVersion(10, 20);
    vm->SetVersion(63, 30);  // chunk boundary
    vm->SetVersion(64, 40);  // next chunk
    vm->Delete(100);

    std::vector<SizeType> vids = {5, 10, 63, 64, 100, 0, 199};
    std::vector<uint8_t> versions;
    vm->BatchGetVersions(vids, versions);

    BOOST_CHECK_EQUAL(versions.size(), 7u);
    BOOST_CHECK_EQUAL(versions[0], 10);
    BOOST_CHECK_EQUAL(versions[1], 20);
    BOOST_CHECK_EQUAL(versions[2], 30);
    BOOST_CHECK_EQUAL(versions[3], 40);
    BOOST_CHECK_EQUAL(versions[4], 0xfe); // deleted
    BOOST_CHECK_EQUAL(versions[5], 0);    // default
    BOOST_CHECK_EQUAL(versions[6], 0);    // default
}

BOOST_AUTO_TEST_CASE(TiKV_BatchGetVersionsOOB)
{
    auto vm = MakeTiKVVersionMap("BatchOOB");
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    std::vector<SizeType> vids = {-1, 100, 200};
    std::vector<uint8_t> versions;
    vm->BatchGetVersions(vids, versions);

    for (auto v : versions) {
        BOOST_CHECK_EQUAL(v, 0xfe);
    }
}

BOOST_AUTO_TEST_CASE(TiKV_BatchGetVersionsEmpty)
{
    auto vm = MakeTiKVVersionMap("BatchEmpty");
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    std::vector<SizeType> vids;
    std::vector<uint8_t> versions;
    vm->BatchGetVersions(vids, versions);
    BOOST_CHECK(versions.empty());
}

// ==============================================================
// Cache Tests
// ==============================================================

BOOST_AUTO_TEST_CASE(TiKV_CacheHitMiss)
{
    auto vm = MakeTiKVVersionMap("CacheHit", 64, 100);
    if (!vm) return;

    vm->Initialize(200, 1, 200);
    vm->SetVersion(5, 42);

    // First read populates cache
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 42);

    // Second read should come from cache
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 42);

    // Read another VID in the same chunk (0..63) — should also be cached now
    BOOST_CHECK_EQUAL(vm->GetVersion(0), 0);
}

BOOST_AUTO_TEST_CASE(TiKV_CacheNoTTLExpiry)
{
    auto vm = MakeTiKVVersionMap("CacheNoTTL", 64, 100);
    if (!vm) return;

    vm->Initialize(100, 1, 100);
    vm->SetVersion(5, 10);

    // Read to populate cache
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 10);

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Cache is capacity-bound only; elapsed time should not expire entries.
    BOOST_CHECK_EQUAL(vm->GetVersion(5), 10);
}

BOOST_AUTO_TEST_CASE(TiKV_CacheEviction)
{
    // Small cache: max 3 chunks, chunkSize=64
    auto vm = MakeTiKVVersionMap("CacheEvict", 64, 3);
    if (!vm) return;

    vm->Initialize(500, 1, 500);

    // Set version in 4 different chunks
    vm->SetVersion(0, 10);    // chunk 0
    vm->SetVersion(64, 20);   // chunk 1
    vm->SetVersion(128, 30);  // chunk 2
    vm->SetVersion(192, 40);  // chunk 3

    // Read all 4 — cache should evict chunk 0 (LRU)
    BOOST_CHECK_EQUAL(vm->GetVersion(0), 10);
    BOOST_CHECK_EQUAL(vm->GetVersion(64), 20);
    BOOST_CHECK_EQUAL(vm->GetVersion(128), 30);
    BOOST_CHECK_EQUAL(vm->GetVersion(192), 40);

    // All data should still be correct (even if evicted, re-fetched from TiKV)
    BOOST_CHECK_EQUAL(vm->GetVersion(0), 10);
}

BOOST_AUTO_TEST_CASE(TiKV_BatchGetWithCache)
{
    auto vm = MakeTiKVVersionMap("BatchCache", 64, 100);
    if (!vm) return;

    vm->Initialize(200, 1, 200);
    vm->SetVersion(5, 10);
    vm->SetVersion(100, 20);

    // First BatchGet populates cache for chunks containing VID 5 and 100
    std::vector<SizeType> vids1 = {5, 100};
    std::vector<uint8_t> vers1;
    vm->BatchGetVersions(vids1, vers1);
    BOOST_CHECK_EQUAL(vers1[0], 10);
    BOOST_CHECK_EQUAL(vers1[1], 20);

    // Second BatchGet should hit cache
    std::vector<SizeType> vids2 = {5, 6, 100, 101};
    std::vector<uint8_t> vers2;
    vm->BatchGetVersions(vids2, vers2);
    BOOST_CHECK_EQUAL(vers2[0], 10);
    BOOST_CHECK_EQUAL(vers2[1], 0);  // VID 6, same chunk as 5
    BOOST_CHECK_EQUAL(vers2[2], 20);
    BOOST_CHECK_EQUAL(vers2[3], 0);  // VID 101, same chunk as 100
}

// ==============================================================
// Concurrent Access Tests
// ==============================================================

BOOST_AUTO_TEST_CASE(TiKV_ConcurrentDelete)
{
    auto vm = MakeTiKVVersionMap("ConcDel", 64, 0);  // no cache, to avoid stale reads
    if (!vm) return;

    const int N = 256;  // exactly 4 chunks of 64
    vm->Initialize(N, 1, N);

    const int numThreads = 4;
    std::atomic<int> successCount{0};
    std::vector<std::thread> threads;

    // Each thread deletes its own range — no overlapping VIDs
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&, t]() {
            int start = t * (N / numThreads);
            int end = (t + 1) * (N / numThreads);
            for (int i = start; i < end; i++) {
                if (vm->Delete(i)) {
                    successCount++;
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // With non-overlapping ranges over distinct chunks, all deletes should succeed.
    BOOST_CHECK_EQUAL(successCount.load(), N);
    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), N);

    for (int i = 0; i < N; i++) {
        BOOST_CHECK(vm->Deleted(i));
    }
}

BOOST_AUTO_TEST_CASE(TiKV_ConcurrentIncVersion)
{
    auto vm = MakeTiKVVersionMap("ConcInc", 64, 0);
    if (!vm) return;

    const int N = 100;
    vm->Initialize(N, 1, N);

    const int numThreads = 4;
    const int incsPerThread = 10;
    std::vector<std::thread> threads;

    // Each thread increments all VIDs. With chunk write mutex, increments are
    // serialized per chunk — total should be numThreads * incsPerThread per VID
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back([&]() {
            for (int round = 0; round < incsPerThread; round++) {
                for (int i = 0; i < N; i++) {
                    uint8_t newVer;
                    vm->IncVersion(i, &newVer);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // Each VID should have been incremented numThreads * incsPerThread times
    int expected = (numThreads * incsPerThread) & 0x7f;
    for (int i = 0; i < N; i++) {
        uint8_t v = vm->GetVersion(i);
        BOOST_CHECK_EQUAL(v, expected);
    }
}

BOOST_AUTO_TEST_CASE(TiKV_ConcurrentReadWrite)
{
    auto vm = MakeTiKVVersionMap("ConcRW", 64, 100);
    if (!vm) return;

    const int N = 200;
    vm->Initialize(N, 1, N);

    std::atomic<bool> stop{false};
    std::atomic<int> readCount{0};

    // Writer thread: increment versions
    std::thread writer([&]() {
        for (int round = 0; round < 5; round++) {
            for (int i = 0; i < N; i++) {
                uint8_t v;
                vm->IncVersion(i, &v);
            }
        }
        stop = true;
    });

    // Reader threads: continuously batch-read versions
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; r++) {
        readers.emplace_back([&]() {
            while (!stop.load()) {
                std::vector<SizeType> vids;
                for (int i = 0; i < 50; i++) {
                    vids.push_back(rand() % N);
                }
                std::vector<uint8_t> versions;
                vm->BatchGetVersions(vids, versions);
                readCount++;
                // Versions should be valid (0-0x7f) since no deletes
                for (auto v : versions) {
                    BOOST_CHECK(v != 0xfe);
                }
            }
        });
    }

    writer.join();
    for (auto& th : readers) th.join();

    BOOST_TEST_MESSAGE("Concurrent read/write: " << readCount.load() << " batch reads completed");

    // Final versions should all be 5 (5 increments)
    for (int i = 0; i < N; i++) {
        BOOST_CHECK_EQUAL(vm->GetVersion(i), 5);
    }
}

// ==============================================================
// Edge Case Tests
// ==============================================================

BOOST_AUTO_TEST_CASE(TiKV_SingleElement)
{
    auto vm = MakeTiKVVersionMap("Single", 64);
    if (!vm) return;

    vm->Initialize(1, 1, 1);
    BOOST_CHECK_EQUAL(vm->Count(), 1);
    BOOST_CHECK(!vm->Deleted(0));

    uint8_t newVer;
    BOOST_CHECK(vm->IncVersion(0, &newVer));
    BOOST_CHECK_EQUAL(newVer, 1);

    BOOST_CHECK(vm->Delete(0));
    BOOST_CHECK(vm->Deleted(0));
}

BOOST_AUTO_TEST_CASE(TiKV_ExactChunkSize)
{
    // Size is exact multiple of chunkSize
    auto vm = MakeTiKVVersionMap("ExactChunk", 64);
    if (!vm) return;

    vm->Initialize(128, 1, 128); // exactly 2 chunks

    // Last VID in chunk 0
    BOOST_CHECK(!vm->Deleted(63));
    vm->SetVersion(63, 50);
    BOOST_CHECK_EQUAL(vm->GetVersion(63), 50);

    // First VID in chunk 1
    vm->SetVersion(64, 60);
    BOOST_CHECK_EQUAL(vm->GetVersion(64), 60);

    // Last VID overall
    vm->SetVersion(127, 70);
    BOOST_CHECK_EQUAL(vm->GetVersion(127), 70);
}

BOOST_AUTO_TEST_CASE(TiKV_PartialLastChunk)
{
    // Size not a multiple of chunkSize (100 VIDs, chunkSize=64)
    // chunk 0: VIDs 0-63, chunk 1: VIDs 64-99 (only 36 used)
    auto vm = MakeTiKVVersionMap("PartialChunk", 64);
    if (!vm) return;

    vm->Initialize(100, 1, 100);

    // Verify VIDs near the end of partial chunk
    BOOST_CHECK(!vm->Deleted(99));
    vm->SetVersion(99, 33);
    BOOST_CHECK_EQUAL(vm->GetVersion(99), 33);
}

BOOST_AUTO_TEST_CASE(TiKV_InitializeWithGlobalIDs)
{
    auto vm = MakeTiKVVersionMap("InitGlobal", 64);
    if (!vm) return;

    // Simulate non-leaf layer: only certain globalIDs are alive
    const int size = 200;
    std::vector<SizeType> globalIDValues = {5, 10, 50, 100, 150, 199};
    COMMON::Dataset<SizeType> globalIDs;
    globalIDs.Initialize(static_cast<SizeType>(globalIDValues.size()), 1, static_cast<SizeType>(globalIDValues.size()), static_cast<SizeType>(globalIDValues.size()));
    for (size_t i = 0; i < globalIDValues.size(); i++) {
        *(globalIDs.At(static_cast<SizeType>(i))) = globalIDValues[i];
    }

    vm->Initialize(size, 1, size, &globalIDs);

    // GlobalIDs should be alive (version 0), all others deleted
    for (int i = 0; i < size; i++) {
        bool shouldBeAlive = (std::find(globalIDValues.begin(), globalIDValues.end(), i) != globalIDValues.end());
        if (shouldBeAlive) {
            BOOST_CHECK_MESSAGE(!vm->Deleted(i), "VID " << i << " should be alive but is deleted");
        } else {
            BOOST_CHECK_MESSAGE(vm->Deleted(i), "VID " << i << " should be deleted but is alive");
        }
    }

    BOOST_CHECK_EQUAL(vm->GetDeleteCount(), size - (int)globalIDValues.size());
}

BOOST_AUTO_TEST_CASE(TiKV_SaveLoadCount)
{
    // Verify that count persists across Load
    auto vm1 = MakeTiKVVersionMap("SaveLoad", 64);
    if (!vm1) return;

    vm1->Initialize(300, 1, 300);
    vm1->Delete(50);
    vm1->Delete(100);

    // Save (no-op for TiKV, but flushes count)
    vm1->Save(std::string(""));

    // Create a new TiKVVersionMap pointing to the same TiKV keys
    // We reuse the same DB and layer to simulate reload
    auto vm2 = std::make_unique<TiKVVersionMap>();
    vm2->SetDB(vm1->GetDB());    // Reuse same DB
    vm2->SetLayer(0);
    vm2->SetChunkSize(64);
    vm2->Load(std::string(""), 1, 300);

    BOOST_CHECK_EQUAL(vm2->Count(), 300);
    BOOST_CHECK(vm2->Deleted(50));
    BOOST_CHECK(vm2->Deleted(100));
    BOOST_CHECK(!vm2->Deleted(0));
}

#endif // TIKV

BOOST_AUTO_TEST_SUITE_END()
