// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/LocalVersionMap.h"
#include "inc/Test.h"

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <set>
#include <thread>

namespace
{
    using SPTAG::COMMON::LocalVersionMap;
    using SPTAG::ErrorCode;
    using SPTAG::SizeType;

    struct Workers {
        std::atomic<bool> start{false};
        std::atomic<bool> stop{false};
        std::vector<std::thread> threads;

        void Launch(std::function<void()> work) {
            threads.emplace_back([this, work]() {
                while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
                if (!stop.load(std::memory_order_acquire)) work();
            });
        }
        void Start() { start.store(true, std::memory_order_release); }
        void Join() {
            for (auto& thread : threads)
                if (thread.joinable()) thread.join();
        }
        ~Workers() {
            stop.store(true, std::memory_order_release);
            Start();
            Join();
        }
    };

    std::shared_ptr<SPTAG::Helper::SimpleBufferIO> BufferIO(std::vector<char>& bytes)
    {
        auto io = std::make_shared<SPTAG::Helper::SimpleBufferIO>();
        BOOST_REQUIRE(io->Initialize(bytes.data(), std::ios::in | std::ios::out, bytes.size()));
        return io;
    }

    std::vector<char> Snapshot(LocalVersionMap& map, SizeType maxRows)
    {
        std::vector<char> bytes(sizeof(SizeType) + maxRows * (sizeof(SizeType) + sizeof(uint8_t)));
        auto io = BufferIO(bytes);
        BOOST_REQUIRE(map.Save(io) == ErrorCode::Success);
        bytes.resize(io->TellP());
        SizeType count;
        memcpy(&count, bytes.data(), sizeof(count));
        BOOST_REQUIRE_GE(count, 0);
        BOOST_REQUIRE_LE(count, maxRows);
        BOOST_CHECK_EQUAL(bytes.size(), sizeof(SizeType) + count * (sizeof(SizeType) + sizeof(uint8_t)));
        return bytes;
    }
}

BOOST_AUTO_TEST_SUITE(LocalVersionMapTest)

BOOST_AUTO_TEST_CASE(SparseKeysVersionsAndLegacySerialization)
{
    LocalVersionMap map;
    BOOST_CHECK_EQUAL(map.Count(), 0);
    BOOST_CHECK(map.Deleted(700000000));
    BOOST_CHECK_EQUAL(map.GetVersion(700000000), 0xfe);
    uint8_t next = 99;
    BOOST_CHECK(!map.IncVersion(700000000, &next));
    BOOST_CHECK_EQUAL(next, 99);
    BOOST_CHECK(!map.Delete(700000000));

    map.SetVersion(0, 0xff);
    map.SetVersion(700000000, 127);
    BOOST_CHECK(map.IncVersion(0, &next));
    BOOST_CHECK_EQUAL(next, 0);
    BOOST_CHECK(map.IncVersion(700000000, &next));
    BOOST_CHECK_EQUAL(next, 0);
    map.SetVersion(700000000, 17);
    BOOST_CHECK_EQUAL(map.Count(), 2);
    BOOST_CHECK_EQUAL(map.BufferSize(), 2 * (sizeof(SizeType) + sizeof(uint8_t)));
    std::vector<SizeType> ids;
    BOOST_REQUIRE(map.GetContainedIDs(ids) == ErrorCode::Success);
    BOOST_CHECK((std::set<SizeType>(ids.begin(), ids.end()) == std::set<SizeType>{0, 700000000}));

    auto bytes = Snapshot(map, 2);
    std::set<SizeType> serializedIDs;
    for (std::size_t offset = sizeof(SizeType); offset < bytes.size(); offset += sizeof(SizeType) + 1) {
        SizeType id;
        memcpy(&id, bytes.data() + offset, sizeof(id));
        BOOST_CHECK(serializedIDs.insert(id).second);
        BOOST_CHECK_EQUAL(static_cast<uint8_t>(bytes[offset + sizeof(id)]), map.GetVersion(id));
    }
    LocalVersionMap loaded;
    loaded.SetVersion(42, 3);
    BOOST_REQUIRE(loaded.Load(BufferIO(bytes), 1024, 1024) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(loaded.Count(), 3);
    BOOST_CHECK_EQUAL(loaded.GetVersion(42), 3);
    BOOST_CHECK_EQUAL(loaded.GetVersion(700000000), 17);
    BOOST_CHECK(loaded.Delete(700000000));
    BOOST_CHECK(!loaded.Delete(700000000));
    BOOST_CHECK_EQUAL(loaded.GetVersion(700000000), 0xfe);
    loaded.DeleteAll();
    BOOST_CHECK_EQUAL(loaded.Count(), 0);
    loaded.SetVersion(700000000, 5);
    BOOST_CHECK_EQUAL(loaded.GetVersion(700000000), 5);
}

BOOST_AUTO_TEST_CASE(ConcurrentIncrementsDoNotLoseUpdates)
{
    LocalVersionMap map;
    map.SetVersion(11, 0);
    constexpr int threadCount = 8;
    constexpr int increments = 10003;
    std::array<std::atomic<int>, 128> observed;
    for (auto& count : observed) count.store(0);
    std::array<int, 128> expected{};
    for (int i = 1; i <= threadCount * increments; ++i) ++expected[i & 0x7f];
    std::atomic<int> failures{0};
    Workers workers;
    for (int i = 0; i < threadCount; ++i) {
        workers.Launch([&]() {
            for (int j = 0; j < increments && !workers.stop.load(); ++j) {
                uint8_t version = 0xfe;
                if (!map.IncVersion(11, &version) || version > 127) ++failures;
                else observed[version].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    workers.Start();
    workers.Join();
    BOOST_CHECK_EQUAL(failures.load(), 0);
    for (int i = 0; i < 128; ++i) BOOST_CHECK_EQUAL(observed[i].load(), expected[i]);
    BOOST_CHECK_EQUAL(map.GetVersion(11), (threadCount * increments) & 0x7f);
    BOOST_CHECK_EQUAL(map.Count(), 1);
}

BOOST_AUTO_TEST_CASE(ConcurrentLookupUpdateAndErase)
{
    LocalVersionMap map;
    constexpr int keys = 128;
    std::atomic<int> failures{0};
    Workers workers;
    for (int writer = 0; writer < 4; ++writer) {
        workers.Launch([&, writer]() {
            for (int i = 0; i < 20000 && !workers.stop.load(); ++i) {
                const SizeType id = (i + writer) % keys;
                map.SetVersion(id, 0);
                uint8_t next;
                map.IncVersion(id, &next);
                map.Delete(id);
            }
        });
    }
    for (int reader = 0; reader < 8; ++reader) {
        workers.Launch([&, reader]() {
            for (int i = 0; i < 30000 && !workers.stop.load(); ++i) {
                const SizeType id = (i + reader) % keys;
                const auto value = map.GetVersion(id);
                if (value > 127 && value != 0xfe) ++failures;
                map.Deleted(id);
                if (map.Count() > keys) ++failures;
                if (map.BufferSize() > keys * (sizeof(SizeType) + sizeof(uint8_t))) ++failures;
            }
        });
    }
    workers.Start();
    workers.Join();
    BOOST_CHECK_EQUAL(failures.load(), 0);
    for (SizeType id = 0; id < keys; ++id) map.SetVersion(id, 9);
    BOOST_CHECK_EQUAL(map.Count(), keys);
    for (SizeType id = 0; id < keys; ++id) BOOST_CHECK_EQUAL(map.GetVersion(id), 9);
}

BOOST_AUTO_TEST_CASE(MaintenanceQuiescesReadersAndWriters)
{
    LocalVersionMap map;
    constexpr int keys = 64;
    std::atomic<int> failures{0};
    std::atomic<int> readyReaders{0}, readyWriters{0};
    for (SizeType id = 0; id < keys; ++id) map.SetVersion(id, static_cast<uint8_t>(id));
    Workers workers;
    for (int writer = 0; writer < 2; ++writer) {
        workers.Launch([&, writer]() {
            for (int i = 0; !workers.stop.load(); ++i) {
                const SizeType id = (i + writer) % keys;
                if (i % 3 == 0) map.Delete(id);
                else map.SetVersion(id, static_cast<uint8_t>(id));
                if (i == 0) ++readyWriters;
            }
        });
    }
    // More readers than operation slots exercise shared-slot admission during maintenance.
    for (int reader = 0; reader < 80; ++reader) {
        workers.Launch([&, reader]() {
            for (int i = 0; !workers.stop.load(); ++i) {
                const SizeType id = (i + reader) % keys;
                const auto value = map.GetVersion(id);
                if (value != id && value != 0xfe) ++failures;
                map.Deleted(id);
                if (i == 0) ++readyReaders;
            }
        });
    }
    workers.Start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((readyReaders.load() != 80 || readyWriters.load() != 2)
        && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    BOOST_REQUIRE_EQUAL(readyReaders.load(), 80);
    BOOST_REQUIRE_EQUAL(readyWriters.load(), 2);
    for (int round = 0; round < 12; ++round) {
        auto bytes = Snapshot(map, keys);
        LocalVersionMap restored;
        BOOST_REQUIRE(restored.Load(BufferIO(bytes), 1024, 1024) == ErrorCode::Success);
        std::vector<SizeType> ids;
        BOOST_REQUIRE(restored.GetContainedIDs(ids) == ErrorCode::Success);
        BOOST_CHECK_EQUAL(ids.size(), restored.Count());
        BOOST_CHECK_EQUAL(std::set<SizeType>(ids.begin(), ids.end()).size(), ids.size());
        for (auto id : ids) BOOST_CHECK_EQUAL(restored.GetVersion(id), id);
        BOOST_REQUIRE(map.Load(BufferIO(bytes), 1024, 1024) == ErrorCode::Success);
        BOOST_REQUIRE(map.GetContainedIDs(ids) == ErrorCode::Success);
        BOOST_CHECK_EQUAL(std::set<SizeType>(ids.begin(), ids.end()).size(), ids.size());
        for (auto id : ids) BOOST_CHECK(id >= 0 && id < keys);
        map.DeleteAll();
    }
    workers.stop.store(true);
    workers.Join();
    BOOST_CHECK_EQUAL(failures.load(), 0);
    map.DeleteAll();
    BOOST_CHECK_EQUAL(map.Count(), 0);
    map.SetVersion(7, 13);
    BOOST_CHECK_EQUAL(map.GetVersion(7), 13);
}

BOOST_AUTO_TEST_CASE(FailedSerializationReopensAdmission)
{
    LocalVersionMap map;
    map.SetVersion(17, 5);
    std::vector<char> shortBuffer(1);
    BOOST_CHECK(map.Save(BufferIO(shortBuffer)) == ErrorCode::DiskIOFail);
    BOOST_CHECK_EQUAL(map.GetVersion(17), 5);
    BOOST_CHECK(map.Load(BufferIO(shortBuffer), 1024, 1024) == ErrorCode::DiskIOFail);
    map.SetVersion(18, 6);
    BOOST_CHECK_EQUAL(map.Count(), 2);
    BOOST_CHECK(map.Delete(17));
    auto bytes = Snapshot(map, 2);
    LocalVersionMap restored;
    BOOST_REQUIRE(restored.Load(BufferIO(bytes), 1024, 1024) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(restored.GetVersion(18), 6);
}

BOOST_AUTO_TEST_SUITE_END()
