// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/ConcurrentSet.h"
#include "inc/Test.h"

#include <atomic>
#include <thread>
#include <vector>

#ifndef FOLLY
BOOST_AUTO_TEST_SUITE(ConcurrentHashMapTest)

BOOST_AUTO_TEST_CASE(InsertAssignEraseAndConditionalUpdates)
{
    SPTAG::Helper::Concurrent::ConcurrentHashMap<int, int> map;
    BOOST_CHECK(map.begin() == map.end());
    BOOST_CHECK(map.find(1) == map.end());
    BOOST_CHECK(map.insert(std::make_pair(1, 10)).second);
    BOOST_CHECK(!map.insert(std::make_pair(1, 20)).second);
    BOOST_CHECK_EQUAL(map.find(1)->second, 10);
    BOOST_CHECK(!map.insert_or_assign(1, 20).second);
    BOOST_CHECK(map.insert_or_assign(2, 30).second);
    BOOST_CHECK(!map.assign_if_equal(1, 10, 40));
    BOOST_CHECK_EQUAL(map.find(1)->second, 20);
    BOOST_CHECK(map.assign_if_equal(1, 20, 40));
    BOOST_CHECK(!map.assign_if_equal(3, 0, 1));
    BOOST_CHECK(map.assign_if(1, 50, [](int value) { return value == 40; }));
    BOOST_CHECK(!map.assign_if(1, 60, [](int value) { return value < 40; }));
    BOOST_CHECK_EQUAL(map.find(1)->second, 50);
    BOOST_CHECK_EQUAL(map.size(), 2);
    BOOST_CHECK_EQUAL(map.erase(1), 1);
    BOOST_CHECK_EQUAL(map.erase(1), 0);
    map.clear();
    BOOST_CHECK_EQUAL(map.size(), 0);
}

BOOST_AUTO_TEST_CASE(SnapshotsSurviveEraseAndClear)
{
    SPTAG::Helper::Concurrent::ConcurrentHashMap<int, int> map;
    map.insert_or_assign(1, 10);
    map.insert_or_assign(2, 20);
    auto found = map.find(1);
    auto snapshot = map.begin();
    auto copied = snapshot;
    BOOST_CHECK(snapshot == copied);
    map.erase(1);
    map.clear();
    BOOST_CHECK_EQUAL(found->second, 10);
    int sum = 0;
    for (; snapshot != map.end(); snapshot++) sum += snapshot->second;
    BOOST_CHECK_EQUAL(sum, 30);
    BOOST_CHECK(snapshot == map.end());
}

BOOST_AUTO_TEST_CASE(ConcurrentCompareAndSwapAndSnapshotReads)
{
    SPTAG::Helper::Concurrent::ConcurrentHashMap<int, int> map;
    map.insert_or_assign(0, 0);
    constexpr int workers = 4;
    constexpr int updates = 500;
    std::atomic<bool> invalidRead{false};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&]() {
            for (int update = 0; update < updates; ++update) {
                while (true) {
                    auto current = map.find(0);
                    if (current == map.end() || current->second < 0) {
                        invalidRead.store(true);
                        return;
                    }
                    if (map.assign_if_equal(0, current->second, current->second + 1)) break;
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
    BOOST_CHECK(!invalidRead.load());
    BOOST_CHECK_EQUAL(map.find(0)->second, workers * updates);
}

BOOST_AUTO_TEST_SUITE_END()
#endif
