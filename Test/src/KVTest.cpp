// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Core/SPANN/ExtraRocksDBController.h"
#include "inc/Core/SPANN/ExtraSPDKController.h"

#include <memory>
#include <chrono>

// enable rocksdb io_uring
extern "C" bool RocksDbIOUringEnable() { return true; }

using namespace SPTAG;
using namespace SPTAG::SPANN;

void Search(std::shared_ptr<Helper::KeyValueIO> db, int internalResultNum, int totalSize, int times, bool debug = false) { 
    std::vector<SizeType> headIDs(internalResultNum, 0);

    std::vector<std::string> values;
    double latency = 0;
    for (int i = 0; i < times; i++) {
        values.clear();
        for (int j = 0; j < internalResultNum; j++) headIDs[j] = (j + i * internalResultNum) % totalSize;
        auto t1 = std::chrono::high_resolution_clock::now();
        db->MultiGet(headIDs, &values);
        auto t2 = std::chrono::high_resolution_clock::now();
        latency += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        if (debug) {
            for (int j = 0; j < internalResultNum; j++) {
                std::cout << values[j].substr(PageSize) << std::endl;
            }
        }
    }
    std::cout << "avg get time: " << (latency / (float)(times)) << "us" << std::endl;

}

void Test(std::string path, std::string type, bool debug = false)
{
    int internalResultNum = 64;
    int totalNum = 1024;
    int mergeIters = 3;
    std::shared_ptr<Helper::KeyValueIO> db;
    if (type == "RocksDB") {
        db.reset(new RocksDBIO(path.c_str(), true));
    } else if (type == "SPDK") {
        db.reset(new SPDKIO(path.c_str(), 1024 * 1024, MaxSize, 64));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < totalNum; i++) {
        int len = std::to_string(i).length();
        std::string val(PageSize - len, '0');
        db->Put(i, val);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "avg put time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(totalNum)) << "us" << std::endl;

    db->ForceCompaction();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < totalNum; i++) {
        for (int j = 0; j < mergeIters; j++) {
            db->Merge(i, std::to_string(i));
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "avg merge time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(totalNum * mergeIters)) << "us" << std::endl;

    Search(db, internalResultNum, totalNum, 10, debug);

    db->ForceCompaction();
    db->ShutDown();

    if (type == "RocksDB") {
        db.reset(new RocksDBIO(path.c_str(), true));
        Search(db, internalResultNum, totalNum, 10, debug);
        db->ForceCompaction();
        db->ShutDown();
    }
}

BOOST_AUTO_TEST_SUITE(KVTest)

BOOST_AUTO_TEST_CASE(RocksDBTest)
{
    Test("tmp_rocksdb", "RocksDB", true);
}

BOOST_AUTO_TEST_CASE(SPDKTest)
{
    Test("tmp_spdk", "SPDK", true);
}

BOOST_AUTO_TEST_SUITE_END()
