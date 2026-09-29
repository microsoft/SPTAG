// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/ExtraFileController.h"
#include "inc/Core/SPANN/IExtraSearcher.h"
#include "inc/Test.h"
#include <chrono>
#include <memory>

// enable rocksdb io_uring

#ifdef ROCKSDB
#include "inc/Core/SPANN/ExtraRocksDBController.h"
// extern "C" bool RocksDbIOUringEnable() { return true; }
#endif

#ifdef SPDK
#include "inc/Core/SPANN/ExtraSPDKController.h"
#endif

using namespace SPTAG;
using namespace SPTAG::SPANN;

void Search(std::shared_ptr<Helper::KeyValueIO> db, int internalResultNum, int totalSize, int times, bool debug,
            SPTAG::SPANN::ExtraWorkSpace &workspace)
{
    std::vector<SizeType> headIDs(internalResultNum, 0);

    std::vector<std::string> values;
    double latency = 0;
    for (int i = 0; i < times; i++)
    {
        values.clear();
        for (int j = 0; j < internalResultNum; j++)
            headIDs[j] = (j + i * internalResultNum) % totalSize;
        auto t1 = std::chrono::high_resolution_clock::now();
        db->MultiGet(headIDs, &values, MaxTimeout, &(workspace.m_diskRequests));
        auto t2 = std::chrono::high_resolution_clock::now();
        latency += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        if (debug)
        {
            for (int j = 0; j < internalResultNum; j++)
            {
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
    SPTAG::SPANN::ExtraWorkSpace workspace;
    SPTAG::SPANN::Options opt;
    auto idx = path.find_last_of(FolderSep);
    opt.m_indexDirectory = path.substr(0, idx);
    opt.m_ssdMappingFile = path.substr(idx + 1);
    workspace.Initialize(4096, 2, internalResultNum, 4 * PageSize, true, false);

    if (type == "RocksDB")
    {
#ifdef ROCKSDB
        db.reset(new RocksDBIO(path.c_str(), true));
#else
        {
            std::cerr << "RocksDB is not supported in this build." << std::endl;
            return;
        }
#endif
    }
    else if (type == "SPDK")
    {
#ifdef SPDK
        db.reset(new SPDKIO(opt));
#else
        {
            std::cerr << "SPDK is not supported in this build." << std::endl;
            return;
        }
#endif
    }
    else
    {
        db.reset(new FileIO(opt));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < totalNum; i++)
    {
        int len = std::to_string(i).length();
        std::string val(PageSize - len, '0');
        db->Put(i, val, MaxTimeout, &(workspace.m_diskRequests));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "avg put time: "
              << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(totalNum)) << "us"
              << std::endl;

    db->ForceCompaction();

    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < totalNum; i++)
    {
        for (int j = 0; j < mergeIters; j++)
        {
            db->Merge(i, std::to_string(i), MaxTimeout, &(workspace.m_diskRequests),
                      [](const void* val, const int size) -> bool { return true; });
        }
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "avg merge time: "
              << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
                  (float)(totalNum * mergeIters))
              << "us" << std::endl;

    Search(db, internalResultNum, totalNum, 10, debug, workspace);

    db->ForceCompaction();
    db->ShutDown();

    // if (type == "RocksDB") {
    //     db.reset(new RocksDBIO(path.c_str(), true));
    //     Search(db, internalResultNum, totalNum, 10, debug);
    //     db->ForceCompaction();
    //     db->ShutDown();
    // }
}

BOOST_AUTO_TEST_SUITE(KVTest)

BOOST_AUTO_TEST_CASE(RocksDBTest)
{
    if (!direxists("tmp_rocksdb"))
        mkdir("tmp_rocksdb");
    Test(std::string("tmp_rocksdb") + FolderSep + "test", "RocksDB", false);
}

BOOST_AUTO_TEST_CASE(SPDKTest)
{
    if (!direxists("tmp_spdk"))
        mkdir("tmp_spdk");
    Test(std::string("tmp_spdk") + FolderSep + "test", "SPDK", false);
}

BOOST_AUTO_TEST_CASE(FileTest)
{
    if (!direxists("tmp_file"))
        mkdir("tmp_file");
    Test(std::string("tmp_file") + FolderSep + "test", "File", false);
}

BOOST_AUTO_TEST_SUITE_END()
