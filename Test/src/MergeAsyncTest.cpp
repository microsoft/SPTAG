// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Test.h"

#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <future>

namespace
{
struct TemporaryIndexDirectory
{
    boost::filesystem::path path = boost::filesystem::temp_directory_path() /
        boost::filesystem::unique_path("sptag-merge-%%%%-%%%%");

    TemporaryIndexDirectory() { boost::filesystem::create_directories(path); }
    ~TemporaryIndexDirectory() { boost::filesystem::remove_all(path); }
};
}

BOOST_AUTO_TEST_SUITE(MergeAsyncTest)

BOOST_AUTO_TEST_CASE(ReadOnlySearcherDoesNotQueueMerges)
{
    TemporaryIndexDirectory directory;
    SPTAG::SPANN::Options options;
    options.m_dim = 1;
    options.m_storage = SPTAG::Storage::FILEIO;
    options.m_indexDirectory = directory.path.string();
    options.m_update = false;
    options.m_asyncMergeInSearch = true;
    options.m_startFileSize = 0;
    options.m_iSSDNumberOfThreads = 1;
    options.m_searchThreadNum = 1;
    options.m_ioThreads = 1;
    options.m_datasetRowsInBlock = 64;
    options.m_datasetCapacity = 64;

    // Scheduling alone needs no posting data or preallocated storage.
    std::ofstream postings((directory.path / (options.m_ssdMappingFile + "_postings")).string());
    BOOST_REQUIRE(postings.good());
    postings.close();

    SPTAG::COMMON::VersionLabel versions;
    versions.Initialize(0, 64, 64);
    BOOST_REQUIRE(versions.Save((directory.path / options.m_deleteIDFile).string()) == SPTAG::ErrorCode::Success);
    SPTAG::COMMON::PostingSizeRecord postingSizes;
    postingSizes.Initialize(0, 64, 64);
    BOOST_REQUIRE(postingSizes.Save((directory.path / options.m_ssdInfoFile).string()) == SPTAG::ErrorCode::Success);
    SPTAG::COMMON::Dataset<SPTAG::ChecksumType> checksums;
    checksums.Initialize(0, 1, 64, 64);
    BOOST_REQUIRE(checksums.Save((directory.path / options.m_checksumFile).string()) == SPTAG::ErrorCode::Success);
    SPTAG::COMMON::Dataset<std::uint64_t> translations;
    auto headIndex = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    BOOST_REQUIRE(headIndex != nullptr);

    SPTAG::SPANN::ExtraDynamicSearcher<float> searcher(options);
    BOOST_REQUIRE(searcher.Available());
    BOOST_REQUIRE(searcher.LoadIndex(options, versions, translations, headIndex));
    bool callbackCalled = false;
    // Read-only search may request a merge even though LoadIndex never creates
    // the update pool. No job should be constructed or run in that state.
    for (SPTAG::SizeType headID : {0, 0, 1})
        searcher.MergeAsync(headIndex.get(), headID, [&] { callbackCalled = true; });
    BOOST_CHECK(!callbackCalled);

    // Initializing the update pool must allow the same head to be scheduled.
    // A guard placed after merge-list insertion would leave head 0 queued and
    // prevent this callback. The empty head index makes the job a harmless no-op.
    options.m_update = true;
    BOOST_REQUIRE(searcher.LoadIndex(options, versions, translations, headIndex));
    auto completed = std::make_shared<std::promise<void>>();
    auto completion = completed->get_future();
    searcher.MergeAsync(headIndex.get(), 0, [completed] { completed->set_value(); });
    BOOST_REQUIRE(completion.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    completion.get();
}

BOOST_AUTO_TEST_SUITE_END()
