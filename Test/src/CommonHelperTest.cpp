// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/AsyncFileReader.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Test.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

BOOST_AUTO_TEST_SUITE(CommonHelperTest)

BOOST_AUTO_TEST_CASE(ToLowerInPlaceTest)
{
    auto runTestCase = [](std::string p_input, const std::string &p_expected) {
        SPTAG::Helper::StrUtils::ToLowerInPlace(p_input);
        BOOST_CHECK(p_input == p_expected);
    };

    runTestCase("abc", "abc");
    runTestCase("ABC", "abc");
    runTestCase("abC", "abc");
    runTestCase("Upper-Case", "upper-case");
    runTestCase("123!-=aBc", "123!-=abc");
}

BOOST_AUTO_TEST_CASE(SplitStringTest)
{
    std::string input("seg1 seg2 seg3  seg4");

    const auto &segs = SPTAG::Helper::StrUtils::SplitString(input, " ");
    BOOST_CHECK(segs.size() == 4);
    BOOST_CHECK(segs[0] == "seg1");
    BOOST_CHECK(segs[1] == "seg2");
    BOOST_CHECK(segs[2] == "seg3");
    BOOST_CHECK(segs[3] == "seg4");
}

BOOST_AUTO_TEST_CASE(FindTrimmedSegmentTest)
{
    using namespace SPTAG::Helper::StrUtils;
    std::string input("\t Space   End    \r\n\t");

    const auto &pos = FindTrimmedSegment(input.c_str(), input.c_str() + input.size(),
                                         [](char p_val) -> bool { return std::isspace(p_val) > 0; });

    BOOST_CHECK(pos.first == input.c_str() + 2);
    BOOST_CHECK(pos.second == input.c_str() + 13);
}

BOOST_AUTO_TEST_CASE(StartsWithTest)
{
    using namespace SPTAG::Helper::StrUtils;

    BOOST_CHECK(StartsWith("Abcd", "A"));
    BOOST_CHECK(StartsWith("Abcd", "Ab"));
    BOOST_CHECK(StartsWith("Abcd", "Abc"));
    BOOST_CHECK(StartsWith("Abcd", "Abcd"));

    BOOST_CHECK(!StartsWith("Abcd", "a"));
    BOOST_CHECK(!StartsWith("Abcd", "F"));
    BOOST_CHECK(!StartsWith("Abcd", "AF"));
    BOOST_CHECK(!StartsWith("Abcd", "AbF"));
    BOOST_CHECK(!StartsWith("Abcd", "AbcF"));
    BOOST_CHECK(!StartsWith("Abcd", "Abcde"));
}

BOOST_AUTO_TEST_CASE(StrEqualIgnoreCaseTest)
{
    using namespace SPTAG::Helper::StrUtils;

    BOOST_CHECK(StrEqualIgnoreCase("Abcd", "Abcd"));
    BOOST_CHECK(StrEqualIgnoreCase("Abcd", "abcd"));
    BOOST_CHECK(StrEqualIgnoreCase("Abcd", "abCD"));
    BOOST_CHECK(StrEqualIgnoreCase("Abcd-123", "abcd-123"));
    BOOST_CHECK(StrEqualIgnoreCase(" ZZZ", " zzz"));

    BOOST_CHECK(!StrEqualIgnoreCase("abcd", "abcd1"));
    BOOST_CHECK(!StrEqualIgnoreCase("Abcd", " abcd"));
    BOOST_CHECK(!StrEqualIgnoreCase("000", "OOO"));
}

#ifndef _MSC_VER
BOOST_AUTO_TEST_CASE(LinuxAsyncBatchReadReportsShortCompletion)
{
    const std::string path = "async_batch_short_completion_test.bin";
    struct FileCleanup
    {
        const std::string& path;
        ~FileCleanup() { std::remove(path.c_str()); }
    } cleanup{path};

    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        BOOST_REQUIRE(output.good());
        const std::vector<char> contents(SPTAG::PageSize, 'x');
        output.write(
            contents.data(),
            static_cast<std::streamsize>(contents.size()));
        BOOST_REQUIRE(output.good());
    }

    auto handler = std::make_shared<SPTAG::Helper::AsyncFileIO>();
    BOOST_REQUIRE(handler->Initialize(
        path.c_str(), O_RDONLY | O_DIRECT, 1, 2, 2, 1));
    std::vector<std::shared_ptr<SPTAG::Helper::DiskIO>> handlers{
        handler};

    SPTAG::Helper::PageBuffer<char> buffer;
    buffer.ReservePageBuffer(2 * SPTAG::PageSize);
    SPTAG::Helper::AsyncReadRequest request;
    request.m_buffer = buffer.GetBuffer();
    request.m_offset = 0;
    request.m_readSize = 2 * SPTAG::PageSize;
    request.m_status = 0;
    int callbackCount = 0;
    bool callbackSuccess = true;
    request.m_callback =
        [&](bool success)
        {
            ++callbackCount;
            callbackSuccess = success;
        };

    BOOST_REQUIRE(
        SPTAG::Helper::BatchReadFileAsync(
            handlers, &request, 1));
    BOOST_CHECK_EQUAL(callbackCount, 1);
    BOOST_CHECK(!callbackSuccess);
    handler->ShutDown();
}
#endif

BOOST_AUTO_TEST_SUITE_END()