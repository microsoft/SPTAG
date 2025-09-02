// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/CommonHelper.h"

#include <memory>

namespace CommonHelperTest {


TEST(CommonHelperTest, ToLowerInPlaceTest) {
    auto runTestCase = [](std::string p_input, const std::string& p_expected)
    {
        SPTAG::Helper::StrUtils::ToLowerInPlace(p_input);
    EXPECT_EQ(p_input, p_expected);
    };

    runTestCase("abc", "abc");
    runTestCase("ABC", "abc");
    runTestCase("abC", "abc");
    runTestCase("Upper-Case", "upper-case");
    runTestCase("123!-=aBc", "123!-=abc");
}


TEST(CommonHelperTest, SplitStringTest) {
    std::string input("seg1 seg2 seg3  seg4");

    const auto& segs = SPTAG::Helper::StrUtils::SplitString(input, " ");
    EXPECT_EQ(segs.size(), 4u);
    EXPECT_EQ(segs[0], "seg1");
    EXPECT_EQ(segs[1], "seg2");
    EXPECT_EQ(segs[2], "seg3");
    EXPECT_EQ(segs[3], "seg4");
}


TEST(CommonHelperTest, FindTrimmedSegmentTest) {
    using namespace SPTAG::Helper::StrUtils;
    std::string input("\t Space   End    \r\n\t");

    const auto& pos = FindTrimmedSegment(input.c_str(),
        input.c_str() + input.size(),
        [](char p_val)->bool
    {
        return std::isspace(p_val) > 0;
    });

    EXPECT_EQ(pos.first, input.c_str() + 2);
    EXPECT_EQ(pos.second, input.c_str() + 13);
}


TEST(CommonHelperTest, StartsWithTest) {
    using namespace SPTAG::Helper::StrUtils;

    EXPECT_TRUE(StartsWith("Abcd", "A"));
    EXPECT_TRUE(StartsWith("Abcd", "Ab"));
    EXPECT_TRUE(StartsWith("Abcd", "Abc"));
    EXPECT_TRUE(StartsWith("Abcd", "Abcd"));

    EXPECT_FALSE(StartsWith("Abcd", "a"));
    EXPECT_FALSE(StartsWith("Abcd", "F"));
    EXPECT_FALSE(StartsWith("Abcd", "AF"));
    EXPECT_FALSE(StartsWith("Abcd", "AbF"));
    EXPECT_FALSE(StartsWith("Abcd", "AbcF"));
    EXPECT_FALSE(StartsWith("Abcd", "Abcde"));
}


TEST(CommonHelperTest, StrEqualIgnoreCaseTest) {
    using namespace SPTAG::Helper::StrUtils;

    EXPECT_TRUE(StrEqualIgnoreCase("Abcd", "Abcd"));
    EXPECT_TRUE(StrEqualIgnoreCase("Abcd", "abcd"));
    EXPECT_TRUE(StrEqualIgnoreCase("Abcd", "abCD"));
    EXPECT_TRUE(StrEqualIgnoreCase("Abcd-123", "abcd-123"));
    EXPECT_TRUE(StrEqualIgnoreCase(" ZZZ", " zzz"));

    EXPECT_FALSE(StrEqualIgnoreCase("abcd", "abcd1"));
    EXPECT_FALSE(StrEqualIgnoreCase("Abcd", " abcd"));
    EXPECT_FALSE(StrEqualIgnoreCase("000", "OOO"));
}

}