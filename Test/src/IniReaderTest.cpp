// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/SimpleIniReader.h"
#include "inc/Test.h"

#include <fstream>

namespace IniReaderTest
{

TEST(IniReaderTest, IniReaderLoadTest)
{
    std::ofstream tmpIni("temp.ini");
    tmpIni << "[Common]" << std::endl;
    tmpIni << "; Comment " << std::endl;
    tmpIni << "Param1=1" << std::endl;
    tmpIni << "Param2=Exp=2" << std::endl;

    tmpIni.close();

    SPTAG::Helper::IniReader reader;
    ASSERT_EQ(SPTAG::ErrorCode::Success, reader.LoadIniFile("temp.ini"));

    EXPECT_TRUE(reader.DoesSectionExist("Common"));
    EXPECT_TRUE(reader.DoesParameterExist("Common", "Param1"));
    EXPECT_TRUE(reader.DoesParameterExist("Common", "Param2"));

    EXPECT_FALSE(reader.DoesSectionExist("NotExist"));
    EXPECT_FALSE(reader.DoesParameterExist("NotExist", "Param1"));
    EXPECT_FALSE(reader.DoesParameterExist("Common", "ParamNotExist"));

    EXPECT_EQ(1, reader.GetParameter<int>("Common", "Param1", 0));
    EXPECT_EQ(0, reader.GetParameter<int>("Common", "ParamNotExist", 0));

    EXPECT_EQ(std::string("Exp=2"), reader.GetParameter<std::string>("Common", "Param2", std::string()));
    EXPECT_EQ(std::string("1"), reader.GetParameter<std::string>("Common", "Param1", std::string()));
    EXPECT_EQ(std::string(), reader.GetParameter<std::string>("Common", "ParamNotExist", std::string()));
}
} // namespace IniReaderTest