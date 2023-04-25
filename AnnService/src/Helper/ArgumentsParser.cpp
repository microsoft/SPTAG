// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/ArgumentsParser.h"

using namespace SPTAG::Helper;


ArgumentsParser::IArgument::IArgument()
{
}


ArgumentsParser::IArgument::~IArgument()
{
}


ArgumentsParser::ArgumentsParser()
{
}


ArgumentsParser::~ArgumentsParser()
{
}


bool
ArgumentsParser::Parse(int p_argc, char** p_args)
{
    while (p_argc > 0)
    {
        int last = p_argc;
        for (auto& option : m_arguments)
        {
            if (!option->ParseValue(p_argc, p_args))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "Failed to parse args around \"%s\"\n", *p_args);
                PrintHelp();
                return false;
            }
        }

        if (last == p_argc)
        {
            p_argc -= 1;
            p_args += 1;
        }
    }

    bool isValid = true;
    for (auto& option : m_arguments)
    {
        if (option->IsRequiredButNotSet())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "Required option not set:\n  ");
            option->PrintDescription();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "\n");
            isValid = false;
        }
    }

    if (!isValid)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "\n");
        PrintHelp();
        return false;
    }

    return true;
}


void
ArgumentsParser::PrintHelp()
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "Usage: ");
    for (auto& option : m_arguments)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "\n  ");
        option->PrintDescription();
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Empty, "\n\n");
}
