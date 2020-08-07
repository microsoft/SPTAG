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
                LOG(Helper::LogLevel::LL_Empty, "Failed to parse args around \"%s\"\n", *p_args);
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
            LOG(Helper::LogLevel::LL_Empty, "Required option not set:\n  ");
            option->PrintDescription();
            LOG(Helper::LogLevel::LL_Empty, "\n");
            isValid = false;
        }
    }

    if (!isValid)
    {
        LOG(Helper::LogLevel::LL_Empty, "\n");
        PrintHelp();
        return false;
    }

    return true;
}


void
ArgumentsParser::PrintHelp()
{
    LOG(Helper::LogLevel::LL_Empty, "Usage: ");
    for (auto& option : m_arguments)
    {
        LOG(Helper::LogLevel::LL_Empty, "\n  ");
        option->PrintDescription();
    }

    LOG(Helper::LogLevel::LL_Empty, "\n\n");
}
