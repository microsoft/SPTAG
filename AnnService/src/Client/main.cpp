// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Client/ClientWrapper.h"
#include "inc/Client/Options.h"
#include "inc/Socket/RemoteInsertDeleteQuery.h"
#include "inc/Core/CommonDataStructure.h"

#include <atomic>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

using namespace SPTAG;

std::unique_ptr<SPTAG::Client::ClientWrapper> g_client;

// Helper function to parse float vector from string (e.g., "1.0|2.0|3.0|4.0")
std::vector<float> ParseFloatVector(const std::string& str)
{
    std::vector<float> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, '|'))
    {
        result.push_back(std::stof(item));
    }
    
    return result;
}

// Helper function to parse int8 vector from string (e.g., "1|-2|3|4")
std::vector<std::int8_t> ParseInt8Vector(const std::string& str)
{
    std::vector<std::int8_t> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, '|'))
    {
        result.push_back(static_cast<std::int8_t>(std::stoi(item)));
    }
    
    return result;
}

// Helper function to parse command with flexible search syntax
bool ParseCommand(const std::string& line, std::string& command, std::string& indexName, std::string& data)
{
    std::istringstream iss(line);
    iss >> command;
    
    if (command == "search")
    {
        // For search, check if next token looks like an index name or search parameter
        std::string next;
        iss >> next;
        
        if (next.find('=') == std::string::npos && next.find(':') == std::string::npos)
        {
            // It's an index name
            indexName = next;
        }
        else
        {
            // No index name provided, use default and include this token in data
            indexName = "default";
            data = next;
        }
        
        // Get the rest of the line as data
        std::string restOfLine;
        std::getline(iss, restOfLine);
        if (!data.empty())
        {
            data += restOfLine;
        }
        else
        {
            data = restOfLine;
        }
        
        data.erase(0, data.find_first_not_of(" \t"));
        return true;
    }
    else
    {
        // For other commands, expect: command indexName data
        iss >> indexName;
        
        // Get the rest of the line as data
        std::getline(iss, data);
        data.erase(0, data.find_first_not_of(" \t"));
        
        return !command.empty() && !indexName.empty();
    }
}

void HandleSearch(const std::string& indexName, const std::string& query, const SPTAG::Client::ClientOptions& options)
{
    SPTAG::Socket::RemoteQuery searchQuery;
    searchQuery.m_type = SPTAG::Socket::RemoteQuery::QueryType::String;
    searchQuery.m_queryString = indexName + " " + query;

    SPTAG::Socket::RemoteSearchResult result;
    auto callback = [&result](SPTAG::Socket::RemoteSearchResult p_result) { result = std::move(p_result); };

    g_client->SendQueryAsync(searchQuery, callback, options);
    g_client->WaitAllFinished();

    std::cout << "Search Status: " << static_cast<std::uint32_t>(result.m_status) << std::endl;

    for (const auto &indexRes : result.m_allIndexResults)
    {
        std::cout << "Index: " << indexRes.m_indexName << std::endl;

        int idx = 0;
        for (const auto &res : indexRes.m_results)
        {
            std::cout << "------------------" << std::endl;
            std::cout << "DocIndex: " << res.VID << " Distance: " << res.Dist;
            if (indexRes.m_results.WithMeta())
            {
                const auto &metadata = indexRes.m_results.GetMetadata(idx);
                std::cout << " MetaData: " << std::string((char *)metadata.Data(), metadata.Length());
            }
            std::cout << std::endl;
            ++idx;
        }
    }
}

void HandleInsertFloat(const std::string& indexName, const std::string& vectorStr, const SPTAG::Client::ClientOptions& options)
{
    auto vector = ParseFloatVector(vectorStr);
    if (vector.empty())
    {
        std::cout << "Error: Invalid vector format. Use pipe-separated values (e.g., 1.0|2.0|3.0)" << std::endl;
        return;
    }

    SPTAG::Socket::RemoteInsertQuery insertQuery;
    insertQuery.m_type = SPTAG::Socket::RemoteInsertQuery::InsertType::Vector;
    insertQuery.m_indexName = indexName;
    insertQuery.m_dimension = static_cast<DimensionType>(vector.size());
    insertQuery.m_valueType = VectorValueType::Float;
    insertQuery.m_vectorCount = 1;
    insertQuery.m_normalized = false;
    insertQuery.m_withMetaIndex = false;

    insertQuery.m_vectorData.resize(vector.size() * sizeof(float));
    std::memcpy(insertQuery.m_vectorData.data(), vector.data(), insertQuery.m_vectorData.size());

    SPTAG::Socket::RemoteInsertDeleteResult result;
    auto callback = [&result](SPTAG::Socket::RemoteInsertDeleteResult p_result) { result = std::move(p_result); };

    g_client->SendInsertAsync(insertQuery, callback, options);
    g_client->WaitAllFinished();

    std::cout << "Insert Status: ";
    switch (result.m_status)
    {
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::Success:
        std::cout << "Success";
        if (!result.m_newVectorIds.empty())
        {
            std::cout << " - Assigned ID(s): ";
            for (auto id : result.m_newVectorIds)
            {
                std::cout << id << " ";
            }
        }
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::Failed:
        std::cout << "Failed";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex:
        std::cout << "Invalid Index";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData:
        std::cout << "Invalid Data";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::MemoryOverflow:
        std::cout << "Memory Overflow";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::DimensionMismatch:
        std::cout << "Dimension Mismatch";
        break;
    }
    
    if (!result.m_message.empty())
    {
        std::cout << " - Message: " << result.m_message;
    }
    std::cout << std::endl;
    std::cout << "Processed Count: " << result.m_processedCount << std::endl;
}

void HandleInsert(const std::string& indexName, const std::string& vectorStr, const SPTAG::Client::ClientOptions& options)
{
    // Parse vector as int8 (default for our MURREN and FBV7 datasets)
    auto vector = ParseInt8Vector(vectorStr);
    if (vector.empty())
    {
        std::cout << "Error: Invalid vector format. Use pipe-separated values (e.g., 1|-2|3)" << std::endl;
        return;
    }

    SPTAG::Socket::RemoteInsertQuery insertQuery;
    insertQuery.m_type = SPTAG::Socket::RemoteInsertQuery::InsertType::Vector;
    insertQuery.m_indexName = indexName;
    insertQuery.m_dimension = static_cast<DimensionType>(vector.size());
    insertQuery.m_valueType = VectorValueType::Int8;
    insertQuery.m_vectorCount = 1;
    insertQuery.m_normalized = false;
    insertQuery.m_withMetaIndex = false;

    insertQuery.m_vectorData.resize(vector.size() * sizeof(std::int8_t));
    std::memcpy(insertQuery.m_vectorData.data(), vector.data(), insertQuery.m_vectorData.size());

    SPTAG::Socket::RemoteInsertDeleteResult result;
    auto callback = [&result](SPTAG::Socket::RemoteInsertDeleteResult p_result) { result = std::move(p_result); };

    g_client->SendInsertAsync(insertQuery, callback, options);
    g_client->WaitAllFinished();

    std::cout << "Insert Status: ";
    switch (result.m_status)
    {
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::Success:
        std::cout << "Success";
        if (!result.m_newVectorIds.empty())
        {
            std::cout << " - Assigned ID(s): ";
            for (auto id : result.m_newVectorIds)
            {
                std::cout << id << " ";
            }
        }
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::Failed:
        std::cout << "Failed";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex:
        std::cout << "Invalid Index";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData:
        std::cout << "Invalid Data";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::MemoryOverflow:
        std::cout << "Memory Overflow";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::DimensionMismatch:
        std::cout << "Dimension Mismatch";
        break;
    }
    
    if (!result.m_message.empty())
    {
        std::cout << " - Message: " << result.m_message;
    }
    std::cout << std::endl;
    std::cout << "Processed Count: " << result.m_processedCount << std::endl;
}

void HandleDeleteById(const std::string& indexName, const std::string& idStr, const SPTAG::Client::ClientOptions& options)
{
    SizeType id = std::stoul(idStr);

    SPTAG::Socket::RemoteDeleteQuery deleteQuery;
    deleteQuery.m_type = SPTAG::Socket::RemoteDeleteQuery::DeleteType::ByVectorId;
    deleteQuery.m_indexName = indexName;
    deleteQuery.m_vectorIds.push_back(id);

    SPTAG::Socket::RemoteInsertDeleteResult result;
    auto callback = [&result](SPTAG::Socket::RemoteInsertDeleteResult p_result) { result = std::move(p_result); };

    g_client->SendDeleteAsync(deleteQuery, callback, options);
    g_client->WaitAllFinished();

    std::cout << "Delete Status: ";
    switch (result.m_status)
    {
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::Success:
        std::cout << "Success";
        if (!result.m_newVectorIds.empty())
        {
            std::cout << " - Deleted ID(s): ";
            for (auto id : result.m_newVectorIds)
            {
                std::cout << id << " ";
            }
        }
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::Failed:
        std::cout << "Failed";
        break;
    case SPTAG::Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex:
        std::cout << "Invalid Index";
        break;
    }
    
    if (!result.m_message.empty())
    {
        std::cout << " - Message: " << result.m_message;
    }
    std::cout << std::endl;
    std::cout << "Processed Count: " << result.m_processedCount << std::endl;
}

void PrintHelp()
{
    std::cout << "\nAvailable commands:" << std::endl;
    std::cout << "  search [<index_name>] <query>   - Search using a query string" << std::endl;
    std::cout << "  insert <index_name> <vector>    - Insert a vector (pipe-separated int8 values)" << std::endl;
    std::cout << "  insertf <index_name> <vector>   - Insert a vector (pipe-separated float values)" << std::endl;
    std::cout << "  delete <index_name> <id>        - Delete a vector by ID" << std::endl;
    std::cout << "  help                            - Show this help message" << std::endl;
    std::cout << "  exit                            - Exit the client" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  search K=10 V:6|-5|4|0|-2|9|1|-5|0|10" << std::endl;
    std::cout << "  search MyIndex K=10 V:6|-5|4|0|-2|9|1|-5|0|10" << std::endl;
    std::cout << "  insert MyIndex 6|-5|4|0|-2|9|1|-5|0|10" << std::endl;
    std::cout << "  insertf MyIndex 1.0|2.0|3.0|4.0" << std::endl;
    std::cout << "  delete MyIndex 12345" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    SPTAG::Client::ClientOptions options;
    if (!options.Parse(argc - 1, argv + 1))
    {
        return 1;
    }

    g_client.reset(new SPTAG::Client::ClientWrapper(options));
    if (!g_client->IsAvailable())
    {
        return 1;
    }

    g_client->WaitAllFinished();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "connection done\n");

    PrintHelp();

    std::string line;
    std::cout << "Command: " << std::flush;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            std::cout << "Command: " << std::flush;
            continue;
        }

        std::string command, indexName, data;
        
        if (line == "help")
        {
            PrintHelp();
        }
        else if (line == "exit")
        {
            break;
        }
        else if (ParseCommand(line, command, indexName, data))
        {
            if (command == "search")
            {
                HandleSearch(indexName, data, options);
            }
            else if (command == "insert")
            {
                HandleInsert(indexName, data, options);
            }
            else if (command == "insertf")
            {
                HandleInsertFloat(indexName, data, options);
            }
            else if (command == "delete")
            {
                HandleDeleteById(indexName, data, options);
            }
            else
            {
                std::cout << "Unknown command: " << command << std::endl;
                std::cout << "Type 'help' for available commands." << std::endl;
            }
        }
        else
        {
            // Fallback to old behavior for backward compatibility
            SPTAG::Socket::RemoteQuery query;
            query.m_type = SPTAG::Socket::RemoteQuery::QueryType::String;
            query.m_queryString = std::move(line);

            SPTAG::Socket::RemoteSearchResult result;
            auto callback = [&result](SPTAG::Socket::RemoteSearchResult p_result) { result = std::move(p_result); };

            g_client->SendQueryAsync(query, callback, options);
            g_client->WaitAllFinished();

            std::cout << "Status: " << static_cast<std::uint32_t>(result.m_status) << std::endl;

            for (const auto &indexRes : result.m_allIndexResults)
            {
                std::cout << "Index: " << indexRes.m_indexName << std::endl;

                int idx = 0;
                for (const auto &res : indexRes.m_results)
                {
                    std::cout << "------------------" << std::endl;
                    std::cout << "DocIndex: " << res.VID << " Distance: " << res.Dist;
                    if (indexRes.m_results.WithMeta())
                    {
                        const auto &metadata = indexRes.m_results.GetMetadata(idx);
                        std::cout << " MetaData: " << std::string((char *)metadata.Data(), metadata.Length());
                    }
                    std::cout << std::endl;
                    ++idx;
                }
            }
        }

        std::cout << "Command: " << std::flush;
    }

    return 0;
}
