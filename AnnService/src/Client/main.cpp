#include "inc/Client/Options.h"
#include "inc/Client/ClientWrapper.h"

#include <cstdio>
#include <atomic>
#include <iostream>

std::unique_ptr<SpaceV::Client::ClientWrapper> g_client;

int main(int argc, char** argv)
{
    SpaceV::Client::ClientOptions options;
    if (!options.Parse(argc - 1, argv + 1))
    {
        return 1;
    }

    g_client.reset(new SpaceV::Client::ClientWrapper(options));
    if (!g_client->IsAvailable())
    {
        return 1;
    }

    g_client->WaitAllFinished();
    fprintf(stdout, "connection done\n");

    std::string line;
    std::cout << "Query: " << std::flush;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            break;
        }

        SpaceV::Socket::RemoteQuery query;
        query.m_type = SpaceV::Socket::RemoteQuery::QueryType::String;
        query.m_queryString = std::move(line);

        SpaceV::Socket::RemoteSearchResult result;
        auto callback = [&result](SpaceV::Socket::RemoteSearchResult p_result)
        {
            result = std::move(p_result);
        };

        g_client->SendQueryAsync(query, callback, options);
        g_client->WaitAllFinished();

        std::cout << "Status: " << static_cast<std::uint32_t>(result.m_status) << std::endl;

        for (const auto& indexRes : result.m_allIndexResults)
        {
            fprintf(stdout, "Index: %s\n", indexRes.m_indexName.c_str());

            int idx = 0;
            for (const auto& res : indexRes.m_results)
            {
                fprintf(stdout, "------------------\n");
                fprintf(stdout, "DocIndex: %d Distance: %f\n", res.Key, res.Dist);
                if (indexRes.m_results.WithMeta())
                {
                    const auto& metadata = indexRes.m_results.GetMetadata(idx);
                    fprintf(stdout, " MetaData: %.*s\n", static_cast<int>(metadata.Length()), metadata.Data());
                }

                ++idx;
            }
        }

        std::cout << "Query: " << std::flush;
    }

    return 0;
}

