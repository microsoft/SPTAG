#include "inc/Aggregator/AggregatorService.h"

SpaceV::Aggregator::AggregatorService g_service;

int main(int argc, char* argv[])
{
    if (!g_service.Initialize())
    {
        return 1;
    }

    g_service.Run();

    return 0;
}

