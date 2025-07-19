#pragma once
#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>

struct RelaxedmonoConfig {
    // Benchmark general
    std::string mode; // "random" or "space1vb"
    int initialCount;
    int appendCount;
    int appendBatchSize;
    int appendRatePerSec;
    int testDurationSec;
    int logIntervalSec;
    std::string indexPath;
    std::vector<int> queryKs;
    int numThreads;

    // Random mode
    int dimensions; // only for mode=random
    int seed = -1;

    // SPACE1VB mode
    std::string space1vbBasePath;

    // SPANN-specific configuration
    std::string distCalcMethod;
    std::string indexAlgoType;
    std::string kvPath;
    std::string ssdInfoFile;
    int selectHeadThreads;
    float selectHeadRatio;
    int buildHeadThreads;
    int refineIterations;
    int buildSSDIndexThreads;
    bool useKV;
    bool buildSSDIndex;
    int postingPageLimit;
    int searchPostingPageLimit;
    int internalResultNum;
    int searchInternalResultNum;
};

class ConfigLoader {
public:
    static RelaxedmonoConfig Load(const std::string& configFilePath);
};
