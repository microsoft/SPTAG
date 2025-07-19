#include "Config.h"
#include <boost/property_tree/ini_parser.hpp>
#include <iostream>
#include <sstream>

RelaxedmonoConfig ConfigLoader::Load(const std::string& configFilePath) {
    RelaxedmonoConfig config;
    boost::property_tree::ptree tree;

    try {
        boost::property_tree::ini_parser::read_ini(configFilePath, tree);
    } catch (const boost::property_tree::ini_parser_error& e) {
        std::cerr << "Error reading config file: " << e.what() << std::endl;
        throw;
    }

    try {
        // Mode-specific handling
        config.mode = tree.get<std::string>("Relaxedmono.mode", "random");

        config.initialCount      = tree.get<int>("Relaxedmono.initialCount");
        config.appendCount       = tree.get<int>("Relaxedmono.appendCount");
        config.appendBatchSize   = tree.get<int>("Relaxedmono.appendBatchSize", 10);
        config.appendRatePerSec  = tree.get<int>("Relaxedmono.appendRatePerSec", 1);
        config.testDurationSec   = tree.get<int>("Relaxedmono.testDurationSec", 60);
        config.logIntervalSec    = tree.get<int>("Relaxedmono.logIntervalSec", 10);
        config.indexPath         = tree.get<std::string>("Relaxedmono.indexPath");
        config.numThreads        = tree.get<int>("Relaxedmono.numThreads", 4);

        // Parse queryKs list
        std::string ksList = tree.get<std::string>("Relaxedmono.queryKs", "1,10,100");
        std::stringstream ss(ksList);
        std::string token;
        while (std::getline(ss, token, ',')) {
            config.queryKs.push_back(std::stoi(token));
        }

        // Mode: random
        if (config.mode == "random") {
            config.dimensions = tree.get<int>("random.dimensions");
            config.seed       = tree.get<int>("random.seed", -1);
        }

        // Mode: space1vb
        else if (config.mode == "space1vb") {
            config.space1vbBasePath = tree.get<std::string>("space1vb.basePath");
        }

        // SPANN configuration
        config.indexAlgoType         = tree.get<std::string>("SPANN.IndexAlgoType", "BKT");
        config.distCalcMethod        = tree.get<std::string>("SPANN.DistCalcMethod", "L2");
        config.kvPath                = tree.get<std::string>("SPANN.KVPath", "/tmp/rocksdb");
        config.ssdInfoFile           = tree.get<std::string>("SPANN.SsdInfoFile", "/tmp/ssdinfo/info.bin");
        config.selectHeadThreads     = tree.get<int>("SPANN.SelectHeadThreads", 4);
        config.selectHeadRatio       = tree.get<float>("SPANN.SelectHeadRatio", 0.2f);
        config.buildHeadThreads      = tree.get<int>("SPANN.BuildHeadThreads", 4);
        config.refineIterations      = tree.get<int>("SPANN.RefineIterations", 3);
        config.buildSSDIndexThreads  = tree.get<int>("SPANN.BuildSSDIndexThreads", 4);
        config.useKV                 = tree.get<bool>("SPANN.UseKV", true);
        config.buildSSDIndex         = tree.get<bool>("SPANN.BuildSSDIndex", true);
        config.postingPageLimit      = tree.get<int>("SPANN.PostingPageLimit", 1000);
        config.searchPostingPageLimit = tree.get<int>("SPANN.SearchPostingPageLimit", 12);
        config.internalResultNum     = tree.get<int>("SPANN.InternalResultNum", 64);
        config.searchInternalResultNum = tree.get<int>("SPANN.SearchInternalResultNum", 64);

    } catch (const boost::property_tree::ptree_bad_path& e) {
        std::cerr << "Missing expected config field: " << e.what() << std::endl;
        throw;
    }

    return config;
}
