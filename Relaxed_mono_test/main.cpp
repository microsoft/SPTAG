#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include "include/Config.h"
#include "include/RelaxedmonoRunner.h"


int main(int argc, char* argv[]) {
    std::string configPath = "config.ini";
    if (argc > 1) {
        configPath = argv[1];
    }

    try {
        RelaxedmonoConfig config = ConfigLoader::Load(configPath);
        RelaxedmonoRunner runner(config);
        runner.run();
    } catch (const std::exception& ex) {
        std::cerr << "Relaxedmono failed: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
