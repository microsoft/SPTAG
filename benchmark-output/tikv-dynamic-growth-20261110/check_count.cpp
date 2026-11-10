#include "inc/Core/SPANN/Index.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
{
    try {
        if (argc != 6) throw std::runtime_error("Usage: check_count INDEX INITIAL TOTAL BATCH LAYERS");
        auto number = [](const char* text) {
            std::size_t parsed = 0;
            const std::string value(text);
            const int result = std::stoi(value, &parsed);
            if (parsed != value.size() || result <= 0) throw std::runtime_error("Invalid positive integer");
            return result;
        };
        const auto initial = number(argv[2]), total = number(argv[3]), batch = number(argv[4]);
        const auto layers = number(argv[5]);
        if (initial > total || (total - initial) % batch)
            throw std::runtime_error("Invalid vector-count boundaries");
        std::shared_ptr<SPTAG::VectorIndex> index;
        if (SPTAG::VectorIndex::LoadIndex(argv[1], index) != SPTAG::ErrorCode::Success || !index)
            throw std::runtime_error("Index reload failed");
        auto spann = std::dynamic_pointer_cast<SPTAG::SPANN::Index<std::uint8_t>>(index);
        if (!spann || index->GetNumSamples() != total || index->GetFeatureDim() != 128 ||
            spann->GetParameter("Layers", "BuildSSDIndex") != std::to_string(layers) ||
            spann->GetParameter("Storage", "BuildSSDIndex") != "TIKVIO" ||
            spann->GetLayerResizeStatus() != SPTAG::ErrorCode::Success)
            throw std::runtime_error("Reloaded count, dimension, topology, backend or resize status mismatch");
        std::vector<SPTAG::SizeType> ids;
        if (spann->GetHeadIndexMapping(0, ids) != SPTAG::ErrorCode::Success || ids.size() != total)
            throw std::runtime_error("Contained-ID count mismatch");
        std::sort(ids.begin(), ids.end());
        for (SPTAG::SizeType vid = 0; vid < total; ++vid)
            if (ids[vid] != vid) throw std::runtime_error("Contained IDs are not exactly [0,TOTAL)");
        std::vector<bool> sources(total - initial, false);
        for (SPTAG::SizeType vid = initial; vid < total; ++vid) {
            auto bytes = index->GetMetadata(vid);
            if (bytes.Length() == 0) throw std::runtime_error("Missing inserted metadata");
            std::string metadata(reinterpret_cast<const char*>(bytes.Data()), bytes.Length());
            std::size_t parsed = 0;
            const auto source = std::stoi(metadata, &parsed);
            if (parsed != metadata.size() || source < initial || source >= total ||
                (source - initial) / batch != (vid - initial) / batch || sources[source - initial])
                throw std::runtime_error("Inserted metadata is not a per-batch source-ID bijection");
            sources[source - initial] = true;
        }
        std::cout << "VERIFIED: samples=" << total << ", layers=" << layers
                  << ", exact live IDs=[0," << total << "), inserted metadata bijection="
                  << total - initial << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
