#include "inc/Core/SPANN/Index.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
{
    try {
        if (argc != 2) throw std::runtime_error("Usage: check_count INDEX");
        constexpr SPTAG::SizeType initial = 9900000, total = 10000000, batch = 10000;
        std::shared_ptr<SPTAG::VectorIndex> index;
        if (SPTAG::VectorIndex::LoadIndex(argv[1], index) != SPTAG::ErrorCode::Success || !index)
            throw std::runtime_error("Final 10M index reload failed");
        if (index->GetNumSamples() != total)
            throw std::runtime_error("Final reloaded GetNumSamples is not 10M");
        auto spann = std::dynamic_pointer_cast<SPTAG::SPANN::Index<std::uint8_t>>(index);
        if (!spann) throw std::runtime_error("Unexpected index type");
        std::vector<SPTAG::SizeType> ids;
        if (spann->GetHeadIndexMapping(0, ids) != SPTAG::ErrorCode::Success || ids.size() != total)
            throw std::runtime_error("Final contained-ID count is not 10M");
        std::sort(ids.begin(), ids.end());
        for (SPTAG::SizeType vid = 0; vid < total; ++vid) {
            if (ids[vid] != vid) throw std::runtime_error("Final contained IDs differ from [0,10M)");
        }
        std::vector<bool> sources(total - initial, false);
        for (SPTAG::SizeType vid = initial; vid < total; ++vid) {
            auto bytes = index->GetMetadata(vid);
            if (bytes.Length() == 0) throw std::runtime_error("Missing inserted metadata");
            std::string metadata(reinterpret_cast<const char*>(bytes.Data()), bytes.Length());
            std::size_t parsed = 0;
            SPTAG::SizeType source = std::stoi(metadata, &parsed);
            if (parsed != metadata.size() || source < initial || source >= total ||
                (source - initial) / batch != (vid - initial) / batch || sources[source - initial])
                throw std::runtime_error("Inserted metadata is not a per-batch source-ID bijection");
            sources[source - initial] = true;
        }
        std::cout << "VERIFIED: saved index reload reports 10000000 samples; contained IDs exactly "
                  << "[0,10000000); all 100000 inserted metadata entries map bijectively within their batches.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
