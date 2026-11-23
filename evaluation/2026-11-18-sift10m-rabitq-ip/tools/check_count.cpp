#include "inc/Core/SPANN/Index.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv)
{
    try {
        if (argc != 5) throw std::runtime_error("Usage: check_count INDEX INITIAL BATCH BATCHES");
        const auto initial = std::stoi(argv[2]), batch = std::stoi(argv[3]), batches = std::stoi(argv[4]);
        const std::int64_t count = std::int64_t(initial) + std::int64_t(batch) * batches;
        if (initial <= 0 || batch <= 0 || batches < 0 || count > (std::numeric_limits<SPTAG::SizeType>::max)())
            throw std::runtime_error("Invalid phase sizes");
        const auto total = static_cast<SPTAG::SizeType>(count);
        std::shared_ptr<SPTAG::VectorIndex> index;
        if (SPTAG::VectorIndex::LoadIndex(argv[1], index) != SPTAG::ErrorCode::Success || !index)
            throw std::runtime_error("Final index reload failed");
        if (index->GetNumSamples() != total)
            throw std::runtime_error("Final reloaded sample count mismatch");
        auto spann = std::dynamic_pointer_cast<SPTAG::SPANN::Index<std::uint8_t>>(index);
        if (!spann) throw std::runtime_error("Unexpected index type");
        std::vector<SPTAG::SizeType> ids;
        if (spann->GetHeadIndexMapping(0, ids) != SPTAG::ErrorCode::Success || ids.size() != total)
            throw std::runtime_error("Final contained-ID count mismatch");
        std::sort(ids.begin(), ids.end());
        std::vector<bool> sources(total, false);
        for (SPTAG::SizeType vid = 0; vid < total; ++vid) {
            if (ids[vid] != vid) throw std::runtime_error("Final contained-ID range mismatch");
            auto bytes = index->GetMetadata(vid);
            if (bytes.Length() == 0) throw std::runtime_error("Missing source-ID metadata");
            std::string metadata(reinterpret_cast<const char*>(bytes.Data()), bytes.Length());
            std::size_t parsed = 0;
            const auto source = std::stoi(metadata, &parsed);
            if (parsed != metadata.size() || source < 0 || source >= total || sources[source] ||
                (vid < initial ? source != vid :
                 source < initial || (source - initial) / batch != (vid - initial) / batch))
                throw std::runtime_error("Metadata is not an initial-ID/per-batch source-ID bijection");
            sources[source] = true;
        }
        std::cout << "VERIFIED: reloaded " << total << " samples; contained IDs exactly [0,"
                  << total << "); all metadata IDs bijective, including " << total - initial
                  << " inserted entries in " << batches << " batches.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
