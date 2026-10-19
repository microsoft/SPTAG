#include "inc/Core/VectorIndex.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr SPTAG::SizeType count = 128;
constexpr SPTAG::DimensionType dimension = 8;
constexpr int neighbors = 3;

void Check(SPTAG::ErrorCode code, const char* operation)
{
    if (code != SPTAG::ErrorCode::Success)
        throw std::runtime_error(std::string(operation) + " failed, error code " +
                                 std::to_string(static_cast<int>(code)));
}

void Search(const std::shared_ptr<SPTAG::VectorIndex>& index, const std::vector<std::string>& businessIDs)
{
    for (SPTAG::SizeType expected : {0, 31, 63})
    {
        std::vector<float> query(dimension);
        for (SPTAG::DimensionType column = 0; column < dimension; ++column)
            query[column] = static_cast<float>(expected) + static_cast<float>(column) / dimension;
        SPTAG::QueryResult result(query.data(), neighbors, false);
        Check(index->SearchIndex(result), "SearchIndex");
        for (int rank = 0; rank < neighbors; ++rank)
        {
            const auto* match = result.GetResult(rank);
            if (match->VID < 0 || static_cast<size_t>(match->VID) >= businessIDs.size())
                throw std::runtime_error("Search returned an invalid row ID");
            std::cout << "query=" << expected << " rank=" << rank
                      << " id=" << businessIDs.at(match->VID) << " distance=" << match->Dist << '\n';
        }
        if (result.GetResult(0)->VID != expected || result.GetResult(0)->Dist != 0.0f)
            throw std::runtime_error("Exact self-match check failed");
    }
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--load"))
            throw std::runtime_error("Usage: sptag_sdk_example <output-directory> [--load]");
        const std::filesystem::path output = std::filesystem::absolute(argv[1]);
        const bool load = argc == 3;
        std::shared_ptr<SPTAG::VectorIndex> index;
        std::vector<std::string> businessIDs;
        if (!load)
        {
            if (std::filesystem::exists(output))
                throw std::runtime_error("Output already exists; use a new directory or --load");
            std::vector<float> vectors(static_cast<size_t>(count) * dimension);
            for (SPTAG::SizeType row = 0; row < count; ++row)
            {
                businessIDs.push_back("document-" + std::to_string(row));
                for (SPTAG::DimensionType column = 0; column < dimension; ++column)
                    vectors[static_cast<size_t>(row) * dimension + column] =
                        static_cast<float>(row) + static_cast<float>(column) / dimension;
            }
            index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
            if (!index) throw std::runtime_error("CreateInstance failed");
            Check(index->SetParameter("DistCalcMethod", "L2"), "SetParameter");
            Check(index->SetParameter("NumberOfThreads", "1"), "SetParameter");
            Check(index->BuildIndex(vectors.data(), count, dimension), "BuildIndex");
            std::filesystem::create_directories(output);
            Check(index->SaveIndex((output / "index").string()), "SaveIndex");
            std::ofstream ids(output / "ids.txt");
            ids.exceptions(std::ios::badbit | std::ios::failbit);
            for (const auto& id : businessIDs) ids << id << '\n';
            ids.close();
        }
        else
        {
            Check(SPTAG::VectorIndex::LoadIndex((output / "index").string(), index), "LoadIndex");
            std::ifstream ids(output / "ids.txt");
            if (!ids) throw std::runtime_error("Cannot open business ID mapping");
            for (std::string id; std::getline(ids, id);) businessIDs.push_back(id);
            if (ids.bad()) throw std::runtime_error("Cannot read business ID mapping");
            if (index->GetFeatureDim() != dimension || index->GetNumSamples() != count ||
                index->GetVectorValueType() != SPTAG::VectorValueType::Float ||
                index->GetDistCalcMethod() != SPTAG::DistCalcMethod::L2 ||
                businessIDs.size() != static_cast<size_t>(count))
                throw std::runtime_error("Saved artifacts do not match this example");
        }
        Search(index, businessIDs);
        std::cout << "C++ SDK example passed: 3/3 self-matches; index and business IDs retained at "
                  << output.string() << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
