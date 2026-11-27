// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Test.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

namespace
{
    using namespace SPTAG;
    constexpr int Dimension = 128, K = 5, QueryCount = 10000;

    std::string Environment(const char* name)
    {
        const auto* value = std::getenv(name);
        if (!value || !*value) throw std::runtime_error(std::string("Missing environment variable: ") + name);
        return value;
    }

    std::vector<uint8_t> ReadVectors(const std::string& path, int required)
    {
        std::ifstream input(path, std::ios::binary);
        int32_t header[2]{};
        if (!input.read(reinterpret_cast<char*>(header), sizeof(header)) ||
            header[0] < required || header[1] != Dimension ||
            std::filesystem::file_size(path) != sizeof(header) + static_cast<uint64_t>(header[0]) * Dimension)
            throw std::runtime_error("Invalid SIFT input: " + path);
        std::vector<uint8_t> result(static_cast<size_t>(required) * Dimension);
        if (!input.read(reinterpret_cast<char*>(result.data()), result.size()))
            throw std::runtime_error("Short SIFT input: " + path);
        return result;
    }

    int SquaredL2(const uint8_t* left, const uint8_t* right)
    {
        int distance = 0;
        for (int d = 0; d < Dimension; ++d) {
            const int delta = static_cast<int>(left[d]) - right[d];
            distance += delta * delta;
        }
        return distance;
    }
}

BOOST_AUTO_TEST_SUITE(SiftLifecycleTest)

BOOST_AUTO_TEST_CASE(DeletionCalibration, *boost::unit_test::disabled())
{
    const std::filesystem::path source = Environment("SPTAG_SIFT_CALIBRATION_INDEX");
    const std::filesystem::path output = Environment("SPTAG_SIFT_OUTPUT");
    BOOST_REQUIRE(!std::filesystem::exists(output));
    std::filesystem::create_directories(output);
    auto queries = ReadVectors(Environment("SPTAG_SIFT_QUERIES"), QueryCount);
    Helper::IniReader profile;
    BOOST_REQUIRE(profile.LoadIniFile((source / "indexloader.ini").string()) == ErrorCode::Success);
    BOOST_REQUIRE(!profile.DoesSectionExist("MetaData"));
    BOOST_REQUIRE(!profile.DoesSectionExist("Quantizer"));
    // This is a fixed-topology measurement, not an automatic-resize acceptance run.
    profile.SetParameter("BuildSSDIndex", "EnableDynamicLayerShrink", "false");
    profile.SetParameter("BuildSSDIndex", "HeadMemoryHighWatermark", "1000000000000");
    profile.SetParameter("BuildSSDIndex", "HeadMemoryTarget", "900000000000");
    profile.SetParameter("BuildSSDIndex", "HeadMemoryLowWatermark", "800000000000");
    auto index = std::make_shared<SPANN::Index<uint8_t>>();
    BOOST_REQUIRE(index->LoadConfig(profile) == ErrorCode::Success);
    BOOST_REQUIRE(index->SetParameter("IndexDirectory", source.string().c_str(), "Base") == ErrorCode::Success);
    std::vector<std::shared_ptr<Helper::DiskIO>> handles;
    const auto files = index->GetIndexFiles();
    for (const auto& file : *files) {
        auto handle = SPTAG::f_createIO();
        BOOST_REQUIRE(handle && handle->Initialize((source / file).string().c_str(), std::ios::binary | std::ios::in));
        handles.push_back(std::move(handle));
    }
    BOOST_REQUIRE(index->LoadIndexData(handles) == ErrorCode::Success);
    index->SetReady(true);
    constexpr int Total = 11000000, Final = 9000000, Batch = 100000;
    BOOST_REQUIRE_EQUAL(index->GetNumSamples(), Total);
    BOOST_REQUIRE_EQUAL(index->GetNumDeleted(), 0);
    std::vector<SizeType> deletion(Total);
    std::iota(deletion.begin(), deletion.end(), 0);
    std::mt19937 random(1729);
    std::shuffle(deletion.begin(), deletion.end(), random);
    std::ofstream deleted(output.parent_path() / "deleted-ids.i32", std::ios::binary);
    BOOST_REQUIRE(deleted.write(reinterpret_cast<const char*>(deletion.data()), (Total - Final) * sizeof(SizeType)));
    std::ofstream metrics(output.parent_path() / "deletion-metrics.jsonl");
    BOOST_REQUIRE(metrics);
    auto wait = [&]() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(20);
        while (!index->AllFinished() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        BOOST_REQUIRE(index->AllFinished());
        BOOST_REQUIRE(index->GetLayerResizeStatus() == ErrorCode::Success);
        BOOST_REQUIRE_EQUAL(index->GetParameter("Layers", "BuildSSDIndex"), "1");
    };
    auto snapshot = [&](const std::string& phase, int live) {
        std::vector<SizeType> routes, users;
        BOOST_REQUIRE(index->GetHeadIndexMapping(1, routes) == ErrorCode::Success);
        BOOST_REQUIRE(index->GetHeadIndexMapping(0, users) == ErrorCode::Success);
        BOOST_REQUIRE_EQUAL(users.size(), live);
        const auto slots = index->GetMemoryIndex()->GetNumSamples();
        BOOST_REQUIRE_GT(slots, 0);
        const auto bytes = index->GetHeadMemoryUsage();
        const uint64_t estimate = (bytes * routes.size() + slots - 1) / slots;
        metrics << "{\"phase\":\"" << phase << "\",\"live\":" << live << ",\"routes\":" << routes.size()
            << ",\"head_slots\":" << slots << ",\"head_bytes\":" << bytes
            << ",\"compact_estimate_bytes\":" << estimate << "}\n";
        metrics.flush();
        BOOST_TEST_MESSAGE("SIFT-CALIBRATION phase=" << phase << " live=" << live
            << " routes=" << routes.size() << " compactBytes=" << estimate);
    };
    auto search = [&](int q) {
        COMMON::QueryResultSet<uint8_t> query(queries.data() + q * Dimension, K);
        return index->SearchIndex(query);
    };
    auto evaluateQueries = [&]() {
        std::atomic_int next{0}, errors{0};
        {
            std::vector<std::jthread> workers;
            for (int thread = 0; thread < 32; ++thread) workers.emplace_back([&]() {
                for (int q = next++; q < QueryCount; q = next++)
                    if (search(q) != ErrorCode::Success) ++errors;
            });
        }
        BOOST_REQUIRE_EQUAL(errors.load(), 0);
        wait();
    };
    wait();
    snapshot("initial", Total);
    for (int begin = 0, iteration = 1; begin < Total - Final; begin += Batch, ++iteration) {
        for (int i = begin; i < begin + Batch; ++i)
            BOOST_REQUIRE(index->DeleteIndex(deletion[i]) == ErrorCode::Success);
        wait();
        for (int q = 0; q < 1000; ++q)
            BOOST_REQUIRE(search((q + iteration * 997) % QueryCount) == ErrorCode::Success);
        wait();
        snapshot("delete-" + std::to_string(iteration), Total - begin - Batch);
        if (iteration == 10 || iteration == 20) {
            for (int pass = 0; pass < 4; ++pass) evaluateQueries();
            snapshot("evaluated-" + std::to_string(iteration), Total - begin - Batch);
            BOOST_REQUIRE(index->SaveIndex(source.string()) == ErrorCode::Success);
        }
    }
    for (int sweep = 0; sweep < 10; ++sweep) {
        evaluateQueries();
        snapshot("converge-" + std::to_string(sweep + 1), Final);
    }
    BOOST_REQUIRE(index->SaveIndex(source.string()) == ErrorCode::Success);
    snapshot("complete", Final);
}

BOOST_AUTO_TEST_CASE(QueryBudgetCalibration, *boost::unit_test::disabled())
{
    constexpr int QueryCount = 64, K = 10;
    const auto directory = Environment("SPTAG_SIFT_OUTPUT");
    const int count = 1100000;
    auto vectors = ReadVectors(Environment("SPTAG_SIFT_VECTORS"), count);
    auto queries = ReadVectors(Environment("SPTAG_SIFT_QUERIES"), QueryCount);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex(directory, loaded) == ErrorCode::Success);
    auto index = std::dynamic_pointer_cast<SPANN::Index<uint8_t>>(loaded);
    BOOST_REQUIRE(index);
    BOOST_REQUIRE_EQUAL(index->GetNumSamples(), count);
    BOOST_REQUIRE(index->SetParameter("AsyncMergeInSearch", "false", "BuildSSDIndex") == ErrorCode::Success);
    std::array<int, QueryCount> boundaries{};
    std::atomic_int next{0};
    {
        std::vector<std::jthread> workers;
        for (int i = 0; i < 8; ++i) workers.emplace_back([&]() {
            for (int q = next++; q < QueryCount; q = next++) {
                std::vector<int> distances(count);
                for (int id = 0; id < count; ++id)
                    distances[id] = SquaredL2(queries.data() + q * Dimension,
                        vectors.data() + static_cast<size_t>(id) * Dimension);
                std::nth_element(distances.begin(), distances.begin() + K - 1, distances.end());
                boundaries[q] = distances[K - 1];
            }
        });
    }
    double best = 0;
    for (int budget : {128, 256, 512}) {
        BOOST_REQUIRE(index->SetParameter("SearchInternalResultNum", std::to_string(budget).c_str(),
            "BuildSSDIndex") == ErrorCode::Success);
        BOOST_REQUIRE(index->SetParameter("InternalResultNum", std::to_string(budget).c_str(),
            "BuildSSDIndex") == ErrorCode::Success);
        int hits = 0;
        const auto started = std::chrono::steady_clock::now();
        for (int q = 0; q < QueryCount; ++q) {
            COMMON::QueryResultSet<uint8_t> query(queries.data() + q * Dimension, K);
            BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
            for (int j = 0; j < K; ++j) {
                const auto* result = query.GetResult(j);
                BOOST_REQUIRE_GE(result->VID, 0);
                BOOST_REQUIRE_LT(result->VID, count);
                BOOST_REQUIRE_EQUAL(result->Dist, SquaredL2(queries.data() + q * Dimension,
                    vectors.data() + static_cast<size_t>(result->VID) * Dimension));
                if (result->Dist <= boundaries[q]) ++hits;
            }
        }
        const double recall = static_cast<double>(hits) / (QueryCount * K);
        best = (std::max)(best, recall);
        BOOST_TEST_MESSAGE("SIFT-BUDGET budget=" << budget << " recallAt10=" << recall
            << " elapsed=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    }
    BOOST_REQUIRE_GE(best, 0.95);
}

BOOST_AUTO_TEST_CASE(AutomaticLifecycle, *boost::unit_test::disabled())
{
    const auto mode = Environment("SPTAG_SIFT_MODE");
    BOOST_REQUIRE(mode == "pilot" || mode == "formal" || mode == "smoke" || mode == "config" ||
        mode == "growth-calibration" || mode == "merge-pilot");
    const auto* pivotSetting = std::getenv("SPTAG_SIFT_PIVOT");
    const std::string pivotText = pivotSetting ? pivotSetting : "10000000";
    size_t parsed = 0;
    const int pivot = std::stoi(pivotText, &parsed);
    BOOST_REQUIRE_EQUAL(parsed, pivotText.size());
    BOOST_REQUIRE(pivot == 16384 || pivot == 5000000 || pivot == 10000000);
    const bool small = pivot == 16384;
    const bool pilot = mode != "formal" && mode != "merge-pilot";
    const int scale = mode == "pilot" || mode == "merge-pilot" ? 10 : 1;
    BOOST_REQUIRE(!small || scale == 1);
    const int initial = small ? 4096 : mode == "smoke" ? 9000 : pivot / scale * 9 / 10;
    const int total = small ? 32768 : mode == "smoke" ? 11000 : pivot / scale * 11 / 10;
    const int batch = small ? 1024 : mode == "smoke" ? 100 : pivot / scale / 100;
    const int finalCount = small ? 1024 : initial;
    const int pivotBatch = (pivot / scale - initial) / batch;
    const int lastInsertBatch = (total - initial) / batch;
    const auto directory = std::filesystem::path(Environment("SPTAG_SIFT_OUTPUT"));
    BOOST_REQUIRE(!std::filesystem::exists(directory));
    std::filesystem::create_directories(directory);
    const auto high = std::stoull(Environment("SPTAG_SIFT_HIGH"));
    const auto target = std::stoull(Environment("SPTAG_SIFT_TARGET"));
    const auto low = std::stoull(Environment("SPTAG_SIFT_LOW"));
    auto index = std::make_shared<SPANN::Index<uint8_t>>();
    auto set = [&](const char* key, const std::string& value, const char* section = "BuildSSDIndex") {
        BOOST_REQUIRE(index->SetParameter(key, value.c_str(), section) == ErrorCode::Success);
    };
    set("IndexDirectory", directory.string(), "Base");
    set("DistCalcMethod", "L2", "Base");
    set("IndexAlgoType", "BKT", "Base");
    set("ValueType", "UInt8", "Base");
    Helper::IniReader profile;
    BOOST_REQUIRE(profile.LoadIniFile(Environment("SPTAG_SIFT_PROFILE")) == ErrorCode::Success);
    for (const char* section : {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"}) {
        BOOST_REQUIRE(profile.DoesSectionExist(section));
        for (const auto& parameter : profile.GetParameters(section)) {
            if (std::string(section) == "Base" &&
                (parameter.first == "vectorpath" || parameter.first == "vectorsize" || parameter.first == "indexdirectory"))
                continue;
            set(parameter.first.c_str(), parameter.second, section);
        }
    }
    set("VectorSize", std::to_string(initial), "Base");
    set("AsyncMergeInSearch", "true");
    set("EnableDynamicLayers", "true");
    set("EnableDynamicLayerShrink", "true");
    const auto* mergeThreshold = std::getenv("SPTAG_SIFT_MERGE_THRESHOLD");
    if (mergeThreshold) {
        size_t consumed = 0;
        const auto threshold = std::stoi(mergeThreshold, &consumed);
        BOOST_REQUIRE_EQUAL(consumed, std::string(mergeThreshold).size());
        BOOST_REQUIRE(threshold == 10 || threshold == 24 || threshold == 32 || threshold == 48);
        set("MergeThreshold", std::to_string(threshold));
    }
    set("HeadMemoryHighWatermark", std::to_string(high));
    set("HeadMemoryTarget", std::to_string(target));
    set("HeadMemoryLowWatermark", std::to_string(low));
    set("LayerResizeCooldownSeconds", small ? "1" : "60");
    set("LayerResizeTimeoutSeconds", small ? "60" : "600");
    if (small) set("GrowthFileSizeGB", "1");
    set("MaxLayers", "2");

    std::ofstream configured(directory.parent_path() / "configured.ini");
    BOOST_REQUIRE(configured);
    for (const char* section : {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"}) {
        configured << '[' << section << "]\n";
        for (const auto& parameter : profile.GetParameters(section))
            configured << parameter.first << '=' << index->GetParameter(parameter.first.c_str(), section) << '\n';
        if (std::string(section) == "BuildSSDIndex")
            for (const char* key : {"EnableDynamicLayers", "EnableDynamicLayerShrink",
                    "HeadMemoryHighWatermark", "HeadMemoryTarget", "HeadMemoryLowWatermark",
                    "LayerResizeCooldownSeconds", "LayerResizeTimeoutSeconds", "MaxLayers"}) {
                std::string name(key);
                std::transform(name.begin(), name.end(), name.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (!profile.GetParameters(section).count(name))
                    configured << key << '=' << index->GetParameter(key, section) << '\n';
            }
    }
    configured << "[Benchmark]\nNumSearchThreads=32\nNumInsertThreads=16\n"
        "NumSearchDuringInsertThreads=0\nNumQueries=" << QueryCount << "\nTopK=" << K << '\n';
    configured << "[Lifecycle]\nPivot=" << pivot << "\nInitial=" << initial
        << "\nTotal=" << total << "\nBatch=" << batch << "\nFinal=" << finalCount << '\n';
    configured.close();
    if (mode == "config") return;
    const bool evaluateGrowth = !pilot ||
        std::stoi(index->GetParameter("MergeThreshold", "BuildSSDIndex")) > 10;
    auto vectors = ReadVectors(Environment("SPTAG_SIFT_VECTORS"), total);
    auto queries = ReadVectors(Environment("SPTAG_SIFT_QUERIES"), QueryCount);

    const auto metricsName = mode == "merge-pilot" ? "merge-pilot-metrics.jsonl" :
        mode == "growth-calibration" ? "growth-metrics.jsonl" :
        pilot ? "pilot-metrics.jsonl" : "metrics.jsonl";
    std::ofstream metrics(directory.parent_path() / metricsName);
    BOOST_REQUIRE(metrics);
    const auto started = std::chrono::steady_clock::now();
    int count = initial;
    std::vector<uint8_t> alive(total, 0);
    std::vector<SizeType> sourceByVID(total);
    std::iota(sourceByVID.begin(), sourceByVID.end(), 0);
    const auto saveMapping = [&]() {
        const auto path = directory.parent_path() / "source-by-vid.i32";
        const auto temporary = path.string() + ".partial";
        std::ofstream mapping(temporary, std::ios::binary | std::ios::trunc);
        BOOST_REQUIRE(mapping.write(reinterpret_cast<const char*>(sourceByVID.data()),
            sourceByVID.size() * sizeof(SizeType)));
        mapping.close();
        BOOST_REQUIRE(mapping);
        std::filesystem::rename(temporary, path);
    };
    std::fill(alive.begin(), alive.begin() + initial, 1);
    auto wait = [&]() {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(small ? 60 : 1200);
        while (!index->AllFinished() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        BOOST_REQUIRE_MESSAGE(index->AllFinished(), "Posting/resize maintenance exceeded its deadline");
        BOOST_REQUIRE(index->GetLayerResizeStatus() == ErrorCode::Success);
    };
    auto layers = [&]() { return std::stoi(index->GetParameter("Layers", "BuildSSDIndex")); };
    auto snapshot = [&](const std::string& phase) {
        std::vector<SizeType> routes;
        BOOST_REQUIRE(index->GetHeadIndexMapping(1, routes) == ErrorCode::Success);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto head = index->GetMemoryIndex();
        metrics << "{\"phase\":\"" << phase << "\",\"live\":" << count << ",\"layers\":" << layers()
            << ",\"routes\":" << routes.size() << ",\"head_bytes\":" << index->GetHeadMemoryUsage()
            << ",\"head_slots\":" << head->GetNumSamples() << ",\"head_deleted\":" << head->GetNumDeleted()
            << ",\"elapsed_seconds\":" << seconds << "}\n";
        metrics.flush();
        BOOST_TEST_MESSAGE("SIFT-LIFECYCLE phase=" << phase << " live=" << count << " layers=" << layers()
            << " routes=" << routes.size() << " headBytes=" << index->GetHeadMemoryUsage()
            << " elapsed=" << seconds);
    };
    auto verify = [&](const std::string& phase) {
        std::vector<SizeType> actual;
        BOOST_REQUIRE(index->GetHeadIndexMapping(0, actual) == ErrorCode::Success);
        BOOST_REQUIRE_EQUAL(actual.size(), count);
        std::sort(actual.begin(), actual.end());
        int position = 0;
        for (int id = 0; id < total; ++id)
            if (alive[id]) BOOST_REQUIRE_EQUAL(actual[position++], id);
        std::array<int, QueryCount> boundaries{};
        std::atomic_int nextQuery{0};
        {
            std::vector<std::jthread> workers;
            for (int thread = 0; thread < 32; ++thread) workers.emplace_back([&]() {
                for (int q = nextQuery++; q < QueryCount; q = nextQuery++) {
                    std::vector<int> distances;
                    distances.reserve(count);
                    for (int id = 0; id < total; ++id)
                        if (alive[id]) distances.push_back(SquaredL2(queries.data() + q * Dimension,
                            vectors.data() + static_cast<size_t>(sourceByVID[id]) * Dimension));
                    std::nth_element(distances.begin(), distances.begin() + K - 1, distances.end());
                    boundaries[q] = distances[K - 1];
                }
            });
        }
        std::atomic_int hits{0}, invalidResults{0}, searchErrors{0};
        auto searchPass = [&](bool measured) {
            nextQuery = 0;
            std::vector<std::jthread> workers;
            for (int thread = 0; thread < 32; ++thread) workers.emplace_back([&]() {
                for (int q = nextQuery++; q < QueryCount; q = nextQuery++) {
                    COMMON::QueryResultSet<uint8_t> query(queries.data() + q * Dimension, K);
                    if (index->SearchIndex(query) != ErrorCode::Success) {
                        ++searchErrors;
                        continue;
                    }
                    std::set<SizeType> seen;
                    for (int j = 0; j < K; ++j) {
                        const auto* result = query.GetResult(j);
                        if (result->VID < 0 || result->VID >= total || !alive[result->VID] ||
                            !seen.insert(result->VID).second) {
                            ++invalidResults;
                            continue;
                        }
                        const auto distance = SquaredL2(queries.data() + q * Dimension,
                            vectors.data() + static_cast<size_t>(sourceByVID[result->VID]) * Dimension);
                        if (result->Dist != distance) ++invalidResults;
                        if (measured && distance <= boundaries[q]) ++hits;
                    }
                }
            });
        };
        for (int round = 1; round <= 2; ++round) {
            searchPass(false);
            hits = 0;
            searchPass(true);
            BOOST_REQUIRE_EQUAL(searchErrors.load(), 0);
            BOOST_REQUIRE_EQUAL(invalidResults.load(), 0);
            const double recall = static_cast<double>(hits.load()) / (QueryCount * K);
            metrics << "{\"verify\":\"" << phase << "\",\"round\":" << round << ",\"live\":" << count
                << ",\"queries\":" << QueryCount << ",\"recall_at_5\":" << recall << ",\"deleted_hits\":0}\n";
            metrics.flush();
            BOOST_TEST_MESSAGE("SIFT-LIFECYCLE verify=" << phase << " round=" << round
                << " recallAt5=" << recall << " deletedHits=0");
            BOOST_REQUIRE_GE(recall, 0.95);
        }
        wait();
    };
    BOOST_TEST_MESSAGE("SIFT-LIFECYCLE build initial=" << initial << " high=" << high
        << " target=" << target << " low=" << low);
    BOOST_REQUIRE(index->BuildIndex(vectors.data(), initial, Dimension) == ErrorCode::Success);
    wait();
    snapshot("initial");
    BOOST_REQUIRE_EQUAL(layers(), 1);
    if (evaluateGrowth) verify("initial");
    BOOST_REQUIRE(index->SaveIndex(directory.string()) == ErrorCode::Success);
    std::atomic_int queryErrors{0};
    std::atomic_uint64_t queryCalls{0};
    std::atomic_bool paused{true};
    std::mutex queryMutex;
    const auto pause = [&]() {
        paused = true;
        std::lock_guard<std::mutex> barrier(queryMutex);
    };
    std::jthread reader([&](std::stop_token stop) {
        int q = 0;
        while (!stop.stop_requested()) {
            {
                std::lock_guard<std::mutex> guard(queryMutex);
                if (!paused) {
                    COMMON::QueryResultSet<uint8_t> query(queries.data() + q * Dimension, K);
                    if (index->SearchIndex(query) != ErrorCode::Success) ++queryErrors;
                    ++queryCalls;
                    q = (q + 1) % 10000;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    for (int begin = initial, iteration = 1; begin < total; begin += batch, ++iteration) {
        std::atomic_int nextOffset{0}, insertErrors{0};
        {
            std::vector<std::jthread> inserters;
            for (int thread = 0; thread < 16; ++thread) inserters.emplace_back([&]() {
                for (int offset = nextOffset++; offset < batch; offset = nextOffset++) {
                    SizeType id = -1;
                    if (index->AddIndex(vectors.data() + static_cast<size_t>(begin + offset) * Dimension,
                            1, Dimension, nullptr, false, true, &id) != ErrorCode::Success) {
                        ++insertErrors;
                        continue;
                    }
                    if (id < begin || id >= begin + batch) ++insertErrors;
                    else sourceByVID[id] = begin + offset;
                }
            });
        }
        BOOST_REQUIRE_EQUAL(insertErrors.load(), 0);
        pause();
        wait();
        std::fill(alive.begin() + begin, alive.begin() + begin + batch, 1);
        std::vector<SizeType> assignedSources(sourceByVID.begin() + begin, sourceByVID.begin() + begin + batch);
        std::sort(assignedSources.begin(), assignedSources.end());
        for (int i = 0; i < batch; ++i) BOOST_REQUIRE_EQUAL(assignedSources[i], begin + i);
        count += batch;
        snapshot("insert-" + std::to_string(iteration));
        if (evaluateGrowth && (iteration == pivotBatch || iteration == lastInsertBatch)) {
            saveMapping();
            verify("insert-" + std::to_string(iteration));
            BOOST_REQUIRE(index->SaveIndex(directory.string()) == ErrorCode::Success);
        }
    }
    pause();
    wait();
    if (!pilot) {
        BOOST_REQUIRE_EQUAL(layers(), 2);
        std::vector<SizeType> deletion(total);
        std::iota(deletion.begin(), deletion.end(), 0);
        std::mt19937 random(1729);
        std::shuffle(deletion.begin(), deletion.end(), random);
        std::ofstream deleted(directory.parent_path() / "deleted-ids.i32", std::ios::binary);
        BOOST_REQUIRE(deleted.write(reinterpret_cast<const char*>(deletion.data()), (total - finalCount) * sizeof(SizeType)));
        for (int begin = 0, iteration = 1; begin < total - finalCount; begin += batch, ++iteration) {
            paused = false;
            for (int i = begin; i < begin + batch; ++i) {
                BOOST_REQUIRE(index->DeleteIndex(deletion[i]) == ErrorCode::Success);
                alive[deletion[i]] = 0;
            }
            pause();
            wait();
            count -= batch;
            // Ordinary public searches provide spatially distributed merge opportunities.
            for (int q = 0; q < 1000; ++q) {
                COMMON::QueryResultSet<uint8_t> query(queries.data() + ((q + iteration * 997) % 10000) * Dimension, K);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
            }
            wait();
            snapshot("delete-" + std::to_string(iteration));
            if (count == pivot / scale || count == finalCount) {
                BOOST_REQUIRE(index->SaveIndex(directory.string()) == ErrorCode::Success);
                verify("delete-" + std::to_string(iteration));
            }
        }
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(small ? 180 : 900);
        int sweep = 0;
        while (layers() != 1 && std::chrono::steady_clock::now() < deadline) {
            for (int q = 0; q < 1000; ++q) {
                COMMON::QueryResultSet<uint8_t> query(queries.data() + ((q + sweep * 997) % 10000) * Dimension, K);
                BOOST_REQUIRE(index->SearchIndex(query) == ErrorCode::Success);
            }
            wait();
            snapshot("converge-" + std::to_string(++sweep));
        }
        BOOST_REQUIRE_MESSAGE(layers() == 1, "Random deletion to the final count did not trigger shrink with the frozen memory thresholds");
    }
    reader.request_stop();
    reader.join();
    BOOST_REQUIRE_EQUAL(queryErrors.load(), 0);
    snapshot("complete");
    metrics << "{\"concurrent_queries\":" << queryCalls.load() << ",\"query_errors\":" << queryErrors.load() << "}\n";
    metrics.flush();
    BOOST_REQUIRE(index->SaveIndex(directory.string()) == ErrorCode::Success);
    saveMapping();
    if (!pilot) {
        index.reset();
        std::shared_ptr<VectorIndex> loaded;
        BOOST_REQUIRE(VectorIndex::LoadIndex(directory.string(), loaded) == ErrorCode::Success);
        index = std::dynamic_pointer_cast<SPANN::Index<uint8_t>>(loaded);
        BOOST_REQUIRE(index);
        wait();
        BOOST_REQUIRE_EQUAL(layers(), 1);
        verify("reloaded");
        BOOST_REQUIRE(index->Check() == ErrorCode::Success);
        BOOST_REQUIRE_EQUAL(layers(), 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
