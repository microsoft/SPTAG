#include "RelaxedmonoRunner.h"
#include "inc/Core/Common.h"
#include "inc/Core/ResultIterator.h"
#include "inc/Core/SearchResult.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <atomic>
#include <future>
#include <algorithm>

RelaxedmonoRunner::RelaxedmonoRunner(const RelaxedmonoConfig& config)
    : m_config(config), m_running(false), m_totalAppendLatency(0.0), m_appendCount(0),
      m_totalQueryLatency(0.0), m_queryCount(0) {}

RelaxedmonoRunner::~RelaxedmonoRunner() {
    stop();
}

void RelaxedmonoRunner::fast_forward_to_phase_2() {
    std::cout << "Entered fast_forward_to_phase_2 ........." << std::endl;
    VectorGenerator::Vector queryVec;
    while (!m_vectorGen->getNext(queryVec));
    std::shared_ptr<ResultIterator> result_iter = m_index->GetIterator(queryVec.data());

    // The median similarity of all points retrieved in previous call to next();
    float previous_median_distance = std::numeric_limits<float>::max();
    int _num_max_points_to_explore =  m_config.initialCount;
    int _num_points_explored = 0;

    bool in_phase_2 = false;
    int next_batch_size = 10;

    while (!in_phase_2) {

        std::shared_ptr<SPTAG::QueryResult> result = result_iter->Next(next_batch_size);
        const auto num_results = result->GetResultNum();

        std::vector<float> distances;
        distances.reserve(num_results);

        for (uint32_t j = 0; j < num_results; ++j) {
            auto& result_j = *result->GetResult(j);
            // Update counters and datastructures.
            ++_num_points_explored;
            distances.emplace_back(result_j.Dist);
        }

        if (num_results < next_batch_size) {
            // We didn't even find enough points, so definitely not finding any more closer points.
            in_phase_2 = true;
        } else {
            std::sort(distances.begin(), distances.end());
            const float current_median_distance = distances[distances.size() / 2];

            // Phase 1: Explore phase where points discovered might be near or far from query.
            // Phase 2: Exploit phase where newer points discovered will strictly be
            //          further from query than already discovered ones.
            // See VBASE paper for more details.
            if (current_median_distance > previous_median_distance) {
                in_phase_2 = true;
            }

            if (!in_phase_2 && _num_points_explored >= _num_max_points_to_explore) {
                std::cout << "ann_index: Stopping search, assuming more search won't give more matches.";
                in_phase_2 = true;
            }

            previous_median_distance = current_median_distance ;
        }
    }

    std::cout << " _num_points_explored = " << _num_points_explored << std::endl;
    std::cout << " previous_median_distance = " << previous_median_distance << std::endl;
}

void RelaxedmonoRunner::relaxMonoQuery() {
    VectorGenerator::Vector queryVec;
    while (!m_vectorGen->getNext(queryVec));
    std::shared_ptr<ResultIterator> result_iter = m_index->GetIterator(queryVec.data());


    bool _stop_search = false;
    bool relaxedMono_stage = false;
    int infinity_sptag_per_batch_size = 10;
    int _num_max_points_to_explore =  m_config.initialCount;
    int _num_points_explored = 0;

    while (!_stop_search && !relaxedMono_stage) {

        std::shared_ptr<SPTAG::QueryResult> result = result_iter->Next(infinity_sptag_per_batch_size);
        const auto num_results = result->GetResultNum();

        for (uint32_t j = 0; j < num_results; ++j) {
            auto& result_j = *result->GetResult(j);
            if (result_j.RelaxedMono) {
                std::cout << " Relaxed mono is set " << std::endl;
            }
            relaxedMono_stage |= result_j.RelaxedMono;
            _num_points_explored++;
        }

        if (num_results < infinity_sptag_per_batch_size || _num_points_explored >= _num_max_points_to_explore) {
            std::cout << "ann_index: Stopping search, assuming more search won't give more matches.";
            std::cout << " num_results = " << num_results << std::endl;
            std::cout << " infinity_sptag_per_batch_size = " << infinity_sptag_per_batch_size << std::endl;
            std::cout << " _num_points_explored = " << _num_points_explored << std::endl;
            _stop_search = true;
        }
    }
    std::cout << " relaxedMono_stage = " << relaxedMono_stage << std::endl;
    std::cout << " _num_points_explored = " << _num_points_explored << std::endl;
}
void RelaxedmonoRunner::run() {
    std::cout << "=== Relaxedmono Starting ===\n";
    buildInitialIndex();

    m_vectorGen = std::make_unique<VectorGenerator>(m_config.dimensions);
    m_vectorGen->start();

    m_running = true;
    relaxMonoQuery();
    fast_forward_to_phase_2();
    return;
}

void RelaxedmonoRunner::stop() {
    m_running = false;
    if (m_vectorGen) m_vectorGen->stop();
    if (m_appendThread.joinable()) m_appendThread.join();
    for (auto& t : m_queryThreads) {
        if (t.joinable()) t.join();
    }
}

void RelaxedmonoRunner::buildInitialIndex() {
    using T = int8_t;
    int dim = m_config.dimensions;
    int count = m_config.initialCount;

    std::vector<T> data(count * dim);
    std::generate(data.begin(), data.end(), [] { return static_cast<T>(rand() % 256); });

    std::vector<uint8_t> meta;
    std::vector<uint64_t> offsets;
    for (int i = 0; i < count; ++i) {
        offsets.push_back(meta.size());
        for (int j = 0; j < 4; ++j)
            meta.push_back(static_cast<uint8_t>((i >> (j * 8)) & 0xFF));
    }
    offsets.push_back(meta.size());

    auto vecSet = std::make_shared<SPTAG::BasicVectorSet>(
        SPTAG::ByteArray(reinterpret_cast<uint8_t*>(data.data()), data.size(), false),
        SPTAG::GetEnumValueType<T>(), dim, count);

    auto metaSet = std::make_shared<SPTAG::MemMetadataSet>(
        SPTAG::ByteArray(meta.data(), meta.size(), false),
        SPTAG::ByteArray(reinterpret_cast<uint8_t*>(offsets.data()), offsets.size() * sizeof(uint64_t), false),
        count);

    //auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN, SPTAG::GetEnumValueType<T>());
    SPTAG::IndexAlgoType algoType = SPTAG::IndexAlgoType::SPANN;
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(m_config.indexAlgoType.c_str(), algoType);
    auto index = SPTAG::VectorIndex::CreateInstance(algoType, SPTAG::GetEnumValueType<T>());
    configureIndexParameters(index);

    auto buildStart = std::chrono::high_resolution_clock::now();
    bool buildSuccess = index->BuildIndex(vecSet, metaSet) == SPTAG::ErrorCode::Success;
    auto buildEnd = std::chrono::high_resolution_clock::now();

    if (!buildSuccess) {
        std::cerr << "Failed to build index\n";
    } else {
        std::cout << "Initial index built with " << count << " vectors.\n";
    }

    auto saveStart = std::chrono::high_resolution_clock::now();
    bool saveSuccess = index->SaveIndex(m_config.indexPath) == SPTAG::ErrorCode::Success;
    auto saveEnd = std::chrono::high_resolution_clock::now();

    if (!saveSuccess) {
        std::cerr << "Failed to save index\n";
    } else {
        std::cout << "Index saved\n";
    }

    index.reset();

    auto loadStart = std::chrono::high_resolution_clock::now();
    bool loadSuccess = SPTAG::VectorIndex::LoadIndex(m_config.indexPath, m_index) == SPTAG::ErrorCode::Success && m_index != nullptr;
    auto loadEnd = std::chrono::high_resolution_clock::now();

    if (!loadSuccess) {
        std::cerr << "Failed to reload index from " << m_config.indexPath << "\n";
    } else {
        std::cout << "Index reloaded from " << m_config.indexPath << "\n";
    }

    auto buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(buildEnd - buildStart).count();
    auto saveTime = std::chrono::duration_cast<std::chrono::milliseconds>(saveEnd - saveStart).count();
    auto loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();

    std::cout << "Timing Summary (in milliseconds):\n"
              << "Build Time: " << buildTime << " ms\n"
              << "Save Time:  " << saveTime << " ms\n"
              << "Load Time:  " << loadTime << " ms\n";
}


void RelaxedmonoRunner::configureIndexParameters(std::shared_ptr<SPTAG::VectorIndex>& index) const {
    index->SetParameter("IndexAlgoType", m_config.indexAlgoType, "Base");
    index->SetParameter("DistCalcMethod", m_config.distCalcMethod, "Base");

    index->SetParameter("isExecute", "true", "SelectHead");
    index->SetParameter("NumberOfThreads", std::to_string(m_config.selectHeadThreads), "SelectHead");
    index->SetParameter("Ratio", std::to_string(m_config.selectHeadRatio), "SelectHead");

    index->SetParameter("isExecute", "true", "BuildHead");
    index->SetParameter("RefineIterations", std::to_string(m_config.refineIterations), "BuildHead");
    index->SetParameter("NumberOfThreads", std::to_string(m_config.buildHeadThreads), "BuildHead");

    index->SetParameter("UseKV", m_config.useKV ? "true" : "false", "BuildSSDIndex");
    index->SetParameter("KVPath", m_config.kvPath, "BuildSSDIndex");
    index->SetParameter("SsdInfoFile", m_config.ssdInfoFile, "BuildSSDIndex");
    index->SetParameter("isExecute", "true", "BuildSSDIndex");
    index->SetParameter("BuildSsdIndex", m_config.buildSSDIndex ? "true" : "false", "BuildSSDIndex");
    index->SetParameter("NumberOfThreads", std::to_string(m_config.buildSSDIndexThreads), "BuildSSDIndex");
    index->SetParameter("PostingPageLimit", std::to_string(m_config.postingPageLimit), "BuildSSDIndex");
    index->SetParameter("SearchPostingPageLimit", std::to_string(m_config.searchPostingPageLimit), "BuildSSDIndex");
    index->SetParameter("InternalResultNum", std::to_string(m_config.internalResultNum), "BuildSSDIndex");
    index->SetParameter("SearchInternalResultNum", std::to_string(m_config.searchInternalResultNum), "BuildSSDIndex");
}

