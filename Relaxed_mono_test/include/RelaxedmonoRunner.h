#pragma once

#include "Config.h"
#include "VectorGenerator.h"

#include "inc/Core/SPANN/Index.h"

#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>

class RelaxedmonoRunner {
public:
    explicit RelaxedmonoRunner(const RelaxedmonoConfig& config);
    ~RelaxedmonoRunner();

    void run();

private:
    void stop();
    void relaxMonoQuery();
    void fast_forward_to_phase_2();
    void buildInitialIndex();
    void configureIndexParameters(std::shared_ptr<SPTAG::VectorIndex>& index) const;

    void appendLoop();
    void queryWorker();

    void printMetrics() const;
    void printFinalMetrics() const;

    RelaxedmonoConfig m_config;
    std::shared_ptr<SPTAG::VectorIndex> m_index;
    std::unique_ptr<VectorGenerator> m_vectorGen;

    std::atomic<bool> m_running;
    std::thread m_appendThread;
    std::vector<std::thread> m_queryThreads;

    mutable std::mutex m_metricsMutex;
    double m_totalAppendLatency;
    double m_totalQueryLatency;
    std::vector<double> m_appendLatencies;
    std::vector<double> m_queryLatencies;
    int m_appendCount;
    int m_queryCount;
};
