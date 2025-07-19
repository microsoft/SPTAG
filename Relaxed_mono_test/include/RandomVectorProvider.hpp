#pragma once

#include "VectorProvider.h"

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <random>
#include <atomic>

class RandomVectorProvider : public IVectorProvider {
public:
    using Vector = std::vector<int8_t>;

    RandomVectorProvider(int dim, size_t maxQueueSize = 100)
        : m_dim(dim), m_maxQueueSize(maxQueueSize), m_running(true) {
        m_thread = std::thread(&RandomVectorProvider::run, this);
    }

    ~RandomVectorProvider() {
        stop();
    }

    bool getNext(Vector& output) override {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&] { return !m_queue.empty() || !m_running; });

        if (m_queue.empty()) return false;

        output = std::move(m_queue.front());
        m_queue.pop();
        m_cv.notify_one();
        return true;
    }

    bool reset() override {
        return false;
    }

private:
    void run() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 255);

        while (m_running) {
            Vector vec(m_dim);
            for (auto& v : vec) {
                v = static_cast<int8_t>(dist(rng));
            }

            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [&] { return m_queue.size() < m_maxQueueSize || !m_running; });

            if (!m_running) break;

            m_queue.push(std::move(vec));
            m_cv.notify_one();
        }
    }

    void stop() {
        m_running = false;
        m_cv.notify_all();
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    int m_dim;
    size_t m_maxQueueSize;
    std::queue<Vector> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::thread m_thread;
    std::atomic<bool> m_running;
};
