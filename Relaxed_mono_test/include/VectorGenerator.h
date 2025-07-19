#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

class VectorGenerator {
public:
    using Vector = std::vector<int8_t>;

    VectorGenerator(int dim, int queueLimit = 1000);
    ~VectorGenerator();

    void start();
    void stop();
    bool getNext(Vector& output);

private:
    void run();

    int m_dim;
    int m_queueLimit;

    std::thread m_thread;
    std::queue<Vector> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::atomic<bool> m_running;
};
