#include "VectorGenerator.h"
#include <random>

VectorGenerator::VectorGenerator(int dim, int queueLimit)
    : m_dim(dim), m_queueLimit(queueLimit), m_running(false) {}

VectorGenerator::~VectorGenerator() {
    stop();
}

void VectorGenerator::start() {
    m_running = true;
    m_thread = std::thread(&VectorGenerator::run, this);
}

void VectorGenerator::stop() {
    m_running = false;
    m_cv.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

bool VectorGenerator::getNext(Vector& output) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [&] { return !m_queue.empty() || !m_running; });

    if (m_queue.empty()) return false;

    output = std::move(m_queue.front());
    m_queue.pop();
    m_cv.notify_one();
    return true;
}

void VectorGenerator::run() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

    while (m_running) {
        Vector vec(m_dim);
        for (auto& v : vec) {
            v = static_cast<int8_t>(dist(rng));
        }

        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [&] { return m_queue.size() < static_cast<size_t>(m_queueLimit) || !m_running; });

            if (!m_running) break;

            m_queue.push(std::move(vec));
        }

        m_cv.notify_one();
    }
}
