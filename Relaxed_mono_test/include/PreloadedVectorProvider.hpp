#pragma once

#include "VectorProvider.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

class PreloadedVectorProvider : public IVectorProvider {
public:
    using Vector = std::vector<int8_t>;

    PreloadedVectorProvider(const std::string& path, int dim, size_t maxQueueSize = 10000)
        : m_dim(dim), m_fileIndex(0), m_maxQueueSize(maxQueueSize), m_running(true) {
        namespace fs = std::filesystem;
        fs::path basePath(path);

        if (fs::is_regular_file(basePath)) {
            m_files.push_back(basePath);
        } else if (fs::is_directory(basePath)) {
            for (const auto& entry : fs::directory_iterator(basePath)) {
                const auto& filename = entry.path().filename().string();
                if (entry.is_regular_file() && filename.rfind("vectors_", 0) == 0) {
                    m_files.push_back(entry.path());
                }
            }
            std::sort(m_files.begin(), m_files.end());
        } else {
            throw std::runtime_error("Invalid path: " + path);
        }

        if (!openNextFile()) {
            throw std::runtime_error("No valid vector files found in path: " + path);
        }

        m_thread = std::thread(&PreloadedVectorProvider::run, this);
    }

    ~PreloadedVectorProvider() {
        m_running = false;
        m_cv.notify_all();
        if (m_thread.joinable()) {
            m_thread.join();
        }
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
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            // Stop current thread if running
            m_running = false;
            m_cv.notify_all();
        }

        if (m_thread.joinable()) {
            m_thread.join();
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Clear state
            m_queue = {};
            m_fileIndex = 0;
            m_firstFile = true;
            m_vectorsRemainingInFirstFile = 0;
            m_vectorsReadFromCurrentFile = 0;
            m_stream.close();
        }

        // Attempt to reopen the first file
        if (!openNextFile()) {
            std::cerr << "Reset failed: unable to reopen vector files." << std::endl;
            return false;
        }

        // Restart the producer thread
        m_running = true;
        m_thread = std::thread(&PreloadedVectorProvider::run, this);
        return true;
    }

private:
    void run() {
        while (m_running) {
            Vector vec(m_dim);
            if (!readNext(vec)) break;

            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [&] { return m_queue.size() < m_maxQueueSize || !m_running; });

            if (!m_running) break;

            m_queue.push(std::move(vec));
            m_cv.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_running = false;
        }
        m_cv.notify_all();
    }

    bool readNext(Vector& output) {
        output.resize(m_dim);

        while (true) {
            if (!m_stream.is_open()) return false;

            // First file: respect count
            if (m_fileIndex == 1) {
                if (m_vectorsRemainingInFirstFile > 0) {
                    if (m_stream.read(reinterpret_cast<char*>(output.data()), m_dim)) {
                        --m_vectorsRemainingInFirstFile;
                        m_vectorsReadFromCurrentFile++;
                        return true;
                    } else {
                        if (!openNextFile()) return false;
                    }
                } else {
                    if (!openNextFile()) return false;
                }
            }
            // All other files: just read raw vectors
            else {
                if (m_stream.read(reinterpret_cast<char*>(output.data()), m_dim)) {
                    m_vectorsReadFromCurrentFile++;
                    return true;
                } else {
                    if (!openNextFile()) return false;
                }
            }
        }
    }

    bool openNextFile() {
        if (m_fileIndex >= m_files.size()) return false;

        m_stream.close();
        const auto& filePath = m_files[m_fileIndex++];
        m_stream.open(filePath, std::ios::binary);
        if (!m_stream) return false;

        std::cout << "Opening vectors from " << filePath << std::endl;
        m_vectorsReadFromCurrentFile = 0;
        if (m_firstFile) {
            int32_t count = 0, dim = 0;
            if (!m_stream.read(reinterpret_cast<char*>(&count), sizeof(int32_t))) return false;
            if (!m_stream.read(reinterpret_cast<char*>(&dim), sizeof(int32_t))) return false;

            if (dim != m_dim) {
                std::cerr << "Dimension mismatch " << dim << " " << m_dim << std::endl;
                throw std::runtime_error("Dimension mismatch in file: " + filePath.string());
            }

            m_vectorsRemainingInFirstFile = count;
            m_firstFile = false;
        }

        return true;
    }


    std::vector<std::filesystem::path> m_files;
    std::ifstream m_stream;
    int m_dim;
    size_t m_fileIndex;
    size_t m_maxQueueSize;

    std::queue<Vector> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::thread m_thread;
    std::atomic<bool> m_running;
    bool m_firstFile = true;
    int32_t m_vectorsRemainingInFirstFile = 0;
    int32_t m_vectorsReadFromCurrentFile = 0;
};
