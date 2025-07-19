#include "RecallProvider.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

class Space1vbRecallProvider : public IRecallProvider {
public:
    Space1vbRecallProvider(const std::string& basePath) : m_basePath(basePath) {}

    bool load(std::vector<int8_t>& queryVectors, std::vector<int32_t>& groundTruth, int& queryCount, int& dim, int& topk) override {
        std::ifstream fq(m_basePath + "/query.bin", std::ios::binary);
        if (!fq) {
            std::cerr << "Failed to open query.bin\n";
            return false;
        }
        int32_t qc, d;
        fq.read(reinterpret_cast<char*>(&qc), sizeof(int32_t));
        fq.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        queryVectors.resize(qc * d);
        fq.read(reinterpret_cast<char*>(queryVectors.data()), qc * d);

        std::cout << "[Recall] Loaded " << qc << " queries of dimension " << d << ". Printing first 2 queries:" << std::endl;
        for (int i = 0; i < std::min(2, qc); ++i) {
            std::cout << "Query " << i << ": ";
            for (int j = 0; j < std::min(d, 16); ++j) {
                std::cout << static_cast<int>(queryVectors[i * d + j]) << " ";
            }
            std::cout << "..." << std::endl;
        }
        fq.close();

        std::ifstream ft(m_basePath + "/truth.bin", std::ios::binary);
        if (!ft) {
            std::cerr << "Failed to open truth.bin\n";
            return false;
        }
        int32_t gtCount;
        ft.read(reinterpret_cast<char*>(&gtCount), sizeof(int32_t));
        ft.read(reinterpret_cast<char*>(&topk), sizeof(int32_t));
        if (gtCount != qc) {
            std::cerr << "Mismatch between query count and ground truth count\n";
            return false;
        }
        std::cout << "[Recall] Loading truth.bin with " << gtCount << " entries and topk = " << topk << std::endl;
        groundTruth.resize(qc * topk);
        std::cout << "[Recall] Ground truth size: " << qc << " x " << topk << " entries" << std::endl;
        std::vector<float> distances(qc * topk);
        ft.read(reinterpret_cast<char*>(groundTruth.data()), groundTruth.size() * sizeof(int32_t));

        std::cout << "[Recall] Sample Ground Truth for first 2 queries:" << std::endl;
        for (int i = 0; i < std::min(2, qc); ++i) {
            std::cout << "GT[" << i << "]: ";
            for (int j = 0; j < std::min(10, topk); ++j) {
                std::cout << groundTruth[i * topk + j] << " ";
            }
            std::cout << "..." << std::endl;
        }
        ft.read(reinterpret_cast<char*>(distances.data()), distances.size() * sizeof(float));
        ft.close();

        std::cout << "[Recall] Successfully loaded ground truth for " << qc << " queries." << std::endl;

        queryCount = qc;
        dim = d;
        return true;
    }


private:
    std::string m_basePath;
};