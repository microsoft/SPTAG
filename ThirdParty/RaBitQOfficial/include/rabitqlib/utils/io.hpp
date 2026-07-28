#pragma once

#include <sys/stat.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace rabitqlib {
// get num of bytes
inline size_t get_filesize(const char* filename) noexcept {
    try {
        return std::filesystem::file_size(filename);
    } catch (const std::filesystem::filesystem_error& e) {
        // Log the error and return -1 to maintain original behavior on error.
        std::cerr << "Error getting file size for '" << filename << "': " << e.what()
                  << '\n';
        return static_cast<size_t>(-1);
    }
}

inline bool file_exists(const char* filename) { return std::filesystem::exists(filename); }

// load .*vecs file to a matrix (e.g., RowMajorFloatMat)
template <typename T, class M>
void load_vecs(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        exit(1);
    }

    assert((std::is_same_v<T*, std::decay_t<decltype(row_mat.data())>> == true));

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));

    size_t cols = tmp;
    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    row_mat = M(rows, cols);

    input.seekg(0, std::ifstream::beg);

    for (size_t i = 0; i < rows; i++) {
        input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&row_mat(i, 0)), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}

// load .*bin file to a matrix (e.g., RowMajorFloatMat)
template <typename T, class M>
void load_bin(const char* filename, M& row_mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        exit(1);
    }

    assert((std::is_same_v<T*, std::decay_t<decltype(row_mat.data())>> == true));

    uint32_t rows;
    uint32_t cols;
    std::ifstream input(filename, std::ios::binary);

    input.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

    row_mat = M(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        input.read(reinterpret_cast<char*>(&row_mat(i, 0)), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}
}  // namespace rabitqlib