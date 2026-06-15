// Offline encoder for REAL (extended) RaBitQ, flat single-centroid, split layout
// (1-bit packed binary code + ex-bits), consumed by SPTAG SPANN::RaBitQ2.
//
// Reads opq_vectors.bin (header int32 N, int32 dim; then N*dim float32 in vid order),
// cosine-normalizes each vector, computes a global centroid (mean of normalized),
// rotates data+centroid with a FhtKacRotator, and encodes with quantize_split_single.
//
// Output rabitq2.bin layout:
//   int32 magic = 0x52425132 ('RBQ2')
//   int32 N, dim, padded_dim, ex_bits, rotator_type(1=FhtKac)
//   int32 rotator_bytes; [rotator dump]
//   float32 centroid_rot[padded_dim]
//   per vec: bin[binBytes] + ex[exBytes]   (binBytes=padded/8+12, exBytes=ex_bits*padded/8+8)
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/rotator.hpp"

using namespace rabitqlib;

static void normalize(float* v, int dim) {
    double n = 0; for (int i = 0; i < dim; i++) n += (double)v[i] * v[i];
    n = std::sqrt(n); if (n > 0) { float inv = (float)(1.0 / n); for (int i = 0; i < dim; i++) v[i] *= inv; }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: rabitq2_encode opq_vectors.bin rabitq2.bin total_bits\n";
        return 1;
    }
    const char* inPath = argv[1];
    const char* outPath = argv[2];
    int total_bits = atoi(argv[3]);
    if (total_bits < 1) { std::cerr << "total_bits must be >= 1\n"; return 1; }
    size_t ex_bits = total_bits - 1;

    std::ifstream in(inPath, std::ios::binary);
    if (!in) { std::cerr << "cannot open " << inPath << "\n"; return 1; }
    int32_t hdr[2] = {0, 0};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    int N = hdr[0], dim = hdr[1];
    std::cout << "input N=" << N << " dim=" << dim << " total_bits=" << total_bits << "\n";

    std::vector<float> data((size_t)N * dim);
    in.read(reinterpret_cast<char*>(data.data()), (size_t)N * dim * sizeof(float));
    if (!in) { std::cerr << "truncated input\n"; return 1; }

    const char* nnEnv = std::getenv("SPTAG_RABITQ_NONORM");
    const bool noNorm = nnEnv && nnEnv[0] == '1';
    std::cout << "normalize=" << (noNorm ? "OFF (L2)" : "ON (cosine)") << "\n";
    if (!noNorm) for (int i = 0; i < N; i++) normalize(&data[(size_t)i * dim], dim);
    std::vector<float> centroid(dim, 0);
    for (int i = 0; i < N; i++) { const float* v = &data[(size_t)i * dim]; for (int d = 0; d < dim; d++) centroid[d] += v[d]; }
    for (int d = 0; d < dim; d++) centroid[d] /= (float)N;

    size_t padded = round_up_to_multiple((size_t)dim, 64);
    Rotator<float>* rot = choose_rotator<float>(dim, RotatorType::FhtKacRotator, padded);
    padded = rot->size();
    std::cout << "padded_dim=" << padded << " ex_bits=" << ex_bits << "\n";

    std::vector<float> centroid_rot(padded);
    rot->rotate(centroid.data(), centroid_rot.data());

    size_t binBytes = BinDataMap<float>::data_bytes(padded);
    size_t exBytes = ExDataMap<float>::data_bytes(padded, ex_bits);
    std::cout << "binBytes=" << binBytes << " exBytes=" << exBytes
              << " bytes/vec=" << (binBytes + exBytes)
              << " storeMB=" << (double)N * (binBytes + exBytes) / 1e6 << "\n";

    quant::RabitqConfig cfg;  // default (accurate) path, matches HNSW non-faster

    std::ofstream out(outPath, std::ios::binary);
    if (!out) { std::cerr << "cannot open " << outPath << "\n"; return 1; }
    int32_t magic = 0x52425132;
    int32_t rtype = 1;
    int32_t pdim32 = (int32_t)padded, ex32 = (int32_t)ex_bits;
    out.write((char*)&magic, 4);
    out.write((char*)&N, 4);
    out.write((char*)&dim, 4);
    out.write((char*)&pdim32, 4);
    out.write((char*)&ex32, 4);
    out.write((char*)&rtype, 4);
    std::vector<char> rotBytes(rot->dump_bytes());
    rot->save(rotBytes.data());
    int32_t rb = (int32_t)rotBytes.size();
    out.write((char*)&rb, 4);
    out.write(rotBytes.data(), rotBytes.size());
    out.write((char*)centroid_rot.data(), padded * sizeof(float));

    std::vector<char> bin(binBytes), ex(exBytes ? exBytes : 1);
    std::vector<float> rd(padded);
    for (int i = 0; i < N; i++) {
        rot->rotate(&data[(size_t)i * dim], rd.data());
        quant::quantize_split_single(rd.data(), centroid_rot.data(), padded, ex_bits,
            bin.data(), exBytes ? ex.data() : nullptr, METRIC_L2, cfg);
        out.write(bin.data(), binBytes);
        if (exBytes) out.write(ex.data(), exBytes);
    }
    out.close();
    std::cout << "wrote " << outPath << "\n";
    delete rot;
    return 0;
}
