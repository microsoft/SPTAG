// Streaming, billion-scale, value-type-aware offline encoder for REAL (extended)
// RaBitQ, flat single-centroid, split layout (1-bit packed binary code + ex-bits),
// consumed by SPTAG SPANN::RaBitQ2 (in-posting code sidecar).
//
// This is the 1B-safe analog of Tools/rabitq2_encode.cpp: instead of loading the
// full N*dim float matrix into RAM (400GB at 1B), it streams the base vector file
// in chunks and makes two passes:
//   pass 1: accumulate the global centroid (mean of cosine-normalized vectors)
//   pass 2: rotate + quantize each vector, append [bin|ex] codes in vid order
//
// Input base file (raw .*bin): header int32 N, int32 dim; then N*dim elements of
// the given value type (Int8/UInt8/Float32) in vid order. Matches spacev1b_base.i8bin.
//
// Output rabitq2.bin layout (identical to rabitq2_encode):
//   int32 magic = 0x52425132 ('RBQ2')
//   int32 N, dim, padded_dim, ex_bits, rotator_type(1=FhtKac)
//   int32 rotator_bytes; [rotator dump]
//   float32 centroid_rot[padded_dim]
//   per vec: bin[binBytes] + ex[exBytes]
//
// Usage: rabitq2_encode_stream <base.bin> <rabitq2.bin> <total_bits> <Int8|UInt8|Float>
//        [chunk_vectors]
// Env: SPTAG_RABITQ_NONORM=1 -> skip cosine normalization (L2 mode).
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/utils/rotator.hpp"

using namespace rabitqlib;

namespace {

size_t ElemSize(const std::string& vt) {
    if (vt == "Float" || vt == "float" || vt == "Float32") return 4;
    if (vt == "Int16" || vt == "int16") return 2;
    if (vt == "Int8" || vt == "int8" || vt == "UInt8" || vt == "uint8") return 1;
    return 0;
}

// Convert one raw element block of the given value type into float (in place buffer).
void ToFloat(const std::string& vt, const void* raw, float* out, int dim) {
    if (vt == "Float" || vt == "float" || vt == "Float32") {
        std::memcpy(out, raw, (size_t)dim * sizeof(float));
    } else if (vt == "Int8" || vt == "int8") {
        const int8_t* p = reinterpret_cast<const int8_t*>(raw);
        for (int d = 0; d < dim; d++) out[d] = (float)p[d];
    } else if (vt == "UInt8" || vt == "uint8") {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(raw);
        for (int d = 0; d < dim; d++) out[d] = (float)p[d];
    } else if (vt == "Int16" || vt == "int16") {
        const int16_t* p = reinterpret_cast<const int16_t*>(raw);
        for (int d = 0; d < dim; d++) out[d] = (float)p[d];
    }
}

void Normalize(float* v, int dim) {
    double n = 0; for (int i = 0; i < dim; i++) n += (double)v[i] * v[i];
    n = std::sqrt(n); if (n > 0) { float inv = (float)(1.0 / n); for (int i = 0; i < dim; i++) v[i] *= inv; }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: rabitq2_encode_stream <base.bin> <rabitq2.bin> <total_bits> "
                     "<Int8|UInt8|Float> [chunk_vectors]\n";
        return 1;
    }
    const std::string inPath = argv[1];
    const std::string outPath = argv[2];
    const int total_bits = atoi(argv[3]);
    const std::string vt = argv[4];
    const size_t chunk = (argc > 5) ? (size_t)atoll(argv[5]) : 1000000;
    if (total_bits < 1) { std::cerr << "total_bits must be >= 1\n"; return 1; }
    const size_t ex_bits = (size_t)(total_bits - 1);
    const size_t esz = ElemSize(vt);
    if (esz == 0) { std::cerr << "bad value type " << vt << "\n"; return 1; }

    std::ifstream in(inPath, std::ios::binary);
    if (!in) { std::cerr << "cannot open " << inPath << "\n"; return 1; }
    int32_t hdr[2] = {0, 0};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    const int N = hdr[0], dim = hdr[1];
    const size_t rowBytes = (size_t)dim * esz;
    const std::streamoff dataStart = sizeof(hdr);
    std::cout << "input N=" << N << " dim=" << dim << " vt=" << vt
              << " total_bits=" << total_bits << " chunk=" << chunk << "\n";

    const char* nnEnv = std::getenv("SPTAG_RABITQ_NONORM");
    const bool noNorm = nnEnv && nnEnv[0] == '1';
    std::cout << "normalize=" << (noNorm ? "OFF (L2)" : "ON (cosine)") << "\n";

    std::vector<char> rawBuf(chunk * rowBytes);
    std::vector<float> fbuf(chunk * (size_t)dim);

    // ---- pass 1: centroid (mean of normalized vectors) ----
    std::vector<double> cacc(dim, 0.0);
    {
        in.clear(); in.seekg(dataStart, std::ios::beg);
        size_t done = 0;
        while (done < (size_t)N) {
            size_t m = std::min(chunk, (size_t)N - done);
            in.read(rawBuf.data(), (std::streamsize)(m * rowBytes));
            if (!in) { std::cerr << "truncated input (pass1)\n"; return 1; }
            for (size_t i = 0; i < m; i++) {
                float* v = &fbuf[i * dim];
                ToFloat(vt, rawBuf.data() + i * rowBytes, v, dim);
                if (!noNorm) Normalize(v, dim);
                for (int d = 0; d < dim; d++) cacc[d] += v[d];
            }
            done += m;
        }
    }
    std::vector<float> centroid(dim);
    for (int d = 0; d < dim; d++) centroid[d] = (float)(cacc[d] / (double)N);

    size_t padded = round_up_to_multiple((size_t)dim, 64);
    Rotator<float>* rot = choose_rotator<float>(dim, RotatorType::FhtKacRotator, padded);
    padded = rot->size();
    std::vector<float> centroid_rot(padded);
    rot->rotate(centroid.data(), centroid_rot.data());

    size_t binBytes = BinDataMap<float>::data_bytes(padded);
    size_t exBytes = ExDataMap<float>::data_bytes(padded, ex_bits);
    std::cout << "padded_dim=" << padded << " ex_bits=" << ex_bits
              << " binBytes=" << binBytes << " exBytes=" << exBytes
              << " bytes/vec=" << (binBytes + exBytes)
              << " storeGB=" << (double)N * (binBytes + exBytes) / 1e9 << "\n";

    quant::RabitqConfig cfg;

    std::ofstream out(outPath, std::ios::binary);
    if (!out) { std::cerr << "cannot open " << outPath << "\n"; return 1; }
    int32_t magic = 0x52425132, rtype = 1, pdim32 = (int32_t)padded, ex32 = (int32_t)ex_bits, N32 = N, dim32 = dim;
    out.write((char*)&magic, 4);
    out.write((char*)&N32, 4);
    out.write((char*)&dim32, 4);
    out.write((char*)&pdim32, 4);
    out.write((char*)&ex32, 4);
    out.write((char*)&rtype, 4);
    std::vector<char> rotBytes(rot->dump_bytes());
    rot->save(rotBytes.data());
    int32_t rb = (int32_t)rotBytes.size();
    out.write((char*)&rb, 4);
    out.write(rotBytes.data(), rotBytes.size());
    out.write((char*)centroid_rot.data(), padded * sizeof(float));

    // ---- pass 2: rotate + quantize, stream codes in vid order ----
    std::vector<char> bin(binBytes), exc(exBytes ? exBytes : 1);
    std::vector<float> rd(padded);
    std::vector<char> outBuf;
    {
        in.clear(); in.seekg(dataStart, std::ios::beg);
        size_t done = 0;
        while (done < (size_t)N) {
            size_t m = std::min(chunk, (size_t)N - done);
            in.read(rawBuf.data(), (std::streamsize)(m * rowBytes));
            if (!in) { std::cerr << "truncated input (pass2)\n"; return 1; }
            outBuf.clear();
            outBuf.reserve(m * (binBytes + exBytes));
            for (size_t i = 0; i < m; i++) {
                float* v = &fbuf[i * dim];
                ToFloat(vt, rawBuf.data() + i * rowBytes, v, dim);
                if (!noNorm) Normalize(v, dim);
                rot->rotate(v, rd.data());
                quant::quantize_split_single(rd.data(), centroid_rot.data(), padded, ex_bits,
                    bin.data(), exBytes ? exc.data() : nullptr, METRIC_L2, cfg);
                outBuf.insert(outBuf.end(), bin.begin(), bin.end());
                if (exBytes) outBuf.insert(outBuf.end(), exc.begin(), exc.end());
            }
            out.write(outBuf.data(), (std::streamsize)outBuf.size());
            done += m;
            if ((done / chunk) % 10 == 0 || done == (size_t)N)
                std::cout << "  encoded " << done << "/" << N << "\r" << std::flush;
        }
    }
    std::cout << "\nwrote " << outPath << "\n";
    out.close();
    delete rot;
    return 0;
}
