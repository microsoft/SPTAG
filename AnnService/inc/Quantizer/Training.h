#pragma once
#include <inc/Core/Common/DistanceUtils.h>
#include <inc/Core/Common/IQuantizer.h>
#include <inc/Core/Common/PQQuantizer.h>
#ifdef RABITQ
#include <inc/Core/Common/RaBitQQuantizer.h>
#endif
#include <inc/Helper/StringConvert.h>
#include <memory>
#include <string>
#include <inc/Core/VectorSet.h>
#include "inc/Core/Common/BKTree.h"

#define blockRows 4096
using namespace SPTAG;

class QuantizerOptions : public Helper::ReaderOptions
{
public:
    QuantizerOptions(SizeType trainingSamples, bool debug, float lambda, SPTAG::QuantizerType qtype, std::string qfile, DimensionType qdim, std::string fullvecs, std::string recvecs, SPTAG::DistCalcMethod distCalcMethod = SPTAG::DistCalcMethod::L2, std::string rabitqMode = "exact") : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32), m_trainingSamples(trainingSamples), m_debug(debug), m_KmeansLambda(lambda), m_quantizerType(qtype), m_outputQuantizerFile(qfile), m_quantizedDim(qdim), m_outputFullVecFile(fullvecs), m_outputReconstructVecFile(recvecs), m_distCalcMethod(distCalcMethod), m_rabitqMode(std::move(rabitqMode))
    {
        AddRequiredOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_outputFile, "-o", "--output", "Output quantized vectors.");
        AddOptionalOption(m_outputMetadataFile, "-om", "--outputmeta", "Output metadata.");
        AddOptionalOption(m_outputMetadataIndexFile, "-omi", "--outputmetaindex", "Output metadata index.");
        AddOptionalOption(m_outputQuantizerFile, "-oq", "--outputquantizer", "Output quantizer.");
        AddOptionalOption(m_quantizerType, "-qt", "--quantizer", "Quantizer type.");
        AddOptionalOption(m_quantizedDim, "-qd", "--quantizeddim", "Quantized Dimension.");
        AddOptionalOption(m_distCalcMethod, "-m", "--dist", "Distance method (L2 or Cosine).");
        AddOptionalOption(m_rabitqMode, "-rqm", "--rabitq_mode", "RaBitQ quantization mode (exact or fast).");

        // We also use this to determine batch size (max number of vectors to load at once)
        AddOptionalOption(m_trainingSamples, "-ts", "--train_samples", "Number of samples for training.");
        AddOptionalOption(m_debug, "-debug", "--debug", "Print debug information.");
        AddOptionalOption(m_KmeansLambda, "-kml", "--lambda", "Kmeans lambda parameter.");
        AddOptionalOption(m_outputFullVecFile, "-ofv", "--output_full", "Output Uncompressed vectors.");
        AddOptionalOption(m_outputFullVecFile, "-orv", "--output_reconstruct", "Output reconstructed vectors.");
    }

    ~QuantizerOptions() {}

    std::string m_inputFiles;

    std::string m_outputFile;

    std::string m_outputFullVecFile;

    std::string m_outputReconstructVecFile;

    std::string m_outputMetadataFile;

    std::string m_outputMetadataIndexFile;

    std::string m_outputQuantizerFile;

    DimensionType m_quantizedDim;

    SizeType m_trainingSamples;

    SPTAG::QuantizerType m_quantizerType;

    bool m_debug;

    float m_KmeansLambda;

    SPTAG::DistCalcMethod m_distCalcMethod;

    std::string m_rabitqMode;

    bool m_hasExplicitQuantizedDim = false;
};

inline bool TryParseRaBitQQuantizationMode(
    const std::string& value,
    COMMON::RaBitQQuantizer::QuantizationMode& mode)
{
    if (Helper::StrUtils::StrEqualIgnoreCase(value.c_str(), "exact")) {
        mode = COMMON::RaBitQQuantizer::QuantizationMode::Exact;
        return true;
    }
    if (Helper::StrUtils::StrEqualIgnoreCase(value.c_str(), "fast")) {
        mode = COMMON::RaBitQQuantizer::QuantizationMode::Fast;
        return true;
    }
    return false;
}

inline bool ValidateLoadedRaBitQQuantizer(
    const std::shared_ptr<COMMON::RaBitQQuantizer>& quantizer,
    const std::shared_ptr<QuantizerOptions>& options,
    DimensionType inputDimension)
{
    if (!quantizer || !options || inputDimension <= 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Invalid RaBitQ validation inputs.\n");
        return false;
    }

    COMMON::RaBitQQuantizer::QuantizationMode expectedMode;
    if (!TryParseRaBitQQuantizationMode(options->m_rabitqMode, expectedMode)) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "RaBitQ quantization mode must be either exact or fast.\n");
        return false;
    }

    if (quantizer->Dimension() != inputDimension) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Loaded RaBitQ dimension %d does not match input dimension %d.\n",
            quantizer->Dimension(),
            inputDimension);
        return false;
    }
    if (options->m_hasExplicitQuantizedDim) {
        const int expectedBits = options->m_quantizedDim > 0 ? options->m_quantizedDim : 2;
        if (quantizer->Bits() != expectedBits) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Loaded RaBitQ bits %d do not match requested bits %d.\n",
                quantizer->Bits(),
                expectedBits);
            return false;
        }
    }
    if (quantizer->GetMetric() != options->m_distCalcMethod) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Loaded RaBitQ metric %s does not match requested metric %s.\n",
            Helper::Convert::ConvertToString(quantizer->GetMetric()).c_str(),
            Helper::Convert::ConvertToString(options->m_distCalcMethod).c_str());
        return false;
    }
    if (quantizer->GetQuantizationMode() != expectedMode) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Loaded RaBitQ mode does not match requested mode %s.\n",
            options->m_rabitqMode.c_str());
        return false;
    }

    return true;
}

inline void CaptureExplicitQuantizerOptions(
    const std::shared_ptr<QuantizerOptions>& options,
    int argc,
    char* argv[])
{
    if (!options || argv == nullptr) {
        return;
    }

    for (int i = 0; i < argc; ++i) {
        const std::string current(argv[i] == nullptr ? "" : argv[i]);
        if (current == "-qd" || current == "--quantizeddim") {
            options->m_hasExplicitQuantizedDim = true;
        }
    }
}

inline bool ShouldOuterNormalizeForQuantizeAndSave(
    const std::shared_ptr<QuantizerOptions>& options,
    const std::shared_ptr<COMMON::IQuantizer>& quantizer)
{
    if (std::dynamic_pointer_cast<COMMON::RaBitQQuantizer>(quantizer)) {
        return false;
    }

    return options != nullptr && options->m_normalized;
}

template <typename T>
std::unique_ptr<T[]> TrainPQQuantizer(std::shared_ptr<QuantizerOptions> options, std::shared_ptr<VectorSet> raw_vectors, std::shared_ptr<VectorSet> quantized_vectors)
{
    SizeType numCentroids = 256;
    if (raw_vectors->Dimension() % options->m_quantizedDim != 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Only n_codebooks that divide dimension are supported.\n");
        return nullptr;
    }
    DimensionType subdim = raw_vectors->Dimension() / options->m_quantizedDim;
    auto codebooks = std::make_unique<T[]>(numCentroids * raw_vectors->Dimension());

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Training Quantizer Codebooks.\n");
    std::vector<std::thread> mythreads;
    mythreads.reserve(options->m_threadNum);
    std::atomic_size_t sent(0);
    for (int tid = 0; tid < options->m_threadNum; tid++)
    {
        mythreads.emplace_back([&, tid]() {
            size_t codebookIdx = 0;
            while (true)
            {
                codebookIdx = sent.fetch_add(1);
                if (codebookIdx < options->m_quantizedDim)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Training Codebook %d.\n", codebookIdx);
                    auto kargs = COMMON::KmeansArgs<T>(numCentroids, subdim, raw_vectors->Count(), options->m_threadNum,
                                                       DistCalcMethod::L2, nullptr);
                    auto dset = COMMON::Dataset<T>(raw_vectors->Count(), subdim, blockRows, raw_vectors->Count());

                    for (int vectorIdx = 0; vectorIdx < raw_vectors->Count(); vectorIdx++)
                    {
                        auto raw_addr =
                            reinterpret_cast<T *>(raw_vectors->GetVector(vectorIdx)) + (codebookIdx * subdim);
                        auto dset_addr = dset[vectorIdx];
                        for (int k = 0; k < subdim; k++)
                        {
                            dset_addr[k] = raw_addr[k];
                        }
                    }

                    std::vector<SizeType> localindices;
                    localindices.resize(dset.R());
                    for (SizeType il = 0; il < localindices.size(); il++)
                        localindices[il] = il;

                    // auto nclusters = COMMON::KmeansClustering<T>(dset, localindices, 0, dset.R(), kargs,
                    // options->m_trainingSamples, options->m_KmeansLambda, options->m_debug, nullptr);

                    std::vector<SizeType> reverselocalindex;
                    reverselocalindex.resize(dset.R());
                    for (SizeType il = 0; il < reverselocalindex.size(); il++)
                    {
                        reverselocalindex[localindices[il]] = il;
                    }

                    for (int vectorIdx = 0; vectorIdx < raw_vectors->Count(); vectorIdx++)
                    {
                        auto localidx = reverselocalindex[vectorIdx];
                        auto quan_addr = reinterpret_cast<uint8_t *>(quantized_vectors->GetVector(vectorIdx));
                        quan_addr[codebookIdx] = kargs.label[localidx];
                    }

                    for (int j = 0; j < numCentroids; j++)
                    {
                        std::cout << kargs.counts[j] << '\t';
                    }
                    std::cout << std::endl;

                    T *cb = codebooks.get() + (numCentroids * subdim * codebookIdx);
                    for (int i = 0; i < numCentroids; i++)
                    {
                        for (int j = 0; j < subdim; j++)
                        {
                            cb[i * subdim + j] = kargs.centers[i * subdim + j];
                        }
                    }
                }
                else
                {
                    return;
                }
            }
        });
    }
    for (auto &t : mythreads)
    {
        t.join();
    }
    mythreads.clear();
    return codebooks;
}

#ifdef RABITQ
inline std::shared_ptr<COMMON::RaBitQQuantizer> TrainRaBitQQuantizer(
    const std::shared_ptr<QuantizerOptions>& options,
    const std::shared_ptr<VectorSet>& raw_vectors)
{
    const int bits = options->m_quantizedDim > 0 ? options->m_quantizedDim : 2;
    if (bits < 1 || bits > 8) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RaBitQ bits must be in [1, 8].\n");
        return nullptr;
    }
    if (options->m_distCalcMethod != DistCalcMethod::L2 &&
        options->m_distCalcMethod != DistCalcMethod::Cosine &&
        options->m_distCalcMethod != DistCalcMethod::InnerProduct) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "RaBitQ global quantization requires DistCalcMethod to be L2, Cosine, or InnerProduct.\n");
        return nullptr;
    }
    COMMON::RaBitQQuantizer::QuantizationMode mode;
    if (!TryParseRaBitQQuantizationMode(options->m_rabitqMode, mode)) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "RaBitQ quantization mode must be either exact or fast.\n");
        return nullptr;
    }
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        raw_vectors->Dimension(),
        bits,
        options->m_normalized,
        options->m_distCalcMethod,
        mode);
    if (quantizer->Train(raw_vectors) != ErrorCode::Success) {
        return nullptr;
    }
    return quantizer;
}
#endif
