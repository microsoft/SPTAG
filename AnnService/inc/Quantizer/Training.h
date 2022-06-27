#pragma once
#include <inc/Core/Common/DistanceUtils.h>
#include <inc/Core/Common/IQuantizer.h>
#include <inc/Core/Common/PQQuantizer.h>

#include <memory>
#include <inc/Core/VectorSet.h>
#include "inc/Core/Common/BKTree.h"

#define blockRows 4096
using namespace SPTAG;

class QuantizerOptions : public Helper::ReaderOptions
{
public:
    QuantizerOptions(SizeType trainingSamples, bool debug, float lambda, SPTAG::QuantizerType qtype, std::string qfile, DimensionType qdim, std::string fullvecs, std::string recvecs) : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32), m_trainingSamples(trainingSamples), m_debug(debug), m_KmeansLambda(lambda), m_quantizerType(qtype), m_outputQuantizerFile(qfile), m_quantizedDim(qdim), m_outputFullVecFile(fullvecs), m_outputReconstructVecFile(recvecs)
    {
        AddRequiredOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_outputFile, "-o", "--output", "Output quantized vectors.");
        AddOptionalOption(m_outputMetadataFile, "-om", "--outputmeta", "Output metadata.");
        AddOptionalOption(m_outputMetadataIndexFile, "-omi", "--outputmetaindex", "Output metadata index.");
        AddOptionalOption(m_outputQuantizerFile, "-oq", "--outputquantizer", "Output quantizer.");
        AddOptionalOption(m_quantizerType, "-qt", "--quantizer", "Quantizer type.");
        AddOptionalOption(m_quantizedDim, "-qd", "--quantizeddim", "Quantized Dimension.");

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
};

template <typename T>
std::unique_ptr<T[]> TrainPQQuantizer(std::shared_ptr<QuantizerOptions> options, std::shared_ptr<VectorSet> raw_vectors, std::shared_ptr<VectorSet> quantized_vectors)
{
    SizeType numCentroids = 256;
    if (raw_vectors->Dimension() % options->m_quantizedDim != 0) {
        LOG(Helper::LogLevel::LL_Error, "Only n_codebooks that divide dimension are supported.\n");
        exit(1);
    }
    DimensionType subdim = raw_vectors->Dimension() / options->m_quantizedDim;
    auto codebooks = std::make_unique<T[]>(numCentroids * raw_vectors->Dimension());

    LOG(Helper::LogLevel::LL_Info, "Begin Training Quantizer Codebooks.\n");
#pragma omp parallel for
    for (int codebookIdx = 0; codebookIdx < options->m_quantizedDim; codebookIdx++) {
        LOG(Helper::LogLevel::LL_Info, "Training Codebook %d.\n", codebookIdx);
        auto kargs = COMMON::KmeansArgs<T>(numCentroids, subdim, raw_vectors->Count(), options->m_threadNum, DistCalcMethod::L2, nullptr);
        auto dset = COMMON::Dataset<T>(raw_vectors->Count(), subdim, blockRows, raw_vectors->Count());

        for (int vectorIdx = 0; vectorIdx < raw_vectors->Count(); vectorIdx++) {
            auto raw_addr = reinterpret_cast<T*>(raw_vectors->GetVector(vectorIdx)) + (codebookIdx * subdim);
            auto dset_addr = dset[vectorIdx];
            for (int k = 0; k < subdim; k++) {
                dset_addr[k] = raw_addr[k];
            }
        }

        std::vector<SizeType> localindices;
        localindices.resize(dset.R());
        for (SizeType il = 0; il < localindices.size(); il++) localindices[il] = il;

        auto nclusters = COMMON::KmeansClustering<T>(dset, localindices, 0, dset.R(), kargs, options->m_trainingSamples, options->m_KmeansLambda, options->m_debug, nullptr);

        std::vector<SizeType> reverselocalindex;
        reverselocalindex.resize(dset.R());
        for (SizeType il = 0; il < reverselocalindex.size(); il++)
        {
            reverselocalindex[localindices[il]] = il;
        }

        for (int vectorIdx = 0; vectorIdx < raw_vectors->Count(); vectorIdx++) {
            auto localidx = reverselocalindex[vectorIdx];
            auto quan_addr = reinterpret_cast<uint8_t*>(quantized_vectors->GetVector(vectorIdx));
            quan_addr[codebookIdx] = kargs.label[localidx];

        }

        for (int j = 0; j < numCentroids; j++) {
            std::cout << kargs.counts[j] << '\t';
        }
        std::cout << std::endl;

        T* cb = codebooks.get() + (numCentroids * subdim * codebookIdx);
        for (int i = 0; i < numCentroids; i++)
        {
            for (int j = 0; j < subdim; j++)
            {
                cb[i * subdim + j] = kargs.centers[i * subdim + j];
            }
        }
    }

    return codebooks;
}