#pragma once
#include <inc/Core/Common/DistanceUtils.h>
#include <inc/Core/Common/IQuantizer.h>
#include <inc/Core/Common/PQQuantizer.h>

#include <inc/Core/Common/OPQQuantizer.h>
#ifdef GPU
#include <inc/Quantizer/OPQUpdate.hxx>
#endif


#include <memory>
#include <inc/Core/VectorSet.h>
#include "inc/Core/Common/BKTree.h"

#define blockRows 4096
using namespace SPTAG;

class QuantizerOptions : public Helper::ReaderOptions
{
public:
    QuantizerOptions(SizeType trainingSamples, bool debug, float lambda, SPTAG::QuantizerType qtype, std::string qfile, DimensionType qdim, std::string fullvecs) : Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32), m_trainingSamples(trainingSamples), m_debug(debug), m_KmeansLambda(lambda), m_quantizerType(qtype), m_outputQuantizerFile(qfile), m_quantizedDim(qdim), m_outputFullVecFile(fullvecs)
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
		AddOptionalOption(m_numIters, "-iters", "--num_iters", "Number of training iterations.");
        AddOptionalOption(m_debug, "-debug", "--debug", "Print debug information.");
        AddOptionalOption(m_KmeansLambda, "-kml", "--lambda", "Kmeans lambda parameter.");
        AddOptionalOption(m_outputFullVecFile, "-ofv", "--output_full", "Output Uncompressed vectors.");
    }

    ~QuantizerOptions() {}

    std::string m_inputFiles;

    std::string m_outputFile;

    std::string m_outputFullVecFile;

    std::string m_outputMetadataFile;

    std::string m_outputMetadataIndexFile;

    std::string m_outputQuantizerFile;

    DimensionType m_quantizedDim;

    SizeType m_trainingSamples;

	 SizeType m_numIters;
	 
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

        if (options->m_debug)
        {
            for (int j = 0; j < numCentroids; j++) {
                std::cout << kargs.counts[j] << '\t';
            }
            std::cout << std::endl;
        }

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

template <typename T>
std::shared_ptr<SPTAG::COMMON::IQuantizer> TrainOPQQuantizer(std::shared_ptr<QuantizerOptions> options, std::shared_ptr<VectorSet> raw_vectors, std::shared_ptr<VectorSet> quantized_vectors)
{
	 #ifdef GPU
	 SizeType dim = raw_vectors->Dimension();

     std::shared_ptr<VectorSet> BaseSet;
     if (options->m_inputValueType != VectorValueType::Float)
     {
         auto base_points = ByteArray::Alloc(sizeof(float) * dim * raw_vectors->Count());
         BaseSet = std::make_shared<BasicVectorSet>(base_points, VectorValueType::Float, dim, raw_vectors->Count());
#pragma omp parallel for
         for (int i = 0; i < raw_vectors->Count(); i++)
         {
             for (int j = 0; j < dim; j++)
             {
                 ((float*)BaseSet->GetVector(i))[j] = (float)((T*)raw_vectors->GetVector(i))[j];
             }
         }
     }
     else
     {
         BaseSet = raw_vectors;
     }

	 float* svd_matrix = new float[dim* dim];
	 float* rotation_matrix = new float[dim* dim];
	 auto rotated_points = ByteArray::Alloc(sizeof(float) * dim * raw_vectors->Count());
	 std::shared_ptr<VectorSet> RotatedSet = std::make_shared<BasicVectorSet>(rotated_points, VectorValueType::Float, dim, raw_vectors->Count());
	 float* reconstructed_points = new float[dim* raw_vectors->Count()];

	 std::shared_ptr<SPTAG::COMMON::IQuantizer> out;

	 for (int i = 0; i < dim; i++)
	 {
		  rotation_matrix[i*dim + i] = 1;
	 }
     auto baseQuan = std::make_shared<SPTAG::COMMON::PQQuantizer<float>>(options->m_quantizedDim, 256, dim / options->m_quantizedDim, false, TrainPQQuantizer<float>(options, BaseSet, quantized_vectors));
#pragma omp parallel for
	 for (int i = 0; i < raw_vectors->Count(); i++)
	 {
		  baseQuan->QuantizeVector(BaseSet->GetVector(i), (std::uint8_t*)quantized_vectors->GetVector(i));
		  baseQuan->ReconstructVector((const std::uint8_t*)quantized_vectors->GetVector(i), ((void*)(&reconstructed_points[dim * i])));
	 }

	 for (int iter = 0; iter < options->m_numIters; iter++)
	 {
		  memset(svd_matrix, 0, sizeof(float)*dim*dim);
#pragma omp parallel for
		  for (int dim1 = 0; dim1 < dim; ++dim1)
		  {
			   for (int dim2 = 0; dim2 < dim; ++dim2)
			   {
					for (int point_id = 0; point_id < raw_vectors->Count(); ++point_id)
					{
                        // column-major order
						 svd_matrix[dim2*dim + dim1] += ((float*)BaseSet->GetVector(point_id))[dim1] * reconstructed_points[(point_id*dim) +dim2];
					}
			   }
		  }
		  
		  OPQRotationUpdate(svd_matrix, rotation_matrix, dim);

		  /* Update R'X*/
#pragma omp parallel for
		  for (int point_id = 0; point_id < raw_vectors->Count(); ++point_id)
		  {
			   for (int dim1 = 0; dim1 < dim; ++dim1)
			   {
					float ele = 0;
					for (int dim2 = 0; dim2 < dim; ++dim2)
					{
                        // column-major
						 ele += rotation_matrix[dim1*dim + dim2] * ((float*)BaseSet->GetVector(point_id))[dim2];
					}
					((float*)RotatedSet->GetVector(point_id))[dim1] = ele;
			   }
		  }
		  auto codebooks = TrainPQQuantizer<float>(options, RotatedSet, quantized_vectors);
		  std::unique_ptr<float[]> opq_rot = std::make_unique<float[]>(dim*dim);

		  memcpy_s(opq_rot.get(), sizeof(float) * dim * dim, rotation_matrix, sizeof(float) * dim * dim);
          /*for (int i = 0; i < dim; i++)
          {
              for (int j = 0; j < dim; j++)
              {
                  // column-major order means already transposed
                  opq_rot[j * dim + i] = rotation_matrix[i * dim + j];
              }
          }*/

          if (iter + 1 == options->m_numIters)
          {
              out = std::make_shared<SPTAG::COMMON::OPQQuantizer<T>>(options->m_quantizedDim, 256, dim / options->m_quantizedDim, false, std::move(codebooks), std::move(opq_rot));
          }
          else {
              out = std::make_shared<SPTAG::COMMON::OPQQuantizer<float>>(options->m_quantizedDim, 256, dim / options->m_quantizedDim, false, std::move(codebooks), std::move(opq_rot));
#pragma omp parallel for
              for (int i = 0; i < raw_vectors->Count(); i++)
              {
                  out->QuantizeVector(BaseSet->GetVector(i), (std::uint8_t*)quantized_vectors->GetVector(i));
                  out->ReconstructVector((const std::uint8_t*)quantized_vectors->GetVector(i), (void*)&(reconstructed_points[dim * i]));
              }
          }

	 }

	 delete svd_matrix;
	 //delete reconstructed_points;
	 delete rotation_matrix;
	 delete reconstructed_points;
	 
	 return out;
#else
	 LOG(Helper::LogLevel::LL_Error, "CPU OPQ Training not supported.\n");
	 exit(1);
#endif

}