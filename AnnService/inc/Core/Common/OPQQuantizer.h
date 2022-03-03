#pragma once

#ifndef _SPTAG_COMMON_OPQQUANTIZER_H_
#define _SPTAG_COMMON_OPQQUANTIZER_H_

#include "PQQuantizer.h"

namespace SPTAG
{
	namespace COMMON
	{
		using OPQMatrixType = float;
		template <typename T>
		class OPQQuantizer : public PQQuantizer<OPQMatrixType>
		{
		public:
			

			OPQQuantizer();

			OPQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, std::unique_ptr<OPQMatrixType[]>&& OPQMatrix);

			virtual void QuantizeVector(const void* vec, std::uint8_t* vecout);

			void ReconstructVector(const std::uint8_t* qvec, void* vecout);

			virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const;

			virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in);

			virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes);

			virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY);

			virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY);

			QuantizerType GetQuantizerType() {
				return QuantizerType::OPQQuantizer;
			}

		protected:
			using PQQuantizer<OPQMatrixType>::m_NumSubvectors;
			using PQQuantizer<OPQMatrixType>::m_DimPerSubvector;
			using PQQuantizer<OPQMatrixType>::m_KsPerSubvector;
			using PQQuantizer<OPQMatrixType>::m_codebooks;
			inline SizeType m_MatrixIndexCalc(SizeType i, SizeType j);

			template <typename I, typename O>
			inline void m_MatrixVectorMultiply(OPQMatrixType* mat, const I* vec, O* mat_vec, bool transpose = false);

			std::unique_ptr<OPQMatrixType[]> m_OPQMatrix;
		};

		template <typename T>
		OPQQuantizer<T>::OPQQuantizer() : PQQuantizer<OPQMatrixType>::PQQuantizer()
		{
		}

		template <typename T>
		OPQQuantizer<T>::OPQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, std::unique_ptr<OPQMatrixType[]>&& OPQMatrix) : m_OPQMatrix(std::move(OPQMatrix)), PQQuantizer<T>::PQQuantizer(NumSubvectors, KsPerSubvector, DimPerSubvector, EnableADC, std::move(Codebooks))
		{
		}

		template <typename T>
		void OPQQuantizer<T>::QuantizeVector(const void* vec, std::uint8_t* vecout)
		{
			OPQMatrixType* mat_vec = (OPQMatrixType*) _mm_malloc(sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			m_MatrixVectorMultiply<T, OPQMatrixType>(m_OPQMatrix.get(), (T*) vec, mat_vec);
			m_EnableADC = false;
			PQQuantizer<OPQMatrixType>::QuantizeVector(mat_vec, vecout);
			_mm_free(mat_vec);
		}

		template <typename T>
		void OPQQuantizer<T>::ReconstructVector(const std::uint8_t* qvec, void* vecout)
		{
			OPQMatrixType* pre_mat_vec = (OPQMatrixType*) _mm_malloc(sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			OPQMatrixType* post_mat_vec = (OPQMatrixType*)_mm_malloc(sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			PQQuantizer<OPQMatrixType>::ReconstructVector(qvec, pre_mat_vec);
			// OPQ Matrix is orthonormal, so inverse = transpose
			m_MatrixVectorMultiply<OPQMatrixType, OPQMatrixType>(m_OPQMatrix.get(), pre_mat_vec, post_mat_vec, true);
			float norm = 0;
			for (int i = 0; i < m_NumSubvectors * m_DimPerSubvector; i++)
			{
				norm += post_mat_vec[i] * post_mat_vec[i];
			}
			norm = ((float) Utils::GetBaseCore<T>())/sqrt(norm);
			T* vecout_T = (T*)vecout;
			for (int i = 0; i < m_NumSubvectors * m_DimPerSubvector; i++)
			{
				vecout_T[i] = (T)(norm * post_mat_vec[i]);
			}
			_mm_free(post_mat_vec);
			_mm_free(pre_mat_vec);
		}

		template <typename T>
		ErrorCode OPQQuantizer<T>::SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
		{
			QuantizerType qtype = QuantizerType::OPQQuantizer;
			VectorValueType rtype = GetEnumValueType<T>();
			IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
			IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);
			IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_NumSubvectors);
			IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_KsPerSubvector);
			IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_DimPerSubvector);
			IOBINARY(p_out, WriteBinary, sizeof(OPQMatrixType) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector, (char*)m_codebooks.get());
			IOBINARY(p_out, WriteBinary, sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector * m_NumSubvectors * m_DimPerSubvector, (char*)m_OPQMatrix.get());
			LOG(Helper::LogLevel::LL_Info, "Saving quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
			return ErrorCode::Success;
		}
		
		template <typename T>
		ErrorCode OPQQuantizer<T>::LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in)
		{
			auto code = PQQuantizer<OPQMatrixType>::LoadQuantizer(p_in);
			if (code != ErrorCode::Success)
			{
				return code;
			}
			
			m_OPQMatrix = std::make_unique<OPQMatrixType[]>((m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector));
			IOBINARY(p_in, ReadBinary, sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector * m_NumSubvectors * m_DimPerSubvector, (char*)m_OPQMatrix.get());
			LOG(Helper::LogLevel::LL_Info, "After read OPQ Matrix.\n");

			return ErrorCode::Success;
		}

		template <typename T>
		ErrorCode OPQQuantizer<T>::LoadQuantizer(std::uint8_t* raw_bytes)
		{
			PQQuantizer<OPQMatrixType>::LoadQuantizer(raw_bytes);
			raw_bytes += sizeof(DimensionType) + sizeof(SizeType) + sizeof(DimensionType) + (sizeof(OPQMatrixType) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector);
			m_OPQMatrix = std::make_unique<OPQMatrixType[]>((m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector));
			std::memcpy(m_OPQMatrix.get(), raw_bytes, sizeof(OPQMatrixType) * (m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector));
			raw_bytes += sizeof(OPQMatrixType) * (m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector);
			return ErrorCode::Success;
		}

		template <typename T>
		inline SizeType OPQQuantizer<T>::m_MatrixIndexCalc(SizeType i, SizeType j)
		{
			return (j * m_NumSubvectors * m_DimPerSubvector) + i;
		}

		template <typename T>
		template <typename I, typename O>
		inline void OPQQuantizer<T>::m_MatrixVectorMultiply(OPQMatrixType* mat, const I* vec, O* mat_vec, bool transpose)
		{
			for (int i = 0; i < m_NumSubvectors * m_DimPerSubvector; i++)
			{
				OPQMatrixType tmp = 0;
				for (int j = 0; j < m_NumSubvectors * m_DimPerSubvector; j++)
				{
					if (transpose)
					{
						tmp += mat[m_MatrixIndexCalc(j, i)] * vec[j];
					}
					else
					{
						tmp += mat[m_MatrixIndexCalc(i, j)] * vec[j];
					}
				}
				mat_vec[i] = (O) tmp;
			}
		}

		template <typename T>
		float OPQQuantizer<T>::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY)
		{
			auto distCalc = DistanceCalcSelector<T>(DistCalcMethod::L2);
			T* RX = (T* )_mm_malloc(sizeof(T) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			T* RY = (T*)_mm_malloc(sizeof(T) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);

			ReconstructVector(pX, (void*)RX);
			ReconstructVector(pY, (void*)RY);
			auto dist = distCalc(RX, RY, m_NumSubvectors * m_DimPerSubvector);
			_mm_free(RX);
			_mm_free(RY);
			return dist;
		}

		template <typename T>
		float OPQQuantizer<T>::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY)
		{
			auto distCalc = DistanceCalcSelector<T>(DistCalcMethod::Cosine);
			T* RX = (T*)_mm_malloc(sizeof(T) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			T* RY = (T*)_mm_malloc(sizeof(T) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);

			ReconstructVector(pX, (void*)RX);
			ReconstructVector(pY, (void*)RY);
			auto dist = distCalc(RX, RY, m_NumSubvectors * m_DimPerSubvector);
			_mm_free(RX);
			_mm_free(RY);
			return dist;
		}
	}
}

#endif  _SPTAG_COMMON_OPQQUANTIZER_H_
