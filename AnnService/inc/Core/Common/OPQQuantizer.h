#pragma once

#ifndef _SPTAG_COMMON_OPQQUANTIZER_H_
#define _SPTAG_COMMON_OPQQUANTIZER_H_

#include "PQQuantizer.h"

namespace SPTAG
{
	namespace COMMON
	{
		template <typename T>
		class OPQQuantizer : public PQQuantizer<T>
		{
		public:
			using OPQMatrixType = float;

			OPQQuantizer();

			OPQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, std::unique_ptr<OPQMatrixType[]>&& OPQMatrix);

			virtual void QuantizeVector(const void* vec, std::uint8_t* vecout);

			void ReconstructVector(const std::uint8_t* qvec, void* vecout);

			virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const;

			virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in);

			virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes);

			QuantizerType GetQuantizerType() {
				return QuantizerType::OPQQuantizer;
			}

		protected:
			using PQQuantizer<T>::m_NumSubvectors;
			using PQQuantizer<T>::m_DimPerSubvector;
			using PQQuantizer<T>::m_KsPerSubvector;
			using PQQuantizer<T>::m_codebooks;
			inline SizeType m_MatrixIndexCalc(SizeType i, SizeType j);

			inline void m_MatrixVectorMultiply(OPQMatrixType* mat, const void* vec, void* mat_vec, bool transpose = false);

			std::unique_ptr<OPQMatrixType[]> m_OPQMatrix;
		};

		template <typename T>
		OPQQuantizer<T>::OPQQuantizer() : PQQuantizer<T>::PQQuantizer()
		{
		}

		template <typename T>
		OPQQuantizer<T>::OPQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, std::unique_ptr<OPQMatrixType[]>&& OPQMatrix) : m_OPQMatrix(std::move(OPQMatrix)), PQQuantizer<T>::PQQuantizer(NumSubvectors, KsPerSubvector, DimPerSubvector, EnableADC, std::move(Codebooks))
		{
		}

		template <typename T>
		void OPQQuantizer<T>::QuantizeVector(const void* vec, std::uint8_t* vecout)
		{
			void* mat_vec = _mm_malloc(sizeof(T) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			m_MatrixVectorMultiply(m_OPQMatrix.get(), vec, mat_vec);
			PQQuantizer<T>::QuantizeVector(mat_vec, vecout);
			_mm_free(mat_vec);
		}

		template <typename T>
		void OPQQuantizer<T>::ReconstructVector(const std::uint8_t* qvec, void* vecout)
		{
			void* pre_mat_vec = _mm_malloc(sizeof(T) * m_NumSubvectors * m_DimPerSubvector, ALIGN_SPTAG);
			PQQuantizer<T>::ReconstructVector(qvec, pre_mat_vec);
			// OPQ Matrix is orthonormal, so inverse = transpose
			m_MatrixVectorMultiply(m_OPQMatrix.get(), pre_mat_vec, vecout, true);
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
			IOBINARY(p_out, WriteBinary, sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector, (char*)m_codebooks.get());
			IOBINARY(p_out, WriteBinary, sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector * m_NumSubvectors * m_DimPerSubvector, (char*)m_OPQMatrix.get());
			LOG(Helper::LogLevel::LL_Info, "Saving quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
			return ErrorCode::Success;
		}
		
		template <typename T>
		ErrorCode OPQQuantizer<T>::LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in)
		{
			auto code = PQQuantizer<T>::LoadQuantizer(p_in);
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
			PQQuantizer<T>::LoadQuantizer(raw_bytes);
			raw_bytes += sizeof(DimensionType) + sizeof(SizeType) + sizeof(DimensionType) + (sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector);
			m_OPQMatrix = std::make_unique<OPQMatrixType[]>((m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector));
			std::memcpy(m_OPQMatrix.get(), raw_bytes, sizeof(OPQMatrixType) * (m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector));
			raw_bytes += sizeof(OPQMatrixType) * (m_NumSubvectors * m_DimPerSubvector) * (m_NumSubvectors * m_DimPerSubvector);
			return ErrorCode::Success;
		}

		template <typename T>
		inline SizeType OPQQuantizer<T>::m_MatrixIndexCalc(SizeType i, SizeType j)
		{
			return (i * m_NumSubvectors * m_DimPerSubvector) + j;
		}

		template <typename T>
		inline void OPQQuantizer<T>::m_MatrixVectorMultiply(OPQMatrixType* mat, const void* vec, void* mat_vec, bool transpose)
		{
			T* vec_T = (T*) vec;
			T* mat_vec_T = (T*)mat_vec;
			for (int i = 0; i < m_NumSubvectors * m_DimPerSubvector; i++)
			{
				mat_vec_T[i] = 0;
				for (int j = 0; j < m_NumSubvectors * m_DimPerSubvector; j++)
				{
					if (transpose)
					{
						mat_vec_T[i] += mat[m_MatrixIndexCalc(j, i)] * vec_T[j];
					}
					else
					{
						mat_vec_T[i] += mat[m_MatrixIndexCalc(i, j)] * vec_T[j];
					}
				}
			}
		}

	}
}

#endif  _SPTAG_COMMON_OPQQUANTIZER_H_
