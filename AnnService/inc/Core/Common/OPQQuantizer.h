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

			virtual void QuantizeVector(const void* vec, std::uint8_t* vecout) const;

			void ReconstructVector(const std::uint8_t* qvec, void* vecout) const;

			virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const;

			virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in);

			virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes);

			virtual SizeType ReconstructSize() const;

			virtual std::uint64_t BufferSize() const;

			virtual int GetBase() const;


			QuantizerType GetQuantizerType() const 
			{
				return QuantizerType::OPQQuantizer;
			}

			VectorValueType GetReconstructType() const
			{
				return GetEnumValueType<T>();
			}


		protected:
			using PQQuantizer<OPQMatrixType>::m_NumSubvectors;
			using PQQuantizer<OPQMatrixType>::m_DimPerSubvector;
			using PQQuantizer<OPQMatrixType>::m_KsPerSubvector;
			using PQQuantizer<OPQMatrixType>::m_codebooks;
			inline SizeType m_MatrixIndexCalc(SizeType i, SizeType j) const;

			template <typename I, typename O>
			inline void m_MatrixVectorMultiply(OPQMatrixType* mat, const I* vec, O* mat_vec, bool transpose = false) const;

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
		void OPQQuantizer<T>::QuantizeVector(const void* vec, std::uint8_t* vecout) const
		{
			OPQMatrixType* mat_vec = (OPQMatrixType*) ALIGN_ALLOC(sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector);
			m_MatrixVectorMultiply<T, OPQMatrixType>(m_OPQMatrix.get(), (T*) vec, mat_vec, true);
			PQQuantizer<OPQMatrixType>::QuantizeVector(mat_vec, vecout);
			ALIGN_FREE(mat_vec);
		}

		template <typename T>
		void OPQQuantizer<T>::ReconstructVector(const std::uint8_t* qvec, void* vecout) const
		{
			OPQMatrixType* pre_mat_vec = (OPQMatrixType*) ALIGN_ALLOC(sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector);
			PQQuantizer<OPQMatrixType>::ReconstructVector(qvec, pre_mat_vec);
			// OPQ Matrix is orthonormal, so inverse = transpose
			m_MatrixVectorMultiply<OPQMatrixType, T>(m_OPQMatrix.get(), pre_mat_vec, (T*) vecout);

			ALIGN_FREE(pre_mat_vec);
		}

		template <typename T>
		ErrorCode OPQQuantizer<T>::SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const
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
		ErrorCode OPQQuantizer<T>::LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in)
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
		SizeType OPQQuantizer<T>::ReconstructSize() const
		{
			return sizeof(T) * ReconstructDim();
		}

		template <typename T>
		std::uint64_t OPQQuantizer<T>::BufferSize() const
		{
			return PQQuantizer<OPQMatrixType>::BufferSize() + (sizeof(OPQMatrixType) * m_NumSubvectors * m_DimPerSubvector * m_NumSubvectors * m_DimPerSubvector);
		}

		template <typename T>
		int OPQQuantizer<T>::GetBase() const
		{
			return COMMON::Utils::GetBase<T>();
		}

		template <typename T>
		inline SizeType OPQQuantizer<T>::m_MatrixIndexCalc(SizeType i, SizeType j) const
		{
			return (i * m_NumSubvectors * m_DimPerSubvector) + j;
		}

		template <typename T>
		template <typename I, typename O>
		inline void OPQQuantizer<T>::m_MatrixVectorMultiply(OPQMatrixType* mat, const I* vec, O* mat_vec, bool transpose) const
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


	}
}

#endif  _SPTAG_COMMON_OPQQUANTIZER_H_
