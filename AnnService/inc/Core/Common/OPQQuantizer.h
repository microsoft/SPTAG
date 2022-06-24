#pragma once

#ifndef _SPTAG_COMMON_OPQQUANTIZER_H_
#define _SPTAG_COMMON_OPQQUANTIZER_H_

#include "PQQuantizer.h"

#if (__cplusplus < 201703L)
#define ISNOTSAME(A, B) if (!std::is_same<A, B>::value)
#else
#define ISNOTSAME(A, B) if constexpr (!std::is_same_v<A, B>)
#endif

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
            DimensionType m_matrixDim;
            const std::function<float(const OPQMatrixType*, const OPQMatrixType*, DimensionType)> m_fdot = SPTAG::COMMON::DistanceCalcSelector<OPQMatrixType>(SPTAG::DistCalcMethod::Cosine);
            const int m_base = COMMON::Utils::GetBase<OPQMatrixType>() * COMMON::Utils::GetBase<OPQMatrixType>();

            void m_InitMatrixTranspose();

            template <typename O>
            inline void m_VectorMatrixMultiply(OPQMatrixType* mat, const OPQMatrixType* vec, O* mat_vec) const;

            std::unique_ptr<OPQMatrixType[]> m_OPQMatrix;
            std::unique_ptr<OPQMatrixType[]> m_OPQMatrix_T;
        };

        template <typename T>
        OPQQuantizer<T>::OPQQuantizer() : PQQuantizer<OPQMatrixType>::PQQuantizer()
        {
        }

        template <typename T>
        OPQQuantizer<T>::OPQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, std::unique_ptr<OPQMatrixType[]>&& OPQMatrix) : m_OPQMatrix(std::move(OPQMatrix)), PQQuantizer<T>::PQQuantizer(NumSubvectors, KsPerSubvector, DimPerSubvector, EnableADC, std::move(Codebooks)), m_matrixDim(NumSubvectors * DimPerSubvector)
        {
            m_InitMatrixTranspose();
        }

        template <typename T>
        void OPQQuantizer<T>::m_InitMatrixTranspose() {
            m_OPQMatrix_T = std::make_unique<OPQMatrixType[]>(m_matrixDim * m_matrixDim);
            for (int i = 0; i < m_matrixDim; i++) {
                for (int j = 0; j < m_matrixDim; j++) {
                    m_OPQMatrix_T[i * m_matrixDim + j] = m_OPQMatrix[j * m_matrixDim + i];
                }
            }
        }

        template <typename T>
        void OPQQuantizer<T>::QuantizeVector(const void* vec, std::uint8_t* vecout) const
        {
            OPQMatrixType* mat_vec = (OPQMatrixType*) ALIGN_ALLOC(sizeof(OPQMatrixType) * m_matrixDim);
            OPQMatrixType* typed_vec;
            ISNOTSAME(T, OPQMatrixType)
            {
                typed_vec = (OPQMatrixType*)ALIGN_ALLOC(sizeof(OPQMatrixType) * m_matrixDim);
                for (int i = 0; i < m_matrixDim; i++)
                {
                    typed_vec[i] = (OPQMatrixType)((T*)vec)[i];
                }
            }
            else {
                typed_vec = (OPQMatrixType*)vec;
            }

            m_VectorMatrixMultiply<OPQMatrixType>(m_OPQMatrix_T.get(), typed_vec, mat_vec);
            PQQuantizer<OPQMatrixType>::QuantizeVector(mat_vec, vecout);
            ALIGN_FREE(mat_vec);

            ISNOTSAME(T, OPQMatrixType)
            {
                ALIGN_FREE(typed_vec);
            }
        }

        template <typename T>
        void OPQQuantizer<T>::ReconstructVector(const std::uint8_t* qvec, void* vecout) const
        {
            OPQMatrixType* pre_mat_vec = (OPQMatrixType*) ALIGN_ALLOC(sizeof(OPQMatrixType) * m_matrixDim);
            PQQuantizer<OPQMatrixType>::ReconstructVector(qvec, pre_mat_vec);
            // OPQ Matrix is orthonormal, so inverse = transpose
            m_VectorMatrixMultiply<T>(m_OPQMatrix.get(), pre_mat_vec, (T*)vecout);
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
            IOBINARY(p_out, WriteBinary, sizeof(OPQMatrixType) * m_matrixDim * m_matrixDim, (char*)m_OPQMatrix.get());
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
            m_matrixDim = m_NumSubvectors * m_DimPerSubvector;
            m_OPQMatrix = std::make_unique<OPQMatrixType[]>(m_matrixDim * m_matrixDim);
            IOBINARY(p_in, ReadBinary, sizeof(OPQMatrixType) * m_matrixDim * m_matrixDim, (char*)m_OPQMatrix.get());
            LOG(Helper::LogLevel::LL_Info, "After read OPQ Matrix.\n");

            m_InitMatrixTranspose();
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode OPQQuantizer<T>::LoadQuantizer(std::uint8_t* raw_bytes)
        {
            PQQuantizer<OPQMatrixType>::LoadQuantizer(raw_bytes);
            raw_bytes += sizeof(DimensionType) + sizeof(SizeType) + sizeof(DimensionType) + (sizeof(OPQMatrixType) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector);
            m_matrixDim = m_NumSubvectors * m_DimPerSubvector;
            m_OPQMatrix = std::make_unique<OPQMatrixType[]>(m_matrixDim * m_matrixDim);
            std::memcpy(m_OPQMatrix.get(), raw_bytes, sizeof(OPQMatrixType) * m_matrixDim * m_matrixDim);
            raw_bytes += sizeof(OPQMatrixType) * m_matrixDim * m_matrixDim;

            m_InitMatrixTranspose();
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
            return PQQuantizer<OPQMatrixType>::BufferSize() + (sizeof(OPQMatrixType) * m_matrixDim * m_matrixDim);
        }

        template <typename T>
        int OPQQuantizer<T>::GetBase() const
        {
            return COMMON::Utils::GetBase<T>();
        }

        template <typename T>
        template <typename O>
        inline void OPQQuantizer<T>::m_VectorMatrixMultiply(OPQMatrixType* mat, const OPQMatrixType* vec, O* mat_vec) const
        {
            for (int i = 0; i < m_matrixDim; i++) {
                mat_vec[i] = (O)(m_base - m_fdot(vec, mat, m_matrixDim));
                mat += m_matrixDim;
            }
        }
    }
}

#endif  _SPTAG_COMMON_OPQQUANTIZER_H_
