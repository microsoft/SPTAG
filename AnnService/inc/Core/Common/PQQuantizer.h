// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_PQQUANTIZER_H_
#define _SPTAG_COMMON_PQQUANTIZER_H_

#include "CommonUtils.h"
#include "DistanceUtils.h"
#include "IQuantizer.h"
#include "inc/Helper/StringConvert.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <cassert>
#include <cstring>


namespace SPTAG
{
    namespace COMMON
    {
        template <typename T>
        class PQQuantizer : public IQuantizer
        {
        public:
            PQQuantizer();

            PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, DistCalcMethod distMethod);

            ~PQQuantizer();

            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const;

            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const;

            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const;
            
            virtual SizeType QuantizeSize() const;

            void ReconstructVector(const std::uint8_t* qvec, void* vecout) const;

            virtual SizeType ReconstructSize() const;

            virtual DimensionType ReconstructDim() const;

            virtual std::uint64_t BufferSize() const;

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const;

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in);

            virtual ErrorCode LoadQuantizer(std::uint8_t* raw_bytes);

            virtual DimensionType GetNumSubvectors() const;

            virtual int GetBase() const;

            SizeType GetKsPerSubvector() const;

            SizeType GetBlockSize() const;

            DimensionType GetDimPerSubvector() const;

            virtual bool GetEnableADC() const;

            virtual void SetEnableADC(bool enableADC);

            VectorValueType GetReconstructType() const
            {
                return GetEnumValueType<T>();
            }

            QuantizerType GetQuantizerType() const {
                return QuantizerType::PQQuantizer;
            }

            float* GetL2DistanceTables();

            T* GetCodebooks();

        protected:
            DimensionType m_NumSubvectors;
            SizeType m_KsPerSubvector;
            DimensionType m_DimPerSubvector;
            SizeType m_BlockSize;
            bool m_EnableADC;
            DistCalcMethod m_distMethod;

            inline SizeType m_DistIndexCalc(SizeType i, SizeType j, SizeType k) const;
            void InitializeDistanceTables();

            std::unique_ptr<T[]> m_codebooks;
            std::unique_ptr<const float[]> m_L2DistanceTables;
        };

        template <typename T>
        PQQuantizer<T>::PQQuantizer() : m_NumSubvectors(0), m_KsPerSubvector(0), m_DimPerSubvector(0), m_BlockSize(0), m_EnableADC(false), m_distMethod(DistCalcMethod::L2)
        {
        }

        template <typename T>
        PQQuantizer<T>::PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, std::unique_ptr<T[]>&& Codebooks, DistCalcMethod distMethod) : 
            m_NumSubvectors(NumSubvectors), m_KsPerSubvector(KsPerSubvector), m_DimPerSubvector(DimPerSubvector), m_BlockSize(KsPerSubvector* KsPerSubvector), m_codebooks(std::move(Codebooks)), m_EnableADC(EnableADC), m_distMethod(distMethod)
        {
            InitializeDistanceTables();
        }

        template <typename T>
        PQQuantizer<T>::~PQQuantizer()
        {}

        template <typename T>
        float PQQuantizer<T>::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const
            // pX must be query distance table for ADC
        {
            float out = 0;
            if (GetEnableADC()) {          
                float* ptr = (float*)pX;
                for (int i = 0; i < m_NumSubvectors; i++) {
                    out += ptr[pY[i]];
                    ptr += m_KsPerSubvector;
                }
            }
            else {
                for (int i = 0; i < m_NumSubvectors; i++) {
                    out += m_L2DistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
                }                
            }
            return out;
        }

        template <typename T>
        float PQQuantizer<T>::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const
            // pX must be query distance table for ADC
        {
            float out = 0;
            if (GetEnableADC()) {
                float* ptr = (float*)pX;
                for (int i = 0; i < m_NumSubvectors; i++) {
                    out += ptr[pY[i]];
                    ptr += m_KsPerSubvector;
                }
            }
            else {
                for (int i = 0; i < m_NumSubvectors; i++) {
                    out += m_L2DistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
                }
            }
            return out;
        }

        template <typename T>
        void PQQuantizer<T>::QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC) const
        {
            if (ADC && GetEnableADC())
            {
                auto distCalc = COMMON::DistanceCalcSelector<T>(m_distMethod);
                float* ADCtable = (float*) vecout;
                T* subcodebooks = m_codebooks.get();
                T* subvec = (T*)vec;
                for (int i = 0; i < m_NumSubvectors; i++)
                {
                    for (int j = 0; j < m_KsPerSubvector; j++)
                    {
                        (*ADCtable) = distCalc(subvec, subcodebooks, m_DimPerSubvector);
                        ADCtable++;
                        subcodebooks += m_DimPerSubvector;
                    }
                    subvec += m_DimPerSubvector;
                }
            }
            else 
            {
                auto distCalc = COMMON::DistanceCalcSelector<T>(DistCalcMethod::L2);
                T* subvec = (T*)vec;
                T* subcodebooks = m_codebooks.get();
                for (int i = 0; i < m_NumSubvectors; i++) {
                    int bestIndex = -1;
                    float minDist = std::numeric_limits<float>::infinity();

                    for (int j = 0; j < m_KsPerSubvector; j++) {
                        float dist = distCalc(subvec, subcodebooks, m_DimPerSubvector);
                        if (dist < minDist) {
                            bestIndex = j;
                            minDist = dist;
                        }
                        subcodebooks += m_DimPerSubvector;
                    }
                    assert(bestIndex != -1);
                    vecout[i] = (std::uint8_t)bestIndex;
                    subvec += m_DimPerSubvector;
                }
            }           
        }

        template <typename T>
        SizeType PQQuantizer<T>::QuantizeSize() const
        {
            if (GetEnableADC())
            {
                return sizeof(float) * m_NumSubvectors * m_KsPerSubvector;
            }
            else
            {
                return m_NumSubvectors;
            }           
        }

        template <typename T>
        void PQQuantizer<T>::ReconstructVector(const std::uint8_t* qvec, void* vecout) const
        {
            T* sub_vecout = (T*)vecout;
            T* subcodebooks = m_codebooks.get();
            for (int i = 0; i < m_NumSubvectors; i++) {
                memcpy(sub_vecout, subcodebooks + qvec[i] * m_DimPerSubvector, sizeof(T) * m_DimPerSubvector);
                sub_vecout += m_DimPerSubvector;
                subcodebooks += m_KsPerSubvector * m_DimPerSubvector;
            }
        }

        template <typename T>
        SizeType PQQuantizer<T>::ReconstructSize() const
        {       
            return sizeof(T) * ReconstructDim();
        }

        template <typename T>
        DimensionType PQQuantizer<T>::ReconstructDim() const
        {
            return m_DimPerSubvector * m_NumSubvectors;
        }

        template <typename T>
        std::uint64_t PQQuantizer<T>::BufferSize() const
        {
            return sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector + 
                sizeof(DimensionType) + sizeof(SizeType) + sizeof(DimensionType) + sizeof(VectorValueType) + sizeof(QuantizerType) + sizeof(DistCalcMethod);
        }

        template <typename T>
        ErrorCode PQQuantizer<T>::SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const
        {
            QuantizerType qtype = QuantizerType::PQQuantizer;
            VectorValueType rtype = GetEnumValueType<T>();
            IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
            IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_NumSubvectors);
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_KsPerSubvector);
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_DimPerSubvector);
            IOBINARY(p_out, WriteBinary, sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector, (char*)m_codebooks.get());
            IOBINARY(p_out, WriteBinary, sizeof(DistCalcMethod), (char*)&m_distMethod);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode PQQuantizer<T>::LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading Quantizer.\n");
            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_NumSubvectors);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read subvecs: %s.\n", std::to_string(m_NumSubvectors).c_str());
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_KsPerSubvector);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read ks: %s.\n", std::to_string(m_KsPerSubvector).c_str());
            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_DimPerSubvector);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read dim: %s.\n", std::to_string(m_DimPerSubvector).c_str());
            m_codebooks = std::make_unique<T[]>(m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "sizeof(T): %s.\n", std::to_string(sizeof(T)).c_str());
            IOBINARY(p_in, ReadBinary, sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector, (char*)m_codebooks.get());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read codebooks.\n");
            
            if (p_in->ReadBinary(sizeof(DistCalcMethod), (char*)&m_distMethod) != sizeof(DistCalcMethod)) {
                m_distMethod = DistCalcMethod::L2;
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read dist: %s.\n", Helper::Convert::ConvertToString(m_distMethod).c_str());
            

            m_BlockSize = m_KsPerSubvector * m_KsPerSubvector;
            InitializeDistanceTables();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
            return ErrorCode::Success;
        }

        template <typename T>
        ErrorCode PQQuantizer<T>::LoadQuantizer(std::uint8_t* raw_bytes)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading Quantizer.\n");
            m_NumSubvectors = *(DimensionType*)raw_bytes;
            raw_bytes += sizeof(DimensionType);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read subvecs: %s.\n", std::to_string(m_NumSubvectors).c_str());
            m_KsPerSubvector = *(SizeType*)raw_bytes;
            raw_bytes += sizeof(SizeType);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read ks: %s.\n", std::to_string(m_KsPerSubvector).c_str());
            m_DimPerSubvector = *(DimensionType*)raw_bytes;
            raw_bytes += sizeof(DimensionType);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read dim: %s.\n", std::to_string(m_DimPerSubvector).c_str());
            m_codebooks = std::make_unique<T[]>(m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "sizeof(T): %s.\n", std::to_string(sizeof(T)).c_str());
            std::memcpy(m_codebooks.get(), raw_bytes, sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector);
            raw_bytes += sizeof(T) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read codebooks.\n");
            
            m_distMethod = *(DistCalcMethod*)raw_bytes;
            raw_bytes += sizeof(DistCalcMethod);
            if (m_distMethod >= DistCalcMethod::Undefined) {
                m_distMethod = DistCalcMethod::L2;
                raw_bytes -= sizeof(DistCalcMethod);
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read dist: %s.\n", Helper::Convert::ConvertToString(m_distMethod).c_str());
            
            m_BlockSize = m_KsPerSubvector * m_KsPerSubvector;
            InitializeDistanceTables();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
            return ErrorCode::Success;
        }

        template <typename T>
        int PQQuantizer<T>::GetBase() const
        {
            return COMMON::Utils::GetBase<T>();
        }

        template <typename T>
        DimensionType PQQuantizer<T>::GetNumSubvectors() const
        {
            return m_NumSubvectors;
        }

        template <typename T>
        SizeType PQQuantizer<T>::GetKsPerSubvector() const
        {
            return m_KsPerSubvector;
        }

        template <typename T>
        SizeType PQQuantizer<T>::GetBlockSize() const
        {
            return m_BlockSize;
        }

        template <typename T>
        DimensionType PQQuantizer<T>::GetDimPerSubvector() const
        {
            return m_DimPerSubvector;
        }

        template <typename T>
        bool PQQuantizer<T>::GetEnableADC() const
        {
            return m_EnableADC;
        }

        template <typename T>
        void PQQuantizer<T>::SetEnableADC(bool enableADC)
        {
            m_EnableADC = enableADC;
        }

        template <typename T>
        inline SizeType PQQuantizer<T>::m_DistIndexCalc(SizeType i, SizeType j, SizeType k) const {
            return m_BlockSize * i + j * m_KsPerSubvector + k;
        }

        template <typename T>
        void PQQuantizer<T>::InitializeDistanceTables()
        {
            auto temp_m_L2DistanceTables = std::make_unique<float[]>(m_BlockSize * m_NumSubvectors);
            auto distFunc = COMMON::DistanceCalcSelector<T>(m_distMethod);

            for (int i = 0; i < m_NumSubvectors; i++) {
                SizeType baseIdx = i * m_KsPerSubvector * m_DimPerSubvector;
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    for (int k = 0; k < m_KsPerSubvector; k++) {
                        temp_m_L2DistanceTables[m_DistIndexCalc(i, j, k)] = distFunc(&m_codebooks[baseIdx + j * m_DimPerSubvector], &m_codebooks[baseIdx + k * m_DimPerSubvector], m_DimPerSubvector);
                    }
                }
            }
            m_L2DistanceTables = std::move(temp_m_L2DistanceTables);
        }

        template <typename T>
        float* PQQuantizer<T>::GetL2DistanceTables() {
            return (float*)(m_L2DistanceTables.get());
        }
 
        template<typename T>
        T* PQQuantizer<T>::GetCodebooks() {
          return (T*)(m_codebooks.get());
        }
    }
}

#endif // _SPTAG_COMMON_PQQUANTIZER_H_
