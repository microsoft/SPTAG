#include <inc/Core/Common/PQQuantizer.h>
#include <iostream>
#include <fstream>
#include <limits>


namespace SPTAG
{
    namespace COMMON
    {

        PQQuantizer::PQQuantizer()
        {
        }

        PQQuantizer::PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, bool EnableADC, float* Codebooks)
        {
            m_NumSubvectors = NumSubvectors;
            m_KsPerSubvector = KsPerSubvector;
            m_DimPerSubvector = DimPerSubvector;
            m_codebooks.reset(Codebooks);
            m_EnableADC = EnableADC;

            m_BlockSize = (m_KsPerSubvector * (m_KsPerSubvector + 1)) / 2;
            m_CosineDistanceTables.reset(new float[m_BlockSize * m_NumSubvectors]);
            m_L2DistanceTables.reset(new float[m_BlockSize * m_NumSubvectors]);

            auto cosineDist = DistanceCalcSelector<float>(DistCalcMethod::Cosine);
            auto L2Dist = DistanceCalcSelector<float>(DistCalcMethod::L2);

            for (int i = 0; i < m_NumSubvectors; i++) {
                float* base = m_codebooks.get() + i * m_KsPerSubvector * m_DimPerSubvector;
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    for (int k = 0; k <= j; k++) {
                        m_CosineDistanceTables[m_DistIndexCalc(i, j, k)] = DistanceUtils::ConvertDistanceBackToCosineSimilarity(cosineDist(base + j * m_DimPerSubvector, base + k * m_DimPerSubvector, m_DimPerSubvector));
                        m_L2DistanceTables[m_DistIndexCalc(i, j, k)] = L2Dist(base + j * m_DimPerSubvector, base + k * m_DimPerSubvector, m_DimPerSubvector);
                    }
                }
            }
        }

        PQQuantizer::~PQQuantizer()
        {
        }

        float PQQuantizer::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY)
        {
            if (GetEnableADC() == false) {
                float out = 0;
                for (int i = 0; i < m_NumSubvectors; i++) {
                    out += m_L2DistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
                }
                return out;
            }
            else {
                float out = 0;
                for (int i = 0; i < m_NumSubvectors; i++) {
                    out += pY[i * m_KsPerSubvector + pX[i]] * pY[i * m_KsPerSubvector + pX[i]];
                }
                return out;
            }
        }

        float PQQuantizer::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY)
        {
            float out = 0;
            for (int i = 0; i < m_NumSubvectors; i++) {
                out += m_CosineDistanceTables[m_DistIndexCalc(i, pX[i], pY[i])];
            }
            return DistanceUtils::ConvertCosineSimilarityToDistance(out);
        }

        void PQQuantizer::QuantizeVector(const float* vec, std::uint8_t* vecout)
        {
            auto distCalc = DistanceCalcSelector<float>(DistCalcMethod::L2);

            for (int i = 0; i < m_NumSubvectors; i++) {
                SizeType bestIndex = 0;
                float minDist = std::numeric_limits<float>::infinity();

                const float* subvec = vec + i * m_DimPerSubvector;
                float* basevec = m_codebooks.get() + i * m_KsPerSubvector * m_DimPerSubvector;
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    float dist = distCalc(subvec, basevec + j * m_DimPerSubvector, m_DimPerSubvector);
                    if (dist < minDist) {
                        bestIndex = j;
                        minDist = dist;
                    }
                }
                vecout[i] = bestIndex;
            }
        }

        std::uint64_t PQQuantizer::BufferSize() const
        {
            return sizeof(float) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector + sizeof(DimensionType) + sizeof(SizeType) + sizeof(DimensionType);
        }

        ErrorCode PQQuantizer::SaveQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
        {
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_NumSubvectors);
            IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_KsPerSubvector);
            IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_DimPerSubvector);
            IOBINARY(p_out, WriteBinary, sizeof(float) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector, (char*)m_codebooks.get());
            LOG(Helper::LogLevel::LL_Info, "Saving quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
            return ErrorCode::Success;
        }

        ErrorCode PQQuantizer::LoadQuantizer(std::shared_ptr<Helper::DiskPriorityIO> p_in)
        {
            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_NumSubvectors);
            IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_KsPerSubvector);
            IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_DimPerSubvector);
            m_codebooks.reset(new float[m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector]);
            IOBINARY(p_in, ReadBinary, sizeof(float) * m_NumSubvectors * m_KsPerSubvector * m_DimPerSubvector, (char*)m_codebooks.get());

            m_BlockSize = (m_KsPerSubvector * (m_KsPerSubvector + 1)) / 2;
            m_CosineDistanceTables.reset(new float[m_BlockSize * m_NumSubvectors]);
            m_L2DistanceTables.reset(new float[m_BlockSize * m_NumSubvectors]);

            auto cosineDist = DistanceCalcSelector<float>(DistCalcMethod::Cosine);
            auto L2Dist = DistanceCalcSelector<float>(DistCalcMethod::L2);

            for (int i = 0; i < m_NumSubvectors; i++) {
                float* base = m_codebooks.get() + i * m_KsPerSubvector * m_DimPerSubvector;
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    for (int k = 0; k <= j; k++) {
                        m_CosineDistanceTables[m_DistIndexCalc(i, j, k)] = DistanceUtils::ConvertDistanceBackToCosineSimilarity(cosineDist(base + j * m_DimPerSubvector, base + k * m_DimPerSubvector, m_DimPerSubvector));
                        m_L2DistanceTables[m_DistIndexCalc(i, j, k)] = L2Dist(base + j * m_DimPerSubvector, base + k * m_DimPerSubvector, m_DimPerSubvector);
                    }
                }
            }
            LOG(Helper::LogLevel::LL_Info, "Load quantizer: Subvectors:%d KsPerSubvector:%d DimPerSubvector:%d\n", m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector);
            return ErrorCode::Success;
        }

        DimensionType PQQuantizer::GetNumSubvectors() const
        {
            return m_NumSubvectors;
        }
        SizeType PQQuantizer::GetKsPerSubvector() const
        {
            return m_KsPerSubvector;
        }
        DimensionType PQQuantizer::GetDimPerSubvector() const
        {
            return m_DimPerSubvector;
        }

        bool PQQuantizer::GetEnableADC() const
        {
            return m_EnableADC;
        }

        inline SizeType PQQuantizer::m_DistIndexCalc(SizeType i, SizeType j, SizeType k) {
            if (k > j) {
                return (m_BlockSize * i) + ((k * (k + 1)) / 2) + j; // exploit symmetry by swapping
            }
            return (m_BlockSize * i) + ((j * (j + 1)) / 2) + k;
        }
    }
}