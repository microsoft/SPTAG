#include <inc/Core/Common/PQQuantizer.h>
#include <iostream>
#include <fstream>
#include <limits>


namespace SPTAG
{
    namespace COMMON
    {

        PQQuantizer::PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, float*** Codebooks)
        {
            m_NumSubvectors = NumSubvectors;
            m_KsPerSubvector = KsPerSubvector;
            m_DimPerSubvector = DimPerSubvector;
            m_codebooks = (const float***) Codebooks;
            m_BlockSize = (m_KsPerSubvector * (m_KsPerSubvector + 1))/2;

            m_CosineDistanceTables = new float[m_BlockSize * m_NumSubvectors];
            m_L2DistanceTables = new float[m_BlockSize * m_NumSubvectors];
            

            auto cosineDist = DistanceCalcSelector<float>(DistCalcMethod::Cosine);
            auto L2Dist = DistanceCalcSelector<float>(DistCalcMethod::L2);

            for (int i = 0; i < m_NumSubvectors; i++) {
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    for (int k = 0; k <= j; k++) {
                        m_CosineDistanceTables[m_DistIndexCalc(i,j,k)] = DistanceUtils::ConvertDistanceBackToCosineSimilarity(cosineDist(m_codebooks[i][j], m_codebooks[i][k], m_DimPerSubvector));
                        m_L2DistanceTables[m_DistIndexCalc(i,j,k)] = L2Dist(m_codebooks[i][j], m_codebooks[i][k], m_DimPerSubvector);
                    }
                }
            }
        }

        PQQuantizer::~PQQuantizer() {
            
            delete[] m_CosineDistanceTables;
            delete[] m_L2DistanceTables;
            
            for (int i = 0; i < m_NumSubvectors; i++) {
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    delete[] m_codebooks[i][j];
                }
                delete[] m_codebooks[i];
            }
            delete[] m_codebooks;
        }

        float PQQuantizer::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) 
        {
            float out = 0;
            for (int i = 0; i < m_NumSubvectors; i++) {
                out += m_L2DistanceTables[m_DistIndexCalc(i,pX[i],pY[i])];
            }
            return out;
        }

        float PQQuantizer::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY)
        {
            float out = 0;
            for (int i = 0; i < m_NumSubvectors; i++) {
                out += m_CosineDistanceTables[m_DistIndexCalc(i,pX[i],pY[i])];
            }
            return DistanceUtils::ConvertCosineSimilarityToDistance(out);
        }

        const std::uint8_t* PQQuantizer::QuantizeVector(const float* vec)
        {
            std::uint8_t* out = new std::uint8_t[m_NumSubvectors];
            auto distCalc = DistanceCalcSelector<float>(DistCalcMethod::L2);
            float* subvec = new float[m_DimPerSubvector];

            for (int i = 0; i < m_NumSubvectors; i++) {
                float minDist = std::numeric_limits<float>::infinity();
                SizeType bestIndex = 0;			
                for (int j = 0; j < m_DimPerSubvector; j++) {
                    subvec[j] = vec[i * m_DimPerSubvector + j];
                }
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    float dist = distCalc((const float*)subvec, m_codebooks[i][j], m_DimPerSubvector);
                    if (dist < minDist) {
                        minDist = dist;
                        bestIndex = j;
                    }
                }
                out[i] = bestIndex;
            }
            delete[] subvec;
            return out;
        }

        void PQQuantizer::SaveQuantizer(std::string path)
        {
            std::ofstream fl(path, std::ios::out | std::ios::binary);
            if (!fl) {
                std::cout << "Could not open quantizer save file" << std::endl;
                return;
            }

            fl.write((char*)&m_NumSubvectors, sizeof(DimensionType));
            fl.write((char*)&m_KsPerSubvector, sizeof(SizeType));
            fl.write((char*)&m_DimPerSubvector, sizeof(DimensionType));

            std::cout << "Saving quantizer with parameters:" << std::endl;
            std::cout << "Subvectors: " << m_NumSubvectors << std::endl;
            std::cout << "KsPerSubvector: " << m_KsPerSubvector << std::endl;
            std::cout << "DimPerSubvector: " << m_DimPerSubvector << std::endl;

            for (int i = 0; i < m_NumSubvectors; i++) {
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    for (int k = 0; k < m_DimPerSubvector; k++) {
                        fl.write((char*)&(m_codebooks[i][j][k]), sizeof(float));
                    }
                }
            }
        }

        void PQQuantizer::LoadQuantizer(std::string path)
        {
            std::ifstream fl(path, std::ios::in | std::ios::binary);
            if (!fl) {
                std::cout << "Could not open quantizer file" << std::endl;
                return;
            }
            DimensionType m_NumSubvectors;
            SizeType m_KsPerSubvector;
            DimensionType m_DimPerSubvector;
            fl.read((char*)&m_NumSubvectors, sizeof(DimensionType));
            fl.read((char*)&m_KsPerSubvector, sizeof(SizeType));
            fl.read((char*)&m_DimPerSubvector, sizeof(DimensionType));

            float*** m_codebooks = new float** [m_NumSubvectors];

            for (int i = 0; i < m_NumSubvectors; i++) {
                m_codebooks[i] = new float* [m_KsPerSubvector];
                for (int j = 0; j < m_KsPerSubvector; j++) {
                    m_codebooks[i][j] = new float[m_DimPerSubvector];
                    for (int k = 0; k < m_DimPerSubvector; k++) {
                        fl.read((char*)&(m_codebooks[i][j][k]), sizeof(float));
                    }
                }
            }

            DistanceUtils::Quantizer = std::static_pointer_cast<Quantizer>(std::make_shared<PQQuantizer>(m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector, m_codebooks));
        }
        DimensionType PQQuantizer::GetNumSubvectors()
        {
            return m_NumSubvectors;
        }
        SizeType PQQuantizer::GetKsPerSubvector()
        {
            return m_KsPerSubvector;
        }
        DimensionType PQQuantizer::GetDimPerSubvector()
        {
            return m_DimPerSubvector;
        }

        inline SizeType PQQuantizer::m_DistIndexCalc(SizeType i, SizeType j, SizeType k) {
            if (k > j) {
                return (m_BlockSize * i) + ((k * (k + 1)) / 2) + j; // exploit symmetry by swapping
            }
            return (m_BlockSize * i) + ((j*(j+1))/2) + k;
        }
    }
}