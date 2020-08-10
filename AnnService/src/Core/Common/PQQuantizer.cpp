#include <inc/Core/Common/PQQuantizer.h>
#include <iostream>
#include <fstream>


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

			m_CosineDistanceTables = new float**[NumSubvectors];
			m_L2DistanceTables = new float**[NumSubvectors];

			for (int i = 0; i < m_NumSubvectors; i++) {
				m_CosineDistanceTables[i] = new float* [KsPerSubvector];
				m_L2DistanceTables[i] = new float* [KsPerSubvector];
				for (int j = 0; j < KsPerSubvector; j++) {
					m_CosineDistanceTables[i][j] = new float[KsPerSubvector];
					m_L2DistanceTables[i][j] = new float[KsPerSubvector];
					for (int k = 0; k < KsPerSubvector; k++) {
						m_CosineDistanceTables[i][j][k] = DistanceUtils::ComputeCosineDistance<EnumInstruction::SPTAG_ELSE>(m_codebooks[i][j], m_codebooks[i][k], m_DimPerSubvector);
						m_L2DistanceTables[i][j][k] = DistanceUtils::ComputeL2Distance<EnumInstruction::SPTAG_ELSE>(m_codebooks[i][j], m_codebooks[i][k], m_DimPerSubvector);
					}
				}
			}
		}

		float PQQuantizer::L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) 
		{
			float out = 0;
			for (int i = 0; i < m_NumSubvectors; i++) {
				out += m_CosineDistanceTables[i][pX[i]][pY[i]];
			}
			return out;
		}

		float PQQuantizer::CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY)
		{
			float out = 0;
			for (int i = 0; i < m_NumSubvectors; i++) {
				out += m_L2DistanceTables[i][pX[i]][pY[i]];
			}
			return out;
		}

		const std::uint8_t* PQQuantizer::QuantizeVector(const float* vec)
		{
			std::uint8_t* out = new std::uint8_t[m_NumSubvectors];
			
			for (int i = 0; i < m_NumSubvectors; i++) {
				float minDist = FLT_MIN;
				SizeType bestIndex = 0;
				float* subvec = new float[m_DimPerSubvector];
				for (int j = 0; j < m_DimPerSubvector; j++) {
					subvec[j] = vec[i * m_DimPerSubvector + j];
				}
				for (int j = 0; j < m_KsPerSubvector; j++) {
					float dist = DistanceUtils::ComputeL2Distance<EnumInstruction::SPTAG_ELSE>((const float*)subvec, m_codebooks[i][j], m_DimPerSubvector);
					if (dist < minDist) {
						minDist = dist;
						bestIndex = j;
					}
				}
				out[i] = bestIndex;
			}
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

			PQQuantizer quantizer = PQQuantizer::PQQuantizer(m_NumSubvectors, m_KsPerSubvector, m_DimPerSubvector, m_codebooks);
			DistanceUtils::PQQuantizer = &quantizer;
		}
	}
}