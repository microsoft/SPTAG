#include <inc/Core/Common/PQQuantizer.h>


namespace SPTAG
{
	namespace COMMON
	{
		PQQuantizer::PQQuantizer(DimensionType NumSubvectors, SizeType KsPerSubvector, DimensionType DimPerSubvector, const float*** Codebooks)
		{
			m_NumSubvectors = NumSubvectors;
			m_KsPerSubvector = KsPerSubvector;
			m_DimPerSubvector = DimPerSubvector;
			m_codebooks = Codebooks;

			m_CosineDistanceTables = new float**[NumSubvectors];
			m_L2DistanceTables = new float**[NumSubvectors];

			for (int i = 0; i < m_NumSubvectors; i++) {
				m_CosineDistanceTables[i] = new float* [KsPerSubvector];
				m_L2DistanceTables[i] = new float* [KsPerSubvector];
				for (int j = 0; j < KsPerSubvector; j++) {
					m_CosineDistanceTables[i][j] = new float[KsPerSubvector];
					m_L2DistanceTables[i][j] = new float[KsPerSubvector];
					for (int k = 0; k < KsPerSubvector; k++) {
						m_CosineDistanceTables[i][j][k] = DistanceUtils::ComputeCosineDistance<EnumInstruction::SPTAG_AVX2>(m_codebooks[i][j], m_codebooks[i][k], m_DimPerSubvector);
						m_L2DistanceTables[i][j][k] = DistanceUtils::ComputeL2Distance<EnumInstruction::SPTAG_AVX2>(m_codebooks[i][j], m_codebooks[i][k], m_DimPerSubvector);
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
					float dist = DistanceUtils::ComputeL2Distance<EnumInstruction::SPTAG_AVX2>(subvec, m_codebooks[i][j], m_DimPerSubvector);
					if (dist < minDist) {
						minDist = dist;
						bestIndex = j;
					}
				}
				out[i] = bestIndex;
			}
			return out;
		}
	}
}