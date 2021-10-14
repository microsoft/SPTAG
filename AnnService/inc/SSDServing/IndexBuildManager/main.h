#pragma once
#include <string>
#include "inc/Core/Common.h"

namespace SPTAG {
	namespace SSDServing {
		int BootProgram(const char* configurationPath);
		int BootProgram(const char* configurationPath, bool forANNIndexTestTool, 
			SPTAG::IndexAlgoType p_algoType = SPTAG::IndexAlgoType::Undefined, 
			SPTAG::VectorValueType valueType = SPTAG::VectorValueType::Undefined, 
			SPTAG::DistCalcMethod distCalcMethod = SPTAG::DistCalcMethod::Undefined, 
			const char* dataFilePath = "", const char* indexFilePath = "");
	}
}