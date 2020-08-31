// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUANTIZER_H_
#define _SPTAG_COMMON_QUANTIZER_H_

#include "CommonUtils.h"

namespace SPTAG
{
	namespace COMMON
	{
		class Quantizer
		{
		public:
			virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) = 0;

			virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) = 0;

			virtual const std::uint8_t* QuantizeVector(const float* vec) = 0;

			virtual void SaveQuantizer(std::string path) = 0;

			static void LoadQuantizer(std::string path, QuantizerType quantizerType);

			virtual QuantizerType GetQuantizerType() = 0;
		};
	}
}

#endif // _SPTAG_COMMON_QUANTIZER_H_