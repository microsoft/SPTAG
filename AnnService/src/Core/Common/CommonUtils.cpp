#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;


#define DefineVectorValueType(Name, Type) template int Utils::GetBase<Type>();
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

template <typename T>
void Utils::BatchNormalize(T* data, SizeType row, DimensionType col, int base, int threads) 
{
#pragma omp parallel for num_threads(threads)
	for (SizeType i = 0; i < row; i++)
	{
		SPTAG::COMMON::Utils::Normalize(data + i * (size_t)col, col, base);
	}
}

#define DefineVectorValueType(Name, Type) template void Utils::BatchNormalize<Type>(Type* data, SizeType row, DimensionType col, int base, int threads);
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType