#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;

template<typename T>
static inline int Utils::GetBase()
{
	if (DistanceUtils::Quantizer)
	{
		return DistanceUtils::Quantizer->GetBase();
	}
	else
	{
		return Utils::GetBaseCore<T>();
	}
}