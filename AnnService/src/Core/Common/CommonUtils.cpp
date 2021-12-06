#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;

template<typename T>
static int Utils::GetBase()
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

#define DefineVectorValueType(Name, Type) template int Utils::GetBase<Type>();
#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
