#include "inc/SSDServing/VectorSearch/BootVectorSearch.h"
#include "inc/SSDServing/VectorSearch/BuildSsdIndex.h"
#include "inc/SSDServing/VectorSearch/SearchSsdIndex.h"
#include "inc/Helper/SimpleIniReader.h"

namespace SPTAG {

	
	std::function<std::shared_ptr<Helper::DiskPriorityIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskPriorityIO> { return std::shared_ptr<Helper::DiskPriorityIO>(new SSDServing::VectorSearch::AsyncFileIO()); };
	
	namespace SSDServing {
		namespace VectorSearch {

			ErrorCode Bootstrap(Options& opts) {
                if (opts.m_buildSsdIndex)
                {
					LOG(Helper::LogLevel::LL_Info, "Start building SSD Index.\n");
					if (false) {}
#define DefineVectorValueType(Name, Type) \
else if (COMMON_OPTS.m_valueType == VectorValueType::Name) { \
BuildSsdIndex<Type>(opts); \
} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
				}
				else {
					LOG(Helper::LogLevel::LL_Info, "Start searching SSD Index.\n");
					if (false) {}
#define DefineVectorValueType(Name, Type) \
else if (COMMON_OPTS.m_valueType == VectorValueType::Name) { \
Search<Type>(opts); \
} \

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
				}
				
				return ErrorCode::Success;
			}
		}
	}
}