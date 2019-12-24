#include "inc/SSDServing/Common/stdafx.h"
#include "inc/SSDServing/SelectHead_BKT/BootSelectHead.h"
#include "inc/SSDServing/SelectHead_BKT/BuildBKT.h"
#include "inc/SSDServing/SelectHead_BKT/AnalyzeTree.h"
#include "inc/SSDServing/SelectHead_BKT/SelectHead.h"

using namespace std;

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {
			ErrorCode Bootstrap(Options& opts) {
				auto start = std::chrono::system_clock::now();

				fprintf(stdout, "Start loading vector file.\n");
				BasicVectorSet vectorSet(opts.m_vectorFile.c_str(), opts.m_valueType);
				fprintf(stdout, "Finish loading vector file.\n");

				fprintf(stdout, "Start generating BKT.\n");
				shared_ptr<COMMON::BKTree> bkt;
				switch (opts.m_valueType)
				{
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        bkt = BuildBKT<Type>(vectorSet, opts); \
		break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

				default: break;
				}
				fprintf(stdout, "Finish generating BKT.\n");

				unordered_map<int, int> counter;

				if (opts.m_calcStd)
				{
					CalcLeafSize(0, bkt, counter);
				}

				if (opts.m_analyzeOnly)
				{
					fprintf(stdout, "Analyze Only.\n");

					std::vector<BKTNodeInfo> bktNodeInfos(bkt->size());

					// Always use the first tree
					DfsAnalyze(0, bkt, vectorSet, opts, 0, bktNodeInfos);

					fprintf(stdout, "Analyze Finish.\n");
				}
				else {
					if (SelectHead(vectorSet, bkt, opts, counter) != ErrorCode::Success)
					{
						return ErrorCode::Fail;
					}
				}

				auto end = std::chrono::system_clock::now();
				std::chrono::minutes elapsedMinutes = std::chrono::duration_cast<std::chrono::minutes>(end - start);
				fprintf(stderr, "Total used time: %d minutes (about %.2lf hours).\n", elapsedMinutes.count(), elapsedMinutes.count() / 60.0);

				return ErrorCode::Success;
			}
		}
	}
}