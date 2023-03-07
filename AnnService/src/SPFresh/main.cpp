// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Core/Common/TruthSet.h"

#include "inc/SPFresh/SPFresh.h"
 
using namespace SPTAG;

// switch between exe and static library by _$(OutputType) 
#ifdef _exe

int main(int argc, char* argv[]) {
	if (argc < 2)
	{
		LOG(Helper::LogLevel::LL_Error,
			"spfresh storePath\n");
		exit(-1);
	}

	auto ret = SSDServing::SPFresh::UpdateTest(argv[1]);
	return ret;
}

#endif
