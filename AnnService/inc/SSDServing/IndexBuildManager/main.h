// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <map>
#include "inc/Core/Common.h"
#include "inc/SSDServing/IndexBuildManager/Options.h"
#include "inc/SSDServing/BuildHead/Options.h"
#include "inc/SSDServing/SelectHead_BKT/Options.h"
#include "inc/SSDServing/VectorSearch/Options.h"

namespace SPTAG {
	namespace SSDServing {
		void BuildConfigFromMap(std::map< std::string, std::map<std::string, std::string> >* config_map,
			SelectHead_BKT::Options& select_head_opts,
			BuildHead::Options& build_head_opts,
			VectorSearch::Options& build_ssd_opts,
			VectorSearch::Options& search_ssd_opts
		);
		void BuildConfigFromFile(const char* configurationPath,
			std::map< std::string, std::map<std::string, std::string> >* config_map,
			SelectHead_BKT::Options& select_head_opts,
			BuildHead::Options& build_head_opts,
			VectorSearch::Options& build_ssd_opts,
			VectorSearch::Options& search_ssd_opts
		);
		int BootProgram(const char* configurationPath);

		int BootProgram(bool forANNIndexTestTool,
			std::map<std::string, std::map<std::string, std::string>>* config_map,
			const char* configurationPath = nullptr,
			SPTAG::VectorValueType valueType = SPTAG::VectorValueType::Undefined,
			SPTAG::DistCalcMethod distCalcMethod = SPTAG::DistCalcMethod::Undefined,
			const char* dataFilePath = nullptr,
			const char* indexFilePath = nullptr);

		const std::string SEC_BASE = "Base";
		const std::string SEC_SELECT_HEAD = "SelectHead";
		const std::string SEC_BUILD_HEAD = "BuildHead";
		const std::string SEC_BUILD_SSD_INDEX = "BuildSSDIndex";
		const std::string SEC_SEARCH_SSD_INDEX = "SearchSSDIndex";
	}
}