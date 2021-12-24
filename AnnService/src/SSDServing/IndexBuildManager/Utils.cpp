// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <inc/SSDServing/IndexBuildManager/Utils.h>

bool SPTAG::SSDServing::readSearchSSDSec(const char* iniFile, SPTAG::SSDServing::VectorSearch::Options& opts) {
	SPTAG::Helper::IniReader iniReader;
	iniReader.LoadIniFile(iniFile);
	return readSearchSSDSec(iniReader, opts);
}
bool SPTAG::SSDServing::readSearchSSDSec(const SPTAG::Helper::IniReader& iniReader, SPTAG::SSDServing::VectorSearch::Options& opts) {
	auto& searchSSDParameters = iniReader.GetParameters("SearchSSDIndex");
	bool ret = !searchSSDParameters.empty();
	if (ret)
	{
		for (const auto& iter : searchSSDParameters)
		{
			opts.SetParameter(iter.first.c_str(), iter.second.c_str());
		}

	}
	return ret;
}
