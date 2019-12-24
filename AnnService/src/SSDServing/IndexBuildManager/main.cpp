#include "inc/SSDServing/Common/stdafx.h"
#include "inc/SSDServing/SelectHead_BKT/BootSelectHead.h"
#include "inc/SSDServing/SelectHead_BKT/Options.h"
#include "inc/SSDServing/BuildHead/BootBuildHead.h"
#include "inc/SSDServing/BuildHead/Options.h"
using namespace SPTAG;

namespace SPTAG {
	namespace SSDServing {
		int internalMain(int argc, char* argv[]) {
			if (argc < 2)
			{
				fprintf(stderr,
					"SSDServing.exe configFilePath\n");
			}

			Helper::IniReader iniReader;
			iniReader.LoadIniFile(argv[1]);

			SSDServing::SelectHead_BKT::Options slOpts;
			for (const auto& iter : iniReader.GetParameters("SelectHead"))
			{
				slOpts.SetParameter(iter.first.c_str(), iter.second.c_str());
			}
			SSDServing::SelectHead_BKT::Bootstrap(slOpts);

			SSDServing::BuildHead::Options bhOpts;
			for (const auto& iter : iniReader.GetParameters("BuildHead"))
			{
				bhOpts.SetParameter(iter.first.c_str(), iter.second.c_str());
			}
			SSDServing::BuildHead::Bootstrap(bhOpts);
			return 0;
		}
	}
}