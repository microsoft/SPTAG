#include <inc/SSDServing/IndexBuildManager/Utils.h>

SPTAG::SSDServing::Neighbor::Neighbor(SPTAG::SizeType k, float d) : key(k), dist(d) {}

SPTAG::SSDServing::Neighbor::Neighbor(const SPTAG::SSDServing::Neighbor& o) : key(o.key), dist(o.dist) {}

bool SPTAG::SSDServing::Neighbor::operator < (const SPTAG::SSDServing::Neighbor& another) const
{
	return this->dist == another.dist ? this->key < another.key : this->dist < another.dist;
}

void SPTAG::SSDServing::writeTruthFile(const std::string truthFile, size_t queryNumber, const int K, std::vector<std::vector<SPTAG::SizeType>>& truthset, std::vector<std::vector<float>>& distset, SPTAG::TruthFileType TFT) {
	auto ptr = SPTAG::f_createIO();
	if (TFT == SPTAG::TruthFileType::TXT)
	{
		if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::out)) {
			LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", truthFile.c_str());
			exit(1);
		}
		for (size_t i = 0; i < queryNumber; i++)
		{
			for (size_t k = 0; k < K; k++)
			{
				if (ptr->WriteString((std::to_string(truthset[i][k]) + " ").c_str()) == 0) {
					LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
					exit(1);
				}
			}
			if (ptr->WriteString("\n") == 0) {
				LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
				exit(1);
			}
		}
	}
	else if (TFT == SPTAG::TruthFileType::XVEC)
	{
		if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::out | std::ios::binary)) {
			LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", truthFile.c_str());
			exit(1);
		}

		for (size_t i = 0; i < queryNumber; i++)
		{
			if (ptr->WriteBinary(sizeof(K), (char*)&K) != sizeof(K) || ptr->WriteBinary(K * 4, (char*)(truthset[i].data())) != K * 4) {
				LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
				exit(1);
			}
		}
	}
	else if (TFT == SPTAG::TruthFileType::DEFAULT) {
		if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::out | std::ios::binary)) {
			LOG(Helper::LogLevel::LL_Error, "Fail to create the file:%s\n", truthFile.c_str());
			exit(1);
		}

		int int32_queryNumber = (int)queryNumber;
		ptr->WriteBinary(4, (char*)&int32_queryNumber);
		ptr->WriteBinary(4, (char*)&K);

		for (size_t i = 0; i < queryNumber; i++)
		{
			if (ptr->WriteBinary(K * 4, (char*)(truthset[i].data())) != K * 4) {
				LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
				exit(1);
			}
		}
		for (size_t i = 0; i < queryNumber; i++)
		{
			if (ptr->WriteBinary(K * 4, (char*)(distset[i].data())) != K * 4) {
				LOG(Helper::LogLevel::LL_Error, "Fail to write the truth file!\n");
				exit(1);
			}
		}
	}
	else {
		LOG(Helper::LogLevel::LL_Error, "Found unsupported file type for generating truth.");
		exit(-1);
	}
}

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
