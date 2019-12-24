#include "inc/SSDServing/Common/stdafx.h"
#include "inc/SSDServing/BuildHead/BootBuildHead.h"

namespace SPTAG {
	namespace SSDServing {
		namespace BuildHead {
			ErrorCode Bootstrap(Options& options) {
                auto indexBuilder = SPTAG::VectorIndex::CreateInstance(options.m_indexAlgoType, options.m_inputValueType);

                SPTAG::Helper::IniReader iniReader;
                if (!options.m_builderConfigFile.empty())
                {
                    iniReader.LoadIniFile(options.m_builderConfigFile);
                }
                // if we don't have one, set by ourselves.
                if (!iniReader.DoesParameterExist("Index", "NumberOfThreads")) {
                    iniReader.SetParameter("Index", "NumberOfThreads", std::to_string(options.m_threadNum));
                }
                for (const auto& iter : iniReader.GetParameters("Index"))
                {
                    indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str());
                }

                SPTAG::ErrorCode code;

                {
                    std::vector<std::string> files = SPTAG::Helper::StrUtils::SplitString(options.m_inputFiles, ",");
                    std::ifstream inputStream(files[0], std::ifstream::binary);
                    if (!inputStream.is_open()) {
                        fprintf(stderr, "Failed to read input file.\n");
                        return ErrorCode::Fail;
                    }
                    SPTAG::SizeType row;
                    SPTAG::DimensionType col;
                    inputStream.read((char*)&row, sizeof(SPTAG::SizeType));
                    inputStream.read((char*)&col, sizeof(SPTAG::DimensionType));
                    std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(options.m_inputValueType)) * row * col;
                    SPTAG::ByteArray vectorSet = SPTAG::ByteArray::Alloc(totalRecordVectorBytes);
                    char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());
                    inputStream.read(vecBuf, totalRecordVectorBytes);
                    inputStream.close();
                    std::shared_ptr<SPTAG::VectorSet> p_vectorSet(new SPTAG::BasicVectorSet(vectorSet, options.m_inputValueType, col, row));

                    std::shared_ptr<SPTAG::MetadataSet> p_metaSet = nullptr;
                    if (files.size() >= 3) {
                        p_metaSet.reset(new SPTAG::FileMetadataSet(files[1], files[2]));
                    }
                    code = indexBuilder->BuildIndex(p_vectorSet, p_metaSet);
                    indexBuilder->SaveIndex(options.m_outputFolder);
                }

                if (SPTAG::ErrorCode::Success != code)
                {
                    fprintf(stderr, "Failed to build index.\n");
                    return ErrorCode::Fail;
                }
				return ErrorCode::Success;
			}
		}
	}
}
