// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/VectorSetReader.h"

#include <inc/Core/Common/DistanceUtils.h>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>

#ifdef RABITQ
#include "inc/Core/Common/RaBitQAutoTuner.h"
#endif

using namespace SPTAG;

#ifdef RABITQ
namespace
{
bool CreateBuildDirectory(const std::filesystem::path& p_final,
                          std::filesystem::path& p_build,
                          std::string& p_error)
{
    namespace fs = std::filesystem;
    std::error_code error;
    fs::create_directories(p_final.parent_path(), error);
    if (error) {
        p_error = "cannot create index parent directory: " + error.message();
        return false;
    }
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 100; ++attempt) {
        p_build = p_final.parent_path() /
            (p_final.filename().string() + ".rabitq-building-" +
             std::to_string(nonce) + "-" + std::to_string(attempt));
        if (fs::create_directory(p_build, error)) {
            return true;
        }
        if (error) {
            p_error = "cannot create unique RaBitQ build directory: " + error.message();
            return false;
        }
    }
    p_error = "cannot allocate a unique RaBitQ build directory";
    return false;
}

bool PublishBuildDirectory(const std::filesystem::path& p_build,
                           const std::filesystem::path& p_final,
                           std::string& p_error)
{
    namespace fs = std::filesystem;
    std::error_code error;
    fs::path backup = p_build;
    backup += ".previous";
    const bool hadPrevious = fs::exists(p_final, error);
    if (error) {
        p_error = "cannot inspect existing index directory: " + error.message();
        return false;
    }
    if (hadPrevious) {
        fs::rename(p_final, backup, error);
        if (error) {
            p_error = "cannot preserve existing index directory: " + error.message();
            return false;
        }
    }
    fs::rename(p_build, p_final, error);
    if (error) {
        const std::string publishError = error.message();
        if (hadPrevious) {
            std::error_code restoreError;
            fs::rename(backup, p_final, restoreError);
            if (restoreError) {
                p_error = "cannot publish new index (" + publishError +
                    ") or restore previous index (" + restoreError.message() + ")";
                return false;
            }
        }
        std::error_code cleanupError;
        fs::remove_all(p_build, cleanupError);
        p_error = "cannot publish new index directory: " + publishError;
        if (cleanupError) {
            p_error += "; cannot remove failed staging directory: " +
                cleanupError.message();
        }
        return false;
    }
    if (hadPrevious) {
        fs::remove_all(backup, error);
        if (error) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                         "Published index but could not remove previous index backup %s: %s\n",
                         backup.string().c_str(), error.message().c_str());
        }
    }
    return true;
}

bool FinalizeStagedConfig(const std::filesystem::path& p_build,
                          const std::filesystem::path& p_final,
                          std::string& p_error)
{
    const std::filesystem::path configPath = p_build / "indexloader.ini";
    std::ifstream input(configPath, std::ios::binary);
    if (!input) {
        p_error = "cannot open staged indexloader.ini";
        return false;
    }
    std::string config(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const std::string stagedRoot = p_build.string();
    const std::string finalRoot = p_final.string();
    std::size_t offset = 0;
    int replacements = 0;
    while ((offset = config.find(stagedRoot, offset)) != std::string::npos) {
        config.replace(offset, stagedRoot.size(), finalRoot);
        offset += finalRoot.size();
        ++replacements;
    }
    if (replacements == 0) {
        p_error = "staged indexloader.ini does not reference its build directory";
        return false;
    }
    input.close();
    std::ofstream output(configPath, std::ios::binary | std::ios::trunc);
    output.write(config.data(), static_cast<std::streamsize>(config.size()));
    if (!output) {
        p_error = "cannot finalize staged indexloader.ini";
        return false;
    }
    return true;
}
} // namespace
#endif

class BuilderOptions : public Helper::ReaderOptions
{
  public:
    BuilderOptions()
        : Helper::ReaderOptions(
              VectorValueType::Float, 0, VectorFileType::TXT, "|", 32, false, false)
    {
        AddRequiredOption(m_outputFolder, "-o", "--outputfolder", "Output folder.");
        AddRequiredOption(m_indexAlgoType, "-a", "--algo", "Index Algorithm type.");
        AddOptionalOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddOptionalOption(m_builderConfigFile, "-c", "--config", "Config file for builder.");
        AddOptionalOption(m_quantizerFile, "-pq", "--quantizer", "Quantizer File");
        AddOptionalOption(m_metaMapping, "-m", "--metaindex", "Enable delete vectors through metadata");
    }

    ~BuilderOptions()
    {
    }

    std::string m_inputFiles;

    std::string m_outputFolder;

    SPTAG::IndexAlgoType m_indexAlgoType;

    std::string m_builderConfigFile;

    std::string m_quantizerFile;

    bool m_metaMapping = false;
};

int main(int argc, char *argv[])
{
    std::shared_ptr<BuilderOptions> options(new BuilderOptions);
    if (!options->Parse(argc - 1, argv + 1))
    {
        exit(1);
    }
    Helper::IniReader iniReader;
    if (!options->m_builderConfigFile.empty() &&
        iniReader.LoadIniFile(options->m_builderConfigFile) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open index configure file!\n");
        return -1;
    }

    const bool configuredAutoTune = iniReader.DoesSectionExist("RaBitQAutoTune") &&
        iniReader.GetParameter("RaBitQAutoTune", "isExecute", false);
    for (int i = 1; i < argc; i++)
    {
        std::string param(argv[i]);
        size_t idx = param.find("=");
        if (idx == std::string::npos)
            continue;

        std::string paramName = param.substr(0, idx);
        std::string paramVal = param.substr(idx + 1);
        std::string sectionName;
        idx = paramName.find(".");
        const bool tunerOverride =
            idx != std::string::npos &&
            Helper::StrUtils::StrEqualIgnoreCase(
                paramName.substr(0, idx).c_str(), "RaBitQAutoTune");
        if (idx != std::string::npos && (configuredAutoTune || tunerOverride))
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "RaBitQ auto-tuning parameters must come only from the INI; "
                "command-line override %s is not allowed.\n",
                param.c_str());
            return 1;
        }
        if (idx != std::string::npos)
        {
            sectionName = paramName.substr(0, idx);
            paramName = paramName.substr(idx + 1);
        }
        iniReader.SetParameter(sectionName, paramName, paramVal);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set [%s]%s = %s\n", sectionName.c_str(), paramName.c_str(),
                     paramVal.c_str());
    }

    std::string quantizerFile = options->m_quantizerFile;
    VectorValueType builderValueType = options->m_inputValueType;
    if (options->m_inputFiles.empty() &&
        iniReader.DoesParameterExist("Base", "ValueType"))
    {
        builderValueType =
            iniReader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
        if (builderValueType == VectorValueType::Undefined)
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error, "Invalid [Base] ValueType.\n");
            return 1;
        }
    }
    const bool autoTuneEnabled = configuredAutoTune;
#ifdef RABITQ
    std::filesystem::path autoTuneBuildDirectory;
    std::filesystem::path autoTuneFinalDirectory;
#endif
    if (autoTuneEnabled)
    {
#ifdef RABITQ
        if (options->m_indexAlgoType != IndexAlgoType::SPANN)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "RaBitQ auto-tuning is supported only for SPANN index construction.\n");
            return 1;
        }
        if (!options->m_inputFiles.empty() || !options->m_quantizerFile.empty())
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "RaBitQ auto-tuning reads vectors and creates its quantizer from the INI; "
                "--input and --quantizer are not allowed.\n");
            return 1;
        }
        const std::string configuredIndexDirectory =
            iniReader.GetParameter("Base", "IndexDirectory", std::string());
        if (configuredIndexDirectory.empty())
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "[Base] IndexDirectory is required for RaBitQ auto-tuning.\n");
            return 1;
        }
        autoTuneFinalDirectory =
            std::filesystem::absolute(configuredIndexDirectory).lexically_normal();
        const auto commandOutput =
            std::filesystem::absolute(options->m_outputFolder).lexically_normal();
        if (commandOutput != autoTuneFinalDirectory)
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "--outputfolder must match the authoritative [Base] IndexDirectory "
                "when RaBitQ auto-tuning is enabled.\n");
            return 1;
        }
        std::string stagingError;
        if (!CreateBuildDirectory(
                autoTuneFinalDirectory, autoTuneBuildDirectory, stagingError))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%s\n", stagingError.c_str());
            return 1;
        }
        COMMON::RaBitQAutoTuneResult tuneResult;
        std::string tuneError;
        ErrorCode tuneStatus = ErrorCode::Fail;
        try
        {
            tuneStatus =
                COMMON::RaBitQAutoTuner::Run(
                    iniReader, autoTuneBuildDirectory.string(), tuneResult, tuneError);
        }
        catch (const std::exception& exception)
        {
            tuneError = exception.what();
        }
        if (tuneStatus != ErrorCode::Success)
        {
            std::error_code cleanupError;
            std::filesystem::remove_all(autoTuneBuildDirectory, cleanupError);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "RaBitQ auto-tuning failed: %s\n", tuneError.c_str());
            return 1;
        }

        builderValueType = VectorValueType::UInt8;
        quantizerFile = tuneResult.quantizerPath;
        options->m_inputFiles.clear();
        iniReader.SetParameter("Base", "ValueType", "UInt8");
        iniReader.SetParameter("Base", "Dim", std::to_string(tuneResult.codeDimension));
        iniReader.SetParameter("Base", "VectorPath", tuneResult.vectorPath);
        iniReader.SetParameter("Base", "VectorType", "DEFAULT");
        iniReader.SetParameter("Base", "VectorSize", std::to_string(tuneResult.vectorCount));
        iniReader.SetParameter("Base", "QuantizerFilePath", tuneResult.quantizerPath);
        iniReader.SetParameter(
            "Base", "IndexDirectory", autoTuneBuildDirectory.string());
        iniReader.SetParameter("BuildSSDIndex", "EnableADC", "true");
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[RaBitQAutoTune] isExecute=true requires a build configured with RABITQ=ON.\n");
        return 1;
#endif
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set QuantizerFile = %s\n", quantizerFile.c_str());
    auto indexBuilder = VectorIndex::CreateInstance(options->m_indexAlgoType, builderValueType);
    if (!indexBuilder)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot create index builder.\n");
        return 1;
    }
    if (!quantizerFile.empty())
    {
        if (indexBuilder->LoadQuantizer(quantizerFile) != ErrorCode::Success ||
            !indexBuilder->m_pQuantizer)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load quantizer file.\n");
            return 1;
        }
    }

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex", "Index"};
    for (int i = 0; i < 5; i++)
    {
        if (!iniReader.DoesParameterExist(sections[i], "NumberOfThreads"))
        {
            iniReader.SetParameter(sections[i], "NumberOfThreads", std::to_string(options->m_threadNum));
        }
        for (const auto &iter : iniReader.GetParameters(sections[i]))
        {
            indexBuilder->SetParameter(iter.first.c_str(), iter.second.c_str(), sections[i]);
        }
    }
    ErrorCode code;
    std::shared_ptr<VectorSet> vecset;
    if (options->m_inputFiles != "")
    {
        if (options->m_dimension <= 0)
        {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "--dimension is required when indexbuilder reads --input directly.\n");
            return 1;
        }
        auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
        if (ErrorCode::Success != vectorReader->LoadFile(options->m_inputFiles))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
            exit(1);
        }
        vecset = vectorReader->GetVectorSet();
        code = indexBuilder->BuildIndex(vecset, vectorReader->GetMetadataSet(), options->m_metaMapping,
                                        options->m_normalized, true);
    }
    else
    {
        if (!quantizerFile.empty())
        {
            indexBuilder->SetQuantizerFileName(
                quantizerFile.substr(quantizerFile.find_last_of("/\\") + 1));
        }
        code = indexBuilder->BuildIndex(options->m_normalized);
    }
    if (code == ErrorCode::Success)
    {
        for (const auto& parameter : iniReader.GetParameters("SearchSSDIndex"))
        {
            std::string name = parameter.first;
            if (Helper::StrUtils::StrEqualIgnoreCase(name.c_str(), "isExecute") ||
                Helper::StrUtils::StrEqualIgnoreCase(name.c_str(), "BuildSsdIndex"))
            {
                continue;
            }
            if (Helper::StrUtils::StrEqualIgnoreCase(name.c_str(), "PostingPageLimit"))
            {
                name = "SearchPostingPageLimit";
            }
            else if (Helper::StrUtils::StrEqualIgnoreCase(name.c_str(), "InternalResultNum"))
            {
                name = "SearchInternalResultNum";
            }
            indexBuilder->SetParameter(
                name.c_str(), parameter.second.c_str(), "BuildSSDIndex");
        }
        std::string saveFolder = options->m_outputFolder;
#ifdef RABITQ
        if (autoTuneEnabled) {
            saveFolder = autoTuneBuildDirectory.string();
        }
#endif
        code = indexBuilder->SaveIndex(saveFolder);
        if (code != ErrorCode::Success)
        {
#ifdef RABITQ
            if (autoTuneEnabled) {
                std::error_code cleanupError;
                std::filesystem::remove_all(autoTuneBuildDirectory, cleanupError);
            }
#endif
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save index.\n");
            return 1;
        }
#ifdef RABITQ
        if (autoTuneEnabled)
        {
            std::string finalizeError;
            if (!FinalizeStagedConfig(
                    autoTuneBuildDirectory, autoTuneFinalDirectory, finalizeError))
            {
                std::error_code cleanupError;
                std::filesystem::remove_all(autoTuneBuildDirectory, cleanupError);
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Failed to finalize auto-tuned index configuration: %s\n",
                    finalizeError.c_str());
                return 1;
            }
            std::string publishError;
            if (!PublishBuildDirectory(
                    autoTuneBuildDirectory, autoTuneFinalDirectory, publishError))
            {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Failed to publish auto-tuned index: %s\n", publishError.c_str());
                return 1;
            }
        }
#endif
    }
    else
    {
#ifdef RABITQ
        if (autoTuneEnabled) {
            std::error_code cleanupError;
            std::filesystem::remove_all(autoTuneBuildDirectory, cleanupError);
        }
#endif
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build index.\n");
        exit(1);
    }
    return 0;
}
