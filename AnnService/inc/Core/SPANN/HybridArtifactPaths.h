// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HYBRID_ARTIFACT_PATHS_H_
#define _SPTAG_SPANN_HYBRID_ARTIFACT_PATHS_H_

#include "Options.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace SPTAG
{
namespace SPANN
{

inline std::string HybridArtifactComparablePath(
    std::string p_value)
{
#ifdef _WIN32
    std::transform(
        p_value.begin(), p_value.end(),
        p_value.begin(),
        [](unsigned char p_char) {
            return static_cast<char>(
                std::tolower(p_char));
        });
#endif
    return p_value;
}

inline std::filesystem::path NormalizeHybridArtifactPath(
    const std::filesystem::path& p_root,
    const std::string& p_path)
{
    std::filesystem::path path(p_path);
    if (path.is_relative()) path = p_root / path;
    std::error_code error;
    const auto normalized =
        std::filesystem::weakly_canonical(path, error);
    if (!error) return normalized;
    error.clear();
    const auto absolute =
        std::filesystem::absolute(path, error);
    return (error ? path : absolute).lexically_normal();
}

inline bool HybridArtifactPathsAlias(
    const std::filesystem::path& p_left,
    const std::filesystem::path& p_right)
{
    if (p_left == p_right) return true;
#ifdef _WIN32
    if (HybridArtifactComparablePath(
            p_left.generic_string()) ==
        HybridArtifactComparablePath(
            p_right.generic_string())) {
        return true;
    }
#endif
    std::error_code error;
    const bool leftExists =
        std::filesystem::exists(p_left, error);
    if (error || !leftExists) return false;
    error.clear();
    const bool rightExists =
        std::filesystem::exists(p_right, error);
    if (error || !rightExists) return false;
    error.clear();
    return std::filesystem::equivalent(
               p_left, p_right, error) &&
        !error;
}

inline bool IsPrimaryStaticShardPath(
    const std::filesystem::path& p_candidate,
    const std::filesystem::path& p_primary)
{
    if (HybridArtifactComparablePath(
            p_candidate.parent_path().generic_string()) !=
        HybridArtifactComparablePath(
            p_primary.parent_path().generic_string())) {
        return false;
    }
    const std::string prefix =
        HybridArtifactComparablePath(
            p_primary.filename().string() + "_");
    const std::string candidate =
        HybridArtifactComparablePath(
            p_candidate.filename().string());
    if (candidate.size() <= prefix.size() ||
        candidate.compare(
            0, prefix.size(), prefix) != 0) {
        return false;
    }
    for (size_t index = prefix.size();
         index < candidate.size(); ++index) {
        if (!std::isdigit(
                static_cast<unsigned char>(
                    candidate[index]))) {
            return false;
        }
    }
    return true;
}

inline bool ValidateHybridArtifactPaths(
    const Options& p_options,
    const std::string& p_indexRoot,
    std::string& p_error)
{
    p_error.clear();
    const auto validBasename =
        [](const std::string& name) {
            return !name.empty() &&
                name != "." && name != ".." &&
                name.find('/') ==
                    std::string::npos &&
                name.find('\\') ==
                    std::string::npos;
        };
    if (!validBasename(
            p_options.m_hybridPostingFile) ||
        !validBasename(
            p_options.m_hybridHeadGraphFile)) {
        p_error =
            "hybrid posting and graph files must be basenames";
        return false;
    }

    std::error_code currentPathError;
    std::filesystem::path currentPath =
        std::filesystem::current_path(
            currentPathError);
    if (currentPathError) currentPath = ".";
    const std::filesystem::path root =
        NormalizeHybridArtifactPath(
            currentPath,
            p_indexRoot);
    const std::filesystem::path headRoot =
        NormalizeHybridArtifactPath(
            root, p_options.m_headIndexFolder);
    const auto posting =
        NormalizeHybridArtifactPath(
            root,
            p_options.m_hybridPostingFile);
    const auto graph =
        NormalizeHybridArtifactPath(
            headRoot,
            p_options.m_hybridHeadGraphFile);
    std::vector<std::filesystem::path> generated = {
        posting,
        NormalizeHybridArtifactPath(
            root,
            p_options.m_hybridPostingFile +
                ".tmp"),
        NormalizeHybridArtifactPath(
            root,
            p_options.m_hybridPostingFile +
                ".stats"),
        NormalizeHybridArtifactPath(
            root,
            p_options.m_hybridPostingFile +
                ".stats.tmp"),
        graph,
        NormalizeHybridArtifactPath(
            headRoot,
            p_options.m_hybridHeadGraphFile +
                ".tmp")};

    std::vector<std::filesystem::path> protectedPaths;
    const auto protect =
        [&](const std::filesystem::path& base,
            const std::string& path) {
            if (!path.empty() &&
                path != "Undefined!") {
                protectedPaths.push_back(
                    NormalizeHybridArtifactPath(
                        base, path));
                protectedPaths.push_back(
                    NormalizeHybridArtifactPath(
                        base, path + ".tmp"));
            }
        };
    const std::array<std::string, 19>
        configuredRootPaths = {
            p_options.m_ssdIndex,
            p_options.m_deleteIDFile,
            p_options.m_headIDFile,
            p_options.m_headVectorFile,
            p_options.m_quantizerFilePath,
            p_options.m_fullDeletedIDFile,
            p_options.m_ssdInfoFile,
            p_options.m_ssdMappingFile,
            p_options.m_checksumFile,
            p_options.m_postingPureCountsFile,
            p_options.m_headRoleFile,
            p_options.m_KVFile,
            p_options.m_primaryHeadCSRFile,
            p_options.m_postingQuantFile,
            p_options.m_pipePQPivotsFile,
            p_options.m_fullVectorFile,
            p_options.m_searchResult,
            p_options.m_logFile,
            p_options.m_updateVectorFile};
    for (const auto& path : configuredRootPaths) {
        protect(root, path);
    }
    const std::array<const char*, 8>
        fixedRootPaths = {
            "ordered_page_starts.bin",
            "indexloader.ini",
            "tag_routing_stats.bin",
            "numeric_meta.bin",
            "tag_level_offsets.bin",
            "signatures_bitmask.bin",
            "sparse_tags.bin",
            "tagpure_meta.bin"};
    for (const char* path : fixedRootPaths) {
        protect(root, path);
    }
    const std::array<const char*, 11>
        fixedHeadPaths = {
            "tree.bin", "graph.bin", "vectors.bin",
            "deletes.bin", "indexloader.ini",
            "head_metaonly.bin", "head_node_meta.bin",
            "tag_node_index.bin", "head_cross_edges.bin",
            "primary_head_csr.bin",
            "head_bundle_manifest.bin"};
    for (const char* path : fixedHeadPaths) {
        protect(headRoot, path);
    }

    for (size_t left = 0;
         left < generated.size(); ++left) {
        for (size_t right = left + 1;
             right < generated.size(); ++right) {
            if (HybridArtifactPathsAlias(
                    generated[left],
                    generated[right])) {
                p_error =
                    "hybrid generated paths alias each other";
                return false;
            }
        }
    }
    const auto primary =
        NormalizeHybridArtifactPath(
            root, p_options.m_ssdIndex);
    for (const auto& path : generated) {
        if (IsPrimaryStaticShardPath(
                path, primary)) {
            p_error =
                "hybrid artifact aliases a primary posting shard";
            return false;
        }
        for (const auto& protectedPath :
             protectedPaths) {
            if (HybridArtifactPathsAlias(
                    path, protectedPath)) {
                p_error =
                    "hybrid artifact aliases an index artifact";
                return false;
            }
        }
    }
    return true;
}

} // namespace SPANN
} // namespace SPTAG

#endif
