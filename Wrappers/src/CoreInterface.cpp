// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CoreInterface.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/TenantPrefixedKeyValueIO.h"
#include "inc/Helper/AtomicFile.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/Common/QueryResultSet.h"
#ifdef ROCKSDB
#include "inc/Core/SPANN/ExtraRocksDBController.h"
#endif

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <thread>
#include <atomic>
#include "inc/Core/SPANN/Options.h"
#include "inc/Core/SPANN/ExtraFileController.h"
#include "inc/Core/SPANN/Index.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cctype>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

struct TagRoutingStatRecord {
    uint32_t column;
    uint32_t tag;
    int32_t vectorCount;
    int32_t postingCount;
};

static_assert(sizeof(TagRoutingStatRecord) == 2 * sizeof(uint32_t) + 2 * sizeof(int32_t),
              "Unexpected TagRoutingStatRecord layout");

struct LegacyTagRoutingStatRecord {
    uint32_t tag;
    int32_t vectorCount;
    int32_t postingCount;
};

static_assert(sizeof(LegacyTagRoutingStatRecord) == sizeof(uint32_t) + 2 * sizeof(int32_t),
              "Unexpected legacy TagRoutingStatRecord layout");

constexpr std::uint32_t kTagRoutingStatsMagic = 0x53525454U; // TTRS
constexpr std::uint32_t kTagRoutingStatsVersion = 4;
constexpr std::uint32_t kLegacyRoutingColumn =
    (std::numeric_limits<std::uint32_t>::max)();

std::uint64_t MakeTagRoutingKey(
    std::uint32_t column,
    std::uint32_t tag)
{
    return
        (static_cast<std::uint64_t>(column) << 32) |
        static_cast<std::uint64_t>(tag);
}

struct TagRoutingStatsFileHeader {
    std::uint32_t magic = kTagRoutingStatsMagic;
    std::uint32_t version = kTagRoutingStatsVersion;
    std::int32_t vectorCount = 0;
    std::int32_t recordCount = 0;
    std::uint64_t generationFingerprint = 0;
};

bool SaveTagRoutingStatsFile(
    const std::string& path,
    int vectorCount,
    std::uint64_t generationFingerprint,
    const TenantIndexManager::TagRoutingStatsMap& stats)
{
    if (vectorCount <= 0 ||
        stats.size() > static_cast<size_t>((std::numeric_limits<std::int32_t>::max)())) {
        return false;
    }
    std::vector<TagRoutingStatRecord> records;
    records.reserve(stats.size());
    for (const auto& entry : stats) {
        if (entry.second.vectorCount < 0 || entry.second.vectorCount > vectorCount ||
            entry.second.postingCount < 0) {
            return false;
        }
        records.push_back({
            static_cast<std::uint32_t>(
                entry.first >> 32),
            static_cast<std::uint32_t>(
                entry.first),
            entry.second.vectorCount,
            entry.second.postingCount});
    }
    std::sort(records.begin(), records.end(),
        [](const TagRoutingStatRecord& left, const TagRoutingStatRecord& right) {
            return left.column == right.column
                ? left.tag < right.tag
                : left.column < right.column;
        });

    const std::string temporary = path + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr) return false;
    TagRoutingStatsFileHeader header;
    header.vectorCount = vectorCount;
    header.recordCount = static_cast<std::int32_t>(records.size());
    header.generationFingerprint =
        generationFingerprint;
    bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
        (records.empty() ||
         std::fwrite(records.data(), sizeof(TagRoutingStatRecord),
                     records.size(), file) == records.size());
    if (std::fclose(file) != 0) ok = false;
    if (!ok) {
        std::remove(temporary.c_str());
        return false;
    }
    if (!SPTAG::Helper::AtomicReplaceFile(
            temporary, path)) {
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

bool LoadTagRoutingStatsFile(
    const std::string& path,
    int expectedVectorCount,
    std::uint64_t& generationFingerprint,
    TenantIndexManager::TagRoutingStatsMap& stats)
{
    stats.clear();
    generationFingerprint = 0;
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    TagRoutingStatsFileHeader header;
    bool ok = std::fread(&header, sizeof(header), 1, file) == 1 &&
        header.magic == kTagRoutingStatsMagic &&
        header.version == kTagRoutingStatsVersion &&
        header.vectorCount == expectedVectorCount &&
        header.recordCount >= 0;
    if (ok) {
        const std::uint64_t recordCount =
            static_cast<std::uint64_t>(
                header.recordCount);
        ok =
            recordCount <=
                ((std::numeric_limits<
                      std::uint64_t>::max)() -
                 sizeof(header)) /
                    sizeof(TagRoutingStatRecord);
        const std::uint64_t expectedBytes = ok
            ? static_cast<std::uint64_t>(
                  sizeof(header)) +
                  recordCount *
                      static_cast<std::uint64_t>(
                          sizeof(TagRoutingStatRecord))
            : 0;
        ok = ok &&
            std::fseek(file, 0, SEEK_END) == 0;
        const long fileBytes = ok
            ? std::ftell(file)
            : -1;
        ok = ok && fileBytes >= 0 &&
            static_cast<std::uint64_t>(
                fileBytes) == expectedBytes &&
            std::fseek(
                file,
                static_cast<long>(sizeof(header)),
                SEEK_SET) == 0;
    }
    if (!ok) {
        std::fclose(file);
        return false;
    }
    stats.reserve((std::min<size_t>)(
        static_cast<size_t>(header.recordCount),
        static_cast<size_t>(1 << 20)));
    for (std::int32_t index = 0;
         index < header.recordCount; ++index) {
        TagRoutingStatRecord record;
        if (std::fread(
                &record, sizeof(record), 1,
                file) != 1) {
            stats.clear();
            std::fclose(file);
            return false;
        }
        if (record.vectorCount < 0 || record.vectorCount > expectedVectorCount ||
            record.postingCount < 0 ||
            !stats.emplace(
                MakeTagRoutingKey(
                    record.column, record.tag),
                TenantIndexManager::TagRoutingStats{
                 record.vectorCount, record.postingCount}).second) {
            stats.clear();
            std::fclose(file);
            return false;
        }
    }
    if (std::fclose(file) != 0) {
        stats.clear();
        return false;
    }
    generationFingerprint =
        header.generationFingerprint;
    return true;
}

bool LoadHybridRoutingStatsHeader(
    const std::string& path,
    SPTAG::SPANN::HybridRoutingStatsHeader& header)
{
    header = SPTAG::SPANN::HybridRoutingStatsHeader();
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    bool ok =
        std::fread(&header, sizeof(header), 1, file) == 1 &&
        header.m_magic ==
            SPTAG::SPANN::kHybridRoutingStatsMagic &&
        header.m_version ==
            SPTAG::SPANN::kHybridRoutingStatsVersion &&
        header.m_categoricalColumnCount >= 0 &&
        header.m_categoricalColumnCount <= 16 &&
        header.m_maskCount ==
            (1 << header.m_categoricalColumnCount) &&
        header.m_numTagColumns > 0 &&
        header.m_headCount > 0 &&
        header.m_generationFingerprint != 0;
    std::uint64_t expectedBytes = sizeof(header);
    const auto addBytes =
        [&expectedBytes](std::uint64_t count,
                         std::uint64_t width) {
            if (count >
                ((std::numeric_limits<
                      std::uint64_t>::max)() -
                 expectedBytes) /
                    width) {
                return false;
            }
            expectedBytes += count * width;
            return true;
        };
    if (ok) {
        ok =
            addBytes(
                static_cast<std::uint64_t>(
                    header.m_categoricalColumnCount),
                sizeof(int)) &&
            addBytes(
                static_cast<std::uint64_t>(
                    header.m_headCount) *
                    static_cast<std::uint64_t>(
                        header.m_numTagColumns),
                sizeof(std::uint32_t)) &&
            addBytes(
                2ULL *
                    (4ULL +
                     static_cast<std::uint64_t>(
                         header.m_maskCount)),
                sizeof(double)) &&
            std::fseek(file, 0, SEEK_END) == 0;
    }
    const long fileBytes = ok
        ? std::ftell(file)
        : -1;
    ok = ok && fileBytes >= 0 &&
        static_cast<std::uint64_t>(fileBytes) ==
            expectedBytes;
    std::fclose(file);
    return ok;
}

std::string ReadBuildSSDIndexValue(
    const std::string& path,
    const std::string& requestedKey)
{
    std::ifstream input(path);
    if (!input) return std::string();
    std::string section;
    std::string line;
    std::string normalizedRequested = requestedKey;
    const auto trim = [](std::string& text) {
        const size_t begin = text.find_first_not_of(" \t\r\n");
        const size_t end = text.find_last_not_of(" \t\r\n");
        text = begin == std::string::npos
            ? std::string()
            : text.substr(begin, end - begin + 1);
    };
    const auto lower = [](std::string& text) {
        std::transform(text.begin(), text.end(), text.begin(),
            [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
    };
    lower(normalizedRequested);
    while (std::getline(input, line)) {
        trim(line);
        if (line.empty() || line[0] == ';') continue;
        if (line.front() == '[') {
            const size_t close = line.find(']');
            section = close == std::string::npos
                ? std::string()
                : line.substr(1, close - 1);
            lower(section);
            continue;
        }
        if (section != "buildssdindex") continue;
        const size_t equal = line.find('=');
        if (equal == std::string::npos) continue;
        std::string key = line.substr(0, equal);
        std::string value = line.substr(equal + 1);
        trim(key);
        trim(value);
        lower(key);
        if (key == normalizedRequested) return value;
    }
    return std::string();
}

bool IniEnablesHybridDistance(const std::string& path)
{
    std::string value = ReadBuildSSDIndexValue(
        path, "EnableHybridDistance");
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char item) {
            return static_cast<char>(
                std::tolower(item));
        });
    return value == "1" || value == "true" ||
        value == "yes" || value == "on";
}

int ReadPositiveEnvironmentInt(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0
        || parsed > std::numeric_limits<int>::max()) {
        fprintf(stderr, "[WARN] Ignoring invalid %s=%s\n", name, value);
        return fallback;
    }
    return static_cast<int>(parsed);
}

bool EnsureDir(const std::string& path)
{
    if (path.empty()) return false;

    std::string cmd = "mkdir -p \"" + path + "\"";
    return std::system(cmd.c_str()) == 0;
}

bool RemovePathRecursive(const std::string& path)
{
    if (path.empty()) return false;

    std::string cmd = "rm -rf \"" + path + "\"";
    return std::system(cmd.c_str()) == 0;
}

bool CopyDirRecursive(const std::string& src, const std::string& dst)
{
    if (src.empty() || dst.empty()) return false;

    if (!RemovePathRecursive(dst)) return false;

    std::string cmd = "cp -a \"" + src + "\" \"" + dst + "\"";
    return std::system(cmd.c_str()) == 0;
}

uint64_t GetPathSizeBytes(const std::string& path)
{
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) {
        return 0;
    }

    if (S_ISREG(st.st_mode)) {
        return static_cast<uint64_t>(st.st_size);
    }

    if (!S_ISDIR(st.st_mode)) {
        return 0;
    }

    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        return 0;
    }

    uint64_t totalBytes = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        totalBytes += GetPathSizeBytes(path + "/" + entry->d_name);
    }

    closedir(dir);
    return totalBytes;
}

bool BuildStaticPostingHierMasks(const std::string& p_snapshotPath,
                                 int p_expectedHeadCount,
                                 int p_expectedTagCount,
                                 int p_aclTagCount,
                                 const SPTAG::Cache::HierWidthTable& p_hierWidths,
                                 std::vector<SPTAG::Cache::HierarchicalPostingMask>& p_masks,
                                 std::unordered_map<std::uint64_t, int>*
                                     p_tagPostingCounts = nullptr)
{
    constexpr std::uint32_t kStaticMetadataMagic = 0x314D5453U; // "STM1"
    constexpr int kHeaderV1IntCount = 9;
    constexpr int kHeaderV2IntCount = 11;
    constexpr size_t kPageSize = 4096;

    std::ifstream input(p_snapshotPath, std::ios::binary);
    if (!input) {
        fprintf(stderr, "[ERROR] Cannot open static snapshot %s for posting masks\n", p_snapshotPath.c_str());
        return false;
    }
    std::int32_t header[kHeaderV2IntCount] = {};
    if (!input.read(
            reinterpret_cast<char*>(header),
            sizeof(std::int32_t) *
                kHeaderV1IntCount)) {
        fprintf(stderr, "[ERROR] Cannot read STM1 header from %s\n", p_snapshotPath.c_str());
        return false;
    }
    const std::uint32_t magic = static_cast<std::uint32_t>(header[0]);
    const int version = header[1];
    const int listCount = header[2];
    const int recordBytes = header[5];
    const int tagCount = header[6];
    if (version == 2 &&
        !input.read(
            reinterpret_cast<char*>(
                header + kHeaderV1IntCount),
            sizeof(std::int32_t) *
                (kHeaderV2IntCount -
                 kHeaderV1IntCount))) {
        fprintf(
            stderr,
            "[ERROR] Cannot read STM1 generation header from %s\n",
            p_snapshotPath.c_str());
        return false;
    }
    const int listPageOffset =
        version == 2 ? header[10] : header[8];
    const int metadataBytes = static_cast<int>(sizeof(std::int32_t) +
                                               static_cast<size_t>(tagCount) * sizeof(std::uint32_t));
    if (magic != kStaticMetadataMagic ||
        (version != 1 && version != 2) ||
        listCount != p_expectedHeadCount ||
        tagCount != p_expectedTagCount || tagCount <= 0 || recordBytes < metadataBytes ||
        listPageOffset < 0) {
        fprintf(stderr,
                "[ERROR] Invalid STM1 header for posting masks: magic=%u version=%d lists=%d expected=%d "
                "record=%d tags=%d expectedTags=%d pages=%d\n",
                magic, version, listCount, p_expectedHeadCount, recordBytes, tagCount,
                p_expectedTagCount, listPageOffset);
        return false;
    }
    const int aclTagCount = p_aclTagCount > 0 ? p_aclTagCount : tagCount;
    if (aclTagCount > tagCount) {
        fprintf(stderr, "[ERROR] STM1 ACL tag count %d exceeds record tag count %d in %s\n",
                aclTagCount, tagCount, p_snapshotPath.c_str());
        return false;
    }

    struct ListInfo {
        std::int32_t pageNum = 0;
        std::uint16_t pageOffset = 0;
        std::int32_t elementCount = 0;
        std::uint16_t pageCount = 0;
        std::int32_t pureCount = 0;
    };
    std::vector<ListInfo> lists(static_cast<size_t>(listCount));
    struct stat snapshotStat {};
    if (stat(p_snapshotPath.c_str(), &snapshotStat) != 0 || snapshotStat.st_size < 0) {
        fprintf(stderr, "[ERROR] Cannot stat STM1 snapshot %s\n", p_snapshotPath.c_str());
        return false;
    }
    const std::uint64_t snapshotBytes = static_cast<std::uint64_t>(snapshotStat.st_size);
    for (auto& list : lists) {
        if (!input.read(reinterpret_cast<char*>(&list.pageNum), sizeof(list.pageNum)) ||
            !input.read(reinterpret_cast<char*>(&list.pageOffset), sizeof(list.pageOffset)) ||
            !input.read(reinterpret_cast<char*>(&list.elementCount), sizeof(list.elementCount)) ||
            !input.read(reinterpret_cast<char*>(&list.pageCount), sizeof(list.pageCount)) ||
            !input.read(reinterpret_cast<char*>(&list.pureCount), sizeof(list.pureCount)) ||
            list.pageNum < 0 || list.pureCount < 0 || list.pureCount > list.elementCount) {
            fprintf(stderr, "[ERROR] Invalid STM1 posting-list metadata in %s\n", p_snapshotPath.c_str());
            return false;
        }
        const std::uint64_t listBytes =
            static_cast<std::uint64_t>(list.elementCount) * static_cast<std::uint64_t>(recordBytes);
        const std::uint64_t declaredPages = (listBytes + kPageSize - 1) / kPageSize;
        const std::uint64_t recordOffset =
            (static_cast<std::uint64_t>(listPageOffset) + static_cast<std::uint64_t>(list.pageNum)) *
                kPageSize +
            list.pageOffset;
        if (list.pageOffset >= kPageSize || declaredPages > std::numeric_limits<std::uint16_t>::max() ||
            list.pageCount != declaredPages ||
            static_cast<std::uint64_t>(list.pageOffset) + listBytes > declaredPages * kPageSize ||
            recordOffset > snapshotBytes || listBytes > snapshotBytes - recordOffset) {
            fprintf(stderr, "[ERROR] Invalid STM1 posting layout in %s\n", p_snapshotPath.c_str());
            return false;
        }
    }

    p_masks.assign(static_cast<size_t>(listCount), SPTAG::Cache::HierarchicalPostingMask());
    if (p_tagPostingCounts != nullptr) {
        p_tagPostingCounts->clear();
    }
    std::vector<std::uint32_t> postingOrder;
    postingOrder.reserve(static_cast<size_t>(listCount));
    for (std::uint32_t postingId = 0; postingId < static_cast<std::uint32_t>(listCount);
         ++postingId) {
        if (lists[postingId].pureCount > 0) postingOrder.push_back(postingId);
    }
    const auto recordOffsetFor = [&lists, listPageOffset](std::uint32_t postingId) {
        const auto& list = lists[postingId];
        return (static_cast<std::uint64_t>(listPageOffset) +
                static_cast<std::uint64_t>(list.pageNum)) * kPageSize + list.pageOffset;
    };
    std::sort(postingOrder.begin(), postingOrder.end(),
              [&recordOffsetFor](std::uint32_t lhs, std::uint32_t rhs) {
                  const std::uint64_t left = recordOffsetFor(lhs);
                  const std::uint64_t right = recordOffsetFor(rhs);
                  return left == right ? lhs < rhs : left < right;
              });

    constexpr std::uint64_t kBatchBytes = 32ULL * 1024ULL * 1024ULL;
    std::vector<char> batch;
    size_t begin = 0;
    size_t nextReport = std::max<size_t>(1, postingOrder.size() / 100);
    fprintf(stderr, "[INFO] Scanning STM1 posting masks in physical order.\n");
    while (begin < postingOrder.size()) {
        const std::uint32_t firstPostingId = postingOrder[begin];
        const std::uint64_t batchStart = recordOffsetFor(firstPostingId);
        std::uint64_t batchEnd = batchStart;
        size_t end = begin;
        while (end < postingOrder.size()) {
            const std::uint32_t postingId = postingOrder[end];
            const auto& list = lists[postingId];
            const std::uint64_t recordOffset = recordOffsetFor(postingId);
            const std::uint64_t listBytes =
                static_cast<std::uint64_t>(list.pureCount) * static_cast<std::uint64_t>(recordBytes);
            if (recordOffset < batchStart || listBytes > std::numeric_limits<uint64_t>::max() - recordOffset) {
                fprintf(stderr, "[ERROR] Invalid STM1 physical-order range in %s\n",
                        p_snapshotPath.c_str());
                return false;
            }
            const std::uint64_t recordEnd = recordOffset + listBytes;
            if (end > begin && recordEnd - batchStart > kBatchBytes) break;
            batchEnd = std::max(batchEnd, recordEnd);
            ++end;
        }
        if (batchEnd < batchStart || batchEnd - batchStart > std::numeric_limits<size_t>::max()) {
            fprintf(stderr, "[ERROR] STM1 mask batch is too large in %s\n", p_snapshotPath.c_str());
            return false;
        }
        batch.resize(static_cast<size_t>(batchEnd - batchStart));
        input.seekg(static_cast<std::streamoff>(batchStart), std::ios::beg);
        if (!input.read(batch.data(), static_cast<std::streamsize>(batch.size()))) {
            fprintf(stderr, "[ERROR] Cannot read STM1 posting batch from %s\n", p_snapshotPath.c_str());
            return false;
        }
        for (size_t index = begin; index < end; ++index) {
            const std::uint32_t postingId = postingOrder[index];
            const auto& list = lists[postingId];
            const char* records = batch.data() + (recordOffsetFor(postingId) - batchStart);
            auto& mask = p_masks[postingId];
            std::vector<std::uint64_t> postingTags;
            if (p_tagPostingCounts != nullptr) {
                postingTags.reserve(
                    static_cast<size_t>(list.pureCount) *
                    static_cast<size_t>(aclTagCount));
            }
            for (int i = 0; i < list.pureCount; ++i) {
                const char* record = records + static_cast<size_t>(i) * static_cast<size_t>(recordBytes);
                for (int tagColumn = 0; tagColumn < aclTagCount; ++tagColumn) {
                    std::uint32_t tag = 0;
                    std::memcpy(&tag, record + sizeof(std::int32_t) +
                                      static_cast<size_t>(tagColumn) * sizeof(tag),
                                sizeof(tag));
                    mask.Insert(
                        tagColumn, tag,
                        p_hierWidths);
                    if (p_tagPostingCounts != nullptr) {
                        postingTags.push_back(
                            MakeTagRoutingKey(
                                static_cast<std::uint32_t>(
                                    tagColumn),
                                tag));
                        postingTags.push_back(
                            MakeTagRoutingKey(
                                kLegacyRoutingColumn,
                                tag));
                    }
                }
            }
            if (p_tagPostingCounts != nullptr) {
                std::sort(postingTags.begin(), postingTags.end());
                postingTags.erase(
                    std::unique(
                        postingTags.begin(), postingTags.end()),
                    postingTags.end());
                for (std::uint64_t key : postingTags) {
                    ++(*p_tagPostingCounts)[key];
                }
            }
        }
        if (end >= nextReport || end == postingOrder.size()) {
            const double percent = 100.0 * static_cast<double>(end) /
                static_cast<double>(postingOrder.size());
            fprintf(stderr, "\r[INFO] STM1 posting mask scan %.1f%%", percent);
            nextReport += std::max<size_t>(1, postingOrder.size() / 100);
        }
        begin = end;
    }
    fprintf(stderr, "\n");
    return true;
}

uint64_t GetCurrentProcessRSSBytes()
{
#ifdef __linux__
    long rssPages = 0;
    FILE* statm = fopen("/proc/self/statm", "r");
    if (statm == nullptr) {
        return 0;
    }

    int scanned = fscanf(statm, "%*s %ld", &rssPages);
    fclose(statm);
    if (scanned != 1 || rssPages <= 0) {
        return 0;
    }

    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) {
        return 0;
    }

    return static_cast<uint64_t>(rssPages) * static_cast<uint64_t>(pageSize);
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        return static_cast<uint64_t>(pmc.WorkingSetSize);
    }
    return 0;
#else
    return 0;
#endif
}

} // namespace

namespace {

int TagLevel(uint32_t tag)
{
    return static_cast<int>(tag / 1000U);
}

float EstimateQueryVectorSelectivity(
    int tenantSize,
    const TenantIndexManager::TagRoutingStatsMap* tagStats,
    const uint32_t* queryTags,
    int numQueryTags,
    const SPTAG::Cache::DNFPredicate* dnf,
    int hierarchyColumnCount,
    int numericBaseColumn,
    const SPTAG::Cache::NumQuantParam*
        numericParams,
    size_t numericParamCount)
{
    if (tenantSize <= 0 || tagStats == nullptr) {
        return 1.0f;
    }

    if (dnf != nullptr && !dnf->Empty()) {
        struct EstimatedClause {
            std::unordered_map<uint32_t, uint32_t>
                categoricalByColumn;
            std::vector<std::tuple<
                uint8_t, uint8_t, uint32_t, uint32_t>>
                canonical;
            double selectivity = 1.0;
        };
        std::vector<EstimatedClause> estimates;
        std::set<std::vector<std::tuple<
            uint8_t, uint8_t, uint32_t, uint32_t>>>
            seenClauses;
        bool sawNonemptyClause = false;
        for (const auto& clause : dnf->clauses) {
            if (clause.lits.empty()) continue;
            sawNonemptyClause = true;
            std::vector<std::tuple<
                uint8_t, uint8_t, uint32_t, uint32_t>>
                canonical;
            canonical.reserve(clause.lits.size());
            for (const auto& literal : clause.lits) {
                canonical.emplace_back(
                    literal.kind, literal.op,
                    literal.col, literal.val);
            }
            std::sort(canonical.begin(),
                      canonical.end());
            canonical.erase(
                std::unique(canonical.begin(),
                            canonical.end()),
                canonical.end());
            if (!seenClauses.insert(canonical).second) {
                continue;
            }
            double hierarchySelectivity = 1.0;
            double independentSelectivity = 1.0;
            double numericSelectivity = 1.0;
            bool estimatedLiteral = false;
            std::unordered_map<uint32_t, uint32_t> categoricalByColumn;
            bool impossibleClause = false;
            for (const auto& literal : clause.lits) {
                if (literal.kind != 0 ||
                    literal.op != SPTAG::Cache::DNF_EQ) {
                    continue;
                }
                const auto inserted =
                    categoricalByColumn.emplace(literal.col, literal.val);
                if (!inserted.second &&
                    inserted.first->second != literal.val) {
                    impossibleClause = true;
                    break;
                }
                const auto stat = tagStats->find(
                    MakeTagRoutingKey(
                        literal.col, literal.val));
                if (stat == tagStats->end()) {
                    impossibleClause = true;
                    break;
                }
                const double literalSelectivity =
                    static_cast<double>(
                        stat->second.vectorCount) /
                    static_cast<double>(tenantSize);
                if (hierarchyColumnCount < 0 ||
                    static_cast<int>(literal.col) <
                        hierarchyColumnCount) {
                    hierarchySelectivity = std::min(
                        hierarchySelectivity,
                        literalSelectivity);
                } else {
                    independentSelectivity *=
                        literalSelectivity;
                }
                estimatedLiteral = true;
            }
            std::unordered_map<
                uint32_t,
                std::pair<std::uint64_t,
                          std::uint64_t>>
                numericRanges;
            for (const auto& literal : clause.lits) {
                if (literal.kind == 0 ||
                    static_cast<int>(literal.col) <
                        numericBaseColumn ||
                    numericParams == nullptr) {
                    continue;
                }
                const size_t numericIndex =
                    static_cast<size_t>(
                        static_cast<int>(literal.col) -
                        numericBaseColumn);
                if (numericIndex >=
                    numericParamCount) {
                    continue;
                }
                const auto& domain =
                    numericParams[numericIndex];
                if (domain.hi < domain.lo) {
                    impossibleClause = true;
                    break;
                }
                auto inserted = numericRanges.emplace(
                    literal.col,
                    std::make_pair(
                        static_cast<std::uint64_t>(
                            domain.lo),
                        static_cast<std::uint64_t>(
                            domain.hi)));
                auto& range = inserted.first->second;
                const std::uint64_t value =
                    literal.val;
                switch (literal.op) {
                case SPTAG::Cache::DNF_EQ:
                    range.first = (std::max)(
                        range.first, value);
                    range.second = (std::min)(
                        range.second, value);
                    break;
                case SPTAG::Cache::DNF_LT:
                    if (value == 0) {
                        impossibleClause = true;
                    } else {
                        range.second = (std::min)(
                            range.second, value - 1);
                    }
                    break;
                case SPTAG::Cache::DNF_LE:
                    range.second = (std::min)(
                        range.second, value);
                    break;
                case SPTAG::Cache::DNF_GT:
                    if (value ==
                        (std::numeric_limits<
                            uint32_t>::max)()) {
                        impossibleClause = true;
                    } else {
                        range.first = (std::max)(
                            range.first, value + 1);
                    }
                    break;
                case SPTAG::Cache::DNF_GE:
                    range.first = (std::max)(
                        range.first, value);
                    break;
                default:
                    impossibleClause = true;
                    break;
                }
                if (impossibleClause ||
                    range.first > range.second) {
                    impossibleClause = true;
                    break;
                }
            }
            if (impossibleClause) continue;
            for (const auto& numeric :
                 numericRanges) {
                const size_t numericIndex =
                    static_cast<size_t>(
                        static_cast<int>(
                            numeric.first) -
                        numericBaseColumn);
                const auto& domain =
                    numericParams[numericIndex];
                const long double selected =
                    static_cast<long double>(
                        numeric.second.second -
                        numeric.second.first) +
                    1.0L;
                const long double total =
                    static_cast<long double>(
                        static_cast<std::uint64_t>(
                            domain.hi) -
                        static_cast<std::uint64_t>(
                            domain.lo)) +
                    1.0L;
                numericSelectivity *=
                    static_cast<double>(
                        selected / total);
                estimatedLiteral = true;
            }
            if (impossibleClause) continue;
            const double clauseSelectivity =
                estimatedLiteral
                ? hierarchySelectivity *
                      independentSelectivity *
                      numericSelectivity
                : 1.0;
            estimates.push_back({
                std::move(categoricalByColumn),
                std::move(canonical),
                clauseSelectivity});
        }
        if (estimates.empty()) {
            return sawNonemptyClause ? 1e-6f : 1.0f;
        }
        for (size_t candidate = 0;
             candidate < estimates.size();) {
            bool redundant = false;
            for (size_t broader = 0;
                 broader < estimates.size(); ++broader) {
                if (candidate == broader ||
                    estimates[broader].canonical.size() >=
                        estimates[candidate]
                            .canonical.size()) {
                    continue;
                }
                if (std::includes(
                        estimates[candidate]
                            .canonical.begin(),
                        estimates[candidate]
                            .canonical.end(),
                        estimates[broader]
                            .canonical.begin(),
                        estimates[broader]
                            .canonical.end())) {
                    redundant = true;
                    break;
                }
            }
            if (redundant) {
                estimates.erase(
                    estimates.begin() + candidate);
            } else {
                ++candidate;
            }
        }
        bool pairwiseDisjoint = estimates.size() > 1;
        for (size_t left = 0;
             left < estimates.size() && pairwiseDisjoint;
             ++left) {
            for (size_t right = left + 1;
                 right < estimates.size(); ++right) {
                bool conflicts = false;
                for (const auto& column :
                     estimates[left].categoricalByColumn) {
                    const auto other = estimates[right]
                        .categoricalByColumn.find(
                            column.first);
                    if (other != estimates[right]
                                     .categoricalByColumn.end() &&
                        other->second != column.second) {
                        conflicts = true;
                        break;
                    }
                }
                if (!conflicts) {
                    pairwiseDisjoint = false;
                    break;
                }
            }
        }
        double unionSelectivity = 0.0;
        if (pairwiseDisjoint) {
            for (const auto& estimate : estimates) {
                unionSelectivity +=
                    estimate.selectivity;
            }
        } else {
            double productNotSelected = 1.0;
            for (const auto& estimate : estimates) {
                productNotSelected *=
                    std::max(
                        0.0,
                        1.0 -
                            estimate.selectivity);
            }
            unionSelectivity =
                1.0 - productNotSelected;
        }
        return static_cast<float>(std::clamp(
            unionSelectivity, 1e-6, 1.0));
    }

    if (queryTags == nullptr || numQueryTags <= 0) return 1.0f;

    std::unordered_set<uint32_t> seenTags;
    std::unordered_map<int, double> levelSelectivities;
    for (int index = 0; index < numQueryTags; ++index) {
        uint32_t tag = queryTags[index];
        if (!seenTags.insert(tag).second) {
            continue;
        }

        std::int64_t vectorCount = 0;
        const auto exactLegacy = tagStats->find(
            MakeTagRoutingKey(
                kLegacyRoutingColumn, tag));
        if (exactLegacy != tagStats->end()) {
            vectorCount =
                exactLegacy->second.vectorCount;
        } else {
            for (const auto& entry : *tagStats) {
                if (static_cast<std::uint32_t>(
                        entry.first) == tag) {
                    vectorCount = (std::min)(
                        static_cast<std::int64_t>(
                            tenantSize),
                        vectorCount +
                            static_cast<std::int64_t>(
                                entry.second.vectorCount));
                }
            }
        }
        if (vectorCount <= 0) {
            continue;
        }

        double tagSelectivity =
            static_cast<double>(vectorCount) /
            static_cast<double>(tenantSize);
        int level = TagLevel(tag);
        double& levelSel = levelSelectivities[level];
        levelSel = std::min(1.0, levelSel + tagSelectivity);
    }

    if (levelSelectivities.empty()) {
        return 1.0f;
    }

    double productNotSelected = 1.0;
    for (const auto& [level, selectivity] : levelSelectivities) {
        (void)level;
        productNotSelected *= std::max(0.0, 1.0 - selectivity);
    }

    double unionSelectivity = 1.0 - productNotSelected;
    unionSelectivity = std::clamp(unionSelectivity, 1e-6, 1.0);
    return static_cast<float>(unionSelectivity);
}

struct PivotEstimatorLevelData {
    std::vector<uint32_t> uniqueTags;
    std::vector<int> counts;
    std::unordered_map<uint32_t, uint32_t> parentByTag;
};

struct PivotEstimatorCandidate {
    int pivotLevel = -1;
    int nodeCount = 0;
    double latencyCost = 0.0;
    double recallPenalty = 0.0;
    double totalCost = std::numeric_limits<double>::infinity();
    std::vector<std::vector<uint32_t>> nodePivotTags;
    std::vector<int> nodeSizes;
};

std::string JsonEscape(const std::string& input)
{
    std::string out;
    out.reserve(input.size() + 8);
    for (char ch : input)
    {
        switch (ch)
        {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

bool ParseLevelWeights(const std::string& csv, int levelCount, std::vector<double>& outWeights)
{
    outWeights.clear();
    if (levelCount <= 0) return false;

    if (csv.empty()) {
        outWeights.assign(levelCount, 1.0 / static_cast<double>(levelCount));
        return true;
    }

    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty()) continue;
        try {
            outWeights.push_back(std::stod(token));
        }
        catch (...) {
            return false;
        }
    }

    if (static_cast<int>(outWeights.size()) != levelCount) return false;

    double sum = 0.0;
    for (double value : outWeights)
    {
        if (value < 0.0) return false;
        sum += value;
    }
    if (sum <= 0.0) return false;

    for (double& value : outWeights)
    {
        value /= sum;
    }
    return true;
}

bool TryGetAncestorTag(uint32_t tag,
                       int fromLevel,
                       int targetLevel,
                       const std::vector<PivotEstimatorLevelData>& levelData,
                       uint32_t& ancestorTag)
{
    ancestorTag = tag;
    if (fromLevel == targetLevel) return true;
    if (fromLevel < targetLevel || fromLevel < 0 || targetLevel < 0 ||
        fromLevel >= static_cast<int>(levelData.size()) ||
        targetLevel >= static_cast<int>(levelData.size())) {
        return false;
    }

    uint32_t currentTag = tag;
    for (int level = fromLevel; level > targetLevel; --level)
    {
        const auto parentIt = levelData[level].parentByTag.find(currentTag);
        if (parentIt == levelData[level].parentByTag.end()) {
            return false;
        }
        currentTag = parentIt->second;
    }

    ancestorTag = currentTag;
    return true;
}

} // namespace

constexpr int32_t kHeadNodeMetaVersion = 3;       // base: categorical only
constexpr int32_t kHeadNodeMetaVersionV4 = 4;     // V3 + quantized numeric block
constexpr int32_t kHeadNodeMetaVersionV5 = 5;     // V4 + per-column hier mask widths

struct HeadNodeMetaFileHeader {
    int32_t version;
    int32_t numSamples;
    int32_t numTagsPerSample;
    int32_t stride;
};

// V5 appends HIER_LEVELS int32 per-column mask widths after the base header.
// V3/V4 files carry no widths and load with the uniform default.

// Parse SPTAG_HIER_LEVEL_WIDTHS (comma-separated per-column bit widths, e.g.
// "256,64,64,128,64") into a build-local layout.
SPTAG::Cache::HierWidthTable HierWidthsFromEnv()
{
    SPTAG::Cache::HierWidthTable table;
    const char* e = std::getenv("SPTAG_HIER_LEVEL_WIDTHS");
    if (e == nullptr || e[0] == '\0') {
        return table;
    }
    int widths[SPTAG::Cache::HIER_LEVELS];
    for (int l = 0; l < SPTAG::Cache::HIER_LEVELS; ++l) widths[l] = SPTAG::Cache::HIER_LEVEL_BITS;
    int n = 0;
    const char* p = e;
    while (*p != '\0' && n < SPTAG::Cache::HIER_LEVELS) {
        widths[n++] = atoi(p);
        const char* comma = strchr(p, ',');
        if (comma == nullptr) break;
        p = comma + 1;
    }
    table.Set(widths, n);
    fprintf(stderr, "[INFO] Hier mask widths: bits=[%d,%d,%d,%d,%d] totalWords=%d (%zu B/head)\n",
            table.bits[0], table.bits[1], table.bits[2], table.bits[3],
            table.bits[4], table.totalWords,
            SPTAG::Cache::HierPostingMaskBytes(table));
    return table;
}

std::string HeadNodeMetaPath(const std::string& workDir)
{
    return workDir + "/HeadIndex/head_node_meta.bin";
}

std::shared_ptr<SPTAG::VectorIndex> GetMemoryIndexForInternal(const std::shared_ptr<SPTAG::VectorIndex>& internalIndex)
{
    auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIndex.get());
    if (spannInternalIdx == nullptr) return nullptr;
    return spannInternalIdx->GetMemoryIndex();
}

bool SaveHeadNodeMetaFile(const std::string& workDir, const std::shared_ptr<SPTAG::VectorIndex>& headIndex)
{
    if (headIndex == nullptr || !headIndex->HasHeadNodeMeta()) return false;

    std::string metaPath = HeadNodeMetaPath(workDir);
    FILE* f = fopen(metaPath.c_str(), "wb");
    if (!f) return false;

    HeadNodeMetaFileHeader header{};
    int32_t quantCols = headIndex->GetHeadNodeNumQuantCols();

    // Persist per-column hier mask widths only when they are non-uniform; a
    // uniform table reproduces the legacy V3/V4 layout exactly, so keep emitting
    // V3/V4 for those (older readers stay compatible).
    const auto& widths =
        headIndex->GetHeadNodeHierWidths();
    bool nonUniformWidths = false;
    for (int l = 0; l < SPTAG::Cache::HIER_LEVELS; ++l)
        if (widths.bits[l] != SPTAG::Cache::HIER_LEVEL_BITS) { nonUniformWidths = true; break; }

    header.version = nonUniformWidths ? kHeadNodeMetaVersionV5
                                      : (quantCols > 0 ? kHeadNodeMetaVersionV4 : kHeadNodeMetaVersion);
    header.numSamples = headIndex->GetHeadNodeMetaSampleCount();
    header.numTagsPerSample = quantCols;  // V4/V5: quantized numeric column count
    header.stride = static_cast<int32_t>(headIndex->GetHeadNodeMetaStride());

    const auto& blob = headIndex->GetHeadNodeMetaBlob();
    bool ok = fwrite(&header, sizeof(header), 1, f) == 1;
    if (ok && header.version == kHeadNodeMetaVersionV5) {
        int32_t wbits[SPTAG::Cache::HIER_LEVELS];
        for (int l = 0; l < SPTAG::Cache::HIER_LEVELS; ++l) wbits[l] = widths.bits[l];
        ok = fwrite(wbits, sizeof(int32_t), SPTAG::Cache::HIER_LEVELS, f) == (size_t)SPTAG::Cache::HIER_LEVELS;
    }
    ok = ok && fwrite(blob.data(), 1, blob.size(), f) == blob.size();
    fclose(f);
    return ok;
}

bool LoadHeadNodeMetaFile(const std::string& workDir, const std::shared_ptr<SPTAG::VectorIndex>& headIndex)
{
    if (headIndex == nullptr) return false;

    std::string metaPath = HeadNodeMetaPath(workDir);
    FILE* f = fopen(metaPath.c_str(), "rb");
    if (!f) return false;

    HeadNodeMetaFileHeader header{};
    bool ok = fread(&header, sizeof(header), 1, f) == 1;
    // The file's sample count must cover every head. For metadata-only ("slim")
    // head roots, GetNumSamples() at load time may still report the small physical
    // root count (the logical total is set later, during the lazy SSD/bundle load),
    // so accept a file that has at least as many entries as the current count;
    // InitializeHeadNodeMeta(header.numSamples) below resizes the blob accordingly.
    if (!ok || header.numSamples < 0 || header.stride <= 0 ||
        header.numSamples < headIndex->GetNumSamples()) {
        fclose(f);
        return false;
    }

    // Version check. Accept V3 (categorical only), V4 (with quantized numeric
    // block) and V5 (V4 + per-column hier mask widths).
    if (header.version != kHeadNodeMetaVersion &&
        header.version != kHeadNodeMetaVersionV4 &&
        header.version != kHeadNodeMetaVersionV5) {
        fprintf(stderr, "[ERROR] head_node_meta.bin version mismatch: file has version %d, expected version %d, %d or %d. "
                        "Please rebuild the index to use the new hierarchical mask format.\n",
                header.version, kHeadNodeMetaVersion, kHeadNodeMetaVersionV4, kHeadNodeMetaVersionV5);
        fclose(f);
        return false;
    }
    const std::uint64_t prefixBytes =
        static_cast<std::uint64_t>(
            sizeof(header)) +
        (header.version == kHeadNodeMetaVersionV5
             ? static_cast<std::uint64_t>(
                   sizeof(int32_t)) *
                   SPTAG::Cache::HIER_LEVELS
             : 0);
    const std::uint64_t sampleCount =
        static_cast<std::uint64_t>(
            header.numSamples);
    const std::uint64_t stride =
        static_cast<std::uint64_t>(
            header.stride);
    const bool sizeSafe =
        sampleCount <=
            ((std::numeric_limits<
                  std::uint64_t>::max)() -
             prefixBytes) /
                stride &&
        fseek(f, 0, SEEK_END) == 0;
    const long fileBytes = sizeSafe
        ? ftell(f)
        : -1;
    const std::uint64_t expectedBytes =
        sizeSafe
        ? prefixBytes + sampleCount * stride
        : 0;
    if (!sizeSafe || fileBytes < 0 ||
        static_cast<std::uint64_t>(fileBytes) !=
            expectedBytes ||
        fseek(
            f, static_cast<long>(sizeof(header)),
            SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    int quantCols = (header.version == kHeadNodeMetaVersionV4 || header.version == kHeadNodeMetaVersionV5)
                    ? header.numTagsPerSample : 0;
    if (quantCols < 0) {
        fclose(f);
        return false;
    }

    SPTAG::Cache::HierWidthTable hierWidths;
    if (header.version == kHeadNodeMetaVersionV5) {
        int32_t wbits[SPTAG::Cache::HIER_LEVELS];
        if (fread(wbits, sizeof(int32_t), SPTAG::Cache::HIER_LEVELS, f) != (size_t)SPTAG::Cache::HIER_LEVELS) {
            fclose(f);
            return false;
        }
        int widths[SPTAG::Cache::HIER_LEVELS];
        for (int l = 0; l < SPTAG::Cache::HIER_LEVELS; ++l) widths[l] = wbits[l];
        hierWidths.Set(
            widths,
            SPTAG::Cache::HIER_LEVELS);
    }

    size_t expectedStride = 0;
    const bool validLayout =
        SPTAG::VectorIndex::TryComputeHeadNodeMetaStride(
            quantCols, hierWidths,
            expectedStride) &&
        expectedStride <=
            static_cast<size_t>(
                (std::numeric_limits<int32_t>::max)()) &&
        expectedStride ==
            static_cast<size_t>(header.stride) &&
        sampleCount <=
            (std::numeric_limits<size_t>::max)() /
                expectedStride &&
        sampleCount * expectedStride <=
            headIndex->GetHeadNodeMetaBlob().max_size();
    if (!validLayout) {
        fprintf(stderr, "[ERROR] head_node_meta.bin stride mismatch: file has stride %d, expected %zu. "
                        "Binary layout has changed.\n",
                header.stride, expectedStride);
        fclose(f);
        return false;
    }
    headIndex->InitializeHeadNodeMeta(
        header.numSamples, quantCols,
        hierWidths);
    if (headIndex->GetHeadNodeMetaStride() !=
            expectedStride ||
        headIndex->GetHeadNodeMetaBlob().size() !=
            sampleCount * expectedStride) {
        headIndex->ClearHeadNodeMeta();
        fclose(f);
        return false;
    }

    auto& blob = headIndex->GetHeadNodeMetaBlob();
    ok = fread(blob.data(), 1, blob.size(), f) == blob.size();
    fclose(f);
    if (!ok) {
        headIndex->ClearHeadNodeMeta();
        return false;
    }
    return true;
}

bool LoadPostingSignaturesIntoHeadIndex(const std::string& workDir,
                                        const std::shared_ptr<SPTAG::VectorIndex>& internalIndex)
{
    if (internalIndex == nullptr) return false;

    auto headIndex = GetMemoryIndexForInternal(internalIndex);
    auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIndex.get());
    if (headIndex == nullptr || spannInternalIdx == nullptr) return false;

    SPTAG::Cache::TenantBitmaskPS sigs;
    std::string sigPath = workDir + "/signatures_bitmask.bin";
    if (!sigs.Load(sigPath)) return false;

    const SizeType numHeadSamples = headIndex->GetNumSamples();
    if (!headIndex->HasHeadNodeMeta()) {
        headIndex->InitializeHeadNodeMeta(numHeadSamples);
    }

    for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
        SizeType globalVID = spannInternalIdx->GetGlobalVID(hid);
        headIndex->SetHeadNodeGlobalVID(hid, globalVID);
        if (hid < sigs.num_postings) {
            headIndex->SetHeadNodePS(hid, sigs.ps[hid]);
        }
    }
    return true;
}

bool EnsureHeadNodeMetaLoaded(const std::string& workDir, const std::shared_ptr<SPTAG::VectorIndex>& internalIndex)
{
    auto headIndex = GetMemoryIndexForInternal(internalIndex);
    if (headIndex == nullptr) return false;
    if (headIndex->HasHeadNodeMeta()) return true;
    if (LoadHeadNodeMetaFile(workDir, headIndex)) return true;
    return LoadPostingSignaturesIntoHeadIndex(workDir, internalIndex);
}

constexpr int32_t kHeadNodeRoutingIndexVersion = 2;

struct HeadNodeRoutingIndexFileHeader {
    int32_t version;
    int32_t pivotLevel;
    int32_t nodeCount;
    int32_t numHeadSamples;
    int32_t numTagMappings;
};

struct PivotEstimatorComputation {
    std::vector<PivotEstimatorLevelData> levelData;
    std::vector<double> levelWeights;
    std::vector<PivotEstimatorCandidate> candidates;
};

constexpr double kGreedyLeafMinLocalSelectivity = 0.05;

struct LeafGreedyPlanEntry {
    uint32_t leafTag = 0;
    int leafCount = 0;
    std::vector<uint32_t> ancestorPath;
};

std::string HeadNodeRoutingIndexPath(const std::string& workDir)
{
    return workDir + "/HeadIndex/tag_node_index.bin";
}

void BuildNodeByPivotTag(const PivotEstimatorCandidate& candidate,
                         std::unordered_map<uint32_t, int>& nodeByPivotTag)
{
    nodeByPivotTag.clear();
    for (int nodeId = 0; nodeId < candidate.nodeCount; ++nodeId)
    {
        if (nodeId < 0 || nodeId >= static_cast<int>(candidate.nodePivotTags.size())) continue;
        for (uint32_t pivotTag : candidate.nodePivotTags[nodeId])
        {
            nodeByPivotTag[pivotTag] = nodeId;
        }
    }
}

void CollectNodesForTag(int tagLevel,
                        uint32_t tag,
                        const PivotEstimatorCandidate& candidate,
                        const std::vector<PivotEstimatorLevelData>& levelData,
                        const std::unordered_map<uint32_t, int>& nodeByPivotTag,
                        std::vector<int>& outNodes)
{
    std::unordered_set<int> touchedNodes;

    if (tagLevel < candidate.pivotLevel)
    {
        for (const auto& nodePivotTags : candidate.nodePivotTags)
        {
            for (uint32_t pivotTag : nodePivotTags)
            {
                uint32_t ancestor = 0;
                if (TryGetAncestorTag(pivotTag, candidate.pivotLevel, tagLevel, levelData, ancestor) &&
                    ancestor == tag)
                {
                    auto nodeIt = nodeByPivotTag.find(pivotTag);
                    if (nodeIt != nodeByPivotTag.end()) {
                        touchedNodes.insert(nodeIt->second);
                    }
                }
            }
        }
    }
    else if (tagLevel == candidate.pivotLevel)
    {
        auto nodeIt = nodeByPivotTag.find(tag);
        if (nodeIt != nodeByPivotTag.end()) {
            touchedNodes.insert(nodeIt->second);
        }
    }
    else
    {
        uint32_t ancestorPivot = 0;
        if (TryGetAncestorTag(tag, tagLevel, candidate.pivotLevel, levelData, ancestorPivot)) {
            auto nodeIt = nodeByPivotTag.find(ancestorPivot);
            if (nodeIt != nodeByPivotTag.end()) {
                touchedNodes.insert(nodeIt->second);
            }
        }
    }

    outNodes.assign(touchedNodes.begin(), touchedNodes.end());
    std::sort(outNodes.begin(), outNodes.end());
}

// Compute latencyCost / recallPenalty / totalCost / nodeSizes for `candidate`,
// given the candidate's `nodePivotTags` (leaf-level tag sets) and `pivotLevel`.
// Used both for the legacy single-shot candidate and for per-step Huffman
// snapshots in the tree-aware merge below.
void EvaluatePivotEstimatorCandidateCost(PivotEstimatorCandidate& candidate,
                                         const std::vector<PivotEstimatorLevelData>& levelData,
                                         const std::vector<double>& levelWeights,
                                         int numTagsPerVec,
                                         double totalVectors,
                                         double recallTarget,
                                         double lambdaRecall,
                                         double estimatedRecall)
{
    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    const int leafLevel = numTagsPerVec - 1;
    std::unordered_map<uint32_t, int> leafTagCount;
    if (leafLevel >= 0 && leafLevel < static_cast<int>(levelData.size())) {
        const auto& leafTags = levelData[leafLevel].uniqueTags;
        const auto& leafCounts = levelData[leafLevel].counts;
        leafTagCount.reserve(leafTags.size());
        for (size_t i = 0; i < leafTags.size() && i < leafCounts.size(); ++i) {
            leafTagCount[leafTags[i]] = leafCounts[i];
        }
    }

    candidate.nodeSizes.assign(candidate.nodeCount, 0);
    for (int nid = 0; nid < candidate.nodeCount && nid < static_cast<int>(candidate.nodePivotTags.size()); ++nid)
    {
        int sz = 0;
        for (uint32_t pivotTag : candidate.nodePivotTags[nid]) {
            auto it = leafTagCount.find(pivotTag);
            if (it != leafTagCount.end()) sz += it->second;
        }
        candidate.nodeSizes[nid] = sz;
    }

    double latencyCost = 0.0;
    for (int queryLevel = 0; queryLevel < numTagsPerVec; ++queryLevel)
    {
        double levelCost = 0.0;
        const auto& qTags = levelData[queryLevel].uniqueTags;
        const auto& qCounts = levelData[queryLevel].counts;
        for (size_t qIdx = 0; qIdx < qTags.size(); ++qIdx)
        {
            std::vector<int> touchedNodes;
            CollectNodesForTag(queryLevel, qTags[qIdx], candidate, levelData, nodeByPivotTag, touchedNodes);
            if (touchedNodes.empty()) continue;

            // Sum per-node search cost (each touched subindex runs its own graph
            // traversal). Using sum(log2(node_size)) instead of log2(sum_sizes)
            // captures per-node IO + traversal overhead and penalises over-fragmentation.
            double tagLatency = 0.0;
            for (int nodeId : touchedNodes)
            {
                if (nodeId >= 0 && nodeId < static_cast<int>(candidate.nodeSizes.size())) {
                    double nodeSize = static_cast<double>(candidate.nodeSizes[nodeId]);
                    if (nodeSize > 0.0) {
                        tagLatency += std::log2(nodeSize + 1.0);
                    }
                }
            }
            if (tagLatency <= 0.0) continue;

            double probability = static_cast<double>(qCounts[qIdx]) / totalVectors;
            levelCost += probability * tagLatency;
        }
        if (queryLevel < static_cast<int>(levelWeights.size())) {
            latencyCost += levelWeights[queryLevel] * levelCost;
        }
    }

    candidate.latencyCost = latencyCost;
    candidate.recallPenalty = lambdaRecall * std::max(0.0, recallTarget - estimatedRecall);
    candidate.totalCost = candidate.latencyCost + candidate.recallPenalty;
}

bool BuildPivotEstimatorComputation(const uint32_t* tags,
                                    int numVectors,
                                    int numTagsPerVec,
                                    int maxNodes,
                                    double recallTarget,
                                    double lambdaRecall,
                                    double estimatedRecall,
                                    const std::string& weightsCsv,
                                    PivotEstimatorComputation& out)
{
    out = PivotEstimatorComputation();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0) {
        return false;
    }

    (void)maxNodes;

    out.levelData.resize(numTagsPerVec);
    for (int level = 0; level < numTagsPerVec; ++level)
    {
        std::unordered_map<uint32_t, int> levelCounts;
        levelCounts.reserve(static_cast<size_t>(numVectors) / 4 + 8);
        std::unordered_map<uint32_t, uint32_t> parentByTag;
        if (level > 0) {
            parentByTag.reserve(static_cast<size_t>(numVectors) / 4 + 8);
        }

        for (int vectorId = 0; vectorId < numVectors; ++vectorId)
        {
            const size_t vectorOffset = static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec);
            uint32_t tag = tags[vectorOffset + static_cast<size_t>(level)];
            levelCounts[tag] += 1;

            if (level > 0)
            {
                uint32_t parentTag = tags[vectorOffset + static_cast<size_t>(level - 1)];
                auto parentIt = parentByTag.find(tag);
                if (parentIt == parentByTag.end()) {
                    parentByTag.emplace(tag, parentTag);
                } else if (parentIt->second != parentTag) {
                    return false;
                }
            }
        }

        std::vector<std::pair<uint32_t, int>> pairs(levelCounts.begin(), levelCounts.end());
        std::sort(pairs.begin(), pairs.end(), [](const auto& left, const auto& right) { return left.first < right.first; });

        out.levelData[level].uniqueTags.reserve(pairs.size());
        out.levelData[level].counts.reserve(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i)
        {
            out.levelData[level].uniqueTags.push_back(pairs[i].first);
            out.levelData[level].counts.push_back(pairs[i].second);
        }
        if (level > 0) {
            out.levelData[level].parentByTag = std::move(parentByTag);
        }
    }

    if (!ParseLevelWeights(weightsCsv, numTagsPerVec, out.levelWeights)) {
        out.levelWeights.assign(numTagsPerVec, 1.0 / static_cast<double>(numTagsPerVec));
    }

    const double totalVectors = static_cast<double>(numVectors);

    const int leafLevel = numTagsPerVec - 1;
    const auto& leafTags = out.levelData[leafLevel].uniqueTags;
    const auto& leafCounts = out.levelData[leafLevel].counts;
    if (leafTags.empty() || leafTags.size() != leafCounts.size()) {
        return false;
    }

    std::vector<LeafGreedyPlanEntry> leafEntries;
    leafEntries.reserve(leafTags.size());
    for (size_t idx = 0; idx < leafTags.size(); ++idx)
    {
        if (leafCounts[idx] <= 0) {
            continue;
        }

        LeafGreedyPlanEntry entry;
        entry.leafTag = leafTags[idx];
        entry.leafCount = leafCounts[idx];
        entry.ancestorPath.resize(static_cast<size_t>(numTagsPerVec));

        bool validPath = true;
        for (int level = 0; level <= leafLevel; ++level)
        {
            uint32_t ancestorTag = 0;
            if (!TryGetAncestorTag(entry.leafTag, leafLevel, level, out.levelData, ancestorTag)) {
                validPath = false;
                break;
            }
            entry.ancestorPath[static_cast<size_t>(level)] = ancestorTag;
        }

        if (!validPath) {
            return false;
        }

        leafEntries.emplace_back(std::move(entry));
    }

    if (leafEntries.empty()) {
        return false;
    }

    // ----- Tree-aware Huffman greedy merge -----
    // Each "group" starts as a single leaf and lives at `currentLevel`. At every
    // step we pop the globally smallest group, find the smallest mergeable
    // sibling (same parent at currentLevel-1), and merge them. When no sibling
    // remains under that parent, the group is left aside; after all merges at
    // currentLevel are done, surviving groups are promoted to currentLevel-1.
    // After each individual merge we snapshot the partition into out.candidates
    // so that FindBestPivotEstimatorCandidate picks the lowest-cost stopping
    // point under the configured cost model.
    struct HuffGroup {
        int level;                       // current depth (leafLevel..0)
        uint32_t anchorTag;              // tag value at `level` representing this group
        int size;                        // total vector count
        std::vector<uint32_t> leafTags;  // leaf-level tags covered
        bool alive;
    };

    std::vector<HuffGroup> groups;
    groups.reserve(leafEntries.size());
    for (const auto& le : leafEntries) {
        HuffGroup g;
        g.level = leafLevel;
        g.anchorTag = le.leafTag;
        g.size = le.leafCount;
        g.leafTags = { le.leafTag };
        g.alive = true;
        groups.push_back(std::move(g));
    }

    auto ancestorAt = [&](uint32_t tag, int fromLevel, int targetLevel) -> uint32_t {
        if (fromLevel == targetLevel) return tag;
        uint32_t a = tag;
        TryGetAncestorTag(tag, fromLevel, targetLevel, out.levelData, a);
        return a;
    };

    // Build a candidate snapshot from currently-alive groups and evaluate cost.
    auto snapshotCandidate = [&]() {
        PivotEstimatorCandidate cand;
        cand.pivotLevel = leafLevel;
        cand.nodePivotTags.reserve(groups.size());
        for (const auto& g : groups) {
            if (!g.alive) continue;
            std::vector<uint32_t> tagsCopy = g.leafTags;
            std::sort(tagsCopy.begin(), tagsCopy.end());
            cand.nodePivotTags.push_back(std::move(tagsCopy));
        }
        cand.nodeCount = static_cast<int>(cand.nodePivotTags.size());
        if (cand.nodeCount > 0) {
            EvaluatePivotEstimatorCandidateCost(cand, out.levelData, out.levelWeights,
                                                numTagsPerVec, totalVectors,
                                                recallTarget, lambdaRecall, estimatedRecall);
            out.candidates.push_back(std::move(cand));
        }
    };

    // Snapshot the initial state (every leaf is its own node).
    snapshotCandidate();

    int currentLevel = leafLevel;
    while (currentLevel >= 0)
    {
        // Bucket alive groups at currentLevel by parent at currentLevel-1.
        // At currentLevel == 0 (top org-level) everyone shares a virtual root,
        // so we collapse them into a single bucket and keep merging until one.
        constexpr uint32_t kVirtualRootTag = std::numeric_limits<uint32_t>::max();
        std::unordered_map<uint32_t, std::unordered_set<int>> sibBuckets;
        for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
            if (!groups[i].alive || groups[i].level != currentLevel) continue;
            uint32_t parentTag = (currentLevel == 0)
                ? kVirtualRootTag
                : ancestorAt(groups[i].anchorTag, currentLevel, currentLevel - 1);
            sibBuckets[parentTag].insert(i);
        }

        // Global min-heap of alive groups at currentLevel by size, breaking ties by idx.
        using HeapEntry = std::tuple<int, int, int>;  // (size, version, idx)
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
        std::vector<int> version(groups.size(), 0);
        for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
            if (groups[i].alive && groups[i].level == currentLevel)
                heap.emplace(groups[i].size, 0, i);
        }

        while (!heap.empty()) {
            HeapEntry top = heap.top(); heap.pop();
            int sz = std::get<0>(top);
            int ver = std::get<1>(top);
            int idx = std::get<2>(top);
            if (!groups[idx].alive || groups[idx].level != currentLevel) continue;
            if (ver != version[idx] || groups[idx].size != sz) continue;  // stale entry

            uint32_t parentTag = (currentLevel == 0)
                ? kVirtualRootTag
                : ancestorAt(groups[idx].anchorTag, currentLevel, currentLevel - 1);
            auto bIt = sibBuckets.find(parentTag);
            if (bIt == sibBuckets.end()) continue;
            auto& sibSet = bIt->second;

            // Find smallest alive sibling (excluding idx).
            int bestSib = -1;
            int bestSibSize = std::numeric_limits<int>::max();
            for (int s : sibSet) {
                if (s == idx) continue;
                if (!groups[s].alive || groups[s].level != currentLevel) continue;
                if (groups[s].size < bestSibSize) {
                    bestSibSize = groups[s].size;
                    bestSib = s;
                }
            }
            if (bestSib < 0) {
                // Sole survivor under this parent at this level → cannot merge
                // further here; remove from bucket so we stop revisiting it.
                sibSet.erase(idx);
                continue;
            }

            // Merge bestSib into idx.
            groups[idx].size += groups[bestSib].size;
            groups[idx].leafTags.insert(groups[idx].leafTags.end(),
                                        groups[bestSib].leafTags.begin(),
                                        groups[bestSib].leafTags.end());
            groups[bestSib].alive = false;
            sibSet.erase(bestSib);

            // Re-push the grown group; bump its version so older entries become stale.
            version[idx] += 1;
            heap.emplace(groups[idx].size, version[idx], idx);

            snapshotCandidate();
        }

        if (currentLevel == 0) break;

        // Promote remaining alive groups to the parent level.
        for (auto& g : groups) {
            if (!g.alive || g.level != currentLevel) continue;
            uint32_t newAnchor = ancestorAt(g.anchorTag, currentLevel, currentLevel - 1);
            g.level = currentLevel - 1;
            g.anchorTag = newAnchor;
        }
        currentLevel -= 1;
    }

    return !out.candidates.empty();
}

const PivotEstimatorCandidate* FindBestPivotEstimatorCandidate(const std::vector<PivotEstimatorCandidate>& candidates)
{
    if (candidates.empty()) return nullptr;

    // Optional override: SPTAG_PIVOT_FORCE_NODE_COUNT=N pins the chosen
    // candidate to the snapshot whose nodeCount is N (smallest |diff| wins).
    // Useful for ablations (e.g. N=1 reproduces a single-shard "original"
    // layout while keeping the node-aware code path warm).
    const char* envForce = std::getenv("SPTAG_PIVOT_FORCE_NODE_COUNT");
    if (envForce != nullptr && envForce[0] != '\0') {
        int target = std::atoi(envForce);
        if (target > 0) {
            const PivotEstimatorCandidate* pick = &candidates.front();
            int bestDiff = std::abs(pick->nodeCount - target);
            for (const auto& c : candidates) {
                int d = std::abs(c.nodeCount - target);
                if (d < bestDiff) { bestDiff = d; pick = &c; }
            }
            return pick;
        }
    }

    const PivotEstimatorCandidate* best = &candidates.front();
    for (const auto& candidate : candidates)
    {
        if (candidate.totalCost < best->totalCost) {
            best = &candidate;
        }
    }
    return best;
}

void BuildTagToNodeIndexForCandidate(const PivotEstimatorCandidate& candidate,
                                     const std::vector<PivotEstimatorLevelData>& levelData,
                                     std::unordered_map<uint32_t, std::vector<int>>& tagToNodes)
{
    tagToNodes.clear();

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    for (int level = 0; level < static_cast<int>(levelData.size()); ++level)
    {
        for (uint32_t tag : levelData[level].uniqueTags)
        {
            std::vector<int> nodes;
            CollectNodesForTag(level, tag, candidate, levelData, nodeByPivotTag, nodes);
            if (!nodes.empty()) {
                auto& merged = tagToNodes[tag];
                merged.insert(
                    merged.end(),
                    nodes.begin(), nodes.end());
            }
        }
    }
    for (auto& entry : tagToNodes) {
        auto& nodes = entry.second;
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(
            std::unique(nodes.begin(), nodes.end()),
            nodes.end());
    }
}

void BuildNodeVectorAssignmentsForTagToNodes(const uint32_t* tags,
                                             int numVectors,
                                             int numTagsPerVec,
                                             const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                             int nodeCount,
                                             std::vector<std::vector<int>>& nodeVectors)
{
    nodeVectors.clear();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0 || nodeCount <= 0) {
        return;
    }

    nodeVectors.assign(nodeCount, std::vector<int>());
    for (int vectorId = 0; vectorId < numVectors; ++vectorId)
    {
        std::unordered_set<int> touchedNodes;
        const size_t vectorOffset = static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec);
        for (int tagIdx = 0; tagIdx < numTagsPerVec; ++tagIdx)
        {
            auto tagIt = tagToNodes.find(tags[vectorOffset + static_cast<size_t>(tagIdx)]);
            if (tagIt == tagToNodes.end()) {
                continue;
            }

            touchedNodes.insert(tagIt->second.begin(), tagIt->second.end());
        }

        if (touchedNodes.empty()) {
            continue;
        }

        for (int nodeId : touchedNodes)
        {
            if (nodeId >= 0 && nodeId < nodeCount) {
                nodeVectors[static_cast<size_t>(nodeId)].push_back(vectorId);
            }
        }
    }
}

void BuildPrimaryNodeVectorAssignmentsForCandidate(const PivotEstimatorCandidate& candidate,
                                                   const uint32_t* tags,
                                                   int numVectors,
                                                   int numTagsPerVec,
                                                   std::vector<std::vector<int>>& nodeVectors)
{
    nodeVectors.clear();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0 || candidate.nodeCount <= 0) {
        return;
    }

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    nodeVectors.assign(candidate.nodeCount, std::vector<int>());
    for (int vectorId = 0; vectorId < numVectors; ++vectorId)
    {
        uint32_t pivotTag = tags[static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec) + static_cast<size_t>(candidate.pivotLevel)];
        auto nodeIt = nodeByPivotTag.find(pivotTag);
        if (nodeIt == nodeByPivotTag.end()) {
            continue;
        }

        int nodeId = nodeIt->second;
        if (nodeId >= 0 && nodeId < candidate.nodeCount) {
            nodeVectors[static_cast<size_t>(nodeId)].push_back(vectorId);
        }
    }
}

void BuildHeadNodeToNodeIndexForCandidate(const PivotEstimatorCandidate& candidate,
                                          const uint32_t* tags,
                                          int numVectors,
                                          int numTagsPerVec,
                                          const std::shared_ptr<SPTAG::VectorIndex>& memoryIndex,
                                          SPTAG::SPANN::ISPANNIndex* spannInternalIdx,
                                          std::vector<int>& headNodeToNode)
{
    headNodeToNode.clear();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0 || memoryIndex == nullptr || spannInternalIdx == nullptr) {
        return;
    }

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    const SizeType numHeadSamples = memoryIndex->GetNumSamples();
    headNodeToNode.assign(numHeadSamples, -1);
    const bool useMeta = memoryIndex->HasHeadNodeMeta();
    for (SizeType hid = 0; hid < numHeadSamples; ++hid)
    {
        SizeType globalVID = useMeta ? memoryIndex->GetHeadNodeGlobalVID(hid)
                                     : spannInternalIdx->GetGlobalVID(hid);
        if (globalVID == SPTAG::MaxSize || globalVID >= static_cast<SizeType>(numVectors)) {
            continue;
        }

        uint32_t pivotTag = tags[static_cast<size_t>(globalVID) * static_cast<size_t>(numTagsPerVec) + static_cast<size_t>(candidate.pivotLevel)];
        auto nodeIt = nodeByPivotTag.find(pivotTag);
        if (nodeIt != nodeByPivotTag.end()) {
            headNodeToNode[hid] = nodeIt->second;
        }
    }
}

bool TryCollectRoutingNodesForQuery(const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                    const uint32_t* queryTags,
                                    int numQueryTags,
                                    std::vector<int>& outNodes)
{
    outNodes.clear();
    if (queryTags == nullptr || numQueryTags <= 0) {
        return false;
    }

    std::unordered_set<int> unionNodes;
    for (int idx = 0; idx < numQueryTags; ++idx)
    {
        auto tagIt = tagToNodes.find(queryTags[idx]);
        if (tagIt == tagToNodes.end() || tagIt->second.empty()) {
            outNodes.clear();
            return false;
        }

        unionNodes.insert(tagIt->second.begin(), tagIt->second.end());
    }

    outNodes.assign(unionNodes.begin(), unionNodes.end());
    std::sort(outNodes.begin(), outNodes.end());
    return !outNodes.empty();
}

bool TryCollectRoutingNodesForDNF(
    const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
    const SPTAG::Cache::DNFPredicate& dnf,
    std::vector<int>& outNodes)
{
    outNodes.clear();
    if (dnf.Empty()) return false;

    std::unordered_set<int> unionNodes;
    for (const auto& clause : dnf.clauses) {
        std::unordered_set<int> clauseNodes;
        bool constrained = false;
        for (const auto& literal : clause.lits) {
            if (literal.kind != 0) continue;
            if (literal.op != SPTAG::Cache::DNF_EQ) {
                outNodes.clear();
                return false;
            }
            const auto tag = tagToNodes.find(literal.val);
            if (tag == tagToNodes.end() || tag->second.empty()) {
                outNodes.clear();
                return false;
            }
            if (!constrained) {
                clauseNodes.insert(
                    tag->second.begin(), tag->second.end());
                constrained = true;
                continue;
            }
            std::unordered_set<int> intersection;
            for (int node : tag->second) {
                if (clauseNodes.count(node) != 0) {
                    intersection.insert(node);
                }
            }
            clauseNodes.swap(intersection);
        }
        if (!constrained || clauseNodes.empty()) {
            outNodes.clear();
            return false;
        }
        unionNodes.insert(
            clauseNodes.begin(), clauseNodes.end());
    }

    outNodes.assign(unionNodes.begin(), unionNodes.end());
    std::sort(outNodes.begin(), outNodes.end());
    return !outNodes.empty();
}

bool SaveHeadNodeRoutingIndexFile(const std::string& workDir,
                                  int pivotLevel,
                                  const std::vector<std::vector<uint32_t>>& nodePivotTags,
                                  const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                  const std::vector<int>& headNodeToNode)
{
    std::string path = HeadNodeRoutingIndexPath(workDir);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    HeadNodeRoutingIndexFileHeader header{};
    header.version = kHeadNodeRoutingIndexVersion;
    header.pivotLevel = pivotLevel;
    header.nodeCount = static_cast<int32_t>(nodePivotTags.size());
    header.numHeadSamples = static_cast<int32_t>(headNodeToNode.size());
    header.numTagMappings = static_cast<int32_t>(tagToNodes.size());

    bool ok = fwrite(&header, sizeof(header), 1, f) == 1;
    for (const auto& tagsForNode : nodePivotTags)
    {
        int32_t tagCount = static_cast<int32_t>(tagsForNode.size());
        ok = ok && fwrite(&tagCount, sizeof(tagCount), 1, f) == 1;
        if (tagCount > 0) {
            ok = ok && fwrite(tagsForNode.data(), sizeof(uint32_t), tagCount, f) == static_cast<size_t>(tagCount);
        }
    }

    std::vector<std::pair<uint32_t, std::vector<int>>> mappings(tagToNodes.begin(), tagToNodes.end());
    std::sort(mappings.begin(), mappings.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
    for (auto& [tag, nodes] : mappings)
    {
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

        int32_t nodeCount = static_cast<int32_t>(nodes.size());
        ok = ok && fwrite(&tag, sizeof(tag), 1, f) == 1;
        ok = ok && fwrite(&nodeCount, sizeof(nodeCount), 1, f) == 1;
        if (nodeCount > 0) {
            ok = ok && fwrite(nodes.data(), sizeof(int32_t), nodeCount, f) == static_cast<size_t>(nodeCount);
        }
    }

    if (!headNodeToNode.empty()) {
        ok = ok && fwrite(headNodeToNode.data(), sizeof(int32_t), headNodeToNode.size(), f) == headNodeToNode.size();
    }
    fclose(f);
    return ok;
}

bool LoadHeadNodeRoutingIndexFile(const std::string& workDir,
                                  int& pivotLevel,
                                  int& nodeCount,
                                  std::vector<std::vector<uint32_t>>& nodePivotTags,
                                  std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                  std::vector<int>& headNodeToNode)
{
    pivotLevel = -1;
    nodeCount = 0;
    nodePivotTags.clear();
    tagToNodes.clear();
    headNodeToNode.clear();

    std::string path = HeadNodeRoutingIndexPath(workDir);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    HeadNodeRoutingIndexFileHeader header{};
    bool ok = fread(&header, sizeof(header), 1, f) == 1;
    if (!ok || header.version != kHeadNodeRoutingIndexVersion || header.nodeCount < 0 ||
        header.numHeadSamples < 0 || header.numTagMappings < 0) {
        fclose(f);
        return false;
    }

    pivotLevel = header.pivotLevel;
    nodeCount = header.nodeCount;
    nodePivotTags.assign(header.nodeCount, std::vector<uint32_t>());
    for (int nodeId = 0; nodeId < header.nodeCount; ++nodeId)
    {
        int32_t tagCount = 0;
        ok = fread(&tagCount, sizeof(tagCount), 1, f) == 1;
        if (!ok || tagCount < 0) {
            fclose(f);
            return false;
        }
        nodePivotTags[nodeId].resize(tagCount);
        if (tagCount > 0) {
            ok = fread(nodePivotTags[nodeId].data(), sizeof(uint32_t), tagCount, f) == static_cast<size_t>(tagCount);
            if (!ok) {
                fclose(f);
                return false;
            }
        }
    }

    for (int mappingId = 0; mappingId < header.numTagMappings; ++mappingId)
    {
        uint32_t tag = 0;
        int32_t mappingNodeCount = 0;
        ok = fread(&tag, sizeof(tag), 1, f) == 1;
        ok = ok && fread(&mappingNodeCount, sizeof(mappingNodeCount), 1, f) == 1;
        if (!ok || mappingNodeCount < 0) {
            fclose(f);
            return false;
        }

        std::vector<int> nodes(mappingNodeCount);
        if (mappingNodeCount > 0) {
            ok = fread(nodes.data(), sizeof(int32_t), mappingNodeCount, f) == static_cast<size_t>(mappingNodeCount);
            if (!ok) {
                fclose(f);
                return false;
            }
        }
        tagToNodes.emplace(tag, std::move(nodes));
    }

    headNodeToNode.resize(header.numHeadSamples);
    if (header.numHeadSamples > 0) {
        ok = fread(headNodeToNode.data(), sizeof(int32_t), header.numHeadSamples, f) == static_cast<size_t>(header.numHeadSamples);
    }
    fclose(f);
    return ok;
}

AnnIndex::AnnIndex(DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::BKT), m_inputValueType(SPTAG::VectorValueType::Float), m_dimension(p_dimension)
{
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::AnnIndex(const char *p_algoType, const char *p_valueType, DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::Undefined), m_inputValueType(SPTAG::VectorValueType::Undefined),
      m_dimension(p_dimension)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_inputValueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::AnnIndex(const std::shared_ptr<SPTAG::VectorIndex> &p_index)
    : m_algoType(p_index->GetIndexAlgoType()), m_inputValueType(p_index->GetVectorValueType()),
      m_dimension(p_index->GetFeatureDim()), m_index(p_index)
{
    m_inputVectorSize = p_index->m_pQuantizer ? p_index->m_pQuantizer->GetNumSubvectors()
                                              : SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::~AnnIndex()
{
}

bool AnnIndex::BuildSPANN(bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index)
        return false;

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

bool AnnIndex::BuildSPANNWithMetaData(ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index)
        return false;

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex)
        m_index->BuildMetaMapping(false);

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

// Build SPANN index with both vector data and metadata (for attribute filtering support)
bool AnnIndex::BuildSPANNWithDataAndMeta(ByteArray p_data, ByteArray p_meta, SizeType p_num,
                                          bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
        return false;

    // Set metadata first (before build, so it's available during search)
    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex)
        m_index->BuildMetaMapping(false);

    // Build with in-memory vector data
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                              (SPTAG::DimensionType)m_dimension, p_normalized));
}

bool AnnIndex::Build(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                             (SPTAG::DimensionType)m_dimension, p_normalized,
                                                             m_shareBuildOwnership));
}

bool AnnIndex::BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex,
                                 bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    auto vectorType = m_index->m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_inputValueType;
    auto vectorSize = m_index->m_pQuantizer ? m_index->m_pQuantizer->GetNumSubvectors() : m_dimension;
    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(
        p_data, vectorType, static_cast<SPTAG::DimensionType>(vectorSize), static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize));
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(vectors, meta, p_withMetaIndex, p_normalized));
}

void AnnIndex::SetBuildParam(const char *p_name, const char *p_value, const char *p_section)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return;
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    m_index->SetParameter(p_name, p_value, p_section);
}

void AnnIndex::SetSearchParam(const char *p_name, const char *p_value, const char *p_section)
{
    if (nullptr != m_index)
        m_index->SetParameter(p_name, p_value, p_section);
}

std::shared_ptr<ResultIterator> AnnIndex::GetIterator(ByteArray p_target)
{
    if (nullptr != m_index)
        return m_index->GetIterator(p_target.Data());
    return nullptr;
}

bool AnnIndex::LoadQuantizer(const char *p_quantizerFile)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return false;
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }

    auto ret = (m_index->LoadQuantizer(p_quantizerFile) == SPTAG::ErrorCode::Success);
    if (ret)
    {
        m_inputVectorSize = m_index->m_pQuantizer->QuantizeSize();
    }
    return ret;
}

void AnnIndex::SetQuantizerADC(bool p_adc)
{
    if (nullptr != m_index)
        return m_index->SetQuantizerADC(p_adc);
}

ByteArray AnnIndex::QuantizeVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr)
    {
        size_t outsize = m_index->GetQuantizer()->GetNumSubvectors() * (size_t)p_num;
        std::uint8_t *outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success !=
            m_index->QuantizeVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false)))
            return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}

ByteArray AnnIndex::ReconstructVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr)
    {
        size_t outsize = m_index->GetQuantizer()->ReconstructSize() * (size_t)p_num;
        std::uint8_t *outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success !=
            m_index->ReconstructVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false)))
            return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}

std::shared_ptr<QueryResult> AnnIndex::Search(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, false);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::SearchWithMetaData(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::BatchSearch(ByteArray p_data, int p_vectorNum, int p_resultNum,
                                                   bool p_withMetaData)
{
    std::shared_ptr<QueryResult> results =
        std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, p_withMetaData);
    if (nullptr != m_index)
    {
        m_index->SearchIndex(p_data.Data(), p_vectorNum, p_resultNum, p_withMetaData, results->GetResults());
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::SearchWithTenantFilter(ByteArray p_data, int p_resultNum, const char* p_tenantId)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);
    
    if (nullptr != m_index && nullptr != p_tenantId)
    {
        // Create filter function that checks if metadata exactly matches tenantId
        std::string tenantId(p_tenantId);
        auto filterFunc = [tenantId](const SPTAG::ByteArray& metadata) -> bool {
            if (metadata.Length() == 0) return false;
            std::string meta(reinterpret_cast<const char*>(metadata.Data()), metadata.Length());
            // Trim trailing whitespace/newline
            while (!meta.empty() && (meta.back() == '\n' || meta.back() == '\r' || meta.back() == ' '))
                meta.pop_back();
            return meta == tenantId;
        };
        
        m_index->SearchIndexWithFilter(*results, filterFunc, 0, false);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::BatchSearchWithTenantFilter(ByteArray p_data, int p_vectorNum, 
                                                                    int p_resultNum, const char* p_tenantId)
{
    std::shared_ptr<QueryResult> results = 
        std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, true);
    
    if (nullptr != m_index && nullptr != p_tenantId && nullptr != p_data.Data())
    {
        // For batch search with filter, we need to process each vector separately
        // since the batch SearchIndex doesn't support filtering
        std::string tenantId(p_tenantId);
        auto filterFunc = [tenantId](const SPTAG::ByteArray& metadata) -> bool {
            if (metadata.Length() == 0) return false;
            std::string meta(reinterpret_cast<const char*>(metadata.Data()), metadata.Length());
            return meta.find(tenantId) != std::string::npos;
        };
        
        SPTAG::BasicResult* results_array = results->GetResults();
        const char* data = reinterpret_cast<const char*>(p_data.Data());
        size_t vectorSize = p_data.Length() / p_vectorNum;
        
        for (int i = 0; i < p_vectorNum; i++)
        {
            SPTAG::QueryResult singleQuery(data + i * vectorSize, p_resultNum, true);
            m_index->SearchIndexWithFilter(singleQuery, filterFunc, 0, false);
            
            // Copy results
            for (int j = 0; j < p_resultNum && j < singleQuery.GetResultNum(); j++)
            {
                auto* one = singleQuery.GetResult(j);
                if (one != nullptr)
                {
                    results_array[i * p_resultNum + j] = *one;
                }
            }
        }
    }
    return std::move(results);
}

bool AnnIndex::ReadyToServe() const
{
    return m_index != nullptr;
}

void AnnIndex::SetVectorTags(const uint32_t* tags, int numVecs, int numTagsPerVec)
{
    if (!m_index) return;
    // Cast to the non-templated SPANN interface to access SetVectorTags
    auto* spannIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(m_index.get());
    if (spannIdx) {
        spannIdx->SetVectorTags(tags, numVecs, numTagsPerVec);
    }
}

void AnnIndex::SetNodeVectorAssignments(const std::vector<std::vector<int>>& nodeVectorAssignments)
{
    if (!m_index) return;
    auto* spannIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(m_index.get());
    if (!spannIdx) return;

    std::vector<std::vector<SPTAG::SizeType>> convertedAssignments;
    convertedAssignments.reserve(nodeVectorAssignments.size());
    for (const auto& nodeVectors : nodeVectorAssignments)
    {
        std::vector<SPTAG::SizeType> convertedNode;
        convertedNode.reserve(nodeVectors.size());
        for (int vectorId : nodeVectors)
        {
            if (vectorId >= 0) {
                convertedNode.push_back(static_cast<SPTAG::SizeType>(vectorId));
            }
        }
        convertedAssignments.emplace_back(std::move(convertedNode));
    }

    spannIdx->SetNodeVectorAssignments(convertedAssignments);
}

void AnnIndex::SetPrimaryNodeVectorAssignments(const std::vector<std::vector<int>>& primaryNodeVectorAssignments)
{
    if (!m_index) return;
    auto* spannIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(m_index.get());
    if (!spannIdx) return;

    std::vector<std::vector<SPTAG::SizeType>> convertedAssignments;
    convertedAssignments.reserve(primaryNodeVectorAssignments.size());
    for (const auto& nodeVectors : primaryNodeVectorAssignments)
    {
        std::vector<SPTAG::SizeType> convertedNode;
        convertedNode.reserve(nodeVectors.size());
        for (int vectorId : nodeVectors)
        {
            if (vectorId >= 0) {
                convertedNode.push_back(static_cast<SPTAG::SizeType>(vectorId));
            }
        }
        convertedAssignments.emplace_back(std::move(convertedNode));
    }

    spannIdx->SetPrimaryNodeVectorAssignments(convertedAssignments);
}

bool AnnIndex::SetSharedDB(std::shared_ptr<SPTAG::Helper::KeyValueIO> p_db)
{
    if (m_index == nullptr)
    {
        if (m_algoType == SPTAG::IndexAlgoType::Undefined ||
            m_inputValueType == SPTAG::VectorValueType::Undefined)
            return false;
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
        if (m_index == nullptr) return false;
    }
    if (m_algoType != SPTAG::IndexAlgoType::SPANN) return false;
    // Any SPANN value type (float/uint8/int8/...) exposes SetSharedDB via the
    // non-templated interface.
    auto* spann = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(m_index.get());
    if (spann == nullptr) return false;
    spann->SetSharedDB(std::move(p_db));
    return true;
}

void AnnIndex::UpdateIndex()
{
    m_index->UpdateIndex();
}

bool AnnIndex::Save(const char *p_savefile) const
{
    return SPTAG::ErrorCode::Success == m_index->SaveIndex(p_savefile);
}

bool AnnIndex::Add(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->AddIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                           (SPTAG::DimensionType)m_dimension, nullptr, false,
                                                           p_normalized));
}

bool AnnIndex::AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex,
                               bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(
        p_data, m_inputValueType, static_cast<SPTAG::DimensionType>(m_dimension), static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num));
    return (SPTAG::ErrorCode::Success == m_index->AddIndex(vectors, meta, p_withMetaIndex, p_normalized));
}

bool AnnIndex::AddWithTags(ByteArray p_data, ByteArray p_tags, SizeType p_num,
                           int p_numTagsPerVec, bool p_normalized)
{
    if (m_index == nullptr || m_algoType != SPTAG::IndexAlgoType::SPANN ||
        p_num == 0 || p_numTagsPerVec <= 0 || m_dimension == 0 ||
        p_data.Length() != p_num * m_inputVectorSize ||
        p_tags.Length() != p_num * static_cast<SizeType>(p_numTagsPerVec) * sizeof(std::uint32_t)) {
        return false;
    }
    auto* spann = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(m_index.get());
    return spann != nullptr &&
           spann->AddIndexWithTags(p_data.Data(), static_cast<SPTAG::SizeType>(p_num),
                                   static_cast<SPTAG::DimensionType>(m_dimension),
                                   reinterpret_cast<const std::uint32_t*>(p_tags.Data()),
                                   p_numTagsPerVec, p_normalized) == SPTAG::ErrorCode::Success;
}

bool AnnIndex::Delete(ByteArray p_data, SizeType p_num)
{
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_data.Data(), (SPTAG::SizeType)p_num));
}

bool AnnIndex::DeleteByMetaData(ByteArray p_meta)
{
    if (nullptr == m_index)
        return false;

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_meta));
}

uint64_t AnnIndex::CalculateBufferSize()
{
    if (nullptr == m_index)
        return 0;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    uint64_t total = sizeof(int) + sizeof(uint64_t) * buffersize->size();
    for (uint64_t bs : *buffersize)
    {
        total += bs;
    }
    return total;
}

ByteArray AnnIndex::Dump(ByteArray p_blobs)
{
    if (nullptr == m_index)
        return ByteArray::c_empty;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    std::uint8_t *ptr = p_blobs.Data(), *pdata = ptr + sizeof(int) + sizeof(uint64_t) * buffersize->size();
    *((int *)ptr) = (int)(buffersize->size());
    ptr += sizeof(int);

    std::vector<SPTAG::ByteArray> indexBlobs;
    for (size_t i = 0; i < buffersize->size(); i++)
    {
        *((uint64_t *)ptr) = buffersize->at(i);
        ptr += sizeof(uint64_t);
        indexBlobs.push_back(SPTAG::ByteArray(pdata, buffersize->at(i), false));
        pdata += buffersize->at(i);
    }

    std::string config;
    if (SPTAG::ErrorCode::Success != m_index->SaveIndex(config, indexBlobs))
    {
        return ByteArray::c_empty;
    }
    std::uint8_t *newdata = new std::uint8_t[config.size()];
    memcpy(newdata, config.c_str(), config.size());
    return ByteArray(newdata, config.size(), false);
}

AnnIndex AnnIndex::LoadFromDump(ByteArray p_config, ByteArray p_blobs)
{
    if (p_config.Length() == 0)
        return AnnIndex(0);

    std::uint8_t *ptr = p_blobs.Data();
    int streamNum = *((int *)ptr);
    ptr += sizeof(int);
    std::uint8_t *pdata = ptr + sizeof(uint64_t) * streamNum;

    std::vector<SPTAG::ByteArray> p_indexBlobs;
    for (int i = 0; i < streamNum; i++)
    {
        std::uint64_t streamSize = *((uint64_t *)ptr);
        ptr += sizeof(uint64_t);
        p_indexBlobs.push_back(SPTAG::ByteArray((std::uint8_t *)pdata, streamSize, false));
        pdata += streamSize;
    }

    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    std::string config((char *)p_config.Data(), p_config.Length());
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(config, p_indexBlobs, vecIndex) ||
        nullptr == vecIndex)
    {
        return AnnIndex(0);
    }
    return AnnIndex(vecIndex);
}

AnnIndex AnnIndex::Load(const char *p_loaderFile)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    auto ret = SPTAG::VectorIndex::LoadIndex(p_loaderFile, vecIndex);
    if (SPTAG::ErrorCode::Success != ret || nullptr == vecIndex)
    {
        return AnnIndex(0);
    }

    return AnnIndex(vecIndex);
}

AnnIndex AnnIndex::Merge(const char *p_indexFilePath1, const char *p_indexFilePath2)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex, addIndex;
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath1, vecIndex) ||
        SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath2, addIndex) ||
        SPTAG::ErrorCode::Success !=
            vecIndex->MergeIndex(addIndex.get(), std::atoi(vecIndex->GetParameter("NumberOfThreads").c_str()), nullptr))
        return AnnIndex(0);

    return AnnIndex(vecIndex);
}

// ============================================================================
// TenantIndexManager Implementation
// ============================================================================

TenantIndexManager::TenantIndexManager(DimensionType p_dimension, const char* p_algoType, const char* p_valueType)
    : m_dimension(p_dimension), m_algoType(SPTAG::IndexAlgoType::Undefined), 
    m_valueType(SPTAG::VectorValueType::Undefined),
    m_headIndexCacheLimitBytes(1024*1024*1024),  // Default 1GB cache limit
    m_headIndexCacheSafetyFactor(1.3),
    m_headCache(nullptr)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_valueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_valueType) * m_dimension;

    // Defaults preserve the service configuration. Benchmarks and deployments that
    // need more independent client channels can opt in before manager construction.
    const int aioContexts = ReadPositiveEnvironmentInt("SPTAG_SHARED_AIO_CONTEXTS", 4);
    const int aioEvents = ReadPositiveEnvironmentInt("SPTAG_SHARED_AIO_EVENTS", 1024);
    SPTAG::Helper::SharedAIOPool::Instance().Initialize(aioContexts, aioEvents);
}

TenantIndexManager::~TenantIndexManager()
{
    if (m_headCache) m_headCache->Clear();
    m_tenantIndices.clear();
    m_lruList.clear();
    m_lruMap.clear();
    m_tenantHeadIndexAccountedBytes.clear();
    m_loadedHeadIndexBytes = 0;
    m_tenantVectorCounts.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantPostingOffsets.clear();
    m_tenantHeadCounts.clear();
    m_tenantTagRoutingStats.clear();
    m_tenantPivotLevels.clear();
    m_tenantPivotNodeCounts.clear();
    m_tenantNodePivotTags.clear();
    m_tenantPlannedNodeVectors.clear();
    m_tenantPlannedPrimaryNodeVectors.clear();
    m_tenantTagToNodes.clear();
    m_tenantHeadNodeToNode.clear();

    // Shut the shared RocksDB down only after every tenant index (which holds
    // a TenantPrefixedKeyValueIO referencing the shared DB through a
    // shared_ptr) has been released.
    if (m_sharedDB)
    {
        m_sharedDB->ShutDown();
        m_sharedDB.reset();
    }
}

bool TenantIndexManager::EnsureSharedDB()
{
#ifndef ROCKSDB
    fprintf(stderr, "[ERROR] TenantIndexManager: shared RocksDB requested but binary built without ROCKSDB.\n");
    return false;
#else
    std::lock_guard<std::mutex> lk(m_sharedDBMutex);
    if (m_sharedDB) return true;
    std::string base = m_baseStoragePath.empty() ? std::string("./tenant_index") : m_baseStoragePath;
    EnsureDir(base);
    if (m_baseStoragePath.empty()) m_baseStoragePath = base;

    // Single shared DB at <baseDir>/rocksdb_shared_0/. The trailing _0 mirrors
    // PrepareDB()'s per-layer suffix; only layer 0 is exercised today.
    std::string dbPath = base + "/rocksdb_shared_0";
    std::shared_ptr<SPTAG::SPANN::RocksDBIO> db;
    try
    {
        db = std::make_shared<SPTAG::SPANN::RocksDBIO>(
            dbPath.c_str(), m_useDirectIO, m_enableWAL, /*recovery=*/false);
    }
    catch (...)
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: failed to open shared RocksDB at %s\n", dbPath.c_str());
        return false;
    }
    if (!db || !db->Available())
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: shared RocksDB unavailable at %s\n", dbPath.c_str());
        return false;
    }
    m_sharedDB = db;
    fprintf(stderr, "[INFO] TenantIndexManager: opened shared RocksDB at %s\n", dbPath.c_str());
    return true;
#endif
}

bool TenantIndexManager::InjectSharedDB(const std::shared_ptr<AnnIndex>& p_idx, int p_internalId)
{
    if (!p_idx) return false;
    if (!m_sharedDB)
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: shared DB not initialised before InjectSharedDB.\n");
        return false;
    }
    auto wrapper = std::make_shared<SPTAG::Helper::TenantPrefixedKeyValueIO>(m_sharedDB, p_internalId);
    if (!p_idx->SetSharedDB(wrapper))
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: tenant %d index is not SPANN<Float>; cannot share DB.\n", p_internalId);
        return false;
    }
    return true;
}

std::shared_ptr<AnnIndex> TenantIndexManager::LoadSpannWithSharedDB(const std::string& p_folder, int p_internalId)
{
    using namespace SPTAG;

    std::string folderPath = p_folder;
    if (!folderPath.empty() && folderPath.back() != FolderSep) folderPath += FolderSep;

    Helper::IniReader iniReader;
    {
        auto fp = SPTAG::f_createIO();
        if (fp == nullptr || !fp->Initialize((folderPath + "indexloader.ini").c_str(), std::ios::in))
            return nullptr;
        if (ErrorCode::Success != iniReader.LoadIni(fp)) return nullptr;
    }

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);
    std::shared_ptr<VectorIndex> vecIndex = VectorIndex::CreateInstance(algoType, valueType);
    if (vecIndex == nullptr) return nullptr;

    if (vecIndex->LoadIndexConfig(iniReader) != ErrorCode::Success) return nullptr;

    if (algoType == IndexAlgoType::SPANN)
    {
        vecIndex->SetParameter("IndexDirectory", p_folder.c_str(), "Base");
        // Disable lazy per-tenant DB creation; the searcher must use m_externalDB.
        vecIndex->SetParameter("ShareDB", "true", "BuildSSDIndex");
        if (!EnsureSharedDB()) return nullptr;
        // SPANN of any value type exposes SetSharedDB via the non-templated
        // interface (previously float-only, which returned nullptr for uint8).
        auto* spann = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(vecIndex.get());
        if (!spann) return nullptr;
        auto wrapper = std::make_shared<SPTAG::Helper::TenantPrefixedKeyValueIO>(m_sharedDB, p_internalId);
        spann->SetSharedDB(wrapper);
    }

    auto indexfiles = vecIndex->GetIndexFiles();
    if (iniReader.DoesSectionExist("MetaData"))
    {
        indexfiles->push_back("metadata.bin");
        indexfiles->push_back("metadataIndex.bin");
    }
    if (iniReader.DoesSectionExist("Quantizer"))
    {
        indexfiles->push_back("quantizer.bin");
    }

    std::vector<std::shared_ptr<Helper::DiskIO>> handles;
    for (std::string& f : *indexfiles)
    {
        auto ptr = SPTAG::f_createIO();
        if (ptr == nullptr || !ptr->Initialize((folderPath + f).c_str(),
                                                std::ios::binary | std::ios::in))
        {
            ptr = nullptr;
        }
        handles.push_back(std::move(ptr));
    }

    if (vecIndex->LoadIndexData(handles) != ErrorCode::Success) return nullptr;

    size_t metaStart = vecIndex->GetIndexFiles()->size();
    if (iniReader.DoesSectionExist("MetaData"))
    {
        vecIndex->SetMetadata(new SPTAG::MemMetadataSet(handles[metaStart], handles[metaStart + 1],
                                                        vecIndex->m_iDataBlockSize, vecIndex->m_iDataCapacity,
                                                        vecIndex->m_iMetaRecordSize));
        if (!(vecIndex->GetMetadata()->Available())) return nullptr;
        if (iniReader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
            vecIndex->BuildMetaMapping();
        metaStart += 2;
    }
    if (iniReader.DoesSectionExist("Quantizer"))
    {
        vecIndex->SetQuantizer(SPTAG::COMMON::IQuantizer::LoadIQuantizer(handles[metaStart]));
        if (!vecIndex->m_pQuantizer) return nullptr;
    }
    vecIndex->SetReady(true);
    return std::make_shared<AnnIndex>(AnnIndex(vecIndex));
}

bool TenantIndexManager::BuildFromData(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum,
                                       bool p_withMetaIndex, bool p_normalized)
{
    if (p_vectorNum == 0 || m_dimension == 0 || p_vectors.Length() != p_vectorNum * m_inputVectorSize)
    {
        return false;
    }

    m_tenantIndices.clear();
    m_lruList.clear();
    m_lruMap.clear();
    m_tenantHeadIndexAccountedBytes.clear();
    m_loadedHeadIndexBytes = 0;
    m_tenantVectorCounts.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantTagRoutingStats.clear();
    m_tenantPivotLevels.clear();
    m_tenantPivotNodeCounts.clear();
    m_tenantNodePivotTags.clear();
    m_tenantTagToNodes.clear();
    m_tenantHeadNodeToNode.clear();

    std::map<int, std::vector<std::pair<const uint8_t*, size_t>>> tenantVectorRanges;
    std::map<int, std::vector<std::string>> tenantMetadataLines;

    const char* metaPtr = reinterpret_cast<const char*>(p_metadata.Data());
    const char* metaEnd = metaPtr + p_metadata.Length();
    const uint8_t* vectorPtr = p_vectors.Data();

    SizeType globalIdx = 0;
    while (metaPtr < metaEnd && globalIdx < p_vectorNum)
    {
        const char* lineEnd = metaPtr;
        while (lineEnd < metaEnd && *lineEnd != '\n')
        {
            lineEnd++;
        }

        if (lineEnd == metaPtr)
        {
            return false;
        }

        std::string metaLine(metaPtr, lineEnd - metaPtr);
        int tenantId = RegisterTenantId(metaLine.c_str());

        tenantVectorRanges[tenantId].push_back({vectorPtr, m_inputVectorSize});
        tenantMetadataLines[tenantId].push_back(metaLine);
        m_tenantGlobalIndices[tenantId].push_back(globalIdx);
        vectorPtr += m_inputVectorSize;
        metaPtr = (lineEnd < metaEnd) ? (lineEnd + 1) : lineEnd;
        globalIdx++;
    }

    std::string algoTypeStr = SPTAG::Helper::Convert::ConvertToString(m_algoType);
    std::string valueTypeStr = SPTAG::Helper::Convert::ConvertToString(m_valueType);

    // Distance metric for tenant index builds. The native [Base]
    // DistCalcMethod parameter is staged before tenant indexes exist, so read it
    // here as well for routing/planning performed before the core index is built.
    std::string distMethod = "Cosine";
    for (const auto& param : m_pendingBuildParams) {
        if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(std::get<2>(param).c_str(), "Base")
            && SPTAG::Helper::StrUtils::StrEqualIgnoreCase(std::get<0>(param).c_str(), "DistCalcMethod")
            && !std::get<1>(param).empty()) {
            distMethod = std::get<1>(param);
        }
    }

    for (auto& tenantEntry : tenantVectorRanges)
    {
        int tenantId = tenantEntry.first;
        std::vector<std::pair<const uint8_t*, size_t>>& vectorRanges = tenantEntry.second;
        if (vectorRanges.empty())
        {
            continue;
        }

        size_t totalVectorSize = vectorRanges.size() * m_inputVectorSize;

        // Zero-copy fast path (SPTAG_BUILD_SHARE_OWNERSHIP=1): when this tenant's
        // vectors are already a contiguous slice of the caller-provided p_vectors
        // buffer (the common single-tenant 1B-scale case where every vector maps
        // to tenant 0 in order), borrow that slice directly instead of allocating
        // and memcpy'ing a full duplicate. Combined with SetShareBuildOwnership
        // (which forwards p_shareOwnership=true to the core BuildIndex) this avoids
        // two full copies of the vector data (wrapper + core), each ~93 GB at 1B
        // int8 vectors. The borrowed p_vectors buffer (a numpy memmap passed via
        // the SWIG buffer protocol) must stay alive for the duration of the build.
        static const bool kShareOwnership = []() {
            const char* e = std::getenv("SPTAG_BUILD_SHARE_OWNERSHIP");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        bool contiguous = kShareOwnership;
        for (size_t i = 1; contiguous && i < vectorRanges.size(); ++i) {
            if (vectorRanges[i].first != vectorRanges[i - 1].first + m_inputVectorSize) {
                contiguous = false;
            }
        }

        ByteArray tenantVectors;
        bool borrowedVectors = false;
        if (contiguous) {
            // own=false: ByteArray borrows the caller's contiguous slice.
            tenantVectors = ByteArray(const_cast<uint8_t*>(vectorRanges[0].first), totalVectorSize, false);
            borrowedVectors = true;
            fprintf(stderr,
                    "[INFO] Tenant %d: zero-copy build (borrowing %zu-byte contiguous slice)\n",
                    tenantId, totalVectorSize);
        } else {
            uint8_t* tenantVectorBuffer = new uint8_t[totalVectorSize];
            uint8_t* out = tenantVectorBuffer;
            for (const auto& vec : vectorRanges)
            {
                memcpy(out, vec.first, vec.second);
                out += vec.second;
            }
            tenantVectors = ByteArray(tenantVectorBuffer, totalVectorSize, true);
        }

        std::string metaStr;
        for (size_t i = 0; i < tenantMetadataLines[tenantId].size(); ++i)
        {
            if (i > 0) metaStr.push_back('\n');
            metaStr += tenantMetadataLines[tenantId][i];
        }
        metaStr.push_back('\n');

        uint8_t* metaBuffer = new uint8_t[metaStr.size()];
        memcpy(metaBuffer, metaStr.data(), metaStr.size());
        ByteArray tenantMetadata(metaBuffer, metaStr.size(), true);

        auto tenantIndex = std::make_shared<AnnIndex>(algoTypeStr.c_str(), valueTypeStr.c_str(), m_dimension);
        bool buildOk = false;
        SizeType tenantVecCount = static_cast<SizeType>(vectorRanges.size());
        std::vector<uint32_t> tenantLocalTags;

        if (m_buildNumTagsPerVec > 0 && m_buildTags.Data() != nullptr) {
            const uint32_t* globalTags = reinterpret_cast<const uint32_t*>(m_buildTags.Data());
            auto gidIt = m_tenantGlobalIndices.find(tenantId);
            if (gidIt != m_tenantGlobalIndices.end()) {
                const auto& gids = gidIt->second;
                tenantLocalTags.resize(static_cast<size_t>(tenantVecCount) * static_cast<size_t>(m_buildNumTagsPerVec));
                for (int i = 0; i < tenantVecCount && i < static_cast<int>(gids.size()); ++i) {
                    int gid = gids[i];
                    for (int t = 0; t < m_buildNumTagsPerVec; ++t) {
                        tenantLocalTags[static_cast<size_t>(i) * static_cast<size_t>(m_buildNumTagsPerVec) + static_cast<size_t>(t)] =
                            globalTags[static_cast<size_t>(gid) * static_cast<size_t>(m_buildNumTagsPerVec) + static_cast<size_t>(t)];
                    }
                }
            }
        }

        // Choose index type based on tenant size (hybrid strategy)
        TenantIndexType indexType = ChooseIndexType(tenantVecCount);
        m_tenantIndexTypes[tenantId] = indexType;

        if (indexType == TenantIndexType::SPANN)
        {
            bool hybridDistanceEnabled = false;
            for (const auto& parameter :
                 m_extraSSDBuildParams) {
                if (!SPTAG::Helper::StrUtils::
                        StrEqualIgnoreCase(
                            parameter.first.c_str(),
                            "EnableHybridDistance")) {
                    continue;
                }
                SPTAG::Helper::Convert::
                    ConvertStringTo<bool>(
                        parameter.second.c_str(),
                        hybridDistanceEnabled);
                break;
            }
            const char* skipPivotEnv = std::getenv("SPTAG_DISABLE_PIVOT_ESTIMATOR");
            const bool skipPivot = skipPivotEnv != nullptr &&
                (skipPivotEnv[0] == '1' ||
                 SPTAG::Helper::StrUtils::StrEqualIgnoreCase(skipPivotEnv, "true") ||
                 SPTAG::Helper::StrUtils::StrEqualIgnoreCase(skipPivotEnv, "yes") ||
                 SPTAG::Helper::StrUtils::StrEqualIgnoreCase(skipPivotEnv, "on"));
            if (skipPivot) {
                fprintf(stderr, "[INFO] Tenant %d: SPTAG_DISABLE_PIVOT_ESTIMATOR set, skipping node-aware planning\n", tenantId);
            }
            if (hybridDistanceEnabled) {
                auto& globalAssignments =
                    m_tenantPlannedNodeVectors[
                        tenantId];
                globalAssignments.assign(1, {});
                globalAssignments[0].reserve(
                    static_cast<size_t>(
                        tenantVecCount));
                for (SizeType vectorId = 0;
                     vectorId < tenantVecCount;
                     ++vectorId) {
                    globalAssignments[0].push_back(
                        vectorId);
                }
                m_tenantPlannedPrimaryNodeVectors[
                    tenantId] = globalAssignments;
                m_tenantPivotLevels[tenantId] = -1;
                m_tenantPivotNodeCounts[tenantId] = 1;
                m_tenantNodePivotTags[tenantId] = {
                    std::vector<std::uint32_t>()};
                auto& tagToNodes =
                    m_tenantTagToNodes[tenantId];
                tagToNodes.clear();
                for (std::uint32_t tag :
                     tenantLocalTags) {
                    tagToNodes[tag] = {0};
                }
                fprintf(
                    stderr,
                    "[INFO] Tenant %d: hybrid distance uses one global "
                    "head/posting node over all %d vectors; attribute pivot "
                    "partitioning is disabled\n",
                    tenantId, tenantVecCount);
            } else if (!skipPivot && !tenantLocalTags.empty()) {
                // Routing/bundle planning must consider only the CATEGORICAL tag
                // columns. Numeric attributes are inlined as the last
                // SPTAG_NUMERIC_COLS columns (raw, high-cardinality values); if
                // fed to the hierarchy pivot estimator they masquerade as deep
                // hierarchy levels with ~unique values, collapsing the plan to a
                // single bundle. Build a categorical-only view (first
                // numBaseCols columns) for all three planning calls. The actual
                // SPANN build below still uses the full m_buildNumTagsPerVec tags.
                int numNumericCols = 0;
                if (const char* e = std::getenv("SPTAG_NUMERIC_COLS")) numNumericCols = atoi(e);
                if (numNumericCols < 0) numNumericCols = 0;
                if (numNumericCols > m_buildNumTagsPerVec) numNumericCols = m_buildNumTagsPerVec;
                const int numBaseCols = m_buildNumTagsPerVec - numNumericCols;

                // Categorical columns that drive routing/bundle planning. By
                // default all categorical columns participate (the legacy ACL
                // containment-chain assumption). When the categorical columns are
                // INDEPENDENT facets that do not form a containment chain (e.g.
                // YFCC year/month/camera), feeding >1 of them to the pivot
                // estimator makes it either collapse the plan or fail the
                // unique-parent check. Two knobs select the routing columns; the
                // remaining categorical columns stay filter-only signatures (hier
                // mask + exact DNF), like numeric columns are kept out of routing:
                //   * SPTAG_ACL_COLS=c0,c1,..  -> explicit, possibly NON-CONTIGUOUS
                //     column list, projected in the given order (level i = col[i]).
                //     Required when the ACL chain is not a leading prefix, e.g.
                //     YFCC country(col0) -> us_state(col4) => SPTAG_ACL_COLS=0,4.
                //   * SPTAG_ROUTING_COLS=K     -> leading-K columns (legacy). Used
                //     only when SPTAG_ACL_COLS is unset.
                // Column indices are clamped to [0, numBaseCols); out-of-range or
                // duplicate entries are dropped.
                std::vector<int> routingCols;
                if (const char* e = std::getenv("SPTAG_ACL_COLS")) {
                    const char* p = e;
                    while (*p != '\0') {
                        int c = atoi(p);
                        if (c >= 0 && c < numBaseCols) {
                            bool dup = false;
                            for (int existing : routingCols) if (existing == c) { dup = true; break; }
                            if (!dup) routingCols.push_back(c);
                        }
                        const char* comma = strchr(p, ',');
                        if (comma == nullptr) break;
                        p = comma + 1;
                    }
                }
                if (routingCols.empty()) {
                    int numRoutingCols = numBaseCols;
                    if (const char* e = std::getenv("SPTAG_ROUTING_COLS")) {
                        int k = atoi(e);
                        if (k > 0) numRoutingCols = (k < numBaseCols) ? k : numBaseCols;
                    }
                    for (int t = 0; t < numRoutingCols; ++t) routingCols.push_back(t);
                }

                const uint32_t* planTags = tenantLocalTags.data();
                int planNumTags = m_buildNumTagsPerVec;
                std::vector<uint32_t> catOnlyTags;
                const int numRoutingCols = static_cast<int>(routingCols.size());
                bool identityProjection = (numRoutingCols == m_buildNumTagsPerVec);
                if (identityProjection)
                    for (int t = 0; t < numRoutingCols; ++t)
                        if (routingCols[t] != t) { identityProjection = false; break; }
                if (numRoutingCols > 0 && !identityProjection) {
                    catOnlyTags.resize(static_cast<size_t>(tenantVecCount) * static_cast<size_t>(numRoutingCols));
                    for (int i = 0; i < tenantVecCount; ++i)
                        for (int t = 0; t < numRoutingCols; ++t)
                            catOnlyTags[static_cast<size_t>(i) * numRoutingCols + t] =
                                tenantLocalTags[static_cast<size_t>(i) * m_buildNumTagsPerVec + routingCols[t]];
                    planTags = catOnlyTags.data();
                    planNumTags = numRoutingCols;
                    std::string colList;
                    for (size_t t = 0; t < routingCols.size(); ++t)
                        colList += (t ? "," : "") + std::to_string(routingCols[t]);
                    fprintf(stderr,
                            "[INFO] Tenant %d: routing-node planning on %d categorical cols [%s] "
                            "(of %d base, %d numeric)\n",
                            tenantId, numRoutingCols, colList.c_str(), numBaseCols, numNumericCols);
                }

                PivotEstimatorComputation pivotComputation;
                const PivotEstimatorCandidate* pivotCandidate = nullptr;
                if (BuildPivotEstimatorComputation(planTags,
                                                   static_cast<int>(tenantVecCount),
                                                   planNumTags,
                                                   0,
                                                   0.99,
                                                   10.0,
                                                   1.0,
                                                   std::string(),
                                                   pivotComputation)) {
                    pivotCandidate = FindBestPivotEstimatorCandidate(pivotComputation.candidates);
                }

                if (pivotCandidate != nullptr) {
                    m_tenantPivotLevels[tenantId] = pivotCandidate->pivotLevel;
                    m_tenantPivotNodeCounts[tenantId] = pivotCandidate->nodeCount;
                    m_tenantNodePivotTags[tenantId] = pivotCandidate->nodePivotTags;
                    BuildTagToNodeIndexForCandidate(*pivotCandidate,
                                                    pivotComputation.levelData,
                                                    m_tenantTagToNodes[tenantId]);
                    BuildPrimaryNodeVectorAssignmentsForCandidate(*pivotCandidate,
                                                                  planTags,
                                                                  static_cast<int>(tenantVecCount),
                                                                  planNumTags,
                                                                  m_tenantPlannedPrimaryNodeVectors[tenantId]);

                    // Keep tree-structured tag-to-node merges for query routing, but
                    // partition postings by the pivot-layer owner only so vectors are
                    // evenly distributed across nodes instead of being replicated by
                    // higher-level ancestor tags.
                    m_tenantPlannedNodeVectors[tenantId] = m_tenantPlannedPrimaryNodeVectors[tenantId];

                    fprintf(stderr,
                            "[INFO] Tenant %d: planned %d routing nodes before SPANN build\n",
                            tenantId,
                            pivotCandidate->nodeCount);
                }
            }

            auto planIt = m_tenantPlannedNodeVectors.find(tenantId);
            auto primaryPlanIt = m_tenantPlannedPrimaryNodeVectors.find(tenantId);
            bool hasNodeAwarePlan = (planIt != m_tenantPlannedNodeVectors.end() && !planIt->second.empty());

            int64_t postingAssignmentCount = static_cast<int64_t>(tenantVecCount);
            if (hasNodeAwarePlan) {
                int64_t plannedAssignmentTotal = 0;
                for (const auto& nodeAssignments : planIt->second) {
                    plannedAssignmentTotal += static_cast<int64_t>(nodeAssignments.size());
                }

                if (plannedAssignmentTotal > 0) {
                    postingAssignmentCount = plannedAssignmentTotal;
                }
            }

            // SPANN scratch/work directory (head index + postings written here
            // during build, then SaveAll copies to the final index dir). Defaults
            // to /tmp; override with SPTAG_SPANN_WORK_DIR to place it on a fast
            // disk with enough space for billion-scale postings.
            //
            // IN-PLACE build: when SPTAG_SPANN_INPLACE_DIR is set, build directly
            // into the final per-tenant dir "<inplace>/tenant_<id>" (which must equal
            // the dir SaveAll/SaveUnifiedStorage writes to). The SSD block pool is
            // pre-allocated and incrementally flushed there during BuildSSDIndex, and
            // SaveAll's srcDir==dstDir check skips the final copy entirely. This avoids
            // the transient 2x disk footprint (work dir + final dir) and the copy time
            // — essential at billion scale. Set SPTAG_SPANN_INPLACE_DIR to the same
            // path passed to SaveAll (i.e. the ini IndexDirectory).
            std::string spannWorkDir;
            bool inPlaceBuild = false;
            if (const char* ip = std::getenv("SPTAG_SPANN_INPLACE_DIR")) {
                if (ip[0] != '\0') {
                    inPlaceBuild = true;
                    spannWorkDir = std::string(ip) + "/tenant_" + std::to_string(tenantId);
                }
            }
            if (!inPlaceBuild) {
                std::string spannWorkBase = "/tmp";
                if (const char* e = std::getenv("SPTAG_SPANN_WORK_DIR")) {
                    if (e[0] != '\0') spannWorkBase = e;
                }
                spannWorkDir = spannWorkBase + "/sptag_spann_tenant_" + std::to_string(tenantId);
            }
            // Resume: keep the existing work dir (head_select_state.bin + per-node head
            // files from the prior run) so BuildIndexInternal can skip the BKT head
            // selection. Wiping it would force a full SelectHead re-run, defeating the
            // SPTAG_PERSIST_SELECTHEAD checkpoint.
            const bool resumeBuild = []() {
                const char* v = std::getenv("SPTAG_RESUME_BUILD");
                return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
            }();
            const std::string checkpointFile = spannWorkDir + "/head_select_state.bin";
            const bool checkpointExists = std::ifstream(checkpointFile, std::ios::binary).good();
            if (resumeBuild && checkpointExists) {
                fprintf(stderr, "[INFO] Tenant %d: RESUME — keeping work dir %s (checkpoint present, skipping wipe)\n",
                        tenantId, spannWorkDir.c_str());
            } else {
                if (resumeBuild) {
                    fprintf(stderr, "[INFO] Tenant %d: RESUME requested but no checkpoint at %s; full rebuild\n",
                            tenantId, checkpointFile.c_str());
                }
                RemovePathRecursive(spannWorkDir);
            }
            EnsureDir(spannWorkDir);
            if (inPlaceBuild) {
                fprintf(stderr, "[INFO] Tenant %d: IN-PLACE build — writing directly to final dir %s (SaveAll will skip the copy)\n",
                        tenantId, spannWorkDir.c_str());
            }
            tenantIndex->SetBuildParam("IndexDirectory", spannWorkDir.c_str(), "Base");
            tenantIndex->SetBuildParam("DistCalcMethod", distMethod.c_str(), "Base");
            tenantIndex->SetBuildParam("isExecute", "true", "SelectHead");
            tenantIndex->SetBuildParam("isExecute", "true", "BuildHead");
            tenantIndex->SetBuildParam("isExecute", "true", "BuildSSDIndex");
            tenantIndex->SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex");
            // SelectHead defaults to the machine width. Explicit native
            // [SelectHead] values are staged below and take precedence.
            {
                int selThreads = (int)std::thread::hardware_concurrency();
                if (selThreads <= 0) selThreads = 16;
                tenantIndex->SetBuildParam("NumberOfThreads", std::to_string(selThreads).c_str(), "SelectHead");
            }
            tenantIndex->SetBuildParam("Storage", m_storageBackend.c_str(), "BuildSSDIndex");

            // Apply staged in-posting quantization config (and any other extra
            // [BuildSSDIndex] params) so they persist into the index's indexloader.ini.
            for (const auto& kv : m_extraSSDBuildParams) {
                tenantIndex->SetBuildParam(kv.first.c_str(), kv.second.c_str(), "BuildSSDIndex");
                fprintf(stderr, "[INFO] Tenant %d: SSD build param %s = %s\n",
                        tenantId, kv.first.c_str(), kv.second.c_str());
            }

            // Scale DataCapacity and SSD file size to tenant size
            // Block pool uses 4KB pages; each vector with replication needs multiple blocks
            // Use the routed posting assignment count instead of raw tenant size,
            // because node-aware builds can replicate vectors across routing nodes.
            int64_t dataCapacity64 = std::max<int64_t>(postingAssignmentCount * 8LL, 4096LL);
            int dataCapacity = static_cast<int>(std::min<int64_t>(dataCapacity64, std::numeric_limits<int>::max()));
            tenantIndex->SetBuildParam("DataCapacity", std::to_string(dataCapacity).c_str(), "Base");
            tenantIndex->SetBuildParam("DataBlockSize", std::to_string(std::min(dataCapacity, 1024 * 1024)).c_str(), "Base");

            // Scale SSD file size: each posting record stores VID + version +
            // per-vector tags + the vector payload. Vector bytes depend on the value
            // type (Int8 = dim*1, Float = dim*4); using a fixed dim*4 here would
            // over-preallocate the block pool 4x for Int8 / UInt8 datasets (e.g.
            // billion-scale SIFT/SPACEV) and can exhaust the disk before build.
            //
            // IMPORTANT: when in-posting quantization (OPQ/RaBitQ) is enabled, the
            // persisted posting record is the SLIM record (quant code + small meta),
            // NOT the full vector. Estimating with the full vector over-allocates the
            // block pool by ~(fullVec/slim) x (e.g. 100B vs ~50B for SPACEV M=25),
            // which at billion scale pre-allocates StartFileSizeGB far past the disk
            // and ENOSPC-fails the build before a single posting is written. So when a
            // posting quantizer is staged, size the pool off the slim record instead.
            const int64_t valueTypeBytes = static_cast<int64_t>(SPTAG::GetValueTypeSize(m_valueType));
            int64_t postingQuantM = 0;
            bool postingQuantized = false;
            // If the caller explicitly pinned StartFileSizeGB / MaxFileSizeGB (e.g. via
            // the spannbuilder --ssd-start-file-gb / --ssd-max-file-gb CLI flags, which
            // stage them into m_extraSSDBuildParams), honor those exactly and DO NOT
            // overwrite them with the auto estimate below. This keeps billion-scale
            // disk budgeting explicit and reproducible from the build script.
            bool explicitStart = false, explicitMax = false;
            // NOTE: keys staged via the native .ini path are lowercased by IniReader
            // (e.g. "startfilesizegb"), so these comparisons MUST be case-insensitive
            // — a case-sensitive match silently fails to detect the explicit pin and
            // lets the auto estimate clobber the reproducible disk budget.
            for (const auto& kv : m_extraSSDBuildParams) {
                if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(kv.first.c_str(), "PostingQuantizer") && kv.second != "None" && !kv.second.empty()) {
                    postingQuantized = true;
                } else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(kv.first.c_str(), "PostingQuantM")) {
                    postingQuantM = std::max<int64_t>(0, atoll(kv.second.c_str()));
                } else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(kv.first.c_str(), "StartFileSizeGB")) {
                    explicitStart = true;
                } else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(kv.first.c_str(), "MaxFileSizeGB")) {
                    explicitMax = true;
                }
            }
            // Slim record = quant code (M bytes) + meta (VID/version/tags/numeric-quant
            // words, ~32B headroom). Full record = dim*valueTypeBytes + tags + 64.
            const int64_t perVecBytes = (postingQuantized && postingQuantM > 0)
                ? (postingQuantM + static_cast<int64_t>(m_buildNumTagsPerVec) * 4 + 32)
                : (static_cast<int64_t>(m_dimension) * valueTypeBytes
                   + static_cast<int64_t>(m_buildNumTagsPerVec) * 4 + 64);
            int64_t estimatedBytes = postingAssignmentCount * perVecBytes * 10LL;
                int startFileSizeGB = std::max(1, (int)(estimatedBytes / (1024LL * 1024LL * 1024LL)) + 1);
                if (hasNodeAwarePlan) {
                startFileSizeGB = std::max(startFileSizeGB, 4);
                }
                int maxFileSizeGB = std::max(startFileSizeGB * 3, hasNodeAwarePlan ? 32 : 10);
            if (!explicitStart) {
                tenantIndex->SetBuildParam("StartFileSizeGB", std::to_string(startFileSizeGB).c_str(), "BuildSSDIndex");
            }
            if (!explicitMax) {
                tenantIndex->SetBuildParam("MaxFileSizeGB", std::to_string(maxFileSizeGB).c_str(), "BuildSSDIndex");
            }

            fprintf(stderr,
                    "[INFO] Tenant %d: posting assignments=%lld raw_vectors=%d StartFileSizeGB=%s MaxFileSizeGB=%s DataCapacity=%d\n",
                    tenantId,
                    static_cast<long long>(postingAssignmentCount),
                    tenantVecCount,
                    explicitStart ? "<explicit>" : std::to_string(startFileSizeGB).c_str(),
                    explicitMax ? "<explicit>" : std::to_string(maxFileSizeGB).c_str(),
                    dataCapacity);

            // Scale graph build defaults by tenant size to avoid fixed overhead on
            // small tenants. Explicit native [BuildHead] values are applied below
            // and therefore override these defaults.
            int tptNumber = 32;
            int refineIter = 2;
            if (tenantVecCount < 10000) {
                tptNumber = 8;
                refineIter = 2;
            } else if (tenantVecCount < 50000) {
                tptNumber = 8;
                refineIter = 2;
            } else if (tenantVecCount < 200000) {
                tptNumber = 16;
                refineIter = 2;
            } else if (tenantVecCount < 500000) {
                tptNumber = 16;
                refineIter = 2;
            }
            tenantIndex->SetBuildParam("TPTNumber", std::to_string(tptNumber).c_str(), "BuildHead");
            tenantIndex->SetBuildParam("RefineIterations", std::to_string(refineIter).c_str(), "BuildHead");
            // Head index (BKT) defaults to the machine width. Explicit native
            // [BuildHead] NumberOfThreads overrides this below.
            {
                int headThreads = (int)std::thread::hardware_concurrency();
                if (headThreads <= 0) headThreads = 16;
                tenantIndex->SetBuildParam("NumberOfThreads", std::to_string(headThreads).c_str(), "BuildHead");
            }

            // Native [Base] head-algorithm, [SelectHead], and [BuildHead]
            // settings are authoritative. Apply them after automatic defaults so
            // explicit INI values such as IndexAlgoType=BKT,
            // RefineIterations=3, and MaxCheckForRefineGraph=16324 are not
            // silently replaced by wrapper heuristics.
            for (const auto& param : m_pendingBuildParams) {
                const std::string& name = std::get<0>(param);
                const std::string& value = std::get<1>(param);
                const std::string& section = std::get<2>(param);
                const bool isHeadSection =
                    SPTAG::Helper::StrUtils::StrEqualIgnoreCase(section.c_str(), "SelectHead")
                    || SPTAG::Helper::StrUtils::StrEqualIgnoreCase(section.c_str(), "BuildHead");
                const bool isHeadAlgorithm =
                    SPTAG::Helper::StrUtils::StrEqualIgnoreCase(section.c_str(), "Base")
                    && SPTAG::Helper::StrUtils::StrEqualIgnoreCase(name.c_str(), "IndexAlgoType");
                if (!isHeadSection && !isHeadAlgorithm) {
                    continue;
                }
                tenantIndex->SetBuildParam(name.c_str(), value.c_str(), section.c_str());
                fprintf(stderr, "[INFO] Tenant %d: native [%s] %s = %s\n",
                        tenantId, section.c_str(), name.c_str(), value.c_str());
            }

            // Set per-vector tags to embed in posting metadata (if available from BuildFromDataWithTags)
            if (!tenantLocalTags.empty()) {
                tenantIndex->SetBuildParam("NumTagsPerVec", std::to_string(m_buildNumTagsPerVec).c_str(), "BuildSSDIndex");
                tenantIndex->SetVectorTags(tenantLocalTags.data(), tenantVecCount, m_buildNumTagsPerVec);
            }
            if (hasNodeAwarePlan) {
                tenantIndex->SetNodeVectorAssignments(planIt->second);
            }
            if (primaryPlanIt != m_tenantPlannedPrimaryNodeVectors.end() && !primaryPlanIt->second.empty()) {
                tenantIndex->SetPrimaryNodeVectorAssignments(primaryPlanIt->second);
            }

            // Shared RocksDB: when enabled, inject a tenant-prefixed wrapper
            // BEFORE Build so SPANN's ExtraDynamicSearcher reuses the shared
            // store instead of opening a per-tenant RocksDB.
            if (m_storageBackend == "ROCKSDBIO" && m_useSharedDB)
            {
                tenantIndex->SetBuildParam("ShareDB", "true", "BuildSSDIndex");
                if (!EnsureSharedDB()) return false;
                if (!InjectSharedDB(tenantIndex, tenantId)) return false;
            }

            tenantIndex->SetShareBuildOwnership(borrowedVectors);
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            m_tenantSpannWorkDirs[tenantId] = spannWorkDir;
            fprintf(stderr, "[INFO] Tenant %d: SPANN build (%d vectors)\n", tenantId, tenantVecCount);
        }
        else if (indexType == TenantIndexType::BKT)
        {
            // Medium tenant: build in-memory BKT index
            tenantIndex = std::make_shared<AnnIndex>("BKT", valueTypeStr.c_str(), m_dimension);
            tenantIndex->SetBuildParam("DistCalcMethod", distMethod.c_str(), "Index");
            tenantIndex->SetShareBuildOwnership(borrowedVectors);
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            fprintf(stderr, "[INFO] Tenant %d: BKT build (%d vectors)\n", tenantId, tenantVecCount);
        }
        else // BRUTEFORCE
        {
            // Small tenant: build trivial BKT index (effectively brute force at this scale)
            tenantIndex = std::make_shared<AnnIndex>("BKT", valueTypeStr.c_str(), m_dimension);
            tenantIndex->SetBuildParam("DistCalcMethod", distMethod.c_str(), "Index");
            tenantIndex->SetShareBuildOwnership(borrowedVectors);
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            fprintf(stderr, "[INFO] Tenant %d: BruteForce build (%d vectors)\n", tenantId, tenantVecCount);
        }

        if (!buildOk)
        {
            return false;
        }

        // A native runtime overlay must not influence construction, but it must
        // be applied before this SPANN tenant is saved-and-released below.
        if (indexType == TenantIndexType::SPANN)
        {
            for (const auto& pendingParam : m_pendingSearchParams)
            {
                const std::string& name = std::get<0>(pendingParam);
                const std::string& value = std::get<1>(pendingParam);
                const std::string& section = std::get<2>(pendingParam);
                tenantIndex->SetSearchParam(name.c_str(), value.c_str(), section.c_str());
                fprintf(stderr, "[INFO] Tenant %d: native [%s] %s = %s\n",
                        tenantId, section.c_str(), name.c_str(), value.c_str());
            }
        }

        m_tenantVectorCounts[tenantId] = static_cast<int>(vectorRanges.size());

        // For SPANN: save the index to its work dir right away, then release the
        // AnnIndex object.  This closes the SSD file descriptor and frees the
        // HeadIndex memory, preventing fd exhaustion when building many tenants.
        if (indexType == TenantIndexType::SPANN)
        {
            std::string workDir = m_tenantSpannWorkDirs[tenantId];
            tenantIndex->Save(workDir.c_str());
            fprintf(stderr, "[INFO] Tenant %d: built & released (%d vectors, dir=%s)\n",
                tenantId, (int)vectorRanges.size(), workDir.c_str());
            tenantIndex.reset();
            continue;
        }

        m_tenantIndices[tenantId] = tenantIndex;
    }

    // For SPANN tenants: compute posting offsets and record head counts
    if (m_algoType == SPTAG::IndexAlgoType::SPANN)
    {
        m_tenantPostingOffsets.clear();
        m_tenantHeadCounts.clear();
        m_totalPostingCount = 0;

        // Iterate all tenants (not just loaded ones — released tenants have work dirs)
        for (const auto& kv : m_tenantVectorCounts)
        {
            int tenantId = kv.first;
            auto typeIt = m_tenantIndexTypes.find(tenantId);
            if (typeIt == m_tenantIndexTypes.end() || typeIt->second != TenantIndexType::SPANN)
            {
                // Non-SPANN tenants don't have postings in the shared SSD
                m_tenantPostingOffsets[tenantId] = -1;
                m_tenantHeadCounts[tenantId] = 0;
                continue;
            }
            // SPTAGHeadVectorIDs.bin is a Dataset file with an 8-byte
            // [rowCount, dimension] header followed by uint64 IDs. Counting
            // file bytes directly includes that header and overstates the head
            // count by one at billion scale.
            std::string headIDFile = m_tenantSpannWorkDirs[tenantId] + "/SPTAGHeadVectorIDs.bin";
            int headCount = 0;

            if (fileexists(headIDFile.c_str()))
            {
                FILE* idFile = fopen(headIDFile.c_str(), "rb");
                int32_t rows = 0, cols = 0;
                if (idFile != nullptr &&
                    fread(&rows, sizeof(rows), 1, idFile) == 1 &&
                    fread(&cols, sizeof(cols), 1, idFile) == 1 &&
                    rows > 0 && cols == 1) {
                    headCount = rows;
                }
                if (idFile != nullptr) fclose(idFile);
            }

            if (headCount <= 0)
            {
                // Fallback: read the [rowCount, dimension] header from the
                // selected head vector Dataset.
                std::string headVecFile = m_tenantSpannWorkDirs[tenantId] + "/SPTAGHeadVectors.bin";
                if (fileexists(headVecFile.c_str()))
                {
                    FILE* vecFile = fopen(headVecFile.c_str(), "rb");
                    int32_t rows = 0, cols = 0;
                    if (vecFile != nullptr &&
                        fread(&rows, sizeof(rows), 1, vecFile) == 1 &&
                        fread(&cols, sizeof(cols), 1, vecFile) == 1 &&
                        rows > 0 && cols == m_dimension) {
                        headCount = rows;
                    }
                    if (vecFile != nullptr) fclose(vecFile);
                }
            }

            if (headCount <= 0)
            {
                fprintf(stderr, "[ERROR] Cannot determine head count for tenant %d\n", tenantId);
                return false;
            }

            m_tenantPostingOffsets[tenantId] = m_totalPostingCount;
            m_tenantHeadCounts[tenantId] = headCount;
            m_totalPostingCount += headCount;
            fprintf(stderr, "[INFO] Tenant %d: headCount=%d, postingOffset=%d\n",
                tenantId, headCount, m_tenantPostingOffsets[tenantId]);
        }

        fprintf(stderr, "[INFO] Total posting count across %d tenants: %d\n",
            (int)m_tenantVectorCounts.size(), m_totalPostingCount);
    }

    return !m_tenantVectorCounts.empty();
}

bool TenantIndexManager::BuildFromDataWithTags(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum,
                                                ByteArray p_tags, int p_numTagsPerVec,
                                                bool p_withMetaIndex, bool p_normalized)
{
    // Store tags and numTagsPerVec for the build process.
    // BuildFromData will be modified to pass tags to each SPANN index.
    m_buildTags = p_tags;
    m_buildNumTagsPerVec = p_numTagsPerVec;

    // Build SPANN indexes — tags will be embedded in postings via SetVectorTags
    if (!BuildFromData(p_vectors, p_metadata, p_vectorNum, p_withMetaIndex, p_normalized))
        return false;

    m_buildTags = ByteArray();  // release reference
    m_buildNumTagsPerVec = 0;

    // BuildSignatures is intentionally explicit. Reconstructing tenant-local
    // tag buffers here duplicates tens of gigabytes for a single billion-vector
    // tenant, and callers such as spannbuilder already invoke BuildSignatures
    // exactly once with their zero-copy tag view.
    fprintf(stderr, "[INFO] BuildFromDataWithTags: tags embedded in postings; signatures pending explicit build\n");
    return true;
}

std::shared_ptr<QueryResult> TenantIndexManager::Search(ByteArray p_queryVector, int p_tenantId, int p_resultNum)
{
    if (!EnsureTenantLoaded(p_tenantId))
    {
        return nullptr;
    }

    // Get index under shared lock (concurrent reads safe)
    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;  // shared_ptr copy under lock, search outside lock
    }

    return indexPtr->Search(p_queryVector, p_resultNum);
}

std::shared_ptr<QueryResult> TenantIndexManager::BatchSearch(ByteArray p_queryVectors, int p_vectorNum,
                                                              int p_tenantId, int p_resultNum)
{
    if (!EnsureTenantLoaded(p_tenantId))
    {
        return nullptr;
    }

    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;
    }
    return indexPtr->BatchSearch(p_queryVectors, p_vectorNum, p_resultNum, false);
}

std::shared_ptr<QueryResult> TenantIndexManager::MultiBatchSearch(
    ByteArray p_queryVectors, int p_vectorNum, ByteArray p_tenantIds, int p_resultNum)
{
    const int32_t* tenantIds = reinterpret_cast<const int32_t*>(p_tenantIds.Data());
    const uint8_t* vectors = p_queryVectors.Data();
    size_t vecSize = m_inputVectorSize;

    // Group queries by tenant: tenant_id → [(original_index, vector_ptr)]
    // Using ordered map ensures deterministic tenant processing order
    std::map<int, std::vector<std::pair<int, const uint8_t*>>> groups;
    for (int i = 0; i < p_vectorNum; i++)
    {
        groups[tenantIds[i]].emplace_back(i, vectors + i * vecSize);
    }

    // OPTIMIZATION 1: Sort tenants by batch count (most queries first)
    // and group same-tenant queries together for better cache locality.
    std::vector<std::pair<int, int>> tenantOrder;  // (tenant_id, query_count)
    for (auto& [tid, qs] : groups)
        tenantOrder.emplace_back(tid, (int)qs.size());
    std::sort(tenantOrder.begin(), tenantOrder.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Allocate output: p_vectorNum × p_resultNum results
    auto output = std::make_shared<QueryResult>(nullptr, p_vectorNum * p_resultNum, false);
    BasicResult* outResults = output->GetResults();

    for (int i = 0; i < p_vectorNum * p_resultNum; i++)
    {
        outResults[i].VID = -1;
        outResults[i].Dist = SPTAG::MaxDist;
    }

    // Pre-load each tenant and IMMEDIATELY grab shared_ptr to prevent
    // subsequent loads from evicting it.  This lets the cache temporarily
    // exceed the limit for one batch; excess is reclaimed on next batch.
    std::map<int, std::shared_ptr<AnnIndex>> heldIndices;
    for (auto& [tid, _] : tenantOrder)
    {
        EnsureTenantLoaded(tid);
        // Immediately pin so the next EnsureTenantLoaded won't evict this one
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(tid);
        if (it != m_tenantIndices.end())
            heldIndices[tid] = it->second;  // use_count > 1 → eviction-proof
    }

    // Dispatch BatchSearch per tenant in parallel
    std::vector<std::thread> threads;

    for (auto& [tid, queryList] : groups)
    {
        auto idxIt = heldIndices.find(tid);
        if (idxIt == heldIndices.end()) continue;
        auto indexPtr = idxIt->second;  // shared_ptr copy → ref count held during search

        threads.emplace_back([&, tid, indexPtr]() {
            int n = (int)queryList.size();
            if (n == 0) return;

            std::vector<uint8_t> buf(n * vecSize);
            for (int i = 0; i < n; i++)
            {
                memcpy(buf.data() + i * vecSize, queryList[i].second, vecSize);
            }

            ByteArray batchData(buf.data(), buf.size(), false);
            auto batchResult = indexPtr->BatchSearch(batchData, n, p_resultNum, false);
            if (!batchResult) return;

            BasicResult* batchRes = batchResult->GetResults();
            for (int i = 0; i < n; i++)
            {
                int origIdx = queryList[i].first;
                memcpy(outResults + origIdx * p_resultNum,
                       batchRes + i * p_resultNum,
                       p_resultNum * sizeof(BasicResult));
            }
        });
    }

    for (auto& t : threads) t.join();

    // Release held indices after all searches complete
    heldIndices.clear();

    return output;
}

void TenantIndexManager::GetTenantIds(int* p_tenants, int* p_count) const
{
    int idx = 0;
    for (const auto& [tenantId, _] : m_tenantVectorCounts)
    {
        p_tenants[idx++] = tenantId;
    }
    *p_count = (int)m_tenantVectorCounts.size();
}

int TenantIndexManager::GetTenantCount() const
{
    return (int)m_tenantVectorCounts.size();
}

int TenantIndexManager::GetTenantVectorCount(int p_tenantId) const
{
    auto it = m_tenantVectorCounts.find(p_tenantId);
    if (it != m_tenantVectorCounts.end())
    {
        return it->second;
    }
    return 0;  // Tenant not found
}

uint64_t TenantIndexManager::GetTenantHeadIndexSize(int p_tenantId) const
{
    auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (workIt == m_tenantSpannWorkDirs.end()) {
        return 0;
    }

    return GetPathSizeBytes(workIt->second + "/HeadIndex");
}

uint64_t TenantIndexManager::EstimateTenantHeadIndexBytes(int p_tenantId) const
{
    uint64_t onDiskBytes = GetTenantHeadIndexSize(p_tenantId);
    if (onDiskBytes > 0) {
        return static_cast<uint64_t>(std::ceil(static_cast<double>(onDiskBytes) * m_headIndexCacheSafetyFactor));
    }

    auto vcIt = m_tenantVectorCounts.find(p_tenantId);
    if (vcIt != m_tenantVectorCounts.end()) {
        return static_cast<uint64_t>(vcIt->second) * 128ULL;
    }

    return 1024ULL * 1024ULL;
}

ByteArray TenantIndexManager::GetTagRoutingStatsBlob(int p_tenantId) const
{
    auto routeIt = m_tenantTagRoutingStats.find(p_tenantId);
    if (routeIt == m_tenantTagRoutingStats.end() || routeIt->second.empty()) {
        return ByteArray();
    }

    std::unordered_map<std::uint32_t, TagRoutingStats>
        aggregate;
    aggregate.reserve(routeIt->second.size());
    bool hasExactLegacyStats = false;
    const auto countIt =
        m_tenantVectorCounts.find(p_tenantId);
    const std::int64_t vectorLimit =
        countIt == m_tenantVectorCounts.end()
            ? (std::numeric_limits<std::int32_t>::max)()
            : countIt->second;
    for (const auto& entry : routeIt->second) {
        if (static_cast<std::uint32_t>(
                entry.first >> 32) ==
            kLegacyRoutingColumn) {
            aggregate[
                static_cast<std::uint32_t>(
                    entry.first)] =
                entry.second;
            hasExactLegacyStats = true;
        }
    }
    if (!hasExactLegacyStats) {
        aggregate.clear();
        for (const auto& entry : routeIt->second) {
            auto& stats = aggregate[
                static_cast<std::uint32_t>(
                    entry.first)];
            stats.vectorCount = static_cast<int>(
                (std::min)(
                    vectorLimit,
                    static_cast<std::int64_t>(
                        stats.vectorCount) +
                        entry.second.vectorCount));
            stats.postingCount = static_cast<int>(
                (std::min)(
                    static_cast<std::int64_t>(
                        (std::numeric_limits<
                             std::int32_t>::max)()),
                    static_cast<std::int64_t>(
                        stats.postingCount) +
                        entry.second.postingCount));
        }
    }

    std::vector<LegacyTagRoutingStatRecord> entries;
    entries.reserve(aggregate.size());
    for (const auto& entry : aggregate) {
        entries.push_back({
            entry.first,
            static_cast<std::int32_t>(
                entry.second.vectorCount),
            static_cast<std::int32_t>(
                entry.second.postingCount)});
    }
    std::sort(
        entries.begin(), entries.end(),
        [](const LegacyTagRoutingStatRecord& left,
           const LegacyTagRoutingStatRecord& right) {
            return left.tag < right.tag;
        });

    ByteArray payload = ByteArray::Alloc(
        entries.size() *
        sizeof(LegacyTagRoutingStatRecord));
    std::memcpy(
        payload.Data(), entries.data(),
        payload.Length());
    return payload;
}

ByteArray TenantIndexManager::
    GetColumnAwareTagRoutingStatsBlob(
        int p_tenantId) const
{
    auto routeIt = m_tenantTagRoutingStats.find(p_tenantId);
    if (routeIt == m_tenantTagRoutingStats.end() || routeIt->second.empty()) {
        return ByteArray();
    }

    std::vector<TagRoutingStatRecord> entries;
    entries.reserve(routeIt->second.size());
    for (const auto& [key, stats] : routeIt->second) {
        if (static_cast<std::uint32_t>(
                key >> 32) ==
            kLegacyRoutingColumn) {
            continue;
        }
        entries.push_back(TagRoutingStatRecord{
            static_cast<std::uint32_t>(
                key >> 32),
            static_cast<std::uint32_t>(key),
            static_cast<int32_t>(stats.vectorCount),
            static_cast<int32_t>(stats.postingCount),
        });
    }

    std::sort(entries.begin(), entries.end(), [](const TagRoutingStatRecord& left, const TagRoutingStatRecord& right) {
        return left.column == right.column
            ? left.tag < right.tag
            : left.column < right.column;
    });

    ByteArray payload = ByteArray::Alloc(entries.size() * sizeof(TagRoutingStatRecord));
    std::memcpy(payload.Data(), entries.data(), payload.Length());
    return payload;
}

ByteArray TenantIndexManager::EstimatePivotBuildPlan(ByteArray p_tags,
                                                     int p_numVectors,
                                                     int p_numTagsPerVec,
                                                     int p_maxNodes,
                                                     float p_recallTarget,
                                                     float p_lambdaRecall,
                                                     float p_estimatedRecall,
                                                     ByteArray p_levelWeightsCsv) const
{
    if (p_numVectors <= 0 || p_numTagsPerVec <= 0 || p_tags.Data() == nullptr) {
        return ByteArray();
    }

    size_t expectedBytes = static_cast<size_t>(p_numVectors) * static_cast<size_t>(p_numTagsPerVec) * sizeof(uint32_t);
    if (p_tags.Length() < expectedBytes) {
        return ByteArray();
    }

    std::string weightsCsv;
    if (p_levelWeightsCsv.Data() != nullptr && p_levelWeightsCsv.Length() > 0) {
        weightsCsv.assign(reinterpret_cast<const char*>(p_levelWeightsCsv.Data()), p_levelWeightsCsv.Length());
    }

    PivotEstimatorComputation computation;
    if (!BuildPivotEstimatorComputation(reinterpret_cast<const uint32_t*>(p_tags.Data()),
                                        p_numVectors,
                                        p_numTagsPerVec,
                                        p_maxNodes,
                                        std::clamp(static_cast<double>(p_recallTarget), 0.01, 1.0),
                                        std::max(0.0, static_cast<double>(p_lambdaRecall)),
                                        std::clamp(static_cast<double>(p_estimatedRecall), 0.0, 1.0),
                                        weightsCsv,
                                        computation)) {
        return ByteArray();
    }

    const PivotEstimatorCandidate* best = FindBestPivotEstimatorCandidate(computation.candidates);
    if (best == nullptr) return ByteArray();

    std::ostringstream json;
    json << "{";
    json << "\"planner_strategy\":\"greedy_leaf_packing\",";
    json << "\"min_local_selectivity\":" << kGreedyLeafMinLocalSelectivity << ",";
    json << "\"num_vectors\":" << p_numVectors << ",";
    json << "\"num_levels\":" << p_numTagsPerVec << ",";
    json << "\"requested_max_nodes\":" << p_maxNodes << ",";
    json << "\"recall_target\":" << std::clamp(static_cast<double>(p_recallTarget), 0.01, 1.0) << ",";
    json << "\"lambda_recall\":" << std::max(0.0, static_cast<double>(p_lambdaRecall)) << ",";
    json << "\"estimated_recall\":" << std::clamp(static_cast<double>(p_estimatedRecall), 0.0, 1.0) << ",";
    json << "\"best_plan\":{";
    json << "\"pivot_level\":" << best->pivotLevel << ",";
    json << "\"node_count\":" << best->nodeCount << ",";
    json << "\"latency_cost\":" << best->latencyCost << ",";
    json << "\"recall_penalty\":" << best->recallPenalty << ",";
    json << "\"total_cost\":" << best->totalCost << "},";
    json << "\"candidates\":[";
    for (size_t idx = 0; idx < computation.candidates.size(); ++idx)
    {
        const auto& candidate = computation.candidates[idx];
        if (idx > 0) json << ",";
        json << "{";
        json << "\"pivot_level\":" << candidate.pivotLevel << ",";
        json << "\"node_count\":" << candidate.nodeCount << ",";
        json << "\"latency_cost\":" << candidate.latencyCost << ",";
        json << "\"recall_penalty\":" << candidate.recallPenalty << ",";
        json << "\"total_cost\":" << candidate.totalCost << ",";
        json << "\"node_sizes\":[";
        for (size_t i = 0; i < candidate.nodeSizes.size(); ++i) {
            if (i > 0) json << ",";
            json << candidate.nodeSizes[i];
        }
        json << "],";
        json << "\"node_pivot_tags\":[";
        for (size_t node = 0; node < candidate.nodePivotTags.size(); ++node)
        {
            if (node > 0) json << ",";
            json << "[";
            for (size_t tagIdx = 0; tagIdx < candidate.nodePivotTags[node].size(); ++tagIdx)
            {
                if (tagIdx > 0) json << ",";
                json << candidate.nodePivotTags[node][tagIdx];
            }
            json << "]";
        }
        json << "]";
        json << "}";
    }
    json << "]";
    json << "}";

    const std::string payload = json.str();
    ByteArray output = ByteArray::Alloc(payload.size());
    std::memcpy(output.Data(), payload.data(), payload.size());
    return output;
}

bool TenantIndexManager::SaveAll(const char* p_baseDir)
{
    m_baseStoragePath = std::string(p_baseDir);
    if (!EnsureDir(m_baseStoragePath))
    {
        return false;
    }

    // For SPANN: copy shared SSD infrastructure once
    // For other algos: save each tenant's data sequentially without directory overhead
    
    if (!SaveUnifiedStorage(p_baseDir))
    {
        return false;
    }

    // Checkpoint the shared RocksDB into <baseDir>/rocksdb_shared_0/ when
    // saving to a directory other than where the live DB lives. RocksDB
    // persists in place, so same-dir saves are a no-op.
    if (m_useSharedDB && m_sharedDB)
    {
        if (m_sharedDB->Checkpoint(std::string(p_baseDir)) != SPTAG::ErrorCode::Success)
        {
            fprintf(stderr, "[ERROR] TenantIndexManager::SaveAll: failed to checkpoint shared RocksDB to %s\n", p_baseDir);
            return false;
        }
    }

    // Write manifest for all tenants
    std::string manifestPath = m_baseStoragePath + "/manifest.txt";
    FILE* manifestFile = fopen(manifestPath.c_str(), "w");
    if (!manifestFile)
    {
        return false;
    }

    fprintf(manifestFile, "dimension %d\n", static_cast<int>(m_dimension));
    fprintf(manifestFile, "algorithm %s\n", m_algoType == SPTAG::IndexAlgoType::SPANN ? "SPANN" : 
            (m_algoType == SPTAG::IndexAlgoType::BKT ? "BKT" : "KDT"));
    fprintf(manifestFile, "unified_storage 1\n");
    fprintf(manifestFile, "total_postings %d\n", m_totalPostingCount);
    
    for (const auto& kv : m_tenantVectorCounts)
    {
        int tenantId = kv.first;
        int count = kv.second;
        int postingOffset = 0;
        int headCount = 0;
        auto offIt = m_tenantPostingOffsets.find(tenantId);
        if (offIt != m_tenantPostingOffsets.end()) postingOffset = offIt->second;
        auto hcIt = m_tenantHeadCounts.find(tenantId);
        if (hcIt != m_tenantHeadCounts.end()) headCount = hcIt->second;
        int typeInt = 0;
        auto typeIt = m_tenantIndexTypes.find(tenantId);
        if (typeIt != m_tenantIndexTypes.end()) typeInt = static_cast<int>(typeIt->second);
        // Format: tenant <id> <vecCount> <postingOffset> <headCount> <indexType>
        fprintf(manifestFile, "tenant %d %d %d %d %d\n", tenantId, count, postingOffset, headCount, typeInt);
    }

    // Save string tenant ID ↔ internal ID mapping
    {
        std::lock_guard<std::mutex> lock(m_tenantIdMutex);
        for (const auto& kv : m_tenantStrToInt)
        {
            // Format: tenant_mapping <internalId> <stringId>
            fprintf(manifestFile, "tenant_mapping %d %s\n", kv.second, kv.first.c_str());
        }
    }

    fclose(manifestFile);

    return true;
}

bool TenantIndexManager::LoadAll(const char* p_baseDir)
{
    m_tenantIndices.clear();
    m_lruList.clear();
    m_lruMap.clear();
    m_tenantHeadIndexAccountedBytes.clear();
    m_loadedHeadIndexBytes = 0;
    m_tenantVectorCounts.clear();
    m_tenantIndexPaths.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantTagRoutingStats.clear();
    m_tenantPivotLevels.clear();
    m_tenantPivotNodeCounts.clear();
    m_tenantNodePivotTags.clear();
    m_tenantTagToNodes.clear();
    m_tenantHeadNodeToNode.clear();

    // Clear tenant ID mapping
    {
        std::lock_guard<std::mutex> lock(m_tenantIdMutex);
        m_tenantStrToInt.clear();
        m_tenantIntToStr.clear();
        m_nextInternalId = 0;
    }

    std::string baseDir(p_baseDir);
    m_baseStoragePath = baseDir;
    std::string manifestPath = baseDir + "/manifest.txt";
    std::ifstream in(manifestPath.c_str());
    if (!in)
    {
        return false;
    }

    // Read manifest
    std::string line;
    bool unifiedStorage = false;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "dimension")
        {
            int dim = 0;
            if (!(iss >> dim) || dim != m_dimension)
            {
                return false;
            }
        }
        else if (key == "unified_storage")
        {
            int val = 0;
            if (iss >> val)
            {
                unifiedStorage = (val != 0);
            }
        }
        else if (key == "total_postings")
        {
            int val = 0;
            if (iss >> val) m_totalPostingCount = val;
        }
        else if (key == "tenant")
        {
            int tenantId = 0;
            int count = 0;
            int postingOffset = 0;
            int headCount = 0;
            int typeInt = 0;
            if (!(iss >> tenantId >> count))
            {
                return false;
            }
            // Optional fields: postingOffset, headCount, indexType
            iss >> postingOffset >> headCount >> typeInt;
            m_tenantVectorCounts[tenantId] = count;
            m_tenantPostingOffsets[tenantId] = postingOffset;
            m_tenantHeadCounts[tenantId] = headCount;
            m_tenantIndexTypes[tenantId] = static_cast<TenantIndexType>(typeInt);
        }
        else if (key == "tenant_mapping")
        {
            int internalId = 0;
            std::string strId;
            if (iss >> internalId >> strId)
            {
                std::lock_guard<std::mutex> lock(m_tenantIdMutex);
                m_tenantStrToInt[strId] = internalId;
                m_tenantIntToStr[internalId] = strId;
                if (internalId >= m_nextInternalId)
                    m_nextInternalId = internalId + 1;
            }
        }
    }
    in.close();

    // Load tenant indices based on storage type
    if (unifiedStorage)
    {
        return LoadUnifiedStorage(p_baseDir);
    }
    else
    {
        // Legacy: load from tenant_XX directories for backward compatibility
        for (const auto& kv : m_tenantVectorCounts)
        {
            int tenantId = kv.first;
            m_tenantSpannWorkDirs[tenantId] = baseDir + "/tenant_" + std::to_string(tenantId) + "/index";
        }
        if (!LoadTenantTagRoutingStats()) return false;
        LoadTenantSparseIndices();
        LoadTenantTagPureIndices();
        return true;
    }
}

bool TenantIndexManager::LoadTenantTagRoutingStats()
{
    for (const auto& entry : m_tenantSpannWorkDirs) {
        const int tenantId = entry.first;
        m_tenantTagRoutingStats.erase(tenantId);
        const auto vectorCount = m_tenantVectorCounts.find(tenantId);
        if (vectorCount == m_tenantVectorCounts.end() ||
            vectorCount->second <= 0) {
            fprintf(stderr,
                    "[ERROR] Tenant %d: invalid vector count for tag routing stats\n",
                    tenantId);
            return false;
        }
        const std::string path =
            entry.second + "/tag_routing_stats.bin";
        TagRoutingStatsMap stats;
        std::uint64_t generationFingerprint = 0;
        if (LoadTagRoutingStatsFile(
                path, vectorCount->second,
                generationFingerprint, stats)) {
            const std::string iniPath =
                entry.second + "/indexloader.ini";
            if (IniEnablesHybridDistance(iniPath)) {
                std::string postingFile =
                    ReadBuildSSDIndexValue(
                        iniPath, "SSDIndex");
                if (postingFile.empty() ||
                    postingFile == "Undefined!") {
                    postingFile =
                        "SPTAGFullList.bin";
                }
                SPTAG::SPANN::HybridRoutingStatsHeader
                    hybridHeader;
                if (!LoadHybridRoutingStatsHeader(
                        entry.second + "/" +
                            postingFile +
                            ".hybrid.stats",
                        hybridHeader) ||
                    generationFingerprint == 0 ||
                    generationFingerprint !=
                        hybridHeader
                            .m_generationFingerprint) {
                    fprintf(
                        stderr,
                        "[WARN] Tenant %d: tag routing stats generation "
                        "does not match the primary hybrid posting; filtered search is "
                        "disabled until BuildSignatures regenerates it\n",
                        tenantId);
                    continue;
                }
            }
            m_tenantTagRoutingStats[tenantId] =
                std::move(stats);
            continue;
        }
        const bool required = IniEnablesHybridDistance(
            entry.second + "/indexloader.ini");
        if (required) {
            fprintf(stderr,
                    "[WARN] Tenant %d: hybrid routing requires a valid %s; "
                    "filtered search is disabled until BuildSignatures "
                    "generates it\n",
                    tenantId, path.c_str());
        }
    }
    return true;
}

void TenantIndexManager::LoadTenantSparseIndices()
{
    // Sparse-tag fast-path index is small (<<1MB/tenant) but the saved
    // m_tenantSparseIdx map is only populated at build time. Without this
    // load step, query-side sparse routing in SearchWithTags is a no-op on
    // any process that only Load()s the index.
    int loadedCount = 0;
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantSparseIdx.count(tenantId)) continue;
        const std::string sparsePath = kv.second + "/sparse_tags.bin";
        struct stat st{};
        if (stat(sparsePath.c_str(), &st) != 0) continue;
        auto sparseIdx = std::make_shared<SPTAG::Cache::SparseTagIndex>();
        if (!sparseIdx->Load(sparsePath))
        {
            fprintf(stderr, "[WARN] Tenant %d: failed to load sparse_tags.bin (%s)\n",
                    tenantId, sparsePath.c_str());
            continue;
        }
        m_tenantSparseIdx[tenantId] = std::move(sparseIdx);
        ++loadedCount;
    }
    if (loadedCount > 0)
    {
        fprintf(stderr, "[INFO] Loaded sparse tag indices for %d tenants\n", loadedCount);
    }

    // Load per-level tag offsets (used to map a raw query tag value to its
    // hierarchical level). Independent of sparse_tags availability.
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantTagLevelOffsets.count(tenantId)) continue;
        const std::string offPath = kv.second + "/tag_level_offsets.bin";
        FILE* of = fopen(offPath.c_str(), "rb");
        if (!of) continue;
        int32_t nLevels = 0;
        if (fread(&nLevels, sizeof(int32_t), 1, of) == 1 && nLevels > 0 && nLevels < 64) {
            std::vector<uint32_t> offs(static_cast<size_t>(nLevels));
            if (fread(offs.data(), sizeof(uint32_t), offs.size(), of) == offs.size()) {
                m_tenantTagLevelOffsets[tenantId] = std::move(offs);
                fprintf(stderr, "[INFO] Tenant %d: loaded tag_level_offsets.bin (%d levels)\n",
                        tenantId, nLevels);
            }
        }
        fclose(of);
    }

    // Load numeric attribute metadata (quantized signature domains). Optional —
    // absent => tenant has no numeric columns.
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantNumericMeta.count(tenantId)) continue;
        const std::string nmPath = kv.second + "/numeric_meta.bin";
        FILE* nf = fopen(nmPath.c_str(), "rb");
        if (!nf) continue;
        int32_t magic = 0, base = 0, ncols = 0;
        if (fread(&magic, sizeof(int32_t), 1, nf) == 1 && magic == 0x54454d4e &&
            fread(&base, sizeof(int32_t), 1, nf) == 1 &&
            fread(&ncols, sizeof(int32_t), 1, nf) == 1 && ncols > 0 && ncols < 4096) {
            NumericMeta nm;
            nm.numBaseCols = base;
            nm.params.resize(static_cast<size_t>(ncols));
            bool ok = true;
            for (int c = 0; c < ncols; ++c) {
                if (fread(&nm.params[c].lo, sizeof(uint32_t), 1, nf) != 1 ||
                    fread(&nm.params[c].hi, sizeof(uint32_t), 1, nf) != 1) { ok = false; break; }
            }
            if (ok) {
                m_tenantNumericMeta[tenantId] = std::move(nm);
                fprintf(stderr, "[INFO] Tenant %d: loaded numeric_meta.bin (base=%d numeric=%d)\n",
                        tenantId, base, ncols);
            }
        }
        fclose(nf);
    }
}

void TenantIndexManager::LoadTenantTagPureIndices()
{
    // Tag-pure chunks live inside the SPANN KV store (FileIO mapping / RocksDB).
    // The chunk data persists across runs because BuildSignatures explicitly
    // calls extra->Checkpoint() after writing chunks. The sidecar holds the
    // metadata (tag → chunkKeys, chunkCounts, count). Without loading it
    // back, SearchWithACL would fall through the fast path.
    //
    // When the OPQ prefilter is enabled, every single-tag query is served by
    // OPQTagPureSearch (resident codes + the canonical vid->vector store),
    // so the tag-pure full-vector chunks are a redundant SECOND copy of the
    // vectors. Skip loading their metadata entirely: this drops the in-memory
    // tag->chunkKeys map and guarantees the dead chunk path is never taken,
    // realizing the single-vector-copy goal. (The chunk bytes embedded in the
    // index KV file remain inert; a clean rebuild without chunks reclaims them.)
    static const bool s_opqPrefilterSkipChunks = []() {
        const char* v = std::getenv("SPTAG_OPQ_PREFILTER");
        return v && v[0] == '1';
    }();
    if (s_opqPrefilterSkipChunks) {
        fprintf(stderr,
            "[INFO] OPQ prefilter ON: skipping tag-pure full-vector chunk load "
            "(single canonical vector copy in opq_vecstore)\n");
        return;
    }
    int loadedCount = 0;
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantTagPurePostings.count(tenantId)) continue;
        const std::string metaPath = kv.second + "/tagpure_meta.bin";
        struct stat st{};
        if (stat(metaPath.c_str(), &st) != 0) continue;

        int loadedDim = 0;
        std::unordered_map<uint32_t, std::shared_ptr<SPTAG::Cache::TagPurePosting>> tags;
        if (!SPTAG::Cache::TagPureBundle::Load(metaPath, loadedDim, tags))
        {
            fprintf(stderr, "[WARN] Tenant %d: failed to load tagpure_meta.bin (%s)\n",
                    tenantId, metaPath.c_str());
            continue;
        }

        // Resolve KV store + page budget from the (now loaded) SPANN index.
        // EnsureTenantLoaded must succeed for the kvDb to be available.
        if (!EnsureTenantLoaded(tenantId))
        {
            fprintf(stderr, "[WARN] Tenant %d: cannot ensure loaded for tag-pure attach\n",
                    tenantId);
            continue;
        }

        std::shared_ptr<SPTAG::Helper::KeyValueIO> kvDb;
        int postingPageLimit = 3, bufferLength = 4;
        SPTAG::SizeType nextKey = 0;
        {
            std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
            auto it = m_tenantIndices.find(tenantId);
            if (it != m_tenantIndices.end()) {
                auto internalIdx = it->second->GetInternalIndex();
                auto* spannIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
                if (spannIdx != nullptr) {
                    if (auto* opts = spannIdx->GetOptions()) {
                        postingPageLimit = std::max(1, opts->m_postingPageLimit);
                        bufferLength = std::max(0, opts->m_bufferLength);
                    }
                    auto extra = spannIdx->GetDiskIndex();
                    if (extra) kvDb = extra->GetKVStore();
                }
            }
        }
        if (kvDb == nullptr) {
            fprintf(stderr, "[WARN] Tenant %d: no KV store while attaching tag-pure metadata\n",
                    tenantId);
            continue;
        }

        // Highest chunk key + 1 — covers future incremental rebuilds.
        for (const auto& tkv : tags) {
            if (!tkv.second) continue;
            for (int k : tkv.second->chunkKeys) {
                if (k + 1 > nextKey) nextKey = k + 1;
            }
        }

        m_tenantTagPurePostings[tenantId] = std::move(tags);
        m_tenantTagPureKV[tenantId] = std::move(kvDb);
        m_tenantTagPureBlockLimit[tenantId] = postingPageLimit + bufferLength + 1;
        m_tenantTagPurePagesPerChunk[tenantId] = postingPageLimit;
        m_tenantTagPureNextKey[tenantId] = nextKey;
        ++loadedCount;
    }
    if (loadedCount > 0)
    {
        fprintf(stderr, "[INFO] Loaded tag-pure indices for %d tenants\n", loadedCount);
    }
}

bool TenantIndexManager::SaveUnifiedStorage(const char* p_baseDir)
{
    std::string baseDir(p_baseDir);

    // Save tenants that are still in memory
    for (const auto& kv : m_tenantIndices)
    {
        int tenantId = kv.first;
        std::string dstTenantDir = baseDir + "/tenant_" + std::to_string(tenantId);
        if (!EnsureDir(dstTenantDir))
            return false;

        auto typeIt = m_tenantIndexTypes.find(tenantId);
        TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;

        if (indexType == TenantIndexType::SPANN)
        {
            if (!kv.second->Save(dstTenantDir.c_str()))
            {
                fprintf(stderr, "[ERROR] Failed to save SPANN index for tenant %d\n", tenantId);
                return false;
            }
            fprintf(stderr, "[INFO] Tenant %d: saved full SPANN index\n", tenantId);
        }
        else
        {
            std::string indexPath = dstTenantDir + "/index";
            if (!kv.second->Save(indexPath.c_str()))
            {
                fprintf(stderr, "[ERROR] Failed to save BKT/BF index for tenant %d\n", tenantId);
                return false;
            }
            fprintf(stderr, "[INFO] Tenant %d: saved BKT/BF index\n", tenantId);
        }
    }

    // Copy tenants that were already saved-and-released during build
    // (they exist in m_tenantSpannWorkDirs but not in m_tenantIndices)
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantIndices.count(tenantId)) continue;  // Already saved above

        std::string srcDir = kv.second;
        std::string dstDir = baseDir + "/tenant_" + std::to_string(tenantId);

        if (srcDir == dstDir) {
            // Already in the right place (saved directly to output dir)
            fprintf(stderr, "[INFO] Tenant %d: already saved in place\n", tenantId);
            continue;
        }

        if (!EnsureDir(dstDir)) return false;
        if (!CopyDirRecursive(srcDir, dstDir))
        {
            fprintf(stderr, "[ERROR] Failed to copy tenant %d from %s to %s\n", tenantId, srcDir.c_str(), dstDir.c_str());
            return false;
        }
        // Update work dir to point to final location
        m_tenantSpannWorkDirs[tenantId] = dstDir;
        fprintf(stderr, "[INFO] Tenant %d: copied from build dir\n", tenantId);
    }

    int totalSaved = (int)m_tenantIndices.size();
    for (const auto& kv : m_tenantSpannWorkDirs)
        if (!m_tenantIndices.count(kv.first)) totalSaved++;
    fprintf(stderr, "[INFO] Unified storage saved: %d tenants (%d SPANN)\n",
        totalSaved,
        (int)std::count_if(m_tenantIndexTypes.begin(), m_tenantIndexTypes.end(),
            [](const auto& kv) { return kv.second == TenantIndexType::SPANN; }));

    return true;
}

bool TenantIndexManager::LoadUnifiedStorage(const char* p_baseDir)
{
    std::string baseDir(p_baseDir);

    for (const auto& kv : m_tenantVectorCounts)
    {
        int tenantId = kv.first;
        auto typeIt = m_tenantIndexTypes.find(tenantId);
        TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;

        std::string tenantDir = baseDir + "/tenant_" + std::to_string(tenantId);

        if (indexType == TenantIndexType::SPANN)
        {
            m_tenantSpannWorkDirs[tenantId] = tenantDir;
        }
        else
        {
            // BKT / BruteForce: store index path for lazy loading
            m_tenantIndexPaths[tenantId] = tenantDir + "/index";
        }
    }

    if (!LoadTenantTagRoutingStats()) return false;
    LoadTenantSparseIndices();
    LoadTenantTagPureIndices();

    return true;
}

void TenantIndexManager::SetBuildParam(const char* p_name, const char* p_value, const char* p_section)
{
    if (p_name == nullptr || p_value == nullptr || p_section == nullptr) {
        return;
    }

    bool updated = false;
    for (auto& pendingParam : m_pendingBuildParams)
    {
        if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(std::get<0>(pendingParam).c_str(), p_name)
            && SPTAG::Helper::StrUtils::StrEqualIgnoreCase(std::get<2>(pendingParam).c_str(), p_section))
        {
            std::get<1>(pendingParam) = p_value;
            updated = true;
            break;
        }
    }
    if (!updated) {
        m_pendingBuildParams.emplace_back(p_name, p_value, p_section);
    }

    for (auto& tenantEntry : m_tenantIndices)
    {
        tenantEntry.second->SetBuildParam(p_name, p_value, p_section);
    }
}

void TenantIndexManager::SetSearchParam(const char* p_name, const char* p_value, const char* p_section)
{
    if (p_name == nullptr || p_value == nullptr || p_section == nullptr) {
        return;
    }

    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    bool updated = false;
    for (auto& pendingParam : m_pendingSearchParams)
    {
        if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(std::get<0>(pendingParam).c_str(), p_name)
            && SPTAG::Helper::StrUtils::StrEqualIgnoreCase(std::get<2>(pendingParam).c_str(), p_section))
        {
            std::get<1>(pendingParam) = p_value;
            updated = true;
            break;
        }
    }
    if (!updated)
    {
        m_pendingSearchParams.emplace_back(p_name, p_value, p_section);
    }

    for (auto& tenantEntry : m_tenantIndices)
    {
        tenantEntry.second->SetSearchParam(p_name, p_value, p_section);
    }
}

bool TenantIndexManager::EnsureTenantLoaded(int p_tenantId)
{
    // Fast path: shared lock check (hot cache)
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        if (m_tenantIndices.count(p_tenantId))
        {
            // Skip LRU update on fast path — splice is not thread-safe under shared_lock.
            // LRU order is approximate; only updated on slow path (exclusive lock).
            return true;
        }
    }

    // Slow path: exclusive lock
    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    // Double-check hot cache
    if (m_tenantIndices.count(p_tenantId))
    {
        auto it = m_lruMap.find(p_tenantId);
        if (it != m_lruMap.end())
            m_lruList.splice(m_lruList.end(), m_lruList, it->second);
        return true;
    }

    // Estimate loaded HeadIndex bytes from on-disk bytes and a safety factor.
    uint64_t estimatedBytes = EstimateTenantHeadIndexBytes(p_tenantId);

    // Soft-evict LRU hot tenants until we have room
    if (m_headIndexCacheLimitBytes > 0)
    {
        int retries = 0;
        while (m_loadedHeadIndexBytes + estimatedBytes > m_headIndexCacheLimitBytes
               && !m_lruList.empty())
        {
            int evictId = m_lruList.front();
            if (evictId == p_tenantId) break;
            bool evicted = UnloadTenantLocked(evictId);
            if (!evicted)
            {
                // Tenant is in use (use_count > 1). Try next LRU candidate.
                // Move to back of LRU to avoid spinning on same tenant.
                m_lruList.pop_front();
                m_lruList.push_back(evictId);
                m_lruMap[evictId] = std::prev(m_lruList.end());
                retries++;
                if (retries > (int)m_lruList.size())
                {
                    // All LRU candidates are pinned (held by caller or another thread).
                    // Break and allow the cache to temporarily exceed the limit.
                    break;
                }
            }
        }
    }

    // Best-effort RSS guard using current process RSS instead of only estimated cache bytes.
    if (m_rssHighWaterMarkBytes > 0)
    {
        uint64_t currentRSSBytes = GetCurrentProcessRSSBytes();
        int retries = 0;
        while (currentRSSBytes > 0
               && currentRSSBytes + estimatedBytes > m_rssHighWaterMarkBytes
               && !m_lruList.empty())
        {
            int evictId = m_lruList.front();
            if (evictId == p_tenantId) break;
            bool evicted = UnloadTenantLocked(evictId);
            if (!evicted)
            {
                m_lruList.pop_front();
                m_lruList.push_back(evictId);
                m_lruMap[evictId] = std::prev(m_lruList.end());
                retries++;
                if (retries > (int)m_lruList.size())
                {
                    break;
                }
            }
            else
            {
                retries = 0;
                currentRSSBytes = GetCurrentProcessRSSBytes();
            }
        }

        if (currentRSSBytes > 0 && currentRSSBytes + estimatedBytes > m_rssHighWaterMarkBytes)
        {
            fprintf(stderr,
                    "[WARN] Rejecting tenant %d load: current RSS %.2f MB + estimated HeadIndex %.2f MB exceeds RSS high-water %.2f MB\n",
                    p_tenantId,
                    currentRSSBytes / (1024.0 * 1024.0),
                    estimatedBytes / (1024.0 * 1024.0),
                    m_rssHighWaterMarkBytes / (1024.0 * 1024.0));
            return false;
        }
    }

    // Full load from disk
    auto typeIt = m_tenantIndexTypes.find(p_tenantId);
    TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;
    std::string loadPath;
    if (indexType == TenantIndexType::SPANN)
    {
        auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (workIt == m_tenantSpannWorkDirs.end()) return false;
        loadPath = workIt->second;
    }
    else
    {
        auto pathIt = m_tenantIndexPaths.find(p_tenantId);
        if (pathIt == m_tenantIndexPaths.end()) return false;
        loadPath = pathIt->second;
    }

    std::shared_ptr<AnnIndex> indexPtr;
    if (indexType == TenantIndexType::SPANN && m_storageBackend == "ROCKSDBIO" && m_useSharedDB)
    {
        indexPtr = LoadSpannWithSharedDB(loadPath, p_tenantId);
        if (indexPtr == nullptr || !indexPtr->ReadyToServe())
        {
            fprintf(stderr, "[ERROR] Failed to load tenant %d (shared-DB) from %s\n", p_tenantId, loadPath.c_str());
            return false;
        }
    }
    else
    {
        AnnIndex tmp = AnnIndex::Load(loadPath.c_str());
        if (!tmp.ReadyToServe())
        {
            fprintf(stderr, "[ERROR] Failed to load tenant %d from %s\n", p_tenantId, loadPath.c_str());
            return false;
        }
        indexPtr = std::make_shared<AnnIndex>(tmp);
    }
    if (m_rssHighWaterMarkBytes > 0)
    {
        uint64_t currentRSSBytes = GetCurrentProcessRSSBytes();
        if (currentRSSBytes > 0 && currentRSSBytes > m_rssHighWaterMarkBytes)
        {
            fprintf(stderr,
                    "[WARN] Rejecting tenant %d after load: current RSS %.2f MB exceeds RSS high-water %.2f MB\n",
                    p_tenantId,
                    currentRSSBytes / (1024.0 * 1024.0),
                    m_rssHighWaterMarkBytes / (1024.0 * 1024.0));
            return false;
        }
    }

    for (const auto& pendingParam : m_pendingSearchParams)
    {
        indexPtr->SetSearchParam(std::get<0>(pendingParam).c_str(),
                                 std::get<1>(pendingParam).c_str(),
                                 std::get<2>(pendingParam).c_str());
    }
    m_tenantIndices[p_tenantId] = indexPtr;
    m_loadedHeadIndexBytes += estimatedBytes;
    m_tenantHeadIndexAccountedBytes[p_tenantId] = estimatedBytes;
    m_lruList.push_back(p_tenantId);
    m_lruMap[p_tenantId] = std::prev(m_lruList.end());

    EnsureHeadNodeMetaLoaded(loadPath, indexPtr->GetInternalIndex());
    if (indexType == TenantIndexType::SPANN) {
        EnsureTenantPivotIndexLoaded(p_tenantId);
    }

    return true;
}

bool TenantIndexManager::EnsureTenantPivotIndexLoaded(int p_tenantId)
{
    if (m_tenantPivotLevels.count(p_tenantId) &&
        m_tenantPivotNodeCounts.count(p_tenantId) &&
        m_tenantNodePivotTags.count(p_tenantId) &&
        m_tenantTagToNodes.count(p_tenantId) &&
        m_tenantHeadNodeToNode.count(p_tenantId)) {
        return true;
    }

    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt == m_tenantSpannWorkDirs.end()) return false;

    int pivotLevel = -1;
    int nodeCount = 0;
    std::vector<std::vector<uint32_t>> nodePivotTags;
    std::unordered_map<uint32_t, std::vector<int>> tagToNodes;
    std::vector<int> headNodeToNode;
    if (!LoadHeadNodeRoutingIndexFile(wdIt->second,
                                      pivotLevel,
                                      nodeCount,
                                      nodePivotTags,
                                      tagToNodes,
                                      headNodeToNode)) {
        return false;
    }

    m_tenantPivotLevels[p_tenantId] = pivotLevel;
    m_tenantPivotNodeCounts[p_tenantId] = nodeCount;
    m_tenantNodePivotTags[p_tenantId] = std::move(nodePivotTags);
    m_tenantTagToNodes[p_tenantId] = std::move(tagToNodes);
    m_tenantHeadNodeToNode[p_tenantId] = std::move(headNodeToNode);
    return true;
}

void TenantIndexManager::InitCache()
{
    SPTAG::Cache::HeadIndexCache::Config cfg;
    cfg.capacity_bytes = m_headIndexCacheLimitBytes;
    cfg.ttl = std::chrono::seconds(600);
    cfg.load_timeout = std::chrono::milliseconds(30000);
    m_headCache = std::make_unique<SPTAG::Cache::HeadIndexCache>(cfg);
}

void TenantIndexManager::SetHeadIndexCacheLimit(uint64_t p_bytesLimit)
{
    m_headIndexCacheLimitBytes = p_bytesLimit;
    if (m_headCache) {
        m_headCache->SetCapacity(p_bytesLimit);
    }
    fprintf(stderr, "[INFO] HeadIndex cache limit set to %lu bytes (%.1f MB)\n",
            (unsigned long)p_bytesLimit, p_bytesLimit / (1024.0 * 1024.0));
}

void TenantIndexManager::SetHeadIndexCacheSafetyFactor(double p_factor)
{
    if (p_factor < 1.0) p_factor = 1.0;
    if (p_factor > 8.0) p_factor = 8.0;
    m_headIndexCacheSafetyFactor = p_factor;
    fprintf(stderr, "[INFO] HeadIndex cache safety factor set to %.3f\n", m_headIndexCacheSafetyFactor);
}

double TenantIndexManager::GetHeadIndexCacheSafetyFactor() const
{
    return m_headIndexCacheSafetyFactor;
}

uint64_t TenantIndexManager::GetHeadIndexCacheUsage() const
{
    std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
    return m_loadedHeadIndexBytes;
}

uint64_t TenantIndexManager::GetCurrentRSSBytes() const
{
    return GetCurrentProcessRSSBytes();
}

void TenantIndexManager::SetRSSHighWaterMark(uint64_t p_bytesLimit)
{
    m_rssHighWaterMarkBytes = p_bytesLimit;
    fprintf(stderr, "[INFO] RSS high-water mark set to %lu bytes (%.1f MB)\n",
            (unsigned long)p_bytesLimit, p_bytesLimit / (1024.0 * 1024.0));
}

uint64_t TenantIndexManager::GetRSSHighWaterMark() const
{
    return m_rssHighWaterMarkBytes;
}

uint64_t TenantIndexManager::GetLastPostingReadCount() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_readPostings;
}

uint64_t TenantIndexManager::GetLastPostingMatchCount() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_matchedPostings;
}

uint64_t TenantIndexManager::GetLastPostingFP() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().FalsePositivePostings();
}

uint64_t TenantIndexManager::GetLastPostingPrePS() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_prePSPostings;
}

uint64_t TenantIndexManager::GetLastScannedVectors() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_scannedVectors;
}

uint64_t TenantIndexManager::GetLastMatchedVectors() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_matchedVectors;
}

uint64_t TenantIndexManager::GetLastPrimaryHeadCandidateCount() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_primaryHeadCandidates;
}

bool TenantIndexManager::UnloadTenant(int p_tenantId)
{
    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    return UnloadTenantLocked(p_tenantId);
}

bool TenantIndexManager::UnloadTenantLocked(int p_tenantId)
{
    // Must be called under exclusive lock (m_tenantIndicesMutex)
    auto it = m_tenantIndices.find(p_tenantId);
    if (it == m_tenantIndices.end()) return false;

    // SAFETY: If another thread holds a shared_ptr to this index (e.g. BatchSearch
    // in progress), skip eviction. The search thread's shared_ptr keeps the object alive.
    // use_count > 1 means: 1 (in map) + N (held by search threads).
    if (it->second.use_count() > 1)
    {
        return false;  // Skip: tenant in use
    }

    uint64_t freedBytes = 0;
    auto accountedIt = m_tenantHeadIndexAccountedBytes.find(p_tenantId);
    if (accountedIt != m_tenantHeadIndexAccountedBytes.end()) {
        freedBytes = accountedIt->second;
        m_tenantHeadIndexAccountedBytes.erase(accountedIt);
    } else {
        freedBytes = EstimateTenantHeadIndexBytes(p_tenantId);
    }

    // With SharedAIOPool: destruction only does close(fd) + free memory (~1ms).
    // AIO contexts are shared and never destroyed.

    // DisableCheckpoint=true (from ini/default): ShutDown never writes back.
    it->second.reset();
    m_tenantIndices.erase(it);

    // Update cache accounting
    if (m_loadedHeadIndexBytes >= freedBytes)
        m_loadedHeadIndexBytes -= freedBytes;
    else
        m_loadedHeadIndexBytes = 0;

    // Remove from LRU
    auto lruIt = m_lruMap.find(p_tenantId);
    if (lruIt != m_lruMap.end())
    {
        m_lruList.erase(lruIt->second);
        m_lruMap.erase(lruIt);
    }

    // Drop OS page cache for this tenant's HeadIndex files.
    // This ensures next load hits real disk IO, not page cache.
    if (m_dropPageCacheOnEvict)
    {
        std::string hiDir;
        auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (wdIt != m_tenantSpannWorkDirs.end())
            hiDir = wdIt->second + "/HeadIndex";
        if (!hiDir.empty())
        {
            DIR* dir = opendir(hiDir.c_str());
            if (dir) {
                struct dirent* ent;
                while ((ent = readdir(dir)) != nullptr) {
                    if (ent->d_name[0] == '.') continue;
                    std::string fp = hiDir + "/" + ent->d_name;
                    int fd = open(fp.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        struct stat st;
                        fstat(fd, &st);
                        posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
                        close(fd);
                    }
                }
                closedir(dir);
            }
        }
    }

    return true;
}

void TenantIndexManager::TouchLRU(int p_tenantId)
{
    // No-op: S3-FIFO handles promotion internally via freq counter
}

void TenantIndexManager::EvictIfNeeded()
{
    // No-op: HeadIndexCache handles eviction internally
}

// ============================================================================
// ACL / Tag Filtered Search — Two-Level Signature Implementation
// ============================================================================

bool TenantIndexManager::BuildSignatures(int p_tenantId, ByteArray p_tags, int p_numVectors, int p_numTagsPerVec)
{
    const uint32_t* p_tagsPtr = reinterpret_cast<const uint32_t*>(p_tags.Data());

    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt == m_tenantSpannWorkDirs.end()) return false;
    std::string workDir = wdIt->second;

    // DirectSparseMaxPostings is a native [BuildSSDIndex] parameter. The
    // sparse-tag sidecar is built after the SPANN store, so retrieve the
    // persisted option from the just-built (or reloaded) index instead of
    // consulting a process environment override.
    int directSparseMaxPostings = 320;
    bool hybridDistanceEnabled = false;
    bool staticStorage = false;
    std::string primaryPostingFile =
        "SPTAGFullList.bin";
    if (!EnsureTenantLoaded(p_tenantId)) return false;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it != m_tenantIndices.end()) {
            auto internalIdx = it->second->GetInternalIndex();
            if (internalIdx != nullptr) {
                const std::string configured =
                    internalIdx->GetParameter("DirectSparseMaxPostings", "BuildSSDIndex");
                int parsed = 0;
                if (SPTAG::Helper::Convert::ConvertStringTo<int>(configured.c_str(), parsed)
                    && parsed > 0) {
                    directSparseMaxPostings = parsed;
                } else if (!configured.empty()) {
                    fprintf(stderr,
                            "[WARN] Tenant %d: invalid DirectSparseMaxPostings=%s; using %d\n",
                            p_tenantId, configured.c_str(), directSparseMaxPostings);
                }
                const std::string hybridConfigured =
                    internalIdx->GetParameter(
                        "EnableHybridDistance",
                        "BuildSSDIndex");
                if (!hybridConfigured.empty()) {
                    SPTAG::Helper::Convert::
                        ConvertStringTo<bool>(
                            hybridConfigured.c_str(),
                            hybridDistanceEnabled);
                }
                if (auto* spann = dynamic_cast<
                        SPTAG::SPANN::ISPANNIndex*>(
                        internalIdx.get())) {
                    const auto* options =
                        spann->GetOptions();
                    staticStorage =
                        options != nullptr &&
                        options->m_storage ==
                            SPTAG::Storage::STATIC;
                    if (options != nullptr &&
                        !options->m_ssdIndex.empty()) {
                        primaryPostingFile =
                            options->m_ssdIndex;
                    }
                }
            }
        }
    }

    // ── Idempotent fast path ─────────────────────────────────────────────
    // If all three persisted artifacts (PS bitmask, sparse tags, tag-pure
    // metadata) are already on disk, the previous BuildSignatures call has
    // saved them and LoadAll has loaded the latter two. We still need PS to
    // be attached to the head index; EnsureTenantLoaded → EnsureHeadNodeMetaLoaded
    // handles that lazily. Just confirm and return so subsequent process
    // startups don't re-scan the entire posting file (~minutes for SIFT-1M).
    {
        struct stat st{};
        const std::string sigPath     = workDir + "/signatures_bitmask.bin";
        const std::string sparsePath  = workDir + "/sparse_tags.bin";
        const std::string tagPurePath = workDir + "/tagpure_meta.bin";
        const std::string routeStatsPath =
            workDir + "/tag_routing_stats.bin";
        const std::string headMetaPath =
            workDir + "/HeadIndex/head_node_meta.bin";
        bool sigOk     = stat(sigPath.c_str(),     &st) == 0;
        bool sparseOk  = stat(sparsePath.c_str(),  &st) == 0;
        bool tagPureOk = stat(tagPurePath.c_str(), &st) == 0;
        bool routeStatsOk =
            stat(routeStatsPath.c_str(), &st) == 0;
        bool headMetaOk =
            stat(headMetaPath.c_str(), &st) == 0;
        const bool baseArtifactsOk = staticStorage
            ? headMetaOk
            : (sigOk && sparseOk && tagPureOk);
        if (baseArtifactsOk &&
            (!hybridDistanceEnabled || routeStatsOk)) {
            // Make sure the PS signatures are attached to the head index.
            EnsureTenantLoaded(p_tenantId);
            bool headMetadataValid =
                !staticStorage;
            {
                std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
                auto it = m_tenantIndices.find(p_tenantId);
                if (it != m_tenantIndices.end()) {
                    const auto internalIndex =
                        it->second->GetInternalIndex();
                    headMetadataValid = staticStorage
                        ? LoadHeadNodeMetaFile(
                              workDir,
                              GetMemoryIndexForInternal(
                                  internalIndex))
                        : EnsureHeadNodeMetaLoaded(
                              workDir, internalIndex);
                }
            }
            // tag-pure + sparse metadata is loaded by LoadAll; touch the
            // loaders again for safety (idempotent — skips already-loaded).
            LoadTenantSparseIndices();
            LoadTenantTagPureIndices();
            if (!LoadTenantTagRoutingStats()) {
                return false;
            }
            const auto loadedStats =
                m_tenantTagRoutingStats.find(
                    p_tenantId);
            if (headMetadataValid &&
                (!hybridDistanceEnabled ||
                 (loadedStats !=
                      m_tenantTagRoutingStats.end() &&
                  !loadedStats->second.empty()))) {
                fprintf(stderr,
                        "[INFO] Tenant %d: BuildSignatures short-circuit "
                        "(static=%d headmeta=%d sig=%d sparse=%d tagpure=%d "
                        "routeStats=%d on disk)\n",
                        p_tenantId, (int)staticStorage, (int)headMetaOk,
                        (int)sigOk, (int)sparseOk,
                        (int)tagPureOk, (int)routeStatsOk);
                return true;
            }
            fprintf(stderr,
                    "[INFO] Tenant %d: rebuilding signatures because persisted "
                    "head metadata or hybrid routing stats are invalid\n",
                    p_tenantId);
        }
    }

    // Read head count from HeadIndex vectors.bin
    auto hcIt = m_tenantHeadCounts.find(p_tenantId);
    int numHeads = (hcIt != m_tenantHeadCounts.end()) ? hcIt->second : 0;
    if (numHeads <= 0) {
        std::string vecPath = workDir + "/HeadIndex/vectors.bin";
        FILE* vf = fopen(vecPath.c_str(), "rb");
        if (vf) {
            int32_t rows = 0;
            if (fread(&rows, sizeof(int32_t), 1, vf) == 1) numHeads = rows;
            fclose(vf);
        }
    }
    if (numHeads <= 0) return false;

    const auto hierWidths =
        HierWidthsFromEnv();

    // ── Routing-only fast path (SPTAG_ROUTING_ONLY=1) ───────────────────────
    // Regenerate ONLY the query-time routing sidecar (tag_node_index.bin) from
    // the loaded head index + per-vector tags, skipping the expensive posting
    // scan / sparse / tag-pure artifacts. Used to repair an index whose build
    // finished the SPANN store but crashed before the full BuildSignatures step
    // (so the per-tag -> bundle-node routing table was never written and every
    // filtered query fans out to all nodes). Honors SPTAG_PIVOT_FORCE_NODE_COUNT
    // / SPTAG_HIER_LEVEL_WIDTHS so the recomputed pivot partition matches the
    // one used at build time.
    if (std::getenv("SPTAG_ROUTING_ONLY") != nullptr) {
        EnsureTenantLoaded(p_tenantId);
        std::shared_ptr<AnnIndex> idxPtr;
        {
            std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
            auto it = m_tenantIndices.find(p_tenantId);
            if (it != m_tenantIndices.end()) idxPtr = it->second;
        }
        if (!idxPtr) return false;
        auto internalIdx = idxPtr->GetInternalIndex();
        auto memoryIndex = GetMemoryIndexForInternal(internalIdx);
        auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
        if (memoryIndex == nullptr || spannInternalIdx == nullptr) return false;

        SizeType numHeadSamples = memoryIndex->GetNumSamples();
        if (numHeadSamples < (SizeType)numHeads) numHeadSamples = (SizeType)numHeads;
        // Preserve the full signature metadata when routing is repaired after a
        // successful signature pass. Reinitializing it would discard the
        // persisted posting masks and numeric signatures before saving the
        // refreshed bundle-node assignments.
        const bool preserveHeadNodeMeta =
            memoryIndex->HasHeadNodeMeta() &&
            memoryIndex->GetHeadNodeMetaSampleCount() >= numHeadSamples;
        if (!preserveHeadNodeMeta) {
            memoryIndex->InitializeHeadNodeMeta(
                numHeadSamples, 0,
                hierWidths);
        }
        spannInternalIdx->PopulateHeadNodeGlobalVIDsFromBundles();

        // Reproduce the build-time routing-column projection so the recomputed
        // pivot partition (and thus node numbering) matches the physical layout.
        // The last SPTAG_NUMERIC_COLS columns are numeric attributes that do NOT
        // form a tag hierarchy; feeding them to the estimator fails its
        // parent-uniqueness check. SPTAG_ACL_COLS selects the categorical routing
        // columns (identity 0..numBaseCols-1 by default). Mirrors lines ~2206-2268.
        int numNumericCols = 0;
        if (const char* e = std::getenv("SPTAG_NUMERIC_COLS")) numNumericCols = atoi(e);
        if (numNumericCols < 0) numNumericCols = 0;
        if (numNumericCols > p_numTagsPerVec) numNumericCols = p_numTagsPerVec;
        const int numBaseCols = p_numTagsPerVec - numNumericCols;
        std::vector<int> routingCols;
        if (const char* e = std::getenv("SPTAG_ACL_COLS")) {
            std::stringstream ss(e);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                int c = atoi(tok.c_str());
                if (c >= 0 && c < numBaseCols) {
                    bool dup = false;
                    for (int existing : routingCols) if (existing == c) { dup = true; break; }
                    if (!dup) routingCols.push_back(c);
                }
            }
        }
        if (routingCols.empty()) {
            int numRoutingCols = numBaseCols;
            if (const char* e = std::getenv("SPTAG_ROUTING_COLS")) {
                int k = atoi(e);
                if (k > 0) numRoutingCols = (k < numBaseCols) ? k : numBaseCols;
            }
            for (int t = 0; t < numRoutingCols; ++t) routingCols.push_back(t);
        }
        const int numRoutingCols = static_cast<int>(routingCols.size());
        const uint32_t* planTags = p_tagsPtr;
        int planNumTags = p_numTagsPerVec;
        std::vector<uint32_t> catOnlyTags;
        bool identityProjection = (numRoutingCols == p_numTagsPerVec);
        if (identityProjection)
            for (int t = 0; t < numRoutingCols; ++t)
                if (routingCols[t] != t) { identityProjection = false; break; }
        if (numRoutingCols > 0 && !identityProjection) {
            catOnlyTags.resize(static_cast<size_t>(p_numVectors) * static_cast<size_t>(numRoutingCols));
            for (SizeType i = 0; i < (SizeType)p_numVectors; ++i)
                for (int t = 0; t < numRoutingCols; ++t)
                    catOnlyTags[static_cast<size_t>(i) * numRoutingCols + t] =
                        p_tagsPtr[static_cast<size_t>(i) * p_numTagsPerVec + routingCols[t]];
            planTags = catOnlyTags.data();
            planNumTags = numRoutingCols;
        }
        fprintf(stderr, "[INFO] Tenant %d: ROUTING_ONLY planning on %d categorical cols "
                "(of %d base, %d numeric)\n", p_tenantId, numRoutingCols, numBaseCols, numNumericCols);

        PivotEstimatorComputation pivotComputation;
        const PivotEstimatorCandidate* pivotCandidate = nullptr;
        if (BuildPivotEstimatorComputation(planTags, p_numVectors, planNumTags,
                                           0, 0.99, 10.0, 1.0, std::string(), pivotComputation)) {
            pivotCandidate = FindBestPivotEstimatorCandidate(pivotComputation.candidates);
        }
        if (pivotCandidate == nullptr) {
            fprintf(stderr, "[ERROR] Tenant %d: ROUTING_ONLY pivot estimation failed\n", p_tenantId);
            return false;
        }
        m_tenantPivotLevels[p_tenantId] = pivotCandidate->pivotLevel;
        m_tenantPivotNodeCounts[p_tenantId] = pivotCandidate->nodeCount;
        m_tenantNodePivotTags[p_tenantId] = pivotCandidate->nodePivotTags;
        BuildTagToNodeIndexForCandidate(*pivotCandidate, pivotComputation.levelData,
                                        m_tenantTagToNodes[p_tenantId]);
        std::vector<int> headNodeToNode;
        BuildHeadNodeToNodeIndexForCandidate(*pivotCandidate, planTags, p_numVectors,
                                             planNumTags, memoryIndex, spannInternalIdx,
                                             headNodeToNode);
        m_tenantHeadNodeToNode[p_tenantId] = headNodeToNode;
        const bool routingOk = SaveHeadNodeRoutingIndexFile(workDir, pivotCandidate->pivotLevel,
                                                            pivotCandidate->nodePivotTags,
                                                            m_tenantTagToNodes[p_tenantId],
                                                            headNodeToNode);
        bool metaOk = true;
        if (preserveHeadNodeMeta) {
            for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                const int16_t nodeId =
                    (hid < static_cast<SizeType>(headNodeToNode.size()) &&
                     headNodeToNode[hid] >= 0)
                    ? static_cast<int16_t>(headNodeToNode[hid])
                    : static_cast<int16_t>(-1);
                memoryIndex->SetHeadNodeBundleNodeId(hid, nodeId);
            }
            metaOk = SaveHeadNodeMetaFile(workDir, memoryIndex);
        }
        const bool ok = routingOk && metaOk;
        fprintf(stderr,
                "[INFO] Tenant %d: ROUTING_ONLY wrote tag_node_index.bin%s "
                "pivotLevel=%d nodeCount=%d heads=%zu tagMappings=%zu ok=%d\n",
                p_tenantId, preserveHeadNodeMeta ? " and refreshed head_node_meta.bin" : "",
                pivotCandidate->pivotLevel, pivotCandidate->nodeCount,
                headNodeToNode.size(), m_tenantTagToNodes[p_tenantId].size(), (int)ok);
        return ok;
    }

    // ── Non-FILEIO posting store (e.g. Storage=STATIC/ROCKSDBIO) fast path ──
    // When the posting store is not a FILEIO block file, the ssdinfo / ssdmapping
    // / ssdmapping_postings artifacts the scan below reads do not exist on disk.
    // The posting-derived filter sidecars (PS / hierarchical masks / sparse_tags
    // / tag-pure full-vector chunks) are unused by the OPQ search paths. The one
    // structure the unfilter cross-subgraph head search requires is head_node_meta
    // carrying each head's global VID — resolvable in-memory from the loaded head
    // index (m_vectorTranslateMap via GetGlobalVID). Generate that minimal head
    // metadata (global VID + own-tag mask + bundle routing) here, then return.
    {
        struct stat sst{};
        const bool hasFileIOPostings = stat((workDir + "/ssdmapping").c_str(), &sst) == 0;
        if (!hasFileIOPostings) {
            fprintf(stderr, "[INFO] Tenant %d: BuildSignatures non-FILEIO posting store "
                    "(no ssdmapping); generating in-memory head_node_meta.\n", p_tenantId);

            EnsureTenantLoaded(p_tenantId);
            std::shared_ptr<AnnIndex> idxPtr;
            {
                std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
                auto it = m_tenantIndices.find(p_tenantId);
                if (it != m_tenantIndices.end()) idxPtr = it->second;
            }
            if (!idxPtr) return false;
            auto internalIdx = idxPtr->GetInternalIndex();
            auto memoryIndex = GetMemoryIndexForInternal(internalIdx);
            auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
            if (memoryIndex == nullptr || spannInternalIdx == nullptr) return false;

            const auto* staticOptions = spannInternalIdx->GetOptions();
            const int staticACLTagCols =
                staticOptions != nullptr && staticOptions->m_staticACLTagCols > 0
                ? staticOptions->m_staticACLTagCols
                : p_numTagsPerVec;
            if (staticACLTagCols <= 0 || staticACLTagCols > p_numTagsPerVec) {
                fprintf(stderr, "[ERROR] Tenant %d: StaticACLTagCols=%d is invalid for %d tag columns.\n",
                        p_tenantId, staticACLTagCols, p_numTagsPerVec);
                return false;
            }
            SizeType numHeadSamples = memoryIndex->GetNumSamples();
            if (numHeadSamples < (SizeType)numHeads) numHeadSamples = (SizeType)numHeads;

            // tag_level_offsets.bin covers categorical hierarchy columns only.
            // Numeric columns are range-filter attributes, not hierarchy levels.
            {
                const int numBaseCols = staticACLTagCols;
                std::vector<uint32_t> levelMin(numBaseCols, std::numeric_limits<uint32_t>::max());
                for (int vid = 0; vid < p_numVectors; ++vid)
                    for (int t = 0; t < numBaseCols; ++t) {
                        uint32_t tag = p_tagsPtr[(size_t)vid * p_numTagsPerVec + t];
                        if (tag < levelMin[t]) levelMin[t] = tag;
                    }
                m_tenantTagLevelOffsets[p_tenantId] = levelMin;
                const std::string offPath = workDir + "/tag_level_offsets.bin";
                if (FILE* of = fopen(offPath.c_str(), "wb")) {
                    int32_t nLevels = numBaseCols;
                    fwrite(&nLevels, sizeof(int32_t), 1, of);
                    fwrite(levelMin.data(), sizeof(uint32_t), levelMin.size(), of);
                    fclose(of);
                }
            }

            // Pivot estimator (tags-only) drives per-bundle tag routing.
            const std::uint32_t* routingTags =
                p_tagsPtr;
            int routingTagCount =
                p_numTagsPerVec;
            std::vector<std::uint32_t>
                categoricalTags;
            if (staticACLTagCols <
                p_numTagsPerVec) {
                categoricalTags.resize(
                    static_cast<size_t>(
                        p_numVectors) *
                    static_cast<size_t>(
                        staticACLTagCols));
                for (int vid = 0;
                     vid < p_numVectors; ++vid) {
                    std::memcpy(
                        categoricalTags.data() +
                            static_cast<size_t>(vid) *
                                staticACLTagCols,
                        p_tagsPtr +
                            static_cast<size_t>(vid) *
                                p_numTagsPerVec,
                        static_cast<size_t>(
                            staticACLTagCols) *
                            sizeof(std::uint32_t));
                }
                routingTags =
                    categoricalTags.data();
                routingTagCount =
                    staticACLTagCols;
            }
            PivotEstimatorComputation pivotComputation;
            const PivotEstimatorCandidate* pivotCandidate = nullptr;
            if (!hybridDistanceEnabled &&
                BuildPivotEstimatorComputation(
                    routingTags, p_numVectors,
                    routingTagCount,
                                               0, 0.99, 10.0, 1.0, std::string(), pivotComputation)) {
                pivotCandidate = FindBestPivotEstimatorCandidate(pivotComputation.candidates);
            }
            if (pivotCandidate != nullptr) {
                m_tenantPivotLevels[p_tenantId] = pivotCandidate->pivotLevel;
                m_tenantPivotNodeCounts[p_tenantId] = pivotCandidate->nodeCount;
                m_tenantNodePivotTags[p_tenantId] = pivotCandidate->nodePivotTags;
                BuildTagToNodeIndexForCandidate(*pivotCandidate, pivotComputation.levelData,
                                                m_tenantTagToNodes[p_tenantId]);
            }

            memoryIndex->InitializeHeadNodeMeta(
                numHeadSamples, 0,
                hierWidths);
            // Monolithic roots resolve through their head-ID map; metadata-only
            // roots fall back to the bundle structures.
            spannInternalIdx->PopulateHeadNodeGlobalVIDsFromBundles();

            // Per-posting member-OR hierarchical mask: the dense filtered path
            // (HeadPostingHierMaskMayIntersect) keeps a posting iff at least one
            // of its member vectors MAY carry a query tag. STATIC STM1 has these
            // tags inline in each pure record; other non-FILEIO slim stores keep
            // them in opq_slim.bin. Never leave an empty mask attached to a static
            // snapshot: that would turn an enabled pre-filter into false negatives.
            std::vector<SPTAG::Cache::HierarchicalPostingMask> postingHierMasks;
            std::unordered_map<std::uint64_t, int>
                tagPostingCounts;
            {
                const std::string slimBinPath = workDir + "/opq_slim.bin";
                const std::string slimIdxPath = workDir + "/opq_slim.idx";
                const std::string staticSnapshotPath = workDir + "/SPTAGFullList.bin";
                struct stat staticSnapshotStat {};
                if (stat(staticSnapshotPath.c_str(), &staticSnapshotStat) == 0) {
                    if (!BuildStaticPostingHierMasks(staticSnapshotPath, numHeads, p_numTagsPerVec,
                                                      staticACLTagCols, hierWidths,
                                                      postingHierMasks,
                                                      &tagPostingCounts)) {
                        return false;
                    }
                    fprintf(stderr, "[INFO] Tenant %d: built posting hier masks for %zu STM1 postings.\n",
                            p_tenantId, postingHierMasks.size());
                } else {
                    std::ifstream idxIn(slimIdxPath, std::ios::binary);
                    std::ifstream binIn(slimBinPath, std::ios::binary);
                    if (idxIn && binIn) {
                    idxIn.seekg(0, std::ios::end);
                    std::streamoff idxBytes = idxIn.tellg();
                    idxIn.seekg(0, std::ios::beg);
                    size_t idxEntries = (idxBytes > 0) ? (size_t)idxBytes / sizeof(std::uint64_t) : 0;
                    if (idxEntries >= 2) {
                        std::vector<std::uint64_t> off(idxEntries);
                        idxIn.read(reinterpret_cast<char*>(off.data()), (std::streamsize)idxEntries * sizeof(std::uint64_t));
                        size_t numSlimPostings = idxEntries - 1;
                        binIn.seekg(0, std::ios::end);
                        std::streamoff slimBytes = binIn.tellg();
                        binIn.seekg(0, std::ios::beg);
                        std::vector<std::uint8_t> slim((size_t)std::max<std::streamoff>(slimBytes, 0));
                        if (slimBytes > 0)
                            binIn.read(reinterpret_cast<char*>(slim.data()), slimBytes);
                        const int recSize = (int)(sizeof(int) + sizeof(std::uint8_t)
                                                  + (size_t)p_numTagsPerVec * sizeof(uint32_t));
                        const int tagByteOff = (int)(sizeof(int) + sizeof(std::uint8_t));
                        postingHierMasks.assign(numSlimPostings, SPTAG::Cache::HierarchicalPostingMask());
                        for (size_t h = 0; h < numSlimPostings; ++h) {
                            postingHierMasks[h].Clear();
                            std::uint64_t o0 = off[h], o1 = off[h + 1];
                            if (o1 <= o0 || o1 > slim.size()) continue;
                            size_t span = (size_t)(o1 - o0);
                            size_t n = span / (size_t)recSize;
                            const std::uint8_t* base = slim.data() + o0;
                            for (size_t i = 0; i < n; ++i) {
                                const std::uint8_t* e = base + i * (size_t)recSize;
                                const uint32_t* vt = reinterpret_cast<const uint32_t*>(e + tagByteOff);
                                for (int t = 0; t < p_numTagsPerVec; ++t)
                                    postingHierMasks[h].Insert(
                                        t, vt[t],
                                        hierWidths);
                            }
                        }
                        fprintf(stderr, "[INFO] Tenant %d: built posting hier masks for %zu slim postings.\n",
                                p_tenantId, numSlimPostings);
                    }
                    } else {
                        fprintf(stderr, "[WARN] Tenant %d: opq_slim.bin/.idx missing; dense filtered "
                                "posting pre-filter will be empty.\n", p_tenantId);
                    }
                }
            }

            const int numNumericCols =
                p_numTagsPerVec - staticACLTagCols;
            if (numNumericCols > 0) {
                std::vector<SPTAG::Cache::NumQuantParam>
                    quantParams(
                        static_cast<size_t>(
                            numNumericCols));
                for (auto& parameter : quantParams) {
                    parameter.lo =
                        (std::numeric_limits<
                            std::uint32_t>::max)();
                    parameter.hi = 0;
                }
                for (int vid = 0; vid < p_numVectors; ++vid) {
                    for (int column = 0;
                         column < numNumericCols;
                         ++column) {
                        const std::uint32_t value =
                            p_tagsPtr[
                                static_cast<size_t>(vid) *
                                    p_numTagsPerVec +
                                staticACLTagCols +
                                column];
                        auto& parameter =
                            quantParams[
                                static_cast<size_t>(
                                    column)];
                        parameter.lo =
                            (std::min)(
                                parameter.lo, value);
                        parameter.hi =
                            (std::max)(
                                parameter.hi, value);
                    }
                }
                const std::string numericPath =
                    workDir + "/numeric_meta.bin";
                FILE* numericFile =
                    fopen(numericPath.c_str(), "wb");
                if (numericFile == nullptr) {
                    return false;
                }
                const std::int32_t magic =
                    0x54454d4e;
                const std::int32_t base =
                    staticACLTagCols;
                const std::int32_t count =
                    numNumericCols;
                bool numericOk =
                    fwrite(
                        &magic, sizeof(magic), 1,
                        numericFile) == 1 &&
                    fwrite(
                        &base, sizeof(base), 1,
                        numericFile) == 1 &&
                    fwrite(
                        &count, sizeof(count), 1,
                        numericFile) == 1;
                for (const auto& parameter :
                     quantParams) {
                    numericOk =
                        numericOk &&
                        fwrite(
                            &parameter.lo,
                            sizeof(parameter.lo), 1,
                            numericFile) == 1 &&
                        fwrite(
                            &parameter.hi,
                            sizeof(parameter.hi), 1,
                            numericFile) == 1;
                }
                numericOk =
                    fclose(numericFile) == 0 &&
                    numericOk;
                if (!numericOk) return false;
                m_tenantNumericMeta[p_tenantId] = {
                    staticACLTagCols,
                    std::move(quantParams)};
            }

            std::unordered_map<std::uint64_t, int>
                tagVectorCounts;
            tagVectorCounts.reserve(
                (std::min)(
                    static_cast<size_t>(
                        p_numVectors) *
                        static_cast<size_t>(
                            staticACLTagCols),
                    static_cast<size_t>(1 << 20)));
            for (int vid = 0; vid < p_numVectors;
                 ++vid) {
                std::vector<std::uint32_t>
                    rawTags;
                rawTags.reserve(
                    static_cast<size_t>(
                        staticACLTagCols));
                for (int column = 0;
                     column < staticACLTagCols;
                     ++column) {
                    const std::uint32_t tag =
                        p_tagsPtr[
                            static_cast<size_t>(vid) *
                                p_numTagsPerVec +
                            column];
                    ++tagVectorCounts[
                        MakeTagRoutingKey(
                            static_cast<std::uint32_t>(
                                column),
                            tag)];
                    rawTags.push_back(tag);
                }
                std::sort(
                    rawTags.begin(), rawTags.end());
                rawTags.erase(
                    std::unique(
                        rawTags.begin(),
                        rawTags.end()),
                    rawTags.end());
                for (std::uint32_t tag : rawTags) {
                    ++tagVectorCounts[
                        MakeTagRoutingKey(
                            kLegacyRoutingColumn,
                            tag)];
                }
            }
            auto& routeStats =
                m_tenantTagRoutingStats[p_tenantId];
            routeStats.clear();
            routeStats.reserve(
                tagVectorCounts.size());
            for (const auto& entry :
                 tagVectorCounts) {
                routeStats[entry.first] = {
                    entry.second,
                    tagPostingCounts[entry.first]};
            }
            std::uint64_t routeGeneration = 0;
            if (hybridDistanceEnabled) {
                SPTAG::SPANN::HybridRoutingStatsHeader
                    hybridHeader;
                if (!LoadHybridRoutingStatsHeader(
                        workDir + "/" +
                            primaryPostingFile +
                            ".hybrid.stats",
                        hybridHeader)) {
                    fprintf(
                        stderr,
                        "[ERROR] Tenant %d: cannot bind "
                        "tag routing stats to the primary "
                        "hybrid posting\n",
                        p_tenantId);
                    return false;
                }
                routeGeneration =
                    hybridHeader
                        .m_generationFingerprint;
            }
            if (!SaveTagRoutingStatsFile(
                    workDir +
                        "/tag_routing_stats.bin",
                    p_numVectors, routeGeneration,
                    routeStats)) {
                return false;
            }

            int resolved = 0;
            for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                SizeType globalVID = memoryIndex->GetHeadNodeGlobalVID(hid);
                memoryIndex->SetHeadNodeHeadOnly(hid, true);
                // Posting-content mask (member-OR) drives the dense-path posting
                // pre-filter; set it independently of the head's own resolved VID.
                if ((size_t)hid < postingHierMasks.size())
                    memoryIndex->SetHeadNodePostingHierMask(hid, postingHierMasks[hid]);
                if (globalVID == SPTAG::MaxSize || globalVID >= (SizeType)p_numVectors) continue;
                ++resolved;
                SPTAG::Cache::HierarchicalOwnTags ownMask;
                ownMask.Clear();
                for (int t = 0; t < staticACLTagCols; ++t) {
                    uint32_t tag = p_tagsPtr[(size_t)globalVID * p_numTagsPerVec + t];
                    ownMask.Insert(t, tag);
                }
                memoryIndex->SetHeadNodeHierMask(hid, ownMask);
            }

            if (pivotCandidate != nullptr) {
                std::vector<int> headNodeToNode;
                BuildHeadNodeToNodeIndexForCandidate(*pivotCandidate, routingTags, p_numVectors,
                                                     routingTagCount, memoryIndex, spannInternalIdx,
                                                     headNodeToNode);
                m_tenantHeadNodeToNode[p_tenantId] = headNodeToNode;
                for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                    int16_t nid = (hid < (SizeType)headNodeToNode.size() && headNodeToNode[hid] >= 0)
                                  ? (int16_t)headNodeToNode[hid] : (int16_t)-1;
                    memoryIndex->SetHeadNodeBundleNodeId(hid, nid);
                }
                SaveHeadNodeRoutingIndexFile(workDir, pivotCandidate->pivotLevel,
                                             pivotCandidate->nodePivotTags,
                                             m_tenantTagToNodes[p_tenantId], headNodeToNode);
            } else if (hybridDistanceEnabled) {
                std::vector<int> headNodeToNode(
                    static_cast<size_t>(
                        numHeadSamples),
                    0);
                m_tenantPivotLevels[p_tenantId] =
                    -1;
                m_tenantPivotNodeCounts[p_tenantId] =
                    1;
                m_tenantNodePivotTags[p_tenantId] = {
                    std::vector<std::uint32_t>()};
                auto& tagToNodes =
                    m_tenantTagToNodes[p_tenantId];
                tagToNodes.clear();
                for (int vid = 0;
                     vid < p_numVectors; ++vid) {
                    for (int column = 0;
                         column < staticACLTagCols;
                         ++column) {
                        tagToNodes[
                            p_tagsPtr[
                                static_cast<size_t>(
                                    vid) *
                                    p_numTagsPerVec +
                                column]] = {0};
                    }
                }
                m_tenantHeadNodeToNode[
                    p_tenantId] =
                    headNodeToNode;
                for (SizeType hid = 0;
                     hid < numHeadSamples;
                     ++hid) {
                    memoryIndex->
                        SetHeadNodeBundleNodeId(
                            hid, 0);
                }
                SaveHeadNodeRoutingIndexFile(
                    workDir, -1,
                    m_tenantNodePivotTags[
                        p_tenantId],
                    tagToNodes,
                    headNodeToNode);
            }
            SaveHeadNodeMetaFile(workDir, memoryIndex);
            fprintf(stderr, "[INFO] Tenant %d: head_node_meta generated for %d heads "
                    "(%d resolved) [non-FILEIO].\n",
                    p_tenantId, (int)numHeadSamples, resolved);
            return true;
        }
    }

    // ── Read real posting→vector assignment from SPANN on-disk data ──
    // Files: ssdinfo (posting sizes), ssdmapping (block addresses), ssdmapping_postings (block data)
    // Posting format per vector: [VID(4B) | Version(1B) | Tags(N*4B) | VectorData(dim*sizeof(T))]
    const int PAGE_SIZE = 4096;
    const int TAG_BYTES = p_numTagsPerVec * (int)sizeof(uint32_t);
    const int META_SIZE = sizeof(int32_t) + sizeof(uint8_t) + TAG_BYTES;
    const int FULL_VEC_INFO_SIZE = m_inputVectorSize + META_SIZE;

    // 1. Read ssdinfo: header (rows, cols=1) then rows × int32 posting sizes
    std::string ssdinfoPath = workDir + "/ssdinfo";
    FILE* infoF = fopen(ssdinfoPath.c_str(), "rb");
    if (!infoF) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", ssdinfoPath.c_str());
        return false;
    }
    int32_t infoHeader[2];
    if (fread(infoHeader, sizeof(int32_t), 2, infoF) != 2) { fclose(infoF); return false; }
    int numPostings = infoHeader[0];
    std::vector<int32_t> postingSizes(numPostings);
    if ((int)fread(postingSizes.data(), sizeof(int32_t), numPostings, infoF) != numPostings) {
        fclose(infoF); return false;
    }
    fclose(infoF);

    // 1b. Load posting_pure_counts.bin (if present). Each posting is laid out as
    //     [pure prefix | unfilter-tail suffix]. The unfilter-tail vectors exist
    //     ONLY to boost unfiltered recall and are never scanned by filtered
    //     queries (runtime caps the read to pure_count * vectorInfoSize). They
    //     must therefore NOT contribute their tags to the per-head tag filter
    //     sidecars (signatures_bitmask / posting_hier_masks / sparse_tags),
    //     otherwise filtered queries lose their pruning and diverge from a
    //     build without U_extra. Same file format as ssdinfo (header + int32s).
    std::vector<int32_t> postingPureCounts;
    bool hasPureCounts = false;
    {
        std::string pcPath = workDir + "/posting_pure_counts.bin";
        FILE* pcF = fopen(pcPath.c_str(), "rb");
        if (pcF) {
            int32_t pcHeader[2];
            if (fread(pcHeader, sizeof(int32_t), 2, pcF) == 2 && pcHeader[0] == numPostings) {
                postingPureCounts.resize(numPostings);
                if ((int)fread(postingPureCounts.data(), sizeof(int32_t), numPostings, pcF)
                    == numPostings) {
                    hasPureCounts = true;
                }
            }
            fclose(pcF);
        }
        if (hasPureCounts) {
            fprintf(stderr, "[INFO] Tenant %d: pure-count sidecar present; building tag filter "
                    "masks over pure prefix only (excluding unfilter-tail).\n", p_tenantId);
        }
    }
    auto pureLimit = [&](int pid, int nVecs) -> int {
        if (!hasPureCounts || pid < 0 || pid >= (int)postingPureCounts.size()) return nVecs;
        int pc = postingPureCounts[pid];
        return (pc < 0 || pc > nVecs) ? nVecs : pc;
    };

    // 2. Read ssdmapping: header (rows, cols) then rows × cols × int64 block addresses
    //    addrs[pid][0] = data size in bytes, addrs[pid][1..] = block addresses
    std::string mappingPath = workDir + "/ssdmapping";
    FILE* mapF = fopen(mappingPath.c_str(), "rb");
    if (!mapF) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", mappingPath.c_str());
        return false;
    }
    int32_t mapHeader[2];
    if (fread(mapHeader, sizeof(int32_t), 2, mapF) != 2) { fclose(mapF); return false; }
    int mapRows = mapHeader[0], mapCols = mapHeader[1];
    std::vector<int64_t> addrFlat((size_t)mapRows * mapCols);
    if ((int)fread(addrFlat.data(), sizeof(int64_t), (size_t)mapRows * mapCols, mapF)
        != mapRows * mapCols) {
        fclose(mapF); return false;
    }
    fclose(mapF);

    // 3. Open posting data file
    std::string postingPath = workDir + "/ssdmapping_postings";
    FILE* postF = fopen(postingPath.c_str(), "rb");
    if (!postF) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", postingPath.c_str());
        return false;
    }

    // 4. For each posting, read its blocks and extract vector IDs
    std::vector<std::vector<uint32_t>> posting_tags(numHeads);
    std::vector<SPTAG::Cache::HierarchicalPostingMask> posting_hier_masks(numHeads);
    std::unordered_map<std::uint64_t, int>
        routingPostingCounts;
    uint64_t totalAssignments = 0;
    std::vector<uint8_t> blockBuf(PAGE_SIZE);
    auto postingRecordStride = [&](const int64_t* rowAddrs, int nVecs) -> int {
        if (nVecs > 0 && rowAddrs[0] > 0 && rowAddrs[0] % nVecs == 0) {
            const int64_t stride = rowAddrs[0] / nVecs;
            if (stride >= META_SIZE && stride <= std::numeric_limits<int>::max()) {
                return static_cast<int>(stride);
            }
        }
        return FULL_VEC_INFO_SIZE;
    };

    // ── Numeric attribute setup (quantized signature) ──────────────────────
    // The last `numNumericCols` tag columns are numeric attributes: the RAW value
    // is stored inline per vector (read by the exact filter) and a quantized
    // bucket is OR-ed into the per-posting numeric signature for range pruning.
    // numNumericCols=0 => pure categorical (layout/behavior unchanged).
    int numNumericCols = 0;
    if (const char* e = std::getenv("SPTAG_NUMERIC_COLS")) numNumericCols = atoi(e);
    if (numNumericCols < 0) numNumericCols = 0;
    if (numNumericCols > p_numTagsPerVec) numNumericCols = p_numTagsPerVec;
    const int numBaseCols = p_numTagsPerVec - numNumericCols;
    std::vector<SPTAG::Cache::NumQuantParam> quantParams(numNumericCols);
    if (numNumericCols > 0) {
        for (int c = 0; c < numNumericCols; ++c) { quantParams[c].lo = UINT32_MAX; quantParams[c].hi = 0; }
        for (int vid = 0; vid < p_numVectors; ++vid)
            for (int c = 0; c < numNumericCols; ++c) {
                uint32_t v = p_tagsPtr[(size_t)vid * p_numTagsPerVec + numBaseCols + c];
                if (v < quantParams[c].lo) quantParams[c].lo = v;
                if (v > quantParams[c].hi) quantParams[c].hi = v;
            }
        // Persist quant metadata so query processes can reconstruct the mapping.
        const std::string nmPath = workDir + "/numeric_meta.bin";
        if (FILE* nf = fopen(nmPath.c_str(), "wb")) {
            int32_t magic = 0x54454d4e;  // 'NMET'
            int32_t base = numBaseCols, ncols = numNumericCols;
            fwrite(&magic, sizeof(int32_t), 1, nf);
            fwrite(&base, sizeof(int32_t), 1, nf);
            fwrite(&ncols, sizeof(int32_t), 1, nf);
            for (int c = 0; c < numNumericCols; ++c) {
                fwrite(&quantParams[c].lo, sizeof(uint32_t), 1, nf);
                fwrite(&quantParams[c].hi, sizeof(uint32_t), 1, nf);
            }
            fclose(nf);
            fprintf(stderr, "[INFO] Tenant %d: wrote numeric_meta.bin (base=%d numeric=%d)\n",
                    p_tenantId, numBaseCols, numNumericCols);
        }
        m_tenantNumericMeta[p_tenantId] = {numBaseCols, quantParams};
    }
    std::vector<uint64_t> posting_num_quant(
        (size_t)numHeads * (size_t)numNumericCols * SPTAG::Cache::NUM_QUANT_WORDS, 0);

    // Tag-pure path: collect first-occurrence vector data per VID so we can
    // materialize per-tag dense lists for very-sparse tags after the loop.
    // Only enabled for Float value type (m_inputVectorSize == dim * 4).
    const bool kTagPureEligible = (m_valueType == SPTAG::VectorValueType::Float)
        && (m_inputVectorSize == (size_t)m_dimension * sizeof(float));
    std::vector<uint8_t> vidSeen;            // 0/1 per VID
    std::vector<float>   vidVecData;         // p_numVectors * dim, row-major
    if (kTagPureEligible) {
        vidSeen.assign(p_numVectors, 0);
        vidVecData.assign((size_t)p_numVectors * (size_t)m_dimension, 0.0f);
    }

    for (int pid = 0; pid < std::min(numPostings, numHeads); pid++) {
        int nVecs = postingSizes[pid];
        if (nVecs <= 0) continue;

        // Gather block addresses (skip index 0 which is data size)
        int64_t* rowAddrs = addrFlat.data() + (int64_t)pid * mapCols;
        // rowAddrs[0] = data size, rowAddrs[1..] = block addresses

        // Read blocks into contiguous buffer
        const int recordStride = postingRecordStride(rowAddrs, nVecs);
        int dataSize = nVecs * recordStride;
        std::vector<uint8_t> raw;
        raw.reserve(dataSize + PAGE_SIZE);
        for (int b = 1; b < mapCols; b++) {
            int64_t blkAddr = rowAddrs[b];
            if (blkAddr < 0) break;  // -1 marks end of block list; 0 is valid
            fseek(postF, blkAddr * PAGE_SIZE, SEEK_SET);
            raw.resize(raw.size() + PAGE_SIZE);
            size_t readBytes = fread(raw.data() + raw.size() - PAGE_SIZE, 1, PAGE_SIZE, postF);
            (void)readBytes;
        }

        if ((int)raw.size() < dataSize) continue;

        // Extract VIDs and map to tags
        const int nPure = pureLimit(pid, nVecs);
        std::vector<std::uint64_t>
            postingRoutingKeys;
        for (int j = 0; j < nVecs; j++) {
            int offset = j * recordStride;
            int32_t vid;
            memcpy(&vid, raw.data() + offset, sizeof(int32_t));
            if (vid < 0 || vid >= p_numVectors) continue;
            // Only the pure prefix drives the filter sidecars. Unfilter-tail
            // vectors (j >= nPure) must not pollute the per-head tag masks.
            if (j < nPure) {
                // Categorical tags (cols 0..numBaseCols-1) -> Bloom + hier mask.
                for (int t = 0; t < numBaseCols; t++) {
                    uint32_t tag = p_tagsPtr[vid * p_numTagsPerVec + t];
                    posting_tags[pid].push_back(tag);
                    postingRoutingKeys.push_back(
                        MakeTagRoutingKey(
                            static_cast<std::uint32_t>(
                                t),
                            tag));
                    postingRoutingKeys.push_back(
                        MakeTagRoutingKey(
                            kLegacyRoutingColumn,
                            tag));
                    // Also insert into hierarchical mask at level t
                    posting_hier_masks[pid].Insert(
                        t, tag, hierWidths);
                }
                // Numeric attributes (cols numBaseCols..) -> quantized signature.
                for (int c = 0; c < numNumericCols; c++) {
                    uint32_t v = p_tagsPtr[vid * p_numTagsPerVec + numBaseCols + c];
                    int b = SPTAG::Cache::NumQuantBucket(quantParams[c], v);
                    SPTAG::Cache::NumQuantInsert(
                        posting_num_quant.data() + (size_t)pid * numNumericCols * SPTAG::Cache::NUM_QUANT_WORDS,
                        c, b);
                }
                totalAssignments++;
            }

            // First-occurrence capture of vector payload for tag-pure path.
            // (Captured for any occurrence; tail copies are byte-identical to the
            // pure home copy, so coverage is unaffected by the pure gate above.)
            if (kTagPureEligible && !vidSeen[vid]) {
                const uint8_t* src = raw.data() + offset + META_SIZE;
                std::memcpy(vidVecData.data() + (size_t)vid * (size_t)m_dimension,
                            src, m_inputVectorSize);
                vidSeen[vid] = 1;
            }
        }
        std::sort(
            postingRoutingKeys.begin(),
            postingRoutingKeys.end());
        postingRoutingKeys.erase(
            std::unique(
                postingRoutingKeys.begin(),
                postingRoutingKeys.end()),
            postingRoutingKeys.end());
        for (std::uint64_t key :
             postingRoutingKeys) {
            ++routingPostingCounts[key];
        }
    }
    fclose(postF);

    auto sigs = std::make_shared<SPTAG::Cache::TenantBitmaskPS>();
    sigs->Build(numHeads, posting_tags);

    std::string sigPath = workDir + "/signatures_bitmask.bin";
    sigs->Save(sigPath);

    // Compute hierarchy-level tag offsets from categorical columns only. Numeric
    // attributes may have low values (including zero) and must not be treated as
    // a later hierarchy level when classifying a categorical query tag.
    // The hierarchical posting mask stores each tag at level == its column
    // index, so the query side must map a raw tag value back to its level.
    // Tag value ranges are disjoint per level, so the per-column minimum is a
    // valid level boundary. Persist these so query processes can load them.
    {
        std::vector<uint32_t> levelMin(numBaseCols, std::numeric_limits<uint32_t>::max());
        for (int vid = 0; vid < p_numVectors; ++vid) {
            for (int t = 0; t < numBaseCols; ++t) {
                uint32_t tag = p_tagsPtr[vid * p_numTagsPerVec + t];
                if (tag < levelMin[t]) levelMin[t] = tag;
            }
        }
        m_tenantTagLevelOffsets[p_tenantId] = levelMin;
        const std::string offPath = workDir + "/tag_level_offsets.bin";
        if (FILE* of = fopen(offPath.c_str(), "wb")) {
            int32_t nLevels = numBaseCols;
            fwrite(&nLevels, sizeof(int32_t), 1, of);
            fwrite(levelMin.data(), sizeof(uint32_t), levelMin.size(), of);
            fclose(of);
            fprintf(stderr, "[INFO] Tenant %d: wrote tag_level_offsets.bin (%d levels)\n",
                    p_tenantId, nLevels);
        }
    }

    std::unordered_map<std::uint64_t, int> tagVectorCounts;
    // The tag vocabulary is usually tiny compared with the vector count (340
    // for the SIFT/SPACEV hierarchy). Reserving p_numVectors * numTags would
    // allocate billions of unused hash buckets at 1B scale.
    tagVectorCounts.reserve(std::min<size_t>(
        static_cast<size_t>(p_numVectors) * static_cast<size_t>(numBaseCols),
        static_cast<size_t>(1 << 20)));
    for (int vid = 0; vid < p_numVectors; ++vid) {
        std::vector<std::uint32_t> rawTags;
        rawTags.reserve(
            static_cast<size_t>(
                numBaseCols));
        for (int t = 0; t < numBaseCols; ++t) {
            uint32_t tag = p_tagsPtr[vid * p_numTagsPerVec + t];
            ++tagVectorCounts[
                MakeTagRoutingKey(
                    static_cast<std::uint32_t>(t),
                    tag)];
            rawTags.push_back(tag);
        }
        std::sort(rawTags.begin(), rawTags.end());
        rawTags.erase(
            std::unique(
                rawTags.begin(), rawTags.end()),
            rawTags.end());
        for (std::uint32_t tag : rawTags) {
            ++tagVectorCounts[
                MakeTagRoutingKey(
                    kLegacyRoutingColumn,
                    tag)];
        }
    }

    std::unordered_map<uint32_t, int> tagPostingCounts;
    tagPostingCounts.reserve(tagVectorCounts.size());
    for (int pid = 0; pid < numHeads; ++pid) {
        std::unordered_set<uint32_t> seenTags;
        for (uint32_t tag : posting_tags[pid]) {
            if (seenTags.insert(tag).second) {
                ++tagPostingCounts[tag];
            }
        }
    }

    auto& routeStats = m_tenantTagRoutingStats[p_tenantId];
    routeStats.clear();
    routeStats.reserve(tagVectorCounts.size());
    for (const auto& [key, vectorCount] : tagVectorCounts) {
        routeStats[key] = TagRoutingStats{
            vectorCount,
            routingPostingCounts[key]};
    }
    std::uint64_t routeGeneration = 0;
    if (hybridDistanceEnabled) {
        SPTAG::SPANN::HybridRoutingStatsHeader
            hybridHeader;
        if (!LoadHybridRoutingStatsHeader(
                workDir + "/" +
                    primaryPostingFile +
                    ".hybrid.stats",
                hybridHeader)) {
            fprintf(
                stderr,
                "[ERROR] Tenant %d: cannot bind tag routing stats to "
                "the primary hybrid posting\n",
                p_tenantId);
            return false;
        }
        routeGeneration =
            hybridHeader.m_generationFingerprint;
    }
    if (!SaveTagRoutingStatsFile(
            workDir + "/tag_routing_stats.bin",
            p_numVectors, routeGeneration,
            routeStats)) {
        fprintf(stderr,
                "[ERROR] Tenant %d: failed to persist tag_routing_stats.bin\n",
                p_tenantId);
        return false;
    }

    // Sparse-path single native knob: [BuildSSDIndex]
    // DirectSparseMaxPostings=N. A tag is materialized into sparse_tags.bin iff
    // it appears in <= N postings.
    // At query time, materialized tags ALWAYS route through the sparse path
    // (no second-stage union-size gate) - this is the single fixed threshold.
    auto sparseIdx = std::make_shared<SPTAG::Cache::SparseTagIndex>();
    sparseIdx->Build(numHeads, posting_tags, tagPostingCounts, directSparseMaxPostings);

    std::string sparsePath = workDir + "/sparse_tags.bin";
    sparseIdx->Save(sparsePath);
    m_tenantSparseIdx[p_tenantId] = sparseIdx;

    // (Tag-pure postings are built after the head-iteration loop below, once
    // both posting-side and head-side vector data have been captured into
    // vidVecData / vidSeen.)

    constexpr int kPivotEstimatorDefaultMaxNodes = 0;
    constexpr double kPivotEstimatorDefaultRecallTarget = 0.99;
    constexpr double kPivotEstimatorDefaultLambdaRecall = 10.0;
    constexpr double kPivotEstimatorDefaultEstimatedRecall = 1.0;

    PivotEstimatorComputation pivotComputation;
    const PivotEstimatorCandidate* pivotCandidate = nullptr;
    if (BuildPivotEstimatorComputation(p_tagsPtr,
                                       p_numVectors,
                                       p_numTagsPerVec,
                                       kPivotEstimatorDefaultMaxNodes,
                                       kPivotEstimatorDefaultRecallTarget,
                                       kPivotEstimatorDefaultLambdaRecall,
                                       kPivotEstimatorDefaultEstimatedRecall,
                                       std::string(),
                                       pivotComputation)) {
        pivotCandidate = FindBestPivotEstimatorCandidate(pivotComputation.candidates);
    }

    if (pivotCandidate != nullptr) {
        m_tenantPivotLevels[p_tenantId] = pivotCandidate->pivotLevel;
        m_tenantPivotNodeCounts[p_tenantId] = pivotCandidate->nodeCount;
        m_tenantNodePivotTags[p_tenantId] = pivotCandidate->nodePivotTags;
        BuildTagToNodeIndexForCandidate(*pivotCandidate,
                                        pivotComputation.levelData,
                                        m_tenantTagToNodes[p_tenantId]);
    } else {
        m_tenantPivotLevels.erase(p_tenantId);
        m_tenantPivotNodeCounts.erase(p_tenantId);
        m_tenantNodePivotTags.erase(p_tenantId);
        m_tenantTagToNodes.erase(p_tenantId);
        m_tenantHeadNodeToNode.erase(p_tenantId);
    }

    // Build head tag table: VIDs NOT found in any posting are head vectors.
    // They need tag metadata for filtered search since inline tag filter
    // can't check them (they're not in posting data).
    std::unordered_set<int> postingVIDs;
    // Re-derive from posting_tags: impossible to get VIDs from tags alone.
    // Instead, use the totalAssignments count: if a VID was found in postings,
    // it contributed to posting_tags. Just re-scan the posting file.
    {
        FILE* pf2 = fopen(postingPath.c_str(), "rb");
        if (pf2) {
            for (int pid = 0; pid < std::min(numPostings, numHeads); pid++) {
                int nVecs = postingSizes[pid];
                if (nVecs <= 0) continue;
                int64_t* rowAddrs2 = addrFlat.data() + (int64_t)pid * mapCols;
                std::vector<uint8_t> raw2;
                const int recordStride2 = postingRecordStride(rowAddrs2, nVecs);
                raw2.reserve(nVecs * recordStride2 + PAGE_SIZE);
                for (int b = 1; b < mapCols; b++) {
                    int64_t blkAddr = rowAddrs2[b];
                    if (blkAddr < 0) break;
                    fseek(pf2, blkAddr * PAGE_SIZE, SEEK_SET);
                    raw2.resize(raw2.size() + PAGE_SIZE);
                    size_t r = fread(raw2.data() + raw2.size() - PAGE_SIZE, 1, PAGE_SIZE, pf2);
                    (void)r;
                }
                for (int j = 0; j < nVecs && j * recordStride2 + 4 <= (int)raw2.size(); j++) {
                    int32_t vid;
                    memcpy(&vid, raw2.data() + j * recordStride2, sizeof(int32_t));
                    if (vid >= 0 && vid < p_numVectors) postingVIDs.insert(vid);
                }
            }
            fclose(pf2);
        }
    }

    // Store per-head-node metadata on the inner head index (if loaded).
    // First ensure the tenant is loaded.
    EnsureTenantLoaded(p_tenantId);
    std::shared_ptr<AnnIndex> idxPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it != m_tenantIndices.end()) idxPtr = it->second;
    }
    int headTagCount = 0;
    if (idxPtr) {
        auto internalIdx = idxPtr->GetInternalIndex();
        auto memoryIndex = GetMemoryIndexForInternal(internalIdx);
        auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
        if (memoryIndex != nullptr && spannInternalIdx != nullptr) {
            const SizeType numHeadSamples = memoryIndex->GetNumSamples();
            memoryIndex->InitializeHeadNodeMeta(
                numHeadSamples, numNumericCols,
                hierWidths);
            for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                SizeType globalVID = spannInternalIdx->GetGlobalVID(hid);
                memoryIndex->SetHeadNodeGlobalVID(hid, globalVID);
                if (hid < sigs->num_postings) {
                    memoryIndex->SetHeadNodePS(hid, sigs->ps[hid]);
                }

                // Posting-content mask: union of all member-vector tags in the
                // head's posting. Drives the dense-path posting pre-filter so
                // that a posting is kept iff at least one of its member vectors
                // MAY carry a tag matching the query (no-false-negative filter).
                SPTAG::Cache::HierarchicalPostingMask postingMask;
                postingMask.Clear();
                if (hid < (SizeType)posting_hier_masks.size()) {
                    postingMask = posting_hier_masks[hid];
                }
                memoryIndex->SetHeadNodePostingHierMask(hid, postingMask);

                // Quantized numeric posting signature: union of member buckets per
                // numeric column. Drives the range pre-filter (MayMatchHierQuant).
                if (numNumericCols > 0 && hid < (SizeType)numHeads) {
                    std::uint64_t* dst = memoryIndex->GetHeadNodeNumQuantMutable(hid);
                    if (dst != nullptr) {
                        std::memcpy(dst,
                            posting_num_quant.data() + (size_t)hid * numNumericCols * SPTAG::Cache::NUM_QUANT_WORDS,
                            (size_t)numNumericCols * SPTAG::Cache::NUM_QUANT_WORDS * sizeof(std::uint64_t));
                    }
                }

                if (globalVID == SPTAG::MaxSize || globalVID >= static_cast<SizeType>(p_numVectors)) {
                    continue;
                }

                // Tag-pure: head VIDs are NOT in any posting; capture their
                // vector data here so the sparse fast path covers them too.
                if (kTagPureEligible
                    && globalVID >= 0 && globalVID < static_cast<SizeType>(p_numVectors)
                    && !vidSeen[globalVID]) {
                    const void* hv = memoryIndex->GetSample(hid);
                    if (hv != nullptr) {
                        std::memcpy(vidVecData.data() + (size_t)globalVID * (size_t)m_dimension,
                                    hv, m_inputVectorSize);
                        vidSeen[globalVID] = 1;
                    }
                }

                // Own-tags mask: a single-vector mask reflecting THIS head
                // centroid's own tags. Used by HeadNodeMatchesQuery to gate
                // whether the head's centroid is admissible as a top-K result
                // (head centroids are real dataset vectors but are never stored
                // as members of any posting, so without this gate they would
                // either be lost or leaked regardless of their own tag).
                SPTAG::Cache::HierarchicalOwnTags ownMask;
                ownMask.Clear();
                for (int t = 0; t < p_numTagsPerVec; ++t) {
                    uint32_t tag = p_tagsPtr[static_cast<size_t>(globalVID) * static_cast<size_t>(p_numTagsPerVec) + static_cast<size_t>(t)];
                    ownMask.Insert(t, tag);
                }
                memoryIndex->SetHeadNodeHierMask(hid, ownMask);

                // Heads are never stored as members of any posting in this
                // SPANN build; the only way for a head centroid vector to be
                // returned as a search result is via the head-as-top-K path
                // (guarded by HeadNodeMatchesQuery's IsHeadNodeHeadOnly check).
                // Mark all heads accordingly so we don't silently lose ~25% of
                // the dataset to ACL queries.
                memoryIndex->SetHeadNodeHeadOnly(hid, true);
                headTagCount++;
            }

            if (pivotCandidate != nullptr) {
                std::vector<int> headNodeToNode;
                BuildHeadNodeToNodeIndexForCandidate(*pivotCandidate,
                                                     p_tagsPtr,
                                                     p_numVectors,
                                                     p_numTagsPerVec,
                                                     memoryIndex,
                                                     spannInternalIdx,
                                                     headNodeToNode);
                m_tenantHeadNodeToNode[p_tenantId] = headNodeToNode;

                // Populate bundleNodeId for each head
                for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                    int16_t nid = (hid < (SizeType)headNodeToNode.size() && headNodeToNode[hid] >= 0)
                                  ? (int16_t)headNodeToNode[hid] : (int16_t)-1;
                    memoryIndex->SetHeadNodeBundleNodeId(hid, nid);
                }

                SaveHeadNodeRoutingIndexFile(workDir,
                                             pivotCandidate->pivotLevel,
                                             pivotCandidate->nodePivotTags,
                                             m_tenantTagToNodes[p_tenantId],
                                             headNodeToNode);
            }
            // Save meta file AFTER bundleNodeId is populated
            SaveHeadNodeMetaFile(workDir, memoryIndex);
        }
    }

    if (pivotCandidate != nullptr) {
        fprintf(stderr,
                "[INFO] Tenant %d: pivot estimator selected level=%d nodes=%d\n",
                p_tenantId,
                pivotCandidate->pivotLevel,
                pivotCandidate->nodeCount);
    }

    // ── Tag-pure postings (chunked, KV-backed) ───────────────────────────
    // For tags with selectivity strictly below SPTAG_TAG_PURE_THRESHOLD
    // (default 0.01), build chunked (VID + normalized vector) payloads and
    // write them into the same KeyValueIO that holds regular SPANN postings.
    // That backend (FileIO ShardedLRUCache or RocksDB block cache) takes care
    // of caching — no separate user-space LRU is introduced. Float dtype only.
    // Runs AFTER the head-iteration loop so vidVecData covers heads too.
    {
        auto& purePostings = m_tenantTagPurePostings[p_tenantId];

        // Resolve the KV store + posting page limit from the loaded SPANN
        // index (configured at index build time, not from defaults).
        std::shared_ptr<SPTAG::Helper::KeyValueIO> kvDb;
        int postingPageLimit = 3;
        int bufferLength = 4;
        {
            std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
            auto it = m_tenantIndices.find(p_tenantId);
            if (it != m_tenantIndices.end()) {
                auto internalIdx2 = it->second->GetInternalIndex();
                auto* spann2 = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx2.get());
                if (spann2 != nullptr) {
                    if (auto* opts = spann2->GetOptions()) {
                        postingPageLimit = std::max(1, opts->m_postingPageLimit);
                        bufferLength = std::max(0, opts->m_bufferLength);
                    }
                    auto extra = spann2->GetDiskIndex();
                    if (extra) kvDb = extra->GetKVStore();
                }
            }
        }

        // Drop chunks written by a previous BuildSignatures call (keep the KV
        // store tidy — no leak in its block pool).
        if (kvDb != nullptr) {
            for (auto& [tag, pp] : purePostings) {
                if (!pp) continue;
                for (int k : pp->chunkKeys) {
                    kvDb->Delete(static_cast<SPTAG::SizeType>(k));
                }
            }
        }
        purePostings.clear();
        m_tenantTagPureKV.erase(p_tenantId);

        if (kTagPureEligible && kvDb != nullptr) {
            double pureThreshold = 0.01;
            if (const char* env = std::getenv("SPTAG_TAG_PURE_THRESHOLD")) {
                double parsed = 0.0;
                if (SPTAG::Helper::Convert::ConvertStringTo<double>(env, parsed)
                    && parsed > 0.0 && parsed <= 1.0) {
                    pureThreshold = parsed;
                }
            }
            int maxCount = (int)std::floor(pureThreshold * (double)p_numVectors);
            if (maxCount < 1) maxCount = 1;

            std::unordered_set<uint32_t> eligibleTags;
            eligibleTags.reserve(tagVectorCounts.size());
            for (const auto& [tag, vectorCount] : tagVectorCounts) {
                if (vectorCount > 0 && vectorCount <= maxCount) {
                    eligibleTags.insert(tag);
                }
            }

            std::unordered_map<uint32_t, std::vector<int>> tagVids;
            tagVids.reserve(eligibleTags.size());
            size_t missingVecs = 0;
            for (int vid = 0; vid < p_numVectors; ++vid) {
                std::unordered_set<uint32_t> seenLocal;
                bool anyHit = false;
                for (int t = 0; t < p_numTagsPerVec; ++t) {
                    uint32_t tag = p_tagsPtr[vid * p_numTagsPerVec + t];
                    if (!seenLocal.insert(tag).second) continue;
                    if (eligibleTags.count(tag) == 0) continue;
                    if (!vidSeen[vid]) { anyHit = true; continue; }
                    tagVids[tag].push_back(vid);
                }
                if (anyHit && !vidSeen[vid]) ++missingVecs;
            }

            // Chunk capacity: ≤ postingPageLimit pages per blob, matching the
            // page budget that regular postings already obey in this index.
            const int kRecBytes = (int)sizeof(int32_t) + m_dimension * (int)sizeof(float);
            int chunkBytes = postingPageLimit * 4096;
            int chunkCap = std::max(1, chunkBytes / kRecBytes);

            // Allocate keys starting just past the head ID range. The
            // FileIO mapping auto-grows; existing head keys are untouched.
            SPTAG::SizeType keyCursor = static_cast<SPTAG::SizeType>(numHeads);
            auto cursorIt = m_tenantTagPureNextKey.find(p_tenantId);
            if (cursorIt != m_tenantTagPureNextKey.end() && cursorIt->second > keyCursor) {
                keyCursor = cursorIt->second;
            }

            std::vector<float> normBuf;
            std::vector<std::string> chunkBlobs;
            int builtTags = 0;
            size_t builtVecs = 0;
            size_t totalChunks = 0;
            size_t failedTags = 0;

            // Workspace providing page-aligned buffers + iocb-initialized
            // AsyncReadRequest entries — required by FileIO::Put's direct
            // write path when the in-process LRU cache is disabled
            // (CacheSizeGB=0). One workspace, reused across all Put calls.
            const int blockLimit = postingPageLimit + bufferLength + 1;
            SPTAG::SPANN::ExtraWorkSpace ws;
            ws.Initialize(/*maxCheck*/16,
                          /*hashExp*/4,
                          /*internalResultNum*/1,
                          /*maxPages bytes*/ blockLimit << SPTAG::PageSizeEx,
                          /*blockIO*/true,
                          /*enableDataCompression*/false);

            for (auto& kv : tagVids) {
                uint32_t tag = kv.first;
                auto& vids = kv.second;
                if (vids.empty()) continue;

                // L2-normalize each vector once for cosine-as-1-IP at query.
                normBuf.assign((size_t)vids.size() * (size_t)m_dimension, 0.0f);
                for (size_t k = 0; k < vids.size(); ++k) {
                    const float* src = vidVecData.data() + (size_t)vids[k] * (size_t)m_dimension;
                    float n2 = 0.0f;
                    for (int i = 0; i < m_dimension; ++i) n2 += src[i] * src[i];
                    float inv = (n2 > 1e-30f) ? 1.0f / std::sqrt(n2) : 0.0f;
                    float* dst = normBuf.data() + k * (size_t)m_dimension;
                    for (int i = 0; i < m_dimension; ++i) dst[i] = src[i] * inv;
                }

                auto pure = std::make_shared<SPTAG::Cache::TagPurePosting>();
                pure->dim = m_dimension;
                pure->Pack(vids, normBuf, chunkCap, chunkBlobs);

                bool ok = true;
                pure->chunkKeys.clear();
                pure->chunkKeys.reserve(chunkBlobs.size());
                for (auto& blob : chunkBlobs) {
                    SPTAG::SizeType key = keyCursor++;
                    auto err = kvDb->Put(key, blob, std::chrono::microseconds(0), &ws.m_diskRequests);
                    if (err != SPTAG::ErrorCode::Success) {
                        fprintf(stderr,
                                "[TagPure] Put failed tenant=%d tag=%u key=%d err=%d size=%zu\n",
                                p_tenantId, tag, (int)key, (int)err, blob.size());
                        ok = false;
                        break;
                    }
                    pure->chunkKeys.push_back((int)key);
                }
                if (!ok) { ++failedTags; continue; }

                builtVecs += pure->count;
                ++builtTags;
                totalChunks += pure->chunkKeys.size();
                purePostings[tag] = std::move(pure);
            }

            m_tenantTagPureNextKey[p_tenantId] = keyCursor;
            m_tenantTagPureKV[p_tenantId] = kvDb;
            m_tenantTagPureBlockLimit[p_tenantId] = blockLimit;
            m_tenantTagPurePagesPerChunk[p_tenantId] = postingPageLimit;

            // Persist: 1) FileIO mapping + block pool so chunk addresses
            // survive process restart; 2) sidecar metadata so we can attach
            // the tag-pure structures at next LoadAll without re-scanning.
            {
                std::shared_ptr<SPTAG::SPANN::IExtraSearcher> extra;
                {
                    std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
                    auto it = m_tenantIndices.find(p_tenantId);
                    if (it != m_tenantIndices.end()) {
                        auto internalIdx = it->second->GetInternalIndex();
                        auto* spannIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
                        if (spannIdx) extra = spannIdx->GetDiskIndex();
                    }
                }
                if (extra) {
                    auto ec = extra->Checkpoint(workDir);
                    if (ec != SPTAG::ErrorCode::Success) {
                        fprintf(stderr,
                                "[TagPure] WARN tenant=%d kvDb checkpoint failed (err=%d); "
                                "chunks will not persist across restart\n",
                                p_tenantId, (int)ec);
                    }
                }
                const std::string metaPath = workDir + "/tagpure_meta.bin";
                if (!SPTAG::Cache::TagPureBundle::Save(metaPath, m_dimension, purePostings)) {
                    fprintf(stderr,
                            "[TagPure] WARN tenant=%d failed to write %s\n",
                            p_tenantId, metaPath.c_str());
                }
            }

            fprintf(stdout,
                    "[TagPure] tenant=%d threshold=%.4f maxCount=%d tags=%d vecs=%zu "
                    "chunks=%zu chunkCap=%d failed=%zu missing=%zu\n",
                    p_tenantId, pureThreshold, maxCount, builtTags, builtVecs,
                    totalChunks, chunkCap, failedTags, missingVecs);
            fflush(stdout);
        } else if (kTagPureEligible) {
            fprintf(stderr, "[TagPure] tenant=%d: no KV store available, skipping\n", p_tenantId);
        }
    }

    fprintf(stderr, "[INFO] Tenant %d: built PS + sparse index + %d head tags (%d postings, %llu assignments, sparse_max_postings=%d)\n",
            p_tenantId, headTagCount, numHeads,
            static_cast<unsigned long long>(totalAssignments),
    directSparseMaxPostings);
    return true;
}

bool TenantIndexManager::BackfillPrimaryHeadCSR(int p_tenantId, ByteArray p_vectors, int p_numVectors,
                                                ByteArray p_tags, int p_numTagsPerVec)
{
    if (p_vectors.Data() == nullptr || p_tags.Data() == nullptr || p_numVectors <= 0 || p_numTagsPerVec < 5) {
        return false;
    }
    if (!EnsureTenantLoaded(p_tenantId)) return false;

    std::shared_ptr<AnnIndex> index;
    {
        std::shared_lock<std::shared_mutex> lock(m_tenantIndicesMutex);
        const auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return false;
        index = it->second;
    }
    auto internal = index->GetInternalIndex();
    auto* spann = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internal.get());
    if (spann == nullptr) return false;

    return spann->BuildPrimaryHeadCSRBackfill(
        p_vectors.Data(), static_cast<SizeType>(p_numVectors),
        reinterpret_cast<const uint32_t*>(p_tags.Data()), p_numTagsPerVec);
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchWithACL(
    ByteArray p_queryVector, int p_tenantId, int p_resultNum,
    ByteArray p_queryTags, int p_numTags)
{
    static const bool s_wrapperTime = (std::getenv("SPTAG_LOG_WRAPPER_TIME") != nullptr);
    auto _wrTotal0 = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                   : std::chrono::high_resolution_clock::time_point{};
    const uint32_t* queryTagsPtr = reinterpret_cast<const uint32_t*>(p_queryTags.Data());
    // ── DNF predicate mode ──────────────────────────────────────────────
    // Sentinel: p_numTags < 0 means p_queryTags is a self-describing DNF blob
    // (uint32 words): [numClauses]{ [numLits] [col,val]xN }... encoding
    // OR-of-AND-clauses. Otherwise p_queryTags is the legacy flat OR/IN list of
    // p_numTags tag values. DNF mode routes only through the dense path; its
    // union of literal values drives the coarse masks/selectivity while the
    // exact per-vector DNF eval runs in the posting scan.
    bool dnfMode = (p_numTags < 0);
    SPTAG::Cache::DNFPredicate dnf;
    std::vector<uint32_t> dnfValues;
    if (dnfMode && queryTagsPtr != nullptr) {
        const uint32_t* w = queryTagsPtr;
        const size_t nWords = p_queryTags.Length() / sizeof(uint32_t);
        size_t pos = 0;
        // Versioned blob. Magic word selects the literal encoding:
        //   0x444E4633 ("DNF3") -> each literal is [kind, col, op, val] (4 words).
        //       kind 0 = categorical tag, kind 1 = numeric (post-filter only).
        //   0x444E4632 ("DNF2") -> each literal is [col, op, val] (3 words).
        //   (no magic)          -> legacy [col, val] (2 words), all equality, tag.
        // Only categorical (kind==0) equality values feed dnfValues (the coarse
        // mask/selectivity union); numeric literals are evaluated purely at the
        // exact per-vector post-filter and never drive retrieval.
        bool dnf3 = (nWords >= 1 && w[0] == 0x444E4633u);
        bool extended = (nWords >= 1 && w[0] == 0x444E4632u);
        if (dnf3 || extended) pos = 1;
        if (nWords > pos) {
            uint32_t numClauses = w[pos++];
            for (uint32_t ci = 0; ci < numClauses && pos < nWords; ++ci) {
                uint32_t numLits = w[pos++];
                SPTAG::Cache::DNFClause clause;
                for (uint32_t li = 0; li < numLits; ++li) {
                    if (dnf3) {
                        if (pos + 3 >= nWords) break;
                        uint32_t kind = w[pos++];
                        uint32_t col  = w[pos++];
                        uint8_t  op   = (uint8_t)w[pos++];
                        uint32_t val  = w[pos++];
                        clause.lits.push_back({col, val, op, (uint8_t)kind});
                        if (kind == 0 && op == SPTAG::Cache::DNF_EQ) dnfValues.push_back(val);
                    } else if (extended) {
                        if (pos + 2 >= nWords) break;
                        uint32_t col = w[pos++];
                        uint8_t  op  = (uint8_t)w[pos++];
                        uint32_t val = w[pos++];
                        clause.lits.push_back({col, val, op, (uint8_t)0});
                        if (op == SPTAG::Cache::DNF_EQ) dnfValues.push_back(val);
                    } else {
                        if (pos + 1 >= nWords) break;
                        uint32_t col = w[pos++];
                        uint32_t val = w[pos++];
                        clause.lits.push_back({col, val});
                        dnfValues.push_back(val);
                    }
                }
                if (!clause.lits.empty()) dnf.clauses.push_back(std::move(clause));
            }
        }
    }
    // Effective flat tag view used by the dense-path mask/selectivity builders.
    // In DNF mode this is the union of all literal values.
    const uint32_t* effTagsPtr = dnfMode ? (dnfValues.empty() ? nullptr : dnfValues.data()) : queryTagsPtr;
    const int effNumTags = dnfMode ? (int)dnfValues.size() : p_numTags;
    auto _ck_a = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    if (!EnsureTenantLoaded(p_tenantId)) return nullptr;
    auto _ck_b = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt == m_tenantSpannWorkDirs.end()) return nullptr;
    const std::string& workDir = wdIt->second;

    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;
    }

    // Unfiltered requests do not need ACL context, tag routing, selectivity
    // estimation, or posting metadata. AnnIndex::Search already executes the
    // PerTagBKT cross-edge unfilter path, so bypass the wrapper-only setup.
    if (p_numTags == 0) {
        return indexPtr->Search(p_queryVector, p_resultNum);
    }

    auto _ck_c = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};

    auto internalIdx = indexPtr->GetInternalIndex();
    auto _ck_d = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    bool forceDenseTagSearch = false;
    bool adaptiveFilteredNprobeEnabled = false;
    bool hybridDistanceEnabled = false;
    float filteredSearchNprobeSafety = 1.0f;
    if (internalIdx != nullptr) {
        const std::string forceDenseParam = internalIdx->GetParameter("ForceDenseTagSearch", "BuildSSDIndex");
        if (!forceDenseParam.empty()) {
            SPTAG::Helper::Convert::ConvertStringTo<bool>(forceDenseParam.c_str(), forceDenseTagSearch);
        }

        const std::string filteredSearchNprobeSafetyParam = internalIdx->GetParameter("FilteredSearchNprobeSafety", "BuildSSDIndex");
        if (!filteredSearchNprobeSafetyParam.empty()) {
            float parsedFilteredSearchNprobeSafety = 1.0f;
            if (SPTAG::Helper::Convert::ConvertStringTo<float>(filteredSearchNprobeSafetyParam.c_str(), parsedFilteredSearchNprobeSafety)
                && parsedFilteredSearchNprobeSafety > 0.0f) {
                filteredSearchNprobeSafety = parsedFilteredSearchNprobeSafety;
            }
        }

        auto* spannIndex = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
        const auto* searchOptions = spannIndex != nullptr ? spannIndex->GetOptions() : nullptr;
        adaptiveFilteredNprobeEnabled =
            searchOptions != nullptr && searchOptions->m_enableAdaptiveFilteredNprobe;
        const std::string hybridDistanceParam =
            internalIdx->GetParameter(
                "EnableHybridDistance", "BuildSSDIndex");
        if (!hybridDistanceParam.empty()) {
            SPTAG::Helper::Convert::ConvertStringTo<bool>(
                hybridDistanceParam.c_str(), hybridDistanceEnabled);
        }
        if (hybridDistanceEnabled) {
            // Hybrid-enabled filtered queries must reach the core cost router.
            // Tag-pure and sparse early returns cannot compare the original and
            // hybrid navigation and posting scan-range costs, so they are not
            // eligible here.
            forceDenseTagSearch = true;
        }
    }

    auto _ck_afterIdx = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                      : std::chrono::high_resolution_clock::time_point{};
    if (s_wrapperTime) {
        double t_a = std::chrono::duration<double, std::milli>(_ck_a - _wrTotal0).count();
        double t_b = std::chrono::duration<double, std::milli>(_ck_b - _ck_a).count();
        double t_c = std::chrono::duration<double, std::milli>(_ck_c - _ck_b).count();
        double t_d = std::chrono::duration<double, std::milli>(_ck_d - _ck_c).count();
        double t_e = std::chrono::duration<double, std::milli>(_ck_afterIdx - _ck_d).count();
        fprintf(stdout,
            "[1] WrapperEntry: tag0=%u  preEnsure=%.4f ensureTenant=%.4f tenantLookups=%.4f getInternal=%.4f getParams=%.4f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            t_a, t_b, t_c, t_d, t_e);
        fflush(stdout);
    }

    // ── Selectivity-based routing gate ──────────────────────────────────
    // The OPQ tag-pure path exhaustively ADC-scans a tag's entire inverted
    // vid list. For broad tags (large list AND high selectivity) that scan is
    // more expensive than the dense BKT-graph ANN path. Route to OPQ tag-pure
    // only when the tag is narrow by EITHER measure:
    //     vidCount < SPTAG_OPQPURE_MAX_VIDS   OR   selectivity < SPTAG_OPQPURE_MAX_SEL
    // Otherwise force the dense graph path. Defaults (MAX_VIDS=50000, MAX_SEL=0.1)
    // enable selectivity routing out of the box: broad tags (e.g. org) go dense,
    // narrower tags stay on tag-pure. Override the env vars to retune or disable.
    // Defaults enable selectivity routing out of the box (Option A): broad tags
    // (vidCount >= 50000 AND selectivity >= 0.1, e.g. org) route to the dense BKT
    // graph; narrower tags stay on the exhaustive OPQ tag-pure path. Override via
    // SPTAG_OPQPURE_MAX_VIDS / SPTAG_OPQPURE_MAX_SEL. Set MAX_VIDS very large and
    // MAX_SEL=1.0 to force every single-tag query back to tag-pure.
    static const std::int64_t s_opqPureMaxVids = []() {
        const char* v = std::getenv("SPTAG_OPQPURE_MAX_VIDS");
        return v ? std::strtoll(v, nullptr, 10) : (std::int64_t)50000;
    }();
    static const double s_opqPureMaxSel = []() {
        const char* v = std::getenv("SPTAG_OPQPURE_MAX_SEL");
        return v ? std::atof(v) : 0.1;
    }();
    static const bool s_gateDebug = (std::getenv("SPTAG_ROUTE_DEBUG") != nullptr);
    if (!forceDenseTagSearch && p_numTags == 1 && queryTagsPtr != nullptr
        && m_valueType == SPTAG::VectorValueType::Float && internalIdx != nullptr) {
        auto* spannGateIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
        if (spannGateIdx != nullptr) {
            auto gateDisk = spannGateIdx->GetDiskIndex();
            if (gateDisk != nullptr) {
                std::int64_t cnt = gateDisk->GetOPQTagVidCount(queryTagsPtr[0]);
                std::int64_t tot = gateDisk->GetOPQTotalVectors();
                // Selectivity routing applies regardless of RaBitQ: broad tags still go to
                // the dense BKT-graph path (graph navigation narrows the candidate set), and
                // RaBitQ simply replaces PQ as the in-RAM scorer on that path. (An earlier
                // experiment bypassed this gate under RaBitQ to force exhaustive tag-pure;
                // that threw away graph navigation and tanked broad/unfilter QPS. Reverted.)
                if (cnt >= 0 && tot > 0) {
                    double sel = static_cast<double>(cnt) / static_cast<double>(tot);
                    bool tagPureOK = (cnt < s_opqPureMaxVids) || (sel < s_opqPureMaxSel);
                    if (!tagPureOK) {
                        forceDenseTagSearch = true;
                        if (s_gateDebug) {
                            static std::atomic<int> g_c{0};
                            if (g_c++ < 8)
                                fprintf(stderr, "[ROUTE] tag=%u cnt=%lld sel=%.4f -> DENSE graph\n",
                                        queryTagsPtr[0], (long long)cnt, sel);
                        }
                    } else if (s_gateDebug) {
                        static std::atomic<int> g_c2{0};
                        if (g_c2++ < 8)
                            fprintf(stderr, "[ROUTE] tag=%u cnt=%lld sel=%.4f -> OPQ tag-pure\n",
                                    queryTagsPtr[0], (long long)cnt, sel);
                    }
                }
            }
        }
    }

    const auto tagStatsIt = m_tenantTagRoutingStats.find(p_tenantId);
    const auto* tagStats = (tagStatsIt != m_tenantTagRoutingStats.end()) ? &tagStatsIt->second : nullptr;
    if (hybridDistanceEnabled &&
        (tagStats == nullptr || tagStats->empty())) {
        fprintf(
            stderr,
            "[ERROR] Tenant %d: hybrid filtered search requires "
            "tag_routing_stats.bin; run BuildSignatures with the native "
            "INI before serving filtered queries\n",
            p_tenantId);
        return nullptr;
    }

    auto _ck_routingStart = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};

    const auto tagToNodesIt = m_tenantTagToNodes.find(p_tenantId);
    const auto headNodeToNodeIt = m_tenantHeadNodeToNode.find(p_tenantId);
    const std::vector<int>* headNodeToNode = (headNodeToNodeIt != m_tenantHeadNodeToNode.end())
        ? &headNodeToNodeIt->second
        : nullptr;
    std::vector<int> routedNodes;
    std::vector<uint8_t> allowedNodeMask;
    bool hasRoutingNodeFilter = false;
    // Flat queries union the nodes containing each requested value. DNF can
    // narrow safely only when every clause has categorical equality literals:
    // intersect within each AND clause, then union across OR clauses. Numeric
    // or non-equality-only clauses conservatively retain every bundle.
    bool routingCollected = false;
    if (tagToNodesIt != m_tenantTagToNodes.end() &&
        headNodeToNode != nullptr &&
        !headNodeToNode->empty()) {
        if (dnfMode && !dnf.Empty()) {
            routingCollected =
                TryCollectRoutingNodesForDNF(
                    tagToNodesIt->second, dnf,
                    routedNodes);
        } else if (p_numTags > 0) {
            routingCollected = TryCollectRoutingNodesForQuery(tagToNodesIt->second, queryTagsPtr, p_numTags, routedNodes);
            static const bool s_routeDbg2 = (std::getenv("SPTAG_ROUTE_DEBUG") != nullptr);
            if (s_routeDbg2) {
                static std::atomic<int> g_rc2{0};
                if (g_rc2++ < 8)
                    fprintf(stderr, "[OR-ROUTE] numTags=%d routedNodes=%zu collected=%d\n",
                            (int)p_numTags, routedNodes.size(), (int)routingCollected);
            }
        }
    }
    if (routingCollected) {
        int nodeCount = 0;
        auto nodeCountIt = m_tenantPivotNodeCounts.find(p_tenantId);
        if (nodeCountIt != m_tenantPivotNodeCounts.end()) {
            nodeCount = nodeCountIt->second;
        }
        if (nodeCount <= 0) {
            for (int nodeId : routedNodes) {
                nodeCount = std::max(nodeCount, nodeId + 1);
            }
        }

        if (nodeCount > 0) {
            allowedNodeMask.assign(static_cast<size_t>(nodeCount), 0);
            for (int nodeId : routedNodes) {
                if (nodeId >= 0 && nodeId < nodeCount) {
                    allowedNodeMask[static_cast<size_t>(nodeId)] = 1;
                    hasRoutingNodeFilter = true;
                }
            }
        }
    }

    // Check if ALL query tags are sparse → use brute-force path
    auto _ck_routing = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                     : std::chrono::high_resolution_clock::time_point{};
    if (s_wrapperTime) {
        double t_idxParam = std::chrono::duration<double, std::milli>(_ck_afterIdx - _wrTotal0).count();
        double t_tagStats = std::chrono::duration<double, std::milli>(_ck_routingStart - _ck_afterIdx).count();
        double t_routeNodes = std::chrono::duration<double, std::milli>(_ck_routing - _ck_routingStart).count();
        fprintf(stdout,
            "[1] WrapperRouting: tag0=%u rn=%zu  idxParam=%.3f tagStats=%.3f routeNodes=%.3f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            (size_t)routedNodes.size(), t_idxParam, t_tagStats, t_routeNodes);
        fflush(stdout);
    }
    static const bool s_disableSparsePath = []() {
        const char* v = std::getenv("SPTAG_DISABLE_SPARSE_PATH");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
    }();
    static const bool s_disableTagPurePath = []() {
        const char* v = std::getenv("SPTAG_DISABLE_TAG_PURE_PATH");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
    }();

    // ── Tag-pure fast path (chunked KV-backed) ──────────────────────────
    // Single-tag query whose tag has materialized tag-pure chunks: MultiGet
    // the chunks from the shared KV store (FileIO ShardedLRUCache / RocksDB
    // block cache handles caching), decode (VID + normVec) entries and
    // flat-scan for top-K. Bypasses BKT/SSD and gives R=1.0 by construction.
    if (!s_disableTagPurePath && !forceDenseTagSearch && p_numTags == 1
        && m_valueType == SPTAG::VectorValueType::Float) {
        // OPQ-prefilter narrow path: load the tag's ids -> ADC screen -> fetch only
        // L survivors -> exact rerank. Preserves the tag-pure exhaustiveness (screens
        // ALL the tag's vids) while cutting read volume ~18x vs full-vector chunks.
        static const bool s_opqPrefilter = []() {
            const char* v = std::getenv("SPTAG_OPQ_PREFILTER");
            return v && v[0] == '1';
        }();
        if (s_opqPrefilter && internalIdx != nullptr) {
            auto* spannIdx = dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
            if (spannIdx != nullptr) {
                auto disk = spannIdx->GetDiskIndex();
                if (disk != nullptr) {
                    auto result = std::make_shared<SPTAG::COMMON::QueryResultSet<float>>(
                        reinterpret_cast<const float*>(p_queryVector.Data()), p_resultNum);
                    if (disk->OPQTagPureSearch(*result, queryTagsPtr[0])) {
                        return result;
                    }
                }
            }
        }
        auto pureIt = m_tenantTagPurePostings.find(p_tenantId);
        auto kvIt = m_tenantTagPureKV.find(p_tenantId);
        if (pureIt != m_tenantTagPurePostings.end() && kvIt != m_tenantTagPureKV.end()
            && kvIt->second != nullptr) {
            auto tagIt = pureIt->second.find(queryTagsPtr[0]);
            if (tagIt != pureIt->second.end() && tagIt->second
                && tagIt->second->count > 0
                && tagIt->second->dim == m_dimension
                && !tagIt->second->chunkKeys.empty()) {
                const auto& pure = *tagIt->second;
                auto& kvDb = kvIt->second;

                std::vector<SPTAG::SizeType> keys;
                keys.reserve(pure.chunkKeys.size());
                for (int k : pure.chunkKeys) keys.push_back(static_cast<SPTAG::SizeType>(k));

                int pagesPerChunk = 16;
                auto pcIt = m_tenantTagPurePagesPerChunk.find(p_tenantId);
                if (pcIt != m_tenantTagPurePagesPerChunk.end()) pagesPerChunk = pcIt->second;

                // Workspace provides page-aligned per-chunk buffers and an
                // AsyncReadRequest vector pre-sized to cover every page of
                // every chunk. Required by FileIO::MultiGet for direct
                // (no-cache) reads.
                SPTAG::SPANN::ExtraWorkSpace ws;
                ws.Initialize(/*maxCheck*/16,
                              /*hashExp*/4,
                              /*internalResultNum*/(int)keys.size(),
                              /*maxPages bytes*/ pagesPerChunk << SPTAG::PageSizeEx,
                              /*blockIO*/true,
                              /*enableDataCompression*/false);

                std::vector<std::string> values(keys.size());
                auto err = kvDb->MultiGet(keys, &values,
                                          std::chrono::microseconds(60000000),
                                          &ws.m_diskRequests);
                if (err == SPTAG::ErrorCode::Success) {
                    std::vector<std::pair<float, int>> topK;
                    pure.SearchTopK(reinterpret_cast<const float*>(p_queryVector.Data()),
                                    values, p_resultNum, topK);
                    auto result = std::make_shared<QueryResult>(
                        p_queryVector.Data(), p_resultNum, false);
                    for (int i = 0; i < p_resultNum; ++i) {
                        if (i < (int)topK.size())
                            result->SetResult(i, topK[i].second, topK[i].first);
                        else
                            result->SetResult(i, -1, SPTAG::MaxDist);
                    }
                    if (s_wrapperTime) {
                        auto _ck_tp = std::chrono::high_resolution_clock::now();
                        double t_total = std::chrono::duration<double, std::milli>(
                            _ck_tp - _wrTotal0).count();
                        fprintf(stdout,
                            "[1] WrapperTagPure: tag=%u cands=%d chunks=%zu topK=%d total=%.3fms\n",
                            queryTagsPtr[0], pure.count, keys.size(), p_resultNum, t_total);
                        fflush(stdout);
                    }
                    return result;
                }
                // MultiGet failure → fall through to existing paths.
                fprintf(stderr, "[TagPure] MultiGet failed tag=%u err=%d, fallback\n",
                        queryTagsPtr[0], (int)err);
            }
        }
    }

    auto sparseIt = m_tenantSparseIdx.find(p_tenantId);
    if (!s_disableSparsePath && !forceDenseTagSearch && sparseIt != m_tenantSparseIdx.end() && p_numTags > 0) {
        auto& sparseIdx = sparseIt->second;
        bool hasDirectPostingListsForAllTags = true;
        // Collect posting IDs for all query tags
        std::unordered_set<int> bfPostings;
        for (int i = 0; i < p_numTags; i++) {
            auto* pids = sparseIdx->GetPostings(queryTagsPtr[i]);
            if (!pids) {
                hasDirectPostingListsForAllTags = false;
                break;
            }
            bfPostings.insert(pids->begin(), pids->end());
        }

        // Sparse fast-path policy: build-time `kSparseIndexBuildMaxPostings` is
        // the single source of truth - if a tag's posting list was materialized
        // at build, query-time always routes through the sparse path. No
        // second-stage union-size cap: that would create a window where the
        // sidecar was paid for but the path silently fell back to ANN.
        if (hasDirectPostingListsForAllTags && !bfPostings.empty()) {
            SPTAG::VectorIndex::ThreadLocalSearchContext searchContext;
            searchContext.m_queryTags.assign(queryTagsPtr, queryTagsPtr + p_numTags);
            searchContext.m_directPostingIDs.assign(bfPostings.begin(), bfPostings.end());
            SPTAG::VectorIndex::ThreadLocalSearchContextGuard searchContextGuard(std::move(searchContext));

            auto result = indexPtr->Search(p_queryVector, p_resultNum);
            return result;
        }
    }

    // Dense tag path: SPANN graph + bitmask PS + inline filter
    // Build query bitmask from requested tags
    auto _ck_sparse = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};
    SPTAG::Cache::PostingBitmask queryMask;
    queryMask.Clear();
    SPTAG::Cache::HierarchicalPostingMask queryHierMask;
    queryHierMask.Clear();
    auto memoryIndex =
        GetMemoryIndexForInternal(internalIdx);
    EnsureHeadNodeMetaLoaded(
        workDir, internalIdx);
    const SPTAG::Cache::HierWidthTable queryHierWidths =
        memoryIndex != nullptr
            ? memoryIndex->GetHeadNodeHierWidths()
            : SPTAG::Cache::HierWidthTable();
    // Map a raw tag value to its hierarchical level using the per-level offsets
    // persisted at build time (tag_level_offsets.bin). Tag value ranges are
    // disjoint and ascending per level, so the level is the largest index i
    // with qtag >= offsets[i]. Falls back to the legacy fixed thresholds only
    // when offsets are unavailable (pre-fix indexes).
    const std::vector<uint32_t>* levelOffsets = nullptr;
    {
        auto offIt = m_tenantTagLevelOffsets.find(p_tenantId);
        if (offIt != m_tenantTagLevelOffsets.end() && !offIt->second.empty())
            levelOffsets = &offIt->second;
    }
    for (int i = 0; i < effNumTags; i++) {
        queryMask.Insert(effTagsPtr[i]);
        // Hierarchical mask uses 0-indexed levels (0=org,1=dept,2=team,3=project)
        // matching the convention used at build time in
        // LoadPostingSignaturesIntoHeadIndex (Insert(t, tag) for t = 0..numTagsPerVec-1).
        uint32_t qtag = effTagsPtr[i];
        int level;
        if (levelOffsets != nullptr) {
            level = 0;
            for (int l = 0; l < (int)levelOffsets->size(); ++l) {
                if (qtag >= (*levelOffsets)[l]) level = l;
                else break;
            }
        } else {
            level = (qtag < 2000) ? 0 : (qtag < 3000) ? 1 : (qtag < 4000) ? 2 : 3;
        }
        queryHierMask.Insert(
            level, qtag,
            queryHierWidths);
    }
    auto _ck_qmask = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                   : std::chrono::high_resolution_clock::time_point{};
    SPTAG::VectorIndex::ThreadLocalSearchContext searchContext;
    if (effNumTags > 0 && effTagsPtr != nullptr) {
        searchContext.m_queryTags.assign(effTagsPtr, effTagsPtr + effNumTags);
    }
    // In DNF mode the exact predicate supersedes the flat OR/IN list. The flat
    // m_queryTags above carries only the union of literal values (for coarse
    // masks + selectivity); the authoritative per-vector test is the DNF.
    if (dnfMode && !dnf.Empty()) {
        searchContext.m_dnf = dnf;
    }
    // Plumb the data-driven tag-level offsets so the SPANN search path maps tag
    // values to hierarchical levels identically to build (column index = level),
    // instead of the stale legacy thresholds in TagLevelFromId.
    if (levelOffsets != nullptr) {
        searchContext.m_tagLevelOffsets = *levelOffsets;
    }
    if (internalIdx) {
            // Dense path posting pre-filter.
            //
            // The previous implementation routed each posting by its head centroid's
            // OWN pivot-level tag (single-value `headNodeToNode`). That was a type
            // error: SPANN places each vector in its K vector-space-nearest heads
            // regardless of tag, so a posting's content has an arbitrary tag
            // distribution that is NOT determined by the centroid's tag. Filtering
            // on the centroid's tag dropped valid postings (whose member vectors
            // matched the query) and capped ACL recall at ~0.91 regardless of nprobe.
            //
            // Fix: use the per-posting HierarchicalPostingMask built in
            // LoadPostingSignaturesIntoHeadIndex by OR-ing ALL member vectors' tags
            // at every level. MayIntersect is a no-false-negative test, so it is
            // safe as a pre-filter (false positives only let extra postings through).
            const bool dnfHasNum = !dnf.Empty() && dnf.HasNumericLiteral();
            const TenantIndexManager::NumericMeta* numMeta = nullptr;
            {
                auto nmIt = m_tenantNumericMeta.find(p_tenantId);
                if (nmIt != m_tenantNumericMeta.end()) numMeta = &nmIt->second;
            }
            if (memoryIndex != nullptr && memoryIndex->HasHeadNodeMeta() && (effNumTags > 0 || dnfHasNum)) {
                // Physical posting signatures are an optional I/O hint, not a
                // default candidate gate: exact inline filtering after
                // distance-first head selection retains replica coverage.
                auto* spannInternalIdx =
                    dynamic_cast<SPTAG::SPANN::ISPANNIndex*>(internalIdx.get());
                auto* searchOptions =
                    spannInternalIdx != nullptr ? spannInternalIdx->GetOptions() : nullptr;
                if (searchOptions != nullptr && searchOptions->m_enableHierPostingFilter) {
                const int quantCols = memoryIndex->GetHeadNodeNumQuantCols();
                const SPTAG::Cache::NumQuantParam* qp =
                    (numMeta != nullptr && !numMeta->params.empty()) ? numMeta->params.data() : nullptr;
                const int numBase = (numMeta != nullptr) ? numMeta->numBaseCols : 0;
                searchContext.m_postingFilter =
                    [memIdx = memoryIndex.get(), queryHierMask, queryHierWidths,
                     dnf, dnfHasNum, quantCols, qp, numBase](int localHid) {
                        // Use the per-posting multi-membership mask (union of all
                        // member-vector tags). MayIntersect / MayMatchHier are
                        // no-false-negative so they are safe as a pre-filter; the
                        // inline per-vector check inside the posting scan does the
                        // exact filtering (DNF eval when present). Numeric range
                        // literals additionally prune on the quantized numeric
                        // signature (MayMatchHierQuant) when present.
                        const auto* hierMask = memIdx->GetHeadNodePostingHierMask(localHid);
                        if (hierMask == nullptr) return true;  // fail open
                        if (!dnf.Empty()) {
                            if (dnfHasNum && quantCols > 0 && qp != nullptr) {
                                const std::uint64_t* quant = memIdx->GetHeadNodeNumQuant(localHid);
                                if (quant != nullptr)
                                    return dnf.MayMatchHierQuant(
                                        *hierMask, quant,
                                        quantCols, qp, numBase,
                                        queryHierWidths);
                            }
                            return dnf.MayMatchHier(
                                *hierMask,
                                queryHierWidths);
                        }
                        return hierMask->MayIntersect(
                            queryHierMask,
                            queryHierWidths);
                    };
                }
            }
            // Bundle-node routing (multi-bundle graph search) is separate from the
            // posting pre-filter above and still useful when manifest has >1 node.
            if (hasRoutingNodeFilter && headNodeToNode != nullptr) {
                searchContext.m_searchHeadBundleNodes = routedNodes;
            }
            (void)allowedNodeMask;
            (void)headNodeToNode;

        if (adaptiveFilteredNprobeEnabled || hybridDistanceEnabled) {
            auto vcIt2 = m_tenantVectorCounts.find(p_tenantId);
            int tenantSize2 = (vcIt2 != m_tenantVectorCounts.end()) ? vcIt2->second : 1;
            float vectorSel = EstimateQueryVectorSelectivity(
                tenantSize2, tagStats, effTagsPtr, effNumTags,
                dnfMode && !dnf.Empty() ? &dnf : nullptr,
                levelOffsets != nullptr
                    ? static_cast<int>(
                          levelOffsets->size())
                    : -1,
                numMeta != nullptr
                    ? numMeta->numBaseCols
                    : 0,
                numMeta != nullptr &&
                        !numMeta->params.empty()
                    ? numMeta->params.data()
                    : nullptr,
                numMeta != nullptr
                    ? numMeta->params.size()
                    : 0);
            searchContext.m_routeSelectivity = vectorSel;
            searchContext.m_filterSelectivity = std::clamp(
                vectorSel / std::max(1.0f, filteredSearchNprobeSafety),
                1e-6f, 1.0f);

            // Per-query selectivity fallback is needed only by adaptive nprobe.
            // Fixed-nprobe searches must not scan global head metadata for a value
            // the core search path will ignore.
            static const bool s_disableSelFallback = []() {
                const char* e = std::getenv("SPTAG_DISABLE_SEL_FALLBACK");
                return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
            }();
            static const SizeType kSelFallbackMaxSamples = []() -> SizeType {
                const char* e = std::getenv("SPTAG_SEL_FALLBACK_MAXSAMPLES");
                int v = e ? atoi(e) : 16384;
                return static_cast<SizeType>(v > 0 ? v : 16384);
            }();
            if (adaptiveFilteredNprobeEnabled &&
                !s_disableSelFallback && memoryIndex != nullptr &&
                memoryIndex->HasHeadNodeMeta() && tenantSize2 > 0 &&
                searchContext.m_filterSelectivity >= 1.0f) {
                SizeType totalHeads = memoryIndex->GetHeadNodeMetaSampleCount();
                SizeType stride = (totalHeads > kSelFallbackMaxSamples)
                    ? (totalHeads + kSelFallbackMaxSamples - 1) / kSelFallbackMaxSamples
                    : 1;
                int passCount = 0, sampled = 0;
                for (SizeType pid = 0; pid < totalHeads; pid += stride) {
                    if (memoryIndex->HeadNodePSMayIntersect(pid, queryMask)) passCount++;
                    sampled++;
                }
                totalHeads = (sampled > 0) ? static_cast<SizeType>(sampled) : totalHeads;
                float fallbackVectorSel = (totalHeads > 0)
                    ? static_cast<float>(passCount) / static_cast<float>(totalHeads)
                    : 1.0f;
                searchContext.m_filterSelectivity =
                    std::clamp(fallbackVectorSel, 1e-6f, 1.0f);
            }
        }
    }
    SPTAG::VectorIndex::ThreadLocalSearchContextGuard searchContextGuard(std::move(searchContext));
    auto _ck_denseEnd = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                      : std::chrono::high_resolution_clock::time_point{};

    auto _wrSearch0 = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};
    if (s_wrapperTime) {
        double t_routing = std::chrono::duration<double, std::milli>(_ck_routing - _wrTotal0).count();
        double t_sparse  = std::chrono::duration<double, std::milli>(_ck_sparse  - _ck_routing).count();
        double t_qmask   = std::chrono::duration<double, std::milli>(_ck_qmask   - _ck_sparse ).count();
        double t_dense   = std::chrono::duration<double, std::milli>(_ck_denseEnd- _ck_qmask  ).count();
        fprintf(stdout,
            "[1] WrapperPre: tag0=%u rn=%zu  routing=%.3f sparse=%.3f qmask=%.3f denseSetup=%.3f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            (size_t)routedNodes.size(), t_routing, t_sparse, t_qmask, t_dense);
        fflush(stdout);
    }
    auto _wrT0 = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    auto result = indexPtr->Search(p_queryVector, p_resultNum);
    if (s_wrapperTime) {
        auto _wrT1 = std::chrono::high_resolution_clock::now();
        double searchMs = std::chrono::duration<double, std::milli>(_wrT1 - _wrSearch0).count();
        double totalMs  = std::chrono::duration<double, std::milli>(_wrT1 - _wrTotal0).count();
        double preMs    = std::chrono::duration<double, std::milli>(_wrSearch0 - _wrTotal0).count();
        fprintf(stdout,
            "[1] WrapperCall: tag0=%u nTags=%d routedNodes=%zu preMs=%.3f searchMs=%.3f totalMs=%.3f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            p_numTags, (size_t)routedNodes.size(), preMs, searchMs, totalMs);
        fflush(stdout);
    }

    return result;
}

bool TenantIndexManager::EnsureTenantCached(int p_tenantId)
{
    return EnsureTenantLoaded(p_tenantId);
}

TenantIndexType TenantIndexManager::ChooseIndexType(int vectorCount) const
{
    // All tenants use SPANN: HeadIndex in memory, postings on SSD
    (void)vectorCount;
    return TenantIndexType::SPANN;
}

// --- String tenant ID mapping ---

int TenantIndexManager::RegisterTenantId(const char* p_tenantStr)
{
    if (p_tenantStr == nullptr) return -1;
    std::string key(p_tenantStr);
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantStrToInt.find(key);
    if (it != m_tenantStrToInt.end())
    {
        return it->second;
    }
    int id = m_nextInternalId++;
    m_tenantStrToInt[key] = id;
    m_tenantIntToStr[id] = key;
    return id;
}

int TenantIndexManager::GetInternalTenantId(const char* p_tenantStr) const
{
    if (p_tenantStr == nullptr) return -1;
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantStrToInt.find(std::string(p_tenantStr));
    return (it != m_tenantStrToInt.end()) ? it->second : -1;
}

const char* TenantIndexManager::GetTenantIdStr(int p_internalId) const
{
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantIntToStr.find(p_internalId);
    return (it != m_tenantIntToStr.end()) ? it->second.c_str() : nullptr;
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchByTenant(
    ByteArray p_queryVector, const char* p_tenantStr, int p_resultNum)
{
    int internalId = GetInternalTenantId(p_tenantStr);
    if (internalId < 0) return nullptr;
    return Search(p_queryVector, internalId, p_resultNum);
}
