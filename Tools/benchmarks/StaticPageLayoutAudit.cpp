// Native metadata-only audit for uncompressed STM1 and legacy raw STATIC posting layouts.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>

namespace {

constexpr std::uint32_t kStaticMetadataMagic = 0x314D5453U; // "STM1", little-endian.
constexpr std::uint64_t kPageSize = 4096;

enum class LayoutFormat {
    STM1,
    LegacyRaw,
};

struct Options {
    std::string postingFile;
    std::string outputFile;
    std::uint32_t valueBytes = 1;
    std::uint32_t pageCap = 0;
};

struct Header {
    LayoutFormat format = LayoutFormat::STM1;
    std::int32_t magic = 0;
    std::int32_t version = 0;
    std::int32_t listCount = 0;
    std::int32_t totalDocumentCount = 0;
    std::int32_t dataDimension = 0;
    std::int32_t recordBytes = 0;
    std::int32_t numTagsPerVec = 0;
    std::int32_t tailPageBudget = 0;
    std::int32_t listPageOffset = 0;
};

struct HistogramBucket {
    std::uint64_t listCount = 0;
    std::uint64_t elementCount = 0;
};

using Histogram = std::map<std::uint64_t, HistogramBucket>;

struct PackingStats {
    std::uint64_t fullPages = 0;
    std::uint64_t listCount = 0;
    std::array<std::uint64_t, kPageSize> remainderCounts = {};
};

struct Totals {
    std::uint64_t nonemptyLists = 0;
    std::uint64_t totalListElements = 0;
    std::uint64_t totalPureElements = 0;
    std::uint64_t maxListElements = 0;
    std::uint64_t currentPayloadBytes = 0;
    std::uint64_t rawPayloadBytes = 0;
    std::uint64_t metadataPayloadBytes = 0;
    std::uint64_t storedPageSum = 0;
    std::uint64_t currentRuntimePageSum = 0;
    std::uint64_t currentZeroOffsetPageSum = 0;
    std::uint64_t rawZeroOffsetPageSum = 0;
    std::uint64_t rawSameOffsetPageSum = 0;
    std::uint64_t storedRuntimePageMismatches = 0;
    std::uint64_t nonzeroPageOffsetLists = 0;
    std::uint64_t pageOffsetPageIncreaseLists = 0;
    std::uint64_t maxPageOffset = 0;
    std::uint64_t pureRuntimePageSum = 0;
    std::uint64_t maxPureRuntimePages = 0;
    std::uint64_t capAffectedLists = 0;
    std::uint64_t capPurePrefixOverflowLists = 0;
    std::uint64_t capRetainedElements = 0;
    std::uint64_t capTrimmedElements = 0;
    std::uint64_t capOverlayPayloadBytes = 0;
    std::uint64_t capOverlayPageAlignedBytes = 0;
    std::uint64_t fileBytes = 0;
    std::uint64_t postingContentOffsetBytes = 0;
    std::uint64_t sourceBestFitContentPages = 0;
    std::uint64_t sourceBestFitFileBytes = 0;
    std::uint64_t currentLayoutPaddingBytes = 0;
    std::uint64_t sourceBestFitPaddingBytes = 0;
    std::uint64_t excessBytesOverSourceBestFit = 0;
};

void Usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " --posting-file INDEX/SPTAGFullList.bin [--output report.json]"
              << " [--value-bytes N] [--page-cap N]\n";
}

bool ParseUnsigned(const char* text, std::uint32_t& value)
{
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' ||
        parsed > static_cast<unsigned long>((std::numeric_limits<std::uint32_t>::max)())) {
        return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

bool ParseArgs(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::string(arg) == "--help" || std::string(arg) == "-h") {
            Usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) return false;
        const char* value = argv[++i];
        if (std::string(arg) == "--posting-file") {
            options.postingFile = value;
        } else if (std::string(arg) == "--output") {
            options.outputFile = value;
        } else if (std::string(arg) == "--value-bytes") {
            if (!ParseUnsigned(value, options.valueBytes) || options.valueBytes == 0) return false;
        } else if (std::string(arg) == "--page-cap") {
            if (!ParseUnsigned(value, options.pageCap)) return false;
        } else {
            return false;
        }
    }
    return !options.postingFile.empty();
}

template <typename T>
bool ReadExact(std::istream& input, T& value)
{
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    return input.gcount() == static_cast<std::streamsize>(sizeof(value));
}

bool ReadHeader(std::istream& input, std::uint32_t valueBytes, Header& header)
{
    if (!ReadExact(input, header.magic)) return false;
    if (static_cast<std::uint32_t>(header.magic) == kStaticMetadataMagic) {
        header.format = LayoutFormat::STM1;
        return ReadExact(input, header.version) &&
            ReadExact(input, header.listCount) &&
            ReadExact(input, header.totalDocumentCount) &&
            ReadExact(input, header.dataDimension) &&
            ReadExact(input, header.recordBytes) &&
            ReadExact(input, header.numTagsPerVec) &&
            ReadExact(input, header.tailPageBudget) &&
            ReadExact(input, header.listPageOffset);
    }

    header.format = LayoutFormat::LegacyRaw;
    header.listCount = header.magic;
    if (!ReadExact(input, header.totalDocumentCount) ||
        !ReadExact(input, header.dataDimension) ||
        !ReadExact(input, header.listPageOffset) ||
        header.dataDimension <= 0) {
        return false;
    }
    const std::uint64_t rawRecordBytes =
        static_cast<std::uint64_t>(header.dataDimension) * valueBytes + sizeof(std::int32_t);
    if (rawRecordBytes > static_cast<std::uint64_t>((std::numeric_limits<std::int32_t>::max)())) {
        return false;
    }
    header.recordBytes = static_cast<std::int32_t>(rawRecordBytes);
    header.version = 0;
    header.numTagsPerVec = 0;
    header.tailPageBudget = 0;
    return true;
}

std::uint64_t CeilingDivide(std::uint64_t numerator, std::uint64_t denominator)
{
    return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

std::uint64_t PageCount(std::uint64_t elementCount, std::uint64_t recordBytes,
                        std::uint64_t pageOffset)
{
    return CeilingDivide(elementCount * recordBytes + pageOffset, kPageSize);
}

int LargestRemainderAtMost(const std::array<std::uint64_t, kPageSize / 64>& masks,
                           std::uint64_t availableBytes)
{
    const std::uint64_t maxRemainder = (std::min)(availableBytes, kPageSize - 1);
    int word = static_cast<int>(maxRemainder / 64);
    const unsigned int bit = static_cast<unsigned int>(maxRemainder % 64);
    std::uint64_t eligible =
        masks[static_cast<std::size_t>(word)] &
        (bit == 63 ? (std::numeric_limits<std::uint64_t>::max)()
                   : ((std::uint64_t{1} << (bit + 1)) - 1));
    if (eligible != 0) {
        return word * 64 + 63 - __builtin_clzll(eligible);
    }
    for (--word; word >= 0; --word) {
        eligible = masks[static_cast<std::size_t>(word)];
        if (eligible != 0) return word * 64 + 63 - __builtin_clzll(eligible);
    }
    return -1;
}

bool SimulateSourceBestFitPages(const PackingStats& packing, std::uint64_t& contentPages)
{
    std::array<std::uint64_t, kPageSize> counts = packing.remainderCounts;
    std::array<std::uint64_t, kPageSize / 64> masks = {};
    for (std::size_t remainder = 0; remainder < counts.size(); ++remainder) {
        if (counts[remainder] != 0) {
            masks[remainder / 64] |= std::uint64_t{1} << (remainder % 64);
        }
    }

    std::uint64_t remainingLists = packing.listCount;
    std::uint64_t residualPages = 0;
    std::uint64_t offset = 0;
    while (remainingLists > 0) {
        const std::uint64_t available = kPageSize - offset;
        const int remainder = LargestRemainderAtMost(masks, available);
        if (remainder < 0 ||
            (available != kPageSize && remainder == 0)) {
            ++residualPages;
            offset = 0;
            continue;
        }

        const std::size_t index = static_cast<std::size_t>(remainder);
        if (--counts[index] == 0) {
            masks[index / 64] &= ~(std::uint64_t{1} << (index % 64));
        }
        --remainingLists;
        offset += index;
        if (offset == kPageSize) {
            ++residualPages;
            offset = 0;
        }
    }
    if (offset > 0) ++residualPages;
    if (packing.fullPages > (std::numeric_limits<std::uint64_t>::max)() - residualPages) {
        return false;
    }
    contentPages = packing.fullPages + residualPages;
    return true;
}

void AddHistogram(Histogram& histogram, std::uint64_t pageCount, std::uint64_t elementCount)
{
    HistogramBucket& bucket = histogram[pageCount];
    ++bucket.listCount;
    bucket.elementCount += elementCount;
}

void WriteHistogram(std::ostream& output, const Histogram& histogram)
{
    output << "[";
    bool first = true;
    for (const auto& entry : histogram) {
        if (!first) output << ",";
        first = false;
        output << "\n      {\"pages\":" << entry.first
               << ",\"list_count\":" << entry.second.listCount
               << ",\"element_count\":" << entry.second.elementCount << "}";
    }
    if (!histogram.empty()) output << "\n    ";
    output << "]";
}

double PercentReduction(std::uint64_t before, std::uint64_t after)
{
    if (before == 0 || after >= before) return 0.0;
    return 100.0 * static_cast<double>(before - after) / static_cast<double>(before);
}

bool Audit(const Options& options, Header& header, Totals& totals,
           Histogram& currentRuntimeHistogram, Histogram& rawZeroOffsetHistogram,
           PackingStats& packing)
{
    std::ifstream input(options.postingFile, std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open posting file: " << options.postingFile << "\n";
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff fileSize = input.tellg();
    input.seekg(0, std::ios::beg);
    if (fileSize <= 0) {
        std::cerr << "Cannot determine posting file size: " << options.postingFile << "\n";
        return false;
    }
    totals.fileBytes = static_cast<std::uint64_t>(fileSize);
    if (!ReadHeader(input, options.valueBytes, header)) {
        std::cerr << "Cannot read STATIC posting header from: " << options.postingFile << "\n";
        return false;
    }
    const bool metadataFormat = header.format == LayoutFormat::STM1;
    if ((metadataFormat && (static_cast<std::uint32_t>(header.magic) != kStaticMetadataMagic ||
                            header.version <= 0 || header.numTagsPerVec <= 0)) ||
        header.listCount <= 0 || header.totalDocumentCount < 0 ||
        header.dataDimension <= 0 || header.recordBytes <= 0 || header.numTagsPerVec < 0 ||
        header.tailPageBudget < -1 || header.listPageOffset < 0) {
        std::cerr << "Posting file is not a valid uncompressed STATIC layout\n";
        return false;
    }

    const std::uint64_t rawRecordBytes =
        static_cast<std::uint64_t>(header.dataDimension) * options.valueBytes + sizeof(std::int32_t);
    const std::uint64_t expectedCurrentRecordBytes =
        rawRecordBytes + static_cast<std::uint64_t>(header.numTagsPerVec) * sizeof(std::uint32_t);
    if (static_cast<std::uint64_t>(header.recordBytes) != expectedCurrentRecordBytes) {
        std::cerr << "STATIC record width " << header.recordBytes
                  << " does not match dim/value/tag metadata (expected "
                  << expectedCurrentRecordBytes << ")\n";
        return false;
    }

    const std::uint64_t metadataBytes =
        sizeof(std::int32_t) * (metadataFormat ? 9 : 4) +
        static_cast<std::uint64_t>(header.listCount) *
            (sizeof(std::int32_t) * (metadataFormat ? 3 : 2) + sizeof(std::uint16_t) * 2);
    const std::uint64_t postingDataOffset =
        static_cast<std::uint64_t>(header.listPageOffset) * kPageSize;
    if (postingDataOffset < metadataBytes || postingDataOffset > totals.fileBytes) {
        std::cerr << "STATIC list-page offset is invalid\n";
        return false;
    }
    totals.postingContentOffsetBytes = postingDataOffset;

    for (std::int32_t list = 0; list < header.listCount; ++list) {
        std::int32_t pageNum = 0;
        std::uint16_t pageOffset = 0;
        std::int32_t listElements = 0;
        std::uint16_t storedPages = 0;
        std::int32_t pureElements = 0;
        if (!ReadExact(input, pageNum) || !ReadExact(input, pageOffset) ||
            !ReadExact(input, listElements) || !ReadExact(input, storedPages) ||
            (metadataFormat && !ReadExact(input, pureElements))) {
            std::cerr << "Failed to read STATIC list metadata at list " << list << "\n";
            return false;
        }
        if (!metadataFormat) pureElements = listElements;
        if (pageNum < 0 || pageOffset >= kPageSize || listElements < 0 ||
            pureElements < 0 || pureElements > listElements) {
            std::cerr << "Invalid STATIC list metadata at list " << list << "\n";
            return false;
        }

        const std::uint64_t elements = static_cast<std::uint64_t>(listElements);
        const std::uint64_t offset = pageOffset;
        const std::uint64_t currentListBytes =
            elements * static_cast<std::uint64_t>(header.recordBytes);
        const std::uint64_t currentRuntimePages =
            PageCount(elements, static_cast<std::uint64_t>(header.recordBytes), offset);
        const std::uint64_t pureRuntimePages =
            PageCount(static_cast<std::uint64_t>(pureElements),
                      static_cast<std::uint64_t>(header.recordBytes), offset);
        const std::uint64_t currentZeroOffsetPages =
            PageCount(elements, static_cast<std::uint64_t>(header.recordBytes), 0);
        const std::uint64_t rawZeroOffsetPages = PageCount(elements, rawRecordBytes, 0);
        const std::uint64_t rawSameOffsetPages = PageCount(elements, rawRecordBytes, offset);

        totals.nonemptyLists += elements > 0;
        totals.totalListElements += elements;
        totals.totalPureElements += static_cast<std::uint64_t>(pureElements);
        totals.maxListElements = (std::max)(totals.maxListElements, elements);
        totals.currentPayloadBytes += elements * static_cast<std::uint64_t>(header.recordBytes);
        totals.rawPayloadBytes += elements * rawRecordBytes;
        totals.metadataPayloadBytes +=
            elements * static_cast<std::uint64_t>(header.numTagsPerVec) * sizeof(std::uint32_t);
        totals.storedPageSum += storedPages;
        totals.currentRuntimePageSum += currentRuntimePages;
        totals.currentZeroOffsetPageSum += currentZeroOffsetPages;
        totals.rawZeroOffsetPageSum += rawZeroOffsetPages;
        totals.rawSameOffsetPageSum += rawSameOffsetPages;
        totals.storedRuntimePageMismatches += storedPages != currentRuntimePages;
        totals.nonzeroPageOffsetLists += offset != 0;
        totals.pageOffsetPageIncreaseLists += currentRuntimePages > currentZeroOffsetPages;
        totals.maxPageOffset = (std::max)(totals.maxPageOffset, offset);
        totals.pureRuntimePageSum += pureRuntimePages;
        totals.maxPureRuntimePages = (std::max)(totals.maxPureRuntimePages, pureRuntimePages);

        if (options.pageCap > 0) {
            const std::uint64_t cappedBytes =
                static_cast<std::uint64_t>(options.pageCap) * kPageSize;
            const std::uint64_t capElements = offset > cappedBytes ? 0 :
                (cappedBytes - offset) / static_cast<std::uint64_t>(header.recordBytes);
            const std::uint64_t retainedElements = (std::min)(elements, capElements);
            if (retainedElements < elements) {
                ++totals.capAffectedLists;
                totals.capRetainedElements += retainedElements;
                totals.capTrimmedElements += elements - retainedElements;
                const std::uint64_t retainedBytes =
                    retainedElements * static_cast<std::uint64_t>(header.recordBytes);
                totals.capOverlayPayloadBytes += retainedBytes;
                totals.capOverlayPageAlignedBytes +=
                    PageCount(retainedElements, static_cast<std::uint64_t>(header.recordBytes), 0) *
                    kPageSize;
            }
            totals.capPurePrefixOverflowLists +=
                static_cast<std::uint64_t>(pureElements) > capElements;
        }

        if (elements > 0) {
            packing.fullPages += currentListBytes / kPageSize;
            ++packing.remainderCounts[static_cast<std::size_t>(currentListBytes % kPageSize)];
            ++packing.listCount;
        }
        AddHistogram(currentRuntimeHistogram, currentRuntimePages, elements);
        AddHistogram(rawZeroOffsetHistogram, rawZeroOffsetPages, elements);
    }

    if (!SimulateSourceBestFitPages(packing, totals.sourceBestFitContentPages) ||
        totals.sourceBestFitContentPages >
            ((std::numeric_limits<std::uint64_t>::max)() - totals.postingContentOffsetBytes) / kPageSize) {
        std::cerr << "STATIC source best-fit layout overflows\n";
        return false;
    }
    totals.sourceBestFitFileBytes = totals.postingContentOffsetBytes +
        totals.sourceBestFitContentPages * kPageSize;
    if (totals.currentPayloadBytes > totals.fileBytes - totals.postingContentOffsetBytes ||
        totals.currentPayloadBytes > totals.sourceBestFitContentPages * kPageSize) {
        std::cerr << "STATIC payload exceeds physical layout\n";
        return false;
    }
    totals.currentLayoutPaddingBytes =
        totals.fileBytes - totals.postingContentOffsetBytes - totals.currentPayloadBytes;
    totals.sourceBestFitPaddingBytes =
        totals.sourceBestFitContentPages * kPageSize - totals.currentPayloadBytes;
    totals.excessBytesOverSourceBestFit =
        totals.fileBytes > totals.sourceBestFitFileBytes
        ? totals.fileBytes - totals.sourceBestFitFileBytes
        : 0;
    return true;
}

bool WriteReport(const Options& options, const Header& header, const Totals& totals,
                 const Histogram& currentRuntimeHistogram, const Histogram& rawZeroOffsetHistogram)
{
    std::ofstream outputFile;
    std::ostream* output = &std::cout;
    if (!options.outputFile.empty()) {
        outputFile.open(options.outputFile);
        if (!outputFile) {
            std::cerr << "Cannot open output file: " << options.outputFile << "\n";
            return false;
        }
        output = &outputFile;
    }

    const std::uint64_t rawRecordBytes =
        static_cast<std::uint64_t>(header.dataDimension) * options.valueBytes + sizeof(std::int32_t);
    const char* format = header.format == LayoutFormat::STM1 ? "STM1" : "legacy_raw";
    *output << std::fixed << std::setprecision(6);
    *output << "{\n"
            << "  \"posting_file\":\"" << options.postingFile << "\",\n"
            << "  \"format\":\"" << format << "\",\n"
            << "  \"header\":{\"version\":" << header.version
            << ",\"list_count\":" << header.listCount
            << ",\"total_document_count\":" << header.totalDocumentCount
            << ",\"data_dimension\":" << header.dataDimension
            << ",\"value_bytes\":" << options.valueBytes
            << ",\"current_record_bytes\":" << header.recordBytes
            << ",\"num_tags_per_vector\":" << header.numTagsPerVec
            << ",\"raw_counterfactual_record_bytes\":" << rawRecordBytes
            << ",\"tail_page_budget\":" << header.tailPageBudget
            << ",\"list_page_offset\":" << header.listPageOffset << "},\n"
            << "  \"page_formulas\":{"
            << "\"current_runtime\":\"ceil((list_ele_count * current_record_bytes + page_offset) / 4096)\","
            << "\"raw_counterfactual\":\"ceil(list_ele_count * raw_record_bytes / 4096)\","
            << "\"raw_same_offsets\":\"ceil((list_ele_count * raw_record_bytes + current_page_offset) / 4096)\""
            << "},\n"
            << "  \"totals\":{\n"
            << "    \"nonempty_list_count\":" << totals.nonemptyLists << ",\n"
            << "    \"total_list_elements\":" << totals.totalListElements << ",\n"
            << "    \"total_pure_elements\":" << totals.totalPureElements << ",\n"
            << "    \"max_list_elements\":" << totals.maxListElements << ",\n"
            << "    \"current_payload_bytes\":" << totals.currentPayloadBytes << ",\n"
            << "    \"raw_counterfactual_payload_bytes\":" << totals.rawPayloadBytes << ",\n"
            << "    \"metadata_payload_bytes\":" << totals.metadataPayloadBytes << ",\n"
            << "    \"stored_page_sum\":" << totals.storedPageSum << ",\n"
            << "    \"current_runtime_per_list_page_sum\":" << totals.currentRuntimePageSum << ",\n"
            << "    \"current_zero_offset_per_list_page_sum\":" << totals.currentZeroOffsetPageSum << ",\n"
            << "    \"raw_zero_offset_per_list_page_sum\":" << totals.rawZeroOffsetPageSum << ",\n"
            << "    \"raw_same_offsets_per_list_page_sum\":" << totals.rawSameOffsetPageSum << ",\n"
            << "    \"raw_zero_offset_page_reduction_percent\":"
            << PercentReduction(totals.currentRuntimePageSum, totals.rawZeroOffsetPageSum) << ",\n"
            << "    \"stored_runtime_page_mismatch_list_count\":"
            << totals.storedRuntimePageMismatches << ",\n"
            << "    \"nonzero_page_offset_list_count\":" << totals.nonzeroPageOffsetLists << ",\n"
            << "    \"page_offset_page_increase_list_count\":"
            << totals.pageOffsetPageIncreaseLists << ",\n"
            << "    \"max_page_offset\":" << totals.maxPageOffset << ",\n"
            << "    \"pure_runtime_per_list_page_sum\":" << totals.pureRuntimePageSum << ",\n"
            << "    \"max_pure_runtime_pages\":" << totals.maxPureRuntimePages;
    if (options.pageCap > 0) {
        *output << ",\n"
                << "    \"page_cap\":" << options.pageCap << ",\n"
                << "    \"cap_affected_lists\":" << totals.capAffectedLists << ",\n"
                << "    \"cap_pure_prefix_overflow_lists\":"
                << totals.capPurePrefixOverflowLists << ",\n"
                << "    \"cap_affected_retained_elements\":" << totals.capRetainedElements << ",\n"
                << "    \"cap_trimmed_elements\":" << totals.capTrimmedElements << ",\n"
                << "    \"cap_overlay_payload_bytes\":" << totals.capOverlayPayloadBytes << ",\n"
                << "    \"cap_overlay_page_aligned_bytes\":"
                << totals.capOverlayPageAlignedBytes;
    }
    *output << "\n"
            << "  },\n"
            << "  \"physical_layout\":{\n"
            << "    \"logical_file_bytes\":" << totals.fileBytes << ",\n"
            << "    \"posting_content_offset_bytes\":" << totals.postingContentOffsetBytes << ",\n"
            << "    \"current_layout_padding_bytes\":" << totals.currentLayoutPaddingBytes << ",\n"
            << "    \"source_best_fit_content_pages\":" << totals.sourceBestFitContentPages << ",\n"
            << "    \"source_best_fit_file_bytes\":" << totals.sourceBestFitFileBytes << ",\n"
            << "    \"source_best_fit_padding_bytes\":" << totals.sourceBestFitPaddingBytes << ",\n"
            << "    \"excess_bytes_over_source_best_fit\":"
            << totals.excessBytesOverSourceBestFit << "\n"
            << "  },\n"
            << "  \"notes\":["
            << "\"Per-list page sums count scan ranges, not unique physical pages in the file.\","
            << "\"The raw counterfactual uses list_ele_count and assumes each list starts at offset zero; it does not model a rebuilt raw bin-packing layout.\","
            << "\"source_best_fit reproduces SelectPostingOffset's largest-fitting-remainder page packing for the current uncompressed list sizes.\","
            << "\"Use total list elements, not pure elements, for the current unfiltered scan path.\""
            << "],\n"
            << "  \"page_histograms\":{\n"
            << "    \"current_runtime\":";
    WriteHistogram(*output, currentRuntimeHistogram);
    *output << ",\n    \"raw_zero_offset_counterfactual\":";
    WriteHistogram(*output, rawZeroOffsetHistogram);
    *output << "\n  }\n}\n";
    return static_cast<bool>(*output);
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!ParseArgs(argc, argv, options)) {
        Usage(argv[0]);
        return 2;
    }

    Header header;
    Totals totals;
    Histogram currentRuntimeHistogram;
    Histogram rawZeroOffsetHistogram;
    PackingStats packing;
    if (!Audit(options, header, totals, currentRuntimeHistogram, rawZeroOffsetHistogram, packing) ||
        !WriteReport(options, header, totals, currentRuntimeHistogram, rawZeroOffsetHistogram)) {
        return 1;
    }
    return 0;
}
