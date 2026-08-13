// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HYBRIDROUTINGSTATS_H_
#define _SPTAG_SPANN_HYBRIDROUTINGSTATS_H_

#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Helper/AtomicFile.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace SPTAG
{
namespace SPANN
{

constexpr std::uint32_t kHybridRoutingStatsMagic = 0x53525948U; // HYRS
constexpr std::uint32_t kHybridRoutingStatsVersion = 3;

struct HybridRoutingStatsHeader
{
    std::uint32_t m_magic = kHybridRoutingStatsMagic;
    std::uint32_t m_version = kHybridRoutingStatsVersion;
    std::int32_t m_categoricalColumnCount = 0;
    std::int32_t m_maskCount = 0;
    std::int32_t m_numTagColumns = 0;
    std::int32_t m_headCount = 0;
    std::uint64_t m_generationFingerprint = 0;
};

struct HybridRouteLayout
{
    HybridPostingLayoutStats m_layout;
    std::vector<double> m_enrichmentByMask;
};

class HybridRoutingStats
{
public:
    std::vector<int> m_categoricalColumns;
    int m_numTagColumns = 0;
    std::vector<std::uint32_t> m_headAttributes;
    std::uint64_t m_generationFingerprint = 0;
    HybridRouteLayout m_original;
    HybridRouteLayout m_hybrid;

    bool Empty() const
    {
        return m_original.m_enrichmentByMask.empty() ||
            m_hybrid.m_enrichmentByMask.empty() ||
            m_numTagColumns <= 0 ||
            m_headAttributes.empty();
    }

    SizeType HeadCount() const
    {
        return m_numTagColumns > 0
            ? static_cast<SizeType>(
                  m_headAttributes.size() /
                  static_cast<size_t>(m_numTagColumns))
            : 0;
    }

    const std::uint32_t* HeadAttributes(SizeType p_head) const
    {
        if (p_head < 0 || p_head >= HeadCount()) {
            return nullptr;
        }
        return m_headAttributes.data() +
            static_cast<size_t>(p_head) *
                static_cast<size_t>(m_numTagColumns);
    }

    double Enrichment(bool p_hybrid, std::uint64_t p_columnMask) const
    {
        // Original navigation is vector-only, so the queried filter is not
        // aligned with the head attributes used to measure posting enrichment.
        if (!p_hybrid) return 1.0;
        const auto& values = m_hybrid.m_enrichmentByMask;
        if (values.empty()) return 1.0;
        if (p_columnMask >= values.size()) return 1.0;
        const double value =
            values[static_cast<size_t>(p_columnMask)];
        return std::isfinite(value) && value > 0.0
            ? value
            : 1.0;
    }

    std::uint64_t ConfiguredMask(
        const Cache::DNFPredicate* p_dnf,
        const std::vector<std::pair<int, std::uint32_t>>&
            p_flatCategoricalValues) const
    {
        std::uint64_t mask = 0;
        const auto addColumn = [&](int p_column) {
            const auto it = std::find(
                m_categoricalColumns.begin(),
                m_categoricalColumns.end(), p_column);
            if (it != m_categoricalColumns.end()) {
                const size_t bit = static_cast<size_t>(
                    it - m_categoricalColumns.begin());
                if (bit < 64) mask |= (1ULL << bit);
            }
        };
        if (p_dnf != nullptr && !p_dnf->Empty()) {
            std::uint64_t commonMask =
                (std::numeric_limits<std::uint64_t>::max)();
            bool haveClause = false;
            for (const auto& clause : p_dnf->clauses) {
                std::uint64_t clauseMask = 0;
                for (const auto& literal : clause.lits) {
                    if (literal.kind != 0) continue;
                    const auto it = std::find(
                        m_categoricalColumns.begin(),
                        m_categoricalColumns.end(),
                        static_cast<int>(literal.col));
                    if (it != m_categoricalColumns.end()) {
                        const size_t bit = static_cast<size_t>(
                            it - m_categoricalColumns.begin());
                        if (bit < 64) clauseMask |= (1ULL << bit);
                    }
                }
                commonMask &= clauseMask;
                haveClause = true;
            }
            return haveClause ? commonMask : 0;
        }
        std::uint64_t flatMask = 0;
        for (const auto& value :
             p_flatCategoricalValues) {
            if (std::find(
                    m_categoricalColumns.begin(),
                    m_categoricalColumns.end(),
                    value.first) ==
                m_categoricalColumns.end()) {
                return 0;
            }
            const std::uint64_t before = mask;
            addColumn(value.first);
            flatMask |= mask ^ before;
        }
        return flatMask != 0 &&
                       (flatMask & (flatMask - 1)) == 0
            ? flatMask
            : 0;
    }

    bool Save(const std::string& p_path, std::string& p_error) const
    {
        p_error.clear();
        if (!Valid()) {
            p_error = "invalid hybrid routing statistics";
            return false;
        }
        const std::string temporary = p_path + ".tmp";
        FILE* file = std::fopen(temporary.c_str(), "wb");
        if (file == nullptr) {
            p_error = "cannot create " + temporary;
            return false;
        }
        HybridRoutingStatsHeader header;
        header.m_categoricalColumnCount =
            static_cast<std::int32_t>(
                m_categoricalColumns.size());
        header.m_maskCount = static_cast<std::int32_t>(
            m_original.m_enrichmentByMask.size());
        header.m_numTagColumns = m_numTagColumns;
        header.m_headCount = HeadCount();
        header.m_generationFingerprint =
            m_generationFingerprint;
        bool ok =
            std::fwrite(&header, sizeof(header), 1, file) == 1 &&
            std::fwrite(
                m_categoricalColumns.data(), sizeof(int),
                m_categoricalColumns.size(), file) ==
                m_categoricalColumns.size() &&
            std::fwrite(
                m_headAttributes.data(),
                sizeof(std::uint32_t),
                m_headAttributes.size(), file) ==
                m_headAttributes.size() &&
            WriteLayout(file, m_original) &&
            WriteLayout(file, m_hybrid);
        if (std::fclose(file) != 0) ok = false;
        if (!ok) {
            std::remove(temporary.c_str());
            p_error =
                "failed to write complete hybrid routing statistics " +
                p_path;
            return false;
        }
        if (!Helper::AtomicReplaceFile(
                temporary, p_path)) {
            std::remove(temporary.c_str());
            p_error =
                "cannot publish hybrid routing statistics " +
                p_path;
            return false;
        }
        return true;
    }

    bool Load(const std::string& p_path,
              int p_expectedNumTagColumns,
              int p_expectedHeadCount,
              std::uint64_t p_expectedGenerationFingerprint,
              std::string& p_error)
    {
        *this = HybridRoutingStats();
        p_error.clear();
        if (p_expectedNumTagColumns <= 0 ||
            p_expectedHeadCount <= 0 ||
            p_expectedGenerationFingerprint == 0) {
            p_error =
                "invalid expected hybrid routing statistics";
            return false;
        }
        FILE* file = std::fopen(p_path.c_str(), "rb");
        if (file == nullptr) {
            p_error = "cannot open " + p_path;
            return false;
        }
        HybridRoutingStatsHeader header;
        bool ok =
            std::fread(&header, sizeof(header), 1, file) == 1 &&
            header.m_magic == kHybridRoutingStatsMagic &&
            header.m_version == kHybridRoutingStatsVersion &&
            header.m_categoricalColumnCount >= 0 &&
            header.m_categoricalColumnCount <= 16 &&
            header.m_numTagColumns ==
                p_expectedNumTagColumns &&
            header.m_headCount ==
                p_expectedHeadCount &&
            header.m_generationFingerprint ==
                p_expectedGenerationFingerprint &&
            header.m_maskCount ==
                (1 << header.m_categoricalColumnCount);
        if (ok) {
            m_generationFingerprint =
                header.m_generationFingerprint;
            m_numTagColumns =
                header.m_numTagColumns;
            m_categoricalColumns.resize(
                static_cast<size_t>(
                    header.m_categoricalColumnCount));
            const size_t headCount =
                static_cast<size_t>(header.m_headCount);
            const size_t tagColumns =
                static_cast<size_t>(
                    header.m_numTagColumns);
            ok = headCount <=
                (std::numeric_limits<size_t>::max)() /
                    tagColumns;
            const size_t attributeCount =
                ok ? headCount * tagColumns : 0;
            if (ok) {
                m_headAttributes.resize(attributeCount);
            }
            ok = ok &&
                std::fread(
                     m_categoricalColumns.data(), sizeof(int),
                     m_categoricalColumns.size(), file) ==
                     m_categoricalColumns.size() &&
                std::fread(
                    m_headAttributes.data(),
                    sizeof(std::uint32_t),
                    m_headAttributes.size(), file) ==
                    m_headAttributes.size() &&
                ReadLayout(
                    file, header.m_maskCount, m_original) &&
                ReadLayout(
                    file, header.m_maskCount, m_hybrid) &&
                std::fgetc(file) == EOF &&
                Valid();
        }
        std::fclose(file);
        if (!ok) {
            *this = HybridRoutingStats();
            p_error =
                "hybrid routing statistics format mismatch at " +
                p_path;
        }
        return ok;
    }

private:
    bool Valid() const
    {
        if (m_categoricalColumns.size() > 16) {
            return false;
        }
        const size_t maskCount =
            static_cast<size_t>(1) <<
            m_categoricalColumns.size();
        return m_generationFingerprint != 0 &&
            m_numTagColumns > 0 &&
            !m_headAttributes.empty() &&
            m_headAttributes.size() %
                    static_cast<size_t>(
                        m_numTagColumns) ==
                0 &&
            m_original.m_enrichmentByMask.size() ==
                   maskCount &&
            m_hybrid.m_enrichmentByMask.size() == maskCount &&
            ValidLayout(m_original) && ValidLayout(m_hybrid);
    }

    static bool ValidLayout(const HybridRouteLayout& p_layout)
    {
        const auto& layout = p_layout.m_layout;
        if (!std::isfinite(layout.m_averageRecords) ||
            !std::isfinite(layout.m_averagePages) ||
            !std::isfinite(layout.m_averageBytes) ||
            !std::isfinite(layout.m_uniqueRatio) ||
            layout.m_averageRecords <= 0.0 ||
            layout.m_averagePages <= 0.0 ||
            layout.m_averageBytes <= 0.0 ||
            layout.m_uniqueRatio <= 0.0 ||
            layout.m_uniqueRatio > 1.0) {
            return false;
        }
        return std::all_of(
            p_layout.m_enrichmentByMask.begin(),
            p_layout.m_enrichmentByMask.end(),
            [](double p_value) {
                return std::isfinite(p_value) &&
                    p_value > 0.0;
            });
    }

    static bool WriteLayout(
        FILE* p_file, const HybridRouteLayout& p_layout)
    {
        const double fixed[] = {
            p_layout.m_layout.m_averageRecords,
            p_layout.m_layout.m_averagePages,
            p_layout.m_layout.m_averageBytes,
            p_layout.m_layout.m_uniqueRatio};
        return std::fwrite(
                   fixed, sizeof(double), 4, p_file) == 4 &&
            std::fwrite(
                p_layout.m_enrichmentByMask.data(),
                sizeof(double),
                p_layout.m_enrichmentByMask.size(),
                p_file) ==
                p_layout.m_enrichmentByMask.size();
    }

    static bool ReadLayout(FILE* p_file,
                           int p_maskCount,
                           HybridRouteLayout& p_layout)
    {
        double fixed[4] = {};
        if (std::fread(
                fixed, sizeof(double), 4, p_file) != 4) {
            return false;
        }
        p_layout.m_layout.m_averageRecords = fixed[0];
        p_layout.m_layout.m_averagePages = fixed[1];
        p_layout.m_layout.m_averageBytes = fixed[2];
        p_layout.m_layout.m_uniqueRatio = fixed[3];
        p_layout.m_enrichmentByMask.resize(
            static_cast<size_t>(p_maskCount));
        return std::fread(
                   p_layout.m_enrichmentByMask.data(),
                   sizeof(double),
                   p_layout.m_enrichmentByMask.size(),
                   p_file) ==
            p_layout.m_enrichmentByMask.size();
    }
};

} // namespace SPANN
} // namespace SPTAG

#endif
