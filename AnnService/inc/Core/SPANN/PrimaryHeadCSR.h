#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace SPTAG::SPANN
{
    static constexpr std::uint64_t kPrimaryHeadCSRMagic = 0x3152534348444850ULL; // "PHDHCSR1"
    static constexpr std::uint32_t kPrimaryHeadCSRVersion = 1;

#pragma pack(push, 1)
    struct PrimaryHeadCSRHeader
    {
        std::uint64_t magic = kPrimaryHeadCSRMagic;
        std::uint32_t version = kPrimaryHeadCSRVersion;
        std::uint32_t headCount = 0;
        std::uint64_t entryCount = 0;
        std::uint32_t tagBases[4] = {};
        std::uint32_t reserved = 0;
    };

    struct PrimaryHeadCSREntry
    {
        std::uint32_t vid = 0;
        // Low 32 bits: one local uint8 tag ID per categorical level.
        // High 32 bits: numeric attribute.
        std::uint64_t attributes = 0;
    };
#pragma pack(pop)

    static_assert(sizeof(PrimaryHeadCSREntry) == 12, "primary CSR entries must remain compact");

    class PrimaryHeadCSR
    {
    public:
        bool Load(const std::string& path, std::uint32_t expectedHeadCount)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input) return false;

            PrimaryHeadCSRHeader header;
            input.read(reinterpret_cast<char*>(&header), sizeof(header));
            if (!input || header.magic != kPrimaryHeadCSRMagic ||
                header.version != kPrimaryHeadCSRVersion ||
                header.headCount != expectedHeadCount ||
                header.entryCount > std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }

            std::vector<std::uint32_t> offsets(static_cast<size_t>(header.headCount) + 1);
            input.read(reinterpret_cast<char*>(offsets.data()),
                       static_cast<std::streamsize>(offsets.size() * sizeof(std::uint32_t)));
            if (!input || offsets.back() != header.entryCount) return false;

            std::vector<PrimaryHeadCSREntry> entries(static_cast<size_t>(header.entryCount));
            input.read(reinterpret_cast<char*>(entries.data()),
                       static_cast<std::streamsize>(entries.size() * sizeof(PrimaryHeadCSREntry)));
            if (!input) return false;

            m_header = header;
            m_offsets = std::move(offsets);
            m_entries = std::move(entries);
            return true;
        }

        bool Loaded() const { return !m_offsets.empty(); }

        std::uint32_t HeadCount() const { return m_header.headCount; }

        const PrimaryHeadCSRHeader& Header() const { return m_header; }

        const PrimaryHeadCSREntry* Begin(std::uint32_t headId) const
        {
            if (headId >= m_header.headCount) return nullptr;
            return m_entries.data() + m_offsets[headId];
        }

        const PrimaryHeadCSREntry* End(std::uint32_t headId) const
        {
            if (headId >= m_header.headCount) return nullptr;
            return m_entries.data() + m_offsets[headId + 1];
        }

        bool IsProjectTag(std::uint32_t tag) const
        {
            const std::uint32_t base = m_header.tagBases[3];
            return tag >= base && tag - base <= std::numeric_limits<std::uint8_t>::max();
        }

        bool MatchesProject(const PrimaryHeadCSREntry& entry, std::uint32_t projectTag) const
        {
            if (!IsProjectTag(projectTag)) return false;
            const std::uint8_t expected = static_cast<std::uint8_t>(projectTag - m_header.tagBases[3]);
            const std::uint8_t actual = static_cast<std::uint8_t>(entry.attributes >> 24);
            return actual == expected;
        }

        void UnpackAttributes(const PrimaryHeadCSREntry& entry, std::uint32_t vecTags[5]) const
        {
            const std::uint32_t packedTags = static_cast<std::uint32_t>(entry.attributes);
            for (int level = 0; level < 4; ++level) {
                vecTags[level] = m_header.tagBases[level] + ((packedTags >> (level * 8)) & 0xffU);
            }
            vecTags[4] = static_cast<std::uint32_t>(entry.attributes >> 32);
        }

    private:
        PrimaryHeadCSRHeader m_header;
        std::vector<std::uint32_t> m_offsets;
        std::vector<PrimaryHeadCSREntry> m_entries;
    };
}
