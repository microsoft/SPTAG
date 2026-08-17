// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_LIMITEDTAGSUPPORT_H_
#define _SPTAG_SPANN_LIMITEDTAGSUPPORT_H_

#include "inc/Core/Common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace SPTAG
{
namespace SPANN
{

class LimitedTagSupport
{
public:
    static constexpr std::uint32_t EmptyTag =
        (std::numeric_limits<std::uint32_t>::max)();

#pragma pack(push, 1)
    struct Header
    {
        std::uint32_t m_magic = 0x3153544cU; // LTS1
        std::uint32_t m_version = 1;
        std::uint32_t m_headerBytes = 48;
        std::uint32_t m_headCount = 0;
        std::uint32_t m_slotsPerHead = 0;
        std::uint32_t m_voteHeadCount = 0;
        std::uint32_t m_minHeadCount = 0;
        std::uint32_t m_tagCount = 0;
        std::uint64_t m_generationFingerprint = 0;
        std::uint64_t m_bodyFingerprint = 0;
    };
#pragma pack(pop)
    static_assert(sizeof(Header) == 48,
                  "LimitedTagSupport header layout changed");

    void Reset()
    {
        m_header = Header();
        m_tags.clear();
        m_headsByTag.clear();
    }

    bool Initialize(
        SizeType p_headCount,
        int p_slotsPerHead,
        int p_voteHeadCount,
        int p_minHeadCount,
        std::uint64_t p_generationFingerprint)
    {
        Reset();
        if (p_headCount <= 0 || p_slotsPerHead <= 0 ||
            p_voteHeadCount <= 0 || p_minHeadCount <= 0 ||
            p_generationFingerprint == 0)
        {
            return false;
        }
        const size_t headCount = static_cast<size_t>(p_headCount);
        const size_t slots = static_cast<size_t>(p_slotsPerHead);
        if (headCount >
            (std::numeric_limits<size_t>::max)() / slots)
        {
            return false;
        }
        m_header.m_headCount =
            static_cast<std::uint32_t>(p_headCount);
        m_header.m_slotsPerHead =
            static_cast<std::uint32_t>(p_slotsPerHead);
        m_header.m_voteHeadCount =
            static_cast<std::uint32_t>(p_voteHeadCount);
        m_header.m_minHeadCount =
            static_cast<std::uint32_t>(p_minHeadCount);
        m_header.m_generationFingerprint =
            p_generationFingerprint;
        m_tags.assign(headCount * slots, EmptyTag);
        return true;
    }

    bool SetHeadTags(
        SizeType p_head,
        const std::vector<std::uint32_t>& p_tags)
    {
        if (p_head < 0 ||
            static_cast<std::uint32_t>(p_head) >=
                m_header.m_headCount ||
            p_tags.size() > m_header.m_slotsPerHead)
        {
            return false;
        }
        const size_t offset =
            static_cast<size_t>(p_head) *
            m_header.m_slotsPerHead;
        std::fill_n(
            m_tags.begin() + offset,
            m_header.m_slotsPerHead, EmptyTag);
        std::unordered_set<std::uint32_t> unique;
        for (size_t slot = 0; slot < p_tags.size(); ++slot)
        {
            if (p_tags[slot] == EmptyTag ||
                !unique.insert(p_tags[slot]).second)
            {
                return false;
            }
            m_tags[offset + slot] = p_tags[slot];
        }
        return true;
    }

    bool Supports(SizeType p_head, std::uint32_t p_tag) const
    {
        if (p_head < 0 ||
            static_cast<std::uint32_t>(p_head) >=
                m_header.m_headCount)
        {
            return false;
        }
        const size_t offset =
            static_cast<size_t>(p_head) *
            m_header.m_slotsPerHead;
        for (std::uint32_t slot = 0;
             slot < m_header.m_slotsPerHead; ++slot)
        {
            if (m_tags[offset + slot] == p_tag) return true;
        }
        return false;
    }

    std::uint32_t TagAt(
        SizeType p_head, int p_slot) const
    {
        if (p_head < 0 || p_slot < 0 ||
            static_cast<std::uint32_t>(p_head) >=
                m_header.m_headCount ||
            static_cast<std::uint32_t>(p_slot) >=
                m_header.m_slotsPerHead)
        {
            return EmptyTag;
        }
        return m_tags[
            static_cast<size_t>(p_head) *
                m_header.m_slotsPerHead +
            static_cast<size_t>(p_slot)];
    }

    std::uint32_t OwnTag(SizeType p_head) const
    {
        return TagAt(p_head, 0);
    }

    std::vector<std::uint32_t> HeadTags(SizeType p_head) const
    {
        std::vector<std::uint32_t> tags;
        if (p_head < 0 ||
            static_cast<std::uint32_t>(p_head) >=
                m_header.m_headCount)
        {
            return tags;
        }
        const size_t offset =
            static_cast<size_t>(p_head) *
            m_header.m_slotsPerHead;
        for (std::uint32_t slot = 0;
             slot < m_header.m_slotsPerHead; ++slot)
        {
            const std::uint32_t tag = m_tags[offset + slot];
            if (tag != EmptyTag) tags.push_back(tag);
        }
        return tags;
    }

    bool Validate(std::string* p_error = nullptr) const
    {
        auto fail = [&](const std::string& p_message) {
            if (p_error != nullptr) *p_error = p_message;
            return false;
        };
        if (m_header.m_magic != 0x3153544cU ||
            m_header.m_version != 1 ||
            m_header.m_headerBytes != sizeof(Header) ||
            m_header.m_headCount == 0 ||
            m_header.m_slotsPerHead == 0 ||
            m_header.m_voteHeadCount == 0 ||
            m_header.m_minHeadCount == 0 ||
            m_header.m_tagCount == 0 ||
            m_header.m_generationFingerprint == 0)
        {
            return fail("invalid limited-tag support header");
        }
        if (m_tags.size() !=
            static_cast<size_t>(m_header.m_headCount) *
                m_header.m_slotsPerHead)
        {
            return fail("limited-tag support body size mismatch");
        }
        if (static_cast<std::uint64_t>(
                m_header.m_tagCount) >
            static_cast<std::uint64_t>(
                m_header.m_headCount) *
                m_header.m_slotsPerHead)
        {
            return fail(
                "limited-tag support tag count exceeds body capacity");
        }

        std::unordered_map<std::uint32_t, std::uint32_t> coverage;
        for (std::uint32_t head = 0;
             head < m_header.m_headCount; ++head)
        {
            std::unordered_set<std::uint32_t> unique;
            const size_t offset =
                static_cast<size_t>(head) *
                m_header.m_slotsPerHead;
            for (std::uint32_t slot = 0;
                 slot < m_header.m_slotsPerHead; ++slot)
            {
                const std::uint32_t tag = m_tags[offset + slot];
                if (tag == EmptyTag) continue;
                if (!unique.insert(tag).second)
                {
                    return fail(
                        "duplicate tag in limited-tag head support");
                }
                ++coverage[tag];
            }
        }
        if (coverage.size() != m_header.m_tagCount)
        {
            return fail("limited-tag support tag count mismatch");
        }
        for (const auto& entry : coverage)
        {
            if (entry.second < m_header.m_minHeadCount)
            {
                return fail(
                    "limited-tag support coverage below minimum");
            }
        }
        return true;
    }

    bool Finalize(std::string* p_error = nullptr)
    {
        std::unordered_set<std::uint32_t> tags;
        for (std::uint32_t tag : m_tags)
        {
            if (tag != EmptyTag) tags.insert(tag);
        }
        m_header.m_tagCount =
            static_cast<std::uint32_t>(tags.size());
        m_header.m_bodyFingerprint = BodyFingerprint(m_tags);
        if (!Validate(p_error)) return false;
        RebuildHeadIndex();
        return true;
    }

    bool Save(const std::string& p_path, std::string* p_error = nullptr)
    {
        if (!Finalize(p_error)) return false;

        const std::string temporary = p_path + ".tmp";
        std::ofstream output(
            temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            if (p_error != nullptr)
                *p_error = "cannot open limited-tag support output";
            return false;
        }
        output.write(
            reinterpret_cast<const char*>(&m_header),
            sizeof(m_header));
        output.write(
            reinterpret_cast<const char*>(m_tags.data()),
            static_cast<std::streamsize>(
                m_tags.size() * sizeof(std::uint32_t)));
        output.close();
        if (!output)
        {
            std::remove(temporary.c_str());
            if (p_error != nullptr)
                *p_error = "cannot write limited-tag support output";
            return false;
        }
        std::remove(p_path.c_str());
        if (std::rename(temporary.c_str(), p_path.c_str()) != 0)
        {
            std::remove(temporary.c_str());
            if (p_error != nullptr)
                *p_error = "cannot publish limited-tag support output";
            return false;
        }
        return true;
    }

    bool Load(
        const std::string& p_path,
        SizeType p_expectedHeadCount,
        int p_expectedSlotsPerHead,
        int p_expectedVoteHeadCount,
        int p_expectedMinHeadCount,
        std::uint64_t p_expectedGeneration,
        std::string* p_error = nullptr)
    {
        Reset();
        std::ifstream input(p_path, std::ios::binary | std::ios::ate);
        if (!input)
        {
            if (p_error != nullptr)
                *p_error = "cannot open limited-tag support input";
            return false;
        }
        const std::streamoff fileBytes = input.tellg();
        input.seekg(0);
        input.read(
            reinterpret_cast<char*>(&m_header),
            sizeof(m_header));
        if (!input ||
            m_header.m_magic != 0x3153544cU ||
            m_header.m_version != 1 ||
            m_header.m_headerBytes != sizeof(Header) ||
            m_header.m_headCount !=
                static_cast<std::uint32_t>(p_expectedHeadCount) ||
            m_header.m_slotsPerHead !=
                static_cast<std::uint32_t>(p_expectedSlotsPerHead) ||
            m_header.m_voteHeadCount !=
                static_cast<std::uint32_t>(p_expectedVoteHeadCount) ||
            m_header.m_minHeadCount !=
                static_cast<std::uint32_t>(p_expectedMinHeadCount) ||
            m_header.m_generationFingerprint !=
                p_expectedGeneration)
        {
            if (p_error != nullptr)
                *p_error =
                    "limited-tag support configuration mismatch";
            Reset();
            return false;
        }
        const size_t tagCount =
            static_cast<size_t>(m_header.m_headCount) *
            m_header.m_slotsPerHead;
        const std::streamoff expectedBytes =
            static_cast<std::streamoff>(sizeof(Header)) +
            static_cast<std::streamoff>(
                tagCount * sizeof(std::uint32_t));
        if (fileBytes != expectedBytes)
        {
            if (p_error != nullptr)
                *p_error = "limited-tag support file size mismatch";
            Reset();
            return false;
        }
        m_tags.resize(tagCount);
        input.read(
            reinterpret_cast<char*>(m_tags.data()),
            static_cast<std::streamsize>(
                tagCount * sizeof(std::uint32_t)));
        if (!input ||
            BodyFingerprint(m_tags) !=
                m_header.m_bodyFingerprint)
        {
            if (p_error != nullptr && p_error->empty())
                *p_error =
                    "limited-tag support body fingerprint mismatch";
            Reset();
            return false;
        }
        if (!Validate(p_error))
        {
            Reset();
            return false;
        }
        RebuildHeadIndex();
        return true;
    }

    SizeType HeadCount() const
    {
        return static_cast<SizeType>(m_header.m_headCount);
    }

    int SlotsPerHead() const
    {
        return static_cast<int>(m_header.m_slotsPerHead);
    }

    std::uint64_t GenerationFingerprint() const
    {
        return m_header.m_generationFingerprint;
    }

    const std::unordered_map<
        std::uint32_t,
        std::vector<SizeType>>& TagHeads() const
    {
        return m_headsByTag;
    }

private:
    void RebuildHeadIndex()
    {
        m_headsByTag.clear();
        m_headsByTag.reserve(
            static_cast<size_t>(
                m_header.m_tagCount) *
                2 + 1);
        for (std::uint32_t head = 0;
             head < m_header.m_headCount; ++head)
        {
            const size_t offset =
                static_cast<size_t>(head) *
                m_header.m_slotsPerHead;
            for (std::uint32_t slot = 0;
                 slot < m_header.m_slotsPerHead;
                 ++slot)
            {
                const std::uint32_t tag =
                    m_tags[offset + slot];
                if (tag != EmptyTag)
                {
                    m_headsByTag[tag].push_back(
                        static_cast<SizeType>(
                            head));
                }
            }
        }
    }

    static std::uint64_t BodyFingerprint(
        const std::vector<std::uint32_t>& p_tags)
    {
        std::uint64_t hash = 1469598103934665603ULL;
        const auto* bytes =
            reinterpret_cast<const std::uint8_t*>(p_tags.data());
        const size_t byteCount =
            p_tags.size() * sizeof(std::uint32_t);
        for (size_t i = 0; i < byteCount; ++i)
        {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    Header m_header;
    std::vector<std::uint32_t> m_tags;
    std::unordered_map<
        std::uint32_t,
        std::vector<SizeType>> m_headsByTag;
};

} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_LIMITEDTAGSUPPORT_H_
