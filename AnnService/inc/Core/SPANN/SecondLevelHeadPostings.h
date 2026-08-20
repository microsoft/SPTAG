// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_SECONDLEVELHEADPOSTINGS_H_
#define _SPTAG_SPANN_SECONDLEVELHEADPOSTINGS_H_

#include "inc/Core/Cache/PostingSignature.h"
#include "inc/Core/Common.h"
#include "inc/Helper/AtomicFile.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace SPTAG
{
namespace SPANN
{

class SecondLevelHeadPostings
{
public:
    using Member = std::uint32_t;
    using Signature = Cache::PostingBitmask;

#pragma pack(push, 1)
    struct Header
    {
        std::uint64_t m_magic = 0x325253324E4E4153ULL; // "SANN2SR2"
        std::uint32_t m_version = 2;
        std::uint32_t m_headerBytes = 88;
        std::uint32_t m_firstLevelHeadCount = 0;
        std::uint32_t m_secondLevelHeadCount = 0;
        std::uint32_t m_replicaCount = 0;
        std::uint32_t m_memberBytes = sizeof(Member);
        std::uint32_t m_signatureBytes = sizeof(Signature);
        std::uint32_t m_reserved = 0;
        std::uint64_t m_memberCount = 0;
        std::uint64_t m_firstLevelIDFingerprint = 0;
        std::uint64_t m_secondLevelIDFingerprint = 0;
        std::uint64_t m_limitedTagSupportFingerprint = 0;
        std::uint64_t m_bodyFingerprint = 0;
        std::uint64_t m_generationFingerprint = 0;
    };
#pragma pack(pop)

    static_assert(
        sizeof(Header) == 88,
        "second-level posting header layout changed");
    static_assert(
        sizeof(Signature) == 32,
        "second-level signature must match PostingBitmask");

    void Reset()
    {
        m_header = Header();
        m_offsets.clear();
        m_members.clear();
        m_signatures.clear();
    }

    static std::uint64_t FingerprintIDs(
        const std::uint64_t* p_ids, size_t p_count)
    {
        return HashBytes(
            kFNVOffset, p_ids,
            p_count * sizeof(std::uint64_t));
    }

    static std::uint64_t BeginIDFingerprint()
    {
        return kFNVOffset;
    }

    static std::uint64_t AddIDFingerprint(
        std::uint64_t p_hash, std::uint64_t p_id)
    {
        return HashBytes(
            p_hash, &p_id, sizeof(p_id));
    }

    bool Initialize(
        SizeType p_firstLevelHeadCount,
        SizeType p_secondLevelHeadCount,
        int p_replicaCount,
        std::uint64_t p_firstLevelIDFingerprint,
        std::uint64_t p_limitedTagSupportFingerprint,
        const std::vector<std::uint64_t>& p_secondLevelToFirst,
        std::vector<std::uint64_t> p_offsets,
        std::vector<Member> p_members,
        std::vector<Signature> p_signatures,
        std::string* p_error = nullptr)
    {
        Reset();
        if (p_firstLevelHeadCount <= 0 ||
            p_secondLevelHeadCount <= 0 ||
            p_replicaCount <= 0 ||
            p_firstLevelIDFingerprint == 0 ||
            p_limitedTagSupportFingerprint == 0 ||
            static_cast<std::uint64_t>(
                p_firstLevelHeadCount) >
                (std::numeric_limits<std::uint32_t>::max)() ||
            static_cast<std::uint64_t>(
                p_secondLevelHeadCount) >
                (std::numeric_limits<std::uint32_t>::max)())
        {
            return Fail(
                p_error,
                "invalid second-level posting configuration");
        }

        m_header.m_firstLevelHeadCount =
            static_cast<std::uint32_t>(
                p_firstLevelHeadCount);
        m_header.m_secondLevelHeadCount =
            static_cast<std::uint32_t>(
                p_secondLevelHeadCount);
        m_header.m_replicaCount =
            static_cast<std::uint32_t>(p_replicaCount);
        m_header.m_memberCount =
            static_cast<std::uint64_t>(p_members.size());
        m_header.m_firstLevelIDFingerprint =
            p_firstLevelIDFingerprint;
        m_header.m_limitedTagSupportFingerprint =
            p_limitedTagSupportFingerprint;
        m_header.m_secondLevelIDFingerprint =
            FingerprintIDs(
                p_secondLevelToFirst.data(),
                p_secondLevelToFirst.size());
        m_offsets = std::move(p_offsets);
        m_members = std::move(p_members);
        m_signatures = std::move(p_signatures);
        m_header.m_bodyFingerprint = BodyFingerprint();
        m_header.m_generationFingerprint =
            GenerationFingerprint(m_header);

        if (!Validate(
                p_secondLevelToFirst, p_error))
        {
            Reset();
            return false;
        }
        return true;
    }

    bool Save(
        const std::string& p_path,
        std::string* p_error = nullptr) const
    {
        if (!Loaded())
            return Fail(p_error, "second-level postings are empty");

        const std::string temporary = p_path + ".tmp";
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc);
        if (!output)
            return Fail(
                p_error,
                "cannot open second-level posting output");

        output.write(
            reinterpret_cast<const char*>(&m_header),
            sizeof(m_header));
        output.write(
            reinterpret_cast<const char*>(m_offsets.data()),
            static_cast<std::streamsize>(
                m_offsets.size() *
                sizeof(std::uint64_t)));
        output.write(
            reinterpret_cast<const char*>(m_members.data()),
            static_cast<std::streamsize>(
                m_members.size() * sizeof(Member)));
        output.write(
            reinterpret_cast<const char*>(
                m_signatures.data()),
            static_cast<std::streamsize>(
                m_signatures.size() *
                sizeof(Signature)));
        output.close();
        if (!output)
        {
            std::remove(temporary.c_str());
            return Fail(
                p_error,
                "cannot write second-level posting output");
        }

        if (!Helper::AtomicReplaceFile(
                temporary, p_path))
        {
            std::remove(temporary.c_str());
            return Fail(
                p_error,
                "cannot publish second-level posting output");
        }
        return true;
    }

    bool Load(
        const std::string& p_path,
        SizeType p_expectedFirstLevelHeadCount,
        SizeType p_expectedSecondLevelHeadCount,
        int p_expectedReplicaCount,
        std::uint64_t p_expectedFirstLevelIDFingerprint,
        std::uint64_t p_expectedLimitedTagSupportFingerprint,
        std::uint64_t p_expectedGeneration,
        const std::vector<std::uint64_t>&
            p_secondLevelToFirst,
        std::string* p_error = nullptr)
    {
        Reset();
        std::ifstream input(
            p_path,
            std::ios::binary | std::ios::ate);
        if (!input)
            return Fail(
                p_error,
                "cannot open second-level posting input");

        const std::streamoff fileBytes = input.tellg();
        input.seekg(0);
        input.read(
            reinterpret_cast<char*>(&m_header),
            sizeof(m_header));
        if (!input ||
            m_header.m_magic != Header().m_magic ||
            m_header.m_version != Header().m_version ||
            m_header.m_headerBytes != sizeof(Header) ||
            m_header.m_memberBytes != sizeof(Member) ||
            m_header.m_signatureBytes !=
                sizeof(Signature) ||
            m_header.m_reserved != 0 ||
            m_header.m_firstLevelHeadCount !=
                static_cast<std::uint32_t>(
                    p_expectedFirstLevelHeadCount) ||
            m_header.m_secondLevelHeadCount !=
                static_cast<std::uint32_t>(
                    p_expectedSecondLevelHeadCount) ||
            m_header.m_replicaCount !=
                static_cast<std::uint32_t>(
                    p_expectedReplicaCount) ||
            m_header.m_firstLevelIDFingerprint !=
                p_expectedFirstLevelIDFingerprint ||
            m_header.m_limitedTagSupportFingerprint !=
                p_expectedLimitedTagSupportFingerprint ||
            m_header.m_secondLevelIDFingerprint !=
                FingerprintIDs(
                    p_secondLevelToFirst.data(),
                    p_secondLevelToFirst.size()) ||
            m_header.m_generationFingerprint !=
                p_expectedGeneration)
        {
            Reset();
            return Fail(
                p_error,
                "second-level posting configuration mismatch");
        }

        const std::uint32_t expectedReplicas =
            (std::min)(
                m_header.m_replicaCount,
                m_header.m_secondLevelHeadCount);
        const std::uint64_t expectedMembers =
            static_cast<std::uint64_t>(
                m_header.m_firstLevelHeadCount) *
            expectedReplicas;
        if (m_header.m_memberCount != expectedMembers)
        {
            Reset();
            return Fail(
                p_error,
                "second-level posting replica count mismatch");
        }

        const std::uint64_t offsetCount =
            static_cast<std::uint64_t>(
                m_header.m_secondLevelHeadCount) + 1;
        std::uint64_t offsetBytes = 0;
        std::uint64_t memberBytes = 0;
        std::uint64_t signatureBytes = 0;
        std::uint64_t expectedBytes = 0;
        if (!CheckedMultiply(
                offsetCount,
                sizeof(std::uint64_t),
                offsetBytes) ||
            !CheckedMultiply(
                m_header.m_memberCount,
                sizeof(Member),
                memberBytes) ||
            !CheckedMultiply(
                m_header.m_secondLevelHeadCount,
                sizeof(Signature),
                signatureBytes) ||
            !CheckedAdd(
                sizeof(Header),
                offsetBytes,
                expectedBytes) ||
            !CheckedAdd(
                expectedBytes,
                memberBytes,
                expectedBytes) ||
            !CheckedAdd(
                expectedBytes,
                signatureBytes,
                expectedBytes) ||
            expectedBytes >
                static_cast<std::uint64_t>(
                    (std::numeric_limits<
                        std::streamoff>::max)()) ||
            offsetBytes >
                static_cast<std::uint64_t>(
                    (std::numeric_limits<
                        std::streamsize>::max)()) ||
            memberBytes >
                static_cast<std::uint64_t>(
                    (std::numeric_limits<
                        std::streamsize>::max)()) ||
            signatureBytes >
                static_cast<std::uint64_t>(
                    (std::numeric_limits<
                        std::streamsize>::max)()) ||
            fileBytes !=
                static_cast<std::streamoff>(
                    expectedBytes) ||
            offsetCount >
                static_cast<std::uint64_t>(
                    m_offsets.max_size()) ||
            m_header.m_memberCount >
                static_cast<std::uint64_t>(
                    m_members.max_size()) ||
            m_header.m_secondLevelHeadCount >
                static_cast<std::uint64_t>(
                    m_signatures.max_size()))
        {
            Reset();
            return Fail(
                p_error,
                "second-level posting file size mismatch");
        }

        try
        {
            m_offsets.resize(
                static_cast<size_t>(offsetCount));
            m_members.resize(
                static_cast<size_t>(
                    m_header.m_memberCount));
            m_signatures.resize(
                static_cast<size_t>(
                    m_header
                        .m_secondLevelHeadCount));
        }
        catch (const std::bad_alloc&)
        {
            Reset();
            return Fail(
                p_error,
                "cannot allocate second-level posting body");
        }
        catch (const std::length_error&)
        {
            Reset();
            return Fail(
                p_error,
                "second-level posting body exceeds vector limits");
        }
        input.read(
            reinterpret_cast<char*>(m_offsets.data()),
            static_cast<std::streamsize>(
                m_offsets.size() *
                sizeof(std::uint64_t)));
        input.read(
            reinterpret_cast<char*>(m_members.data()),
            static_cast<std::streamsize>(
                m_members.size() * sizeof(Member)));
        input.read(
            reinterpret_cast<char*>(
                m_signatures.data()),
            static_cast<std::streamsize>(
                m_signatures.size() *
                sizeof(Signature)));
        if (!input ||
            BodyFingerprint() !=
                m_header.m_bodyFingerprint ||
            GenerationFingerprint(m_header) !=
                m_header.m_generationFingerprint ||
            !Validate(
                p_secondLevelToFirst, p_error))
        {
            Reset();
            if (p_error != nullptr && p_error->empty())
                *p_error =
                    "second-level posting body mismatch";
            return false;
        }
        return true;
    }

    bool Loaded() const
    {
        return
            m_offsets.size() ==
                static_cast<size_t>(
                    m_header.m_secondLevelHeadCount) + 1 &&
            !m_members.empty() &&
            m_signatures.size() ==
                static_cast<size_t>(
                    m_header.m_secondLevelHeadCount);
    }

    SizeType FirstLevelHeadCount() const
    {
        return static_cast<SizeType>(
            m_header.m_firstLevelHeadCount);
    }

    SizeType SecondLevelHeadCount() const
    {
        return static_cast<SizeType>(
            m_header.m_secondLevelHeadCount);
    }

    int ReplicaCount() const
    {
        return static_cast<int>(
            m_header.m_replicaCount);
    }

    std::uint64_t MemberCount() const
    {
        return m_header.m_memberCount;
    }

    std::uint64_t GenerationFingerprint() const
    {
        return m_header.m_generationFingerprint;
    }

    const Member* Begin(SizeType p_secondLevelHead) const
    {
        if (p_secondLevelHead < 0 ||
            static_cast<std::uint32_t>(
                p_secondLevelHead) >=
                m_header.m_secondLevelHeadCount)
            return nullptr;
        return m_members.data() +
            m_offsets[
                static_cast<size_t>(
                    p_secondLevelHead)];
    }

    const Member* End(SizeType p_secondLevelHead) const
    {
        if (p_secondLevelHead < 0 ||
            static_cast<std::uint32_t>(
                p_secondLevelHead) >=
                m_header.m_secondLevelHeadCount)
            return nullptr;
        return m_members.data() +
            m_offsets[
                static_cast<size_t>(
                    p_secondLevelHead) + 1];
    }

    const Signature* SignatureAt(
        SizeType p_secondLevelHead) const
    {
        if (p_secondLevelHead < 0 ||
            static_cast<std::uint32_t>(
                p_secondLevelHead) >=
                m_header.m_secondLevelHeadCount ||
            m_signatures.size() !=
                static_cast<size_t>(
                    m_header.m_secondLevelHeadCount))
        {
            return nullptr;
        }
        return &m_signatures[
            static_cast<size_t>(
                p_secondLevelHead)];
    }

private:
    static constexpr std::uint64_t kFNVOffset =
        1469598103934665603ULL;
    static constexpr std::uint64_t kFNVPrime =
        1099511628211ULL;

    static bool CheckedMultiply(
        std::uint64_t p_left,
        std::uint64_t p_right,
        std::uint64_t& p_result)
    {
        if (p_right != 0 &&
            p_left >
                (std::numeric_limits<
                    std::uint64_t>::max)() /
                    p_right)
        {
            return false;
        }
        p_result = p_left * p_right;
        return true;
    }

    static bool CheckedAdd(
        std::uint64_t p_left,
        std::uint64_t p_right,
        std::uint64_t& p_result)
    {
        if (p_left >
            (std::numeric_limits<
                std::uint64_t>::max)() -
                p_right)
        {
            return false;
        }
        p_result = p_left + p_right;
        return true;
    }

    static bool Fail(
        std::string* p_error,
        const char* p_message)
    {
        if (p_error != nullptr) *p_error = p_message;
        return false;
    }

    static std::uint64_t HashBytes(
        std::uint64_t p_hash,
        const void* p_data,
        size_t p_bytes)
    {
        const auto* bytes =
            reinterpret_cast<const std::uint8_t*>(
                p_data);
        for (size_t i = 0; i < p_bytes; ++i)
        {
            p_hash ^= bytes[i];
            p_hash *= kFNVPrime;
        }
        return p_hash;
    }

    std::uint64_t BodyFingerprint() const
    {
        std::uint64_t hash = HashBytes(
            kFNVOffset, m_offsets.data(),
            m_offsets.size() *
                sizeof(std::uint64_t));
        hash = HashBytes(
            hash, m_members.data(),
            m_members.size() * sizeof(Member));
        return HashBytes(
            hash, m_signatures.data(),
            m_signatures.size() *
                sizeof(Signature));
    }

    static std::uint64_t GenerationFingerprint(
        const Header& p_header)
    {
        std::uint64_t hash = kFNVOffset;
        hash = HashBytes(
            hash,
            &p_header.m_firstLevelHeadCount,
            sizeof(p_header.m_firstLevelHeadCount));
        hash = HashBytes(
            hash,
            &p_header.m_secondLevelHeadCount,
            sizeof(p_header.m_secondLevelHeadCount));
        hash = HashBytes(
            hash,
            &p_header.m_replicaCount,
            sizeof(p_header.m_replicaCount));
        hash = HashBytes(
            hash,
            &p_header.m_signatureBytes,
            sizeof(p_header.m_signatureBytes));
        hash = HashBytes(
            hash,
            &p_header.m_memberCount,
            sizeof(p_header.m_memberCount));
        hash = HashBytes(
            hash,
            &p_header.m_firstLevelIDFingerprint,
            sizeof(
                p_header
                    .m_firstLevelIDFingerprint));
        hash = HashBytes(
            hash,
            &p_header.m_secondLevelIDFingerprint,
            sizeof(
                p_header
                    .m_secondLevelIDFingerprint));
        hash = HashBytes(
            hash,
            &p_header
                 .m_limitedTagSupportFingerprint,
            sizeof(
                p_header
                    .m_limitedTagSupportFingerprint));
        return HashBytes(
            hash,
            &p_header.m_bodyFingerprint,
            sizeof(p_header.m_bodyFingerprint));
    }

    bool Validate(
        const std::vector<std::uint64_t>&
            p_secondLevelToFirst,
        std::string* p_error) const
    {
        if (m_header.m_magic != Header().m_magic ||
            m_header.m_version != Header().m_version ||
            m_header.m_headerBytes != sizeof(Header) ||
            m_header.m_memberBytes != sizeof(Member) ||
            m_header.m_signatureBytes !=
                sizeof(Signature) ||
            m_header.m_reserved != 0 ||
            m_header.m_firstLevelHeadCount == 0 ||
            m_header.m_secondLevelHeadCount == 0 ||
            m_header.m_replicaCount == 0 ||
            m_header.m_firstLevelIDFingerprint == 0 ||
            m_header.m_secondLevelIDFingerprint == 0 ||
            m_header.m_limitedTagSupportFingerprint == 0 ||
            m_header.m_bodyFingerprint == 0 ||
            m_header.m_generationFingerprint == 0 ||
            p_secondLevelToFirst.size() !=
                m_header.m_secondLevelHeadCount ||
            m_offsets.size() !=
                static_cast<size_t>(
                    m_header.m_secondLevelHeadCount) + 1 ||
            m_offsets.front() != 0 ||
            m_offsets.back() !=
                m_header.m_memberCount ||
            m_members.size() !=
                static_cast<size_t>(
                    m_header.m_memberCount) ||
            m_signatures.size() !=
                static_cast<size_t>(
                    m_header
                        .m_secondLevelHeadCount))
        {
            return Fail(
                p_error,
                "invalid second-level posting header");
        }

        const std::uint32_t expectedReplicas =
            (std::min)(
                m_header.m_replicaCount,
                m_header.m_secondLevelHeadCount);
        const std::uint64_t expectedMembers =
            static_cast<std::uint64_t>(
                m_header.m_firstLevelHeadCount) *
            expectedReplicas;
        if (m_header.m_memberCount != expectedMembers)
            return Fail(
                p_error,
                "second-level posting replica count mismatch");

        std::vector<std::uint32_t> coverage(
            m_header.m_firstLevelHeadCount, 0);
        for (std::uint32_t second = 0;
             second <
                 m_header.m_secondLevelHeadCount;
             ++second)
        {
            const std::uint64_t begin =
                m_offsets[second];
            const std::uint64_t end =
                m_offsets[second + 1];
            if (begin >= end ||
                end > m_members.size())
            {
                return Fail(
                    p_error,
                    "empty or invalid second-level posting");
            }

            bool hasSelf = false;
            Member previous = 0;
            for (std::uint64_t offset = begin;
                 offset < end; ++offset)
            {
                const Member member =
                    m_members[
                        static_cast<size_t>(
                            offset)];
                if (member >=
                        m_header
                            .m_firstLevelHeadCount ||
                    (offset > begin &&
                     member <= previous))
                {
                    return Fail(
                        p_error,
                        "invalid second-level posting member order");
                }
                previous = member;
                ++coverage[member];
                if (member ==
                    p_secondLevelToFirst[second])
                    hasSelf = true;
            }
            if (!hasSelf)
                return Fail(
                    p_error,
                    "second-level head is absent from its own posting");
        }

        for (std::uint32_t count : coverage)
        {
            if (count != expectedReplicas)
                return Fail(
                    p_error,
                    "first-level head replica coverage mismatch");
        }
        return true;
    }

    Header m_header;
    std::vector<std::uint64_t> m_offsets;
    std::vector<Member> m_members;
    std::vector<Signature> m_signatures;
};

} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_SECONDLEVELHEADPOSTINGS_H_
