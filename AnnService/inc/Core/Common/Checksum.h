// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_CHECKSUM_H_
#define _SPTAG_COMMON_CHECKSUM_H_

#include <string>
#include "inc/Core/Common.h"

namespace SPTAG
{
typedef uint8_t ChecksumType;

namespace COMMON
{
class Checksum
{
  public:
    Checksum() : m_type(0), m_seed(0), m_skip(false)
    {
    }

    void Initialize(bool p_skip, uint8_t p_type, int p_seed = 0)
    {
        m_type = p_type;
        m_seed = p_seed;
        m_skip = p_skip;
    }

    ChecksumType CalcChecksum(const char *p_data, int p_length)
    {
        uint8_t cs = m_seed;
        for (int i = 0; i < p_length; i++)
            cs ^= p_data[i];
        return cs;
    }

    ChecksumType AppendChecksum(ChecksumType p_checksum, const char* p_data, int p_length)
    {
        for (int i = 0; i < p_length; i++)
            p_checksum ^= p_data[i];
        return p_checksum;
    }

    bool ValidateChecksum(const char *p_data, int p_length, ChecksumType p_checksum)
    {
        if (m_skip) return true;
        return (CalcChecksum(p_data, p_length) == p_checksum);
    }

  private:
    uint8_t m_type;
    int m_seed;
    bool m_skip;
};
} // namespace COMMON
} // namespace SPTAG

#endif // _SPTAG_COMMON_CHECKSUM_H_