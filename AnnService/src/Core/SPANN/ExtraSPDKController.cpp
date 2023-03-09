// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/ExtraSPDKController.h"

namespace SPTAG::SPANN
{

std::unique_ptr<char[]> SPDKIO::BlockController::m_memBuffer;

bool SPDKIO::BlockController::Initialize() {
    const char* kUseMemImplEnvStr = getenv(kUseMemImplEnv);
    m_useMemImpl = kUseMemImplEnvStr && !strcmp(kUseMemImplEnvStr, "1");
    if (m_useMemImpl) {
        if (m_memBuffer == nullptr) {
            m_memBuffer.reset(new char[kMemImplMaxNumBlocks * PageSize]);
        }
        for (AddressType i = 0; i < kMemImplMaxNumBlocks; i++) {
            m_blockAddresses.push(i);
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::Initialize failed\n");
        return false;
    }
}

// get p_size blocks from front, and fill in p_data array
bool SPDKIO::BlockController::GetBlocks(AddressType* p_data, int p_size) {
    AddressType currBlockAddress = 0;
    if (m_useMemImpl) {
        for (int i = 0; i < p_size; i++) {
            while (!m_blockAddresses.try_pop(currBlockAddress));
            p_data[i] = currBlockAddress;
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::GetBlocks failed\n");
        return false;
    }
}

// release p_size blocks, put them at the end of the queue
bool SPDKIO::BlockController::ReleaseBlocks(AddressType* p_data, int p_size) {
    if (m_useMemImpl) {
        for (int i = 0; i < p_size; i++) {
            m_blockAddresses.push(p_data[i]);
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::ReleaseBlocks failed\n");
        return false;
    }
}

// read a posting list. p_data[0] is the total data size, 
// p_data[1], p_data[2], ..., p_data[((p_data[0] + PageSize - 1) >> PageSizeEx)] are the addresses of the blocks
// concat all the block contents together into p_value string.
bool SPDKIO::BlockController::ReadBlocks(AddressType* p_data, std::string* p_value) {
    if (m_useMemImpl) {
        p_value->resize(p_data[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;
        while (currOffset < p_data[0]) {
            AddressType readSize = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
            memcpy(p_value->data() + currOffset, m_memBuffer.get() + p_data[dataIdx] * PageSize, readSize);
            currOffset += PageSize;
            dataIdx++;
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::ReadBlocks single failed\n");
        return false;
    }
}

// parallel read a list of posting lists.
bool SPDKIO::BlockController::ReadBlocks(std::vector<AddressType*>& p_data, std::vector<std::string>* p_values) {
    if (m_useMemImpl) {
        p_values->resize(p_data.size());
        for (size_t i = 0; i < p_data.size(); i++) {
            ReadBlocks(p_data[i], &((*p_values)[i]));
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::ReadBlocks batch failed\n");
        return false;
    }
}

// write p_value into p_size blocks start from p_data
bool SPDKIO::BlockController::WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value) {
    if (m_useMemImpl) {
        for (int i = 0; i < p_size; i++) {
            memcpy(m_memBuffer.get() + p_data[i] * PageSize, p_value.data() + i * PageSize, PageSize);
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::ReadBlocks single failed\n");
        return false;
    }
}

bool SPDKIO::BlockController::ShutDown() {
    if (m_useMemImpl) {
        while (!m_blockAddresses.empty()) {
            AddressType currBlockAddress;
            m_blockAddresses.try_pop(currBlockAddress);
        }
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::ShutDown failed\n");
        return false;
    }
}

}
