// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/ExtraSPDKController.h"

namespace SPTAG::SPANN
{

std::unique_ptr<char[]> SPDKIO::BlockController::m_memBuffer;

void SPDKIO::BlockController::SpdkBdevEventCallback(enum spdk_bdev_event_type type, struct spdk_bdev *bdev, void *event_ctx) {
    fprintf(stderr, "SpdkBdevEventCallback: supported bdev event type %d\n", type);
}

void SPDKIO::BlockController::SpdkStart(void *arg) {
    SPDKIO::BlockController* ctrl = (SPDKIO::BlockController *)arg;

    fprintf(stdout, "SPDKIO::BlockController::SpdkStart: using bdev %s\n", ctrl->m_ssdBdevName);

    int rc = 0;
    ctrl->m_ssdSpdkBdev = NULL;
    ctrl->m_ssdSpdkBdevDesc = NULL;

    // Open bdev
    rc = spdk_bdev_open_ext(ctrl->m_ssdBdevName, true, SpdkBdevEventCallback, NULL, &ctrl->m_ssdSpdkBdevDesc);
    if (rc) {
        fprintf(stderr, "SPDKIO::BlockController::SpdkStart: spdk_bdev_open_ext failed, %d\n", rc);
        ctrl->m_ssdSpdkThreadStartFailed = true;
        spdk_app_stop(-1);
        return;
    }
    ctrl->m_ssdSpdkBdev = spdk_bdev_desc_get_bdev(ctrl->m_ssdSpdkBdevDesc);

    // Open I/O channel
    ctrl->m_ssdSpdkBdevIoChannel = spdk_bdev_get_io_channel(ctrl->m_ssdSpdkBdevDesc);
    if (ctrl->m_ssdSpdkBdevIoChannel == NULL) {
        fprintf(stderr, "SPDKIO::BlockController::SpdkStart: spdk_bdev_get_io_channel failed\n");
        spdk_bdev_close(ctrl->m_ssdSpdkBdevDesc);
        ctrl->m_ssdSpdkThreadStartFailed = true;
        spdk_app_stop(-1);
        return;
    }

    // Close I/O channel and bdev
    spdk_put_io_channel(ctrl->m_ssdSpdkBdevIoChannel);
    spdk_bdev_close(ctrl->m_ssdSpdkBdevDesc);
    fprintf(stdout, "SPDKIO::BlockController::SpdkStart: finalized\n");
    spdk_app_stop(0);

    ctrl->m_ssdSpdkThreadReady = true;
}

void* SPDKIO::BlockController::InitializeSpdk(void *arg) {
    SPDKIO::BlockController* ctrl = (SPDKIO::BlockController *)arg;

    struct spdk_app_opts opts;
    spdk_app_opts_init(&opts, sizeof(opts));
    opts.name = "spfresh";
    const char* spdkConf = getenv(kSpdkConfEnv);
    opts.json_config_file = spdkConf ? spdkConf : "";
    const char* spdkBdev = getenv(kSpdkBdevEnv);
    ctrl->m_ssdBdevName = spdkBdev? spdkBdev : "";

    int rc;
    rc = spdk_app_start(&opts, &SPTAG::SPANN::SPDKIO::BlockController::SpdkStart, arg);
    if (rc) {
        ctrl->m_ssdSpdkThreadStartFailed = true;
    } else {
        spdk_app_fini();
    }
    pthread_exit(NULL);
}

bool SPDKIO::BlockController::Initialize() {
    const char* useMemImplEnvStr = getenv(kUseMemImplEnv);
    m_useMemImpl = useMemImplEnvStr && !strcmp(useMemImplEnvStr, "1");
    const char* useSsdImplEnvStr = getenv(kUseSsdImplEnv);
    m_useSsdImpl = useSsdImplEnvStr && !strcmp(useSsdImplEnvStr, "1");
    if (m_useMemImpl) {
        if (m_memBuffer == nullptr) {
            m_memBuffer.reset(new char[kMemImplMaxNumBlocks * PageSize]);
        }
        for (AddressType i = 0; i < kMemImplMaxNumBlocks; i++) {
            m_blockAddresses.push(i);
        }
        return true;
    } else if (m_useSsdImpl) {
        pthread_create(&m_ssdSpdkTid, NULL, &InitializeSpdk, this);
        while (!m_ssdSpdkThreadReady && !m_ssdSpdkThreadStartFailed);
        if (m_ssdSpdkThreadStartFailed) {
            fprintf(stderr, "SPDKIO::BlockController::Initialize failed\n");
            return false;
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
    if (m_useMemImpl || m_useSsdImpl) {
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
    if (m_useMemImpl || m_useSsdImpl) {
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
    } else if (m_useSsdImpl) {
        p_value->resize(p_data[0]);
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
    } else if (m_useSsdImpl) {
        p_values->resize(p_data.size());
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
    } else if (m_useSsdImpl) {
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
    } else if (m_useSsdImpl) {
        spdk_app_start_shutdown();
        pthread_join(m_ssdSpdkTid, NULL);
        return true;
    } else {
        fprintf(stderr, "SPDKIO::BlockController::ShutDown failed\n");
        return false;
    }
}

}
