#include "inc/Helper/DiskIO.h"

std::function<void(SPTAG::Helper::AsyncReadRequest*)> SPTAG::Helper::DiskPriorityIO::g_fCleanup = [](SPTAG::Helper::AsyncReadRequest* req) {};