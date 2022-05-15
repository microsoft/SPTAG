#pragma once

#include "inc/Core/Common.h"
#include "inc/Core/Common/cuda/params.h"

int OPQRotationUpdate(float* svd_mat, float* rotation, SPTAG::SizeType dim);
