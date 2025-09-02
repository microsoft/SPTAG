// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GoogleTest main
#include "inc/Test.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

