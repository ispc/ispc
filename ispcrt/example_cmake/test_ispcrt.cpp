// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// ispcrt
#include "ispcrt.h"

int main() {
    ISPCRTDevice device = ispcrtGetDevice(ISPCRT_DEVICE_TYPE_AUTO, 0);
    ispcrtRelease(device);

    return 0;
}