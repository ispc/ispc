// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

extern "C" __regcall float ISPC_CALLEE(uniform float A[], float b, uniform int i) {
    if (programIndex % 2 == 0) {
        return A[programIndex + i] + b;
    } else {
        return A[programIndex + i] - b;
    }
}