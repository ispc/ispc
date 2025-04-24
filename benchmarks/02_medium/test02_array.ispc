// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export void ArrayMean(const uniform float src[], uniform float &dst, const uniform int n) {
    float sum = 0.0f;
    foreach (i = 0 ... n) {
        sum += src[i];
    }
    dst = reduce_add(sum) / n;
}
