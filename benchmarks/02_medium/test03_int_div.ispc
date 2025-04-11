// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export void ArrayIntDivFull(const uniform int src0[], const uniform int src1[], uniform int dst[],
                            const uniform int mask[], const uniform int n) {
    // generated data does not trigger a division by zero
    foreach (i = 0 ... n) {
        dst[i] = src0[i] / src1[i];
    }
}

export void ArrayIntDivMask(const uniform int src0[], const uniform int src1[], uniform int dst[],
                            const uniform int mask[], const uniform int n) {
    foreach (i = 0 ... n) {
        dst[i] = (mask[i] == 1) ? (src0[i] / src1[i]) : 0;
    }
}
