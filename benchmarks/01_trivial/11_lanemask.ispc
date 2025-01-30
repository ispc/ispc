// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export uniform int get_ispc_program_count() { return TARGET_WIDTH; }

export void lanemask_test(uniform int *uniform src, uniform uint64 *uniform dst, uniform int count) {
    for (uniform int i = 0, j = 0; i < count; i += programCount, j++) {
        varying int p = src[i + programIndex];
        uniform uint64 mask = 0;
        if (p) {
            mask = lanemask();
        }
        dst[j] = mask;
    }
}
