// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// [int32, int64] x [uniform, varying]
#define op(arg) count_trailing_zeros(arg)
#define cttz8(arg) op(op(op(op(op(op(op(op(arg))))))))

export void cttz_uniform_int32(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i++) {
        dst[i] = count_trailing_zeros(src[i]);
    }
}

export void cttz_varying_int32(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    foreach (i = 0...count) {
        dst[i] = count_trailing_zeros(src[i]);
    }
}

export void cttz_uniform_int64(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i++) {
        dst[i] = count_trailing_zeros(src[i]);
    }
}

export void cttz_varying_int64(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    foreach (i = 0...count) {
        dst[i] = count_trailing_zeros(src[i]);
    }
}

export void ctlz_uniform_int32(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i++) {
        dst[i] = count_leading_zeros(src[i]);
    }
}

export void ctlz_varying_int32(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    foreach (i = 0...count) {
        dst[i] = count_leading_zeros(src[i]);
    }
}

export void ctlz_uniform_int64(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i++) {
        dst[i] = count_leading_zeros(src[i]);
    }
}

export void ctlz_varying_int64(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    foreach (i = 0...count) {
        dst[i] = count_leading_zeros(src[i]);
    }
}
