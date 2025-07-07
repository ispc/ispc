// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Test reduce_equal function performance with different data types
// Using macros to reduce code duplication

export uniform int width() { return programCount; }

#define REDUCE_EQUAL_FUNC(T, name)                                                                                     \
    export void reduce_equal_##name(uniform T *uniform src, uniform bool *uniform dst, uniform int count) {            \
        foreach (i = 0...count) {                                                                                      \
            dst[i] = reduce_equal(src[i]);                                                                             \
        }                                                                                                              \
    }

// Generate functions for all supported data types
REDUCE_EQUAL_FUNC(int32, int32)
REDUCE_EQUAL_FUNC(int64, int64)
REDUCE_EQUAL_FUNC(float, float)
REDUCE_EQUAL_FUNC(double, double)
REDUCE_EQUAL_FUNC(int8, int8)
REDUCE_EQUAL_FUNC(int16, int16)
#ifdef HAS_FP16
REDUCE_EQUAL_FUNC(float16, float16)
#endif
