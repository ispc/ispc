// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export uniform int width() { return programCount; }

// Macro for shift functions with different data types
#define SHIFT_FUNC(TYPE)                                                                                               \
    export void shift_##TYPE(uniform TYPE *uniform input, uniform TYPE *uniform output, uniform int n,                 \
                             uniform int offset) {                                                                     \
        foreach (i = 0 ... n) {                                                                                        \
            TYPE value = input[i];                                                                                     \
            TYPE shifted = shift(value, offset);                                                                       \
            output[i] = shifted;                                                                                       \
        }                                                                                                              \
    }

// Macro for const shift functions with different data types
#define SHIFT_CONST_FUNC(TYPE)                                                                                         \
    export void shift_const_##TYPE(uniform TYPE *uniform input, uniform TYPE *uniform output, uniform int n) {         \
        foreach (i = 0 ... n) {                                                                                        \
            TYPE value = input[i];                                                                                     \
            TYPE shifted = shift(value, -1);                                                                           \
            output[i] = shifted;                                                                                       \
        }                                                                                                              \
    }

// Test shifting different data types
#ifdef HAS_FP16
SHIFT_FUNC(float16)
SHIFT_CONST_FUNC(float16)
#endif

SHIFT_FUNC(float)
SHIFT_CONST_FUNC(float)
SHIFT_FUNC(double)
SHIFT_CONST_FUNC(double)
SHIFT_FUNC(int8)
SHIFT_CONST_FUNC(int8)
SHIFT_FUNC(int16)
SHIFT_CONST_FUNC(int16)
SHIFT_FUNC(int32)
SHIFT_CONST_FUNC(int32)
SHIFT_FUNC(int64)
SHIFT_CONST_FUNC(int64)
