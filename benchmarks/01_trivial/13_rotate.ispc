// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export uniform int width() { return programCount; }

// Simple rotate test - rotate value with offset -1
export void rotate_const_id(uniform int *uniform input, uniform int *uniform output, uniform int n) {
    foreach (i = 0 ... n) {
        int value = input[i];
        int rotated = rotate(value, -1);
        output[i] = rotated;
    }
}

// Macro for rotate functions with different data types
#define ROTATE_FUNC(TYPE)                                                                                              \
    export void rotate_##TYPE(uniform TYPE *uniform input, uniform TYPE *uniform output, uniform int n,                \
                              uniform int offset) {                                                                    \
        foreach (i = 0 ... n) {                                                                                        \
            TYPE value = input[i];                                                                                     \
            TYPE rotated = rotate(value, offset);                                                                      \
            output[i] = rotated;                                                                                       \
        }                                                                                                              \
    }

// Test rotating different data types
#ifdef HAS_FP16
ROTATE_FUNC(float16)
#endif

ROTATE_FUNC(float)
ROTATE_FUNC(double)
ROTATE_FUNC(int8)
ROTATE_FUNC(int16)
ROTATE_FUNC(int32)
ROTATE_FUNC(int64)
ROTATE_FUNC(uint8)
ROTATE_FUNC(uint16)
ROTATE_FUNC(uint32)
ROTATE_FUNC(uint64)
