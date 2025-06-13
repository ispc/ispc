// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export uniform int width() { return programCount; }

// Broadcast from different indices test
export void broadcast_varying_index(uniform int *uniform input, uniform int *uniform output, uniform int n,
                                    uniform int src_index) {
    foreach (i = 0 ... n) {
        int value = input[i];
        int broadcasted = broadcast(value, src_index);
        output[i] = broadcasted;
    }
}

// Broadcast with computation test
export void broadcast_with_computation(uniform float *uniform input, uniform float *uniform output, uniform int n) {
    foreach (i = 0 ... n) {
        float v = input[i];
        float c = v * v + 1.0f;
        float b = broadcast(c, 0);
        output[i] = b;
    }
}

// Macro for broadcast functions with different data types
#define BROADCAST_FUNC(TYPE)                                                                                           \
    export void broadcast_##TYPE(uniform TYPE *uniform input, uniform TYPE *uniform output, uniform int n) {           \
        foreach (i = 0 ... n) {                                                                                        \
            TYPE value = input[i];                                                                                     \
            TYPE broadcasted = broadcast(value, 0);                                                                    \
            output[i] = broadcasted;                                                                                   \
        }                                                                                                              \
    }

// Test broadcasting different data types
BROADCAST_FUNC(float)
BROADCAST_FUNC(double)
BROADCAST_FUNC(int64)
BROADCAST_FUNC(int8)
BROADCAST_FUNC(int16)
BROADCAST_FUNC(int32)
BROADCAST_FUNC(uint8)
BROADCAST_FUNC(uint16)
BROADCAST_FUNC(uint32)
BROADCAST_FUNC(uint64)

#ifdef HAS_FP16
BROADCAST_FUNC(float16)
#endif
