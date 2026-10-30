// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Copies of structs of byte fields (#2778), contiguous, indexed and masked, and the same copies through integer types
// of the same size as references.
struct Bytes8 {
    uint8 b[8];
};
struct Bytes12 {
    uint8 b[12];
};
struct Bytes16 {
    uint8 b[16];
};
struct Int32x3 {
    int32 v[3];
};
struct Int64x2 {
    int64 v[2];
};

#define COPIES(NAME, T)                                                                                                \
    export void copy_##NAME(uniform T dst[], const uniform T src[], uniform int n) {                                   \
        foreach (i = 0 ... n) {                                                                                        \
            dst[i] = src[i];                                                                                           \
        }                                                                                                              \
    }                                                                                                                  \
    export void copy_indexed_##NAME(uniform T dst[], const uniform T src[], const uniform int idx[], uniform int n) {  \
        foreach (i = 0 ... n) {                                                                                        \
            int k = idx[i];                                                                                            \
            dst[k] = src[k];                                                                                           \
        }                                                                                                              \
    }                                                                                                                  \
    export void copy_masked_##NAME(uniform T dst[], const uniform T src[], const uniform int8 on[], uniform int n) {   \
        foreach (i = 0 ... n) {                                                                                        \
            if (on[i]) {                                                                                               \
                dst[i] = src[i];                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    }

COPIES(bytes8, Bytes8)
COPIES(int64, int64)
COPIES(bytes12, Bytes12)
COPIES(int32x3, Int32x3)
COPIES(bytes16, Bytes16)
COPIES(int64x2, Int64x2)
