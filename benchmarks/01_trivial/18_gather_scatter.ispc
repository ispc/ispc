// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

export uniform int width() { return programCount; }

// Random patterns: dst[i] = src[idx[i]] and dst[idx[i]] = src[i], with an index array of type IT.
#define GATHER_RANDOM(ET, IT)                                                                                          \
    export void gather_random_##ET##_##IT(uniform ET *uniform src, uniform IT *uniform idx, uniform ET *uniform dst,   \
                                          uniform int n) {                                                             \
        foreach (i = 0 ... n) {                                                                                        \
            dst[i] = src[idx[i]];                                                                                      \
        }                                                                                                              \
    }

#define SCATTER_RANDOM(ET, IT)                                                                                         \
    export void scatter_random_##ET##_##IT(uniform ET *uniform src, uniform IT *uniform idx, uniform ET *uniform dst,  \
                                           uniform int n) {                                                            \
        foreach (i = 0 ... n) {                                                                                        \
            dst[idx[i]] = src[i];                                                                                      \
        }                                                                                                              \
    }

// The same with a partial execution mask: only the elements with on[i] != 0.
#define GATHER_RANDOM_MASKED(ET, IT)                                                                                   \
    export void gather_random_masked_##ET##_##IT(uniform ET *uniform src, uniform IT *uniform idx,                     \
                                                 uniform int8 *uniform on, uniform ET *uniform dst, uniform int n) {   \
        foreach (i = 0 ... n) {                                                                                        \
            if (on[i]) {                                                                                               \
                dst[i] = src[idx[i]];                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    }

#define SCATTER_RANDOM_MASKED(ET, IT)                                                                                  \
    export void scatter_random_masked_##ET##_##IT(uniform ET *uniform src, uniform IT *uniform idx,                    \
                                                  uniform int8 *uniform on, uniform ET *uniform dst, uniform int n) {  \
        foreach (i = 0 ... n) {                                                                                        \
            if (on[i]) {                                                                                               \
                dst[idx[i]] = src[i];                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    }

// Patterns the compiler can see: every block of 4 elements reads its first element, reverse order, stride 2 and 4.
#define GATHER_PATTERNS(ET)                                                                                            \
    export void gather_broadcast4_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {              \
        foreach (i = 0 ... n) {                                                                                        \
            dst[i] = src[i & ~3];                                                                                      \
        }                                                                                                              \
    }                                                                                                                  \
    export void gather_reverse_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {                 \
        foreach (i = 0 ... n) {                                                                                        \
            dst[i] = src[n - 1 - i];                                                                                   \
        }                                                                                                              \
    }                                                                                                                  \
    export void gather_stride2_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {                 \
        foreach (i = 0 ... n) {                                                                                        \
            dst[i] = src[2 * i];                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    export void gather_stride4_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {                 \
        foreach (i = 0 ... n) {                                                                                        \
            dst[i] = src[4 * i];                                                                                       \
        }                                                                                                              \
    }

#define SCATTER_PATTERNS(ET)                                                                                           \
    export void scatter_reverse_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {                \
        foreach (i = 0 ... n) {                                                                                        \
            dst[n - 1 - i] = src[i];                                                                                   \
        }                                                                                                              \
    }                                                                                                                  \
    export void scatter_stride2_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {                \
        foreach (i = 0 ... n) {                                                                                        \
            dst[2 * i] = src[i];                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    export void scatter_stride4_##ET(uniform ET *uniform src, uniform ET *uniform dst, uniform int n) {                \
        foreach (i = 0 ... n) {                                                                                        \
            dst[4 * i] = src[i];                                                                                       \
        }                                                                                                              \
    }

GATHER_RANDOM(int32, int32)
GATHER_RANDOM(int32, uint32)
GATHER_RANDOM(int32, int16)
GATHER_RANDOM(int32, uint16)
GATHER_RANDOM(int8, int32)
GATHER_RANDOM(int8, uint32)
GATHER_RANDOM(double, int32)
GATHER_RANDOM(double, uint32)

SCATTER_RANDOM(int32, int32)
SCATTER_RANDOM(int32, uint32)
SCATTER_RANDOM(int32, int16)
SCATTER_RANDOM(int8, int32)
SCATTER_RANDOM(double, int32)

GATHER_RANDOM_MASKED(int32, int32)
GATHER_RANDOM_MASKED(double, int32)
SCATTER_RANDOM_MASKED(int32, int32)

GATHER_PATTERNS(int8)
GATHER_PATTERNS(int32)
GATHER_PATTERNS(double)

SCATTER_PATTERNS(int32)
SCATTER_PATTERNS(double)
