// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Moments of point records of 2, 3 or 4 floats in an array of structures: the sum of every coordinate and the sum of
// the squared norms. The gang reads the records in three ways: packed (record i), strided (record 2 * i) and indexed
// (record idx[i]). Each way has a gather variant (a varying struct access), a transpose variant (aos_to_soa* of
// contiguous records, or of records copied lane by lane for the indexed way) and, where the records are contiguous in
// memory, a wide variant that reads the floats as one contiguous array.
struct point2 {
    float c[2];
};
struct point3 {
    float c[3];
};
struct point4 {
    float c[4];
};

static inline void transpose2(uniform float buf[], float c[2]) { aos_to_soa2(buf, &c[0], &c[1]); }
static inline void transpose3(uniform float buf[], float c[3]) { aos_to_soa3(buf, &c[0], &c[1], &c[2]); }
static inline void transpose4(uniform float buf[], float c[4]) { aos_to_soa4(buf, &c[0], &c[1], &c[2], &c[3]); }

// out[0 .. N - 1] = the sums of the coordinates, out[N] = the sum of the squared norms.
#define STORE_MOMENTS(N)                                                                                               \
    for (uniform int k = 0; k < N; ++k)                                                                                \
        out[k] = reduce_add(s[k]);                                                                                     \
    out[N] = reduce_add(q);

#define ACCUMULATE(N, C)                                                                                               \
    for (uniform int k = 0; k < N; ++k) {                                                                              \
        s[k] += C[k];                                                                                                  \
        q += C[k] * C[k];                                                                                              \
    }

#define MOMENTS(N)                                                                                                     \
    export void moments_gather_packed_##N(const uniform point##N pts[], uniform int n, uniform float out[]) {          \
        float s[N] = {0}, q = 0;                                                                                       \
        foreach (i = 0 ... n) {                                                                                        \
            point##N p = pts[i];                                                                                       \
            ACCUMULATE(N, p.c)                                                                                         \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    export void moments_transpose_packed_##N(const uniform point##N pts[], uniform int n, uniform float out[]) {       \
        float s[N] = {0}, q = 0;                                                                                       \
        uniform int i = 0;                                                                                             \
        for (; i + programCount <= n; i += programCount) {                                                             \
            float c[N];                                                                                                \
            transpose##N((uniform float *uniform) & pts[i], c);                                                        \
            ACCUMULATE(N, c)                                                                                           \
        }                                                                                                              \
        foreach (j = i ... n) {                                                                                        \
            point##N p = pts[j];                                                                                       \
            ACCUMULATE(N, p.c)                                                                                         \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    /* The coordinates as one float array: float j is coordinate j % N. */                                             \
    export void moments_wide_packed_##N(const uniform point##N pts[], uniform int n, uniform float out[]) {            \
        const uniform float *uniform f = (const uniform float *uniform) & pts[0];                                      \
        float s[N] = {0}, q = 0;                                                                                       \
        foreach (j = 0 ... N * n) {                                                                                    \
            float v = f[j];                                                                                            \
            int k = j % N;                                                                                             \
            for (uniform int m = 0; m < N; ++m)                                                                        \
                s[m] += k == m ? v : 0;                                                                                \
            q += v * v;                                                                                                \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    /* Strided: records 0, 2, 4, ... of 2 * n records. */                                                              \
    export void moments_gather_strided_##N(const uniform point##N pts[], uniform int n, uniform float out[]) {         \
        float s[N] = {0}, q = 0;                                                                                       \
        foreach (i = 0 ... n) {                                                                                        \
            point##N p = pts[2 * i];                                                                                   \
            ACCUMULATE(N, p.c)                                                                                         \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    /* Transposes 2 * programCount contiguous records and keeps the even ones. */                                      \
    export void moments_transpose_strided_##N(const uniform point##N pts[], uniform int n, uniform float out[]) {      \
        float s[N] = {0}, q = 0;                                                                                       \
        uniform int i = 0;                                                                                             \
        for (; i + programCount <= n; i += programCount) {                                                             \
            float lo[N], hi[N];                                                                                        \
            transpose##N((uniform float *uniform) & pts[2 * i], lo);                                                   \
            transpose##N((uniform float *uniform) & pts[2 * i + programCount], hi);                                    \
            float c[N];                                                                                                \
            for (uniform int k = 0; k < N; ++k) {                                                                      \
                /* Lane l takes record 2l: lane 2l of lo, or lane 2l - programCount of hi. */                          \
                int src = (2 * programIndex) % programCount;                                                           \
                float a = shuffle(lo[k], src), b = shuffle(hi[k], src);                                                \
                c[k] = programIndex < programCount / 2 ? a : b;                                                        \
            }                                                                                                          \
            ACCUMULATE(N, c)                                                                                           \
        }                                                                                                              \
        foreach (j = i ... n) {                                                                                        \
            point##N p = pts[2 * j];                                                                                   \
            ACCUMULATE(N, p.c)                                                                                         \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    /* The coordinates of all 2 * n records as one float array, keeping the even records. */                           \
    export void moments_wide_strided_##N(const uniform point##N pts[], uniform int n, uniform float out[]) {           \
        const uniform float *uniform f = (const uniform float *uniform) & pts[0];                                      \
        float s[N] = {0}, q = 0;                                                                                       \
        foreach (j = 0 ... 2 * N * n) {                                                                                \
            float v = (j / N) % 2 == 0 ? f[j] : 0;                                                                     \
            int k = j % N;                                                                                             \
            for (uniform int m = 0; m < N; ++m)                                                                        \
                s[m] += k == m ? v : 0;                                                                                \
            q += v * v;                                                                                                \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    /* Indexed: records idx[0 .. n - 1]. */                                                                            \
    export void moments_gather_indexed_##N(const uniform point##N pts[], const uniform int idx[], uniform int n,       \
                                           uniform float out[]) {                                                      \
        float s[N] = {0}, q = 0;                                                                                       \
        foreach (i = 0 ... n) {                                                                                        \
            point##N p = pts[idx[i]];                                                                                  \
            ACCUMULATE(N, p.c)                                                                                         \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }                                                                                                                  \
    /* Copies the records lane by lane into a buffer, then transposes it. */                                           \
    export void moments_transpose_indexed_##N(const uniform point##N pts[], const uniform int idx[], uniform int n,    \
                                              uniform float out[]) {                                                   \
        float s[N] = {0}, q = 0;                                                                                       \
        uniform int i = 0;                                                                                             \
        for (; i + programCount <= n; i += programCount) {                                                             \
            uniform float buf[N * programCount];                                                                       \
            for (uniform int l = 0; l < programCount; ++l)                                                             \
                for (uniform int k = 0; k < N; ++k)                                                                    \
                    buf[N * l + k] = pts[idx[i + l]].c[k];                                                             \
            float c[N];                                                                                                \
            transpose##N(buf, c);                                                                                      \
            ACCUMULATE(N, c)                                                                                           \
        }                                                                                                              \
        foreach (j = i ... n) {                                                                                        \
            point##N p = pts[idx[j]];                                                                                  \
            ACCUMULATE(N, p.c)                                                                                         \
        }                                                                                                              \
        STORE_MOMENTS(N)                                                                                               \
    }

MOMENTS(2)
MOMENTS(3)
MOMENTS(4)
