// Copyright (c) 2024-2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// For easier checking on the C++ side:
// The value of PERMUTE_VAL should match the value of 'perm' parameter defined in the C++ code.
// This ensures that during runtime, the shuffle index simplifies to just programIndex.
#define PERMUTE_VAL 1

template <typename T> inline void Shuffle1(uniform T a[], uniform T ret[], int perm, uniform int count) {
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

template <typename T> inline void Shuffle2(uniform T a[], uniform T b[], uniform T ret[], int perm, uniform int count) {
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], b[i], programIndex + perm - PERMUTE_VAL);
    }
}

#define DEFINE_SHUFFLE1(TYPE)                                                                                          \
    export void Shuffle1_##TYPE(uniform TYPE a[], uniform TYPE ret[], uniform int perm, uniform int count) {           \
        Shuffle1<TYPE>(a, ret, perm, count);                                                                           \
    }

#define DEFINE_SHUFFLE2(TYPE)                                                                                          \
    export void Shuffle2_##TYPE(uniform TYPE a[], uniform TYPE b[], uniform TYPE ret[], uniform int perm,              \
                                uniform int count) {                                                                   \
        Shuffle2<TYPE>(a, b, ret, perm, count);                                                                        \
    }

// Define Shuffle1 functions for each type
DEFINE_SHUFFLE1(int8)
DEFINE_SHUFFLE1(int16)
DEFINE_SHUFFLE1(int)
#ifdef HAS_FP16
DEFINE_SHUFFLE1(float16)
#endif
DEFINE_SHUFFLE1(float)
DEFINE_SHUFFLE1(int64)
DEFINE_SHUFFLE1(double)

// Define Shuffle2 functions for each type
DEFINE_SHUFFLE2(int8)
DEFINE_SHUFFLE2(int16)
DEFINE_SHUFFLE2(int)
#ifdef HAS_FP16
DEFINE_SHUFFLE2(float16)
#endif
DEFINE_SHUFFLE2(float)
DEFINE_SHUFFLE2(int64)
DEFINE_SHUFFLE2(double)
