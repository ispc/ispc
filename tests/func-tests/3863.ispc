// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Test for _mm_shuffle_pd SSE2 intrinsic correctness (#3863)
// This runtime test verifies that _mm_shuffle_pd correctly uses 1-bit selector
// fields for each output lane, not 2-bit fields like _mm_shuffle_ps.
// rule: skip on arch=*
// rule: run on arch=x86
// rule: run on arch=x86-64
// rule: skip on target=generic.*

#include "test_static.isph"
#include "../../stdlib/include/intrinsics/emmintrin.isph"

// Test all four possible shuffle masks: (0,0), (0,1), (1,0), (1,1)
// _MM_SHUFFLE2(x, y) produces ((x << 1) | y)
// Expected behavior:
//   result[0] = A[y]
//   result[1] = B[x]

task void f_v(uniform float RET[]) {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);
    uniform int errors = 0;

    // Test 1: _MM_SHUFFLE2(0, 0) = 0: select A[0], B[0] -> {1.0, 3.0}
    uniform __m128d r0 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
    if (r0.m128d_f64[0] != 1.0 || r0.m128d_f64[1] != 3.0)
        ++errors;

    // Test 2: _MM_SHUFFLE2(0, 1) = 1: select A[1], B[0] -> {2.0, 3.0}
    uniform __m128d r1 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 1));
    if (r1.m128d_f64[0] != 2.0 || r1.m128d_f64[1] != 3.0)
        ++errors;

    // Test 3: _MM_SHUFFLE2(1, 0) = 2: select A[0], B[1] -> {1.0, 4.0}
    uniform __m128d r2 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 0));
    if (r2.m128d_f64[0] != 1.0 || r2.m128d_f64[1] != 4.0)
        ++errors;

    // Test 4: _MM_SHUFFLE2(1, 1) = 3: select A[1], B[1] -> {2.0, 4.0}
    uniform __m128d r3 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
    if (r3.m128d_f64[0] != 2.0 || r3.m128d_f64[1] != 4.0)
        ++errors;

    RET[programIndex] = errors;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
