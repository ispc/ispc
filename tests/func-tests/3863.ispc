// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Test for _mm_shuffle_pd SSE2 intrinsic correctness (#3863)
// This runtime test verifies that _mm_shuffle_pd correctly uses 1-bit selector
// fields for each output lane, not 2-bit fields like _mm_shuffle_ps.
//
// RUN: %{ispc} %s -o %t.o
// RUN: %{cc} %S/../test_static.c %t.o -o %t.exe
// RUN: %t.exe
//
// REQUIRES: X86_ENABLED

#include "../test_static.isph"
#include "intrinsics/emmintrin.isph"

// Test all four possible shuffle masks: (0,0), (0,1), (1,0), (1,1)
// _MM_SHUFFLE2(x, y) produces ((x << 1) | y)
// Expected behavior:
//   result[0] = A[y]
//   result[1] = B[x]

task void f_v(uniform float RET[]) {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);

    // Test 1: _MM_SHUFFLE2(0, 0) = 0: select A[0], B[0] -> {1.0, 3.0}
    uniform __m128d r0 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
    uniform double result0[2];
    *((uniform __m128d * uniform)result0) = r0;
    RET[0] = (result0[0] == 1.0 && result0[1] == 3.0) ? 1.0 : 0.0;

    // Test 2: _MM_SHUFFLE2(0, 1) = 1: select A[1], B[0] -> {2.0, 3.0}
    uniform __m128d r1 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 1));
    uniform double result1[2];
    *((uniform __m128d * uniform)result1) = r1;
    RET[1] = (result1[0] == 2.0 && result1[1] == 3.0) ? 1.0 : 0.0;

    // Test 3: _MM_SHUFFLE2(1, 0) = 2: select A[0], B[1] -> {1.0, 4.0}
    uniform __m128d r2 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 0));
    uniform double result2[2];
    *((uniform __m128d * uniform)result2) = r2;
    RET[2] = (result2[0] == 1.0 && result2[1] == 4.0) ? 1.0 : 0.0;

    // Test 4: _MM_SHUFFLE2(1, 1) = 3: select A[1], B[1] -> {2.0, 4.0}
    uniform __m128d r3 = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
    uniform double result3[2];
    *((uniform __m128d * uniform)result3) = r3;
    RET[3] = (result3[0] == 2.0 && result3[1] == 4.0) ? 1.0 : 0.0;
}

task void result(uniform float RET[]) {
    // All tests should pass (return 1.0)
    RET[0] = 1.0;
    RET[1] = 1.0;
    RET[2] = 1.0;
    RET[3] = 1.0;
}
