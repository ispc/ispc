// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Test for _mm_shuffle_pd SSE2 intrinsic correctness (#3863)
// This test verifies that _mm_shuffle_pd correctly uses 1-bit selector fields
// for each output lane, not 2-bit fields like _mm_shuffle_ps.
//
// RUN: %{ispc} %s --target=sse2-i32x4 -I %S/../../stdlib/include -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse4-i32x4 -I %S/../../stdlib/include -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x4 -I %S/../../stdlib/include -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 -I %S/../../stdlib/include -O2 --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

#include "intrinsics/emmintrin.isph"

// Test all four possible shuffle masks: (0,0), (0,1), (1,0), (1,1)
// _MM_SHUFFLE2(x, y) produces ((x << 1) | y)
// Expected behavior:
//   result[0] = A[y]
//   result[1] = B[x]

export uniform double test_shuffle_pd_00() {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);
    // _MM_SHUFFLE2(0, 0) = 0: select A[0], B[0] -> {1.0, 3.0}
    uniform __m128d r = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 0));
    return r.m128d_f64[0] * 10.0 + r.m128d_f64[1];
    // CHECK-LABEL: define double @test_shuffle_pd_00()
    // CHECK: ret double 1.300000e+01
}

export uniform double test_shuffle_pd_01() {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);
    // _MM_SHUFFLE2(0, 1) = 1: select A[1], B[0] -> {2.0, 3.0}
    uniform __m128d r = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(0, 1));
    return r.m128d_f64[0] * 10.0 + r.m128d_f64[1];
    // CHECK-LABEL: define double @test_shuffle_pd_01()
    // CHECK: ret double 2.300000e+01
}

export uniform double test_shuffle_pd_10() {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);
    // _MM_SHUFFLE2(1, 0) = 2: select A[0], B[1] -> {1.0, 4.0}
    uniform __m128d r = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 0));
    return r.m128d_f64[0] * 10.0 + r.m128d_f64[1];
    // CHECK-LABEL: define double @test_shuffle_pd_10()
    // CHECK: ret double 1.400000e+01
}

export uniform double test_shuffle_pd_11() {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);
    // _MM_SHUFFLE2(1, 1) = 3: select A[1], B[1] -> {2.0, 4.0}
    uniform __m128d r = _mm_shuffle_pd(a, b, _MM_SHUFFLE2(1, 1));
    return r.m128d_f64[0] * 10.0 + r.m128d_f64[1];
    // CHECK-LABEL: define double @test_shuffle_pd_11()
    // CHECK: ret double 2.400000e+01
}
