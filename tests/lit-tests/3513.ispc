// This test checks whether ISPC produces correct representation of bool type
// on ABI boundary for targets with i8 masks.

// RUN: %{cc_ispc_run} sse2-i32x4 sse2i32x4 | FileCheck %s
// RUN: %{cxx_ispc_run} sse2-i32x4 sse2i32x4 | FileCheck %s
// RUN: %{cc_ispc_run} sse2-i32x8 sse2i32x8 | FileCheck %s
// RUN: %{cxx_ispc_run} sse2-i32x8 sse2i32x8 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.1-i16x8 sse41i16x8 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.1-i16x8 sse41i16x8 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.1-i32x4 sse41i32x4 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.1-i32x4 sse41i32x4 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.1-i32x8 sse41i32x8 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.1-i32x8 sse41i32x8 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.1-i8x16 sse41i8x16 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.1-i8x16 sse41i8x16 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.2-i16x8 sse42i16x8 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.2-i16x8 sse42i16x8 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.2-i32x4 sse42i32x4 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.2-i32x4 sse42i32x4 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.2-i32x8 sse42i32x8 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.2-i32x8 sse42i32x8 | FileCheck %s
// RUN: %{cc_ispc_run} sse4.2-i8x16 sse42i8x16 | FileCheck %s
// RUN: %{cxx_ispc_run} sse4.2-i8x16 sse42i8x16 | FileCheck %s
// RUN: %{cc_ispc_run} avx2-i16x16 avx2i16x16 | FileCheck %s
// RUN: %{cxx_ispc_run} avx2-i16x16 avx2i16x16 | FileCheck %s
// RUN: %{cc_ispc_run} avx2-i32x16 avx2i32x16 | FileCheck %s
// RUN: %{cxx_ispc_run} avx2-i32x16 avx2i32x16 | FileCheck %s
// RUN: %{cc_ispc_run} avx2-i32x4 avx2i32x4 | FileCheck %s
// RUN: %{cxx_ispc_run} avx2-i32x4 avx2i32x4 | FileCheck %s
// RUN: %{cc_ispc_run} avx2-i32x8 avx2i32x8 | FileCheck %s
// RUN: %{cxx_ispc_run} avx2-i32x8 avx2i32x8 | FileCheck %s
// RUN: %{cc_ispc_run} avx2-i64x4 avx2i64x4 | FileCheck %s
// RUN: %{cxx_ispc_run} avx2-i64x4 avx2i64x4 | FileCheck %s
// RUN: %{cc_ispc_run} avx2-i8x32 avx2i8x32 | FileCheck %s
// RUN: %{cxx_ispc_run} avx2-i8x32 avx2i8x32 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i16x16 generici16x16 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i16x16 generici16x16 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i16x8 generici16x8 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i16x8 generici16x8 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i1x16 generici1x16 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i1x16 generici1x16 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i1x32 generici1x32 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i1x32 generici1x32 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i1x4 generici1x4 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i1x4 generici1x4 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i1x64 generici1x64 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i1x64 generici1x64 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i1x8 generici1x8 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i1x8 generici1x8 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i32x16 generici32x16 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i32x16 generici32x16 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i32x4 generici32x4 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i32x4 generici32x4 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i32x8 generici32x8 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i32x8 generici32x8 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i64x4 generici64x4 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i64x4 generici64x4 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i8x16 generici8x16 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i8x16 generici8x16 | FileCheck %s
// RUN: %{cc_ispc_run} generic-i8x32 generici8x32 | FileCheck %s
// RUN: %{cxx_ispc_run} generic-i8x32 generici8x32 | FileCheck %s

// TODO!: ideally we need to introduce AVX2_HOST and similar features to
// reliably test tests like this but for now assume that AVX2 is always
// available practically on any x86_64 hosts

// REQUIRES: !MACOS_HOST && X86_ENABLED && X86_64_HOST

// CHECK: dst = true, bit representation: 0x01

#ifdef ISPC
export void test_bool_bug(uniform bool *uniform dst) {
    foreach (i = 0...1) {
        dst[i] = true;
    }
}
#else
#include <stdio.h>

#if defined(__cplusplus)
using namespace ispc;
#endif

int main() {
    bool dst;

    // Call ISPC function that sets dst[0] = true
    test_bool_bug(&dst);

    // Check the bool representation
    printf("dst = %s, bit representation: 0x%02x\n",
           dst ? "true" : "false", (unsigned char)dst);

    return 0;
}
#endif // ISPC
