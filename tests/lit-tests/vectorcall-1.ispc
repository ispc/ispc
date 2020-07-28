// Check that calling convention is not adding any overhead and function is optimized to only
// add and ret instructions.
// RUN: %{ispc}  %s --emit-asm --vectorcall --target=avx2-i32x8 -o - | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc}  %s --emit-asm --vectorcall --target=sse2-i32x4 -o - | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: %{ispc}  %s --emit-asm --vectorcall --target=sse4-i16x8 -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc}  %s --emit-asm --vectorcall --target=avx512skx-i32x16 -o - | FileCheck %s -check-prefix=CHECK_AVX512

// REQUIRES: X86_ENABLED

// CHECK_AVX2: bb.0:
// CHECK_AVX2-NEXT: vpaddd %ymm0, %ymm1, %ymm0
// CHECK_AVX2-NEXT: retq

// CHECK_SSE2: bb.0:
// CHECK_SSE2-NEXT: paddd %xmm1, %xmm0
// CHECK_SSE2-NEXT: retq

// CHECK_SSE4: bb.0:
// CHECK_SSE4-NEXT: paddd %xmm2, %xmm0
// CHECK_SSE4-NEXT:	paddd %xmm3, %xmm1
// CHECK_SSE4-NEXT: retq

// CHECK_AVX512: bb.0:
// CHECK_AVX512-NEXT: vpaddd %zmm0, %zmm1, %zmm0
// CHECK_AVX512-NEXT: retq

int addVal( int val1, int val2) {
    return val1 + val2;
}
