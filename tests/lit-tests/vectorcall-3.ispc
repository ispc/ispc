// RUN: %{ispc}  %s --emit-asm --target=avx2-i32x8 --disable-vectorcall -o - | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc}  %s --emit-asm --target=sse2-i32x4 --disable-vectorcall -o - | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: %{ispc}  %s --emit-asm --target=sse4-i16x8 --disable-vectorcall -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc}  %s --emit-asm  --target=avx512skx-i32x16 --disable-vectorcall -o - | FileCheck %s -check-prefix=CHECK_AVX512

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK_AVX2: bb.0:
// CHECK_AVX2-NEXT: vmovdqa	(%rdx), %ymm0
// CHECK_AVX2-NEXT: vpaddd (%rcx), %ymm0, %ymm0
// CHECK_AVX2-NEXT: retq

// CHECK_SSE2: bb.0:
// CHECK_SSE2-NEXT: movdqa	(%rdx), %xmm0
// CHECK_SSE2-NEXT: paddd	(%rcx), %xmm0
// CHECK_SSE2-NEXT: retq

// CHECK_SSE4: bb.0:
// CHECK_SSE4-NEXT: movdqa	(%r9), %xmm1
// CHECK_SSE4-NEXT: movdqa	(%r8), %xmm0
// CHECK_SSE4-NEXT: paddd (%rcx), %xmm0
// CHECK_SSE4-NEXT: paddd	(%rdx), %xmm1
// CHECK_SSE4-NEXT: retq

// CHECK_AVX512: bb.0:
// CHECK_AVX512-NEXT: vmovdqa64	(%rdx), %zmm0
// CHECK_AVX512-NEXT: vpaddd (%rcx), %zmm0, %zmm0
// CHECK_AVX512-NEXT: retq

int addVal( int val1, int val2) {
    return val1 + val2;
}
