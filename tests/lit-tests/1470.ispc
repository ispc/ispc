// This test checks that swizzle operation does not cause any compiler 
// internal error when it is applied to binary expression.
// Swizzle operation is implemented through memory using insert instructions.
// The test also checks that inserts are transformed to shuffles during
// optimization when vector size equals number of program instances.

// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=sse2-i32x4 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=sse2-i32x8 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-asm -h %t.h --target=sse2-i32x4 -o - | FileCheck %s -check-prefix=CHECK_SSE

// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=sse4-i32x4 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=sse4-i32x8 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=sse4-i16x8 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=sse4-i8x16 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-asm -h %t.h --target=sse4-i32x4 -o - | FileCheck %s -check-prefix=CHECK_SSE

// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx1-i32x4 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx1-i32x8 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx1-i32x16 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx1-i64x4 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-asm -h %t.h --target=avx1-i32x4 -o - | FileCheck %s -check-prefix=CHECK_AVX

// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx2-i32x8 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx2-i32x16 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx2-i64x4 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx512knl-x16 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc}  %s --emit-llvm -h %t.h --target=avx512skx-x16 -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// REQUIRES: X86_ENABLED

typedef float<2> vec2;
typedef float<4> vec4;

inline vec4 MakeVec4(vec2 a, vec2 b) {
    const vec4 res = {a.x, a.y, b.x, b.y};
    return res;
}

unmasked vec4 test_swizzle(uniform vec4 offs0, uniform vec4 offs1, uniform vec4 offs2) {
    vec4 uv = { 1.0f, 2.0f, 1.0f, 2.0f };

    vec4 a1 = uv.xzyw + offs0;
    vec4 a2 = (uv.xyxy + offs0).xwwy;
    vec4 a3 = MakeVec4((uv.xyxy + offs0).xz, (uv.xyxy + offs1).yw) + offs2;
    vec4 a4 = (MakeVec4((uv.xyyy + offs0).xz, (uv.xxxy + offs1).yw) + offs2).xyyw;
    if (a1.x+a2.y > a3.z+a4.w) return a1+a3;
    return a2+a4;
}

unmasked vec4 test_shuffle(uniform vec4 offs0) {
    vec4 uv = { 1.0f, 2.0f, 1.0f, 2.0f };
// CHECK: @test_shuffle
// CHECK-COUNT-3: shufflevector

// CHECK_SSE: test_shuffle
// CHECK_SSE-COUNT-3: shufps {{.*#+}}

// CHECK_AVX: test_shuffle
// CHECK_AVX-COUNT-3: vpermilps {{.*#+}}

    return (uv.xyxy + offs0).xwwz;
}

export void test() {
    const uniform vec4 offs0 = { 0, 1, 2, 3 };
    const uniform vec4 offs1 = { 2, 3, 4, 5 };
    const uniform vec4 offs2 = { 3, 4, 5, 6 };
    vec4 a = test_swizzle(offs0, offs1, offs2);
    vec4 b = test_shuffle(offs0);
}

