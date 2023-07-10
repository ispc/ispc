// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// The goal of this test is to check that code generation for both versions is the same.

struct FVector4 {
    float V[4];
};

inline uniform FVector4 Mul3(const uniform FVector4 &A, const uniform FVector4 &B) {
    uniform float<4> S0 = {A.V[0], A.V[1], A.V[2], A.V[3]};
    uniform float<4> S1 = {B.V[0], B.V[1], B.V[2], B.V[3]};
    uniform float<4> D = S0 * S1;

    uniform FVector4 Result;
    Result.V[0] = D[0];
    Result.V[1] = D[1];
    Result.V[2] = D[2];
    Result.V[3] = D[3];
    return Result;
}

inline uniform FVector4 Mul4(const uniform FVector4 &A, const uniform FVector4 &B) {
    uniform float<4> S0 = {A.V[0], A.V[1], A.V[2], A.V[3]};
    uniform float<4> S1 = {B.V[0], B.V[1], B.V[2], B.V[3]};
    uniform float<4> D = S0 * S1;

    uniform FVector4 Result;
#pragma unroll
    for (uniform int i = 0; i < 4; i++) {
        Result.V[i] = D[i];
    }

    return Result;
}

// CHECK-LABEL: TestUniform3___
// CHECK-COUNT-1: vmovups
// CHECK-COUNT-1: vmulps
// CHECK-COUNT-1: vmovups
// CHECK-NOT: vmovaps
// CHECK-NOT: vmovups
unmasked void TestUniform3(uniform FVector4 Dst[], const uniform FVector4 Src0[], const uniform FVector4 Src1[],
                           const uniform FVector4 Src2[], const uniform int Count) {
    for (uniform int i = 0; i < Count; i++) {
        Dst[i] = Mul3(Src0[i], Src1[i]);
    }
}

// CHECK-LABEL: TestUniform4___
// CHECK-COUNT-1: vmovups
// CHECK-COUNT-1: vmulps
// CHECK-COUNT-1: vmovups
// CHECK-NOT: vmovaps
// CHECK-NOT: vmovups
unmasked void TestUniform4(uniform FVector4 Dst[], const uniform FVector4 Src0[], const uniform FVector4 Src1[],
                           const uniform FVector4 Src2[], const uniform int Count) {
    for (uniform int i = 0; i < Count; i++) {
        Dst[i] = Mul4(Src0[i], Src1[i]);
    }
}
