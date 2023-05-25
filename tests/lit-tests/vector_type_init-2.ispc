// This test checks that uniform vectors conversion during initialization is perfomed using vector instructions.
// The example is used from https://github.com/ispc/ispc/issues/2147

// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK_IR
// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_ASM
// REQUIRES: X86_ENABLED

// CHECK_IR: define <4 x double> @TestCross
// CHECK_IR: fpext <4 x float>
// CHECK_IR-NOT: fpext float

// CHECK_ASM: TestCross
// CHECK_ASM-COUNT-2: vcvtps2pd
// CHECK_ASM-NOT: vcvtss2sd

struct FVector {
    float<4> V;
};

typedef double<4> QVec3;

inline uniform QVec3 SetQVec3(const uniform double X, const uniform double Y, const uniform double Z) {
    const uniform QVec3 Result = {X, Y, Z, 0.0};
    return Result;
}

inline uniform QVec3 SetQVec3(uniform const FVector fp0) {
    const uniform QVec3 Result = {fp0.V[0], fp0.V[1], fp0.V[2], 0.0};
    return Result;
}

inline uniform QVec3 QVec3Swizzle(const uniform QVec3 Vec, const uniform int X, const uniform int Y,
                                  const uniform int Z) {
    return SetQVec3(Vec[X], Vec[Y], Vec[Z]);
}

inline uniform QVec3 QVec3Cross(const uniform QVec3 Vec1, const uniform QVec3 Vec2) {
    uniform QVec3 Tmp0 = QVec3Swizzle(Vec2, 1, 2, 0);
    uniform QVec3 Tmp1 = QVec3Swizzle(Vec1, 1, 2, 0);
    Tmp0 = Tmp0 * Vec1;
    Tmp1 = Tmp1 * Vec2;
    uniform QVec3 Tmp2 = Tmp0 - Tmp1;
    return QVec3Swizzle(Tmp2, 1, 2, 0);
}

uniform QVec3 TestCross(const uniform FVector a, const uniform FVector b) {
    const uniform QVec3 tmp1 = SetQVec3(a);
    const uniform QVec3 tmp2 = SetQVec3(b);
    return QVec3Cross(tmp1, tmp2);
}