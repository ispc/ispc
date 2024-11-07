// RUN: %{ispc} -O2 --target=avx2-i32x8 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// XFAIL: !LLVM_20_0+

// REQUIRES: X86_ENABLED

// CHECK-NOT: vmovq

struct FVector4 {
    double V[4];
};

uniform FVector4 VectorBitwiseAnd(const uniform FVector4 &Vec1, const uniform FVector4 &Vec2) {
    uniform FVector4 Result;

    foreach (i = 0 ... 4) {
        Result.V[i] = doublebits(intbits(Vec1.V[i]) & intbits(Vec2.V[i]));
    }

    return Result;
}
