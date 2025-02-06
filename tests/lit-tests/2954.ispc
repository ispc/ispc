// RUN: %{ispc} -O2 --target=avx2-i32x8 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// This test passes when built with LLVM 20.0 or with LLVM 18/19 built with the
// backport patch llvm_patches/18_1_19_1_InstCombine-foldVecExtTruncToExtElt-extend.patch
// It is better to skip it as UNSUPPORTED for the release because many package
// maintainers will not have older LLVMs patched.

// UNSUPPORTED: !LLVM_20_0+

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
