// This test checks that no redundant vmovaps instructions are generated for the code below.
// The assembler used to be the following:
//     vmovaps    LCPI0_0(%rip), %ymm0    ## ymm0 = [4294967295,4294967295,4294967295,4294967295,0,0,0,0]
//     vmaskmovps    (%rdi), %ymm0, %ymm1
//     vmaskmovps    (%rsi), %ymm0, %ymm2
//     vmovaps    %xmm1, %xmm1   // <-- redundant, as upper part of ymm1 is already zeros.
//     vmovaps    %xmm2, %xmm2   // <-- redundant, as upper part of ymm1 is already zeros.
//     vmulps    %ymm2, %ymm1, %ymm1
//     vmaskmovps    %ymm1, %ymm0, -16(%rsp)

// RUN: %{ispc} %s --target=avx2-i32x8 --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

struct FVector4 {
    float V[4];
};

unmasked uniform FVector4 Mul1(const uniform FVector4 &A, const uniform FVector4 &B) {
    uniform FVector4 Result;
// CHECK-NOT: vmovaps [[REG:%xmm[0-9]+]], [[REG]]
    foreach(i = 0 ... 4) {
        Result.V[i] = A.V[i] * B.V[i];
    }
    return Result;
}
