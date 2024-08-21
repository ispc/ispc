// RUN: %{ispc} --target=avx2-i32x8 --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-NOT: vandps  ymm{{.*}}, ymm{{.*}}, ymm{{.*}}

struct FVector3 {
    float V[3];
};

uniform FVector3 VectorBitwiseAnd(const uniform FVector3 &Vec1, const uniform FVector3 &Vec2) {
    uniform FVector3 Result;

    foreach (i = 0 ... 3) {
        Result.V[i] = floatbits(intbits(Vec1.V[i]) & intbits(Vec2.V[i]));
    }

    return Result;
}
