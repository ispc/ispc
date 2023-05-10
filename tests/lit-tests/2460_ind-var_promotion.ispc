// This test checks for the simplification of induction variables for loops and efficient
// generation of accumulator widening/arithmetic.
// See [GitHub issue ISPC/#2460](https://github.com/ispc/ispc/issues/2460) for details.

// RUN: %{ispc} -DVECTOR=0 %s -o - -O2 --nostdlib --emit-asm --target=avx2-i32x4 --arch=x86-64 --x86-asm-syntax=intel | FileCheck %s --check-prefix=CHECK_ASM
// RUN: %{ispc} -DVECTOR=1 %s -o - -O2 --nostdlib --emit-asm --target=avx2-i32x4 --arch=x86-64 --x86-asm-syntax=intel | FileCheck %s --check-prefix=CHECK_ASM

// CHECK_ASM-LABEL:  Max_Uniform_FVector4f:
// CHECK_ASM:        .{{.*}}:
// CHECK_ASM-NOT:      cdqe
// CHECK_ASM-NOT:      movsxd
// CHECK_ASM-LABEL:  Max1_Uniform_FVector4f:
// CHECK_ASM:        .{{.*}}:
// CHECK_ASM-NOT:      cdqe
// CHECK_ASM-NOT:      movsxd

// REQUIRES: X86_ENABLED

struct FVector4f {
#if VECTOR == 0
    float V[4];
#else
    float<4> V;
#endif
};

extern float max(float, float);

inline unmasked uniform FVector4f VectorMax(const uniform FVector4f &V1, const uniform FVector4f &V2) {
    varying float S0, S1, Result;
    *((uniform FVector4f * uniform) & S0) = *((uniform FVector4f * uniform) & V1);
    *((uniform FVector4f * uniform) & S1) = *((uniform FVector4f * uniform) & V2);

    Result = max(S0, S1);

    return *((uniform FVector4f * uniform) & Result);
}

export void Max_Uniform_FVector4f(uniform float A[], uniform float B[], uniform float C[], uniform int Num) {
    for (uniform int i = 0; i < Num; i += 4) {
        uniform FVector4f *uniform pA = (uniform FVector4f * uniform) & A[i];
        uniform FVector4f *uniform pB = (uniform FVector4f * uniform) & B[i];
        uniform FVector4f *uniform pC = (uniform FVector4f * uniform) & C[i];

        *pA = VectorMax(VectorMax(*pB, *pC), *pA);
    }
}

inline unmasked uniform FVector4f VectorMax1(const uniform FVector4f &V1, const uniform FVector4f &V2) {
    uniform FVector4f Result;

    foreach (i = 0 ... 4) {
        Result.V[i] = max(V1.V[i], V2.V[i]);
    }

    return Result;
}

export void Max1_Uniform_FVector4f(uniform float A[], uniform float B[], uniform float C[], uniform int Num) {
    for (uniform int i = 0; i < Num; i += 4) {
        uniform FVector4f *uniform pA = (uniform FVector4f * uniform) & A[i];
        uniform FVector4f *uniform pB = (uniform FVector4f * uniform) & B[i];
        uniform FVector4f *uniform pC = (uniform FVector4f * uniform) & C[i];

        *pA = VectorMax1(VectorMax1(*pB, *pC), *pA);
    }
}
