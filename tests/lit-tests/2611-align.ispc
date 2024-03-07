// RUN: %{ispc} --target=avx512skx-x16 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-NOT: vmovapd ymm{{.*}}
// CHECK: vmovupd ymm{{.*}}

typedef double FReal;

#define REAL_BITS doublebits
#define DOUBLE_NAN 0xFFFFFFFFFFFFFFFF
#define NAN DOUBLE_NAN

struct FVector4 {
    double V[4];
};

static inline uniform FVector4 VectorCompareGE1(const uniform FVector4 &A, const uniform FVector4 &B) {
    varying FReal Result;

    foreach (i = 0 ... 4) {
        Result = A.V[i] >= B.V[i] ? REAL_BITS(NAN) : REAL_BITS(0);
    }

    return *((uniform FVector4 * uniform) & Result);
}

export void CompareGE1(uniform FVector4 Result[], const uniform FVector4 Src0[], const uniform FVector4 Src1[],
                       const uniform int NumCompares) {
    for (uniform int i = 0; i < NumCompares; i++) {
        Result[i] = VectorCompareGE1(Src0[i], Src1[i]);
    }
}
