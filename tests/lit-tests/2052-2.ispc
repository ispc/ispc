// The tests checks that no extra load/store instructions is generated.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --emit-asm  -o - | FileCheck %s
// REQUIRES: ARM_ENABLED

// CHECK-LABEL: VectorMatrixMultiply
// CHECK-COUNT-2: ldp
// CHECK-NOT: ld1r
// CHECK-NOT: add
// CHECK: {{LBB0_[0-9]}}:
struct FMatrix {
    float<16> M;
};
FMatrix VectorMatrixMultiply(const FMatrix &A, const uniform FMatrix &B) {
    FMatrix Result;

    for (uniform unsigned int m = 0; m < 4; m++) {
        varying float Sum;
        for (uniform unsigned int k = 0; k < 4; k++) {
            Sum = 0.0f;
            for (uniform unsigned int n = 0; n < 4; n++) {
                Sum += A.M[m * 4 + n] * B.M[n * 4 + k];
            }

            Result.M[m * 4 + k] = Sum;
        }
    }

    return Result;
}