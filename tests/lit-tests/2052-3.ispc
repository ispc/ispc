// The tests checks that no extra load/store instructions is generated.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --emit-asm  -o - | FileCheck %s
// REQUIRES: ARM_ENABLED

// CHECK-COUNT-2: mov
// CHECK-NOT: add
// CHECK-LABEL: This Inner Loop Header
// CHECK-COUNT-2: ldr
// CHECK: fcm
// CHECK-NOT: ldp
typedef struct {
    uniform float<4> v;
} FVector4;

unmasked void mm_cmpgt_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        uniform FVector4 S1;
        S1.v[0] = Source1[k];

        uniform FVector4 S2;
        S2.v[0] = Source2[k];

        uniform FVector4 R;
        R = S1;
        R.v[0] = S1.v[0] > S2.v[0] ? floatbits(0xFFFFFFFF) : 0.0f;

        // Inline _mm_store_ss(&Result[k], R);
        Result[k] = R.v[0];
    }
}
