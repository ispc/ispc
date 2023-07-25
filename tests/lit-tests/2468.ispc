// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// The goal of this test is to check that code generation results into a pair of loads, an add, and a store.
// In assembly we expect only one vmovups instruction for load since the second load is embedded into vaddps.

// CHECK: vmovups
// CHECK-NOT: vgather
// CHECK-NEXT: vaddps
// CHECK-NEXT: vmovups

#define SOA_IX(a, v, nv) ((((a) / TARGET_WIDTH)*(nv) + (v))*TARGET_WIDTH + (a) % TARGET_WIDTH)

void simple(uniform float vin[], uniform float vout[]) {
    foreach (index = 0 ... TARGET_WIDTH) {
        // Load the appropriate input value for this program instance.
        float v1 = vin[SOA_IX(index, 0, 3)];
        float v2 = vin[SOA_IX(index, 2, 3)];

        // And write the result to the output array.
        vout[index] = v1 + v2;
    }
}