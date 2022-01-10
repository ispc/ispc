// RUN: %{ispc} %s --target=avx2-i32x4 --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-i32x16 --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// The test checks that aos_to_soa uses unaligned stores by default, but aligning pointers
// makes them aligned.

// The test supposed to work on single-pumped targets only.

// CHECK-LABEL: Begin function unaligned
// CHECK-COUNT-3: vmovup{{[sd]}} %{{[xyz]mm[0-9]+}}, {{[0-9]*}}(%{{.*}})
// CHECK-NOT: vmovup{{[sd]}} %{{[xyz]mm[0-9]+}}, {{[0-9]*}}(%{{.*}})
void unaligned(uniform float * uniform input,
       varying float * uniform out1,
       varying float * uniform out2,
       varying float * uniform out3) {
    aos_to_soa3(input, out1, out2, out3);
}

// CHECK-LABEL: Begin function aligned
// CHECK-COUNT-3: vmovap{{[sd]}} %{{[xyz]mm[0-9]+}}, {{[0-9]*}}(%{{.*}})
// CHECK-NOT: vmovap{{[sd]}} %{{[xyz]mm[0-9]+}}, {{[0-9]*}}(%{{.*}})
void aligned(uniform float * uniform input,
       varying float * uniform out1,
       varying float * uniform out2,
       varying float * uniform out3) {
    assume((((uniform int64)input) % (TARGET_WIDTH * sizeof(float))) == 0);
    assume((((uniform int64)out1) % (TARGET_WIDTH * sizeof(float))) == 0);
    assume((((uniform int64)out2) % (TARGET_WIDTH * sizeof(float))) == 0);
    assume((((uniform int64)out3) % (TARGET_WIDTH * sizeof(float))) == 0);
    aos_to_soa3(input, out1, out2, out3);
}
