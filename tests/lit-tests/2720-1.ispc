// Pins today's code for #2720 on x86 (2720.ispc covers ARM): the same
// uniform value is stored to 4 adjacent floats.
// Today: 4 scalar float stores on every x86 target.
// Ideal: one 16-byte vector store.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: @foo___
// CHECK-NOT: store <{{[0-9]+}} x float>
// CHECK: store float
// CHECK-NOT: store <{{[0-9]+}} x float>
// CHECK: store float
// CHECK-NOT: store <{{[0-9]+}} x float>
// CHECK: store float
// CHECK-NOT: store <{{[0-9]+}} x float>
// CHECK: store float
// CHECK-NOT: store
// CHECK: ret i32
uniform int foo(uniform float v1[], uniform float v2[], uniform float v3[], uniform float result[]) {
    uniform float invArea = 1.f / ((v3[0] - v1[0]) * (v2[1] - v1[1]) - ((v3[1] - v1[1]) * (v2[0] - v1[0])));
    result[0] = invArea;
    result[1] = invArea;
    result[2] = invArea;
    result[3] = invArea;
}
