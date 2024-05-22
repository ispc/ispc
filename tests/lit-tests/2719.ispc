// RUN: %{ispc} --target=avx2-i32x8 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: foo
// CHECK-NOT: vbroadcastss    ymm0, xmm0
// CHECK-NOT: vmovaps    ymm1, dword ptr [{{.*}}]
// CHECK-NOT: vmaskmovps      ymmword ptr [{{.*}}], ymm1, ymm0
uniform int foo(uniform float v1[], uniform float v2[], uniform float v3[], uniform float result[]) {
    float invArea = 1.f / ((v3[0] - v1[0]) * (v2[1] - v1[1]) - ((v3[1] - v1[1]) * (v2[0] - v1[0])));
    foreach (i = 0 ... 4) {
        result[i] = invArea;
    }
}
