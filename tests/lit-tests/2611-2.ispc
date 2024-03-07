// RUN: %{ispc} --target=avx2-i32x4 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

struct vec2 {
    float V[2];
};

uniform vec2 vmax(const uniform vec2 &V1, const uniform vec2 &V2) {
    uniform vec2 Result;

    foreach (i = 0 ... 2) {
        Result.V[i] = max(V1.V[i], V2.V[i]);
    }

    return Result;
}

// CHECK-LABEL: foo:
// CHECK-NEXT: # %bb.0:
// CHECK-NEXT: vmovsd  xmm0, qword ptr [r{{.*}}]
// CHECK-NEXT: vmovsd  xmm1, qword ptr [r{{.*}}]
// CHECK-NEXT: vmaxps  xmm0, xmm0, xmm1
// CHECK-NEXT: vmovss  dword ptr [r{{.*}}], xmm0
// CHECK-NEXT: vextractps      dword ptr [r{{.*}} + 4], xmm0, 1
// CHECK-NEXT: ret
export void foo(uniform float A[], uniform float B[]) {

    uniform vec2 *uniform pA = (uniform vec2 * uniform) A;
    uniform vec2 *uniform pB = (uniform vec2 * uniform) B;

    *pA = vmax(*pA, *pB);
}
