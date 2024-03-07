// RUN: %{ispc} --target=avx2-i32x8 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s
// RUN: %{ispc} --target=avx512skx-x8 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s
// RUN: %{ispc} --target=avx2-i32x16 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s
// RUN: %{ispc} --target=avx512skx-x16 --nowrap --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED
// REQUIRES: LLVM_17_0+

struct vec4 {
    double V[4];
};

uniform vec4 vmax(const uniform vec4 &V1, const uniform vec4 &V2) {
    uniform vec4 Result;

    foreach (i = 0 ... 4) {
        Result.V[i] = max(V1.V[i], V2.V[i]);
    }

    return Result;
}

// CHECK-LABEL: foo:
// CHECK: vmovupd ymm{{.*}}, ymmword ptr [r{{.*}}]
// CHECK-NEXT: vmaxpd  ymm{{.*}}, ymm{{.*}}, ymmword ptr [r{{.*}}]
// CHECK-NEXT: vmovupd ymmword ptr [r{{.*}}], ymm{{.*}}
export void foo(uniform double A[], uniform double B[]) {

    uniform vec4 *uniform pA = (uniform vec4 * uniform) A;
    uniform vec4 *uniform pB = (uniform vec4 * uniform) B;

    *pA = vmax(*pA, *pB);
}
