// This test checks that the ISPC compiler does not crash when compiling
// IndexExpr with FunctionCallExpr returning pointer type.

// RUN: %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap %s -o - 2>&1 | FileCheck %s
// RUN: %{ispc} --target=avx512icl-x16 -O2 --emit-asm --x86-asm-syntax=intel --nostdlib --nowrap %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AVX512

// REQUIRES: X86_ENABLED

// CHECK-NOT: .*expr.cpp:.*: Assertion failed: "ptrType != nullptr".

// CHECK-AVX512-LABEL: foo___{{.*}}:
// CHECK-AVX512-NEXT: # %bb.0:
// CHECK-AVX512-NEXT:  movsxd  [[REG:r[a-z]+]], {{.*}}
// CHECK-AVX512-NEXT:  shl     [[REG]], 6
// CHECK-AVX512-NEXT:  vmovups zmm0, zmmword ptr [r{{.*}} + [[REG]]]
// CHECK-AVX512-NEXT:  ret

struct S {
    float b[100];
};
varying float *uniform func(const varying struct S &t) {
    return (varying float *uniform) & t;
};
varying float foo(const varying struct S &x, uniform int c) {
    return (func(x))[c];
}
