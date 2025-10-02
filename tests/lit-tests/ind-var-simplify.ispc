// This test checks that induction variables are canonicalized and widened on CPU.

// RUN: %{ispc} %s --nostdlib --emit-llvm-text --target=avx2-i32x4 --arch=x86-64 -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: for_loop:
// CHECK:           %indvars.iv = phi i64
// CHECK:           %indvars.iv.next = add{{.*}} i64 %indvars.iv

void widen(uniform float A[], uniform int64 Num, uniform int64 N) {
    uniform float result = 0;
    for (uniform int i = 0; i < Num; i += N) {
        result += A[i];
    }
    A[0] = result;
}
