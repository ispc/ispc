// This test checks that induction variables are not canonicalized or widened on GPU.

// RUN: %{ispc} %s --nostdlib --emit-llvm-text --target=xehpg-x8 --arch=xe64 -o - | FileCheck %s

// REQUIRES: XE_ENABLED

// CHECK-LABEL: for_test:
// CHECK:           %i{{.*}} = phi i32
// CHECK:           %add{{.*}} = add{{.*}} i32 %i{{.*}}

task void widen(uniform float A[], uniform int64 Num, uniform int64 N) {
    uniform float result = 0;
    for (uniform int i = 0; i < Num; i += N) {
        result += A[i];
    }
}
