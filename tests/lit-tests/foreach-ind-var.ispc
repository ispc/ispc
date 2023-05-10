// This tests ensures arithmetic on the foreach induction variable is signed.
// RUN: %{ispc} %s --emit-llvm-text --target=host -o - -O0 --nowrap | FileCheck %s

// REQUIRES: !XE_ENABLED

// CHECK-LABEL: define void @fold_add_0

// CHECK-LABEL: allocas:
// CHECK:           %nitems = sub{{.}}nsw
// CHECK:           %aligned_end = sub{{.*}}nsw

// CHECK-LABEL: foreach_full_body:
// CHECK:           %new_counter = add{{.*}}nsw

task void fold_add_0(uniform float A[], uniform int64 Num, uniform float Result[]) {
    Result[programIndex] = 0;
    foreach (i = 0 ... Num) {
        Result[programIndex] += A[i];
    }
    Result[0] = reduce_add(Result[programIndex]);
}
