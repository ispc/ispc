// RUN: %{ispc} --target=sse4,avx2 %s --emit-llvm-text -o %t.ll
// RUN: FileCheck %s --check-prefix=CHECK-TARGET < %t_sse4.ll
// RUN: FileCheck %s --check-prefix=CHECK-TARGET < %t_avx2.ll
// RUN: FileCheck %s --check-prefix=CHECK-DISPATCHER < %t.ll

// REQUIRES: X86_ENABLED

// CHECK-DISPATCHER: @arr = {{.*}} global [2 x float]
// CHECK-TARGET: @arr = external {{.*}} global [2 x float]
uniform float arr[] = { 1.0f, 2.0f };
