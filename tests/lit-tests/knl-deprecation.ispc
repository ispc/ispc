// RUN: %{ispc} --nowrap --target=avx512knl-x16 --emit-asm -o %t.s %s 2>&1 | FileCheck %s
// RUN: %{ispc} --nowrap --target=avx512knl-i32x16 --emit-asm -o %t.s %s 2>&1 | FileCheck %s
// RUN: %{ispc} --nowrap --target=avx512knl-x16,avx2-i32x4 --emit-asm -o %t.s %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: Warning: The target avx512knl_x16 is deprecated and will be removed in the future.

void foo() {}
