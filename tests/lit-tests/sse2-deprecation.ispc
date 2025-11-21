// RUN: %{ispc} --nowrap --target=sse2 --emit-asm -o %t.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} --nowrap --target=sse2-i32x4 --emit-asm -o %t.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} --nowrap --target=sse2-x2 --emit-asm -o %t.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} --nowrap --target=sse2-i32x8 --emit-asm -o %t.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} --nowrap --target=sse2-i32x8,avx2-i32x4 --emit-asm -o %t.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-X8

// REQUIRES: X86_ENABLED

// CHECK-X4: Warning: The target sse2-i32x4 is deprecated and will be removed in the future.
// CHECK-X8: Warning: The target sse2-i32x8 is deprecated and will be removed in the future.

void foo() {}
