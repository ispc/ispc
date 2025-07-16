
// This test verifies that ISPC correctly reports an error when --mcmodel=large
// is used with --arch=x86, as the large code model is not supported on 32-bit x86.
// It also verifies that compiler doesn't crash.

// RUN: not %{ispc} %s --nowrap --target=avx2-i32x8 --arch=x86 --mcmodel=large -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1
// RUN: %{ispc} %s --nowrap --target=avx2-i32x8 --arch=x86 --mcmodel=small -o %t.o
// RUN: %{ispc} %s --nowrap --target=avx2-i32x8 --arch=x86-64 --mcmodel=large -o %t.o
// RUN: %{ispc} %s --nowrap --target=avx2-i32x8 --arch=x86-64 --mcmodel=small -o %t.o

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK_ERROR_1: Error: --mcmodel=large is not supported for x86 architecture. Use x86-64 architecture instead.
// CHECK_ERROR_1-NOT: FATAL ERROR

uniform int i;
