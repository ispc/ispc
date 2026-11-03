// Known address gap: with 32-bit addressing, "uniform new uniform int8[n]"
// sign extends an unsigned 32-bit n, so for n of 2^31 or more it asks
// posix_memalign for about 2^64 bytes. The allocation builtin doesn't check
// the result, so the program then uses an uninitialized pointer. With 64-bit
// addressing n is zero extended.
//
// XFAIL until the size of an unsigned count is zero extended.

// RUN: %{ispc} %s --target=avx2-i32x8 --target-os=linux -O2 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LINUX_ENABLED
// XFAIL: *

// CHECK-LABEL: @new_u32___
// CHECK: zext i32 %n to i64
// CHECK-NOT: sext i32 %n
// CHECK: @posix_memalign
uniform int8 *uniform new_u32(uniform unsigned int n) { return uniform new uniform int8[n]; }
