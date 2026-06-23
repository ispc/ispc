// Check the unsigned reduce_add call produces the (v)psadbw instruction on x86 targets.

// RUN: %{ispc} -O2 --target=avx2-i32x8 %s --emit-asm --x86-asm-syntax=intel -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: test_uint8___UM_vyT:
// CHECK: psadbw
unmasked uniform uint8 test_uint8(uint8 v) { return reduce_add(v); }
