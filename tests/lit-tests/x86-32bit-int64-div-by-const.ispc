// Verifies that signed division/remainder of varying int64 by a constant
// compiles on 32-bit x86 targets without AVX512DQ (v4i64 and v8i64). LLVM 23
// X86 backend used to crash with "Do not know how to expand the result of this
// operator!" on i64 mulhs when i64 is not a legal scalar type.

// RUN: %{ispc} %s --arch=x86 --target=avx2-i64x4 --nostdlib --nowrap --woff --emit-asm -O0 -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=avx2-i64x4 --nostdlib --nowrap --woff --emit-asm -O2 -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=avx2-i32x8 --nostdlib --nowrap --woff --emit-asm -O0 -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=avx2-i32x8 --nostdlib --nowrap --woff --emit-asm -O2 -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: srem_const___
// CHECK-LABEL: sdiv_const___

int64 srem_const(int64 a) { return a % 62; }

int64 sdiv_const(int64 a) { return a / 62; }
