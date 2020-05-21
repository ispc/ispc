// RUN: %{ispc} %s --target=avx1-i32x8 -h %t.h --emit-llvm -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x8 -h %t.h --emit-llvm -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x16 -h %t.h --emit-llvm -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i64x4 -h %t.h --emit-llvm -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: global i8 -128
uniform unsigned int8 a_min8 = -0x80;
// CHECK: global i16 -32768
uniform unsigned int16 a_min16 = -0x8000;
// CHECK: global i32 -2147483648
uniform unsigned int32 a_min32 = -0x80000000;
// CHECK: global i64 -9223372036854775808
uniform unsigned int64 a_min64 = -0x8000000000000000;
