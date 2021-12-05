//; RUN: %{ispc} %s --target=host -O0 --nostdlib --emit-llvm-text -o - 2>&1 | FileCheck %s

// CHECK: global i8 18
// CHECK: global i16 4660
// CHECK: global i32 305419896
// CHECK: global i64 1311768467294899695
uniform uint8  i1 = 0X12;
uniform uint16 i2 = 0X1234;
uniform uint32 i3 = 0X12345678;
uniform uint64 i4 = 0X1234567890abcdef;

// CHECK: global half 0xH4000
// CHECK: global float 2.000000e+00
// CHECK: global double 2.000000e+00
uniform float16 f1 = 0X1p1f16;
uniform float   f2 = 0X1p1f;
uniform double  f3 = 0X1p1d;
