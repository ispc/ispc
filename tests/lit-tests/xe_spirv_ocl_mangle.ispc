// This test checks that ISPC correctly mangles SPIR-V builtins with pointer types.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// REQUIRES: XE_ENABLED && !WINDOWS_HOST

// SPIR-V builtin, no pointer type:
// CHECK: _Z28__spirv_BuiltInWorkgroupSizei
// SPIR-V ocl builtin with pointer types, varying
// CHECK: _Z18__spirv_ocl_sincosDv16_dPS_
// SPIR-V ocl builtin with pointer types, uniform
// CHECK: _Z18__spirv_ocl_sincosdPd
// SPIR-V printf builtin
// CHECK: _Z18__spirv_ocl_printfPU3AS2c
task void assert_kernel(uniform double aFOO[], uniform double RES[]) {
  double x_var = aFOO[programIndex];
  varying double res1_var;
  varying double res2_var;
  sincos(x_var, &res1_var, &res2_var);
  RES[programIndex] = res1_var + res2_var;

  uniform double x = aFOO[0];
  uniform double res1;
  uniform double res2;
  sincos(x, &res1, &res2);
  print("%x\n", x);
}