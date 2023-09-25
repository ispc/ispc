// This test checks that ISPC correctly mangles SPIR-V builtins.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// REQUIRES: XE_ENABLED && !WINDOWS_HOST

// CHECK-LABEL: @kernel_double
// SPIR-V builtin, no pointer type:
// CHECK: _Z28__spirv_BuiltInWorkgroupSizei

// SPIR-V ocl builtins:
// rcp and rsqrt for double type are emulated through float, no need to check them here
// CHECK: _Z15__spirv_ocl_expDv16_d
// CHECK: _Z15__spirv_ocl_cosDv16_d
// CHECK: _Z15__spirv_ocl_sinDv16_d
// CHECK: _Z15__spirv_ocl_logDv16_d
// CHECK: @llvm.genx.ieee.sqrt.v16f64
// CHECK: _Z15__spirv_ocl_powDv16_dS_
// CHECK: _Z18__spirv_ocl_sincosDv16_dPS_

// CHECK: _Z15__spirv_ocl_expd
// CHECK: _Z15__spirv_ocl_cosd
// CHECK: _Z15__spirv_ocl_sind
// CHECK: _Z15__spirv_ocl_logd
// CHECK: @llvm.genx.ieee.sqrt.f64
// CHECK: _Z15__spirv_ocl_powdd
// CHECK: _Z18__spirv_ocl_sincosdPd

// SPIR-V printf builtin
// CHECK: _Z18__spirv_ocl_printfPU3AS2c
task void kernel_double(uniform double aFOO[], uniform double RES[]) {
  double x_var = aFOO[programIndex];
  x_var = exp(x_var);
  x_var = cos(x_var);
  x_var = sin(x_var);
  x_var = log(x_var);
  x_var = sqrt(x_var);
  x_var = pow(x_var, 2);
  varying double res1_var;
  varying double res2_var;
  sincos(x_var, &res1_var, &res2_var);
  RES[programIndex] = x_var;

  uniform double x = aFOO[0];
  x = exp(x);
  x = cos(x);
  x = sin(x);
  x = log(x);
  x = sqrt(x);
  x = pow(x, 2);
  uniform double res1;
  uniform double res2;
  sincos(x, &res1, &res2);
  print("%x\n", x);
}

// CHECK-LABEL: @kernel_float
// exp is implemented through pow
// CHECK: _Z23__spirv_ocl_native_powrDv16_fS_
// CHECK: _Z24__spirv_ocl_native_recipDv16_f
// CHECK: _Z22__spirv_ocl_native_cosDv16_f
// CHECK: _Z22__spirv_ocl_native_sinDv16_f
// CHECK: _Z23__spirv_ocl_native_log2Dv16_f
// CHECK: _Z23__spirv_ocl_native_sqrtDv16_f
// CHECK: _Z24__spirv_ocl_native_rsqrtDv16_f
// CHECK: _Z23__spirv_ocl_native_powrDv16_fS_

// CHECK: _Z23__spirv_ocl_native_powrff
// CHECK: _Z24__spirv_ocl_native_recipf
// CHECK: _Z22__spirv_ocl_native_cosf
// CHECK: _Z22__spirv_ocl_native_sinf
// CHECK: _Z23__spirv_ocl_native_log2f
// CHECK: _Z23__spirv_ocl_native_sqrtf
// CHECK: _Z24__spirv_ocl_native_rsqrtf
// CHECK: _Z23__spirv_ocl_native_powrff
task void kernel_float(uniform float aFOO[], uniform float RES[]) {
  float x_var = aFOO[programIndex];
  x_var = exp(x_var);
  x_var = rcp(x_var);
  x_var = cos(x_var);
  x_var = sin(x_var);
  x_var = log(x_var);
  x_var = sqrt(x_var);
  x_var = rsqrt(x_var);
  x_var = pow(x_var, 2);
  RES[programIndex] = x_var;

  uniform float x = aFOO[0];
  x = exp(x);
  x = rcp(x);
  x = cos(x);
  x = sin(x);
  x = log(x);
  x = sqrt(x);
  x = rsqrt(x);
  x = pow(x, 2);
  print("%x\n", x);
}

// CHECK-LABEL: @kernel_float16
// exp is implemented through pow
// CHECK: _Z23__spirv_ocl_native_powrDv16_DhS_
// CHECK: _Z22__spirv_ocl_native_cosDv16_Dh
// CHECK: _Z22__spirv_ocl_native_sinDv16_Dh
// CHECK: _Z23__spirv_ocl_native_log2Dv16_Dh
// CHECK: _Z23__spirv_ocl_native_sqrtDv16_Dh
// CHECK: _Z24__spirv_ocl_native_rsqrtDv16_Dh
// CHECK: _Z23__spirv_ocl_native_powrDv16_DhS_

// CHECK: _Z23__spirv_ocl_native_powrDhDh
// CHECK: _Z24__spirv_ocl_native_recipDh
// CHECK: _Z22__spirv_ocl_native_cosDh
// CHECK: _Z22__spirv_ocl_native_sinDh
// CHECK: _Z23__spirv_ocl_native_log2Dh
// CHECK: _Z23__spirv_ocl_native_sqrtDh
// CHECK: _Z24__spirv_ocl_native_rsqrtDh
// CHECK: _Z23__spirv_ocl_native_powrDhDh
task void kernel_float16(uniform float aFOO[], uniform float RES[]) {
  float16 x_var = aFOO[programIndex];
  x_var = exp(x_var);
  x_var = rcp(x_var);
  x_var = cos(x_var);
  x_var = sin(x_var);
  x_var = log(x_var);
  x_var = sqrt(x_var);
  x_var = rsqrt(x_var);
  x_var = pow(x_var, 2);
  RES[programIndex] = x_var;

  uniform float16 x = aFOO[0];
  x = exp(x);
  x = rcp(x);
  x = cos(x);
  x = sin(x);
  x = log(x);
  x = sqrt(x);
  x = rsqrt(x);
  x = pow(x, 2);
  print("%x\n", x);
}