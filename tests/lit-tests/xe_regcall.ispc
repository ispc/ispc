// The test checks code generation for __regcall extern "C" function definitions on GPU.

// Check CPU one-target compilation:
// RUN: %{ispc} %s --target=gen9-x8 --emit-llvm-text --nowrap -o - | FileCheck %s -check-prefix=CHECK_GPU

// REQUIRES: XE_ENABLED

// Check that calling convention and function name are correct

// CHECK_GPU: define x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r
// CHECK_GPU-NOT: define x86_regcallcc <8 x i32> @extern_func_def_r(
// CHECK_GPU: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_r(
// CHECK-GPU: spir_kernel void @ispc_task

extern "C" __regcall int extern_func_def_r(uniform float RET[], int b) {
  return b*100;
}

extern "C" __regcall int extern_sycl_decl_r(uniform float RET[]);

task void ispc_task(uniform float RET[]) {
  RET[programIndex] = extern_func_def_r(RET, programIndex);
  RET[programIndex] = extern_sycl_decl_r(RET);
}
