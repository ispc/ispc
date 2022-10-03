// The test checks code generation for __regcall extern "C" function definitions on CPU.

// Check CPU one-target compilation:
// RUN: %{ispc} %s --target=avx2-i32x8 --emit-llvm-text --nowrap -o - | FileCheck %s -check-prefix=CHECK_CPU

// Check CPU multi-target compilation:
// RUN: %{ispc} %s --target=avx2-i32x8,avx1-i32x8 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck %s -check-prefix=CHECK_MULTI_CPU --input-file=%t.ll
// RUN: FileCheck %s -check-prefix=CHECK_MULTI_CPU1 --input-file=%t_avx.ll

// RUN: not %{ispc} %s --target=host -DREGCALL_NOT_EXTERN --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_REGCALL_NOT_EXTERN

// CHECK_REGCALL_NOT_EXTERN: Illegal to use "__regcall" qualifier on non-extern function "extern_func_def_not_extern"

// Check that calling convention and function name are correct
// CHECK_CPU: x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r(
// CHECK_CPU-NOT: <8 x i32> @extern_func_def_r(
// CHECK_CPU: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_r(
// CHECK_CPU: void @ispc_task__

// Dispatch function must be generated for extern C definitions
// CHECK_MULTI_CPU: declare x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r_avx(
// CHECK_MULTI_CPU: declare x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r_avx2(
// CHECK_MULTI_CPU: define x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r(

// CHECK_MULTI_CPU1: declare x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r(
// CHECK_MULTI_CPU1: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_r(
// CHECK_MULTI_CPU1: define void @ispc_task___
// CHECK_MULTI_CPU1: call x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r(
// CHECK_MULTI_CPU1: call x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_r(
// CHECK_MULTI_CPU1: define x86_regcallcc <8 x i32> @__regcall3__extern_func_def_r_avx(

// REQUIRES: X86_ENABLED

extern "C" __regcall int extern_func_def_r(uniform float RET[], int b) {
  return b*100;
}

#ifdef REGCALL_NOT_EXTERN
 __regcall int extern_func_def_not_extern(uniform float RET[], int b) {
  return b*100;
}
#endif

extern "C" __regcall int extern_sycl_decl_r(uniform float RET[]);

task void ispc_task(uniform float RET[]) {
  RET[programIndex] = extern_func_def_r(RET, programIndex);
  RET[programIndex] = extern_sycl_decl_r(RET);
}
