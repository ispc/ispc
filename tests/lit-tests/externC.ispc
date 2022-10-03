// The test checks code generation for extern "C" function definitions on CPU.

// Check CPU one-target compilation:
// RUN: %{ispc} %s --target=avx2-i32x8 --emit-llvm-text --nowrap -o - | FileCheck %s -check-prefix=CHECK_CPU

// Check CPU multi-target compilation:
// RUN: %{ispc} %s --target=avx2-i32x8,avx1-i32x8 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck %s -check-prefix=CHECK_MULTI_CPU --input-file=%t.ll
// RUN: FileCheck %s -check-prefix=CHECK_MULTI_CPU1 --input-file=%t_avx.ll

// Not-mangled definition of extern function only
// CHECK_CPU: <8 x i32> @extern_func_def(
// CHECK_CPU-NOT: <8 x i32> @extern_func_def__
// CHECK_CPU: declare <8 x i32> @extern_func_decl
// CHECK_CPU: void @export_func_def__
// CHECK_CPU: void @ispc_task__
// CHECK_CPU: void @export_func_def(

// CHECK_MULTI_CPU: declare void @export_func_def_avx(
// CHECK_MULTI_CPU: declare void @export_func_def_avx2(
// CHECK_MULTI_CPU: define void @export_func_def(
// Dispatch function must be generated for extern C definitions
// CHECK_MULTI_CPU: declare <8 x i32> @extern_func_def_avx(
// CHECK_MULTI_CPU: declare <8 x i32> @extern_func_def_avx2(
// CHECK_MULTI_CPU: define <8 x i32> @extern_func_def(

// CHECK_MULTI_CPU1: declare <8 x i32> @extern_func_def(
// CHECK_MULTI_CPU1: declare <8 x i32> @extern_func_decl(
// CHECK_MULTI_CPU1-NOT: define <8 x i32> @extern_func_def(
// CHECK_MULTI_CPU1: define void @ispc_task___
// CHECK_MULTI_CPU1: call void @export_func_def__
// CHECK_MULTI_CPU1: call <8 x i32> @extern_func_def(
// CHECK_MULTI_CPU1: call <8 x i32> @extern_func_decl(
// CHECK_MULTI_CPU1: define <8 x i32> @extern_func_def_avx(

// REQUIRES: X86_ENABLED

extern "C" int extern_func_def(uniform float RET[], int b) {
  return RET[b]*100;
}

extern "C" int extern_func_decl(uniform float RET[], int b);

noinline export void export_func_def(uniform float RET[]) {
  RET[programIndex] = programIndex;
}

task void ispc_task(uniform float RET[]) {
  export_func_def(RET);
  RET[programIndex] = extern_func_def(RET, programIndex);
  RET[programIndex] += extern_func_decl(RET, programIndex);
}
