// The test checks code generation for extern "C" function definitions on GPU.

// Check GPU:
// RUN: %{ispc} %s --target=gen9-x8 --emit-llvm-text --nowrap -o - | FileCheck %s -check-prefix=CHECK_GPU

// REQUIRES: XE_ENABLED

// CHECK_GPU: spir_func <8 x i32> @extern_func_def(
// CHECK_GPU-NOT: spir_func <8 x i32> @extern_func_def__(
// CHECK_GPU: spir_func <8 x i32> @extern_func_decl
// CHECK_GPU: spir_func void @export_func_def__
// CHECK_GPU: spir_func void @export_func_def(
// CHECK-GPU: spir_kernel void @ispc_task

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