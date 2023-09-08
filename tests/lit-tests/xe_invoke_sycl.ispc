// The test checks code generation and proper diagnostics for invoke_sycl expression.

// Check CPU:
// RUN: %{ispc} %s --target=avx2-i32x8 --emit-llvm-text --nowrap -o - | FileCheck %s -check-prefix=CHECK_CPU

// Check multitarget CPU:
// RUN: %{ispc} %s --target=avx2-i32x8,avx1-i32x8 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck %s -check-prefix=CHECK_MULTI_CPU1 --input-file=%t_avx.ll
// RUN: FileCheck %s -check-prefix=CHECK_MULTI_CPU2 --input-file=%t_avx2.ll

// Check GPU:
// RUN: %{ispc} %s --target=gen9-x8 --emit-llvm-text --nowrap -o - | FileCheck %s -check-prefix=CHECK_GPU

// Check errors
// RUN: not %{ispc} %s --target=avx2-i32x8 -DINVOKE_SYCL_NOT_EXTERN --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_INVOKE_SYCL_NOT_EXTERN_SYCL
// RUN: not %{ispc} %s --target=avx2-i32x8 -DCHECK_EXTERN_SYCL_DEF --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_EXTERN_SYCL_DEF
// RUN: not %{ispc} %s --target=avx2-i32x8 -DSYCL_NO_INVOKE --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=SYCL_NO_INVOKE
// RUN: not %{ispc} %s --target=avx2-i32x8 -DINVOKE_SYCL_NO_REGCALL --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=INVOKE_SYCL_NO_REGCALL

// RUN: not %{ispc} %s --target=gen9-x8 -DINVOKE_SYCL_NOT_EXTERN --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_INVOKE_SYCL_NOT_EXTERN_SYCL
// RUN: not %{ispc} %s --target=gen9-x8 -DCHECK_EXTERN_SYCL_DEF --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_EXTERN_SYCL_DEF
// RUN: not %{ispc} %s --target=gen9-x8 -DSYCL_NO_INVOKE --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=SYCL_NO_INVOKE
// RUN: not %{ispc} %s --target=gen9-x8 -DINVOKE_SYCL_NO_REGCALL --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=INVOKE_SYCL_NO_REGCALL

// CHECK_INVOKE_SYCL_NOT_EXTERN_SYCL-COUNT-2: "invoke_sycl" expression illegal with non-'extern "SYCL"'-qualified function.
// CHECK_EXTERN_SYCL_DEF: 'extern "SYCL"' function "extern_sycl_func_def" cannot be defined in ISPC
// SYCL_NO_INVOKE: Illegal to call 'extern "SYCL"'-qualified function without "invoke_sycl" expression.
// INVOKE_SYCL_NO_REGCALL: "invoke_sycl" expression can be only used with '__regcall'-qualified function.
// REQUIRES: XE_ENABLED

// CHECK_CPU: x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl({{(float\*)|ptr}}, i32)
// CHECK_CPU: x86_regcallcc i32 @__regcall3__extern_sycl_decl_uniform({{(float\*)|ptr}})

// CHECK_CPU: call x86_regcallcc i32 @__regcall3__extern_sycl_decl_uniform({{(float\*)|ptr}} {{.*}})
// CHECK_CPU: call x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl({{(float\*)|ptr}} {{.*}}, i32 {{.*}})
// CHECK_CPU: call x86_regcallcc void @__regcall3__extern_sycl_decl_void({{(float\*)|ptr}} {{.*}})
// CHECK_CPU: call x86_regcallcc void @__regcall3__extern_sycl_decl_noarg()
// CHECK_CPU: call x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_ptr({{(float\*)|ptr}} {{.*}})

// CHECK_MULTI_CPU1: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl({{(float\*)|ptr}}, i32)
// CHECK_MULTI_CPU1: declare x86_regcallcc i32 @__regcall3__extern_sycl_decl_uniform({{(float\*)|ptr}})
// CHECK_MULTI_CPU1: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_ptr(

// CHECK_MULTI_CPU2: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl({{(float\*)|ptr}}, i32)
// CHECK_MULTI_CPU2: declare x86_regcallcc i32 @__regcall3__extern_sycl_decl_uniform({{(float\*)|ptr}})
// CHECK_MULTI_CPU2: declare x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_ptr(

// CHECK-GPU: x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl(<8 x i64> %{{.*}}), <8 x i32> %{{.*}})>)
// CHECK-GPU: x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_uniform(<8 x i64> %{{.*}}))
// CHECK-GPU: spir_kernel void @ispc_task

// Eventually external call should be surrounded by goto/join intrinsics to set HW mask
// but it is not supported in backend yet.
// TODO: enable it when support is available
// @llvm.genx.simdcf.goto
// Check that broadcast of uniform return values work as expected.
// CHECK_GPU: call x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_uniform(<8 x i64> %{{.*}})
// CHECK_GPU: extractelement <8 x i32> %{{.*}}, i{{(32|64)}} 0
// CHECK_GPU: call x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl(<8 x i64> %{{.*}}, <8 x i32> {{.*}})
// @llvm.genx.simdcf.join

// CHECK_GPU: store <8 x float>
// CHECK_GPU: store float
// CHECK_GPU: call x86_regcallcc void @__regcall3__extern_sycl_decl_void(<8 x i64> {{.*}})
// CHECK_GPU: call x86_regcallcc void @__regcall3__extern_sycl_decl_noarg()
// CHECK_GPU: call x86_regcallcc <8 x i32> @__regcall3__extern_sycl_decl_ptr(<8 x i64> {{.*}})

// REQUIRES: XE_ENABLED

#ifdef CHECK_EXTERN_SYCL_DEF
extern "SYCL" int extern_sycl_func_def(uniform float RET[], int b) {
  return b*100;
}
#endif

extern "SYCL" __regcall int extern_sycl_decl(uniform float RET[], uniform int b);
extern "SYCL" __regcall uniform int extern_sycl_decl_uniform(uniform float RET[]);
extern "SYCL" __regcall void extern_sycl_decl_void(uniform float RET[]);
extern "SYCL" __regcall void extern_sycl_decl_noarg();
extern "SYCL" __regcall int extern_sycl_decl_ptr(uniform float* uniform RET);

static int static_func();
extern "C" int extern_c_func();

extern "SYCL" int extern_sycl_decl_no_rc(uniform float RET[]);

export void ispc_task(uniform float RET[]) {
  uniform int res = invoke_sycl(extern_sycl_decl_uniform, RET);
  RET[programIndex] = invoke_sycl(extern_sycl_decl, RET, res);
  RET[0] = res;
  invoke_sycl(extern_sycl_decl_void, RET);
  invoke_sycl(extern_sycl_decl_noarg);
  invoke_sycl(extern_sycl_decl_ptr, RET);
// Checking compiler diagnostics
#ifdef INVOKE_SYCL_NOT_EXTERN
  RET[programIndex] = invoke_sycl(static_func);
  RET[programIndex] = invoke_sycl(extern_c_func);
#endif
#ifdef SYCL_NO_INVOKE
  extern_sycl_decl_uniform(RET);
#endif
#ifdef INVOKE_SYCL_NO_REGCALL
  invoke_sycl(extern_sycl_decl_no_rc, RET);
#endif
}
