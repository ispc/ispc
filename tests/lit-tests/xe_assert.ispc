// This test checks correct LLVM IR generation when assert is used on Xe target.
// TODO: add test case with assert inside exported function when support is added to VC backend.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o - | FileCheck %s
// REQUIRES: XE_ENABLED

//CHECK: @lz_format_str = internal unnamed_addr addrspace(2) constant [{{.*}} x i8] c"{{.*}}xe_assert.ispc:13:3: Assertion failed: a > 0 \0A\00", align 1
task void assert_kernel(uniform float aFOO[]) {
//CHECK: fail{{.*}}:
//CHECK-NEXT: call i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8] addrspace(2)* @lz_format_str, i64 0, i64 0))
//CHECK-NEXT: call void @llvm.genx.raw.send.noresult.v32i16.v16i1.v32i16(i32 0, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i32 39, i32 33554448, <32 x i16> zeroinitializer)
  float a = aFOO[programIndex];
  assert(a > 0);
}