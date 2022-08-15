// The test checks correct IR generation for print of bool types.
// It doesn't make much sense with opaque pointers since the most of the instructions will be optimized.
// RUN: %{ispc} %s --target=avx2-i32x4 --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED

// UNSUPPORTED: OPAQUE_PTRS_ENABLED
// CHECK: [[PRTARG_PTR_BITCAST:%[a-zA-Z0-9_.]+]] = bitcast [1 x i8*]*  [[PRTARG_PTR:%[a-zA-Z0-9_.]+]] to i8**
// CHECK: [[PRTARG_BITCAST:%[a-zA-Z0-9_.]+]] = bitcast <4 x i32>* {{%[a-zA-Z0-9_.]+}} to i8*
// CHECK: [[SCT_OFF:%[a-zA-Z0-9_.]+]] = getelementptr [1 x i8*], [1 x i8*]* [[PRTARG_PTR]], i32 0, i32 0
// CHECK: store i8* [[PRTARG_BITCAST]], i8** [[SCT_OFF]]
// CHECK: __do_print(i8* {{%[a-zA-Z0-9_.]+}}, i8* {{%[a-zA-Z0-9_.]+}}, i32 4, i64 {{%[a-zA-Z0-9_.]+}}, i8** [[PRTARG_PTR_BITCAST]])
void foo(bool bool_var) {
    print("\n bool_var = %\n", bool_var);
}
