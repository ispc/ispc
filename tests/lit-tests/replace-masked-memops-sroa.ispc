// RUN: %{ispc} %s -O2 --emit-llvm-text --target=avx512skx-i32x4 -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --emit-llvm-text --target=avx512skx-i32x8 -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --emit-llvm-text --target=avx512skx-i32x16 -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// For the following code, we expect that the SROA pass will successfully
// promote the local variable T from memory to registers, transforming
// the alloca/store/load instruction sequence into more efficient
// insertelement instructions.

// CHECK: define void @foo___REFs_5B_unData_5D_(ptr noalias nocapture %D
// CHECK-NEXT: allocas:
// CHECK-NOT:    alloca
// CHECK-NOT:    store
// CHECK:        [[T_VEC_INSERT_0:%.*]] = insertelement <4 x double> poison, double {{.*}}, i64 0
// CHECK-NEXT:   [[T_VEC_INSERT_1:%.*]] = insertelement <4 x double> [[T_VEC_INSERT_0]], double {{.*}}, i64 1
// CHECK-NEXT:   [[T_VEC_INSERT_2:%.*]] = insertelement <4 x double> [[T_VEC_INSERT_1]], double {{.*}}, i64 2
// CHECK-NEXT:   [[T_VEC_INSERT_3:%.*]] = insertelement <4 x double> [[T_VEC_INSERT_2]], double {{.*}}, i64 3
// CHECK-NEXT:   [[T_VEC_INSERT_4:%.*]] = insertelement <4 x double> poison, double {{.*}}, i64 0
// CHECK-NEXT:   [[T_VEC_INSERT_5:%.*]] = insertelement <4 x double> [[T_VEC_INSERT_4]], double {{.*}}, i64 1
// CHECK-NEXT:   [[T_VEC_INSERT_6:%.*]] = insertelement <4 x double> [[T_VEC_INSERT_5]], double {{.*}}, i64 2
// CHECK-NEXT:   [[T_VEC_INSERT_7:%.*]] = insertelement <4 x double> [[T_VEC_INSERT_6]], double {{.*}}, i64 3
// CHECK-NEXT:   [[ADD:%.*]] = fadd <4 x double> [[T_VEC_INSERT_3]], [[T_VEC_INSERT_7]]
// CHECK-NEXT:   store <4 x double> [[ADD]], ptr %D

struct Data {
  double Value[3][4];
};

void foo(uniform Data &D) {
  uniform Data T = D;
  foreach (i = 0 ... 4)
    D.Value[0][i] = T.Value[0][i] + T.Value[1][i];
}
