// This test is checking correct memory intrinsics generation for double/i64 types and 32 SIMD width.
// In particular it checks that for these types no svm_block.ld/svm_block_st or load/store are generated
// due to size limization, and that gathers/scatters are used instead.

// double simd32
// RUN: %{ispc} %s --target=xehpc-x32 --arch=xe64 -DUNIFORM -DTEST_DOUBLE --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM_DOUBLE %s
// RUN: %{ispc} %s --target=xehpc-x32 --arch=xe64 -DVARYING -DTEST_DOUBLE --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN:  FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING_DOUBLE_SIMD32
// RUN: %{ispc} %s --target=xehpc-x32 --arch=xe64 -DMASKED_ST -DTEST_DOUBLE --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_DOUBLE_SIMD32

// i64 simd32
// RUN: %{ispc} %s --target=xehpc-x32 --arch=xe64 -DUNIFORM -DTEST_I64 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM_I64 %s
// RUN: %{ispc} %s --target=xehpc-x32 --arch=xe64 -DVARYING -DTEST_I64 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN:  FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING_I64_SIMD32
// RUN: %{ispc} %s --target=xehpc-x32 --arch=xe64 -DMASKED_ST -DTEST_I64 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_I64_SIMD32

// REQUIRES: XE_ENABLED

#ifdef TEST_DOUBLE
typedef double T;
#elif TEST_I64
typedef int64 T;
#else
typedef float T;
#endif
task void vadd(uniform int n, uniform T aIN[], uniform T bIN[], uniform T cOUT[]) {
#ifdef UNIFORM
  for (uniform int i = 0; i < n; i += programCount) {
// CHECK_UNIFORM_DOUBLE:  %{{.*}} = load <32 x double>
// CHECK_UNIFORM_I64:     %{{.*}} = load <32 x i64>
    T a = aIN[i + programIndex];
// CHECK_UNIFORM_DOUBLE:  %{{.*}} = load <32 x double>
// CHECK_UNIFORM_I64:     %{{.*}} = load <32 x i64>
    T b = bIN[i + programIndex];
// CHECK_UNIFORM_DOUBLE:  store <32 x double> %add_a_load_b_load
// CHECK_UNIFORM_I64:     store <32 x i64> %add_a_load_b_load
    cOUT[i + programIndex] = a + b;
  }
#elif defined VARYING
// CHECK_VARYING_WARN: Performance Warning: Scatter required to store value.
// CHECK_VARYING_WARN: Performance Warning: Gather required to load value.
// CHECK_VARYING_WARN: Performance Warning: Gather required to load value.
  for (int i = 0; i < n; i += programCount) {
// CHECK_VARYING_DOUBLE_SIMD32: call <16 x double> @llvm.genx.svm.gather.v16f64.v16i1.v16i64
// CHECK_VARYING_DOUBLE_SIMD32-NEXT: call <16 x double> @llvm.genx.svm.gather.v16f64.v16i1.v16i64
// CHECK_VARYING_I64_SIMD32: call <16 x i64> @llvm.genx.svm.gather.v16i64.v16i1.v16i64
// CHECK_VARYING_I64_SIMD32-NEXT: call <16 x i64> @llvm.genx.svm.gather.v16i64.v16i1.v16i64
    T a = aIN[i + programIndex/2];
// CHECK_VARYING_DOUBLE_SIMD32: call <16 x double> @llvm.genx.svm.gather.v16f64.v16i1.v16i64
// CHECK_VARYING_DOUBLE_SIMD32-NEXT: call <16 x double> @llvm.genx.svm.gather.v16f64.v16i1.v16i64
// CHECK_VARYING_I64_SIMD32: call <16 x i64> @llvm.genx.svm.gather.v16i64.v16i1.v16i64
// CHECK_VARYING_I64_SIMD32-NEXT: call <16 x i64> @llvm.genx.svm.gather.v16i64.v16i1.v16i64
    T b = bIN[i + programIndex/2];
// CHECK_VARYING_DOUBLE_SIMD32: call void @llvm.genx.svm.scatter.v16i1.v16i64.v16f64
// CHECK_VARYING_DOUBLE_SIMD32-NEXT: call void @llvm.genx.svm.scatter.v16i1.v16i64.v16f64
// CHECK_VARYING_I64_SIMD32: call void @llvm.genx.svm.scatter.v16i1.v16i64.v16i64
// CHECK_VARYING_I64_SIMD32-NEXT: call void @llvm.genx.svm.scatter.v16i1.v16i64.v16i64
    cOUT[i + programIndex] = a + b;
  }
#elif defined MASKED_ST
// CHECK_MASKED_ST_WARN-NOT: Performance Warning: Gather required to load value.
// CHECK_MASKED_ST_WARN: Performance Warning: Scatter required to store value.
// CHECK_MASKED_ST_SAFE_WARN-NOT: Performance Warning: Gather required to load value.
// CHECK_MASKED_ST_SAFE_WARN: Performance Warning: Scatter required to store value.
  for (int i = 0; i < n; n += programCount) {
// CHECK_MASKED_ST_DOUBLE_SIMD32-COUNT-4: call <16 x double> @llvm.genx.svm.gather.v16f64.v16i1.v16i64
// CHECK_MASKED_ST_DOUBLE_SIMD32-COUNT-2: call void @llvm.genx.svm.scatter.v16i1.v16i64.v16f64

// CHECK_MASKED_ST_I64_SIMD32-COUNT-4: call <16 x i64> @llvm.genx.svm.gather.v16i64.v16i1.v16i64
// CHECK_MASKED_ST_I64_SIMD32-COUNT-2: call void @llvm.genx.svm.scatter.v16i1.v16i64.v16i64
    T a = aIN[i + programIndex];
    T b = bIN[i + programIndex];
    cOUT[i + programIndex] = a + b;
  }
#endif
}
