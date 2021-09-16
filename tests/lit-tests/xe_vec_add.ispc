// This test is checking correct svm.ld/st svm.gather/scatter generation for different types and SIMD width.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DVARYING --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DMASKED_ST --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DMASKED_ST --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE

// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DVARYING --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DMASKED_ST --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DMASKED_ST --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe32 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll 2>&1
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe32 -DVARYING --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe32 -DMASKED_ST --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe32 -DMASKED_ST --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE

// RUN: %{ispc} %s --target=gen9-x8 --arch=xe32 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll 2>&1
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe32 -DVARYING --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe32 -DMASKED_ST --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe32 -DMASKED_ST --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE

// Now process special cases
// i8 simd16
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DUNIFORM -DTEST_I8 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DVARYING -DTEST_I8 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING_I8_SIMD16
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DMASKED_ST -DTEST_I8 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_I8_SIMD16
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DMASKED_ST -DTEST_I8 --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE_I8_SIMD16

// for i8 we always generate gathers/scatters since i8x8 is too small for svm ld/st
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DUNIFORM -DTEST_I8 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM_I8_SIMD8 %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DVARYING -DTEST_I8 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING_I8_SIMD8
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DMASKED_ST -DTEST_I8 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_I8_SIMD8
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DMASKED_ST -DTEST_I8 --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE_I8_SIMD8

// i16 simd16
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DUNIFORM -DTEST_I16 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DVARYING -DTEST_I16 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING_I16_SIMD16
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DMASKED_ST -DTEST_I16 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_I16_SIMD16
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -DMASKED_ST -DTEST_I16 --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE_I16_SIMD16

// i16 simd8
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DUNIFORM -DTEST_I16 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DVARYING -DTEST_I16 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING_I16_SIMD8
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DMASKED_ST -DTEST_I16 --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_I16_SIMD8
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -DMASKED_ST -DTEST_I16 --emit-llvm-text --opt=enable-xe-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE_I16_SIMD8

// REQUIRES: XE_ENABLED
#ifdef TEST_I8
typedef int8 T;
#elif TEST_I16
typedef int16 T;
// masked_load/masked_store implementation for i32, f32, i64 and f64 is identical
#else
typedef float T;
#endif
export void vadd(uniform int n, uniform T aIN[], uniform T bIN[], uniform T cOUT[]) {
#ifdef UNIFORM
  for (uniform int i = 0; i < n; i += programCount) {
// CHECK_UNIFORM: @llvm.genx.svm.block.ld.unaligned
// CHECK_UNIFORM_I8_SIMD8: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_UNIFORM_I8_SIMD8-NEXT: call <8 x i8> @llvm.genx.rdregioni.v8i8.v32i8.i16(<32 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 undef)
    T a = aIN[i + programIndex];
// CHECK_UNIFORM: @llvm.genx.svm.block.ld.unaligned
// CHECK_UNIFORM_I8_SIMD8: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_UNIFORM_I8_SIMD8-NEXT: call <8 x i8> @llvm.genx.rdregioni.v8i8.v32i8.i16(<32 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 undef)
    T b = bIN[i + programIndex];
// CHECK_UNIFORM: @llvm.genx.svm.block.st
// CHECK_UNIFORM_I8_SIMD8: call <32 x i8> @llvm.genx.wrregioni.v32i8.v8i8.i16.v8i1(<32 x i8> undef, <8 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_UNIFORM_I8_SIMD8-NEXT: call void @llvm.genx.svm.scatter.v8i1.v8i64.v32i8(<8 x i1> {{.*}}, i32 0, {{.*}})
    cOUT[i + programIndex] = a + b;
  }
#elif defined VARYING
// CHECK_VARYING_WARN: Performance Warning: Scatter required to store value.
// CHECK_VARYING_WARN: Performance Warning: Gather required to load value.
// CHECK_VARYING_WARN: Performance Warning: Gather required to load value.
  for (int i = 0; i < n; i += programCount) {
// CHECK_VARYING: @llvm.genx.svm.gather
// CHECK_VARYING_I8_SIMD16: call <64 x i8> @llvm.genx.svm.gather.v64i8.v16i1.v16i64(<16 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_VARYING_I8_SIMD16-NEXT: call <16 x i8> @llvm.genx.rdregioni.v16i8.v64i8.i16(<64 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 undef)
// CHECK_VARYING_I8_SIMD8: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_VARYING_I8_SIMD8-NEXT: call <8 x i8> @llvm.genx.rdregioni.v8i8.v32i8.i16(<32 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 undef)
// CHECK_VARYING_I16_SIMD16: call <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_VARYING_I16_SIMD16-NEXT: call <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16(<32 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 undef)
// CHECK_VARYING_I16_SIMD8: call <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_VARYING_I16_SIMD8-NEXT: call <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16(<16 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 undef)
    T a = aIN[i + programIndex/2];
// CHECK_VARYING: @llvm.genx.svm.gather
// CHECK_VARYING_I8_SIMD16: call <64 x i8> @llvm.genx.svm.gather.v64i8.v16i1.v16i64(<16 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_VARYING_I8_SIMD16-NEXT: call <16 x i8> @llvm.genx.rdregioni.v16i8.v64i8.i16(<64 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 undef)
// CHECK_VARYING_I8_SIMD8: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_VARYING_I8_SIMD8-NEXT: call <8 x i8> @llvm.genx.rdregioni.v8i8.v32i8.i16(<32 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 undef)
// CHECK_VARYING_I16_SIMD16: call <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_VARYING_I16_SIMD16-NEXT: call <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16(<32 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 undef)
// CHECK_VARYING_I16_SIMD8: call <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_VARYING_I16_SIMD8-NEXT: call <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16(<16 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 undef)
    T b = bIN[i + programIndex/2];
// CHECK_VARYING: @llvm.genx.svm.scatter
// CHECK_VARYING_I8_SIMD16: call <64 x i8> @llvm.genx.wrregioni.v64i8.v16i8.i16.v16i1(<64 x i8> undef, <16 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 0, <16 x i1> {{.*}})
// CHECK_VARYING_I8_SIMD16-NEXT: call void @llvm.genx.svm.scatter.v16i1.v16i64.v64i8(<16 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_VARYING_I8_SIMD8: call <32 x i8> @llvm.genx.wrregioni.v32i8.v8i8.i16.v8i1(<32 x i8> undef, <8 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_VARYING_I8_SIMD8-NEXT: call void @llvm.genx.svm.scatter.v8i1.v8i64.v32i8(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_VARYING_I16_SIMD16: call <32 x i16> @llvm.genx.wrregioni.v32i16.v16i16.i16.v16i1(<32 x i16> undef, <16 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 0, <16 x i1> {{.*}})
// CHECK_VARYING_I16_SIMD16-NEXT: call void @llvm.genx.svm.scatter.v16i1.v16i64.v32i16(<16 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_VARYING_I16_SIMD8: call <16 x i16> @llvm.genx.wrregioni.v16i16.v8i16.i16.v8i1(<16 x i16> undef, <8 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_VARYING_I16_SIMD8-NEXT: call void @llvm.genx.svm.scatter.v8i1.v8i64.v16i16(<8 x i1> {{.*}}, i32 1, <8 x i64> {{.*}})
    cOUT[i + programIndex] = a + b;
  }
#elif defined MASKED_ST
// CHECK_MASKED_ST_WARN-NOT: Performance Warning: Gather required to load value.
// CHECK_MASKED_ST_WARN: Performance Warning: Scatter required to store value.
// CHECK_MASKED_ST_SAFE_WARN-NOT: Performance Warning: Gather required to load value.
// CHECK_MASKED_ST_SAFE_WARN: Performance Warning: Scatter required to store value.
  for (int i = 0; i < n; n += programCount) {
// CHECK_MASKED_ST: load
// CHECK_MASKED_ST: load
// CHECK_MASKED_ST: @llvm.genx.svm.gather
// CHECK_MASKED_ST: @llvm.genx.svm.gather
// CHECK_MASKED_ST: @llvm.genx.svm.scatter

// CHECK_MASKED_ST_I8_SIMD16: load <16 x i8>
// CHECK_MASKED_ST_I8_SIMD16: load <16 x i8>
// CHECK_MASKED_ST_I8_SIMD16: call <64 x i8> @llvm.genx.svm.gather.v64i8.v16i1.v16i64(<16 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_MASKED_ST_I8_SIMD16-NEXT: call <16 x i8> @llvm.genx.rdregioni.v16i8.v64i8.i16(<64 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 undef)
// CHECK_MASKED_ST_I8_SIMD16: call <64 x i8> @llvm.genx.svm.gather.v64i8.v16i1.v16i64(<16 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_MASKED_ST_I8_SIMD16-NEXT: call <16 x i8> @llvm.genx.rdregioni.v16i8.v64i8.i16(<64 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 undef)
// CHECK_MASKED_ST_I8_SIMD16: call <64 x i8> @llvm.genx.wrregioni.v64i8.v16i8.i16.v16i1(<64 x i8> undef, <16 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 0, <16 x i1> {{.*}})
// CHECK_MASKED_ST_I8_SIMD16: call void @llvm.genx.svm.scatter.v16i1.v16i64.v64i8(<16 x i1> {{.*}}, i32 0, {{.*}})

// CHECK_MASKED_ST_I8_SIMD8-NOT: load <8 x i8>
// CHECK_MASKED_ST_I8_SIMD8-NOT: load <8 x i8>
// CHECK_MASKED_ST_I8_SIMD8: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_MASKED_ST_I8_SIMD8-NEXT: call <8 x i8> @llvm.genx.rdregioni.v8i8.v32i8.i16(<32 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 undef)
// CHECK_MASKED_ST_I8_SIMD8: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{.*}}, i32 0, {{.*}})
// CHECK_MASKED_ST_I8_SIMD8-NEXT: call <8 x i8> @llvm.genx.rdregioni.v8i8.v32i8.i16(<32 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 undef)
// CHECK_MASKED_ST_I8_SIMD8: call <32 x i8> @llvm.genx.wrregioni.v32i8.v8i8.i16.v8i1(<32 x i8> undef, <8 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_MASKED_ST_I8_SIMD8-NEXT: call void @llvm.genx.svm.scatter.v8i1.v8i64.v32i8(<8 x i1> {{.*}}, i32 0, {{.*}})

// CHECK_MASKED_ST_I16_SIMD16: load <16 x i16>
// CHECK_MASKED_ST_I16_SIMD16: load <16 x i16>
// CHECK_MASKED_ST_I16_SIMD16: call <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_MASKED_ST_I16_SIMD16-NEXT: call <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16(<32 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 undef)
// CHECK_MASKED_ST_I16_SIMD16: call <32 x i16> @llvm.genx.svm.gather.v32i16.v16i1.v16i64(<16 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_MASKED_ST_I16_SIMD16-NEXT: call <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16(<32 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 undef)
// CHECK_MASKED_ST_I16_SIMD16: call <32 x i16> @llvm.genx.wrregioni.v32i16.v16i16.i16.v16i1(<32 x i16> undef, <16 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 0, <16 x i1> {{.*}})
// CHECK_MASKED_ST_I16_SIMD16: call void @llvm.genx.svm.scatter.v16i1.v16i64.v32i16(<16 x i1> {{.*}}, i32 1, {{.*}})

// CHECK_MASKED_ST_I16_SIMD8: load <8 x i16>
// CHECK_MASKED_ST_I16_SIMD8: load <8 x i16>
// CHECK_MASKED_ST_I16_SIMD8: call <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_MASKED_ST_I16_SIMD8-NEXT: call <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16(<16 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 undef)
// CHECK_MASKED_ST_I16_SIMD8: call <16 x i16> @llvm.genx.svm.gather.v16i16.v8i1.v8i64(<8 x i1> {{.*}}, i32 1, {{.*}})
// CHECK_MASKED_ST_I16_SIMD8-NEXT: call <8 x i16> @llvm.genx.rdregioni.v8i16.v16i16.i16(<16 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 undef)
// CHECK_MASKED_ST_I16_SIMD8: call <16 x i16> @llvm.genx.wrregioni.v16i16.v8i16.i16.v8i1(<16 x i16> undef, <8 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_MASKED_ST_I16_SIMD8: call void @llvm.genx.svm.scatter.v8i1.v8i64.v16i16(<8 x i1> {{.*}}, i32 1, <8 x i64> {{.*}})

// CHECK_MASKED_ST_UNSAFE: load
// CHECK_MASKED_ST_UNSAFE: load
// CHECK_MASKED_ST_UNSAFE-NOT: @llvm.genx.svm.gather
// CHECK_MASKED_ST_UNSAFE: @llvm.genx.svm.scatter

// CHECK_MASKED_ST_UNSAFE_I8_SIMD16: load <16 x i8>
// CHECK_MASKED_ST_UNSAFE_I8_SIMD16: load <16 x i8>
// CHECK_MASKED_ST_UNSAFE_I8_SIMD16-NOT: @llvm.genx.svm.gather
// CHECK_MASKED_ST_UNSAFE_I8_SIMD16: call <64 x i8> @llvm.genx.wrregioni.v64i8.v16i8.i16.v16i1(<64 x i8> undef, <16 x i8> {{.*}}, i32 0, i32 16, i32 4, i16 0, i32 0, <16 x i1> {{.*}})
// CHECK_MASKED_ST_UNSAFE_I8_SIMD16: call void @llvm.genx.svm.scatter.v16i1.v16i64.v64i8(<16 x i1> {{.*}}, i32 0, {{.*}})

// CHECK_MASKED_ST_UNSAFE_I8_SIMD8: load <8 x i8>
// CHECK_MASKED_ST_UNSAFE_I8_SIMD8: load <8 x i8>
// CHECK_MASKED_ST_UNSAFE_I8_SIMD8-NOT: @llvm.genx.svm.gather
// CHECK_MASKED_ST_UNSAFE_I8_SIMD8: call <32 x i8> @llvm.genx.wrregioni.v32i8.v8i8.i16.v8i1(<32 x i8> undef, <8 x i8> {{.*}}, i32 0, i32 8, i32 4, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_MASKED_ST_UNSAFE_I8_SIMD8: call void @llvm.genx.svm.scatter.v8i1.v8i64.v32i8(<8 x i1> {{.*}}, i32 0, {{.*}})

// CHECK_MASKED_ST_UNSAFE_I16_SIMD16: load <16 x i16>
// CHECK_MASKED_ST_UNSAFE_I16_SIMD16: load <16 x i16>
// CHECK_MASKED_ST_UNSAFE_I16_SIMD16-NOT: @llvm.genx.svm.gather
// CHECK_MASKED_ST_UNSAFE_I16_SIMD16: call <32 x i16> @llvm.genx.wrregioni.v32i16.v16i16.i16.v16i1(<32 x i16> undef, <16 x i16> {{.*}}, i32 0, i32 16, i32 2, i16 0, i32 0, <16 x i1> {{.*}})
// CHECK_MASKED_ST_UNSAFE_I16_SIMD16: call void @llvm.genx.svm.scatter.v16i1.v16i64.v32i16(<16 x i1> {{.*}}, i32 1, {{.*}})

// CHECK_MASKED_ST_UNSAFE_I16_SIMD8: load <8 x i16>
// CHECK_MASKED_ST_UNSAFE_I16_SIMD8: load <8 x i16>
// CHECK_MASKED_ST_UNSAFE_I16_SIMD8-NOT: @llvm.genx.svm.gather
// CHECK_MASKED_ST_UNSAFE_I16_SIMD8: call <16 x i16> @llvm.genx.wrregioni.v16i16.v8i16.i16.v8i1(<16 x i16> undef, <8 x i16> {{.*}}, i32 0, i32 8, i32 2, i16 0, i32 0, <8 x i1> {{.*}})
// CHECK_MASKED_ST_UNSAFE_I16_SIMD8: call void @llvm.genx.svm.scatter.v8i1.v8i64.v16i16(<8 x i1> {{.*}}, i32 1, <8 x i64> {{.*}})

    T a = aIN[i + programIndex];
    T b = bIN[i + programIndex];
    cOUT[i + programIndex] = a + b;
  }
#endif
}
