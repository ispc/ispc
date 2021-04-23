// RUN: %{ispc} %s --target=genx-x16 --arch=genx64 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=genx-x16 --arch=genx64 -DVARYING --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING
// RUN: %{ispc} %s --target=genx-x16 --arch=genx64 -DMASKED_ST --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST
// RUN: %{ispc} %s --target=genx-x16 --arch=genx64 -DMASKED_ST --emit-llvm-text --opt=enable-genx-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE

// RUN: %{ispc} %s --target=genx-x16 --arch=genx32 -DUNIFORM --emit-llvm-text --nowrap -o %t.ll 2>&1
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_UNIFORM %s
// RUN: %{ispc} %s --target=genx-x16 --arch=genx32 -DVARYING --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_VARYING_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_VARYING
// RUN: %{ispc} %s --target=genx-x16 --arch=genx32 -DMASKED_ST --emit-llvm-text --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST
// RUN: %{ispc} %s --target=genx-x16 --arch=genx32 -DMASKED_ST --emit-llvm-text --opt=enable-genx-unsafe-masked-load --nowrap -o %t.ll 2>&1 | FileCheck %s -check-prefix=CHECK_MASKED_ST_WARN
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_MASKED_ST_UNSAFE

// REQUIRES: GENX_ENABLED
export void vadd(uniform int n, uniform float aIN[], uniform float bIN[], uniform float cOUT[]) {
#ifdef UNIFORM
  for (uniform int i = 0; i < n; i += programCount) {
// CHECK_UNIFORM: @llvm.genx.svm.block.ld
    float a = aIN[i + programIndex];
// CHECK_UNIFORM: @llvm.genx.svm.block.ld
    float b = bIN[i + programIndex];
// CHECK_UNIFORM: @llvm.genx.svm.block.st
    cOUT[i + programIndex] = a + b;
  }
#elif defined VARYING
// CHECK_VARYING_WARN: Performance Warning: Scatter required to store value.
// CHECK_VARYING_WARN: Performance Warning: Gather required to load value.
// CHECK_VARYING_WARN: Performance Warning: Gather required to load value.
  for (int i = 0; i < n; i += programCount) {
// CHECK_VARYING: @llvm.genx.svm.gather
    float a = aIN[i + programIndex/2];
// CHECK_VARYING: @llvm.genx.svm.gather
    float b = bIN[i + programIndex/2];
// CHECK_VARYING: @llvm.genx.svm.scatter
    cOUT[i + programIndex] = a + b;
  }
#elif defined MASKED_ST
// CHECK_MASKED_ST_WARN-NOT: Performance Warning: Gather required to load value.
// CHECK_MASKED_ST_WARN: Performance Warning: Scatter required to store value.
// CHECK_MASKED_ST_SAFE_WARN-NOT: Performance Warning: Gather required to load value.
// CHECK_MASKED_ST_SAFE_WARN: Performance Warning: Scatter required to store value.
  for (int i = 0; i < n; n += programCount) {
// CHECK_MASKED_ST: @llvm.genx.svm.block.ld
// CHECK_MASKED_ST: @llvm.genx.svm.block.ld
// CHECK_MASKED_ST: @llvm.genx.svm.gather
// CHECK_MASKED_ST: @llvm.genx.svm.gather
// CHECK_MASKED_ST: @llvm.genx.svm.scatter
// CHECK_MASKED_ST_UNSAFE: @llvm.genx.svm.block.ld
// CHECK_MASKED_ST_UNSAFE: @llvm.genx.svm.block.ld
// CHECK_MASKED_ST_UNSAFE-NOT: @llvm.genx.svm.gather
// CHECK_MASKED_ST_UNSAFE: @llvm.genx.svm.scatter
    float a = aIN[i + programIndex];
    float b = bIN[i + programIndex];
    cOUT[i + programIndex] = a + b;
  }
#endif
}
