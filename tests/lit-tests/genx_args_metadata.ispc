// RUN: %{ispc} %s --target=genx-x16 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F1 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F1 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F2 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F2 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F3 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F3 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F4 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F4 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F5 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F5 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F6 %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F6 %s
// REQUIRES: GENX_ENABLED

// CHECK_EXPORT_F1: spir_kernel void @f1(i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %RET)
export void f1(uniform float RET[]) {}

// CHECK_TASK_F1: spir_kernel void @f1_task(i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %RET)
task void f1_task(uniform float RET[]) {}

// CHECK_EXPORT_F2: spir_kernel void @f2(i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aF, i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aD, i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aI)
export void f2(uniform float aF[], uniform double aD[], uniform int aI[]) {}

// CHECK_TASK_F2: spir_kernel void @f2_task(i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aF, i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aD, i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aI)
task void f2_task(uniform float aF[], uniform double aD[], uniform int aI[]) {}

// CHECK_EXPORT_F3: spir_kernel void @f3(i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aF1, i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)
export void f3(uniform float aF1[], uniform float aF2[], uniform float a) {}

// CHECK_TASK_F3: spir_kernel void @f3_task(i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aF1, i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)
task void f3_task(uniform float aF1[], uniform float aF2[], uniform float a) {}

// CHECK_EXPORT_F4: spir_kernel void @f4(i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aF1, i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)
export void f4(uniform float aF1[], double *uniform aF2, uniform float a) {}

// CHECK_TASK_F4: spir_kernel void @f4_task(i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aF1, i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)
task void f4_task(uniform float aF1[], double *uniform aF2, uniform float a) {}

// CHECK_EXPORT_F5: spir_kernel void @f5(i8 addrspace(1)* noalias nocapture "VCArgumentIOKind"="0" %_p)
export void f5(void *uniform _p) {}

// CHECK_TASK_F5: spir_kernel void @f5_task(i8 addrspace(1)* nocapture "VCArgumentIOKind"="0" %_p)
task void f5_task(void *uniform _p) {}

// CHECK_EXPORT_F6: spir_kernel void @f6(float "VCArgumentIOKind"="0" %a)
export void f6(uniform float a) {}

// CHECK_TASK_F6: spir_kernel void @f6_task(float "VCArgumentIOKind"="0" %a)
task void f6_task(uniform float a) {}
