// This test checks correct emitting of LLVM IR functions definitions for Xe tasks, export and extern "C" functions.
// It also checks datalayout and metadata.
// RUN: %{ispc} %s --target=gen9-x16 --emit-llvm-text --no-discard-value-names -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F1_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F1_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F2_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F2_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F3_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F3_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F4_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F4_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F5_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F5_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F6_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F6_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F7_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F7_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_DATALAYOUT %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_FUNC_GPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_METADATA %s
// RUN: %{ispc} %s --target=gen9-x16 --emit-llvm-text -g --nowrap --no-discard-value-names -o - | FileCheck -check-prefix=CHECK_METADATA_DEBUG %s
// RUN: %{ispc} %s --target=gen9-x16 -DCHECK_DIAG --nowrap --emit-spirv -o %t.spv 2>&1
// RUN: not %{ispc} %s --target=gen9-x16 -DCHECK_DIAG_TASK --nowrap --emit-spirv -o %t.spv 2>&1 | FileCheck %s -check-prefix=CHECK_RET_VAL_TASK_ERROR

// CPU run to avoid regressions
// RUN: %{ispc} %s --target=avx2-i32x16 --emit-llvm-text --no-discard-value-names -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F1_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F1_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F2_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F2_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F3_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F3_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F4_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F4_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F5_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F5_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F6_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F6_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_EXPORT_F7_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_TASK_F7_CPU %s
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_FUNC_CPU %s
// RUN: %{ispc} %s --target=avx2-i32x16 -DCHECK_DIAG --nowrap -o %t.o 2>&1
// RUN: not %{ispc} %s --target=avx2-i32x16 -DCHECK_DIAG_TASK --nowrap -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_RET_VAL_TASK_ERROR
// REQUIRES: XE_ENABLED

// CHECK_DATALAYOUT: target datalayout = "e-p:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
// CHECK_DATALAYOUT: target triple = "spir64-unknown-unknown"

// CHECK_EXPORT_F1_GPU: void @f1__{{.*}}({{(float\*)|ptr}} noalias nocapture %RET, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F1_GPU: spir_func void @f1({{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %RET) {{.*}}
// CHECK_EXPORT_F1_GPU-NOT: spir_kernel void @f1({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %RET)
// CHECK_EXPORT_F1_CPU: void @f1__{{.*}}({{(float\*)|ptr}} noalias nocapture %RET, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F1_CPU: void @f1({{(float\*)|ptr}} noalias nocapture %RET)
export void f1(uniform float RET[]) {}

// CHECK_TASK_F1_GPU: dllexport spir_kernel void @f1_task({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %RET) {{.*}}
// CHECK_TASK_F1_GPU-NOT: void @f1_task__
// CHECK_TASK_F1_CPU: void @f1_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F1_CPU-NOT: void @f1_task({{(float\*)|ptr}} noalias nocapture %RET)
task void f1_task(uniform float RET[]) {}

// CHECK_EXPORT_F2_GPU: void @f2__{{.*}}({{(float\*)|ptr}} noalias nocapture %aF, {{(double\*)|ptr}} noalias nocapture %aD, {{(i32\*)|ptr}} noalias nocapture %aI, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F2_GPU: spir_func void @f2({{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aF, {{(double addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aD, {{(i32 addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aI) {{.*}}
// CHECK_EXPORT_F2_GPU-NOT: spir_kernel void @f2({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aF, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aD, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aI)
// CHECK_EXPORT_F2_CPU: void @f2__{{.*}}({{(float\*)|ptr}} noalias nocapture %aF, {{(double\*)|ptr}} noalias nocapture %aD, {{(i32\*)|ptr}} noalias nocapture %aI, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F2_CPU: void @f2({{(float\*)|ptr}} noalias nocapture %aF, {{(double\*)|ptr}} noalias nocapture %aD, {{(i32\*)|ptr}} noalias nocapture %aI)
export void f2(uniform float aF[], uniform double aD[], uniform int aI[]) {}

// CHECK_TASK_F2_GPU: dllexport spir_kernel void @f2_task({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aF, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aD, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aI)  {{.*}}
// CHECK_TASK_F2_GPU-NOT: void @f2_task__
// CHECK_TASK_F2_CPU: void @f2_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F2_CPU-NOT: void @f2_task({{(float\*)|ptr}} noalias nocapture %aF, {{(double\*)|ptr}} noalias nocapture %aD, {{(i32\*)|ptr}} noalias nocapture %aI)
task void f2_task(uniform float aF[], uniform double aD[], uniform int aI[]) {}

// CHECK_EXPORT_F3_GPU: void @f3__{{.*}}({{(float\*)|ptr}} noalias nocapture %aF1, {{(float\*)|ptr}} noalias nocapture %aF2, float %a, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F3_GPU: spir_func void @f3({{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aF1, {{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aF2, float %a) {{.*}}
// CHECK_EXPORT_F3_GPU-NOT: spir_kernel void @f3({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aF1, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)
// CHECK_EXPORT_F3_CPU: void @f3__{{.*}}({{(float\*)|ptr}} noalias nocapture %aF1, {{(float\*)|ptr}} noalias nocapture %aF2, float %a, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F3_CPU: void @f3({{(float\*)|ptr}} noalias nocapture %aF1, {{(float\*)|ptr}} noalias nocapture %aF2, float %a)
export void f3(uniform float aF1[], uniform float aF2[], uniform float a) {}

// CHECK_TASK_F3_GPU: dllexport spir_kernel void @f3_task({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aF1, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)  {{.*}}
// CHECK_TASK_F3_GPU-NOT: void @f3_task__
// CHECK_TASK_F3_CPU: void @f3_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F3_CPU-NOT: void @f3_task({{(float\*)|ptr}} noalias nocapture %aF1, {{(float\*)|ptr}} noalias nocapture %aF2, float %a)
task void f3_task(uniform float aF1[], uniform float aF2[], uniform float a) {}

// CHECK_EXPORT_F4_GPU: void @f4__{{.*}}({{(float\*)|ptr}} noalias nocapture %aF1, {{(double\*)|ptr}} noalias nocapture %aF2, float %a, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F4_GPU: spir_func void @f4({{(float addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aF1, {{(double addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias nocapture %aF2, float %a) {{.*}}
// CHECK_EXPORT_F4_GPU-NOT: spir_kernel void @f4({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aF1, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)
// CHECK_EXPORT_F4_CPU: void @f4__{{.*}}({{(float\*)|ptr}} noalias nocapture %aF1, {{(double\*)|ptr}} noalias nocapture %aF2, float %a, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F4_CPU: void @f4({{(float\*)|ptr}} noalias nocapture %aF1, {{(double\*)|ptr}} noalias nocapture %aF2, float %a)
export void f4(uniform float aF1[], double *uniform aF2, uniform float a) {}

// CHECK_TASK_F4_GPU: dllexport spir_kernel void @f4_task({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aF1, {{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %aF2, float "VCArgumentIOKind"="0" %a)  {{.*}}
// CHECK_TASK_F4_GPU-NOT: void @f4_task__
// CHECK_TASK_F4_CPU: void @f4_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F4_CPU-NOT: void @f4_task({{(float\*)|ptr}} noalias nocapture %aF1, {{(float\*)|ptr}} noalias nocapture %aF2, float %a)
task void f4_task(uniform float aF1[], double *uniform aF2, uniform float a) {}

// CHECK_EXPORT_F5_GPU: void @f5__{{.*}}({{(i8\*)|ptr}} noalias nocapture %_p, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F5_GPU: spir_func void @f5({{(i8 addrspace\(4\)\*|ptr addrspace\(4\))}} noalias nocapture %_p) {{.*}}
// CHECK_EXPORT_F5_GPU-NOT: spir_kernel void @f5({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} noalias nocapture "VCArgumentIOKind"="0" %_p)
// CHECK_EXPORT_F5_CPU: void @f5__{{.*}}({{(i8\*)|ptr}} noalias nocapture %_p, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F5_CPU: void @f5({{(i8\*)|ptr}} noalias nocapture %_p)
export void f5(void *uniform _p) {}

// CHECK_TASK_F5_GPU: dllexport spir_kernel void @f5_task({{(i8 addrspace\(1\)\*|ptr addrspace\(1\))}} {{.*}} "VCArgumentIOKind"="0" %_p)  {{.*}}
// CHECK_TASK_F5_GPU-NOT: void @f5_task__
// CHECK_TASK_F5_CPU: void @f5_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F5_CPU-NOT: void @f5_task({{(i8\*)|ptr}} noalias nocapture %_p)
task void f5_task(void *uniform _p) {}

// CHECK_EXPORT_F6_GPU: void @f6__{{.*}}(float %a, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F6_GPU: spir_func void @f6(float %a) {{.*}}
// CHECK_EXPORT_F6_GPU-NOT: spir_kernel void @f6(float "VCArgumentIOKind"="0" %a)
// CHECK_EXPORT_F6_CPU: void @f6__{{.*}}(float %a, <16 x {{.*}}> %__mask)
// CHECK_EXPORT_F6_CPU: void @f6(float %a)
export void f6(uniform float a) {}

// CHECK_TASK_F6_GPU: dllexport spir_kernel void @f6_task(float "VCArgumentIOKind"="0" %a)  {{.*}}
// CHECK_TASK_F6_GPU-NOT: void @f6_task__
// CHECK_TASK_F6_CPU: void @f6_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F6_CPU-NOT: void @f6_task(float %a)
task void f6_task(uniform float a) {}

// CHECK_EXPORT_F7_GPU: void @f7__{{.*}}(<16 x {{.*}}> %__mask)
// CHECK_EXPORT_F7_GPU: spir_func void @f7() {{.*}}
// CHECK_EXPORT_F7_GPU-NOT: spir_kernel void @f7()
// CHECK_EXPORT_F7_CPU: void @f7__{{.*}}(<16 x {{.*}}> %__mask)
// CHECK_EXPORT_F7_CPU: void @f7()
export void f7() {}

// CHECK_TASK_F7_GPU: dllexport spir_kernel void @f7_task()  {{.*}}
// CHECK_TASK_F7_GPU-NOT: void @f7_task__
// CHECK_TASK_F7_CPU: void @f7_task__{{.*}} noalias {{.*}} %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10)
// CHECK_TASK_F7_CPU-NOT: void @f7_task()
task void f7_task() {}

// If there is no "CMGenxMain" attribute for function, "VCFunction" attribute
// should always go with "VCStackCall" attribute for function
// definitions and declarations
// CHECK_METADATA: { {{.*}} "CMStackCall" "VCFunction" "VCStackCall" }
// CHECK_METADATA-NOT: { {{.*}} "VCFunction" }
// CHECK_METADATA: { {{.*}} "CMGenxMain" "VCFunction"{{.*}} "oclrt"="1" }

// CHECK_METADATA-NOT: !"Debug Info Version"
// CHECK_METADATA_DEBUG: !"Debug Info Version"

// Check linkage and inlining of non-kernel/non-export functions
// On GPU the functions have alwaysinline attribute by default
// CHECK_FUNC_GPU: alwaysinline
// CHECK_FUNC_GPU-NEXT: @t1__
// CHECK_FUNC_CPU-NOT: alwaysinline
// CHECK_FUNC_CPU: @t1__
uniform int t1() {
    return programCount;
}
// CHECK_FUNC_GPU-NOT: alwaysinline
// CHECK_FUNC_GPU: @t1_noinline__
// CHECK_FUNC_CPU-NOT: alwaysinline
// CHECK_FUNC_CPU: @t1_noinline__
noinline uniform int t1_noinline() {
    return programCount;
}
// CHECK_FUNC_GPU-NOT: @t2__
// CHECK_FUNC_CPU-NOT: @t2__
static uniform int t2() {
    return programCount;
}
// CHECK_FUNC_GPU: alwaysinline
// CHECK_FUNC_GPU: @t3__
// CHECK_FUNC_CPU-NOT: alwaysinline
// CHECK_FUNC_CPU: @t3__
extern uniform int t3() {
    return programCount;
}
// CHECK_FUNC_GPU-NOT: alwaysinline
// CHECK_FUNC_GPU: @t4(
// CHECK_FUNC_CPU-NOT: alwaysinline
// CHECK_FUNC_CPU: @t4(
extern "C" uniform int t4(int p);
extern uniform int t4_call() {
    return t4(programCount);
}

// Check signature of extern "C" functions
// CHECK_FUNC_GPU: declare spir_func void @t5({{(i32 addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias) {{.*}}
// CHECK_FUNC_CPU: declare void @t5({{(i32\*)|ptr}} noalias)
extern "C" void t5(uniform int p[]);
// CHECK_FUNC_GPU: void @t5_call___{{.*}}
// CHECK_FUNC_GPU: addrspacecast {{(i32\*)|ptr}} %p to {{(i32 addrspace\(4\)\*)|(ptr addrspace\(4\))}}
// CHECK_FUNC_GPU: call spir_func void @t5({{(i32 addrspace\(4\)\*)|(ptr addrspace\(4\))}} %{{.*}})

// CHECK_FUNC_GPU: spir_func void @t5_call({{(i32 addrspace\(4\)\*)|(ptr addrspace\(4\))}} noalias %p) {{.*}}
// CHECK_FUNC_GPU: call spir_func void @t5({{(i32 addrspace\(4\)\*)|(ptr addrspace\(4\))}} %p)

// CHECK_FUNC_CPU: void @t5_call___{{.*}}
// CHECK_FUNC_CPU: call void @t5({{(i32\*)|ptr}} %p)
// CHECK_FUNC_CPU: void @t5_call({{(i32\*)|ptr}} noalias %p)
// CHECK_FUNC_CPU: call void @t5({{(i32\*)|ptr}} %p)

export void t5_call(uniform int p[]) {
    t5(p);
}

// Diagnostics
#ifdef CHECK_DIAG
export uniform int check_diag() {
    return programCount;
}
#elif CHECK_DIAG_TASK
// CHECK_RET_VAL_TASK_ERROR: Error: Task-qualified functions must have void return type.
task uniform int check_diag_task() {
    return programCount;
}
#endif
