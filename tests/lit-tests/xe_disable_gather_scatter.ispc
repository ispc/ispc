// Check that gather/scatter instructions are not generated on Xe targets with native gathers/scatters support
// when --opt=disable-gathers/--opt=disable-scatters options are used.

// RUN: %{ispc} %s -O2 --woff --target=xehpg-x8 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=xehpg-x16 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=xehpc-x32 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s

// REQUIRES: XE_ENABLED

// CHECK-LABEL: gather_scatter_base_offset_int32___
// CHECK-NOT: @llvm.{{.*}}.gather{{.*}}
// CHECK-NOT: @llvm.{{.*}}.scatter{{.*}}

// CHECK-LABEL: gather_scatter_base_offset_float___
// CHECK-NOT: @llvm.{{.*}}.gather{{.*}}
// CHECK-NOT: @llvm.{{.*}}.scatter{{.*}}

// CHECK-LABEL: gather_scatter_base_offset_int64___
// CHECK-NOT: @llvm.{{.*}}.gather{{.*}}
// CHECK-NOT: @llvm.{{.*}}.scatter{{.*}}

// CHECK-LABEL: gather_scatter_base_offset_double___
// CHECK-NOT: @llvm.{{.*}}.gather{{.*}}
// CHECK-NOT: @llvm.{{.*}}.scatter{{.*}}

void *gptr;

void gather_scatter_base_offset_int32(uniform int32 pIn[], uniform int32 pOut[], int i) {
  pOut[programIndex+i] = pIn[i];
}

void gather_scatter_int32(int i) {
    int32 *ptr = (int32 * varying)gptr;
    int32 g = *ptr;
    ptr[programIndex+i] = g;
}

void gather_scatter_base_offset_float(uniform float pIn[], uniform float pOut[], int i) {
  pOut[programIndex+i] = pIn[i];
}

void gather_scatter_float(int i) {
    float *ptr = (float * varying)gptr;
    float g = *ptr;
    ptr[programIndex+i] = g;
}

void gather_scatter_base_offset_int64(uniform int64 pIn[], uniform int64 pOut[], int i) {
  pOut[programIndex+i] = pIn[i];
}

void gather_scatter_int64(int i) {
    int64 *ptr = (int64 * varying)gptr;
    int64 g = *ptr;
    ptr[programIndex+i] = g;
}

void gather_scatter_base_offset_double(uniform double pIn[], uniform double pOut[], int i) {
  pOut[programIndex+i] = pIn[i];
}

void gather_scatter_double(int i) {
    double *ptr = (double * varying)gptr;
    double g = *ptr;
    ptr[programIndex+i] = g;
}
