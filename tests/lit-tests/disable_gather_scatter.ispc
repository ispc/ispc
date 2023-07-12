// Check that gather/scatter instructions are not generated on targets with native gathers/scatters support
// when --opt=disable-gathers/--opt=disable-scatters options are used.

// On targets without native gathers/scatters support, just check that compilation with --opt=disable-gathers/--opt=disable-scatters is successful
// RUN: %{ispc} %s -O2 --woff --target=sse2-i32x4 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse2-i32x4 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse2-i32x8 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse2-i32x8 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i8x16 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i8x16 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i16x8 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i16x8 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i32x4 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i32x4 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i32x8 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=sse4.2-i32x8 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i32x4 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i32x4 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i32x8 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i32x8 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i32x16 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i32x16 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i64x4 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx1-i64x4 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx2-i8x32 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx2-i8x32 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx2-i16x16 --addressing=32 --nostdlib --opt=disable-gathers --opt=disable-scatters
// RUN: %{ispc} %s -O2 --woff --target=avx2-i16x16 --addressing=64 --nostdlib --opt=disable-gathers --opt=disable-scatters

// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x4 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x4 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x8 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x8 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x16 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i32x16 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i64x4 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx2-i64x4 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512knl-x16 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512knl-x16 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x4 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x4 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x8 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x8 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x16 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x16 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x32 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x32 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x64 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512skx-x64 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x4 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x4 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x8 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x8 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x16 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x16 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x32 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x32 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x64 --addressing=32 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --woff --target=avx512spr-x64 --addressing=64 --nostdlib --emit-llvm-text --opt=disable-gathers --opt=disable-scatters -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_14_0+ && !MACOS_HOST

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
