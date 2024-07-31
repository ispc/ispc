// This test fails on avx512-x64 targets due to issue in InstCombine.

// RUN: %{ispc} --target=avx512skx-x64 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s
// REQUIRES: X86_ENABLED
// XFAIL: *

template <typename T>
unmasked void shuf1(uniform T a[], uniform T ret[], int perm) {
    ret[programIndex] = shuffle(a[programIndex], perm);
}

// CHECK-LABEL: shuf1_int8
// CHECK: shufflevector
unmasked void shuf1_int8(uniform int8 a[], uniform int8 ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

// CHECK-LABEL: shuf1_int16
// CHECK: shufflevector
unmasked void shuf1_int16(uniform int16 a[], uniform int16 ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

// CHECK-LABEL: shuf1_float16
// CHECK: shufflevector
unmasked void shuf1_float16(uniform float16 a[], uniform float16 ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

// CHECK-LABEL: shuf1_int
// CHECK: shufflevector
unmasked void shuf1_int(uniform int a[], uniform int ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

// CHECK-LABEL: shuf1_float
// CHECK: shufflevector
unmasked void shuf1_float(uniform float a[], uniform float ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

// CHECK-LABEL: shuf1_int64
// CHECK: shufflevector
unmasked void shuf1_int64(uniform int64 a[], uniform int64 ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

// CHECK-LABEL: shuf1_double
// CHECK: shufflevector
unmasked void shuf1_double(uniform double a[], uniform double ret[]) {
    shuf1(a, ret, programCount - 1 - programIndex);
}

template <typename T>
unmasked void shuf2(uniform T a[], uniform T ret[], int perm) {
    T aa = a[programIndex]; 
    T bb = aa + programCount;
    ret[programIndex] = shuffle(aa, bb, perm);
}

// CHECK-LABEL: shuf2_int8
// CHECK: shufflevector
unmasked void shuf2_int8(uniform int8 a[], uniform int8 ret[]) {
    shuf2(a, ret, programCount + 1);
}

// CHECK-LABEL: shuf2_int16
// CHECK: shufflevector
unmasked void shuf2_int16(uniform int16 a[], uniform int16 ret[]) {
    shuf2(a, ret, programCount + 1);
}

// CHECK-LABEL: shuf2_float16
// CHECK: shufflevector
unmasked void shuf2_float16(uniform float16 a[], uniform float16 ret[]) {
    shuf2(a, ret, programCount + 1);
}

// CHECK-LABEL: shuf2_int
// CHECK: shufflevector
unmasked void shuf2_int(uniform int a[], uniform int ret[]) {
    shuf2(a, ret, programCount + 1);
}

// CHECK-LABEL: shuf2_float
// CHECK: shufflevector
unmasked void shuf2_float(uniform float a[], uniform float ret[]) {
    shuf2(a, ret, programCount + 1);
}

// CHECK-LABEL: shuf2_int64
// CHECK: shufflevector
unmasked void shuf2_int64(uniform int64 a[], uniform int64 ret[]) {
    shuf2(a, ret, programCount + 1);
}

// CHECK-LABEL: shuf2_double
// CHECK: shufflevector
unmasked void shuf2_double(uniform double a[], uniform double ret[]) {
    shuf2(a, ret, programCount + 1);
}