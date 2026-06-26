// This test checks that result of shuffle(T, int) and shuffle(T, T, int) with constant indexes is shufflevector.

// RUN: %{ispc} --target=avx512icl-x64 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// Previously XFAIL'd: __shuffle2_i8 for icl-x64 used to have its shuf2 call
// inlined too late for IsCompileTimeConstant to fold __is_compile_time_constant
// to true. The deferred-inlining + post-inline cleanup added for #3804 now
// folds the constant path correctly.

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
