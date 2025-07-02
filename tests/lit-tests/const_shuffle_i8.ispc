// This test checks that result of shuffle(T, int) and shuffle(T, T, int) with constant indexes is shufflevector.

// RUN: %{ispc} --target=avx512icl-x64 --nowrap -O2 --emit-llvm-text %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// XFAIL: *
// __shuffle2_i8 for icl-x64 has const path as other width, but optimization
// pipeline fails to inline shuf2 call before the IsCompileTimeConstant pass
// substitutes call to __is_compile_time_constant_varying_int32 to constant (it
// should be true in this case), so this test is expected to fail for icl-x64
// unless the inlining for this case is fixed.

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
