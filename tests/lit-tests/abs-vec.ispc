// This test checks abs functions for short vectors. They are now defined in standard library.

// RUN: %{ispc} -O2 --vectorcall --target=avx512icl-x4 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

#include <short_vec.isph>

// CHECK-LABEL: uniform_abs_4___uni_3C_4_3E_
// CHECK-DAG: vpabsd xmm0, xmm0
// CHECK-NEXT: ret
uniform int<4> uniform_abs_4(uniform int<4> a) {
  return abs(a);
}

// CHECK-LABEL: varying_abs___vyi_3C_3_3E_
// CHECK-DAG: vpabsd xmm0, xmm0
// CHECK-DAG: vpabsd xmm1, xmm1
// CHECK-DAG: vpabsd xmm2, xmm2
// CHECK-NEXT: ret
varying int<3> varying_abs(varying int<3> a) {
  return abs(a);
}
