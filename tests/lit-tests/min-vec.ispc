// This test checks min functions for short vectors. They are now defined in standard library.

// RUN: %{ispc} -O2 --vectorcall --target=avx512icl-x4 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

#include <short_vec.isph>

// REQUIRES: X86_ENABLED

// Note: current codegeneration is not ideal due to size rounding up to the
// next power of 2. See issue #3106 for more details.
// CHECK-LABEL: uniform_min_ded___uni_3C_3_3E_uni_3C_3_3E_
// CHECK-DAG: vpminsd xmm0, xmm0, xmm1
uniform int<3> uniform_min_ded(uniform int<3> a, uniform int<3> b) {
  return min(a, b);
}

// CHECK-LABEL: uniform_min_4_ded___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-DAG: vpminsd xmm0, xmm0, xmm1
// CHECK-NEXT: ret
uniform int<4> uniform_min_4_ded(uniform int<4> a, uniform int<4> b) {
  return min(a, b);
}

// CHECK-LABEL: varying_min_ded___vyi_3C_3_3E_vyi_3C_3_3E_
// CHECK-DAG: vpminsd xmm0, xmm0, xmm3
// CHECK-DAG: vpminsd xmm1, xmm1, xmm4
// CHECK-DAG: vpminsd xmm2, xmm2, xmm5
// CHECK-NEXT: ret
varying int<3> varying_min_ded(varying int<3> a, varying int<3> b) {
  return min(a, b);
}
