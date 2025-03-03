// This test checks max functions for short vectors. They are now defined in standard library.

// RUN: %{ispc} -O2 --vectorcall --target=avx512icl-x4 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// Note: current codegeneration is not ideal due to size rounding up to the
// next power of 2. See issue #3106 for more details.
// CHECK-LABEL: uniform_max___uni_3C_3_3E_uni_3C_3_3E_
// CHECK-DAG: vpmaxsd xmm0, xmm0, xmm1
uniform int<3> uniform_max(uniform int<3> a, uniform int<3> b) {
  return max<int, 3>(a, b);
}

// CHECK-LABEL: uniform_max_ded___uni_3C_3_3E_uni_3C_3_3E_
// CHECK-DAG: vpmaxsd xmm0, xmm0, xmm1
uniform int<3> uniform_max_ded(uniform int<3> a, uniform int<3> b) {
  return max(a, b);
}

// CHECK-LABEL: uniform_max_4___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-DAG: vpmaxsd xmm0, xmm0, xmm1
// CHECK-NEXT: ret
uniform int<4> uniform_max_4(uniform int<4> a, uniform int<4> b) {
  return max<int, 4>(a, b);
}

// CHECK-LABEL: uniform_max_4_ded___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-DAG: vpmaxsd xmm0, xmm0, xmm1
// CHECK-NEXT: ret
uniform int<4> uniform_max_4_ded(uniform int<4> a, uniform int<4> b) {
  return max(a, b);
}

// CHECK-LABEL: varying_max___vyi_3C_3_3E_vyi_3C_3_3E_
// CHECK-DAG: vpmaxsd xmm0, xmm0, xmm3
// CHECK-DAG: vpmaxsd xmm1, xmm1, xmm4
// CHECK-DAG: vpmaxsd xmm2, xmm2, xmm5
// CHECK-NEXT: ret
varying int<3> varying_max(varying int<3> a, varying int<3> b) {
  return max<int, 3>(a, b);
}

// CHECK-LABEL: varying_max_ded___vyi_3C_3_3E_vyi_3C_3_3E_
// CHECK-DAG: vpmaxsd xmm0, xmm0, xmm3
// CHECK-DAG: vpmaxsd xmm1, xmm1, xmm4
// CHECK-DAG: vpmaxsd xmm2, xmm2, xmm5
// CHECK-NEXT: ret
varying int<3> varying_max_ded(varying int<3> a, varying int<3> b) {
  return max(a, b);
}
