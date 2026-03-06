// This test checks the optimizations performed by the different fast-math modes on divisions at the IR level.

// REQUIRES: X86_ENABLED

// RUN: %{ispc} %s --target=avx512skx-x16 --opt=fast-math -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_IR_DEFAULT
// RUN: %{ispc} %s --target=avx512skx-x16 --opt=fast-math:legacy -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_IR_LEGACY
// RUN: %{ispc} %s --target=avx512skx-x16 --opt=fast-math:balanced -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_IR_BALANCED
// RUN: %{ispc} %s --target=avx512skx-x16 --opt=fast-math:aggressive -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_IR_AGGRESSIVE

// The default mode is the legacy one
// CHECK_X64_IR_DEFAULT-NOT: fdiv <16 x {{(half|float|double)}}>
// CHECK_X64_IR_DEFAULT-COUNT-2: fmul <16 x half> {{%a|.*, %a}}
// CHECK_X64_IR_DEFAULT-COUNT-2: fmul <16 x float> {{%a|.*, %a}}
// CHECK_X64_IR_DEFAULT-COUNT-2: fmul <16 x double> {{%a|.*, %a}}

// In legacy mode, rcp+fmul shall be performed systematically instead of basic divisions
// CHECK_X64_IR_LEGACY-NOT: fdiv <16 x {{(half|float|double)}}>
// CHECK_X64_IR_LEGACY-COUNT-2: fmul <16 x half> {{%a|.*, %a}}
// CHECK_X64_IR_LEGACY-COUNT-2: fmul <16 x float> {{%a|.*, %a}}
// CHECK_X64_IR_LEGACY-COUNT-2: fmul <16 x double> {{%a|.*, %a}}

// In balanced mode, math operations shall be tagged and divisions shall not be approximated.
// Divisions can be replace by a multiplication for constants here.
// CHECK_X64_IR_BALANCED-COUNT-1: fmul {{((contract|reassoc|nsz|arcp) )+}}<16 x half> {{%a|.*, %a}}
// CHECK_X64_IR_BALANCED-COUNT-1: fdiv {{((contract|reassoc|nsz|arcp) )+}}<16 x half> {{%a|.*, %a}}
// CHECK_X64_IR_BALANCED-COUNT-1: fmul {{((contract|reassoc|nsz|arcp) )+}}<16 x float> {{%a|.*, %a}}
// CHECK_X64_IR_BALANCED-COUNT-1: fdiv {{((contract|reassoc|nsz|arcp) )+}}<16 x float> {{%a|.*, %a}}
// CHECK_X64_IR_BALANCED-COUNT-1: fmul {{((contract|reassoc|nsz|arcp) )+}}<16 x double> {{%a|.*, %a}}
// CHECK_X64_IR_BALANCED-COUNT-1: fdiv {{((contract|reassoc|nsz|arcp) )+}}<16 x double> {{%a|.*, %a}}

// In aggressive mode, math operations shall be tagged, divisions can be optimized out and may even be aproximated.
// That being said, approximations are typically done in the generated assembly and not the in the IR.
// It shall at least optimize divisions by constants in the IR.
// CHECK_X64_IR_AGGRESSIVE: fmul {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )+}}<16 x half> {{%a|.*, %a}}
// CHECK_X64_IR_AGGRESSIVE: fmul {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )+}}<16 x float> {{%a|.*, %a}}
// CHECK_X64_IR_AGGRESSIVE: fmul {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )+}}<16 x double> {{%a|.*, %a}}

float16 f1(float16 a) {
  return a/2.f16;
}

float16 f2(float16 a, float16 b) {
  return a/b;
}

float f3(float a) {
  return a/2.f;
}

float f4(float a, float b) {
  return a/b;
}

double f5(double a) {
  return a/2.d;
}

double f6(double a, double b) {
  return a/b;
}
