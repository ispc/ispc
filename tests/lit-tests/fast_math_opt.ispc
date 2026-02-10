// RUN: %{ispc} %s --target=avx512spr-x16 --opt=fast-math -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_ASM_DEFAULT
// RUN: %{ispc} %s --target=avx512spr-x16 --opt=fast-math:legacy -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_ASM_LEGACY
// RUN: %{ispc} %s --target=avx512spr-x16 --opt=fast-math:balanced -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_ASM_BALANCED
// RUN: %{ispc} %s --target=avx512spr-x16 --opt=fast-math:aggressive -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_ASM_AGGRESSIVE

// Generic target stdlib's rcp implementation uses 1.0/x (fdiv) rather than
// a hardware rcp instruction, so the CHECK_NOFDIV check fails on ppc64le.
// XFAIL: PPC64LE_HOST

// The default mode is the legacy one
// CHECK_X64_ASM_DEFAULT-NOT: fdiv <16 x {{(half|float|double)}}>
// CHECK_X64_ASM_DEFAULT-COUNT-2: fmul <16 x half> %a
// CHECK_X64_ASM_DEFAULT-COUNT-2: fmul <16 x float> %a
// CHECK_X64_ASM_DEFAULT-COUNT-2: fmul <16 x double> %a

// In legacy mode, rcp+fmul should be performed systematically instead of plain divisions
// CHECK_X64_ASM_LEGACY-NOT: fdiv <16 x {{(half|float|double)}}>
// CHECK_X64_ASM_LEGACY-COUNT-2: fmul <16 x half> %a
// CHECK_X64_ASM_LEGACY-COUNT-2: fmul <16 x float> %a
// CHECK_X64_ASM_LEGACY-COUNT-2: fmul <16 x double> %a

// In balanced mode, math operations should be tagged and divisions should not be optimized out, except for constants.
// CHECK_X64_ASM_BALANCED-COUNT-1: fmul {{((contract|reassoc|nsz) )+}}<16 x half> %a
// CHECK_X64_ASM_BALANCED-COUNT-1: fdiv {{((contract|reassoc|nsz) )+}}<16 x half> %a
// CHECK_X64_ASM_BALANCED-COUNT-1: fmul {{((contract|reassoc|nsz) )+}}<16 x float> %a
// CHECK_X64_ASM_BALANCED-COUNT-1: fdiv {{((contract|reassoc|nsz) )+}}<16 x float> %a
// CHECK_X64_ASM_BALANCED-COUNT-1: fmul {{((contract|reassoc|nsz) )+}}<16 x double> %a
// CHECK_X64_ASM_BALANCED-COUNT-1: fdiv {{((contract|reassoc|nsz) )+}}<16 x double> %a

// In aggressive mode, math operations should be tagged and division can be optimized out (but may not always be).
// It should at least optimize divisions by constants.
// CHECK_X64_ASM_AGGRESSIVE: fmul {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )+}}<16 x half> %a
// CHECK_X64_ASM_AGGRESSIVE: fmul {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )+}}<16 x float> %a
// CHECK_X64_ASM_AGGRESSIVE: fmul {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )+}}<16 x double> %a

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
