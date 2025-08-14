// RUN: %{ispc} %s --target=avx512spr-x16 --opt=fast-math -O2 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_X64_ASM

// CHECK_X64_ASM-COUNT-1: fdiv {{((contract|reassoc|arcp|nsz|nnan|ninf|afn|fast) )*}}<8 x double> %a, %b


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
