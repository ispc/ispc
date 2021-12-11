// RUN: %{ispc} %s --target=host --opt=fast-math --nostdlib --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK_WARN
// RUN: %{ispc} %s --target=host --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV

// CHECK_NOFDIV-NOT: fdiv


float16 f1(float16 a) {
  return a/2.f16;
}

// CHECK_WARN: Warning: rcp(varying float16) not found from stdlib. Can't apply fast-math rcp optimization.
float16 f2(float16 a, float16 b) {
  return a/b;
}

float f3(float a) {
  return a/2.f;
}

// CHECK_WARN: Warning: rcp(varying float) not found from stdlib. Can't apply fast-math rcp optimization.
float f4(float a, float b) {
  return a/b;
}

double f5(double a) {
  return a/2.d;
}

// CHECK_WARN: Warning: rcp(varying double) not found from stdlib. Can't apply fast-math rcp optimization.
double f6(double a, double b) {
  return a/b;
}
