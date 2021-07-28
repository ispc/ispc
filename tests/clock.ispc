#include "../test_static.isph"
// issue #1783 for ARM support.
// rule: skip on arch=arm
// rule: skip on arch=aarch64


float x;

task void f_f(uniform float RET[], uniform float aFOO[]) {
  unsigned uniform int64 a = clock();
  x = pow(sqrt(aFOO[programIndex]), 5.5);
  unsigned uniform int64 b = clock();
  RET[programIndex] = (b - a) > 0 ? 1 : 0;
}

task void result(uniform float RET[]) {
  RET[programIndex] = 1;
}
