#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
bool ok(float16 x, float16 ref) { return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-5; }
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
  varying float16 arg = aFOO[programIndex] / (programCount + 2.0f16);
  varying float16 ref = asin((float)arg);
  varying float16 res = asin(arg);
  RET[programIndex] = ok(res, ref) ? 0. : 1.;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
