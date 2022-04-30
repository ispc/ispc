#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
uniform bool ok(uniform float16 x, uniform float16 ref) { return (abs(x - ref) < 1e-7) || abs((x-ref)/ref) < 1e-6; }
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform float16 arg = aFOO[i];
    uniform float16 arg2 = aFOO[i] + b;
    uniform float16 ref = atan2((float)arg, (float)arg2);
    uniform float16 res = atan2(arg, arg2);
    RET[i] = ok(res, ref) ? 0. : 1.;
  }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
