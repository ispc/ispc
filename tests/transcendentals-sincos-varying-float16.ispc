#include "../test_static.isph"
// sincos() function is available only on Linux.
// rule: skip on OS=*
// rule: run on OS=linux
// rule: skip on arch=x86
// rule: skip on arch=x86-64
bool ok(float16 x, float16 ref) { return (abs(x - ref) < 1e-4) || abs((x-ref)/ref) < 1e-3; }
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
  varying float16 arg = aFOO[programIndex];
  varying float ref;
  varying float ref2;
  sincos((float)arg, &ref, &ref2);
  varying float16 res;
  varying float16 res2;
  sincos(arg, &res, &res2);
  RET[programIndex] = ok(res, ref) && ok(res2, ref2) ? 0.0 : 1.0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
