#include "../test_static.isph"
// sincos() function is available only on Linux.
// rule: skip on OS=*
// rule: run on OS=linux
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-5) || abs((x-ref)/ref) < 1d-4; }
task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i];
    uniform float ref;
    uniform float ref2;
    sincos((float)arg, &ref, &ref2);
    uniform double res;
    uniform double res2;
    sincos(arg, &res, &res2);
    // native single precision sin and cos on Xe is pretty imprecise
    RET[i] = ok(res, ref) && ok(res2, ref2) ? 0.0 : 1.0;
  }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
