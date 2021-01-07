#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-7) || abs((x-ref)/ref) < 1d-6; }
task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i] / (programCount + 2.0);
    uniform double ref = acos((float)arg);
    uniform double res = acos(arg);
    RET[i] = ok(res, ref) ? 0. : 1.;
  }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
