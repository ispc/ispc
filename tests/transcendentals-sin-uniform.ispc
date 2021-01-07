#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
uniform bool ok(uniform double x, uniform double ref) { return (abs(x - ref) < 1d-5) || abs((x-ref)/ref) < 1d-4; }
task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i] + b;
    uniform double ref = sin((float)arg);
    uniform double res = sin(arg);
    uniform double delta = abs((res - ref) / ref);
    // native single precision sin on Xe is pretty imprecise
    RET[i] = ok(res, ref) ? 0. : 1.;
  }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
