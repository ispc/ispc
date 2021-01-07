#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
bool ok(double x, double ref) { return (abs(x - ref) < 1d-6) || abs((x-ref)/ref) < 1d-5; }
task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex] / (programCount + 2.0);
  varying double ref = asin((float)arg);
  varying double res = asin(arg);
  RET[programIndex] = ok(res, ref) ? 0. : 1.;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
