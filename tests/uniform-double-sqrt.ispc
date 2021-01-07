#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

bool ok(double x, double ref) { return (abs(x - ref) < 1d-15) || abs((x-ref)/ref) < 1d-14; }

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform double f = aFOO[programCount/2];
    uniform double calc = sqrt(f) * sqrt(f) - aFOO[programCount/2];
    RET[programIndex] = ok(calc, 0) ? 0. : 1.;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
