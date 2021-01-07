#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

static double double4(uniform double a, uniform double b, uniform double c, 
                    uniform double d) {
    double ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}



bool ok(double x, double ref) { return (abs(x - ref) < 1d-15) || abs((x-ref)/ref) < 1d-14; }

task void f_v(uniform float RET[]) {
    double v = double4((1.000000000000000d0),(-1.000000000000000d0),(-1.5707963705062866d0),(1.5707963705062866d0));
    double ref = double4((0.8414709848078965d0),(-0.8414709848078965d0),(-1.000000000000000d0),(1.000000000000000d0));
    RET[programIndex] = ok(sin(v), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
