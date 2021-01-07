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
    double v = double4((-3.141592741012573d0),(0.000000000000000d0),(3.141592741012573d0),(6.283185482025146d0));
    double ref = double4((-8.742277955963587d-08),(0.000000000000000d0),(8.742277955963587d-08),(1.7484555911927306d-07));
    RET[programIndex] = ok(tan(v), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
