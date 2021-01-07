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



bool ok(double x, double ref) { return (abs(x - ref) < 1d-15) || abs((x-ref)/ref) < 1d-13; }

task void f_v(uniform float RET[]) {
    double v = double4((3.141602741012573d),(0.000010000000000d),(3.141602741012573d),(6.283195482025146d));
    double ref = double4((1.144733096770642d0),(-11.512925464970229d0),(1.144733096770642d0),(1.8378786857850002d0));
    RET[programIndex] = ok(log(v), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
