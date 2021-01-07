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



bool ok(double x, double ref) {
    bool r = (abs(x - ref) < 1d-14) || abs((x-ref)/ref) < 1d-14;
    return r;
}

task void f_v(uniform float RET[]) {
    double v = double4((-9.424777984619141d0),(4.000000000000000d0),(10.000000000000000d0),(-10.000000000000000d0));
    double ref = double4((2.3849760909612067d-08),(-0.7568024953079282d0),(-0.5440211108893699d0),(0.5440211108893699d0));
    RET[programIndex] = ok(sin(v), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
