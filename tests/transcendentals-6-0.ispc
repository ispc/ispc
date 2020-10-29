// rule: skip on cpu=TGLLP

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

export void f_v(uniform float RET[]) {
    double v = double4((-3.141592653589793d0),(0.000000000000000d0),(3.141592653589793d0),(6.283185307179586d0));
    double ref = double4((-1.000000000000000d0),(1.000000000000000d0),(-1.000000000000000d0),(1.000000000000000d0));
    RET[programIndex] = ok(cos(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
