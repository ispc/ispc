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



bool ok(double x, double ref) { 
    bool r = (abs(x - ref) < 1d-14) || abs((x-ref)/ref) < 1d-14;
    if (any(r == false))
#ifndef ISPC_TARGET_GENX
        print("mismatch got %, expected %\n", x, ref);
#endif
    return r;
}


export void f_v(uniform float RET[]) {
    double v = double4((14.300000190734863d0),(-6.699999809265137d0),(-21.200000762939453d0),(9.000000000000000d0));
    double ref = double4((-0.1621146247115303d0),(0.9143832254542971d0),(-0.7030295000381365d0),(-0.9111302618846769d0));
    RET[programIndex] = ok(cos(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
