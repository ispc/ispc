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
    double v = double4((14.300010190734863d0),(6.700009809265136d0),(21.200010762939453d0),(9.000010000000000d0));
    double ref = double4((2.6602602499044092d0),(1.902108990465272d0),(3.0540016893636612d0),(2.197225688446713d0));
    RET[programIndex] = ok(log(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[4]) { RET[programIndex] = 1.; }
