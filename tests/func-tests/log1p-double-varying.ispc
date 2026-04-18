#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

static double double2(uniform double a, uniform double b) {
    double ret = 0;
    for (uniform int i = 0; i < programCount; i += 2) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
    }
    return ret;
}

bool ok(double x, double ref) {
    // log1p is meant to be much more precise than log(1+x) for x~=0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_v(uniform float RET[]) {
    double x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = double2(0.019d, 0.49d);
    ref = double2(0.01882175424058776d, 0.3987761199573678d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = double2(2e-2d, 1e-12d);
    ref = double2(0.019802627296179712d, 9.999999999995e-13d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Medium values
    x = double2(2.11d, 9.03d);
    ref = double2(1.1346227261911428d, 2.305580601973844d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Big values
    x = double2(73.5d, 6485.2d);
    ref = double2(4.310799125385514d, 8.777432122037862d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Bigger values
    x = double2(647913.5d, 123456789.0123777d);
    ref = double2(13.381514022184533d, 18.63140177436828d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 6: Negative values
    x = double2(-0.25d, -0.5d);
    ref = double2(-0.2876820724517809d, -0.6931471805599453d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 7: More negative values
    x = double2(-0.77d, -0.99d);
    ref = double2(-1.4696759700589417d, -4.60517018598809d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 8: Special values
    x = double2(-0.0d, +0.0d);
    ref = double2(-0.0d,  +0.0d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 9: Extreme values
    x = double2(-0.9999999999999d, 3.42e179d);
    ref = double2(-29.933295312068765d, 413.3923721970087d);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
