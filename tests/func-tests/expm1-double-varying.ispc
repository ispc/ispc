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
    // expm1 is meant to be much more precise than exp for x ~= 0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_v(uniform float RET[]) {
    double x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = double2(0.2d, 0.42d);
    ref = double2(0.22140275816016985d, 0.5219615556186338d);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = double2(9e-2d, 3e-4d);
    ref = double2(0.09417428370521036d, 0.00030004500450033747d);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Tiny values
    x = double2(6e-10d, 9e-120d);
    ref = double2(6.0000000018e-10d, 9e-120d);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Big values
    x = double2(2.3d, 4.7d);
    ref = double2(8.974182454814718d, 108.94717245212351d);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Bigger values
    x = double2(11.1d, 397.3d);
    ref = double2(66170.16016837658d, 3.5091154775566545e+172d);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 6: Negative values
    x = double2(-3.0d, -0.1d);
    ref = double2(-0.950212931632136d, -0.09516258196404043d);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
