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
    // erfc is meant to be much more precise than erf for x>1
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_v(uniform float RET[]) {
    double x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = double2(0.27d, 0.99d);
    ref = double2(0.7025817814529872d, 0.16149193044463017d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = double2(2e-2d, 1e-12d);
    ref = double2(0.9774354253081551d, 0.9999999999988716d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Big values
    x = double2(3.0d, 4.0d);
    ref = double2(2.2090496998585445e-05d, 1.541725790028002e-08d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Bigger values
    x = double2(5.0d, 7.0d);
    ref = double2(1.5374597944280347e-12d, 4.183825607779414e-23d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Even bigger values
    x = double2(9.7d, 26.1d);
    ref = double2(7.936107563214489e-43d, 3.0812174933147107e-298d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 6: Negative values
    x = double2(-3.0d, -1e-2d);
    ref = double2(1.9999779095030015d, 1.0112834155558497d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 7: Special values
    x = double2(-0.0d, +0.0d);
    ref = double2(1.0d, 1.0d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 8: Extreme values
    x = double2(-1e160d, 1e160d);
    ref = double2(2.0d,  0.0d);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
