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
    return (abs(x - ref) < 1e-15d) || abs((x-ref)/ref) < 1e-12d;
}

task void f_v(uniform float RET[]) {
    double x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = double2(0.27d, 0.99d);
    ref = double2(0.27329247819819713d, 1.1598288906636083d);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = double2(2e-2d, 1e-12d);
    ref = double2(0.020001333360000255d, 1e-12d);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Big values
    x = double2(5.0d, 7.0d);
    ref = double2(74.20321057778875d, 548.3161232732465d);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Negative values
    x = double2(-3.0d, -1e-2d);
    ref = double2(-10.017874927409903d, -0.010000166667500003d);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Special values
    x = double2(-0.0d, +0.0d);
    ref = double2(-0.0d,  +0.0d);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 6: Extreme values
    x = double2(80.0d, 379.1d);
    ref = double2(2.770311192196755e+34d, 2.1878023972022845e+164d);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
