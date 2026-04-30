#include "test_static.isph"

static float float4(uniform float a, uniform float b, uniform float c, uniform float d) {
    float ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}

bool ok(float x, float ref) {
    // erfc is meant to be much more precise than erf for x>1
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-35) || abs((x - ref) / ref) < 2e-5;
}

task void f_v(uniform float RET[]) {
    float x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = float4(0.27, 0.99, 1.17, 2.67);
    ref = float4(0.70258176, 0.16149193, 0.09799961, 0.00015939877);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = float4(4e-2, 3e-4, 2e-6, 1e-8);
    ref = float4(0.9548889, 0.9996615, 0.99999774, 1.0);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Big values
    x = float4(5.0, 7.0, 9.0, 11.0);
    ref = float4(1.5374598e-12, 4.1838257e-23, 4.1370317e-37, 0.0000000e+00);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Negative values
    x = float4(-11.0, -3.0, -1e-2, -1e-8);
    ref = float4(2.0, 1.999978, 1.0112834, 1.0);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Special and extreme values
    x = float4(-0.0, +0.0, -1e25, 1e25);
    ref = float4(1.0, 1.0, 2.0, 0.0);
    res = erfc(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
