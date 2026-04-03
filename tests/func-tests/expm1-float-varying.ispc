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
    // expm1 is meant to be much more precise than exp for x ~= 0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-35) || abs((x - ref) / ref) < 2e-5;
}

task void f_v(uniform float RET[]) {
    float x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = float4(0.2, 0.42, 0.6, 2.1);
    ref = float4(0.22140276, 0.5219615, 8.2211882e-01, 7.1661692e+00);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = float4(9e-2, 4e-2, 3e-4, 2e-6);
    ref = float4(9.4174288e-02, 4.0810775e-02, 3.0004501e-04, 2.0000020e-06);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Tiny values
    x = float4(6e-12, 7e-16, 8e-20, 9e-24);
    ref = float4(6.0e-12, 7.0e-16, 8.0e-20, 9.0e-24);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Big values
    x = float4(5.9, 9.1, 17.2, 31.3);
    ref = float4(3.6403751e+02, 8.9542959e+03, 2.9502948e+07, 3.9211814e+13);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Negative values
    x = float4(-11.0, -0.3, -1e-2, -1e-8);
    ref = float4(-9.999833e-01, -2.591818e-01, -9.950166e-03, -1.000000e-08);
    res = expm1(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
