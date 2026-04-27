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
    // log1p is meant to be much more precise than log(1+x) for x~=0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-35) || abs((x - ref) / ref) < 2e-5;
}

task void f_v(uniform float RET[]) {
    float x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = float4(-0.17, 0.019, 0.49, 0.93);
    ref = float4(-0.18632957, 0.01882175, 0.3987761, 0.65752);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = float4(4e-2, 3e-4, 2e-6, 1e-8);
    ref = float4(3.9220713e-02, 2.9995502e-04, 1.9999979e-06, 9.9999999e-09);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Medium values
    x = float4(2.11, 5.28, 9.03, 20.44);
    ref = float4(1.1346227, 1.83737, 2.3055806, 3.0652583);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Big values
    x = float4(73.5, 942.1, 6485.2, 647913.5);
    ref = float4(4.310799,  6.849172,  8.777432, 13.381514);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Negative values
    x = float4(-0.25, -0.5, -0.75, -0.99);
    ref = float4(-0.2876821, -0.6931472, -1.3862944, -4.605171);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 6: Special and extreme values
    x = float4(-0.0, +0.0, -0.99999, 1e25);
    ref = float4(-0.0, +0.0, -11.511568, 57.56463);
    res = log1p(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
