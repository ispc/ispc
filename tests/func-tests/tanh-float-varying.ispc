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
    return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-5;
}

task void f_v(uniform float RET[]) {
    float x, ref, res;

    RET[programIndex] = 0;

    // Case 1: Basic values
    x = float4(0.27, 0.99, 1.17, 2.67);
    ref = float4(0.26362484, 0.75736232, 0.82427217, 0.99045404);
    res = tanh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = float4(4e-2, 3e-4, 2e-6, 1e-8);
    ref = float4(3.99786803e-02, 2.99999991e-04, 2.00000000e-06, 1.00000000e-08);
    res = tanh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Big values
    x = float4(5.0, 7.0, 9.0, 11.0);
    ref = float4(0.9999092 , 0.99999834, 0.99999997, 1.0);
    res = tanh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Negative values
    x = float4(-11.0, -3.0, -1e-2, -1e-8);
    ref = float4(-9.99999999e-01, -9.95054754e-01, -9.99966668e-03, -1.00000000e-08);
    res = tanh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Special and extreme values
    x = float4(-0.0, +0.0, -1e25, 1e25);
    ref = float4(-0.0,  +0.0,  -1.0,  1.0);
    res = tanh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
