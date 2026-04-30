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
    ref = float4(0.27329248, 1.1598289, 1.4558128, 7.185359);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 2: Small values
    x = float4(4e-2, 3e-4, 2e-6, 1e-8);
    ref = float4(4.001067e-02, 3.000000e-04, 2.000000e-06, 1.000000e-08);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 3: Big values
    x = float4(5.0, 7.0, 9.0, 11.0);
    ref = float4(74.20321, 548.3161, 4051.542, 29937.07);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 4: Negative values
    x = float4(-11.0, -3.0, -1e-2, -1e-8);
    ref = float4(-2.99370703e+04, -1.00178747e+01, -1.00001665e-02, -9.99999994e-09);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;

    // Case 5: Special and extreme values
    x = float4(-0.0, +0.0, 21.7, 80.0);
    ref = float4(-0.0, +0.0, 1.3278854e+09, 2.7703112e+34);
    res = sinh(x);
    RET[programIndex] += ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
