#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
static float float4(uniform float16 a, uniform float16 b, uniform float16 c, uniform float16 d) {
    float ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}

bool ok(uniform float16 x, uniform float16 ref) { return (abs(x - ref) < 1e-2f16) || abs((x - ref) / ref) < 1e-1f16; }

task void f_v(uniform float RET[]) {

    uniform float16 a = 1.96484;
    uniform float16 b = 6.80859;
    uniform float16 ref = 99.375;
    uniform float16 res = pow(a, b);
    RET[programIndex] = ok(res, ref) ? 1. : 0.;

    a = 2.4668;
    b = -3.62695;
    ref = 99.375;
    res = pow(a, b);
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;

    a = 0.155884;
    b = 1.68164;
    ref = 0.0439148;
    res = pow(a, b);
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;

    a = 3.83789;
    b = 0.000000;
    ref = 1.000000;
    res = pow(a, b);
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
