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

bool ok(float16 x, float16 ref) { return (abs(x - ref) < 1e-2f16) || abs((x - ref) / ref) < 1e-1f16; }

task void f_v(uniform float RET[]) {
    float16 a = float4((1.96484), (2.4668), (0.155884), (3.83789));
    float16 b = float4((6.80859), (-3.62695), (1.68164), (0.000000));
    float16 ref = float4((99.375), (0.0378113), (0.0439148), (1.00000));
    float16 res = pow(a, b);
    RET[programIndex] = ok(res, ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
