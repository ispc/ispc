#include "../test_static.isph"
static float float4(uniform float a, uniform float b, uniform float c, 
                    uniform float d) {
    float ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}



bool ok(float x, float ref) { return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-4; }

task void f_v(uniform float RET[]) {
    float v = float4((14.300000190734863),(-6.699999809265137),(-21.200000762939453),(9.000000000000000));
    float ref = float4((0.986771941184998),(-0.404849767684937),(-0.711160659790039),(0.412118494510651));
    RET[programIndex] = ok(sin(v), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
