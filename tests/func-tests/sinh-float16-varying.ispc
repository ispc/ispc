#include "test_static.isph"

// rule: skip on arch=ppc64le

bool ok(float16 x, float16 ref) {
    return (abs(x - ref) < 0.001f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 arg = ((aFOO[programIndex] / programCount) * 3 - 1.5);
    float16 ref = sinh((float)arg);
    float16 res = sinh((float16)arg);
    RET[programIndex] = ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
