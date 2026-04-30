#include "test_static.isph"

// rule: skip on arch=ppc64le

bool ok(float16 x, float16 ref) {
    // expm1 is meant to be much more precise than exp for x ~= 0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 arg = ((aFOO[programIndex] / programCount) * 3 - 1.5);
    float16 ref = expm1((float)arg);
    float16 res = expm1((float16)arg);
    RET[programIndex] = ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
