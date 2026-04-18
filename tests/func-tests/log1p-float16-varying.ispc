#include "test_static.isph"

// rule: skip on arch=ppc64le

bool ok(float16 x, float16 ref) {
    // log1p is meant to be much more precise than log(1+x) for x~=0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float x = aFOO[programIndex];

    // Ensure the input is valid for all log1p implementations
    x = select(x > -1.0, x, abs(x));

    float16 ref = log1p((float)x);
    float16 res = log1p((float16)x);
    RET[programIndex] = ok(res, ref) ? 0 : 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
