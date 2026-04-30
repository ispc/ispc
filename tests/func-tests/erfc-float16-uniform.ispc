#include "test_static.isph"

// rule: skip on arch=ppc64le

uniform bool ok(uniform float16 x, uniform float16 ref) {
    // See the varying version
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    for (uniform int i = 0; i != programCount; ++i) {
        uniform float16 ref = erfc((uniform float16)((aFOO[i] / programCount) * 3 - 1.5));
        uniform float16 res = erfc((aFOO[i] / programCount) * 3 - 1.5);
        RET[i] = ok(res, ref) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
