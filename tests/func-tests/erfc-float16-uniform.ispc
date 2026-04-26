#include "test_static.isph"

// rule: skip on arch=ppc64le

uniform bool ok(uniform float16 x, uniform float16 ref) {
    // See the varying version
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    const float16 x = aFOO[programIndex];
    const float16 ref = erfc(x);

    foreach_active(i) {
        const uniform float16 res = erfc(extract(x, i));
        RET[i] = ok(res, extract(ref, i)) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
