#include "test_static.isph"

// rule: skip on arch=ppc64le

uniform bool ok(uniform float16 x, uniform float16 ref) {
    // See the varying version
    return (abs(x - ref) < 0.001f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 x = aFOO[programIndex];

    // Ensure the input is valid for all log1p implementations
    x = select(x > -1.0f16, x, abs(x));

    const varying float16 ref = log1p(x);

    foreach_active(i) {
        const uniform float16 res = log1p(extract(x, i));
        RET[i] = ok(res, extract(ref, i)) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
