#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    // See the varying version
    return (abs(x - ref) < 1e-6) || abs((x - ref) / ref) < 1e-5;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float x = aFOO[programIndex];

    // Ensure the input is valid for all log1p implementations
    x = select(x > -1.0, x, abs(x));

    const varying float ref = log1p(x);

    foreach_active(i) {
        const uniform float res = log1p(extract(x, i));
        RET[i] = ok(res, extract(ref, i)) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
