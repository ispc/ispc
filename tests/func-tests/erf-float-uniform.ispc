#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    return (abs(x - ref) < 1e-6) || abs((x - ref) / ref) < 1e-5;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    const float x = aFOO[programIndex];
    const float ref = erf(x);

    foreach_active(i) {
        const uniform float res = erf(extract(x, i));
        RET[i] = ok(res, extract(ref, i)) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
