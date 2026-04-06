#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    return (abs(x - ref) < 1e-6) || abs((x - ref) / ref) < 1e-5;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    for (uniform int i = 0; i != programCount; ++i) {
        uniform float ref = cosh((uniform float)((aFOO[i] / programCount) * 3 - 1.5));
        uniform float res = cosh((aFOO[i] / programCount) * 3 - 1.5);
        RET[i] = ok(res, ref) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
