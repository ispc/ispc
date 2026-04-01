#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

uniform bool ok(uniform double x, uniform double ref) {
    // See the varying version
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    for (uniform int i = 0; i != programCount; ++i) {
        uniform double ref = erfc((uniform double)((aFOO[i] / programCount) * 3.0d - 1.5d));
        uniform double res = erfc((aFOO[i] / programCount) * 3.0d - 1.5d);
        RET[i] = ok(res, ref) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
