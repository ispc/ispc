#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

uniform bool ok(uniform double x, uniform double ref) {
    // See the varying version
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    const double x = aFOO[programIndex];
    const double ref = expm1(x);

    foreach_active(i) {
        const uniform double res = expm1(extract(x, i));
        RET[i] = ok(res, extract(ref, i)) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
