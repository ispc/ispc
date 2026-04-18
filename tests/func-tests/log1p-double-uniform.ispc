#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

uniform bool ok(uniform double x, uniform double ref) {
    // See the varying version
    return (abs(x - ref) < 1e-15d) || abs((x - ref) / ref) < 1e-12d;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    double x = aFOO[programIndex];

    // Ensure the input is valid for all log1p implementations
    x = select(x > -1.0d, x, abs(x));

    const varying double ref = log1p(x);

    foreach_active(i) {
        const uniform double res = log1p(extract(x, i));
        RET[i] = ok(res, extract(ref, i)) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
