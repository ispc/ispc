#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

uniform bool ok(uniform double x, uniform double ref) {
    return (abs(x - ref) < 1e-15d) || abs((x - ref) / ref) < 1e-12d;
}

task void f_v(uniform float RET[]) {
    const uniform double x[] = {
        0.27d, 0.99d,
        2e-2d, 1e-12d,
        5.0d, 7.0d,
        -3.0d, -1e-2d,
        -0.0d, +0.0d,
        -1e160d, 1e160d
    };

    const uniform double ref[] = {
        +2.974182185470127804e-01d, +8.385080695553698282e-01d,
        +2.256457469184494635e-02d, +1.128379167095512566e-12d,
        +9.999999999984625632e-01d, +1.000000000000000000e+00d,
        -9.999779095030013609e-01d, -1.128341555584961783e-02d,
        -0.000000000000000000e+00d, +0.000000000000000000e+00d,
        -1.000000000000000000e+00d, +1.000000000000000000e+00d
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform double res = erf(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
