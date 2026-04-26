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
        80.0d, 379.1d,
        -1e160d, 1e160d
    };

    const uniform double ref[] = {
        +2.636248354722033338e-01d, +7.573623242165262726e-01d,
        +1.999733375993093323e-02d, +9.999999999999999799e-13d,
        +9.999092042625951082e-01d, +9.999983369439446879e-01d,
        -9.950547536867304643e-01d, -9.999666679999460323e-03d,
        -0.000000000000000000e+00d, +0.000000000000000000e+00d,
        +1.000000000000000000e+00d, +1.000000000000000000e+00d,
        -1.000000000000000000e+00d, +1.000000000000000000e+00d
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform double res = tanh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
