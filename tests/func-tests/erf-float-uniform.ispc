#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    return (abs(x - ref) < 1e-6) || abs((x - ref) / ref) < 1e-5;
}

task void f_v(uniform float RET[]) {
    const uniform float x[] = {
        0.27, 0.99,
        1.17, 2.67,
        4e-2, 3e-4,
        2e-6, 1e-8,
        5.0, 7.0,
        9.0, 11.0,
        -11.0, -3.0,
        -1e-2, -1e-8,
        -0.0, +0.0,
        -1e25, 1e25
    };

    const uniform float ref[] = {
        +2.9741823673e-01, +8.3850806952e-01,
        +9.0200036764e-01, +9.9984061718e-01,
        +4.5111104846e-02, +3.3851375338e-04,
        +2.2567583073e-06, +1.1283791679e-08,
        +1.0000000000e+00, +1.0000000000e+00,
        +1.0000000000e+00, +1.0000000000e+00,
        -1.0000000000e+00, -9.9997788668e-01,
        -1.1283415370e-02, -1.1283791679e-08,
        -0.0000000000e+00, +0.0000000000e+00,
        -1.0000000000e+00, +1.0000000000e+00
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float res = erf(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
