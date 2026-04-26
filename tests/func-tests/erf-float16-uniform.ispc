#include "test_static.isph"

// rule: skip on arch=ppc64le

uniform bool ok(uniform float16 x, uniform float16 ref) {
    return (abs(x - ref) < 0.001f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_v(uniform float RET[]) {
    const uniform float16 x[] = {
        0.17, 0.99,
        1.17, 1.67,
        4e-1, 3e-2,
        2e-3, 1e-4,
        2.0, 3.0,
        4.0, 5.0,
        -5.0, -2.0,
        -1e-2, -1e-6,
        -0.0, +0.0,
        -1e4, 1e4
    };

    const uniform float16 ref[] = {
        +1.9006e-01, +8.3838e-01,
        +9.0186e-01, +9.8193e-01,
        +4.2822e-01, +3.3844e-02,
        +2.2583e-03, +1.1283e-04,
        +9.9512e-01, +1.0000e+00,
        +1.0000e+00, +1.0000e+00,
        -1.0000e+00, -9.9512e-01,
        -1.1284e-02, -1.1325e-06,
        -0.0000e+00, +0.0000e+00,
        -1.0000e+00, +1.0000e+00
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float16 res = erf(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
