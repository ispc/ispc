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
        7.1, 10.3
    };

    const uniform float16 ref[] = {
        +1.0146e+00, +1.5312e+00,
        +1.7656e+00, +2.7500e+00,
        +1.0811e+00, +1.0000e+00,
        +1.0000e+00, +1.0000e+00,
        +3.7617e+00, +1.0070e+01,
        +2.7312e+01, +7.4188e+01,
        +7.4188e+01, +3.7617e+00,
        +1.0000e+00, +1.0000e+00,
        +1.0000e+00, +1.0000e+00,
        +6.0700e+02, +1.4816e+04
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float16 res = cosh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
