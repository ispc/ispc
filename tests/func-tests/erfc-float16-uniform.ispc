#include "test_static.isph"

// rule: skip on arch=ppc64le

uniform bool ok(uniform float16 x, uniform float16 ref) {
    // See the varying version
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
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
        +8.1006e-01, +1.6138e-01,
        +9.8022e-02, +1.8188e-02,
        +5.7178e-01, +9.6631e-01,
        +9.9756e-01, +1.0000e+00,
        +4.6768e-03, +2.2113e-05,
        +0.0000e+00, +0.0000e+00,
        +2.0000e+00, +1.9951e+00,
        +1.0117e+00, +1.0000e+00,
        +1.0000e+00, +1.0000e+00,
        +2.0000e+00, +0.0000e+00
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float16 res = erfc(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
