#include "test_static.isph"

// rule: skip on arch=ppc64le

bool ok(float16 x, float16 ref) {
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
        +1.6846e-01, +7.5732e-01,
        +8.2422e-01, +9.3164e-01,
        +3.7988e-01, +2.9984e-02,
        +2.0008e-03, +1.0002e-04,
        +9.6387e-01, +9.9512e-01,
        +9.9951e-01, +1.0000e+00,
        -1.0000e+00, -9.6387e-01,
        -1.0002e-02, -1.0133e-06,
        -0.0000e+00, +0.0000e+00,
        -1.0000e+00, +1.0000e+00
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const float16 res = tanh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
