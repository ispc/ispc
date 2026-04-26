#include "test_static.isph"

// rule: skip on arch=ppc64le

bool ok(float16 x, float16 ref) {
    // expm1 is meant to be much more precise than exp for x ~= 0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_v(uniform float RET[]) {
    const uniform float16 x[] = {
        0.2, 0.42, 0.6, 1.7,
        9e-1, 4e-1, 3e-2, 2e-3,
        3.9, 5.1, 7.2, 9.3,
        -11.0, -0.3, -1e-5, -1e-6
    };

    const uniform float16 ref[] = {
        +2.2131e-01, +5.2197e-01,
        +8.2227e-01, +4.4766e+00,
        +1.4590e+00, +4.9170e-01,
        +3.0457e-02, +2.0027e-03,
        +4.8406e+01, +1.6325e+02,
        +1.3370e+03, +1.0904e+04,
        -1.0000e+00, -2.5928e-01,
        -1.0014e-05, -1.0133e-06
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const float16 res = expm1(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
