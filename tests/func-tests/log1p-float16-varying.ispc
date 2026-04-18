#include "test_static.isph"

// rule: skip on arch=ppc64le

bool ok(float16 x, float16 ref) {
    // log1p is meant to be much more precise than log(1+x) for x~=0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-4f16) || abs((x - ref) / ref) < 0.05f16;
}

task void f_v(uniform float RET[]) {
    const uniform float16 x[] = {
        -0.17, 0.019,
        0.49, 0.93,
        4e-1, 3e-2,
        2e-3, 1e-4,
        2.11, 5.28,
        9.03, 20.44,
        53.5, 242.1,
        648.2, 64713.5,
        -0.25, -0.5,
        -0.75, -0.99,
        -0.0, +0.0,
        -0.999, 3e4
    };

    const uniform float16 ref[] = {
        -1.8640e-01, +1.8814e-02,
        +3.9868e-01, +6.5771e-01,
        +3.3643e-01, +2.9556e-02,
        +1.9989e-03, +1.0002e-04,
        +1.1348e+00, +1.8379e+00,
        +2.3066e+00, +3.0645e+00,
        +3.9980e+00, +5.4922e+00,
        +6.4766e+00, +1.1078e+01,
        -2.8760e-01, -6.9336e-01,
        -1.3867e+00, -4.6289e+00,
        -0.0000e+00, +0.0000e+00,
        -6.9297e+00, +1.0312e+01
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const float16 res = log1p(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
