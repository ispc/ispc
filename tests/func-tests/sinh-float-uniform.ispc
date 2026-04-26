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
        21.7, 80.0
    };

    const uniform float ref[] = {
        +2.7329248190e-01, +1.1598289013e+00,
        +1.4558128119e+00, +7.1853590012e+00,
        +4.0010668337e-02, +3.0000001425e-04,
        +1.9999999950e-06, +9.9999999392e-09,
        +7.4203208923e+01, +5.4831610107e+02,
        +4.0515419922e+03, +2.9937070312e+04,
        -2.9937070312e+04, -1.0017874718e+01,
        -1.0000166483e-02, -9.9999999392e-09,
        -0.0000000000e+00, +0.0000000000e+00,
        +1.3278854400e+09, +2.7703112423e+34
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float res = sinh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
