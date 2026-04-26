#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    // See the varying version
    return (abs(x - ref) < 1e-35) || abs((x - ref) / ref) < 2e-5;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    const uniform float x[] = {
        0.2, 0.42, 0.6, 2.1,
        9e-2, 4e-2, 3e-4, 2e-6,
        6e-12, 7e-16, 8e-20, 9e-24,
        5.9, 9.1, 17.2, 31.3,
        -11.0, -0.3, -1e-2, -1e-8
    };

    const uniform float ref[] = {
        +2.2140276432e-01, +5.2196151018e-01,
        +8.2211881876e-01, +7.1661691666e+00,
        +9.4174288213e-02, +4.0810775012e-02,
        +3.0004500877e-04, +2.0000020413e-06,
        +5.9999999760e-12, +6.9999999195e-16,
        +7.9999997461e-20, +8.9999996828e-24,
        +3.6403750610e+02, +8.9542958984e+03,
        +2.9502948000e+07, +3.9211814093e+13,
        -9.9998331070e-01, -2.5918179750e-01,
        -9.9501656368e-03, -9.9999999392e-09
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float res = expm1(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
