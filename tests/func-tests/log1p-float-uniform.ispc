#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    // See the varying version
    return (abs(x - ref) < 1e-6) || abs((x - ref) / ref) < 1e-5;
}

task void f_v(uniform float RET[]) {
    const uniform float x[] = {
        -0.17, 0.019,
        0.49, 0.93,
        4e-2, 3e-4,
        2e-6, 1e-8,
        2.11, 5.28,
        9.03, 20.44,
        73.5, 942.1,
        6485.2, 647913.5,
        -0.25, -0.5,
        -0.75, -0.99,
        -0.0, +0.0,
        -0.99999, 1e25
    };

    const uniform float ref[] = {
        -1.8632957339e-01, +1.8821753561e-02,
        +3.9877611399e-01, +6.5751999617e-01,
        +3.9220713079e-02, +2.9995501973e-04,
        +1.9999979486e-06, +9.9999999392e-09,
        +1.1346226931e+00, +1.8373700380e+00,
        +2.3055806160e+00, +3.0652582645e+00,
        +4.3107991219e+00, +6.8491721153e+00,
        +8.7774324417e+00, +1.3381513596e+01,
        -2.8768208623e-01, -6.9314718246e-01,
        -1.3862943649e+00, -4.6051712036e+00,
        -0.0000000000e+00, +0.0000000000e+00,
        -1.1511568069e+01, +5.7564628601e+01
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float res = log1p(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
