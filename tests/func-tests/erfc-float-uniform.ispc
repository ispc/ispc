#include "test_static.isph"

uniform bool ok(uniform float x, uniform float ref) {
    // See the varying version
    return (abs(x - ref) < 1e-35) || abs((x - ref) / ref) < 2e-5;
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
        +7.0258176327e-01, +1.6149193048e-01,
        +9.7999610007e-02, +1.5939876903e-04,
        +9.5488888025e-01, +9.9966150522e-01,
        +9.9999773502e-01, +1.0000000000e+00,
        +1.5374598296e-12, +4.1838257317e-23,
        +4.1370317081e-37, +0.0000000000e+00,
        +2.0000000000e+00, +1.9999779463e+00,
        +1.0112833977e+00, +1.0000000000e+00,
        +1.0000000000e+00, +1.0000000000e+00,
        +2.0000000000e+00, +0.0000000000e+00
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform float res = erfc(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
