#include "test_static.isph"

bool ok(float x, float ref) {
    return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-5;
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
        +2.6362484694e-01, +7.5736230612e-01,
        +8.2427215576e-01, +9.9045401812e-01,
        +3.9978679270e-02, +3.0000001425e-04,
        +1.9999999950e-06, +9.9999999392e-09,
        +9.9990922213e-01, +9.9999833107e-01,
        +9.9999994040e-01, +1.0000000000e+00,
        -1.0000000000e+00, -9.9505478144e-01,
        -9.9996663630e-03, -9.9999999392e-09,
        -0.0000000000e+00, +0.0000000000e+00,
        -1.0000000000e+00, +1.0000000000e+00
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const float res = tanh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
