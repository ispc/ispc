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
        21.7, 80.0
    };

    const uniform float ref[] = {
        +1.0366719961e+00, +1.5314055681e+00,
        +1.7661796808e+00, +7.2546114922e+00,
        +1.0008001328e+00, +1.0000000000e+00,
        +1.0000000000e+00, +1.0000000000e+00,
        +7.4209945679e+01, +5.4831701660e+02,
        +4.0515419922e+03, +2.9937070312e+04,
        +2.9937070312e+04, +1.0067662239e+01,
        +1.0000499487e+00, +1.0000000000e+00,
        +1.0000000000e+00, +1.0000000000e+00,
        +1.3278854400e+09, +2.7703112423e+34
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const float res = cosh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
