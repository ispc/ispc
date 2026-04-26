#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

uniform bool ok(uniform double x, uniform double ref) {
    return (abs(x - ref) < 1e-15d) || abs((x - ref) / ref) < 1e-12d;
}

task void f_v(uniform float RET[]) {
    const uniform double x[] = {
        0.27d, 0.99d,
        2e-2d, 1e-12d,
        5.0d, 7.0d,
        -3.0d, -1e-2d,
        -0.0d, +0.0d,
        80.0d, 379.1d
    };

    const uniform double ref[] = {
        +2.732924781981971307e-01d, +1.159828890663608281e+00d,
        +2.000133336000025491e-02d, +9.999999999999999799e-13d,
        +7.420321057778875229e+01d, +5.483161232732464896e+02d,
        -1.001787492740990260e+01d, -1.000016666750000276e-02d,
        -0.000000000000000000e+00d, +0.000000000000000000e+00d,
        +2.770311192196754917e+34d, +2.187802397202284545e+164d
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform double res = sinh(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
