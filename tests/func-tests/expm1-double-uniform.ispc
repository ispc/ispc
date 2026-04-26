#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

uniform bool ok(uniform double x, uniform double ref) {
    // See the varying version
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    const uniform double x[] = {
        0.2d, 0.42d,
        9e-2d, 3e-4d,
        6e-10d, 9e-120d,
        2.3d, 4.7d,
        11.1d, 397.3d,
        -3.0d, -0.1d
    };

    const uniform double ref[] = {
        +2.214027581601698547e-01d, +5.219615556186337768e-01d,
        +9.417428370521035985e-02d, +3.000450045003374717e-04d,
        +6.000000001799999774e-10d, +8.999999999999999979e-120d,
        +8.974182454814718213e+00d, +1.089471724521235245e+02d,
        +6.617016016837657662e+04d, +3.509115477556654489e+172d,
        -9.502129316321360486e-01d, -9.516258196404042691e-02d
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    for (uniform int i = 0; i < count; ++i)
    {
        const uniform double res = expm1(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
