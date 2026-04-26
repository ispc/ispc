#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

bool ok(double x, double ref) {
    // erfc is meant to be much more precise than erf for x>1
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_v(uniform float RET[]) {
    const uniform double x[] = {
        0.27d, 0.99d,
        2e-2d, 1e-12d,
        3.0d, 4.0d,
        5.0d, 7.0d,
        9.7d, 26.1d,
        -3.0d, -1e-2d,
        -0.0d, +0.0d,
        -1e160d, 1e160d
    };

    const uniform double ref[] = {
        +7.025817814529872196e-01d, +1.614919304446301995e-01d,
        +9.774354253081550814e-01d, +9.999999999988715693e-01d,
        +2.209049699858544122e-05d, +1.541725790028001999e-08d,
        +1.537459794428034936e-12d, +4.183825607779414210e-23d,
        +7.936107563214511110e-43d, +3.081217493314593186e-298d,
        +1.999977909503001472e+00d, +1.011283415555849663e+00d,
        +1.000000000000000000e+00d, +1.000000000000000000e+00d,
        +2.000000000000000000e+00d, +0.000000000000000000e+00d
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const double res = erfc(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
