#include "test_static.isph"

// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

bool ok(double x, double ref) {
    // log1p is meant to be much more precise than log(1+x) for x~=0
    // so the absolute precision threshold must be much smaller.
    return (abs(x - ref) < 1e-300d) || abs((x - ref) / ref) < 2e-12d;
}

task void f_v(uniform float RET[]) {
    const uniform double x[] = {
        0.019d, 0.49d,
        2e-2d, 1e-12d,
        2.11d, 9.03d,
        73.5d, 6485.2d,
        647913.5d, 123456789.0123777d,
        -0.25d, -0.5d,
        -0.77d, -0.99d,
        -0.0d, +0.0d,
        -0.9999999999999d, 3.42e179d
    };

    const uniform double ref[] = {
        +1.882175424058776109e-02d, +3.987761199573677806e-01d,
        +1.980262729617971243e-02d, +9.999999999994999557e-13d,
        +1.134622726191142794e+00d, +2.305580601973844068e+00d,
        +4.310799125385513797e+00d, +8.777432122037861717e+00d,
        +1.338151402218453256e+01d, +1.863140177436827827e+01d,
        -2.876820724517809014e-01d, -6.931471805599452862e-01d,
        -1.469675970058941683e+00d, -4.605170185988090914e+00d,
        -0.000000000000000000e+00d, +0.000000000000000000e+00d,
        -2.993329531206876482e+01d, +4.133923721970086831e+02d
    };

    const uniform count = sizeof(x) / sizeof(x[0]);
    assert(count == sizeof(ref) / sizeof(ref[0]));

    RET[programIndex] = 0;

    foreach(i = 0...count)
    {
        const double res = log1p(x[i]);
        RET[programIndex] += ok(res, ref[i]) ? 0 : 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
