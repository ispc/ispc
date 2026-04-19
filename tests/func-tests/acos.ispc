#include "test_static.isph"

bool ok(float x, float ref) {
    return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-5;
}

task void f_v(uniform float RET[]) {
    uniform float vals[15] = { 
        0.0, 1.0, 0.5, 0.79, 1e-3, 0.995, 0.9999998, 0.999999,
        -1.0, -0.87, -0.25, -0.01, -0.995, -0.9999998, -0.999999
    };

    uniform float r[15];

    foreach (i = 0 ... 15) {
        // TODO: Possible optimization opportunity.
        #pragma ignore warning(perf)
        r[i] = cos(acos(vals[i % 15]));
    }

    int errors = 0;
    for (uniform int i = 0; i < 15; ++i) {
        if (ok(r[i], vals[i % 15]) == false) {
            ++errors;
        }
    }

    // Just to be sure the function actually compute something
    if (!ok(acos(0.972381), 0.235572))
        ++errors;

    RET[programIndex] = errors;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
