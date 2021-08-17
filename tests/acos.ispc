#include "../test_static.isph"
bool ok(float x, float ref) { return (abs(x - ref) < 1e-4) || abs((x-ref)/ref) < 1e-4; }

task void f_v(uniform float RET[]) {
    uniform float vals[8] = { 0, 1, 0.5, -1, -.87, -.25, 1e-3, -.99999999 };
    uniform float r[8];
    foreach (i = 0 ... 8) {
        // TODO: Possible optimization opportunity.
        #pragma ignore warning(perf)
        r[i] = cos(acos(vals[i % 8]));
    }

    int errors = 0;
    for (uniform int i = 0; i < 8; ++i) {
        if (ok(r[i], vals[i%8]) == false) {
            ++errors;
        }
    }
    RET[programIndex] = errors;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
