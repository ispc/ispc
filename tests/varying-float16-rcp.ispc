#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
#pragma ignore warning(perf)
    varying float16 x = aFOO[programIndex];
    varying float16 d, ix;
    ix = rcp(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    if (d < 1e-4) {
        RET[programIndex] = 1.;
    } else {
        RET[programIndex] = 0.;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 1.; }
