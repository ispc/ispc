#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
#pragma ignore warning(perf)
    uniform float16 x = aFOO[programCount / 2];
    float16 d, ix;
    ix = rcp(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-4) ? 1. : 0.;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.; }
