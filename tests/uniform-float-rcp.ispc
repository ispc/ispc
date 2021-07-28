#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    uniform float x = aFOO[programCount/2];
    float d, ix;
    ix = rcp(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-7) ? 1. : 0.;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
