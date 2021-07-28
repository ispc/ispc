#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    float x = aFOO[programIndex&0x3];
    float d, ix;
    ix = rcp_fast(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-2) ? 1. : 0.;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
