#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    double x = aFOO[programIndex&0x3]*1d100;
    double d, ix; 
    ix = rcp(x);
    d = (ix - 1.0d0 / x);
    d = (d < 0.0d0) ? -d : d;
    RET[programIndex] = (d < 1d-15 && !isnan(d)) ? 1.0d0 : 0.0d0; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1.0d0;
}
