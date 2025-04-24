#include "test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform double a[programCount];
    #pragma ignore warning(perf)
    a[programIndex] = aFOO[programIndex];
    double aa;
    packed_load_active(a, &aa);
    #pragma ignore warning(perf)
    RET[programIndex] = aa;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
