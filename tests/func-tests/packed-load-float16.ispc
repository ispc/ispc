#include "test_static.isph"
// rule: skip on arch=ppc64le
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float16 a[programCount];
    #pragma ignore warning(perf)
    a[programIndex] = aFOO[programIndex];
    float16 aa;
    packed_load_active(a, &aa);
    #pragma ignore warning(perf)
    RET[programIndex] = aa;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
