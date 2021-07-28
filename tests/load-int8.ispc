#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[programCount];
    x[programIndex] = aFOO[programIndex];
    unsigned int8 v = x[programIndex];
    #pragma ignore warning(perf)
    RET[programIndex] = v;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
