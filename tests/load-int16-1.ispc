#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 x[programCount];
    x[programIndex] = aFOO[programIndex];
    unsigned int16 v = 0;
    if (programIndex & 1)
        v = x[programIndex];
    #pragma ignore warning(perf)
    RET[programIndex] = v;
}

task void result(uniform float RET[]) {
    if (programIndex & 1)
        RET[programIndex] = 1+programIndex;
    else
        RET[programIndex] = 0;
}
