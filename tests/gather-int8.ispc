#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[programCount];
    x[programIndex] = programIndex;
    int a = aFOO[programIndex]-1;
    #pragma ignore warning(perf)
    unsigned int8 v = x[a];
    #pragma ignore warning(perf)
    RET[programIndex] = v;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
