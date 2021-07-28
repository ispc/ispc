#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[programCount];
    x[programIndex] = 0;
    int a = aFOO[programIndex]-1;
    #pragma ignore warning(perf)
    x[a] = programIndex;
    RET[programIndex] = x[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
