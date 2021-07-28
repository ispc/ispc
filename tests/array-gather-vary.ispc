#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float x[programCount+10];
    uniform int i;
    for (i = 0; i < programCount+10; ++i)
        x[i] = a+b;
    #pragma ignore warning(perf)
    RET[programIndex] = x[a];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 6 + programIndex;
}
