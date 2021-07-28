#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[programCount];
    x[programIndex] = -1;
    int a = aFOO[programIndex]-1;
    if (programIndex < 3) {
        #pragma ignore warning(perf)
        x[a] = programIndex;
    }
    RET[programIndex] = x[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = -1;
    RET[0] = 0;
    RET[1] = 1;
    RET[2] = 2;
}
