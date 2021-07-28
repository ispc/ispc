#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = 1ul << (programIndex % 28);
    if (programIndex & 1)
        a = -a;
    RET[programIndex] = ldexp(a, 2);
}

task void result(uniform float RET[]) {
    int pi = programIndex % 28;
    #pragma ignore warning(perf)
    RET[programIndex] = (1ul << (pi + 2));
    if (programIndex & 1)
        RET[programIndex] = -RET[programIndex];
}
