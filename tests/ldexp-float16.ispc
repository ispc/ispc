#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float16 a = 1ul << ((programIndex/4) % 28);
    if (programIndex & 1)
        a = -a;
    RET[programIndex] = ldexp(a, 2);
}

task void result(uniform float RET[]) {
    int pi = (programIndex/4) % 28;
    #pragma ignore warning(perf)
    RET[programIndex] = (1ul << (pi + 2));
    if (programIndex & 1)
        RET[programIndex] = -RET[programIndex];
}
