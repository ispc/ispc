#include "../test_static.isph"
uniform unsigned int32 s[programCount];

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    float delta = 1;
    atomic_add_global(&s[programCount-1-programIndex], programIndex);
    #pragma ignore warning(perf)
    RET[programIndex] = s[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount-1-programIndex;
}
