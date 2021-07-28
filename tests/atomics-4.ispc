#include "../test_static.isph"
uniform int32 s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = atomic_or_global(&s, (1ul<<min(programIndex,30)));
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (1ul<<min(programCount,31))-1;
}
