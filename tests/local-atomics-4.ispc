#include "../test_static.isph"
uniform int32 s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0;
    if (programIndex < 29)
        atomic_or_local(&s, (1ul<<programIndex));
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (1ul<<min(29,programCount))-1;
}
