#include "../test_static.isph"
uniform int32 s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 a = aFOO[programIndex]; 
    float b = 0;
    if (programIndex < 32)
        atomic_min_local(&s, a);
    RET[programIndex] = reduce_min(b);
}

task void result(uniform float RET[]) {
    int pi = (programIndex < 32) ? programIndex : 0;
    RET[programIndex] = reduce_min(pi);
}
