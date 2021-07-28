#include "../test_static.isph"
uniform int32 s = 0xbeef;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = atomic_swap_local(&s, programIndex);
    RET[programIndex] = reduce_max(b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0xbeef;
}
