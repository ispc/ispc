#include "../test_static.isph"
uniform int32 s = 2;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = atomic_compare_exchange_global(&s, programIndex, a*1000);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 2 : 3000;
}
