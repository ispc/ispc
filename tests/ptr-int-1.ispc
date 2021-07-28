#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = &a;
    uintptr_t pi = (uintptr_t)b;
    RET[programIndex] = *((uniform int * varying)pi);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
