#include "../test_static.isph"
uniform float s = 100.;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform float b = atomic_swap_global(&s, 1.);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
