#include "../test_static.isph"
uniform float s = 100.;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform float b = atomic_compare_exchange_local(&s, 1., -100.);
    RET[programIndex] = b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 100.;
}
