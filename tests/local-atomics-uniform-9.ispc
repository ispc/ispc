#include "../test_static.isph"
uniform int64 s = 100.;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int64 b = atomic_compare_exchange_local(&s, 100, -100);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = -100.;
}
