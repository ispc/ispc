#include "test_static.isph"
uniform double s = 0xffff;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex];
    uniform double b = atomic_min_global(&s, 1);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
