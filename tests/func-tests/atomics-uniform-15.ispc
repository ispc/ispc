#include "test_static.isph"
// rule: skip on arch=xe64
uniform double s = 0xffff;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex];
    uniform double b = atomic_max_global(&s, 1);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0xffff;
}
