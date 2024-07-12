// rule: skip on arch=xe64
#include "../test_static.isph"
uniform double s = 10;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex];
    atomic_add_global(&s, 1);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 11;
}
