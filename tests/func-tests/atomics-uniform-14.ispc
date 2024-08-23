#include "test_static.isph"
// rule: skip on arch=xe64
uniform float s = 0xffff;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform float b = atomic_max_global(&s, 1);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0xffff;
}
