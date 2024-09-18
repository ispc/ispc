// rule: skip on arch=xe64
#include "test_static.isph"
uniform float s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = atomic_min_global(&s, a);
    RET[programIndex] = reduce_min(b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = reduce_min(programIndex);
}
