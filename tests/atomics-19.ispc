// rule: skip on arch=xe64
#include "test_static.isph"
uniform float s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    if (programIndex & 1)
        b = atomic_max_global(&s, a);
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 0 : programCount;
}
