#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform int8 m = 42;
    if (v >= 3)
        m = reduce_max(-(int8)v);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) { RET[programIndex] = (programCount == 1) ? 42 : -3; }
