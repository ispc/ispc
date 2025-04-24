#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform int16 m;
    if (v >= 0 && v < 2)
        m = reduce_min(-(int16)v);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) { RET[programIndex] = -1; }
