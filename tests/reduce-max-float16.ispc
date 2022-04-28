#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m = 42;
    if (v >= 3)
        m = reduce_max(-(float16)v);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) { RET[programIndex] = (programCount == 1) ? 42 : -3; }
