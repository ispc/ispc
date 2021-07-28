#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    typedef float Float;
    Float a = aFOO[programIndex];
    RET[programIndex] = a;
}

task void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
