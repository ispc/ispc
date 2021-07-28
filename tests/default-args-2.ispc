#include "../test_static.isph"
float foo(uniform float x, float f = 1) { return f+x; }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = foo(b);
}

task void result(uniform float RET[]) { RET[programIndex] = 6; }
