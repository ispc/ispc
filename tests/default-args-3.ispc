#include "../test_static.isph"
static const float g = 100;

float foo(uniform float x, float f = 2*2) { return f+x; }
float foo(uniform float x, int i) { return x; }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = foo(b);
}

task void result(uniform float RET[]) { RET[programIndex] = 9; }
