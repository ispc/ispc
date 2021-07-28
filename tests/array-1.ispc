#include "../test_static.isph"
static float x[1][2];

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    x[0][1] = a;
    RET[programIndex] = x[0][1];
}

task void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
