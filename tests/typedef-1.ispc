#include "../test_static.isph"

typedef float Float;

task void f_fu(uniform float RET[], uniform Float aFOO[], uniform Float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = a+b;
}

task void result(uniform float RET[]) { RET[programIndex] = 6 + programIndex; }
