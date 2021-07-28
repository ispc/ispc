#include "../test_static.isph"
float foo(float f = 1) { return f; }

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = foo();
}

task void result(uniform float RET[]) { RET[programIndex] = 1; }
