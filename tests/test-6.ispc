#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform float b = 1.;
    RET[programIndex] = a + b;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 2 + programIndex;
}
