#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying b = aFOO;
    uniform float * uniform c = &aFOO[10];
    RET[programIndex] = c - b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
