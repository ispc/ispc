#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform ap = &aFOO[0];
    RET[programIndex] = ap[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
