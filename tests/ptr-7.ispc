#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying ap = &aFOO[programIndex];
    RET[programIndex] = *ap;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
