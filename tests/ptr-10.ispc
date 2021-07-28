#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    RET[programIndex] = *b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
