#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    uniform float * uniform c = aFOO;
    RET[programIndex] = (c <= b) ? 10 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
