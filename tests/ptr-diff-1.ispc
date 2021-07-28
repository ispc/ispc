#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    b += 5;
    RET[programIndex] = b - aFOO;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
