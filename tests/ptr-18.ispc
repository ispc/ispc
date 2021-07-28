#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    b += 10;
    RET[programIndex] = b[-5];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 6;
}
