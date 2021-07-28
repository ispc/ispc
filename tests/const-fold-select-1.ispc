#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = (programIndex >= 0) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
